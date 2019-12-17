import argparse
import os
import time

import lucene
import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm, trange

import utils
from dcg import dcg
from expman import Experiment
from lucene_index_payload import LuceneIndex
from scalar_quantization import crelu, threshold
    

def surrogate_text(x, boost=False):
    surrogate = []
    x = utils.compute_if_dask(x, progress=False)
    for term, freq in enumerate(x):
        if freq:
            if boost:
                surrogate.append('{}^{}'.format(str(term), freq))
            else:
                try:
                    surrogate.extend([str(term)] * freq)
                except:
                    print(freq, type(freq))

    return ' '.join(surrogate)
                    

def features_to_str(x, batch_size=1, boost=False):
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        xb = utils.compute_if_dask(xb, progress=False)
        for xi in xb:
            yield surrogate_text(xi, boost=boost)
        del xb


def batch_features(x, batch_size=1):
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        xb = utils.compute_if_dask(xb, progress=False)
        for xi in xb:
            yield xi
        del xb


def main(args):
    lucene_vm = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    lucene_vm.attachCurrentThread()

    dataset, q, x = utils.load_benchmark(args.dataset, args.features)

    q = utils.load_features(q, chunks=(5000, 2048))
    x = utils.load_features(x, chunks=(5000, 2048))
    
    if args.limit:
        x = x[:args.limit]
        
    n_queries, n_samples = q.shape[0], x.shape[0]

    if args.crelu:
        q = crelu(q)
        x = crelu(x)

    params = vars(args)
    ignore = ('output', 'force')
    progress = tqdm(zip(args.threshold, args.sq_factor), total=len(args.threshold))
    for thr, s in progress:
        params['threshold'] = thr
        params['sq_factor'] = s
        progress.set_postfix({k: v for k, v in params.items() if k not in ignore})
        exp = Experiment(params, root=args.output, ignore=ignore)

        density, density_file = exp.require_csv(f'density.csv')
        if 'query_density' not in density:
            progress.write('Computing query density ...')
            q_re = q.rechunk({0: -1, 1: 'auto'}) if utils.is_dask(q) else q
            q_sq = threshold(q_re, thr, s)
            q_density = (q_sq != 0).mean(axis=0)
            q_density = utils.compute_if_dask(q_density)
            density['query_density'] = q_density
            density.to_csv(density_file, index=False)

        if 'database_density' not in density:
            progress.write('Computing database density ...')
            x_re = q.rechunk({0: -1, 1: 'auto'}) if utils.is_dask(x) else x
            x_sq = threshold(x_re, thr, s)
            x_density = (x_sq != 0).mean(axis=0)
            x_density = utils.compute_if_dask(x_density)
            density['database_density'] = x_density
            density.to_csv(density_file, index=False)

        index_stats, index_stats_file = exp.require_csv('index_stats.csv')

        index_name = exp.name.lower()
        index_path = exp.path_to('lucene_index')
        with LuceneIndex(index_path) as idx:
            if idx.count() < n_samples:
                x_sq = threshold(x, thr, s)
                x_sq = batch_features(x_sq, 5000)
                # x_str = features_to_str(x_sq, 5000)

                progress.write(f'Indexing: {index_name}')

                start = time.time()
                for i, xi in enumerate(tqdm(x_sq, total=n_samples)):
                    idx.add(str(i), xi)

                add_time = time.time() - start
                progress.write(f'Index time: {add_time}')

                index_stats.at[0, 'add_time'] = add_time

            if 'size' not in index_stats.columns:
                index_stats.at[0, 'size'] = utils.get_folder_size(index_path)

            index_stats.to_csv(index_stats_file, index=False)

        metrics, metrics_file = exp.require_csv(f'metrics.csv')

        scores = None
        scores_file = exp.path_to(f'scores.h5')
        if not os.path.exists(scores_file):
            progress.write('Computing scores...')

            q_sq = threshold(q, thr, s)
            q_sq = utils.compute_if_dask(q_sq, progress=False)
            # q_str = features_to_str(q_sq, n_queries, boost=True)

            scores = np.zeros((n_queries, n_samples), dtype=np.float32)
            query_times = []

            if True:  # sequential version
                for i, qi in enumerate(tqdm(q_sq, total=n_queries)):
                    start = time.time()
                    if qi.any():
                        for j, score in tqdm(idx.query(qi, n_samples), total=n_samples):
                            scores[i, int(j)] = score
                        query_times.append(time.time() - start)
                    else:
                        query_times.append(None)

            else:  # Parallel version (currently slower)
                idx._init_searcher()

                def _search(i, qi):
                    lucene_vm.attachCurrentThread()
                    scores_i = np.zeros(n_samples, dtype=np.float32)
                    start = time.time()
                    if qi.any():
                        for j, score in idx.query(qi, n_samples):
                            scores_i[int(j)] = score
                        query_time = time.time() - start
                    else:
                        query_time = None

                    return scores_i, query_time

                queries = enumerate(tqdm(q_sq, total=n_queries))
                scores_n_times = Parallel(n_jobs=6, prefer="threads")(delayed(_search)(i, qi) for i, qi in queries)
                scores, query_times = zip(*scores_n_times)
                scores = np.vstack(scores)

            metrics['query_time'] = query_times
            metrics.to_csv(metrics_file, index=False)
            progress.write(f'Query time: {metrics.query_time.sum()}')
            utils.save_as_hdf5(scores, scores_file, progress=True)

        if 'ap' not in metrics:
            dataset._load()  # TODO in y_true getter
            if scores is None:
                progress.write('Loading scores...')
                scores = utils.load_features(scores_file)[...]
            progress.write('Computing mAP...')
            metrics['ap'] = dataset.score(scores, reduction=False, progress=True)
            metrics.to_csv(metrics_file, index=False)
            progress.write(f'mAP: {metrics.ap.mean()}')

        if 'ndcg' not in metrics:
            dataset._load()  # TODO in y_true getter
            if scores is None:
                progress.write('Loading scores...')
                scores = utils.load_features(scores_file)[...]
            progress.write('Computing nDCG...')
            y_true = dataset.y_true[:, :args.limit] if args.limit else dataset.y_true
            bs = 50
            ndcg = []
            for i in trange(0, y_true.shape[0], bs):
                ndcg.append(dcg(y_true[i:i + bs], scores[i:i + bs], normalized=True))

            metrics['ndcg'] = np.concatenate(ndcg)
            # metrics['ndcg'] = dcg(dataset.y_true, scores, normalized=True)            
            metrics.to_csv(metrics_file, index=False)
            progress.write(f'nDCG: {metrics.ndcg.mean()}')
        
        if 'ndcg@25' not in metrics:
            dataset._load()  # TODO in y_true getter
            if scores is None:
                progress.write('Loading scores...')
                scores = utils.load_features(scores_file)[...]
            progress.write('Computing nDCG@25...')
            y_true = dataset.y_true[:, :args.limit] if args.limit else dataset.y_true
            bs = 50
            ndcg = []
            for i in trange(0, y_true.shape[0], bs):
                ndcg.append(dcg(y_true[i:i + bs], scores[i:i + bs], p=25, normalized=True))

            metrics['ndcg@25'] = np.concatenate(ndcg)
            # metrics['ndcg'] = dcg(dataset.y_true, scores, normalized=True)            
            metrics.to_csv(metrics_file, index=False)
            progress.write(f'nDCG@25: {metrics["ndcg@25"].mean()}')


if __name__ == '__main__':
    # thrs = (30, 26, 22, 18)
    thrs = (16, 18, 20, 22)
    """
    thrs = (range(20, 50, 4), # TODO fino a 30 max x ox+fl
            range(50, 100, 10),  # 50 to 100 excl. step 10
            range(10, 50, 2),  # 10 to 50 excl. step 2
            range(100, 1000, 100),  # 100 to 1000 excl. step 100
            range(1000, 5001, 1000),  # 1000 to 5000 step 1000
            range(2, 10))  # 2 to 10 excl. (less important)
    thrs = list(itertools.chain.from_iterable(thrs))
    """

    benchmarks = ('oxford', 'paris', 'holidays', 'oxford+flickr100k', 'holidays+mirflickr1m')

    parser = argparse.ArgumentParser(description='Scalar Quantization scoring')
    parser.add_argument('dataset', choices=benchmarks, help='Benchmark')
    parser.add_argument('features', help='Features dirname')
    parser.add_argument('output', help='Output dir for results')

    parser.add_argument('-f', '--force', default=False, action='store_true', help='Force indexing')
    parser.add_argument('-l', '--limit', type=int, default=0, help='Distractor set limit (0 = no limit)')

    parser.add_argument('-c', '--crelu', action='store_true', default=False, help='Use CReLU')
    parser.add_argument('-t', '--threshold', type=int, nargs='+', default=thrs, help='Thresholding factor (multiple values accepted)')
    parser.add_argument('-q', '--sq-factor', type=int, nargs='+', default=None, help='Scalar quantization factor (multiple values accepted)')

    args = parser.parse_args()

    if args.sq_factor is None:
        args.sq_factor = (0,) * len(args.threshold)

    main(args)
