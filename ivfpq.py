import argparse
import os
import time

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import utils
from dcg import dcg
from expman import Experiment


def main(args):
    dataset, q, x = utils.load_benchmark(args.dataset, args.features)

    q = utils.load_features(q, chunks=(2500, 2048))
    x = utils.load_features(x, chunks=(2500, 2048))

    if args.limit:
        x = x[:args.limit]

    n_points, dim = x.shape

    if args.n_cells is None:
        step_k = 2500
        min_points_per_centroid = 39.0
        max_points_per_centroid = 256.0

        # n_train_points = min(n_points, 120000) # train index with less points or it crashes..
        min_k = np.ceil(n_points / (step_k * max_points_per_centroid)).astype(int) * step_k
        max_k = np.floor(n_points / (step_k * min_points_per_centroid)).astype(int) * step_k
        args.n_cells = min_k
        print('Using min suggested cells:', args.n_cells)

    exp = Experiment(args, root=args.output, ignore=('output', 'pretrained'))
    print(exp)

    # create or load faiss index
    index_file = exp.path_to('index.faiss')
    if not os.path.exists(index_file):
        if args.pretrained:
            print('Loading pre-trained empty index ...')
            index = faiss.read_index(args.pretrained)
            train_time = None
        else:
            tmp = utils.compute_if_dask(x)
            print('Creating index: training ...')
            index = faiss.index_factory(dim, 'IVF{},PQ{}'.format(args.n_cells, args.code_size))
            # index = faiss.index_factory(dim, 'IVF{},Flat'.format(args.n_cells))
            start = time.time()
            index.train(tmp)
            train_time = time.time() - start
            del tmp

        print('Creating index: adding ...')
        start = time.time()
        bs = 2 ** 14
        for i in trange(0, x.shape[0], bs):
            batch = utils.compute_if_dask(x[i:i + bs])
            index.add(batch)
        add_time = time.time() - start

        faiss.write_index(index, index_file)
        size = os.path.getsize(index_file)
        index_stats_file = exp.path_to('index_stats.csv')
        index_stats = pd.DataFrame({'size': size, 'train_time': train_time, 'add_time': add_time}, index=[0])
        index_stats.to_csv(index_stats_file, index=False)
    else:
        print('Loading pre-built index ...')
        index = faiss.read_index(index_file)

    n_probes = (1, 2, 5, 10, 25) # , 50, 100, 250, 500, 1000, 2500, 5000)
    n_probes = filter(lambda x: x <= args.n_cells, n_probes)
    params = vars(args)
    progress = tqdm(n_probes)
    for p in progress:
        index.nprobe = p
        params['nprobe'] = p
        progress.set_postfix({k: v for k, v in params.items() if k != 'output'})

        scores = None
        scores_file = exp.path_to(f'scores_np{p}.h5')
        if not os.path.exists(scores_file):
            print('Computing scores:', scores_file)
            q = utils.compute_if_dask(q)
            # execute kNN search using k = dataset size
            ranked_sim, ranked_ids = index.search(q, n_points)
            # we need a similarity matrix, we construct it from the ranked results.
            # we fill it initially with the lowest score (not recovered IDs has infinity score)
            if False:  # XXX OPTIMIZED VERSION NOT WORKING!!!!
                ranked_ids = np.ma.array(ranked_ids, mask=(ranked_ids < 0))
                id_order = ranked_ids.argsort(axis=1)
                scores = - ranked_sim[np.arange(q.shape[0]).reshape(-1, 1), id_order]
                del ranked_sim, ranked_ids, id_order
            else:
                scores = np.full((q.shape[0], n_points), np.inf)
                for i, (rsims, rids) in enumerate(zip(ranked_sim, ranked_ids)):
                    for rsim, rid in zip(rsims, rids):
                        if rid > 0:
                            scores[i, rid] = rsim
                scores = -scores

            utils.save_as_hdf5(scores, scores_file, progress=True)

        query_times, query_times_file = exp.require_csv('query_times.csv', index='n_probes')
        for i in trange(1, 6):
            if utils.value_missing(query_times, p, f'query_time_run{i}'):
                q = utils.compute_if_dask(q)
                start = time.time()
                index.search(q, n_points)
                query_time = time.time() - start
                query_times.at[p, f'query_time_run{i}'] = query_time
                query_times.to_csv(query_times_file)

        metrics, metrics_file = exp.require_csv(f'metrics_np{p}.csv')

        if 'ap' not in metrics:
            if scores is None:
                print('Loading scores...')
                scores = utils.load_features(scores_file)
            print('Computing mAP...')
            metrics['ap'] = dataset.score(scores[...], reduction=False, progress=True)
            metrics.to_csv(metrics_file, index=False)

        if 'ndcg' not in metrics:
            dataset._load()  # TODO in y_true getter
            if scores is None:
                print('Loading scores...')
                scores = utils.load_features(scores_file)
            print('Computing nDCG...')
            y_true = dataset.y_true[:, :args.limit] if args.limit else dataset.y_true

            bs = 5
            ndcg = []
            for i in trange(0, y_true.shape[0], bs):
                ndcg.append(dcg(y_true[i:i + bs], scores[i:i + bs], normalized=True))
            ndcg = np.concatenate(ndcg)

            # metrics['ndcg'] = dcg(y_true, scores, normalized=True)
            metrics['ndcg'] = ndcg
            metrics.to_csv(metrics_file, index=False)

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

        metrics['n_probes'] = p
        metrics.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    benchmarks = ('oxford', 'paris', 'holidays', 'oxford+flickr100k', 'holidays+mirflickr1m')

    parser = argparse.ArgumentParser(description='IVFPQ scoring (faiss)')
    parser.add_argument('dataset', choices=benchmarks, help='Benchmark')
    parser.add_argument('features', help='Features dirname')
    parser.add_argument('output', help='Output dir for results')

    parser.add_argument('-l', '--limit', type=int, default=0, help='Distractor set limit (0 = no limit)')

    parser.add_argument('-c', '--code-size', type=int, default=512, help='Code size in bytes')
    parser.add_argument('-n', '--n-cells', type=int, default=None, help='Number of Voronoi cells')

    parser.add_argument('-p', '--pretrained', type=str, help='Path to pretrained index')

    args = parser.parse_args()
    main(args)
