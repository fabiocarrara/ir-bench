import argparse
import itertools
import os
import time

import dask.array as da
import numpy as np
import pandas as pd

from collections import deque
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk, streaming_bulk, scan
from tqdm import tqdm

import utils
from dcg import dcg
from expman import Experiment


def crelu(x):
    fw = da if isinstance(x, da.core.Array) else np
    return fw.hstack([fw.maximum(x, 0), - fw.minimum(x, 0)])


def thr_sq(x, thr, s):
    fw = da if isinstance(x, da.core.Array) else np
    # threshold
    x = fw.maximum(x - (1. / thr), 0)
    x += (1. / thr) * (x > 0)
    # quantize
    x = fw.floor(s * x).astype(int)
    return x


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
    
    
def generate_index_actions(es, index_name, x, x_ids, thr, s, batch_size=1):
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        xb = thr_sq(xb, thr, s)
        xb = utils.compute_if_dask(xb, progress=False)
        id_b = x_ids[i:i + batch_size]
        for xi_id, xi in zip(id_b, xb):
            # if es.exists(index_name, xi_id):
                # tqdm.write(f'Skipping: {xi_id}')
            #    continue
            yield {'_index': index_name, '_id': xi_id, 'repr': surrogate_text(xi)}
        del xb
                    

def main(args):
    es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
    dataset, q, x = utils.load_benchmark(args.dataset, args.features)

    q = utils.load_features(q, chunks=(5000, 2048))
    x = utils.load_features(x, chunks=(5000, 2048))
    n_queries, n_samples = q.shape[0], x.shape[0]

    if args.limit:
        x = x[:args.limit]

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
            q_sq = thr_sq(q, thr, s)
            q_density = (q_sq != 0).mean(axis=0)
            q_density = utils.compute_if_dask(q_density)
            density['query_density'] = q_density
            density.to_csv(density_file, index=False)

        if 'database_density' not in density:
            progress.write('Computing database density ...')
            x_sq = thr_sq(x, thr, s)
            x_density = (x_sq != 0).mean(axis=0)
            x_density = utils.compute_if_dask(x_density)
            density['database_density'] = x_density
            density.to_csv(density_file, index=False)

        index_name = exp.name.lower()
        if not es.indices.exists(index_name) or es.count(index=index_name)['count'] < n_samples or args.force:
            # x_sq = thr_sq(x, thr, s)
            x_ids, _ = dataset.images()

            index_actions = generate_index_actions(es, index_name, x, x_ids, thr, s, 50)
            # index_actions = tqdm(index_actions, total=n_samples)

            progress.write(f'Indexing: {index_name}')

            index_config = {
                "mappings": {
                    "_source": {"enabled": False},  # do not store STR
                    "properties": {"repr": {"type": "text"}}  # FULLTEXT
                },
                "settings": {
                    "index": {"number_of_shards": 1, "number_of_replicas": 0},
                    "analysis": {"analyzer": {"first": {"type": "whitespace"}}}
                }
            }
            
            # es.indices.delete(index_name, ignore=(400, 404))
            es.indices.create(index_name, index_config, ignore=400)
            es.indices.put_settings({"index": {"refresh_interval": "-1", "number_of_replicas": 0}}, index_name)

            indexing = parallel_bulk(es, index_actions, thread_count=4, chunk_size=150, max_chunk_bytes=2**26)
            indexing = tqdm(indexing, total=n_samples)
            start = time.time()            
            deque(indexing, maxlen=0)
            add_time = time.time() - start
            progress.write(f'Index time: {add_time}')

            es.indices.put_settings({"index": {"refresh_interval": "1s"}}, index_name)
            es.indices.refresh()

            index_stats_file = exp.path_to('index_stats.csv')
            index_stats = pd.DataFrame({'add_time': add_time}, index=[0])
            index_stats.to_csv(index_stats_file, index=False)

        metrics, metrics_file = exp.require_csv(f'metrics.csv')

        scores = None
        scores_file = exp.path_to(f'scores.h5')
        if not os.path.exists(scores_file):
            progress.write('Computing scores...')

            xid2idx = {k: i for i, k in enumerate(dataset.images()[0])}
            q_sq = thr_sq(q, thr, s)
            q_sq = utils.compute_if_dask(q_sq, progress=False)

            scores = np.zeros((n_queries, n_samples), dtype=np.float32)
            query_times = []
            
            for i, qi in enumerate(tqdm(q_sq)):
                query = {
                    "query": {"query_string": {"default_field": "repr", "query": surrogate_text(qi, boost=True)}},
                    # "from": 0, "size": n_samples
                }
                start = time.time()
                for hit in tqdm(scan(es, query, index=index_name, preserve_order=True), total=n_samples):
                    j = xid2idx[hit['_id']]
                    scores[i, j] = hit['_score']

                query_times.append(time.time() - start)
            metrics['query_time'] = query_times
            metrics.to_csv(metrics_file, index=False)
            progress.write(f'Query time: {metrics.query_time.sum()}')
            utils.save_as_hdf5(scores, scores_file, progress=True)

        if 'ap' not in metrics:
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
            metrics['ndcg'] = dcg(dataset.y_true, scores, normalized=True)
            metrics.to_csv(metrics_file, index=False)
            progress.write(f'nDCG: {metrics.ndcg.mean()}')


if __name__ == '__main__':
    thrs = (range(2, 10),  # 2 to 10 excl.
            range(10, 50, 2),  # 10 to 50 excl. step 2
            range(50, 100, 10),  # 50 to 100 excl. step 10
            range(100, 1000, 100),  # 100 to 1000 excl. step 100
            range(1000, 5001, 1000))  # 1000 to 5000 step 1000
    thrs = list(itertools.chain.from_iterable(thrs))
    sq_f = thrs

    benchmarks = ('oxford', 'paris', 'holidays', 'oxford+flickr100k', 'holidays+mirflickr1m')

    parser = argparse.ArgumentParser(description='Scalar Quantization scoring')
    parser.add_argument('dataset', choices=benchmarks, help='Benchmark')
    parser.add_argument('features', help='Features dirname')
    parser.add_argument('output', help='Output dir for results')

    parser.add_argument('-f', '--force', default=False, action='store_true', help='Force indexing')
    parser.add_argument('-l', '--limit', type=int, default=0, help='Distractor set limit (0 = no limit)')

    parser.add_argument('-c', '--crelu', action='store_true', default=False, help='Use CReLU')
    parser.add_argument('-t', '--threshold', type=int, nargs='+', default=thrs,
                        help='Thresholding factor (multiple values accepted)')
    parser.add_argument('-q', '--sq-factor', type=int, nargs='+', default=sq_f,
                        help='Scalar quantization factor (multiple values accepted)')

    args = parser.parse_args()
    main(args)
