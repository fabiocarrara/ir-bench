import argparse
import itertools
import os

import dask.array as da
import numpy as np
from tqdm import tqdm

import utils
from dcg import dcg
from expman import Experiment


def crelu(x):
    fw = da if isinstance(x, da.core.Array) else np
    return fw.hstack([fw.maximum(x, 0), - fw.minimum(x, 0)])


def threshold(x, thr, s=0):
    fw = da if isinstance(x, da.core.Array) else np
    # threshold
    tmp = fw.maximum(fw.fabs(x) - (1. / thr), 0)
    tmp += (1. / thr) * (tmp > 0)
    x = fw.copysign(tmp, x)
    # scalar quantization (optional)
    x = fw.fix(s * x).astype(int) if s else x  
    return x


if __name__ == '__main__':

    thrs = (range(20, 50, 4),
            range(50, 100, 10),  # 50 to 100 excl. step 10
            range(10, 50, 2),  # 10 to 50 excl. step 2
            range(100, 1000, 100),  # 100 to 1000 excl. step 100
            range(1000, 5001, 1000),  # 1000 to 5000 step 1000
            range(2, 10))  # 2 to 10 excl. (less important)
    thrs = list(itertools.chain.from_iterable(thrs))

    benchmarks = ('oxford', 'paris', 'holidays', 'oxford+flickr100k', 'holidays+mirflickr1m')

    parser = argparse.ArgumentParser(description='Scalar Quantization scoring')
    parser.add_argument('dataset', choices=benchmarks, help='Benchmark')
    parser.add_argument('features', help='Features dirname')
    parser.add_argument('output', help='Output dir for results')

    parser.add_argument('-c', '--crelu', action='store_true', default=False, help='Use CReLU')
    parser.add_argument('-t', '--threshold', nargs='+', default=thrs, help='Thresholding factor (multiple values accepted)')
    parser.add_argument('-l', '--limit', default=0, help='Distractor set limit (0 = no limit)')

    args = parser.parse_args()
    params = vars(args)

    dataset, q, x = utils.load_benchmark(args.dataset, args.features)

    q = utils.load_features(q, chunks=(2500, 2048))
    x = utils.load_features(x, chunks=(2500, 2048))

    if args.limit:
        x = x[:args.limit]

    if args.crelu:
        q = crelu(q)
        x = crelu(x)

    progress = tqdm(args.threshold)
    for thr in progress:
        params['threshold'] = thr
        progress.set_postfix({k: v for k, v in params.items() if k != 'output'})
        exp = Experiment(params, root=args.output, ignore=('output',))

        density, density_file = exp.require_csv(f'density.csv')
        if 'query_density' not in density:
            q_thr = threshold(q, thr)
            q_density = (q_thr != 0).mean(axis=0)
            q_density = utils.compute_if_dask(q_density)
            density['query_density'] = q_density
            density.to_csv(density_file, index=False)

        if 'database_density' not in density:
            x_thr = threshold(x, thr)
            x_density = (x_thr != 0).mean(axis=0)
            x_density = utils.compute_if_dask(x_density)
            density['database_density'] = x_density
            density.to_csv(density_file, index=False)

        scores = None
        scores_file = exp.path_to(f'scores.h5')
        if not os.path.exists(scores_file):
            print('Computing scores...')
            q_thr = threshold(q, thr)
            x_thr = threshold(x, thr)

            scores = q_thr.dot(x_thr.T)
            scores = utils.compute_if_dask(scores)
            utils.save_as_hdf5(scores, scores_file, progress=True)

        metrics, metrics_file = exp.require_csv(f'metrics.csv')
        if 'ap' not in metrics:
            if scores is None:
                print('Loading scores...')
                scores = utils.load_features(scores_file)[...]
            print('Computing mAP...')
            metrics['ap'] = dataset.score(scores, reduction=False, progress=True)
            metrics.to_csv(metrics_file, index=False)

        if 'ndcg' not in metrics:
            dataset._load()  # TODO in y_true getter
            if scores is None:
                print('Loading scores...')
                scores = utils.load_features(scores_file)[...]
            print('Computing nDCG...')
            metrics['ndcg'] = dcg(dataset.y_true, scores, normalized=True)
            metrics.to_csv(metrics_file, index=False)
