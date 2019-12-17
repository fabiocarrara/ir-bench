import argparse
import os
import numpy as np

import utils


def main(args):

    _, q, x = utils.load_benchmark(args.dataset, args.features)

    q = utils.load_features(q, chunks=(2500, 2048))
    x = utils.load_features(x, chunks=(2500, 2048))
    dim = q.shape[1]

    if args.random_rot is not None:
        rot = args.random_rot
        rot = os.path.join('features', 'random_ortho', f'rand_ortho_{dim}_{rot}.npy')
        rot = np.load(rot).astype(np.float32)
        q = q.dot(rot.T)
        x = x.dot(rot.T)

    # centering
    x_mean = x.mean(axis=0)
    q -= x_mean
    x -= x_mean

    out_dir = os.path.join('features', args.output)
    os.makedirs(out_dir, exist_ok=True)
    _, q_out, x_out = utils.load_benchmark(args.dataset, args.output)

    if not os.path.exists(q_out) or args.force:
        utils.save_as_hdf5(q, q_out, progress=True)
    if not os.path.exists(x_out) or args.force:
        utils.save_as_hdf5(x, x_out, progress=True)


if __name__ == '__main__':
    benchmarks = ('oxford', 'paris', 'holidays', 'oxford+flickr100k', 'holidays+mirflickr1m')

    parser = argparse.ArgumentParser(description='Rotate and Center Features')
    parser.add_argument('dataset', choices=benchmarks, help='Benchmark')
    parser.add_argument('features', help='Input features dirname')
    parser.add_argument('output', help='Output features dirname')

    parser.add_argument('-r', '--random-rot', type=int, default=None, help='Path to npy random orthogonal transformation')
    parser.add_argument('-f', '--force', default=False, action='store_true', help='Force computation')

    args = parser.parse_args()
    main(args)
