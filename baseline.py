import argparse
import utils
import subprocess

import numpy as np
import dask.array as da

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline scoring (dot product)')
    parser.add_argument('dataset'), #choices=('oxford', 'paris'), help='Benchmark')
    parser.add_argument('features', help='Features dirname')
    parser.add_argument('-r', '--rotate', help='.npy file containing the random rotation to apply')
    args = parser.parse_args()

    dataset, q, x = utils.load_benchmark(args.dataset, args.features)
    x = utils.load_features(x, chunks=(1000, 2048))
    q = utils.load_features(q, chunks=(1000, 2048))

    x /= da.sqrt((x**2).sum(axis=1, keepdims=True))
    q /= da.sqrt((q**2).sum(axis=1, keepdims=True))

    if args.rotate:
        R = np.load(args.rotate)
        q = q.dot(R.T)
        x = x.dot(R.T)
        x -= x.mean(axis=0)
    
    scores = q.dot(x.T)
    scores = utils.compute_if_dask(scores)
    dataset._load()
    
    mean_ap = dataset.score(scores)
    print(mean_ap)
    
    """ CONFIRMED THAT compute_ap WORKS
    eval_bin = 'eval_bin/compute_ap'

    aps = []
    for i, scores_i in enumerate(tqdm(scores)):
        tmp_rnk = f'tmp/{dataset.query_ids[i]}.rnk'
        rank = scores_i.argsort()[::-1]
        
        with open(tmp_rnk, 'w') as f:
            f.write('\n'.join(dataset.image_ids[rank]))
            
        q_id = 'data/oxford-buildings/lab/' + dataset.query_ids[i]

        p = subprocess.Popen([eval_bin, q_id, tmp_rnk], stdout=subprocess.PIPE)
        out, _ = p.communicate()
        aps.append(float(out))
    
    print(np.mean(aps))
    """
        
        


