import argparse
import utils

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotate and zero-mean features')
    parser.add_argument('features', help='Features to be rotated (prefix)')
    parser.add_argument('rotation', help='Path to npy rotation matrix')
    parser.add_argument('output', help='Output file for rotated features (prefix)')
    args = parser.parse_args()

    q = args.features + '_queries.h5'
    x = args.features + '_dataset.h5'
    
    q = utils.load_features(q, chunks=(5000, 2048))
    x = utils.load_features(x, chunks=(5000, 2048))
    
    R = np.load(args.rotation).astype(np.float32)
    x = x.dot(R.T)
    x -= x.mean(axis=0)
    
    q = q.dot(R.T)
    
    q_out = args.output + '_queries.h5'
    x_out = args.output + '_dataset.h5'
    
    utils.save_as_hdf5(q, q_out, progress=True)
    utils.save_as_hdf5(x, x_out, progress=True)
    

