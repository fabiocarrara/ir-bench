import os

import dask.array as da
import h5py
import numpy as np
import subprocess
from dask.diagnostics import ProgressBar

from dataloader import Oxford, Paris, Holidays, Flickr100K, MIRFlickr1M


def get_folder_size(root):
    p = subprocess.Popen(['du', root], stdout=subprocess.PIPE)
    out, err = p.communicate()
    return int(out.split('\t')[0])


def value_missing(dataframe, row, col):
    if (dataframe.empty or
            col not in dataframe or
            row not in dataframe.index):
        return True

    return np.isnan(dataframe.at[row, col])


def is_dask(x):
    return isinstance(x, da.core.Array)


def load_features(path, hdf5_path='data', chunks=None):
    if path.endswith('.npy'):
        x = np.load(path, mmap_mode='r')

    elif path.endswith('.hdf5') or path.endswith('.h5'):
        x = h5py.File(path, 'r')[hdf5_path]

    if chunks:
        x = da.from_array(x, chunks=chunks)

    return x


def load_benchmark(dataset, features=None, preproc='rot'):
    if 'oxford' in dataset:
        data_class = Oxford
    elif 'paris' in dataset:
        data_class = Paris
    elif 'holidays' in dataset:
        data_class = Holidays

    distractor = None
    if '+' in dataset:
        if '+flickr100k' in dataset:
            distractor_data_class = Flickr100K
        elif '+mirflickr1m' in dataset:
            distractor_data_class = MIRFlickr1M

        distractor = distractor_data_class()

    data = data_class(distractor=distractor)

    if features:
        queries_f = os.path.join('features', features, f'{dataset}_queries.h5')
        dataset_f = os.path.join('features', features, f'{dataset}_dataset.h5')
        return data, queries_f, dataset_f

    return data


def compute_if_dask(x, progress=True):
    if not is_dask(x):
        return x

    if progress:
        with ProgressBar():
            return x.compute()

    return x.compute()


def free_memory():
    with open('/proc/meminfo', 'r') as f:
        lines = f.readlines()
        lines = filter(lambda x: 'MemAvailable' in x, lines)
        line = next(lines)

    mem_free_kb = line.split(':')[1].strip()[:-3]
    mem_free_kb = int(mem_free_kb)

    return mem_free_kb


def save_as_hdf5(x, path, progress=False):
    if is_dask(x):
        if progress:
            with ProgressBar():
                da.to_hdf5(path, 'data', x, compression="gzip", compression_opts=9)
        else:
            da.to_hdf5(path, 'data', x, compression="gzip", compression_opts=9)
    else:
        f = h5py.File(path, 'w')
        f.create_dataset('data', data=x, compression="gzip", compression_opts=9)
