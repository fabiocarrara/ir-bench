import glob
import itertools
import os

import numpy as np
from .abs_dataset import ComputeAPDataset


class Holidays(ComputeAPDataset):
    name = 'inria-holidays'
    URL1 = 'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz'
    URL2 = 'ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz'

    def __init__(self, *args, **kwargs):
        super(Holidays, self).__init__(*args, **kwargs)

    def download_if_missing(self, data_dir, download):
        jpg_dir = os.path.join(data_dir, 'jpg')
        if os.path.exists(jpg_dir):
            return

        assert download, 'Data not found, not downloading..'

        print('Downloading and extracting INRIA Holidays: jpg1.tar.gz')
        os.system('wget {} -O - | tar -xz -C {}'.format(self.URL1, data_dir))
        print('Downloading and extracting INRIA Holidays: jpg2.tar.gz')
        os.system('wget {} -O - | tar -xz -C {}'.format(self.URL2, data_dir))

    def prepare_gt(self):
        gt_dir = os.path.join(self.data_dir, 'lab')
        if os.path.exists(gt_dir):
            return

        os.makedirs(gt_dir)
        for query_name, relevant in itertools.groupby(self.image_ids, key=lambda x: x[:4]):
            query = next(relevant)
            relevant = list(relevant)

            query_file = os.path.join(gt_dir, '{}_query.txt'.format(query_name))
            good_file = os.path.join(gt_dir, '{}_good.txt'.format(query_name))
            junk_file = os.path.join(gt_dir, '{}_junk.txt'.format(query_name))
            ok_file = os.path.join(gt_dir, '{}_ok.txt'.format(query_name))

            open(query_file, 'w').write(query)               # create QUERY file with the query itself
            open(good_file, 'w').write('\n'.join(relevant))  # create GOOK file excluding the query itself
            open(junk_file, 'a').write(query + '\n')         # create JUNK file with the query only
            open(ok_file, 'a').close()                       # create OK file empty

    def load(self):
        # Loading images
        image_files = os.path.join(self.data_dir, 'jpg', '*.jpg')
        image_files = glob.glob(image_files)
        image_files.sort()

        image_ids = map(os.path.basename, image_files)
        image_ids = map(lambda x: os.path.splitext(x)[0], image_ids)
        image_ids = list(image_ids)

        self.image_files = np.array(image_files)
        self.image_ids = np.array(image_ids)

        self.prepare_gt()

        # Loading queries
        query_files = os.path.join(self.data_dir, 'lab', '*_query.txt')
        query_files = glob.glob(query_files)
        query_files.sort()

        self.y_true = np.zeros((len(query_files), len(image_ids)), dtype=bool)  # TODO sparse!

        def parse_query(idx, query_file):

            def _parse_query_gt(gt_type):
                query_gt = query_file.replace('_query.txt', '_{}.txt'.format(gt_type))
                if os.path.exists(query_gt):
                    for relevant_id in open(query_gt, 'r'):
                        relevant_idx = np.where(self.image_ids == relevant_id.rstrip())
                        self.y_true[idx][relevant_idx] = 1

            _parse_query_gt('good')
            _parse_query_gt('ok')

            query_id = os.path.basename(query_file)[:-len('_query.txt')]
            query_image_basename = open(query_file, 'r').readline().rstrip()
            query_prefix = os.path.join(self.data_dir, 'lab', query_id)

            query_image_file = os.path.join(self.data_dir, 'jpg', query_image_basename + '.jpg')

            return query_id, query_prefix, query_image_file

        queries = enumerate(query_files)
        queries = map(lambda x: parse_query(x[0], x[1]), queries)
        queries = zip(*queries)
        queries = map(np.array, queries)

        self.query_ids, self.query_prefixes, self.query_image_files = queries


if __name__ == '__main__':
    # from flickr100k import Flickr100K

    # flickr_dir = '/home/fabio/SLOW/Datasets/Flickr100k'
    # flickr100k = Flickr100K(flickr_dir)

    data_dir = '/home/fabio/SLOW/Datasets/holidays'
    eval_bin = '/home/fabio/Code/ir-bench/eval_bin/compute_ap_stdin'
    dset = Holidays(data_dir, eval_bin=eval_bin, download=True)  # , distractor=flickr100k)

    ids, _ = dset.images()
    q_ids = dset.queries()[0]

    sim = np.random.rand(len(q_ids), len(ids))
    print(sim.shape)
    print(dset.score(sim))
    print(dset.score(dset.y_true))
    dset.save_order()
