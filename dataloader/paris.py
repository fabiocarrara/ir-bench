import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from .abs_dataset import ComputeAPDataset


class Paris(ComputeAPDataset):

    name = 'oxford-paris'

    def __init__(self, *args, **kwargs):
        self.blacklisted = {"paris_louvre_000136",
                            "paris_louvre_000146",
                            "paris_moulinrouge_000422",
                            "paris_museedorsay_001059",
                            "paris_notredame_000188",
                            "paris_pantheon_000284",
                            "paris_pantheon_000960",
                            "paris_pantheon_000974",
                            "paris_pompidou_000195",
                            "paris_pompidou_000196",
                            "paris_pompidou_000201",
                            "paris_pompidou_000467",
                            "paris_pompidou_000640",
                            "paris_sacrecoeur_000299",
                            "paris_sacrecoeur_000330",
                            "paris_sacrecoeur_000353",
                            "paris_triomphe_000662",
                            "paris_triomphe_000833",
                            "paris_triomphe_000863",
                            "paris_triomphe_000867"}
                                
        super(Paris, self).__init__(*args, **kwargs)
        self.crop_queries()

    def crop_queries(self):
        cropped_queries_dir = os.path.join(self.data_dir, 'roi')
        if os.path.exists(cropped_queries_dir):
            return

        for image_file, roi, roi_file in tqdm(zip(self.query_image_files, self.query_rois, self.query_roi_files)):
            destination_dir = os.path.dirname(roi_file)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            roi = list(map(int, map(round, map(float, roi))))
            Image.open(image_file).crop(roi).save(roi_file)

    def load(self):
        # Loading images
        image_files = os.path.join(self.data_dir, 'jpg', 'paris', '**', '*.jpg')
        image_files = glob.glob(image_files)
        image_files.sort()

        image_ids = map(os.path.basename, image_files)
        image_ids = map(lambda x: os.path.splitext(x)[0], image_ids)
        # filter corrupted images
        image_ids_files = zip(image_ids, image_files)
        image_ids_files = filter(lambda x: x[0] not in self.blacklisted, image_ids_files)
        image_ids, image_files = zip(*image_ids_files)

        self.image_files = np.array(image_files)
        self.image_ids = np.array(image_ids)

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
            query_line = open(query_file, 'r').readline().split()
            query_image_basename, query_roi = query_line[0], query_line[1:]
            query_prefix = os.path.join(self.data_dir, 'lab', query_id)
            query_class = query_image_basename.split('_')[1]
            query_image_file = os.path.join(self.data_dir, 'jpg', 'paris', query_class, query_image_basename + '.jpg')
            query_roi_file = os.path.join(self.data_dir, 'roi', 'paris', query_class, query_image_basename + '.jpg')

            return query_id, query_prefix, query_image_file, query_roi, query_roi_file

        queries = enumerate(query_files)
        queries = map(lambda x: parse_query(x[0], x[1]), queries)
        queries = zip(*queries)
        queries = map(np.array, queries)

        self.query_ids, self.query_prefixes, self.query_image_files, self.query_rois, self.query_roi_files = queries


if __name__ == '__main__':
    paris = Paris()
    ids = paris.images()[0]
    q_ids = paris.queries()[0]

    sim = np.random.rand(len(q_ids), len(ids))
    print(paris.score(sim))
    print(paris.score(paris.y_true))
