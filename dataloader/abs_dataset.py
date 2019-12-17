import os
import subprocess

import numpy as np
from joblib import delayed, Parallel
from tqdm import tqdm


class IRDataset(object):

    name = None

    def __init__(self, data_dir='data', download=False):
        self.data_dir = os.path.join(data_dir, self.name)
        self.download = download
        
        self.image_ids = None
        self.image_files = None

        self.loaded = False

    def download_if_missing(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _load(self):
        if not self.loaded:
            if self.download:
                self.download_if_missing()
            self.load()
            self.loaded = True

    def images(self):
        self._load()
        return self.image_ids, self.image_files


class ScoreableDataset(IRDataset):

    def __init__(self, *args, distractor=None, **kwargs):
        super(ScoreableDataset, self).__init__(*args, **kwargs)
        
        self.query_ids = None
        self.query_image_files = None

        self.distractor = distractor

        self.y_true = None  # Query x Dataset

    def _load(self):
        if not self.loaded:
            self.load()

            if isinstance(self.distractor, IRDataset):
                self.distractor._load()
                self.image_ids = np.array(self.image_ids.tolist() + self.distractor.image_ids.tolist())
                self.image_files = np.array(self.image_files.tolist() + self.distractor.image_files.tolist())
                d = len(self.distractor.image_ids)
                self.y_true = np.pad(self.y_true, ((0, 0), (0, d)), 'constant', constant_values=0)

            self.loaded = True

    def _score(self, sim):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        self._load()
        return self._score(*args, **kwargs)


def _score_rnk_with_computeap(rnk, query_prefix, image_ids, eval_bin):
    ranked_ids = '\n'.join(image_ids[rnk])
    p = subprocess.Popen([eval_bin, query_prefix], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = p.communicate(input=ranked_ids.encode('utf8'))
    return float(out)


def _sort_score_rnk_with_computeap(sim, query_prefix, image_ids, eval_bin):
    rnk = sim.argsort()[::-1]
    ranked_ids = '\n'.join(image_ids[rnk])
    p = subprocess.Popen([eval_bin, query_prefix], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = p.communicate(input=ranked_ids.encode('utf8'))
    return float(out)


class ComputeAPDataset(ScoreableDataset):

    def __init__(self, *args, eval_bin='./eval_bin/compute_ap_stdin', **kwargs):
        super(ComputeAPDataset, self).__init__(*args, **kwargs)
        
        self.eval_bin = eval_bin
        
        self.query_rois = None
        self.query_prefixes = None
        self.query_roi_files = None

    def queries(self):
        return self.query_ids, self.query_prefixes, self.query_image_files, self.query_rois, self.query_roi_files

    def _score(self, sim, reduction=True, progress=False):
        # rnks = sim.argsort(axis=1)[:, ::-1]

        progress = tqdm(zip(sim, self.query_prefixes), total=sim.shape[0], disable=not progress)

        aps = Parallel(n_jobs=-1, backend="threading")(
            delayed(_sort_score_rnk_with_computeap)(r, qp, self.image_ids, self.eval_bin) for r, qp in progress
        )
        
        if reduction:
            return np.mean(aps)
        
        return aps

    def save_order(self, dest_dir=None):

        if dest_dir is None:
            dest_dir = self.data_dir

        def relative(x):
            return x[len(self.data_dir)+1:]

        images_list = os.path.join(dest_dir, 'images.txt')
        with open(images_list, 'w') as f:
            rel_image_files = map(relative, self.image_files)
            lines = map('\t'.join, zip(self.image_ids, rel_image_files))
            f.write('\n'.join(lines))

        query_list = os.path.join(dest_dir, 'queries.txt')
        with open(query_list, 'w') as f:
            rel_query_image_files = map(relative, self.query_image_files)
            things_to_write = [self.query_ids, rel_query_image_files]

            if self.query_rois is not None:
                rois_as_str = map(lambda x: ' '.join(map(str, x.tolist())), self.query_rois)
                rel_query_roi_files = map(relative, self.query_roi_files)

                things_to_write.extend([rois_as_str, rel_query_roi_files])

            lines = map('\t'.join, zip(*things_to_write))
            f.write('\n'.join(lines))
