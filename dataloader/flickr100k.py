import os
import numpy as np

from .abs_dataset import IRDataset


class Flickr100K(IRDataset):

    name = 'flickr100k'

    def __init__(self, *args, **kwargs):
        super(Flickr100K, self).__init__(*args, **kwargs)

        self.corrupted = {
            '06/06_000443.jpg',
            'autumn/autumn_000335.jpg',
            'baby/baby_001317.jpg',
            'cat/cat_001894.jpg',
            'festival/festival_001907.jpg',
            'green/green_002662.jpg',
            'park/park_001319.jpg',
            'portrait/portrait_000801.jpg',
            'river/river_001733.jpg',
            'school/school_001535.jpg',
        }

    def load(self):
        image_files = os.path.join(self.data_dir, 'image_list.txt')
        image_dir = os.path.join(self.data_dir, 'oxc1_100k')
        if not os.path.exists(image_files):
            cmd = 'find {} -name "*.jpg" | rev | cut -d/ -f-2 | rev | sort > {}'.format(image_dir, image_files)
            print('Creating image list: {}'.format(image_files))
            os.system(cmd)

        with open(image_files, 'r') as f:
            image_files = map(str.rstrip, f.readlines())
            image_files = filter(lambda x: x not in self.corrupted, image_files)
            image_files = list(image_files)
            
        image_ids = map(lambda x: x.replace('/','_').split('.')[0], image_files)
        image_ids = [f'flickr100k_{i}' for i in image_ids]
        
        image_files = map(lambda x: os.path.join(image_dir, x), image_files)
        image_files = list(image_files)

        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)


if __name__ == '__main__':
    dataset = Flickr100K('/home/fabio/SLOW/Datasets/Flickr100k')
    print(len(dataset.image_ids))
