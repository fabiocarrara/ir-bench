import os
import numpy as np

from .abs_dataset import IRDataset


class MIRFlickr1M(IRDataset):

    name = 'mirflickr1m'

    def __init__(self, *args, **kwargs):
        super(MIRFlickr1M, self).__init__(*args, **kwargs)

        self.corrupted = {
            '5/59898.jpg',
            '10/104442.jpg',
            '10/107349.jpg',
            '10/108460.jpg',
            '68/686806.jpg'
        }

    def load(self):
        image_files = os.path.join(self.data_dir, 'image_list.txt')
        image_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(image_files):
            cmd = 'find {} -name "*.jpg" | rev | cut -d/ -f-2 | rev | sort > {}'.format(image_dir, image_files)
            print('Creating image list: {}'.format(image_files))
            os.system(cmd)

        with open(image_files, 'r') as f:
            image_files = map(str.rstrip, f.readlines())
            image_files = filter(lambda x: x not in self.corrupted, image_files)
            image_files = list(image_files)
            
        image_ids = map(lambda x: x.replace('/','_').split('.')[0], image_files)
        image_ids = [f'mirflickr1m_{i}' for i in image_ids]
        
        image_files = map(lambda x: os.path.join(image_dir, x), image_files)
        image_files = list(image_files)

        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)


if __name__ == '__main__':
    dataset = MIRFlickr1M()
    print(len(dataset.image_ids))
