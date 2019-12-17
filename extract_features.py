import argparse
import os

import h5py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.models import resnext101_32x8d
from torchvision.transforms import transforms
from tqdm import tqdm

import utils


class URLDataset(torch.utils.data.Dataset):

    def __init__(self, urls, transform=None):
        self.urls = urls
        self.transform = transform

    def __getitem__(self, item):
        image = default_loader(self.urls[item])
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.urls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Features from Pretrained Models')
    parser.add_argument('dataset', type=str,
                        choices=('oxford', 'paris', 'holidays', 'oxford+flickr100k', 'holidays+mirflickr1m'),
                        help='Image Dataset')
    parser.add_argument('features', type=str, help='Features name')

    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')

    args = parser.parse_args()

    dataset, q, x = utils.load_benchmark(args.dataset, args.features)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = resnext101_32x8d(pretrained=True).eval()  # your model here
    model.fc = torch.nn.Sequential()  # remove last classification layer
    model = model.cuda()
    with torch.no_grad():
        ex = model(torch.empty(1, 3, 224, 224).cuda()).squeeze().cpu().numpy()


    def _extract_from_urls(urls, out_file):
        data = URLDataset(urls, transform=transform)
        data = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        progress = tqdm(data)
        with torch.no_grad(), h5py.File(out_file, 'a') as out:
            shape = (len(urls),) + ex.shape
            features = out.require_dataset('data', shape=shape, dtype=ex.dtype)
            n_processed = 0
            for x in progress:
                b = x.shape[0]
                if n_processed > 5060:
                    f = model(x.cuda()).cpu().numpy()
                    features[n_processed:n_processed + b, :] = f
                n_processed += b

    base_dir = os.path.dirname(x)
    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(x):
        print('Extracting:', x)
        _, urls = dataset.images()
        print(len(urls))
        _extract_from_urls(urls, x)

    if not os.path.exists(q):
        print('Extracting:', q)
        dataset._load()  # TODO move in dataset class
        query_data = dataset.queries()
        query_ids, query_prefixes, query_image_files, query_rois, query_roi_files = query_data
        query_urls = query_roi_files if query_roi_files is not None else query_image_files
        _extract_from_urls(query_urls, q)
