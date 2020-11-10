# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import itertools
import os
from collections import defaultdict

import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def center_crop(img, size):
    width, height = img.size

    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2

    return img.crop((left, top, right, bottom))


class Aircraft(VisionDataset):
    folder_name = 'fgvc-aircraft-2013b'
    processed_folder = os.path.join('fgvc-aircraft-2013b/processed')
    url = 'https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'

    def __init__(self, root, split, task='variant', download=False,
                 img_size=(72, 72), transform=None, target_transform=None):
        super(Aircraft, self).__init__(root, transform=transform,
                                  target_transform=target_transform)

        pr_file = self._get_processed_name(root, task, split, img_size)
        if download and not os.path.exists(pr_file):
            self._download_and_prepare(root, task, split, img_size)

        assert os.path.exists(pr_file)
        self.data, self.labels, self.classes = torch.load(pr_file)

    def _download_and_prepare(self, root, task, split, img_size):
        download_and_extract_archive(self.url, download_root=root, )
        self._prepare(root, task, split, img_size)

    def _prepare(self, root, task, split, img_size):
        extracted_path = os.path.join(root, self.folder_name)
        assert os.path.exists(extracted_path)
        os.makedirs(os.path.join(root, self.processed_folder), exist_ok=True)

        images = defaultdict(list)
        classes = []
        data = []
        labels = []

        labels_path = os.path.join(extracted_path, 'data')
        p = os.path.join(labels_path, 'images_{}_{}.txt'.format(task, split))
        files = glob.glob(p)
        if split == 'train':
            p = os.path.join(labels_path, 'images_{}_val.txt'.format(task))
            files = itertools.chain(files, glob.glob(p))
        for file in files:
            with open(file, 'r') as f:
                for line in tqdm(f):
                    img, cat = line.split(maxsplit=1)
                    cat = cat[:-1]
                    img = pil_loader(os.path.join(extracted_path, 'data',
                                                  'images', f'{img}.jpg'))
                    # Remove the 20 pixel bottom banner
                    width, height = img.size
                    img = img.crop((0, 0, width, height - 20))
                    width, height = img.size
                    ratio = min(img_size[0] / width, img_size[1] / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(img_size)
                    # img = center_crop(img, img_size)
                    assert img.size == img_size
                    images[cat].append(np.array(img))

        for i, (cl, imgs) in enumerate(sorted(images.items())):
            classes.append(cl)
            data.append(np.stack(imgs))
            labels.append(np.ones(len(imgs)) * i)

        data = np.concatenate(data)
        labels = np.concatenate(labels)
        file_name = self._get_processed_name(root, task, split, img_size)
        print('Saving data to {}'.format(file_name))
        torch.save((data, labels, classes), file_name)

    def _get_processed_name(self, root, task, split, img_size):
        pr_file = 'aircraft-{}-{}-{}x{}.th'.format(task, split, *img_size)
        return os.path.join(root, self.processed_folder, pr_file)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

