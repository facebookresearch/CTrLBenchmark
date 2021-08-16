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
from torchvision.datasets.utils import gen_bar_updater

from tqdm import tqdm
import tarfile
import urllib

def center_crop(img, size):
    width, height = img.size

    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2

    return img.crop((left, top, right, bottom))

def untar(path):
    directory_path = os.path.dirname(path)
    with tarfile.open(path) as tar_file:
        tar_file.extractall(directory_path)

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, url.split("/")[-1])
    if os.path.exists(file_name):
        print(f"Dataset already downloaded at {file_name}.")
    else:
        urllib.request.urlretrieve(url, file_name, reporthook=gen_bar_updater())
    return file_name

class DTD(VisionDataset):
    folder_name = 'dtd'
    processed_folder = 'dtd/processed'
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'

    def __init__(self, root, split, download=False, img_size=(32, 32),
                 transform=None, target_transform=None):
        super(DTD, self).__init__(root, transform=transform,
                                  target_transform=target_transform)

        pr_file = self._get_processed_name(root, split, img_size)
        if download and not os.path.exists(pr_file):
            self._download_and_prepare(root, split, img_size)

        assert os.path.exists(pr_file)
        self.data, self.labels, self.classes = torch.load(pr_file)

    def _download_and_prepare(self, root, split, img_size):
        archive_path = os.path.join(root, "dtd-r1.0.1.tar.gz")
        if not os.path.exists(archive_path):
            print("Downloading DTD dataset...")
            download(self.url, root)

        if not os.path.exists(os.path.join(root, "dtd")):
            print("Uncompressing images...")
            untar(archive_path)
        self._prepare(root, split, img_size)

    def _prepare(self, root, split, img_size):
        extracted_path = os.path.join(root, self.folder_name)
        assert os.path.exists(extracted_path)
        os.makedirs(os.path.join(root, self.processed_folder), exist_ok=True)

        images = defaultdict(list)
        classes = []
        data = []
        labels = []

        labels_path = os.path.join(extracted_path, 'labels')
        p = os.path.join(labels_path, '{}1.txt'.format(split))
        files = glob.glob(p)
        if split == 'train':
            p = os.path.join(labels_path, 'val1.txt')
            files = itertools.chain(files, glob.glob(p))
        files = list(files)
        for file in tqdm(files):
            with open(file, 'r') as f:
                for line in f:
                    line = line.rstrip('\n')
                    img = pil_loader(os.path.join(extracted_path, 'images', line))
                    width, height = img.size
                    ratio = max(img_size[0] / width, img_size[1] / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size)
                    img = center_crop(img, img_size)
                    assert img.size == img_size
                    cat, img_name = line.split('/')
                    images[cat].append(np.array(img))

        for i, (cl, imgs) in enumerate(sorted(images.items())):
            classes.append(cl)
            data.append(np.stack(imgs))
            labels.append(np.ones(len(imgs)) * i)

        data = np.concatenate(data)
        labels = np.concatenate(labels)
        file_name = self._get_processed_name(root, split, img_size)
        print('Saving data to {}'.format(file_name))
        torch.save((data, labels, classes), file_name)

    def _get_processed_name(self, root, split, img_size):
        pr_file = 'dtd-{}-{}x{}.th'.format(split, *img_size)
        return os.path.join(root, self.processed_folder, pr_file)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

