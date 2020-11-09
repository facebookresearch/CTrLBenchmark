# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
from collections import defaultdict

import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class Task(object):
    def __init__(self, name, samples, loss_fn, transformation=None,
                 split_names=None, creator=None, source_concepts=None,
                 attributes=None, dim_red='PCA', generator=None,
                 n_samples_per_class=None, save_path=None, id=None):
        """

        :param samples: Iterable containing the data and labels for each split. The length corresponds to the number of
            splits. Each split i should be composed of two Tensors:

            - a `(N_i x ...)` tensor containing the features of the N_i samples for this splits
            - a `(N_i x n_labels)` tensor containing the labels for each attribute that we want to classify. The attributes in different splits are not forced to overlap, allowing to generate ZSL tasks.
        :param transformation:
        :param creator:
        """
        self._infos = {
            'src_concepts': [] if source_concepts is None else source_concepts,
            'transformation': transformation,
            'attributes': attributes
        }
        self.name = name
        self.save_path = None
        self.loss_fn = loss_fn
        self.id = id

        self.split_names = split_names
        self.datasets = [TensorDataset(s_samples, labels.long()) for
                         s_samples, labels in samples]
        self.n_classes = [dataset.tensors[1].max(dim=0).values + 1 for dataset
                          in self.datasets]
        self.x_dim = list(self.datasets[0].tensors[0].size()[1:])
        assert all(list(split.tensors[0].size()[1:]) == self.x_dim for split in
                   self.datasets)
        self.n_samples = [dataset.tensors[0].size(0) for dataset in
                          self.datasets]
        self.n_samples_per_class = n_samples_per_class
        assert all([torch.equal(self.n_classes[0], t) for t in self.n_classes])
        self.n_classes = self.n_classes[0]
        self._dim_reduction = PCA(n_components=3) \
            if dim_red == 'PCA' else TSNE(n_components=3)

        self.creator = creator
        # self.generator = generator
        self.statistics = self.compute_statistics()

        if save_path:
            self.save_path = self.save(save_path)

    def compute_statistics(self):
        train_split = self.datasets[0].tensors[0]
        if train_split[0].dim() == 3:
            # Images
            # assert train_split.size(1) == 3
            n_channels = train_split.size(1)
            mean = [train_split[:, i, :, :].mean() for i in range(n_channels)]
            std = [train_split[:, i, :, :].std() for i in range(n_channels)]
        else:
            # Vectors
            mean = train_split.mean()
            std = train_split.std()

        return {'mean': mean, 'std': std}

    @property
    def concepts(self):
        return [concept for cat_concepts in self.src_concepts for concept in
                cat_concepts]

    @property
    def transformation(self):
        return self._infos['transformation']

    @property
    def src_concepts(self):
        """
        :return: A copy of the concepts list of this task
        """
        return self._infos['src_concepts'].copy()

    @property
    def attributes(self):
        return self._infos['attributes']

    def get_data(self, split:str):
        """

        :param split:
        :type split:
        :return:
        :rtype:
        """
        return self.datasets[split].tensors[0]

    def get_labels(self, split, prop):
        return self.datasets[split].tensors[1][:, prop]

    def plot_task(self, viz, name):
        legend = [str(c) for c in self.src_concepts]

        selected_means = []
        cat_ids = []
        for cat_id, cat in enumerate(self.src_concepts):
            for c in cat:
                if hasattr(c, 'mean'):
                    selected_means.append(c.mean)
                    cat_ids.append(cat_id + 1)

        if len(selected_means) > 2:
            data = torch.stack(selected_means)
            title = '{} selected concepts'.format(name)
            if selected_means[0].numel() > 3:
                title = '{} of {}'.format(
                    self._dim_reduction.__class__.__name__, title)
                data = self._dim_reduction.fit_transform(data)

            viz.scatter(data, Y=cat_ids,
                        opts={'title': title, 'markersize': 3,
                              'legend': legend})

        plot_data = self.get_data(split=0)
        title = '{} features'.format(name)
        if plot_data[0].ndimension() == 3 and plot_data[0].size(0) in [1, 3]:
            # We have an image
            imgs_per_label = defaultdict(list)
            for ds in self.datasets:
                x, y = ds.tensors
                y = y.squeeze()
                for y_val in y.unique():
                    x_sample = random.choice(x[y == y_val])
                    imgs_per_label[y_val.item()].append(x_sample)

            for y, images in imgs_per_label.items():
                grid = torchvision.utils.make_grid(images)
                viz.image(grid, opts={
                    'title': '{} ({})'.format(self.src_concepts[y], y),
                    'width': grid.size(2) * 3,
                    'height': grid.size(1) * 3.2})
        else:
            # Vectorial data
            if plot_data[0].numel() > 3:
                plot_data = self._dim_reduction.fit_transform(
                    plot_data.view(plot_data.size(0), -1))
                title = '{} of {}'.format(
                    self._dim_reduction.__class__.__name__, title)

            viz.scatter(plot_data, Y=self.get_labels(split=0, prop=0) + 1,
                        opts={'title': title, 'webgl': True, 'markersize': 3,
                              'legend': legend})

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        task_datasets = []
        save_paths = []
        for split_data, split_name in zip(self.datasets,
                                          ['train', 'val', 'test']):
            save_path = os.path.join(path,
                                     '{}_{}.pth'.format(self.name, split_name))
            save_paths.append(save_path)
            torch.save(split_data.tensors, save_path)
            task_datasets.append(save_path)
        logger.info('Task saved to {} ...'.format(save_paths))
        metadata_file = os.path.join(path, '{}.meta'.format(self.name))
        torch.save(self._meta(), metadata_file)
        return task_datasets

    def _meta(self):
        meta = {
            'source_concepts': [tuple(str(c) for c in cat) for cat in
                                self.src_concepts],
            'transformation': str(self.transformation),
            'creator': self.creator
        }
        return meta

    def info(self, full=True):
        infos = {
            'data_path': self.save_path,
            'split_names': self.split_names,
            'id': self.id,
            'x_dim': self.x_dim,
            'n_classes': self.n_classes.tolist(),
            'descriptor': self.name,
            'full_descr': str(self),
        }
        if full:
            infos['loss_fn'] = self.loss_fn
            infos['statistics'] = self.statistics

        return infos

    def __repr__(self):
        return "{}-way classification".format(len(self.src_concepts))

    def __str__(self):
        categories = '\n\t-'.join([str(c) for c in self.src_concepts])
        descr = "{}-way classification created by {} ({} samples): \n\t {} \n\t-{}"
        trans_descr = self.transformation
        return descr.format(self.n_classes[0].item(), self.creator,
                            self.n_samples, trans_descr, categories)
