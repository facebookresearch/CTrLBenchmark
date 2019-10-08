# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import visdom
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset

from src.tasks.Category import Category


class Task(object):
    def __init__(self, concepts, transformations, available_transformations, n_samples_per_class):
        self.concepts = concepts
        self.transformations = transformations
        self.n_samples_per_class = n_samples_per_class

        categories = []
        samples = []
        labels = []

        for class_id, (concept, n_samples) in enumerate(zip(concepts, self.n_samples_per_class)):
            sampling_distrib = available_transformations.apply_transformation(concept, transformations)
            new_cat = Category(concept, sampling_distrib)
            categories.append(new_cat)
            samples.append(torch.from_numpy(new_cat.get_samples(n_samples)).float())
            labels.append(torch.ones(n_samples) * class_id)

        self.dataset = TensorDataset(torch.cat(samples), torch.cat(labels).long())

    def mutate(self):

        return self

    def plot_task(self, viz):
        if self.dataset[0][0].numel() <= 3:
            viz.scatter(self.dataset.tensors[0], Y=self.dataset.tensors[1] + 1,
                        opts={'title': 'Real features',
                              'webgl': True,
                              'markersize': 3,
                              'legend': [c.descriptor for c in self.concepts]})

        X_pca = PCA(n_components=3).fit_transform(self.dataset.tensors[0])
        viz.scatter(X_pca, Y=self.dataset.tensors[1] + 1,
                    opts={'title': 'PCA',
                          'webgl': True,
                          'markersize': 3,
                          'legend': [c.descriptor for c in self.concepts]})

        X_embedded = TSNE(n_components=3).fit_transform(self.dataset.tensors[0])

        viz.scatter(X_embedded, Y=self.dataset.tensors[1] + 1,
                    opts={'title': 't-sne',
                          'webgl': True,
                          'markersize': 3,
                          'legend': [c.descriptor for c in self.concepts]})

    def __str__(self):
        return "{}-way classification: {}".format(len(self.concepts), ' vs '.join([c.descriptor for c in self.concepts]))
