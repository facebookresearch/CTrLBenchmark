# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from ctrl.concepts.concept import ComposedConcept, Concept
from ctrl.concepts.concept_tree import ConceptTree
from ctrl.instances.DTD import DTD
from ctrl.instances.taxonomies import TAXONOMY
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, \
    FashionMNIST

logger = logging.getLogger(__name__)


IMAGE_DS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'mnist': MNIST,
    'fashion-mnist': FashionMNIST,
    'svhn': SVHN,
    'dtd': DTD,
}


class ImageConcept(Concept):
    def __init__(self, samples, split_rnd, val_size=None, val_ratio=None,
                 *args, **kwargs):
        super(ImageConcept, self).__init__(*args, **kwargs)
        if val_size is not None:
            assert val_ratio is None
        else:
            assert val_ratio is not None
        self.is_static = True
        self._samples = samples
        self.val_size = val_size
        self.val_ratio = val_ratio
        self.split_rnd = split_rnd

        self.attrs = None

    def init_samples(self, n_per_split):
        n_splits = len(n_per_split)
        self.attrs = [torch.Tensor().long() for _ in range(n_splits)]
        if n_splits == len(self._samples):
            pass
        elif n_splits == 3 and len(self._samples) == 2:
            # We need to split the train_set
            train_samples = self._samples[0]
            train_size = train_samples.shape[0]
            # assert train_size in [500, 5000]
            idx = self.split_rnd.permutation(train_size)

            if self.val_size is None:
                val_size = int(np.floor(self.val_ratio * train_size))
            else:
                val_size = self.val_size

            self._samples = (train_samples[idx[:-val_size]],
                             train_samples[idx[-val_size:]],
                             self._samples[1])

        else:
            raise NotImplementedError

    def _get_samples(self, n, attributes, split_id, rng):
        assert not attributes, 'Can\'t use attributes with Images for now'
        n_samples = self._samples[split_id].size(0)
        if n_samples > n:
            idx = rng.choice(n_samples, n, replace=False)
            samples = self._samples[split_id][idx]
        else:
            samples = self._samples[split_id]
        if attributes:
            selected_attributes = self.attrs[split_id][idx][:, attributes]
        else:
            selected_attributes = None

        return samples, selected_attributes

    def get_atomic_concepts(self):
        return [self]

    def get_attr(self, x, y):
        assert self._samples[0].size(1) == 1, 'Don\'t know what an attribute' \
                                             ' is for 3D images'
        tresh = self._samples[0].max().float() / 2

        new_attr = []
        for i, samples_split in enumerate(self._samples):
            split_attr = samples_split[:, 0, y, x].float() > tresh
            new_attr.append(split_attr.long().unsqueeze(1))
        return new_attr


class ImageDatasetTree(ConceptTree):
    def __init__(self, data_path, split_seed, img_size, val_size=None,
                 val_ratio=None, expand_channels=False, *args, **kwargs):
        if not data_path:
            data_path = Path.home()/'.ctrl_data'
            logger.warning(f'No data path specified, default to {data_path}. '
                           f'Can be set in default_datasets.yaml:img_ds_tree')
        self.data_path = data_path
        self.img_size = img_size

        # We already have rnd available from the Tree parent class, but we
        # need a new random state that won't vary across runs to create the
        # val split
        self.split_rnd = np.random.RandomState(split_seed)

        self.val_size = val_size
        self.val_ratio = val_ratio

        self.expand_channels = expand_channels

        super().__init__(*args, **kwargs)

    def build_tree(self):
        if self.name not in IMAGE_DS:
            raise ValueError('Unknown image dataset: {}'.format(self.name))

        ds_class = IMAGE_DS[self.name]
        common_params = dict(root=self.data_path, download=True)
        if self.name in ['svhn', 'dtd', 'aircraft']:
            self.trainset = ds_class(split='train', **common_params)
            self.testset = ds_class(split='test', **common_params)
        else:
            self.trainset = ds_class(train=True, **common_params)
            self.testset = ds_class(train=False, **common_params)
        self.train_samples, self.train_targets = self._format_ds(self.trainset)
        self.test_samples, self.test_targets = self._format_ds(self.testset)
        self.height, self.width = self.train_samples.size()[-2:]

        taxonomy = TAXONOMY[self.name]

        concepts = self._build_tree(taxonomy)
        root_concept = ComposedConcept(concepts, id=self.name)
        self.tree.add_node(root_concept)
        for sub_concept in concepts:
            self.tree.add_edge(root_concept, sub_concept)

        del self.trainset, self.testset, \
            self.train_samples, self.test_samples, \
            self.train_targets, self.test_targets

        return root_concept

    def _build_tree(self, current_level_concepts):
        new_concepts = []
        if isinstance(current_level_concepts, dict):
            # Not yet at the lowest level
            for name, lower_concepts in current_level_concepts.items():
                concepts = self._build_tree(lower_concepts)

                concept_name = '{} {}'.format(self.name, name)
                new_concept = ComposedConcept(concepts=concepts,
                                              id=concept_name)
                self.tree.add_node(new_concept)
                self.all_nodes.add(new_concept)
                for c in concepts:
                    self.tree.add_edge(new_concept, c)

                new_concepts.append(new_concept)

        elif isinstance(current_level_concepts, list):
            # Adding lowest level concepts
            for c in current_level_concepts:
                samples = self._get_samples(c)
                concept_name = '{} {}'.format(self.name, c)
                concept = ImageConcept(id=concept_name, samples=samples,
                                       split_rnd=self.split_rnd,
                                       val_size=self.val_size,
                                       val_ratio=self.val_ratio)
                self.tree.add_node(concept)
                self.all_nodes.add(concept)
                self.leaf_nodes.add(concept)
                new_concepts.append(concept)

        else:
            raise NotImplementedError()

        return new_concepts

    def _format_ds(self, dataset):
        """
        Unify the format of the datasets so that they all have the same type,
        shape and size.
        """
        if hasattr(dataset, 'targets'):
            # Cifar, Mnist
            targets = dataset.targets
        else:
            # SVHN
            targets = dataset.labels

        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)

        samples = dataset.data
        if isinstance(samples, np.ndarray):
            if samples.shape[-1] in [1, 3]:
                # CIFAR Channels are at the end
                samples = samples.transpose((0, 3, 1, 2))
            assert samples.shape[1] in [1, 3]
            samples = torch.from_numpy(samples)

        if samples.ndimension() == 3:
            # Add the channel dim
            samples = samples.unsqueeze(1)
        if self.expand_channels and samples.size(1) == 1:
            samples = samples.expand(-1, 3, -1, -1)

        # samples = samples.float()
        cur_size = list(samples.shape[-2:])
        if cur_size != self.img_size:
            logger.warning('Resizing {} ({}->{})'.format(self.name,
                                                         cur_size,
                                                         self.img_size))
            samples = F.interpolate(samples.float(), self.img_size,
                                    mode='bilinear',
                                    align_corners=False).byte()

        return samples, targets

    def _get_samples(self, concept):
        concept_idx = self._get_class_to_idx()[concept]

        train_concept_mask = self.train_targets == concept_idx
        test_concept_mask = self.test_targets == concept_idx

        train_samples = self.train_samples[train_concept_mask]
        test_samples = self.test_samples[test_concept_mask]

        assert train_samples.size(1) in [1, 3] and \
               test_samples.size(1) in [1, 3]

        return train_samples, test_samples

    def _get_class_to_idx(self):
        if hasattr(self.trainset, 'class_to_idx'):
            assert self.trainset.class_to_idx == self.testset.class_to_idx
            return self.trainset.class_to_idx
        else:
            taxo = TAXONOMY[self.name]
            assert isinstance(taxo, list)
            return {_class: i for i, _class in enumerate(taxo)}

    def plot_concepts(self, viz):
        for c in self.leaf_concepts:
            images = [self.rnd.choice(s) for s in c._samples]
            grid = torchvision.utils.make_grid(images)
            viz.image(grid, opts={'title': c.descriptor,
                                  'width': grid.size(2) * 3,
                                  'height': grid.size(1) * 3.2})

    def draw_attrs(self, viz):
        pass
