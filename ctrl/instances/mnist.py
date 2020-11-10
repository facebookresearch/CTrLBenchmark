# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import torch
from ctrl.concepts.concept import Concept, ComposedConcept
from ctrl.concepts.concept_tree import ConceptTree
from torchvision.datasets import MNIST

logger = logging.getLogger(__name__)


def format_data(data):
    if data.dim() == 3:
        # Add the channel dimension if it doesn't exists
        data = data.unsqueeze(1)
    if isinstance(data, torch.ByteTensor):
        data = data.float().div(255)
    return data


class DigitConcept(Concept):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_data, test_data = data
        logger.info('{}: train:\t {}   -   test: {}'.format(self.descriptor, train_data.size(0), test_data.size(0)))
        self.data = [format_data(split) for split in data]
        self.indices = None

        self.norm_mean = torch.as_tensor((0.1307,))
        self.norm_std = torch.as_tensor((0.3081,))

    def init_samples(self, n_per_split):
        assert len(n_per_split) == len(self.data)
        self._samples = []
        for split, n in zip(self.data, n_per_split):
            if n >= 0:
                selected_indices = torch.randperm(split.size(0))[:n]
                split = split[selected_indices]
            self._samples.append(split)


class MnistTree(ConceptTree):
    def __init__(self, n_samples_per_concept, *args, **kwargs):
        super().__init__(n_levels=1, n_children=10, n_samples_per_concept=n_samples_per_concept, name='MNIST',
                         *args, **kwargs)

    def build_tree(self):
        train = MNIST(download=True, root=os.environ['HOME']+'/data', train=True)
        test = MNIST(download=True, root=os.environ['HOME']+'/data', train=False)
        assert train.class_to_idx == test.class_to_idx

        for digit, idx in train.class_to_idx.items():
            train_mask = train.targets == idx
            train_samples = train.data[train_mask]
            test_mask = test.targets == idx
            test_samples = test.data[test_mask]

            concept = DigitConcept(id=digit, data=(train_samples, test_samples))
            self.tree.add_node(concept)
            self.all_nodes.add(concept)
            self.leaf_nodes.add(concept)

        root_concept = ComposedConcept(self.leaf_concepts, id='all_digits')
        for concept in self.leaf_concepts:
            self.tree.add_edge(root_concept, concept)
        return root_concept

    def plot_concepts(self, viz):
        pass
