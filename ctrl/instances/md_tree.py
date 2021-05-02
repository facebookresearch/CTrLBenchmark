# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import networkx as nx
import numpy as np
import torch
import torchvision

from ctrl.concepts.concept import ComposedConcept
from ctrl.concepts.concept_tree import ConceptTree
from ctrl.instances.taxonomies import TAXONOMY


class MultiDomainDatasetTree(ConceptTree):
    def __init__(self, **kwargs):
        self.children = []
        extracted = []
        for k, v in kwargs.items():
            if 'child' in k:
                extracted.append(k)
                self.children.append(v)
        for k in extracted:
            del kwargs[k]
        n_levels = max(child.n_levels for child in self.children)
        super().__init__(n_levels=n_levels, n_children=None,
                         n_samples_per_concept=None, **kwargs)

    def build_tree(self):
        child_concepts = [child.root_node for child in self.children]
        root_concept = ComposedConcept(child_concepts, id=self.name)
        self.tree.add_node(root_concept)
        for child in self.children:
            self.tree = nx.compose(self.tree, child.tree)
            self.tree.add_edge(root_concept, child.root_node)
            self.leaf_nodes = self.leaf_nodes.union(child.leaf_nodes)
            self.all_nodes = self.all_nodes.union(child.all_nodes)

        return root_concept

    def init_data(self, n_samples_per_concept):
        pass

    def _format_ds(self, dataset):
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
