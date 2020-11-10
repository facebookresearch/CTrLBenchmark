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
        # if self.name == 'cifar10':
        #     ds_class = CIFAR10
        # elif self.name == 'cifar100':
        #     ds_class = CIFAR100
        # elif self.name == 'mnist':
        #     ds_class = MNIST
        # elif self.name == 'fashion-mnist':
        #     ds_class = FashionMNIST
        # elif self.name == 'svhn':
        #     ds_class = SVHN
        # else:
        #     raise ValueError('Unknown image dataset: {}'.format(self.name))
        # common_params = dict(root=self.data_path, download=True)
        # if self.name == 'svhn':
        #     self.trainset = ds_class(split='train', **common_params)
        #     self.testset = ds_class(split='test', **common_params)
        # else:
        #     self.trainset = ds_class(train=True, **common_params)
        #     self.testset = ds_class(train=False, **common_params)
        # self.train_samples, self.train_targets = self._format_ds(self.trainset)
        # self.test_samples, self.test_targets = self._format_ds(self.testset)
        # self.height, self.width = self.train_samples.size()[2:]
        #
        # taxonomy = TAXONOMY[self.name]
        #
        # concepts = self._build_tree(taxonomy)
        child_concepts = [child.root_node for child in self.children]
        root_concept = ComposedConcept(child_concepts, id=self.name)
        self.tree.add_node(root_concept)
        for child in self.children:
            self.tree = nx.compose(self.tree, child.tree)
            self.tree.add_edge(root_concept, child.root_node)
            self.leaf_nodes = self.leaf_nodes.union(child.leaf_nodes)
            self.all_nodes = self.all_nodes.union(child.all_nodes)
        # print(trainset)dd
        # del self.trainset, self.testset, \
        #     self.train_samples, self.test_samples, \
        #     self.train_targets, self.test_targets

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

    def init_attributes(self, n_attrs):
        for i in range(n_attrs):
            attr_ok = False
            while not attr_ok:
                attr_x = self.rnd.randrange(self.width)
                attr_y = self.rnd.randrange(self.height)
                self.attributes.append((attr_x, attr_y))
                attrs = []
                for concept in self.leaf_concepts:
                    attrs.append(concept.get_attr(attr_x, attr_y))
                splits = [torch.cat(s) for s in zip(*attrs)]
                # values = [s.unique() for s in splits]
                all_n = []
                for s in splits:
                    for val in s.unique():
                        all_n.append((s == val).sum().item())
                # all_n = [(s == val).sum().item() for s in splits for val in s.unique()]
                m = min(all_n)
                if m > 1000 and len(all_n) == 6:
                    attr_ok = True
                else:
                    self.attributes.pop()

            for concept, attr in zip(self.leaf_concepts, attrs):
                for j, s_attr in enumerate(attr):
                    concept.attrs[j] = torch.cat([concept.attrs[j], s_attr], dim=1)

        # for concept in self.leaf_concepts:
        #     all_pos = []
        #     all_neg = []
        #     for s_attr in concept.attrs:
        #         all_pos.append(s_attr.sum())
        #         all_neg.append((s_attr == 0).sum())
        #     print('{}: {} positive attributes, {} neg'.format(concept.descriptor, all_pos, all_neg    ))
        if n_attrs:
            self.attribute_similarities = torch.ones(n_attrs, n_attrs)

    def plot_concepts(self, viz):
        for c in self.leaf_concepts:
            images = [self.rnd.choice(s) for s in c._samples]
            grid = torchvision.utils.make_grid(images)
            viz.image(grid, opts={'title': c.descriptor,
                                  'width': grid.size(2) * 3,
                                  'height': grid.size(1) * 3.2})

    def draw_attrs(self, viz):
        pass
