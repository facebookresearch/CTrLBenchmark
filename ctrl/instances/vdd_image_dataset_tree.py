# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from os import path

import numpy as np
import torch
import torchvision
from ctrl.commons.utils import format_path
from ctrl.concepts.concept import ComposedConcept
from ctrl.concepts.concept_tree import ConceptTree
from ctrl.instances.image_dataset_tree import ImageConcept
from ctrl.instances.taxonomies import TAXONOMY
from pycocotools.coco import COCO


class VDDImageDatasetTree(ConceptTree):
    def __init__(self, data_path, split_seed, val_size=None,
                 val_ratio=None, *args, **kwargs):
        self.data_path = format_path(data_path)

        # We already have rnd available from the Tree parent class, but we
        # need a new random state that won't vary across runs to create the
        # val split
        self.split_rnd = np.random.RandomState(split_seed)

        self.val_size = val_size
        self.val_ratio = val_ratio

        super().__init__(n_levels=None, n_children=None, *args, **kwargs)

    def build_tree(self):
        # if self.name == 'cifar10':
        #     ds_class = CIFAR10
        # elif self.name == 'cifar100':
        #     ds_class = CIFAR100
        # elif self.name == 'mnist':
        #     ds_class = MNIST
        # else:
        #     raise ValueError('Unknown image dataset: {}'.format(self.name))
        #
        # self.trainset = ds_class(root=self.data_path, train=True, download=True)
        # self.testset = ds_class(root=self.data_path, train=False, download=True)
        # self.train_samples, self.train_targets = self._format_ds(self.trainset)
        # self.test_samples, self.test_targets = self._format_ds(self.testset)
        # self.height, self.width = self.train_samples.size()[2:]

        taxonomy = TAXONOMY[self.name]

        tasks = self._build_tree(taxonomy)
        vdd_task = ComposedConcept(tasks, id=self.name)
        self.tree.add_node(vdd_task)
        for task in tasks:
            self.tree.add_edge(vdd_task, task)
        # print(trainset)dd
        # del self.trainset, self.testset, \
        #     self.train_samples, self.test_samples, \
        #     self.train_targets, self.test_targets

        return vdd_task

    def _build_tree(self, current_level_concepts):
        all_task_concepts = []
        for task, classes in current_level_concepts.items():
            task_concepts = self.add_task(task)
            new_concept = ComposedConcept(concepts=task_concepts, id=task)

            self.tree.add_node(new_concept)
            self.all_nodes.add(new_concept)
            for c in task_concepts:
                self.tree.add_edge(new_concept, c)

            all_task_concepts.append(new_concept)

        return all_task_concepts

    def add_task(self, task_name):
        new_concepts = []
        train_index_path = path.join(self.data_path, 'annotations',
                                     '{}_train.json'.format(task_name))
        # We are using the val split for testing since the test ground
        # truths aren't available for VDD
        test_index_path = path.join(self.data_path, 'annotations',
                                    '{}_val.json'.format(task_name))

        coco_train = COCO(train_index_path)
        coco_test = COCO(test_index_path)
        assert coco_train.cats == coco_test.cats
        for cat, cat_props in coco_train.cats.items():
            train_img_ids = coco_train.catToImgs[cat]
            train_imgs = coco_train.loadImgs(train_img_ids)
            train_img_paths = map(lambda img: path.join(self.data_path,
                                                        img['file_name']),
                                  train_imgs)
            test_img_ids = coco_test.catToImgs[cat]
            test_imgs = coco_test.loadImgs(test_img_ids)
            test_img_paths = map(lambda img: path.join(self.data_path,
                                                        img['file_name']),
                                 test_imgs)
            samples = (np.array(list(train_img_paths)),
                       np.array(list(test_img_paths)))
            concept = ImageConcept(id='{}.{}'.format(task_name,
                                                     cat_props['name']),
                                   samples=samples,
                                   split_rnd=self.split_rnd,
                                   val_size=self.val_size,
                                   val_ratio=self.val_ratio)
            self.tree.add_node(concept)
            self.all_nodes.add(concept)
            self.leaf_nodes.add(concept)
            new_concepts.append(concept)
        return new_concepts
    #
    #
    # def _build_tree(self, current_level_concepts):
    #
    #     new_concepts = []
    #     if isinstance(current_level_concepts, dict):
    #         # Not yet at the lowest level
    #         for name, lower_concepts in current_level_concepts.items():
    #             concepts = self._build_tree(lower_concepts)
    #
    #             new_concept = ComposedConcept(concepts=concepts, id=name)
    #             self.tree.add_node(new_concept)
    #             self.all_nodes.add(new_concept)
    #             for c in concepts:
    #                 self.tree.add_edge(new_concept, c)
    #
    #             new_concepts.append(new_concept)
    #
    #     elif isinstance(current_level_concepts, list):
    #         # Adding lowest level concepts
    #
    #         for c in current_level_concepts:
    #             samples = self._get_samples(c)
    #             concept = ImageConcept(id=c, samples=samples,
    #                                    split_rnd=self.split_rnd,
    #                                    val_size=self.val_size,
    #                                    val_ratio=self.val_ratio)
    #             self.tree.add_node(concept)
    #             self.all_nodes.add(concept)
    #             self.leaf_nodes.add(concept)
    #             new_concepts.append(concept)
    #
    #     else:
    #         raise NotImplementedError()
    #
    #     return new_concepts

    def _get_samples(self, concept):
        concept_idx = self.trainset.class_to_idx[concept]
        assert concept_idx == self.testset.class_to_idx[concept]

        train_concept_mask = self.train_targets == concept_idx
        test_concept_mask = self.test_targets == concept_idx

        train_samples = self.train_samples[train_concept_mask]
        test_samples = self.test_samples[test_concept_mask]

        assert train_samples.size(1) in [1, 3] and \
               test_samples.size(1) in [1, 3]

        return train_samples, test_samples

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
