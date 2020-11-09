# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from ctrl.commons.tree import Tree
from ctrl.concepts.concept import ComposedConcept, AtomicConcept
from ctrl.concepts.concept_tree import ConceptTree
from ctrl.tasks.task import Task
from torchvision import transforms

logger = logging.getLogger(__name__)


def loss(y_hat, y, reduction: str = 'none'):
    """

    :param y_hat: Model predictions
    :param y: Ground Truth
    :param reduction:
    :return:
    """
    assert y.size(1) == 1 and torch.is_tensor(y_hat)
    y = y.squeeze(1)
    loss_val = F.cross_entropy(y_hat, y, reduction=reduction)
    assert loss_val.dim() == 1
    return loss_val


def augment_samples(samples):
    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()
        ])
    aug_samples = []
    for sample in samples:
        for i in range(4):
            aug_samples.append(trans(sample))
    for sample in samples:
        aug_samples.append(transforms.ToTensor()(transforms.ToPILImage()(sample)))

    return torch.stack(aug_samples)


def _generate_samples_from_descr(categories, attributes, n_samples_per_class,
                                 augment):
    use_cat_id, attributes = attributes
    assert use_cat_id or attributes, 'Each task should at least use the ' \
                                     'category id or an attribute as labels'
    if not use_cat_id:
        all_concepts = np.array(
            [concept for cat in categories for concept in cat])
        all_attrs = np.array([c.attrs for c in all_concepts])
        selected_attr = all_attrs[:, attributes[0]]

        categories = [tuple(all_concepts[selected_attr == val])
                    for val in np.unique(selected_attr)]

    if use_cat_id or isinstance(all_concepts[0], AtomicConcept):
        samples = []
        labels = []
        for i, cat_concepts in enumerate(categories):
            mixture = ComposedConcept(cat_concepts, id=None)
            cat_samples = []
            cat_labels = []
            for s_id, n in enumerate(n_samples_per_class):
                split_samples, split_attrs = mixture._get_samples(n, attributes,
                                                                  split_id=s_id)
                if s_id in augment:
                     split_samples = augment_samples(split_samples)
                split_labels = torch.Tensor().long()
                if use_cat_id:
                    cat_id = torch.tensor([i]).expand(split_samples.shape[0], 1)
                    split_labels = torch.cat([split_labels, cat_id], dim=1)

                if attributes:
                    raise NotImplementedError('Attrs aren\'t supported '
                                              'anymore')
                    split_labels = torch.cat([split_labels, split_attrs], dim=1)
                cat_samples.append(split_samples)
                cat_labels.append(split_labels)
            samples.append(cat_samples)
            labels.append(cat_labels)
        if torch.is_tensor(samples[0][0]):
            cat_func = torch.cat
        else:
            cat_func = np.concatenate
        samples = (cat_func(split) for split in zip(*samples))
        labels = (torch.cat(split) for split in zip(*labels))
    else:
        # Grouping the concepts by attribute value to create the categories
        all_concepts = np.array(
            [concept for cat in categories for concept in cat])

        samples, labels = get_samples_using_attrs(all_concepts, attributes,
                                                  n_samples_per_class)

    return samples, labels


def get_samples_using_attrs(concepts, attrs, n_samples_per_class):
    samples, labels = [], []
    if isinstance(concepts[0], AtomicConcept):
        all_attrs = np.array([c.attrs for c in concepts])
        selected_attr = all_attrs[:, attrs[0]]

        concepts = [tuple(concepts[selected_attr == val])
                    for val in np.unique(selected_attr)]
    else:
        if all(c.is_static for c in concepts):
            all_samples = []
            all_attrs = []
            for i in range(len(n_samples_per_class)):
                split_samples = []
                split_attrs = []
                for c in concepts:
                    c_samples, c_attributes = c._get_samples(None, attrs, i)
                    split_samples.append(c_samples)
                    split_attrs.append(c_attributes)
                all_samples.append(split_samples)
                all_attrs.append(split_attrs)

            all_samples = [torch.cat(split) for split in all_samples]
            all_attrs = [torch.cat(split) for split in all_attrs]

            for n, s_samples, s_attrs in zip(n_samples_per_class, all_samples, all_attrs):
                samples.append([])
                labels.append([])
                attr_values = s_attrs.unique().tolist()
                assert len(attr_values) == 2
                for attr_val in attr_values:
                    mask = s_attrs[:, 0] == attr_val
                    candiates_samples = s_samples[mask]
                    candiates_attrs = s_attrs[mask]
                    # assert candiates_samples.size(0) >= n
                    selected_idx = torch.randperm(candiates_samples.size(0))[:n]
                    samples[-1].append(candiates_samples[selected_idx])
                    labels[-1].append(candiates_attrs[selected_idx])

    samples = [torch.cat(split) for split in samples]
    labels = [torch.cat(split) for split in labels]

    return samples, labels


def _get_samples_from_descr(main_classes, attributes,
                            n_samples_per_class, rnd):
    """
    todo: handle attributes for composed concepts and for concepts of which
    instances can have different values for the same attribute


    :param main_classes: Iterable of Iterables of Concepts:
           [[classA_concept1, classA_concept2,], [classB_concept1, ...],...]
    :param attributes:
    :param transformation:
    :return:
    """
    use_cat_id, attributes = attributes
    assert use_cat_id or attributes, 'Each task should at least use the ' \
                                     'category id or an attribute as labels'
    samples = []
    labels = []
    for i, cat_concepts in enumerate(main_classes):
        # First, gather all the potential samples for the current category
        class_samples = []
        class_labels = []
        for concept in cat_concepts:
            concept_samples = concept.get_samples()
            class_samples.append(concept_samples)

            concepts_labels = []
            concept_attributes = concept.get_attributes(attributes)
            for j, split_samples in enumerate(concept_samples):
                split_labels = torch.Tensor().long()
                if use_cat_id:
                    cat_id = torch.tensor([i]).expand(split_samples.size(0),
                                                      1)
                    split_labels = torch.cat([split_labels, cat_id], dim=1)
                if concept_attributes:
                    split_attrs = concept_attributes[j]
                    split_labels = torch.cat([split_labels, split_attrs], dim=1)
                concepts_labels.append(split_labels)
            class_labels.append(concepts_labels)

        class_samples = [torch.cat(split) for split in zip(*class_samples)]
        class_labels = [torch.cat(split) for split in zip(*class_labels)]

        # Then select the correct number of samples for the current class
        samples_selection = []
        labels_selection = []
        for x, y, n in zip(class_samples, class_labels,
                           n_samples_per_class):
            if n > 0:
                selected_idx = rnd.sample(range(x.size(0)), n)
                x = x[selected_idx]
                y = y[selected_idx]
            samples_selection.append(x)
            labels_selection.append(y)

        samples.append(samples_selection)
        labels.append(labels_selection)

    # Concatenate all samples for each split
    samples = (torch.cat(x_split) for x_split in zip(*samples))
    labels = (torch.cat(y_split) for y_split in zip(*labels))

    return samples, labels


class TaskGenerator(object):
    def __init__(self, concepts: ConceptTree, transformation_pool,
                 samples_per_class, split_names, strat, reuse_samples,
                 seed: int, flatten, n_initial_classes, use_cat_id, tta,
                 *args, **kwargs):
        """

        :param concepts: Concept pool from which we will sample when creating
            new tasks.
        :param transformation_pool: Transformation pool from which we will
            select the operations to be applied on  the data of new tasks.
        :param samples_per_class: Initial number of samples per class
        :param split_names: Name of the different data splits usually
            (train, val, test)
        :param strat: Strategy to use for the creation of new tasks
        :param reuse_samples: wether to allow the usage of the same data
            samples in different tasks.
        :param seed: The seed used for the samples selection
        :param flatten:
        :param n_initial_classes:
        :param use_cat_id: Legacy prop used with attributes.
        :param tta: use Test Time Augmentation
        """
        super(TaskGenerator, self).__init__(*args, **kwargs)
        self.task_pool = []

        self.reuse_samples = reuse_samples
        self.concept_pool = concepts
        self.transformation_pool = transformation_pool
        assert len(samples_per_class) == len(split_names)
        self.n_samples_per_class = samples_per_class
        self.split_names = split_names

        self.rnd = random.Random(seed)

        self.flatten = flatten
        self.tta = tta

        # For default task creation
        self.n_initial_classes = n_initial_classes
        self.use_cat_id = use_cat_id

        self.strat = strat
        self.contains_loaded_tasks = False

    @property
    def n_tasks(self):
        return len(self.task_pool)

    def add_task(self, name=None, save_path=None):
        """
        Adds a new task to the current pool.
        This task will be created using the current strategy `slef.strat`
        :param name: The name of the new task
        :param save_path: If provided, the task will be saved under this path
        :return: The new Task
        """
        new_task_id = len(self.task_pool)

        if new_task_id == 0:
            concepts, attrs, trans, n = self._create_new_task(
                self.concept_pool, self.transformation_pool)
        else:
            concepts = self.task_pool[-1].src_concepts
            attrs = self.task_pool[-1].attributes
            trans = self.task_pool[-1].transformation
            n = self.task_pool[-1].n_samples_per_class

        cur_task_spec = SimpleNamespace(src_concepts=concepts,
                                        attributes=attrs,
                                        transformation=trans,
                                        n_samples_per_class=n,
                                        )

        cur_task_spec = self.strat.new_task(cur_task_spec, self.concept_pool,
                                       self.transformation_pool,
                                       self.task_pool)
        assert len(cur_task_spec.n_samples_per_class) == len(self.split_names)

        new_task = self._create_task(cur_task_spec, name, save_path)
        new_task.id = new_task_id
        self.task_pool.append(new_task)
        return new_task

    def load_task(self, task_name, load_path):
        splits = ['train', 'val', 'test']
        samples = []
        save_paths = []
        for split in splits:
            file_path = os.path.join(load_path, '{}_{}.pth'.format(task_name, split))
            save_paths.append(file_path)
            assert os.path.isfile(file_path), file_path
            xs, ys = torch.load(file_path)
            samples.append((xs, ys))
        metadata_file = os.path.join(load_path, '{}.meta'.format(task_name))
        if os.path.isfile(metadata_file):
            meta = torch.load(metadata_file)
        else:
            meta = {}
        task = Task(task_name, samples, loss, split_names=self.split_names,
                    id=len(self.task_pool), **meta)
        task.save_path = save_paths
        self.task_pool.append(task)
        self.contains_loaded_tasks = True
        return task

    def _create_task(self, task_spec, name, save_path):
        concepts = task_spec.src_concepts
        attributes = task_spec.attributes
        transformation = task_spec.transformation
        n_samples_per_class = task_spec.n_samples_per_class

        samples = self.get_samples(concepts, attributes, transformation,
                                   n_samples_per_class)
        if self.flatten:
            samples = [(x.view(x.size(0), -1), y) for x, y in samples]
        task = Task(name, samples, loss, transformation, self.split_names,
                    source_concepts=concepts, attributes=attributes,
                    creator=self.strat.descr(), generator=self,
                    n_samples_per_class=n_samples_per_class,
                    save_path=save_path)
        return task

    def get_similarities(self, component=None):
        """
        :param component: String representing the components across which the
            similarities should be computed, can be any combination of :

            - 'x' for p(x|z)
            - 'y' for p(y|z)
            - 'z' for p(z)
        :return: A dict associating each component to an n_tasks x n_tasks
         tensor containing the similarities between tasks over this component.
        """
        if component is None:
            component = 'xyz'

        similarities = torch.zeros(self.n_tasks, self.n_tasks, len(component))
        times = torch.zeros(len(component))
        for i, t1 in enumerate(self.task_pool):
            for j, t2 in enumerate(self.task_pool[i:]):
                sim, time = self.get_similarity(t1, t2, component)
                sim = torch.tensor(sim)
                # Similarities are symmetric
                similarities[i, i + j] = sim
                similarities[i + j, i] = sim
                times += torch.tensor(time)
        for comp, time in zip(component, times.unbind()):
            if time > 1:
                logger.warning(
                    "Comparison of {} took {:4.2f}s".format(comp, time))

        sim_dict = dict(zip(component, similarities.unbind(-1)))
        return sim_dict

    def get_similarity(self, t1, t2, component=None):
        if component is None:
            component = 'xyz'
        res = []
        times = []
        for char in component:
            start_time = time.time()
            if char == 'x':
                res.append(self.transformation_pool.transformations_sim(
                    t1.transformation, t2.transformation))
            elif char == 'y':
                res.append(self.concept_pool.y_attributes_sim(t1.attributes,
                                                              t2.attributes))
            elif char == 'z':
                res.append(self.concept_pool.categories_sim(t1.src_concepts,
                                                            t2.src_concepts))
            else:
                raise ValueError('Unknown component {}'.format(char))
            times.append(time.time() - start_time)
        return res, times

    def get_samples(self, concepts, attributes, transformation,
                    n_samples_per_class):
        augment = [1] if self.tta else []
        if self.reuse_samples:
            samples, labels = _get_samples_from_descr(concepts,
                                                      attributes,
                                                      n_samples_per_class,
                                                      self.rnd)
        else:
            samples, labels = _generate_samples_from_descr(concepts,
                                                           attributes,
                                                           n_samples_per_class,
                                                           augment)
        # Apply the input transformation
        samples = [transformation(x) for x in samples]

        return [(x, y) for x, y in zip(samples, labels)]

    def stream_infos(self, full=True):
        """
        return a list containing the information of each task in the task_pool,
        useful when the stream needs to be serialized (e.g. to be sent to
        workers.)
        """
        return [t.info(full) for t in self.task_pool]

    def _create_new_task(self, concept_pool, transformation_pool, n_attributes=0):
        logger.info('Creating new task from scratch')
        concepts = concept_pool.get_compatible_concepts(self.n_initial_classes,
                                                        leaf_only=True,)

        n_avail_attrs = len(concept_pool.attributes)
        if n_attributes > n_avail_attrs:
            raise ValueError('Can\'t select {} attributes, only {} available'
                             .format(n_attributes, n_avail_attrs))
        attributes = self.rnd.sample(range(n_avail_attrs), n_attributes)

        transformation = transformation_pool.get_transformation()
        concepts = [(c,) for c in concepts]

        return concepts, (self.use_cat_id, attributes), transformation, \
               self.n_samples_per_class

    def __str__(self):
        descr = "Task stream containing {} tasks:\n\t".format(self.n_tasks)
        tasks = '\n\t'.join(map(str, self.task_pool))
        return descr + tasks


t = TaskGenerator()