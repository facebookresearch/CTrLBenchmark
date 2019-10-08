# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from numbers import Number

import numpy as np
import torch
import visdom
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset

from src.concepts.concept import UniformConceptPool, HierarchicalConceptPool
from src.concepts.concept_tree import ConceptTree
from src.tasks.Category import Category
from src.tasks.task import Task

logger = logging.getLogger(__name__)

class TaskGenerator(object):
    def __init__(self, concept_pool, transformation_pool, p_mutate=.8, p_last=.5):
        self.task_pool = []

        self.p_mutate = p_mutate
        self.p_last = p_last

        # self.available_concepts = UniformConceptPool(intrinsic_dim, 50)
        # self.available_concepts = ClusteredConceptPool(intrinsic_dim, 2, 10)
        # self.available_concepts = HierarchicalConceptPool(2, 7, 3)
        self.available_concepts = concept_pool
        self.available_transformation = transformation_pool

    def add_task(self, ambiant_dim, out_dim=None, n_samples_per_class=None):
        """
        Adds a new task to the current pool.
        This task will be a mutation of an existing task with probability `p_mutate` or a totally task
        with probability 1-`p_mutate`.
        :param in_dim:
        :param out_dim:
        :param n_samples_per_class:
        :return:
        """
        if self.task_pool and np.random.uniform() < self.p_mutate:
            task = self.choose_task()
            logger.info('Mutating existing task: {}'.format(task))
            new_task = self.mutate_task(task)
        else:
            logger.info('Creating new task from scratch')
            new_task = self.create_task(ambiant_dim, out_dim, n_samples_per_class)
        logger.info(new_task)
        self.task_pool.append(new_task)

    def choose_task(self):
        assert self.task_pool, "Can't choose a task from empty pool"
        if np.random.uniform() < self.p_last:
            return self.task_pool[-1]
        else:
            return np.random.choice(self.task_pool)

    def create_task(self, n_dims, n_concepts, n_samples_per_class):
        if isinstance(n_samples_per_class, Number):
            n_samples_per_class = [n_samples_per_class] * n_concepts
        assert len(n_samples_per_class) == n_concepts

        concepts = self.available_concepts.get_concepts(n_concepts)
        concepts = sorted(concepts, key=lambda c: c.descriptor)

        transformations = self.available_transformation.get_tranformation()

        task = Task(concepts, transformations, self.available_transformation, n_samples_per_class)

        return task

    def mutate_task(self, task):
        p_add = 1

        # Make a copy since we will modify this list
        concepts = list(task.concepts)

        mutated = False
        if np.random.uniform() < p_add:
            new_concept = self.available_concepts.get_compatible_concept(task.concepts)
            if new_concept is not None:
                concepts.append(new_concept)
                mutated = True

        if not mutated and len(concepts) > 2:
            choice = np.random.choice(concepts)
            concepts.remove(choice)
            mutated = True

        if not mutated:
            raise ValueError("Impossible to mutate this Task")

        concepts = sorted(concepts, key=lambda c: c.descriptor)
        transformations = task.transformations
        n_samples_per_class = task.n_samples_per_class

        return Task(concepts, transformations, self.available_transformation, n_samples_per_class)

    def __str__(self):
        descr = "Task stream containing {} tasks:\n\t".format(len(self.task_pool))
        tasks = '\n\t'.join(map(str, self.task_pool))
        return descr + tasks






