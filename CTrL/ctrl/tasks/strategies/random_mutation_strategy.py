# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from ctrl.concepts.concept import ComposedConcept
from ctrl.tasks.strategies.input_domain_strategy import TaskCreationStrategy

logger = logging.getLogger(__name__)


class MutationException(Exception):
    pass


class RandomMutationStrategy(TaskCreationStrategy):
    def __init__(self, operations, p_mutate, p_last, n_attr_min, n_attr_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_mutate = p_mutate
        self.p_last = p_last
        self.n_attr_min = n_attr_min
        self.n_attr_max = n_attr_max

        self._all_operators = {'add': self.add_category,
                               'remove': self.remove_category,
                               'merge': self.merge_classes,
                               'split': self.split_classes,
                               'input_transform': self.random_transformation}

        if operations == 'all':
            self.operations = list(self._all_operators.keys())
        else:
            assert all(op in self._all_operators for op in operations)
            self.operations = operations

    def new_task(self, concepts, transformations, previous_tasks, n_samples_per_class):
        mutate = self.rnd.random() < self.p_mutate

        if previous_tasks and mutate:
            old_task = self.choose_task(previous_tasks)
            try:
                next_task = self.mutate_task(old_task, concepts, transformations)
            except MutationException:
                mutate = False

        if not mutate or not previous_tasks:
            logger.info('Creating new task from scratch')
            n_attributes = self.rnd.randint(self.n_attr_min, self.n_attr_max)
            next_task = self._create_new_task(concepts, transformations, n_attributes)
        return next_task

    def choose_task(self, task_pool):
        assert task_pool, "Can't choose a task from empty pool"
        if self.rnd.random() < self.p_last:
            return task_pool[-1]
        else:
            return self.rnd.choice(task_pool)

    def mutate_task(self, task, concept_pool, transfo_pool, samples_per_class):
        logger.info('Mutate existing task: "{}"'.format(task))
        new_task = None
        avail_ops = self.operations.copy()
        while avail_ops and new_task is None:
            op = self.rnd.choice(avail_ops)
            avail_ops.remove(op)
            mutation_func = self._all_operators[op]
            try:
                new_task = mutation_func(task, concept_pool, transfo_pool, samples_per_class)
                new_task.creator = op
            except MutationException as e:
                logger.exception(e)

        if new_task is None:
            raise MutationException("Impossible to mutate this Task")

        return new_task

    def add_category(self, task, concept_pool, transformation_pool):
        logger.info('Trying to add_category')
        concepts = list(task.concepts)
        new_concept = concept_pool.get_compatible_concepts(exclude_concepts=concepts)
        if new_concept is None:
            raise MutationException('No compatible concept found.')
        new_category = tuple(new_concept)
        categories = task.src_concepts + [new_category]

        return categories, task.attributes, task.transformation

    def remove_category(self, task, concept_pool, transformation_pool):
        logger.info('Trying to remove_category')
        categories = task.src_concepts
        if len(categories) < 3:
            raise MutationException('Not enough classes to remove one.')
        choice = self.rnd.choice(categories)
        categories.remove(choice)

        return categories, task.attributes, task.transformation

    def merge_classes(self, task, concept_pool, transformation_pool):
        logger.info('Trying to merge_classes')
        categories = task.src_concepts
        if len(categories) < 3:
            raise MutationException('Not enough classes to merge.')

        new_categories = concept_pool.merge_closest_categories(categories)
        if new_categories is None:
            raise MutationException('No couple of categories can be merged.')

        return new_categories, task.attributes, task.transformation

    def split_classes(self, task, concept_pool, transformation_pool):
        logger.info('Trying to split_classes')
        categories = task.src_concepts

        split_candidates = concept_pool.get_widest_categories(categories)

        def is_valid(candidate):
            # A category can only be split if it contains several concepts or a higher level ComposedConcept
            return len(candidate) > 1 or isinstance(candidate[0], ComposedConcept)

        split_candidates = list(filter(is_valid, split_candidates))
        if not split_candidates:
            raise MutationException('No category can be split.')

        cat_to_split = self.rnd.choice(split_candidates)
        logger.info("splitting {}".format(cat_to_split))

        new_categories = concept_pool.split_category(cat_to_split)
        new_categories = [tuple(cat) for cat in new_categories]

        categories.remove(cat_to_split)
        categories.extend(new_categories)

        return categories, task.attributes, task.transformation

    def random_transformation(self, task, concept_pool, transformation_pool):
        logger.info('Trying to random_transformation')
        new_transformation = transformation_pool.get_transformation()

        return task.src_concepts, task.attributes, new_transformation
