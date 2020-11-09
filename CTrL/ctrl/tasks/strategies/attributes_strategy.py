# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ctrl.tasks.strategies.input_domain_strategy import TaskCreationStrategy


class AttributeStrategy(TaskCreationStrategy):
    def __init__(self, n_attrs_per_task, resample_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_attrs_per_task = n_attrs_per_task
        self.resample_classes = resample_classes

    def new_task(self, concept_pool, transformations, previous_tasks, n_samples_per_class):
        if not previous_tasks:
            concepts, (use_cat_id, attrs), transfo = \
                self._create_new_task(concept_pool, transformations, self.n_attrs_per_task)
        else:
            prev_attrs = set(attr for t in previous_tasks for attr in t.attributes[1])
            avail_attrs = set(range(len(concept_pool.attributes))).difference(prev_attrs)
            n_avail_attrs = len(avail_attrs)
            if self.n_attrs_per_task > n_avail_attrs:
                raise ValueError(
                    'Can\'t select {} attributes, only {} available'.format(
                        self.n_attrs_per_task, n_avail_attrs))
            attributes = self.rnd.sample(list(avail_attrs), self.n_attrs_per_task)

            prev_task = previous_tasks[-1]
            if self.resample_classes:
                concepts = concept_pool.get_compatible_concepts(
                    self.n_initial_classes,
                    leaf_only=True,
                    preferred_lca_dist=self.concepts_preferred_lca_dist,
                    max_lca_dist=self.concepts_max_lca_dist)
                concepts = [(c,) for c in concepts]
            else:
                concepts = prev_task.src_concepts
            use_cat_id = self.use_cat_id
            attrs = attributes
            transfo = prev_task.transformation

        if not use_cat_id:
            assert len(attrs) == 1, "Should have at max one attribute " \
                                         "when not using the category ids, " \
                                         "otherwise it's unclear what a class" \
                                         " is (for n samples per class)."

        return concepts, (use_cat_id, attrs), transfo
