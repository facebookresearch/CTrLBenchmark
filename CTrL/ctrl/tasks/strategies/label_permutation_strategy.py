# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ctrl.tasks.strategies.task_creation_strategy import TaskCreationStrategy


class LabelPermutationStrategy(TaskCreationStrategy):
    def __init__(self, *args, **kwargs):
        super(LabelPermutationStrategy, self).__init__(*args, **kwargs)

    def new_task(self, concepts, transformations, previous_tasks, n_samples_per_class):
        if not previous_tasks:
            return self._create_new_task(concepts, transformations)

        prev_task = previous_tasks[-1]
        new_concepts = prev_task.src_concepts.copy()
        self.rnd.shuffle(new_concepts)
        return new_concepts, prev_task.attributes, prev_task.transformation
