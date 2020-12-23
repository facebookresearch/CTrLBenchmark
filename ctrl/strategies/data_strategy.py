# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from ctrl.strategies.task_creation_strategy import TaskCreationStrategy


class DataStrategy(TaskCreationStrategy):
    def __init__(self, n_samples_per_class_options, random,
                 with_replacement, max_samples, min_samples, decay_rate, steps,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples_per_class_options = n_samples_per_class_options
        self.random = random
        self.with_replacement = with_replacement

        if self.random and not self.with_replacement:
            self.rnd.shuffle(self.n_samples_per_class_options)

        self.max_samples = max_samples
        self.min_samples = min_samples
        self.decay_rate = decay_rate
        self.steps = steps

        self.idx = 0

    def new_task(self, task_spec, concepts, transformations, previous_tasks):
        if self.steps is not None:
            n_samples = self._get_n_samples_schedule(len(previous_tasks))
        elif self.max_samples is not None:
            n_samples = self._decay_n_samples(len(previous_tasks))
        else:
            n_samples = self._get_n_samples_classic()

        self.idx += 1

        if isinstance(n_samples, int):
            # If a single number is provided, it corresponds to the trains set
            # size. We need to add the default sizes for remaining sets.
            n_samples = [n_samples]
        if len(n_samples) != len(task_spec.n_samples_per_class):
            n_samples = [*n_samples,
                         *task_spec.n_samples_per_class[len(n_samples):]]


        task_spec.n_samples_per_class = n_samples
        return task_spec

    def _get_n_samples_classic(self):
        if self.with_replacement and self.random:
            n_samples = self.rnd.choice(self.n_samples_per_class_options)
        elif self.with_replacement:
            # We use replacement but without random selection: we cycle through
            # the list of options
            idx = self.idx % len(self.n_samples_per_class_options)
            n_samples = self.n_samples_per_class_options[idx]
        else:
            assert self.n_samples_per_class_options, 'Not enough data options'
            n_samples = self.n_samples_per_class_options.pop(0)
        return n_samples

    def _decay_n_samples(self, t):
        n_samples = self.max_samples * np.exp(-self.decay_rate * t)
        res = [int(round(n_samples)), int(round(n_samples/2))]
        print(f'Using {res} samples')
        return res

    def _get_n_samples_schedule(self, t):
        cur_idx = 0
        # next_step = self.n_samples_per_class_options[0]
        while cur_idx < len(self.steps) and t >= self.steps[cur_idx]:
            cur_idx += 1
        print(f"CHOOSING FROM {self.n_samples_per_class_options[cur_idx]}")
        return self.rnd.choice(self.n_samples_per_class_options[cur_idx])