# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ctrl.tasks.strategies.task_creation_strategy import TaskCreationStrategy


class MixedStrategy(TaskCreationStrategy):
    def __init__(self, strategies, random_select, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategies = strategies
        self.strategies_list = list(strategies.values())
        # self.last_strat = None
        self.random_select = random_select
        assert not self.random_select
        self.idx = 0

    def new_task(self, task_spec, concepts, transformations, previous_tasks):
        for strat in self.strategies_list:
            task_spec = strat.new_task(task_spec, concepts, transformations,
                                       previous_tasks)
        return task_spec

    def descr(self):
        return 'Mixed<{}>'.format(list(self.strategies.keys()))
