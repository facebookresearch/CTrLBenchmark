# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import logging
import random

logger = logging.getLogger(__name__)


class TaskCreationStrategy(abc.ABC):
    def __init__(self, domain, seed,
                 concepts_preferred_lca_dist, concepts_max_lca_dist):
        self.rnd = random.Random(seed)
        self.domain = domain
        self.concepts_preferred_lca_dist = concepts_preferred_lca_dist
        self.concepts_max_lca_dist = concepts_max_lca_dist

    @abc.abstractmethod
    def new_task(self, task_spec, concepts, transformations, previous_tasks):
        raise NotImplementedError

    def descr(self):
        return type(self).__name__
