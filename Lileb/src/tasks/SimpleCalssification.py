# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from src.tasks import Task


class SimpleClassification(Task):
    def __init__(self, in_dim, n_classes, descr=None):
        super(SimpleClassification, self).__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.descr = descr

        if self.descr is None:
            self.descr = [(np.random.uniform(-10, 10, self.in_dim),
                      np.eye(self.in_dim) * np.random.uniform(-5, 5)) for _ in range(self.n_classes)]

    def mutate(*args, **kwargs):
        pass

    def _add_class(self):
        self.descr.append((np.random.uniform(-10, 10, self.in_dim),
                      np.eye(self.in_dim) * np.random.uniform(-5, 5)))

    def get_code(self):
        return self.descr

    def __repr__(self):
        return str(self.get_code())
