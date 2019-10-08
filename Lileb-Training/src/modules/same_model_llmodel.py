# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import make_model, MultiHead


class SameModelLLModel(LifelongLearningModel):
    def __init__(self, *args, **kwargs):
        super(SameModelLLModel, self).__init__(*args, **kwargs)
        self.model = None
        self.n_classes = None

    def _new_model(self, x_dim, n_classes, **kwargs):
        if self.model is None:
            self.x_dim = x_dim
            self.n_classes = n_classes

            all_dims = [x_dim, *self.hidden_size]
            base_model = self.base_model_func(all_dims)
            new_head = MultiHead(all_dims[-1], n_classes)

            self.model = make_model(base_model, new_head)

        if n_classes != self.n_classes:
            raise ValueError(
                'SingleHead model can only be used with the same number of '
                'classes. Was initialized with {} but got {} on the last '
                'task'.format(self.n_classes, n_classes))

        if x_dim != self.x_dim:
            raise ValueError(
                'SingleHead model can only be used with the same input '
                'dimensions. Was initialized with '
                '{} but got {} on the last task'.format(self.x_dim, x_dim))

        return self.model

    def finish_task(self, dataset):
        pass
