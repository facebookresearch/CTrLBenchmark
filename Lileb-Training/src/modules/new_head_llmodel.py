# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import make_model, MultiHead


class NewHeadLLModel(LifelongLearningModel):
    def __init__(self, *args, **kwargs):
        super(NewHeadLLModel, self).__init__(*args, **kwargs)
        self.common_model = None

    def _new_model(self, x_dim, n_classes, **kwargs):
        if self.common_model is None:
            self.all_dims = [x_dim, *self.hidden_size]
            self.common_model = self.base_model_func(self.all_dims)

        assert x_dim == self.all_dims[0], 'SingleHead model can only be used ' \
                                          'with the same input dimensions. ' \
                                          'Was initialized with {} but got {}' \
                                          ' on the last task'.format(
                                          self.all_dims[0], x_dim)

        new_head = MultiHead(self.all_dims[-1], n_classes)
        new_model = make_model(self.common_model, new_head)
        return new_model

    def finish_task(self, dataset):
        pass
