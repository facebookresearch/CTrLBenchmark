# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import make_model, MultiHead


class IndependentLLModel(LifelongLearningModel):
    def _new_model(self, x_dim, n_classes, **kwargs):
        all_dims = [x_dim, *self.hidden_size]
        new_base = self.base_model_func(all_dims)

        new_head = MultiHead(all_dims[-1], n_classes)
        new_model = make_model(new_base, new_head)
        return new_model

    def finish_task(self, dataset):
        pass
