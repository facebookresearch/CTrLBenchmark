# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import make_model, MultiHead
from src.train.ray_training import SimpleMultiTaskTrainable, \
    OSMultiTaskTrainable


class MultitaskLegLLModel(LifelongLearningModel):
    def __init__(self, *args, **kwargs):
        super(MultitaskLegLLModel, self).__init__(*args, **kwargs)
        self.legs = []
        self.n_classes = None

        self.common_model = None
        if len(self.hidden_size) == 0:
            raise NotImplementedError("NewLeg and NewHead are equivalent "
                                      "when no hidden layers are used, "
                                      "please use NewHead instead.")

    def _new_model(self, x_dim, n_classes, **kwargs):
        if self.common_model is None:
            self.n_classes = n_classes
            if len(x_dim) == 1:
                n_convs = 0
            else:
                k = 3
                stride = 2
                second_layer_map_size = [(s - (k-1) + 1)//stride for s in x_dim[1:]]
                self.hidden_size[0] = (self.hidden_size[0], *second_layer_map_size)
                n_convs = 1
            self.all_dims = [x_dim, *self.hidden_size]

            common_trunc = self.base_model_func(self.all_dims[1:], n_convs=n_convs)
            head = MultiHead(self.all_dims[-1], n_classes)
            self.common_model = make_model(common_trunc, head)

        assert n_classes == self.n_classes, 'Number of classes should always ' \
                                            'be the same with MTNewLegLLModel'

        new_leg = self.base_model_func(self.all_dims[:2])
        self.legs.append(new_leg)
        return MTLegModel(nn.ModuleList(self.legs),
                          self.common_model,
                          self.all_dims[1])

    def finish_task(self, dataset):
        pass

    def get_trainable(self, use_ray_logging):
        return SimpleMultiTaskTrainable if use_ray_logging else OSMultiTaskTrainable


class MTLegModel(nn.Module):
    def __init__(self, legs, trunc, trunc_input_size):
        super(MTLegModel, self).__init__()
        self.legs = legs
        self.trunc = trunc
        if isinstance(trunc_input_size, int):
            self.trunc_in_size = (trunc_input_size,)
        else:
            self.trunc_in_size = trunc_input_size

    def forward(self, *input):
        batch_size = input[0].size(0)
        if len(input) == 2:
            input, task_ids = input
        else:
            input = input[0]
            task_id = len(self.legs)-1
            task_ids = torch.ones(batch_size, dtype=torch.long) * task_id

        feats = torch.empty(batch_size, *self.trunc_in_size).to(input.device)

        for t_id in task_ids.unique().tolist():
            mask = task_ids == t_id
            feats[mask] = self.legs[t_id](input[mask])
        out = self.trunc(feats)
        return out

    @property
    def n_out(self):
        return self.trunc.n_out if self.trunc else self.legs[0].n_out
