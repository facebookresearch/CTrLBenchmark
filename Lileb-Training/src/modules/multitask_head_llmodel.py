# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import MultiHead
from src.train.ray_training import SimpleMultiTaskTrainable, OSMultiTaskTrainable


class MultitaskHeadLLModel(LifelongLearningModel):
    def __init__(self, *args, **kwargs):
        super(MultitaskHeadLLModel, self).__init__(*args, **kwargs)
        self.heads = []
        self.n_classes = []

        self.common_model = None

    def _new_model(self, x_dim, n_classes, **kwargs):
        if self.common_model is None:
            self.all_dims = [x_dim, *self.hidden_size]
            self.common_model = self.base_model_func(self.all_dims)

        self.n_classes.append(n_classes)

        new_head = MultiHead(self.all_dims[-1], n_classes)
        self.heads.append(new_head)
        return MTHeadModel(self.common_model,
                           nn.ModuleList(self.heads),
                           self.n_classes.copy())

    def finish_task(self, dataset):
        pass

    def get_trainable(self, use_ray_logging):
        return SimpleMultiTaskTrainable if use_ray_logging else OSMultiTaskTrainable


class MTHeadModel(nn.Module):
    def __init__(self, base, heads, n_classes):
        super(MTHeadModel, self).__init__()
        self.base = base
        assert all(n == n_classes[0] for n in n_classes)
        self.n_classes = n_classes[0]
        # Check that all heads have the same number of outputs
        assert all(head.n_out == heads[0].n_out for head in heads)
        self.heads = heads

    def forward(self, *input):
        batch_size = input[0].size(0)

        if len(input) == 2:
            input, task_ids = input
        else:
            # By default, if no task id is provided we consider that the batch
            # is from the last task.
            input = input[0]
            task_id = len(self.heads)-1
            task_ids = torch.ones(batch_size, dtype=torch.long) * task_id

        feats = self.base(input)
        out = [torch.empty(batch_size, n_out).to(input.device)
               for n_out in self.n_classes]
        batch_tasks = task_ids.unique().tolist()
        # assert len(batch_tasks) == 1
        for t_id in batch_tasks:
            mask = task_ids == t_id
            task_head = self.heads[t_id]
            task_out = task_head(feats[mask])
            for attr_out, head_out in zip(out, task_out):
                attr_out[mask] = head_out
        return out

    @property
    def n_out(self):
        return self.heads[0].n_out



