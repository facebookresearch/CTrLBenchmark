# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

from src.modules.ll_model import LifelongLearningModel
from src.modules.multitask_head_llmodel import MTHeadModel
from src.modules.utils import MultiHead, make_model


class FineTuneHeadLLModel(LifelongLearningModel):
    def __init__(self,  *args, **kwargs):
        super(FineTuneHeadLLModel, self).__init__(*args, **kwargs)
        self.source_ll_model = None
        self.finished_tasks = 0
        self.all_dims = None

    def get_model(self, task_id, **task_infos):
        task_model = super().get_model(task_id, **task_infos)

        if task_id >= self.finished_tasks and task_id > 0:
            source_model = self.source_ll_model.get_model(task_id - 1)
            task_model.load_source(source_model)

        return task_model

    def _new_model(self, x_dim, n_classes, task_id, **kwargs):
        if self.all_dims is None:
            self.all_dims = [x_dim, *self.hidden_size]
        if self.source_ll_model is None:
            raise ValueError('Source model should be set before creating the '
                             'first model for FineTuneHeadLLModel !')

        new_base = self.base_model_func(self.all_dims)
        new_head = MultiHead(self.all_dims[-1], n_classes)
        new_model = make_model(new_base, new_head, seq=FineTuneHeadMultiTask)

        return new_model

    def set_source_model(self, source):
        assert self.source_ll_model is None, \
            'Source already set for FineTuneHead.'
        self.source_ll_model = source

    def finish_task(self, dataset):
        self.finished_tasks += 1


class FineTuneHeadMultiTask(nn.Module):
    def __init__(self, base, head):
        super(FineTuneHeadMultiTask, self).__init__()
        self.base = base
        self.head = head

    def forward(self, input):
        return self.head(self.base(input))

    def load_source(self, source):
        assert type(source) == MTHeadModel
        self.base.load_state_dict(source.base.state_dict())
        self.head.load_state_dict(source.heads[-1].state_dict())
