# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

from src.modules.ll_model import LifelongLearningModel
from src.modules.multitask_leg_llmodel import MTLegModel
from src.modules.utils import MultiHead, make_model


class FineTuneLegLLModel(LifelongLearningModel):
    def __init__(self, *args, **kwargs):
        super(FineTuneLegLLModel, self).__init__(*args, **kwargs)
        self.source_ll_model = None
        self.finished_tasks = 0
        self.all_dims = None
        if len(self.hidden_size) == 0:
            raise NotImplementedError("NewLeg and NewHead are equivalent "
                                      "when no hidden layers are used, "
                                      "please use NewHead instead.")

    def get_model(self, task_id, **task_infos):
        task_model = super().get_model(task_id, **task_infos)

        if task_id >= self.finished_tasks and task_id > 0:
            source_model = self.source_ll_model.get_model(task_id - 1)
            task_model.load_source(source_model)

        return task_model

    def _new_model(self, x_dim, n_classes, task_id, **kwargs):
        if self.all_dims is None:
            if len(x_dim) == 1:
                self.n_convs = 0
            else:
                k = 3
                stride = 2
                second_layer_map_size = [(s - (k - 1) + 1) // stride for s in
                                     x_dim[1:]]
                self.hidden_size[0] = (self.hidden_size[0], *second_layer_map_size)
                self.n_convs = 1
            self.all_dims = [x_dim, *self.hidden_size]
        if self.source_ll_model is None:
            raise ValueError('Source model should be set before creating the '
                             'first model for FineTuneLegLLModel !')

        head = MultiHead(self.all_dims[-1], n_classes)
        common_trunc = self.base_model_func(self.all_dims[1:], n_convs=self.n_convs)
        new_trunc = make_model(common_trunc, head)

        new_leg = self.base_model_func(self.all_dims[:2])
        new_model = make_model(new_leg, new_trunc, seq=FineTuneLegMultiTask)

        return new_model

    def set_source_model(self, source):
        assert self.source_ll_model is None, 'Source already set for FineTune.'
        self.source_ll_model = source

    def finish_task(self, dataset):
        self.finished_tasks += 1


class FineTuneLegMultiTask(nn.Module):
    def __init__(self, leg, trunc=None):
        super(FineTuneLegMultiTask, self).__init__()
        self.trunc = trunc
        self.leg = leg

    def forward(self, input):
        return self.trunc(self.leg(input))

    def load_source(self, source):
        assert type(source) == MTLegModel
        self.leg.load_state_dict(source.legs[-1].state_dict())
        self.trunc.load_state_dict(source.trunc.state_dict())
