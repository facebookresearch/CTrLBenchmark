# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import make_model, MultiHead


class NewLegLLModel(LifelongLearningModel):
    def __init__(self, *args, **kwargs):
        super(NewLegLLModel, self).__init__(*args, **kwargs)

        self.common_model = None
        self.n_classes = None
        if len(self.hidden_size) == 0:
            raise NotImplementedError("NewLeg and NewHead are equivalent "
                                      "when no hidden layers are used, "
                                      "please use NewHead instead.")

    def _new_model(self, x_dim, n_classes, **kwargs):
        if self.common_model is None:
            # This is the first task
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
            common_trunc = self.base_model_func(self.hidden_size,
                                                n_convs=n_convs)
            head = MultiHead(self.all_dims[-1], n_classes)
            self.common_model = make_model(common_trunc, head)

        assert n_classes == self.n_classes, 'Number of classes should always ' \
                                            'be the same with NewLegLLModel'

        new_leg = self.base_model_func(self.all_dims[:2])
        return make_model(new_leg, self.common_model)

    def finish_task(self, dataset):
        pass
