# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torch import nn

from src.modules.ll_model import LifelongLearningModel
from src.modules.resnet import _make_layer, BasicBlock
from src.modules.utils import get_conv_out_size, Flatten

logger = logging.getLogger(__name__)


def _conv_block(in_dim, out_dim, k, stride, is_last):
    assert len(in_dim) == 3
    layers = [nn.Conv2d(in_dim[0], out_dim[0], k, stride)]
    if not is_last:
        layers.append(nn.ReLU())
    return layers


def _residual_block(in_dim, out_dim, stride, is_first, is_last):
    return _make_layer(BasicBlock, in_dim, out_dim, 2, stride=stride,
                       is_first=is_first, is_last=is_last)


def _lin_block(in_dim, out_dim, dropout_p, is_last):
    layers = []
    if not isinstance(in_dim, int):
        in_dim = in_dim[0] * in_dim[1] * in_dim[2]
        layers.append(Flatten())

    layers.append(nn.Linear(in_dim, out_dim))

    if not is_last:
        layers += [nn.ReLU(),
                   nn.Dropout(dropout_p)]

    return layers


def get_block(in_dim, out_dim, dropout_p, k, stride, is_first, is_last, residual):
    if residual:
        layers = _residual_block(in_dim, out_dim, stride, is_first, is_last)
    elif isinstance(out_dim, int) or len(out_dim) == 1:
        # Linear layer
        layers = _lin_block(in_dim, out_dim, dropout_p, is_last)
    elif len(out_dim) == 3:
        layers = _conv_block(in_dim, out_dim, k, stride, is_last)
    else:
        raise ValueError('Don\'t know which kind of layer to use for input '
                         'size {} and output size {}.'.format(in_dim, out_dim))

    return nn.Sequential(*layers)


class ChangeLayerLLModel(LifelongLearningModel):
    def __init__(self, share_layer, k, stride, residual, *args, **kwargs):
        super(ChangeLayerLLModel, self).__init__(*args, **kwargs)
        self.common_model = None
        self.share_layer = share_layer
        self.common_layers = None

        self.k = k
        self.padding = 0

        if residual:
            assert self.k == 3, 'Can only use kernels of size 3 with the resnet'
            stride = [1, 1, 2, 2, 2, None]
            self.padding = 1
            if len(self.hidden_size) > 1:
                logger.warning('Several hidden sizes where specified for '
                               'resnet, only the first one is going to be used')
                first = self.hidden_size[0]
                self.hidden_size = [first] + [first * 2**i for i in range(4)]
                self.n_convs = 5

        elif isinstance(stride, int):
            stride = [stride] * len(self.share_layer)

        # self.sh
        self.residual = residual
        self.strides = stride

    def _new_model(self, x_dim, n_classes, **kwargs):
        if len(x_dim) == 1:
            x_dim = x_dim[0]
        assert len(n_classes) == 1, 'Only supports single output'
        n_classes = n_classes[0]

        # Put all dimensions together for current model.
        model_dims = [x_dim, *self.hidden_size, n_classes]

        if not isinstance(x_dim, int):
            # Input is an image, we need to calculate the intermediate map sizes
            for i in range(self.n_convs):
                img_size = model_dims[i][1:]
                stride = self.strides[i]
                out_size = get_conv_out_size(img_size, self.k,
                                             self.padding, stride)
                model_dims[i+1] = [model_dims[i+1], *out_size]

        if self.common_layers is None:
            # Need to init the shared layers
            self._init_common_layers(model_dims)

        model = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(model_dims, model_dims[1:])):
            layer = self._get_layer(i, in_dim, out_dim)
            model.add_module(str(i), layer)
        model.n_out = 1
        return model

    def _init_common_layers(self, model_dims):
        self.common_layers = {}
        for i, (in_dim, out_dim) in enumerate(zip(model_dims, model_dims[1:])):
            if self.share_layer[i]:
                is_first = i == 0
                is_last = i == len(self.share_layer) - 1
                stride = self.strides[i]
                self.common_layers[i] = get_block(in_dim, out_dim,
                                                  self.dropout_p, self.k,
                                                  stride, is_first, is_last,
                                                  self.residual)

    def _get_layer(self, i, in_dim, out_dim):
        if self.share_layer[i]:
            layer = self.common_layers[i]
        else:
            is_first = i == 0
            is_last = i == len(self.share_layer) - 1
            stride = self.strides[i]
            layer = get_block(in_dim, out_dim, self.dropout_p,
                              self.k, stride, is_first, is_last,
                              self.residual)
        return layer


    def finish_task(self, dataset):
        pass
