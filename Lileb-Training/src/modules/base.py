# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


def LinearNet(sizes, dropout_p):
        layers = []
        last_size = sizes[0]
        if isinstance(last_size, torch.Size):
            assert len(last_size) == 1
            last_size = last_size[0]
        for size in sizes[1:]:
            layers.extend([nn.Linear(last_size, size),
                           nn.ReLU(),
                           nn.Dropout(dropout_p)])
            last_size = size
        return nn.Sequential(*layers)

class ConvNet(nn.Module):
    def __init__(self, sizes, n_convs, dropout_p, k, stride):
        super(ConvNet, self).__init__()
        # first n_convs layers are conv
        conv_sizes = sizes[1:n_convs + 1]

        # following layers are fc
        lin_sizes = sizes[n_convs + 1:]

        layers = []
        first_layer_dims = sizes[0]
        assert len(first_layer_dims) == 3
        img_size = first_layer_dims[1:]

        last_n_filters = first_layer_dims[0]
        for size in conv_sizes:
            if not isinstance(size, int):
                # get the number of filters.
                assert len(size) == 3
                size = size[0]
            layers.extend([nn.Conv2d(last_n_filters, size, k, stride),
                           nn.ReLU()])

            img_size = [(s - (k-1) + 1)//stride for s in img_size]
            last_n_filters = size

        n_features = last_n_filters * img_size[0] * img_size[1]

        self.convs = nn.Sequential(*layers)
        self.fcs = LinearNet([n_features, *lin_sizes], dropout_p=dropout_p)

    def forward(self, x):
        x = self.convs(x)
        if self.fcs:
            x = self.fcs(x.view(x.size(0), -1))
        return x


def get_block_model(dims, dropout_p, n_convs=0):
    net = []
    if dims:
        if isinstance(dims[0], int) or len(dims[0]) == 1:
            assert n_convs == 0
            net.append(LinearNet(dims, dropout_p=dropout_p))
        elif len(dims[0]) == 3:
            if n_convs == 0:
                n_convs = 2
            net.append(ConvNet(dims, n_convs, dropout_p, k=3, stride=2))

    return nn.Sequential(*net)
