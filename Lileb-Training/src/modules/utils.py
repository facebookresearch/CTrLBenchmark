# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class MultiHead(nn.Module):
    def __init__(self, in_size, out_sizes, *args, **kwargs):
        super(MultiHead, self).__init__(*args, **kwargs)
        if isinstance(in_size, torch.Size):
            assert len(in_size) == 1, 'Multhihead expect 1d inputs, got {}'\
               .format(in_size)
            in_size = in_size[0]

        heads = [nn.Linear(in_size, out) for i, out in enumerate(out_sizes)]
        # heads = [nn.Linear(in_size, 1 if out in [1, 2] else out) for i, out in enumerate(out_sizes)]
        self.heads = nn.ModuleList(heads)
        self.n_out = len(out_sizes)

    def forward(self, input):
        return [head(input) for head in self.heads]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def make_model(*blocks, seq=nn.Sequential):
    blocks_list = []
    for block in blocks:
        if isinstance(block, nn.Module):
            block = [block]
        assert isinstance(block, list)
        blocks_list += block

    model = seq(*blocks_list)

    model.n_out = blocks_list[-1].n_out
    return model


def get_conv_out_size(in_size, kernel_size, padding, stride):
    return [(d+padding - (kernel_size-1) + 1) // stride for d in in_size]
