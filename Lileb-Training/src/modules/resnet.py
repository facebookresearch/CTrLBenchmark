# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchvision.models.resnet import conv3x3, conv1x1

from src.modules.utils import Flatten


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _make_layer(block, in_dim, out_dim, blocks, stride=1, norm_layer=None,
                is_first=False, is_last=False):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    if is_last:
        return [nn.AvgPool2d(in_dim[1:]),
                Flatten(),
                nn.Linear(in_dim[0], out_dim)]

    in_planes = in_dim[0]
    planes = out_dim[0]
    if is_first:
        return [conv3x3(in_planes, planes),
                norm_layer(planes),
                nn.ReLU()]


    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(in_planes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(in_planes, planes, stride, downsample, norm_layer))
    in_planes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(in_planes, planes, norm_layer=norm_layer))

    return layers
