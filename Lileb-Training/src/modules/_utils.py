# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch.nn as nn
from torch.nn import init

# from src.utils.misc import pretty_wrap


def load_state_dict(module, state_dict_path):
    print(f'Loading {_name} from {state_dict_path} ...')
    state_dict = torch.load(state_dict_path)#.state_dict()
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    module.load_state_dict(state_dict)
    module.eval()


def to(module, gpu_id):
    if len(gpu_id) == 1:
        module.to(f'cuda:{gpu_id[0]}')
    elif len(gpu_id) > 1:
        module = torch.nn.DataParallel(module, gpu_id).cuda()
    return module


def init_weights(net, name='normal', gain=0.02, gain_bn=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if name == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif name == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif name == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif name == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {name} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.constant_(m.weight.data, gain_bn)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch1d':
        norm_layer = partial(nn.BatchNorm1d, affine=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'instance1d':
        norm_layer = partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        return partial(nn.ReLU, inplace=True)
    if layer_type == 'lrelu':
        return partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    if layer_type == 'elu':
        return partial(nn.ELU, inplace=True)
    if layer_type == 'tanh':
        return nn.Tanh
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)


def num_params(module):
    num_params = 0
    for param in module.parameters():
        num_params += param.numel()
    return num_params


def print_net(name, net, init_name, init_gain, init_gain_bn=None, init_gain_blocks=None, init_gain_blocks_bn=None):
    s = f'Class: {net.__class__.__name__}\n' \
        f'Init: {init_name}, Gain={init_gain} Gain BN: {init_gain_bn}\n' \
        f'    Blocks: Gain {init_gain_blocks} Gain BN: {init_gain_blocks_bn}\n' \
        f'Number of parameters : {num_params(net) / 1e6:.6f}\n'
    print(pretty_wrap(title=name, text=s))
