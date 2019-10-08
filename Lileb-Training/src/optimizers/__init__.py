# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.optim as optim


def get_optim_by_name(name):
    if name == 'sgd':
        return optim.SGD
    elif name == 'adam':
        return optim.Adam
    raise NotImplementedError(name)


def get_lr_scheduler_by_name(name):
    if name == 'None':
        return None
    elif name == 'exponential':
        return optim.lr_scheduler.exponential
    else:
        raise NotImplementedError(name)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_lr_scheduler(optimizer, _name, **kwargs):
    return get_lr_scheduler_by_name(_name)(optimizer, **kwargs)


def init_optimizer(modules, _name, _modules, lr_scheduler=None, **kwargs):
    """
    Will select all modules if the _modules attribute is an empty list
    :param modules:
    :param _name:
    :param _modules:
    :param lr_scheduler:
    :param kwargs:
    :return:
    """
    if isinstance(_modules, str):
        _modules = [_modules]

    if not _modules:
        _modules = list(modules.keys())

    optim = get_optim_by_name(_name)
    optim_func = partial(optim, **kwargs)

    for name in _modules:
        module = modules[name]
        module.optim_func = optim_func
