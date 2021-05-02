"""
This module contains a bunch of code extracted from
https://github.com/TomVeniat/MNTDP in order to allow the usage of automatic
configuration and initialization on the CTrL benchmark.
"""
import collections
import os
from os import path
from copy import deepcopy

import yaml
from numpy.random import default_rng

from ctrl.instances.image_dataset_tree import ImageDatasetTree
from ctrl.instances.md_tree import MultiDomainDatasetTree
from ctrl.strategies import InputDomainMutationStrategy, SplitStrategy, \
    IncrementalStrategy, RandomMutationStrategy, DataStrategy, \
    AttributeStrategy, MixedStrategy, LabelPermutationStrategy
from ctrl.tasks.task_generator import TaskGenerator
from ctrl.transformations import RandomNNTransformationTree, \
    ImgRotationTransformationTree, RandomPermutationsTransformation, \
    IdentityTransformation, NoisyNNTransformationTree, \
    RainbowTransformationTree


def get_component_by_name(name):
    if name in ['cifar10_tree', 'cifar100_tree', 'mnist_tree', 'svhn_tree',
                'fashion_mnist_tree', 'dtd_tree', 'aircraft_tree']:
        return ImageDatasetTree
    if name.startswith('md_tree'):
        return MultiDomainDatasetTree

    if name == 'nn_x_transformation':
        return RandomNNTransformationTree
    if name == 'img_rot_x_transformation':
        return ImgRotationTransformationTree
    if name == 'randperm_x_transformation':
        return RandomPermutationsTransformation
    if name == 'id_x_transformation':
        return IdentityTransformation
    if name == 'noisy_nn_x_transformation':
        return NoisyNNTransformationTree
    if name == 'rainbow_x_transformation':
        return RainbowTransformationTree

    if name == 'transfo':
        return InputDomainMutationStrategy
    if name == 'split':
        return SplitStrategy
    if name == 'incremental':
        return IncrementalStrategy
    if name == 'random':
        return RandomMutationStrategy
    if name == 'data':
        return DataStrategy
    if name == 'attributes':
        return AttributeStrategy
    if name.startswith('mixed'):
        return MixedStrategy
    if name == 'label_permut':
        return LabelPermutationStrategy

    if name == 'task_gen':
        return TaskGenerator

    raise NotImplementedError(name)


def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def recursive_update(d, u):
    """
    From Sacred (https://github.com/IDSIA/sacred).
    Given two dictionaries d and u, update dict d recursively.

    E.g.:
    d = {'a': {'b' : 1}}
    u = {'c': 2, 'a': {'d': 3}}
    => {'a': {'b': 1, 'd': 3}, 'c': 2}
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            r = recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def load_component_default_config(component_config, all_default_configs):
    component_default_config = {}
    if '_name' in component_config:
        elt_default = deepcopy(all_default_configs.get(
            component_config['_name'], {}))
        default = load_component_default_config(elt_default,
                                                all_default_configs)
        recursive_update(default, elt_default)
        component_default_config.update(default)
    for key, val in component_config.items():
        if isinstance(val, dict):
            conf = load_component_default_config(val, all_default_configs)
            if conf:
                component_default_config[key] = conf

    return component_default_config


def load_default_config(config):
    fn = path.join(path.dirname(__file__), 'default_datasets.yaml')
    # fn = f'./streams/default_datasets.yaml'
    if os.path.isfile(fn):# and ('_name' not in comp_conf):
        comp_default_configs = load_yaml(fn)
        if '_name' in config:
            comp_default_config = load_component_default_config(config, comp_default_configs)
        else:
            comp_default_config = {}
            for mod, mod_config in config.items():
                if isinstance(mod_config, dict):
                    comp_default_config[mod] = load_component_default_config(mod_config, comp_default_configs)

    return comp_default_config


def init_component(_rnd, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, dict):
            v = init_component(_rnd=_rnd, **v)
        kwargs[k] = v
    if '_name' in kwargs:
        comp_class = get_component_by_name(kwargs.pop('_name'))
        return comp_class(seed=_rnd.integers(0, 1e9), **kwargs)
    else:
        return kwargs


def get_stream(name, seed=None):
    config_path = path.join(path.dirname(__file__), f'{name}.yaml')
    stream_config = load_yaml(config_path)
    config = load_default_config(stream_config)
    recursive_update(config, stream_config)
    return init_component(default_rng(seed), **config)['task_gen']




