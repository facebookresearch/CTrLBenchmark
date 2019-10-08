# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

from sacred import Experiment
from sacred.config import load_config_file
from sacred import SETTINGS
from sacred.utils import recursive_update


SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = True
SETTINGS.HOST_INFO.CAPTURED_ENV = [
    'OMP_NUM_THREADS',
    'CUDA_VISIBLE_DEVICES',
    'SLURM_JOB_ID'
]


def load_component_default_config(component_config, all_default_configs):
    component_default_config = {}
    if '_name' in component_config:
        elt_default = deepcopy(all_default_configs.get(component_config['_name'], {}))
        default = load_component_default_config(elt_default, all_default_configs)
        recursive_update(default, elt_default)
        component_default_config.update(default)
    for key, val in component_config.items():
        if isinstance(val, dict):
            conf = load_component_default_config(val, all_default_configs)
            if conf:
                component_default_config[key] = conf

    return component_default_config


def load_default_config(config, command_name, logger):
    default_config = {}
    for comp, comp_conf in config.items():
        fn = f'configs/default/{comp}.yaml'
        if os.path.isfile(fn):# and ('_name' not in comp_conf):
            comp_default_configs = load_config_file(fn)
            if '_name' in comp_conf:
                comp_default_config = load_component_default_config(comp_conf, comp_default_configs)
            else:
                comp_default_config = {}
                for mod, mod_config in comp_conf.items():
                    if isinstance(mod_config, dict):
                        comp_default_config[mod] = load_component_default_config(mod_config, comp_default_configs)

            default_config[comp] = comp_default_config

    return default_config


def wrap_with_sacred(command, ex_hook):
    ex = Experiment()
    ex_hook(ex)
    ex.config_hook(load_default_config)
    ex.main(command)
    ex.run_commandline()
