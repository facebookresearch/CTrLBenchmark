# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import sacred
from sacred.randomness import get_seed
from tqdm import tqdm
from sacred.observers import FileStorageObserver

from src.experiments import init_experiment
from src.datasets import init_dataset
from src.modules import init_module
from src.optimizers import init_optimizer
from src.utils import MONGO_CONF_PATH, LOCAL_SAVE_PATH, VISDOM_CONF_PATH, load_conf
from src.utils.external_resources import get_mongo_obs
from src.utils.log_observer import LogObserver
from src.utils.sacred import wrap_with_sacred


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end='')


logging.basicConfig(level=logging.INFO, stream=TqdmStream)


# def simple_wrapper(datasets, experiment, modules, optimizers, **kwargs):
#     config = dict(ds=datasets, opt=optimizers, mod=modules, ex=experiment)
#     config = flatten_config(config)
#
#     main(config)


def init_and_run(experiment, modules, datasets, optimizers={}, _run=None, _rnd=None):
    # initializing datasets

    dsets = {}
    for dataset_name, dataset_config in sorted(datasets.items()):
        dsets[dataset_name] = init_dataset(**dataset_config, _rnd=_rnd)

    # initializing modules
    mods = {}
    for module_name, module_config in modules.items():
        mods[module_name] = init_module(**module_config)

    # initializing optimizers
    for optimizer_name, optimizer_config in optimizers.items():
        init_optimizer(mods['ll_models'], **optimizer_config)

    # initializing experiment and running it
    exp = init_experiment(sacred_run=_run, seed=get_seed(_rnd),
                    **dsets, **mods, **experiment)
    return exp.run()


def set_observers(experiment):
    if os.path.isfile(VISDOM_CONF_PATH):
        visdom_conf = load_conf(VISDOM_CONF_PATH)
        experiment.observers.append(LogObserver.create(visdom_conf))

    if os.path.isfile(MONGO_CONF_PATH):
        experiment.observers.append(get_mongo_obs(mongo_path=MONGO_CONF_PATH))
    else:
        experiment.observers.append(FileStorageObserver.create(LOCAL_SAVE_PATH))


if __name__ == '__main__':

    sacred.SETTINGS['CAPTURE_MODE'] = 'no'
    sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
    wrap_with_sacred(init_and_run, ex_hook=set_observers)
