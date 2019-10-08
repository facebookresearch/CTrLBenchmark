# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import shutil
import tempfile
import time

import visdom
from tqdm import tqdm

import src.utils.external_resources as external
from src.utils.misc import get_env_url, get_runs, replay_run

MONGO_CONF_PATH = 'resources/mongo_replay.yaml'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser('Replay')
    parser.add_argument('--sacred-ids', type=int, nargs='+',
                        # default=[604, 605, 606]
                        )
    parser.add_argument('--slurm-ids', type=int, nargs='+',
                        # default=[17205191,
                        #          17205193,
                        #          17205195,
                        #          17205196]
                        # default=[
                        #     17209924,
                        #     17209925,
                        #     17209926,
                        #     17209928,
                        #     17209929,
                        #     17209930,
                        #     17209931,
                        #     17209933,
                        # ]

                        default=[
                            17214247,
                            17214249,
                            17214252,
                            17214253
                            ]
                        )
    parser.add_argument('--host', '--visdom-hostname', type=str,
                        default='localhost')
    parser.add_argument('--port', '--visdom-port', type=str,
                        default='8097')
    parser.add_argument('--name-end', type=str, default='traces.zip')
    parser.add_argument('--main-only', action='store_true', default=False)

    return parser.parse_args()


def main(args):
    if not os.path.isfile(MONGO_CONF_PATH):
        raise ValueError('File {} must exist'.format(MONGO_CONF_PATH))

    runs = get_runs(args.sacred_ids, args.slurm_ids, MONGO_CONF_PATH)

    viz = visdom.Visdom(args.host, port=args.port)

    envs = []
    tot_time = 0
    n_replayed = 0
    for run in runs:
        env, n, time = replay_run(run, viz, args, MONGO_CONF_PATH, logger)
        envs.append(env)
        tot_time += time
        n_replayed += n

    logger.info('Done.')
    logger.info('Replayed {} envs in {:.3f} seconds.'.format(n_replayed,
                                                             tot_time))
    logger.info(envs)


if __name__ == '__main__':
    args = get_args()
    main(args)
