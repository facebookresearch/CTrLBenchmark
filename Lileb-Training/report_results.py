# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import os

import visdom
import pandas as pd

from src.utils.misc import get_env_url, get_runs
from datadiff import diff

MONGO_CONF_PATH = 'resources/mongo_replay.yaml'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_key(config):
    return config['datasets']['task_gen']['samples_per_class'][0]

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
                        default=[
                            17214247,
                            17214249,
                            17214252,
                            17214253,
                        #     # 17209929,
                        #     # 17209930,
                        #     # 17209931,
                        #     # 17209933,
                        ]
                # default = [17214573,
                #            # 17214574,
                #            17214575,
                #            17214576,
                #            17214577,
                #            17214578,
                #            17214579,
                #            17214580]

    )
    parser.add_argument('--host', '--visdom-hostname', type=str,
                        default='localhost')
    parser.add_argument('--port', '--visdom-port', type=str,
                        default='8097')
    # parser.add_argument('--key', type=str,  default='datasets.task_gen.samples_per_class.0')

    return parser.parse_args()


def main(args):
    if not os.path.isfile(MONGO_CONF_PATH):
        raise ValueError('File {} must exist'.format(MONGO_CONF_PATH))

    runs = get_runs(args.sacred_ids, args.slurm_ids, MONGO_CONF_PATH)

    results = {}
    index = None
    for r in runs:
        res = r['result']
        if index is None:
            index = res['model']
        assert res['model'] == index
        results[(get_key(r['config']), r['_id'])] = res['accuracy']

    res = pd.DataFrame(results, index=index)
    res.sort_index(axis=1, inplace=True)
    print(res.to_clipboard())

    viz = visdom.Visdom(args.host, port=args.port)
    viz.text(res.to_html(classes=['table', 'table-bordered', 'table-hover']))

    logger.info(get_env_url(viz))

