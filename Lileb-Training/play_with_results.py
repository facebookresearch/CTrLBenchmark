# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import logging
import os
import pprint
from collections import OrderedDict

import pandas as pd
import numpy as np
import visdom
from deepdiff import DeepDiff
from pandas.io.clipboard import clipboard_set

from src.utils.misc import get_runs, replay_run, get_env_url

MONGO_CONF_PATH = 'resources/mongo_replay.yaml'

EXCLUDED_CONFIG_PATH = {"root['seed']",
                        "root['experiment']['log_steps']",
                        "root['datasets']['task_gen']['concept_pool']"
                        "['n_samples_per_concept']",
                        "root['experiment']['grace_period']",
                        "root['experiment']['n_it_max']",}

KEY_ALIASES = {"root['datasets']['task_gen']['samples_per_class'][0]":
                   "samples_per_class"}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser('Replay')
    parser.add_argument('--sacred-ids', type=int, nargs='+',
                        # default=[659]
                        # default=[834, 835, 836, 837, 838, 839, 841, 842, 843, 801, 802, 821, 819, 820, 825, 822, 823, 824]
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

                        # default=[
                        #     17214247,
                        #     17214249,
                        #     17214252,
                        #     17214253
                        #     ]
                        )
    parser.add_argument('--host', '--visdom-hostname', type=str,
                        default='localhost')
    parser.add_argument('--port', '--visdom-port', type=str,
                        default='8097')
    parser.add_argument('--name-end', type=str, default='traces.zip')
    parser.add_argument('--replay', '-r', action='store_true', default=False)
    parser.add_argument('--all-envs', '-a', action='store_true', default=False)
    parser.add_argument('--group', action='store_true', default=False)

    return parser.parse_args()


def get_key(config):
    return config['datasets']['task_gen']['samples_per_class'][0]


def validate_configurations(configurations):
    diffs = {}
    ref_id, ref_conf = next(iter(configurations.items()))
    logger.info('Reference conf:')
    logger.info(pprint.pformat(ref_conf))
    for id, config in configurations.items():
        diff = DeepDiff(ref_conf,
                        config,
                        exclude_paths=EXCLUDED_CONFIG_PATH)
        if any(k.endswith('item_added') or k.endswith('item_removed')
               for k in diff.keys()):
            logging.warning('Suspicious diff: {}'.format(pprint.pformat(diff)))
        if 'values_changed' not in diff:
            continue
        for k, v in diff['values_changed'].items():
            k = KEY_ALIASES.get(k, k)
            if k not in diffs:
                diffs[k] = OrderedDict()
                diffs[k][ref_id] = v['old_value']
            diffs[k][id] = v['new_value']
    if len(diffs) > 1:

        logger.warning('More than one varying thing in configs:{}'
                       .format(pprint.pformat(diffs)))
        raise ValueError
    return next(iter(diffs.items()))


def log_failed(failed_runs):
    for run in failed_runs:
        logger.warning('Exp {} (slurm: {}) didn\'t return anything, STATUS'
                       '={}'.format(run['_id'],
                                    run['host']['ENV'].get('SLURM_JOB_ID', None),
                                    run['status']))


def main(args):
    if not os.path.isfile(MONGO_CONF_PATH):
        raise ValueError('File {} must exist'.format(MONGO_CONF_PATH))

    runs = get_runs(args.sacred_ids, args.slurm_ids, MONGO_CONF_PATH)

    viz = visdom.Visdom(args.host, port=args.port)

    envs = []
    tot_time = 0
    n_replayed = 0

    configs = {}
    results = {}
    failed = []
    index = None
    for run in runs:
        slurm_id = run['host']['ENV'].get('SLURM_JOB_ID', None)
        logger.info('\nProcessing run {} ({})'.format(run['_id'], slurm_id))
        if args.replay:
            # Replay part
            env, n, time = replay_run(run, viz, args, MONGO_CONF_PATH, logger)
            envs.append(env)
            tot_time += time
            n_replayed += n

        # Results aggregation part
        res = run['result']
        if not res:
            failed.append(run)
            continue
        configs[run['_id']] = run['config']
        if index is None:
            index = res['model']
        assert res['model'] == index
        # results[run['_id']] = {'Avg acc': res['accuracy']}
        metrics = ['accuracy', 'speed']
        accs = [((metric, mod), val) for metric in metrics
                for mod, val in zip(res['model'], res[metric])]
        results[run['_id']] = dict(accs)
        # results[(get_key(run['config']), run['_id'])] = res['accuracy']

    logger.info('Done.')
    logger.info('Replayed {} envs in {:.3f} seconds.'.format(n_replayed,
                                                             tot_time))
    logger.info(envs)

    log_failed(failed)

    key, values = validate_configurations(configs)
    # res = pd.DataFrame(results, index=index)
    res = pd.DataFrame(results)
    # res.loc['ids'] = res.columns

    ref_val = next(iter(values.values()))
    new_cols = [(values.get(k, ref_val), k) for k in res.columns]
    new_cols = pd.MultiIndex.from_tuples(new_cols, names=[key, '_id'])
    res.columns = new_cols
    res.sort_index(axis=1, inplace=True)
    if args.group:
        # Group and compute statistics over runs
        res = res.transpose().groupby(level=key).agg([np.mean, np.std]).transpose()

        # agg_funcs = {'accuracy': lambda grp: grp.groupby(level=0).apply(lambda x: x.round(3).astype(str).apply('±'.join, 0)),
        #              'speed': lambda grp: grp.groupby(level=0).apply(lambda x: x.round(1).astype(str).apply('±'.join, 0))}
        # Process the results to get everything in the "mean±std" format
        res = res.groupby(level=[0, 1]).apply(
            lambda x: x.round(3).astype(str).apply('±'.join, 0))

    sacred_ids = list(results.keys())
    print(sacred_ids)
    print(res)

    id_str = ' '.join(map(str, sacred_ids))
    id_row = 'ids\t{}'.format(id_str)

    buf = io.StringIO()
    res.to_csv(buf, sep='\t', encoding='utf-8', )
    txt = buf.getvalue()
    txt += id_row
    clipboard_set(txt)
    viz = visdom.Visdom(args.host, port=args.port)
    viz.text(res.to_html(classes=['table', 'table-bordered', 'table-hover']))
    logger.info(get_env_url(viz))

    log_failed(failed)

if __name__ == '__main__':
    args = get_args()
    main(args)
