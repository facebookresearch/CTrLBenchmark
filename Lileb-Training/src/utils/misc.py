# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import time

from tqdm import tqdm

import src.utils.external_resources as external


def get_env_url(visdom_client, replace=('devfair054', 'localhost')):
    res = '{}:{}/env/{}'.format(visdom_client.server, visdom_client.port, visdom_client.env)
    if replace:
        res = res.replace(*replace)
    return res


def rename_class(cls, new_name):
    cls.__name__ = new_name
    qualname = cls.__qualname__.split('.')
    qualname[-1] = new_name
    cls.__qualname__ = '.'.join(qualname)
    return cls


def get_runs(sacred_ids=None, slurm_ids=None, mongo_conf_path=None):
    if sacred_ids:
        assert not slurm_ids, 'Can\'t specify both sacred and slurm ids'
        ids = sacred_ids
        req = {'_id': {"$in": ids}}
    elif slurm_ids:
        ids = [str(id) for id in slurm_ids]
        req = {'host.ENV.SLURM_JOB_ID': {"$in": ids}}
    else:
        raise ValueError('Should specify one of --sacred-ids or --slurm-ids.')

    mongo_collection = external.get_mongo_collection(mongo_path=mongo_conf_path)
    runs = mongo_collection.find(req)
    n_res = mongo_collection.count_documents(req)
    if n_res < len(ids):
        retrieved = set(r['_id'] if sacred_ids else
                        r['host']['ENV']['SLURM_JOB_ID'] for r in runs)
        missing = set(ids) - retrieved
        raise ValueError('Missing runs, expected {} but got {} (missing {}).'
                         .format(len(ids), runs.count(), missing))
    if n_res > len(ids):
        raise ValueError('Big problem: More results that ids (runs coming from '
                         'different Slurm clusters ?)')
    return runs


def replay_run(run, viz, args, mongo_path, logger):
    if not run['artifacts']:
        logger.warning(
            'Run {} doesn\'t have any stored file'.format(run['_id']))
        return None, 0, 0

    selected_artifact = None
    for artifact in reversed(run['artifacts']):
        if artifact['name'].endswith(args.name_end):
            selected_artifact = artifact
            break

    if selected_artifact is None:
        available = [a['name'] for a in run['artifacts']]
        raise ValueError('No artifact ending with \'{}\' in run {}. Available'
                         ' artifacts are {}'.format(args.name_end, args.id,
                                                    available))
    start_time = time.time()
    gridfs = external.get_gridfs(mongo_path=mongo_path)
    object = gridfs.get(selected_artifact['file_id'])

    with tempfile.TemporaryDirectory() as dir:
        file_path = os.path.join(dir, selected_artifact['name'])
        with open(file_path, 'wb') as file:
            file.write(object.read())

        target = os.path.join(dir, selected_artifact['name'][:-4])
        shutil.unpack_archive(file_path, target)
        env_files = os.listdir(target)
        logger.info('Replaying envs for {}'.format(run['_id']))
        n_replayed = 0
        main_env = None
        for file in tqdm(env_files):
            if 'main' in file:
                main_env = file
            if args.all_envs or 'main' in file or 'tasks' in file:
                viz.delete_env(file)
                viz.replay_log(os.path.join(target, file))
                n_replayed += 1
    tot_time = time.time() - start_time
    logger.info('Replayed {} envs in {:.3f} seconds.'.format(n_replayed,
                                                             tot_time))
    if main_env is None:
        logger.warning('No main env file found.')
    else:
        logger.info('Main env is {}.'.format(main_env))
        viz.env = main_env
    logger.info(get_env_url(viz))
    return get_env_url(viz), n_replayed, tot_time

