# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gridfs
from pymongo import MongoClient
from sacred.observers import MongoObserver
from visdom import Visdom

from src.utils import load_conf

###
# Mongodb
###


def get_mongo_connection_url(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)
    if 'user' in mongo_conf and 'passwd' in mongo_conf:
        db_user = '{}:{}'.format(mongo_conf['user'], mongo_conf['passwd'])
    else:
        db_user = None

    db_host = '{}:{}'.format(mongo_conf['host'], mongo_conf['port'])
    auth_db = mongo_conf.get('auth_db', mongo_conf['db'])
    return f'mongodb://{db_user}@{db_host}/{auth_db}' if db_user else f'mongodb://{db_host}'


def get_mongo_db(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)

    connection_url = get_mongo_connection_url(mongo_conf)
    return MongoClient(connection_url)[mongo_conf['db']]


def get_mongo_collection(mongo_conf=None, mongo_path=None, collection=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)
    db = get_mongo_db(mongo_conf)

    if collection is None:
        collection = mongo_conf['collection']

    return db[collection]


def get_mongo_obs(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)

    db_url = get_mongo_connection_url(mongo_conf)
    return MongoObserver.create(url=db_url, db_name=mongo_conf['db'], collection=mongo_conf['collection'])


def get_gridfs(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)

    return gridfs.GridFS(get_mongo_db(mongo_conf))


###
# Visdom
###
def get_visdom(visdom_path=None, **conf_updates):
    visdom_conf = dict(raise_exceptions=True)

    if visdom_path is not None:
        visdom_conf.update(load_conf(visdom_path))

    visdom_conf.update(conf_updates)
    return Visdom(**visdom_conf)
