# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

VISDOM_CONF_PATH = 'resources/visdom.yaml'
MONGO_CONF_PATH = 'resources/mongo.yaml'
LOCAL_SAVE_PATH = '/checkpoint/veniat/runs'


def load_conf(path):
    _, ext = os.path.splitext(path)
    with open(path) as file:
        if ext == '.json':
            import json
            conf = json.load(file)
        elif ext in ['.yaml', '.yml']:
            import yaml
            conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf
