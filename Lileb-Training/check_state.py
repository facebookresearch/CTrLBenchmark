# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pprint import pprint

import ray

def get_arg_parser():
    parser = argparse.ArgumentParser('Monitoring')
    parser.add_argument('--port', type=str, default=6385)
    parser.add_argument('--hostname', type=str)
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    ray.init(redis_address="{}:{}".format(args.hostname, args.port))
    key = 'NodeManagerAddress'
    val = 'Resources'
    pprint(sorted([{node[key]: node[val]} for node in ray.nodes()], key=lambda x: x[key]))
    print(ray.cluster_resources())
    print(ray.available_resources())
