# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


class Transformation(object):
    def __init__(self, transfo_pool, path, trans_descr):
        assert path[0] == transfo_pool.root_node
        self.transfo_pool = transfo_pool
        self.path = path
        self.trans_descr = trans_descr

    def __call__(self, X):
        with torch.no_grad():
            for u, v in zip(self.path, self.path[1:]):
                f = self.transfo_pool.tree.edges()[u, v]['f']
                X = f(X)
        return X

    def __str__(self):
        return self.trans_descr