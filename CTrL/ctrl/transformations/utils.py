# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


class BatchedTransformation(object):
    def __init__(self, transfo, descr=None):
        self.transfo = transfo
        self.descr = descr

    def __call__(self, batch):
        if torch.is_tensor(batch):
            batch = batch.unbind(0)
        res = [self.transfo(elt) for elt in batch]
        return torch.stack(res, 0)

    def __str__(self):
        if self.descr is None:
            return super().__str__()
        else:
            return self.descr