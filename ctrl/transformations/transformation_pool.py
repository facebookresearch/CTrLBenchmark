# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc


class TransformationPool(abc.ABC):
    @abc.abstractmethod
    def get_transformation(self, exclude_trans=None):
        raise NotImplementedError

    @abc.abstractmethod
    def transformations_sim(self, t1, t2):
        raise NotImplementedError

