# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import scipy


from src.gmeval.distributions import _FuzzedExpansion, FlatGaussian, IsometricTransform


class Category(object):
    def __init__(self, concept, sampling_distrib):
        # assert concept.intrinsic_dim == n_dims
        self.concept = concept
        self.sampling_distrib = sampling_distrib



    def get_samples(self, N):
        # return self.concept.sample(N)
        return self.sampling_distrib.sample(N)
