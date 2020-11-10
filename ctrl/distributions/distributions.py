# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Inspired from GMeval
Credits to Anton Bakhtin, Arthur Szlam & Marc'Aurelio Ranzato
"""

import numpy as np
import scipy
import scipy.stats
import torch
from torch.distributions import MultivariateNormal


class FuzzedExpansion(object):
    def __init__(self, new_dims, fuzz_scale):
        self.new_dims = new_dims
        if new_dims > 0:
            self._fuzz_distribution = MultivariateNormal(torch.zeros(new_dims), torch.eye(new_dims) * fuzz_scale)

    def __call__(self, X):
        if self.new_dims < 1:
            return X
        fuzz = self._fuzz_distribution.sample((X.size(0),))
        return torch.cat([X, fuzz], dim=1)


class IsometricTransform(object):
    def __init__(self, rotation_mat, bias=None):
        self.rotation_mat = rotation_mat
        if bias is None:
            bias = torch.zeros(rotation_mat.shape[0])
        self.bias = bias

    def __call__(self, X):
        return X @ self.rotation_mat + self.bias


class Mixture(object):
    def __init__(self, weights, components):
        assert len(weights) == len(components)
        self._weights = weights.copy() / weights.sum()
        self._components = components
        self._random = np.random.RandomState()
        # self._dim = self._components[0].dim

    def sample(self, N):
        counts = self._random.multinomial(N, self._weights)
        samples = [component.sample(k) if k else None
                   for k, component in zip(counts, self._components)]
        samples = [x for x in samples if x is not None]
        samples = np.concatenate(samples, axis=0)
        self._random.shuffle(samples)
        return samples

    def logprob_per_component(self, X):
        logpdfs = [component.logprob(X) for component in self._components]
        logpdfs = np.array(logpdfs).T
        logpdfs += np.expand_dims(np.log(self._weights), 0)
        return logpdfs

    def logprob(self, X):
        return scipy.special.logsumexp(self.logprob_per_component(X), axis=1)

    @property
    def num_components(self):
        return len(self._components)

    def restore_components(self, X):
        return np.argmax(self.logprob_per_component(X), axis=1)
