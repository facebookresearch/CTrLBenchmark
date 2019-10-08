# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number

import numpy as np
import torch
from torch.distributions import MultivariateNormal, uniform

from src.gmeval.distributions import Gaussian


class Concept(object):
    def __init__(self, mean, cov, id):
        super(Concept, self).__init__()
        self.intrinsic_dim = mean.size(0)
        # self.intrinsic_distrib = MultivariateNormal(mean, cov)
        self.intrinsic_distrib = Gaussian(dict(mean=mean, cov=cov))

        self.descriptor = id

    def get_samples(self, N):
        return self.intrinsic_distrib.sample(N)
        # return self.intrinsic_distrib.sample((N,))


class UniformConceptPool(object):
    def __init__(self, n_dims, n_concepts, low=-1, high=1):
        low = torch.ones(n_dims)*low
        high = torch.ones(n_dims)*high
        means = uniform.Uniform(low, high).sample((n_concepts,))
        self.concepts = [Concept(m, torch.eye(n_dims), i) for i, m in enumerate(means)]

    def get_concepts(self, N):
        choice = np.random.choice(self.concepts, N, replace=False)
        return choice


class HierarchicalConceptPool(object):
    def __init__(self, n_levels, n_children, n_dims, low=-1000, high=1000, mean=0):
        if isinstance(low, Number):
            low = torch.ones(n_dims) * low
        if isinstance(high, Number):
            high = torch.ones(n_dims) * high

        self.n_levels = n_levels
        self.cluster_means = uniform.Uniform(low+mean, high+mean).sample((n_children,))
        self.concepts = []
        for i, cluster_mean in enumerate(self.cluster_means):
            pref = '\t' * (3-n_levels)
            if n_levels > 0:
                print(pref + 'New cluster centered on {}'.format(cluster_mean))
                self.concepts.append(HierarchicalConceptPool(n_levels-1, n_children, n_dims, low=low/10, high=high/10, mean=cluster_mean))
            else:
                print(pref + 'New concept centered on {}'.format(cluster_mean))
                self.concepts.append(Concept(cluster_mean, np.eye(n_dims), i))
            # concept_means = MultivariateNormal(cluster_mean, torch.eye(n_dims)).sample((n_concepts_per_cluster,))
            # self.concepts_means.append(concept_means)
            # for j, m in enumerate(concept_means):
            #     self.concepts.append(Concept(m, torch.eye(n_dims), (i, i*n_concepts_per_cluster + j)))

    def get_concepts(self, N):
        all_concepts = self.get_leaf_concepts()
        return np.random.choice(all_concepts, N, replace=False)


    def get_leaf_concepts(self):
        if self.n_levels == 0:
            return self.concepts
        else:
            return sum([c.get_leaf_concepts() for c in self.concepts], [])
        # choice = np.random.choice(self.concepts, N, replace=False)
        # return choice
#
# class NaryConceptNode(object):
#     def __init__(self, n_levels, n_children, n_dims, low, high):
#

