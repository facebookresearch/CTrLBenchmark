# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import hashlib

import numpy as np
import torch
from torch.distributions import Multinomial


class Concept(abc.ABC):
    def __init__(self, id):
        self._samples = None
        self.descriptor = id

    def get_samples(self):
        return self._samples

    def get_attributes(self, attribute_ids):
        raise NotImplementedError('Attrs not supported anymore')

    @abc.abstractmethod
    def init_samples(self, n_per_split):
        raise NotImplementedError

    def __repr__(self):
        return self.descriptor

    def __hash__(self):
        h = hashlib.md5(self.descriptor.encode('utf-8')).hexdigest()
        return int(h, 16)


class ComposedConcept(Concept):
    def __init__(self, concepts, cluster_mean=None, weights=None, *args,
                 **kwargs):
        super(ComposedConcept, self).__init__(*args, **kwargs)
        self._concepts = concepts
        self.mean = cluster_mean
        if weights is None:
            weights = torch.ones(len(self.get_atomic_concepts()))
        self.weights = weights / weights.sum()

    def init_samples(self, N):
        assert self._samples is None
        all_samples = [c.get_samples() for c in self._concepts]
        if torch.is_tensor(all_samples[0][0]):
            cat_func = torch.cat
        else:
            cat_func = np.concatenate
        self._samples = [cat_func(split) for split in zip(*all_samples)]

    def _get_samples(self, n, attributes=None, split_id=None, rng=None):
        if not attributes:
            attributes = []
        if not rng:
            rng = np.random.default_rng()
        samples = []
        sample_attributes = []
        samples_per_concept = rng.multinomial(n, self.weights)
        for concept, n_samples in zip(self.get_atomic_concepts(),
                                      samples_per_concept):
            if n_samples == 0:
                continue
            c_samples, c_attrs = concept._get_samples(n_samples,
                                                      attributes=attributes,
                                                      split_id=split_id,
                                                      rng=rng)
            samples.append(c_samples)
            sample_attributes.append(c_attrs)

        if attributes:
            sample_attributes = torch.cat(sample_attributes)
        else:
            sample_attributes = torch.Tensor()

        if torch.is_tensor(samples[0]):
            cat_func = torch.cat
        else:
            cat_func = np.concatenate
        return cat_func(samples), sample_attributes

    def get_atomic_concepts(self):
        return sum([c.get_atomic_concepts() for c in self._concepts], [])
