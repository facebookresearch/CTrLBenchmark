# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from numbers import Number

import numpy as np
import sklearn
import torch
from ctrl.concepts.concept import ComposedConcept, AtomicConcept
from ctrl.concepts.concept_tree import ConceptTree
from torch.distributions import uniform
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SyntheticGaussianTree(ConceptTree):
    def __init__(self, intrinsic_dims, low_bound, high_bound, scale_fact, cov_scale, mean, *args, **kwargs):
        self.intrinsic_dims = intrinsic_dims
        self.mean = mean
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.scale_fact = scale_fact
        self.cov_scale = cov_scale
        super().__init__(*args, **kwargs)

    def build_tree(self):
        concepts = self._build_tree(self.n_levels, self.n_children, self.intrinsic_dims, self.low_bound, self.high_bound, self.scale_fact,
                         self.cov_scale, self.mean, self.name)

        concept = ComposedConcept(concepts, cluster_mean=self.mean, id=self.name)
        self.tree.add_node(concept)

        for c in concepts:
            self.tree.add_edge(concept, c)
        return concept

    def _build_tree(self, n_levels, n_children, n_dims, low, high, scale_fact, cov_scale, mean, parent_name):
        if isinstance(low, Number):
            low = torch.ones(n_dims) * low
        if isinstance(high, Number):
            high = torch.ones(n_dims) * high

        cluster_means = uniform.Uniform(low + mean, high + mean).sample((n_children[0],))
        pref = '\t' * (self.n_levels - n_levels)
        concepts = []
        for i, cluster_mean in enumerate(cluster_means):
            logger.debug(pref + 'New cluster centered on {}'.format(cluster_mean))
            concept_name = '{}{}'.format(parent_name, i)
            if n_levels > 1:
                lower_concepts = self._build_tree(n_levels - 1, n_children[1:], n_dims, low=low * scale_fact,
                                                  high=high * scale_fact, scale_fact=scale_fact, cov_scale=cov_scale,
                                                  mean=cluster_mean, parent_name=concept_name)
                concept = ComposedConcept(lower_concepts, cluster_mean=cluster_mean, id=concept_name)
                self.tree.add_node(concept)

                for c in lower_concepts:
                    self.tree.add_edge(concept, c)

                self.all_nodes.add(concept)

            else:
                concept = AtomicConcept(cluster_mean, torch.eye(n_dims) * cov_scale, concept_name)
                self.tree.add_node(concept)
                self.all_nodes.add(concept)
                self.leaf_nodes.add(concept)
            concepts.append(concept)
        return concepts

    def plot_concepts(self, viz):
        if self.intrinsic_dims > 3:
            return
        all_leafs = sorted([(elt.descriptor, elt.mean) for elt in self.leaf_concepts])
        concepts_desc, concept_means = zip(*all_leafs)
        viz.scatter(torch.stack(concept_means), Y=range(1, len(concepts_desc) + 1),
                    opts={'title': 'All concepts',
                          'markersize': 3,
                          'legend': list(concepts_desc)})

    def init_attributes(self, n_attrs, viz=None, plot=True):
        all_concepts = list(self.leaf_concepts)
        pivot = int(len(all_concepts) / 2)

        plot = plot and all_concepts[0].mean.size(0) == 2 and viz

        for i in tqdm(range(n_attrs), desc='Init attributes'):
            self.rnd.shuffle(all_concepts)

            data = torch.stack([c.mean for c in all_concepts])
            labels = torch.ones(len(all_concepts), 1)
            labels[pivot:] -= 1

            if plot:
                win = viz.scatter(data, labels + 1,
                                  opts={'xtickmin': -1.5, 'xtickmax': 1.5,
                                        'ytickmin': -1.5, 'ytickmax': 1.5,
                                        'markersize': 3})
            tries = 0
            found_lin_sep = False
            while not found_lin_sep:
                tries += 1
                if tries % 100 == 0:
                    logger.warning('Tried to find sep {} times for attr {}'
                                   .format(tries, len(self.attributes)))
                perceptron = torch.nn.Linear(data.size(1), 1)
                optimizer = torch.optim.SGD(perceptron.parameters(), lr=1)
                loss = torch.nn.BCELoss()

                for j in range(100):
                    optimizer.zero_grad()
                    y_hat = perceptron(data).sigmoid()
                    loss(y_hat, labels).backward()
                    optimizer.step()

                if plot:
                    x = torch.tensor([-1, 1]).view(2, 1).float()

                    w = perceptron.weight.squeeze()
                    a = -w[0] / w[1]
                    b = -perceptron.bias / w[1]
                    y = a * x + b

                    viz.scatter(X=x.squeeze(), Y=y.squeeze(),
                                opts=dict(mode='lines', markers=False), name='sep',
                                update='new', win=win)

                new_labels = (perceptron(data).sigmoid().squeeze() > 0.5).long()

                svm = sklearn.svm.LinearSVC(C=1e9, max_iter=10000).fit(data,
                                                                       new_labels)

                if plot:
                    viz.scatter(data, new_labels + 1, win=win,
                                opts={'title': 'attribute {}'.format(i),
                                      'xtickmin': -1.5, 'xtickmax': 1.5,
                                      'ytickmin': -1.5, 'ytickmax': 1.5,
                                      'markersize': 3,})
                    viz.scatter(X=x.squeeze(), Y=y.squeeze(), name='perceptron',
                                opts=dict(mode='lines', markers=False),
                                update='new', win=win)

                    w = svm.coef_[0]

                    a = -w[0] / w[1]
                    b = -svm.intercept_[0] / w[1]
                    y = a * x + b

                    viz.scatter(X=x.squeeze(), Y=y.squeeze(),
                                opts=dict(mode='lines', markers=False), name='svm',
                                update='new', win=win)

                found_lin_sep = all(svm.predict(data) == new_labels.numpy())

            for concept, attr_val in zip(all_concepts, new_labels):
                concept.attrs.append(attr_val.item())
            self.attributes.append((svm.coef_, svm.intercept_))

        if n_attrs > 0:
            self.attribute_similarities = torch.empty(n_attrs, n_attrs)
            for i, (w1, _) in enumerate(self.attributes):
                for j, (w2, _) in enumerate(self.attributes):
                    self.attribute_similarities[i, j] = (
                            w1 @ w2.transpose() / (
                                np.linalg.norm(w1) * np.linalg.norm(w2))).item()
