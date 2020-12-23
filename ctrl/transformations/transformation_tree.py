# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from abc import ABC
from collections import defaultdict
from numbers import Number

import networkx as nx
from torch import nn

from ctrl.commons.tree import Tree
from ctrl.transformations.transformation import Transformation
from ctrl.transformations.transformation_pool import TransformationPool

logger = logging.getLogger(__name__)


class TransformationTree(TransformationPool, Tree, ABC):
    def __init__(self, *args, **kwargs):
        self._node_index = defaultdict()
        self._node_index.default_factory = self._node_index.__len__
        super().__init__(*args, **kwargs)

    def get_transformation(self, exclude_trans=None, allowed_trans=None):
        if exclude_trans is None:
            exclude_trans = []
        exclude_nodes = [trans.path[-1] for trans in exclude_trans]
        if allowed_trans is not None:
            allowed_nodes = set(trans.path[-1] for trans in allowed_trans)
        else:
            allowed_nodes = None
        node = self.get_compatible_nodes(exclude_nodes=exclude_nodes,
                                         force_nodes=allowed_nodes,
                                         leaf_only=True)
        all_paths = list(nx.all_simple_paths(self.tree, self.root_node, node))

        selected_path = random.choice(all_paths)
        path_descr = self.get_path_descr(selected_path)
        return Transformation(self, selected_path, path_descr)

    def transformations_sim(self, t1, t2):
        return self.wu_palmer(t1.path[-1], t2.path[-1])

    def edit_transformation(self, transformation, min_dist, max_dist):
        dist = random.randint(min_dist, max_dist)
        old_path = transformation.path.copy()
        old_path = old_path[:-dist]
        new_candidates = list(nx.all_simple_paths(self.tree, old_path[-1],
                                                  self.out_nodes))
        selected_path = random.choice(new_candidates)
        new_path = old_path + selected_path[1:]
        return Transformation(self, new_path)

    def get_path_descr(self, path):
        return '->'.join([self.tree.nodes[itm]['name'] for itm in path])


class RandomNNTransformationTree(TransformationTree):
    def __init__(self, depth, degree, x_dim, z_dim, non_lin, *args, **kwargs):
        self.depth = depth
        self.n_children = self._format_property(degree)
        self.hidden_sizes = self._format_property(x_dim)
        self.z_dim = z_dim

        if non_lin == 'relu':
            self.non_linearity = nn.ReLU
        elif non_lin == 'tanh':
            self.non_linearity = nn.Tanh

        super().__init__(*args, **kwargs)

    def _format_property(self, prop):
        if isinstance(prop, Number):
            prop = [prop]
        if len(prop) == 1:
            prop = prop * self.depth
        assert len(prop) == self.depth
        return prop

    def build_tree(self):
        self._build_tree(self.depth-1, self.n_children, self.name, self.z_dim, self.hidden_sizes)
        self.tree.add_node(self._node_index[self.name], name=self.name)
        return self._node_index[self.name]

    def _build_tree(self, depth, n_children, parent_node, parent_dim, hidden_dims):
        for i in range(n_children[0]):
            module = nn.Sequential(
                nn.Linear(parent_dim, hidden_dims[0]),
                self.non_linearity())
            node_name = '{}{}'.format(parent_node, i)
            self.tree.add_node(self._node_index[node_name], name=node_name)
            self.tree.add_edge(self._node_index[parent_node], self._node_index[node_name], f=module)
            if depth > 0:
                self._build_tree(depth - 1, n_children[1:], node_name, hidden_dims[0], hidden_dims[1:])
            else:
                self.leaf_nodes.add(self._node_index[node_name])
