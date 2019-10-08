# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import networkx as nx
import numpy as np
import scipy

from src.concepts.concept import Concept
from src.gmeval.distributions import _FuzzedExpansion, IsometricTransform


class TansformationTree(object):
    ROOT = 'root'

    def __init__(self, intrinsic_dim, out_dim, n_rotations):
        super(TansformationTree, self).__init__()
        self.intrinsic_dim = intrinsic_dim
        self.out_dim = out_dim

        self.tree = nx.DiGraph()
        self.tree.add_edge(self.ROOT, 'expand')
        self.tree.add_node('expand',
                           func=_FuzzedExpansion,
                           args=dict(
                               # base_distribution=self.concept.intrinsic_distrib,
                               new_dims=self.out_dim - self.intrinsic_dim,
                               fuzz_scale=1)
                           )
        self.out_nodes = []
        for i in range(n_rotations):
            node_name = 'rotate_{}'.format(i)
            self.out_nodes.append(node_name)

            self.tree.add_edge('expand', node_name)
            rot = scipy.stats.ortho_group.rvs(self.out_dim)
            self.tree.add_node(node_name,
                               func=IsometricTransform,
                               args=dict(
                                   # base_distribution=self.expended_distrib,
                                   rotation=rot,
                                   bias=None)
                               )

    def get_tranformation(self):
        paths = list(nx.all_simple_paths(self.tree, self.ROOT, self.out_nodes))
        return paths[np.random.choice(len(paths))]

    def apply_transformation(self, source, trans_path):
        assert trans_path[0] == self.ROOT
        if isinstance(source, Concept):
            source = source.intrinsic_distrib

        for node_name in trans_path[1:]:
            node = self.tree.nodes()[node_name]
            args = node['args']
            args['base_distribution'] = source
            source = node['func'](args)

        return source


