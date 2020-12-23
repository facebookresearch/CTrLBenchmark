# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from ctrl.transformations.transformation_tree import TransformationTree
from torch import nn
from tqdm import tqdm


class NoisyNNTransformationTree(TransformationTree):
    def __init__(self, noise_min, noise_max, x_dim, z_dim, n_canonic_transfo,
                 n_var_per_trans, *args, **kwargs):
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_canonic_transfo = n_canonic_transfo
        self.n_var_per_trans = n_var_per_trans
        self.depth = 2

        super().__init__(*args, **kwargs)
        self._inv_index = {v: k for k, v in self._node_index.items()}

    def build_tree(self):
        first_module = nn.Sequential(nn.Linear(self.z_dim, self.z_dim),
                                     nn.ReLU())
        # node_name = '{}{}'.format(self.name, 'front')
        node_name = 'front'
        self.tree.add_node(self._node_index[self.name], name=self.name)
        self.tree.add_node(self._node_index[node_name], name=node_name)
        self.tree.add_edge(self._node_index[self.name],
                           self._node_index[node_name], f=first_module)
        noise_source = torch.distributions.uniform.Uniform(self.noise_min,
                                                           self.noise_max)
        for i in tqdm(range(self.n_canonic_transfo), desc='Init noisy x',
                      disable=self.n_canonic_transfo < 30):
            lin = nn.Linear(self.z_dim, self.x_dim)
            for j in range(self.n_var_per_trans):
                mod = mod_lin(lin, noise_source)
                node_name = (i, j)
                self.tree.add_node(self._node_index[node_name], name=str(node_name))
                self.tree.add_edge(self._node_index['front'],
                                   self._node_index[node_name],
                                   f=nn.Sequential(mod, nn.ReLU()))
                self.leaf_nodes.add(self._node_index[node_name])
        return self._node_index[self.name]

    def transformations_sim(self, t1, t2):
        t1 = self._inv_index[t1.path[-1]]
        t2 = self._inv_index[t2.path[-1]]
        return 0 if t1[0] != t2[0] else 1


def mod_lin(lin, noise_source):
    noise = noise_source.sample(lin.weight.size())
    new_lin = nn.Linear(lin.in_features, lin.out_features)
    state_dict = lin.state_dict()
    state_dict['weight'] = state_dict['weight'] + noise
    new_lin.load_state_dict(state_dict)
    return new_lin
