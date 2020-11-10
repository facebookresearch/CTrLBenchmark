# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from ctrl.transformations.transformation_tree import TransformationTree
from torchvision import transforms
from torchvision.transforms import RandomRotation


class ImgRotationTransformationTree(TransformationTree):
    def __init__(self, n_rotations, max_degrees, *args, **kwargs):
        self.n_rotations = n_rotations
        self.max_degrees = max_degrees

        super(ImgRotationTransformationTree, self).__init__(*args, **kwargs)

    def build_tree(self):

        self.tree.add_node(self._node_index[self.name], name=self.name)

        for i in range(self.n_rotations):
            node_name = 'rotate_{}'.format(i)
            self.leaf_nodes.add(self._node_index[node_name])
            degrees = self.rnd.uniform(-self.max_degrees, self.max_degrees)

            trans = transforms.Compose([
                transforms.ToPILImage(),
                RandomRotation((degrees, degrees)),
                transforms.ToTensor()
            ])
            f = BatchedTransformation(trans)

            self.tree.add_node(self._node_index[node_name], name=node_name)
            self.tree.add_edge(self._node_index[self.name],
                               self._node_index[node_name],
                               f=f, degrees=degrees)

        self.depth = 1

        return self._node_index[self.name]

    def transformations_sim(self, t1, t2):
        """
        arccos((tr(R)−1)/2)
        :param t1:
        :param t2:
        :return:
        """
        theta_1 = self.tree.in_edges()[t1.path[-2:]]['degrees']
        theta_2 = self.tree.in_edges()[t2.path[-2:]]['degrees']

        theta = abs(theta_1 - theta_2) * np.pi/180
        min_angle = np.arccos(np.cos(theta))
        return 1 - min_angle / np.pi
