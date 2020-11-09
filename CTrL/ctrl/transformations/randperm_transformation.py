# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from ctrl.transformations.TransformationTree import TransformationTree
from ctrl.transformations.utils import BatchedTransformation
from torchvision import transforms


class RandomPermutationsTransformation(TransformationTree):
    def __init__(self, n_permutations, x_off, y_off, width, height, flatten,
                 *args, **kwargs):
        self.n_permutations = n_permutations
        self.x_off = x_off
        self.y_off = y_off
        self.width = width
        self.height = height
        self.flatten = flatten

        super(RandomPermutationsTransformation, self).__init__(*args, **kwargs)

    def build_tree(self):
        self.tree.add_node(self._node_index[self.name], name=self.name)
        for i in range(self.n_permutations):
            node_name = 'permutation_{}'.format(i)
            self.leaf_nodes.add(self._node_index[node_name])

            perm = RandomPermutation(self.x_off, self.y_off, self.width,
                                     self.height, self.flatten)
            trans = transforms.Compose(
                [BatchedTransformation(transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor()])),
                    perm])
            # f = BatchedTransformation(trans)
            self.tree.add_node(self._node_index[node_name], name=node_name)
            self.tree.add_edge(self._node_index[self.name],
                               self._node_index[node_name], f=trans)
        self.depth = 1
        return self._node_index[self.name]


class RandomPermutation(object):
    """
    Applies a constant random permutation to the images.
    """

    def __init__(self, x_off=0, y_off=0, width=None, height=None,
                 flatten=False):
        self.x_off = x_off
        self.y_off = y_off
        self.width = width
        self.height = height
        self.x_max = x_off + width
        self.y_max = y_off + height
        self.kernel = torch.randperm(width * height)
        self.flatten = flatten

    def __call__(self, input):
        return rand_perm_(input.clone(), self.x_off, self.y_off, self.x_max,
                          self.y_max, self.kernel, self.flatten)


def rand_perm_(img, x, y, x_max, y_max, kernel, flatten):
    """
    Applies INPLACE the random permutation defined in `kernel` to the image
    `img` on the zone defined by `x`, `y`, `x_max`, `y_max`
    :param img: Input image of dimension (B*C*W*H)
    :param x: offset on x axis
    :param y: offset on y axis
    :param x_max: end of the zone to permute on the x axis
    :param y_max: end of the zone to permute on the y axis
    :param kernel: LongTensor of dim 1 containing one value for each point in
    the zone to permute
    :return: the permuted image.
    """
    assert img.dim() == 4
    if img.size(1) != 1:
        raise NotImplementedError('Not Implemented for multi-channel images')
    zone = img[:, :, x:x_max, y:y_max].contiguous()
    img[:, :, x:x_max, y:y_max] = zone.view(zone.size(0), -1)\
        .index_select(1, kernel).view(zone.size())
    return img.view(img.size(0), -1) if flatten else img
