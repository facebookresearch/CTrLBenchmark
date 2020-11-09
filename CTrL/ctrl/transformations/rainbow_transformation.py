# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import torch
from ctrl.transformations.TransformationTree import TransformationTree
from ctrl.transformations.utils import BatchedTransformation
from torchvision import transforms
from torchvision.transforms import RandomAffine

ROTATIONS = {
    '0': 0,
    # '90': 90,
    # '180': 180,
    # '270': 270
}

COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
OLD_BACKGOUND = [0]

SCALES = {
    'full': 1,
    # '3/4': 0.75,
    # 'half': 0.5,
    # '1/4': 0.25
}


def get_rotations():
    transformations = {}
    for name, angle in ROTATIONS.items():
        trans = transforms.Compose([
            transforms.ToPILImage(),
            RandomAffine(degrees=(angle, angle)),
            transforms.ToTensor()
        ])
        transformations[name] = BatchedTransformation(trans)
    return transformations


def get_scales():
    transformations = {}
    for name, scale in SCALES.items():
        trans = transforms.Compose([
            transforms.ToPILImage(),
            RandomAffine(degrees=0, scale=(scale, scale)),
            transforms.ToTensor()
        ])
        transformations[name] = BatchedTransformation(trans)
    return transformations


def change_background_color(images, old_background, new_background):
    """
    :param images: BCHW
    :return:
    """
    assert old_background == [0]
    if not torch.is_tensor(new_background):
        new_background = torch.tensor(new_background, dtype=images.dtype)
        if images.max() <= 1 and new_background.max() > 1:
            new_background /= 255

    if images.size(1) == 1 and len(new_background) == 3:
        images = images.expand(-1, 3, -1, -1)
    else:
        assert images.size(1) == len(new_background)
        # raise NotImplementedError(images.size(), new_background)

    images = images.clone()

    new_background = new_background.view(-1, 1, 1)

    bg_ratio = images.max() - images
    bg = bg_ratio * new_background
    imgs = images + bg
    # print(images[:, 0, :, :].std().item(),images[:, 1, :, :].std().item(),images[:, 2, :, :].std().item())
    # print(imgs[:, 0, :, :].std().item(), imgs[:, 1, :, :].std().item(), imgs[:, 2, :, :].std().item())
    return imgs


def get_colors():
    transformations = {}
    for color in COLORS:
        trans = partial(change_background_color, old_background=OLD_BACKGOUND,
                        new_background=color)
        transformations[str(color)] = trans
    return transformations


class RainbowTransformationTree(TransformationTree):
    def __init__(self, *args, **kwargs):
        self.n_rotations = None
        self.n_colors = None
        self.n_scaless = None
        super(RainbowTransformationTree, self).__init__(*args, **kwargs)

    def build_tree(self):

        self.tree.add_node(self._node_index[self.name], name=self.name)

        rotations = get_rotations()
        colors = get_colors()
        scales = get_scales()
        levels = [rotations, scales, colors]

        prev_nodes = [self.name]

        for domain in levels:
            prev_nodes = self._add_transfos(prev_nodes, domain)

        self.leaf_nodes.update([self._node_index[node] for node in prev_nodes])

        self.depth = len(levels)

        return self._node_index[self.name]

    def _add_transfos(self, parent_nodes, transfos):
        nodes = []
        for parent in parent_nodes:
            for name, transfo in transfos.items():
                node_name = '{}_{}'.format(parent, name)
                self.tree.add_node(self._node_index[node_name], name=node_name,
                                   last_transfo=name)

                self.tree.add_edge(self._node_index[parent],
                                   self._node_index[node_name],
                                   f=transfo, )
                nodes.append(node_name)
        return nodes

    def transformations_sim(self, t1, t2):
        """
        arccos((tr(R)−1)/2)
        :param t1:
        :param t2:
        :return:
        """
        t1_nodes = [t1.transfo_pool.tree.nodes()[id]['last_transfo'] for id in
                    t1.path[1:]]
        t2_nodes = [t2.transfo_pool.tree.nodes()[id]['last_transfo'] for id in
                    t2.path[1:]]
        n_eq = 0
        for op1, op2 in zip(t1_nodes, t2_nodes):
            if op1 == op2:
                n_eq += 1

        return n_eq / (len(t1_nodes))
