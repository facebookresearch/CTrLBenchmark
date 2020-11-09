# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from ctrl.transformations.TransformationTree import TransformationTree
from ctrl.transformations.utils import BatchedTransformation
from torchvision.transforms import transforms


def load_or_convert_to_image(img):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
        img = F.to_pil_image(img)
    assert isinstance(img, Image.Image)
    return img


def crop_if_not_square(img, max_size=72):
    if min(img.size) > max_size:
        img = F.resize(img, max_size, Image.BILINEAR)
    if img.size[0] != img.size[1]:
        img = F.center_crop(img, min(img.size))
    return img


class IdentityTransformation(TransformationTree):
    def __init__(self, format_image, *args, **kwargs):
        self.format_image = format_image
        super(IdentityTransformation, self).__init__(*args, **kwargs)

    def build_tree(self):
        self.tree.add_node(self._node_index[self.name], name=self.name)
        node_name = 'Id'
        self.leaf_nodes.add(self._node_index[node_name])
        self.tree.add_node(self._node_index[node_name], name=node_name)
        if self.format_image:
            trans = transforms.Compose([
                load_or_convert_to_image,
                # transforms.ToPILImage(),
                crop_if_not_square,
                transforms.ToTensor()
            ])
            f = BatchedTransformation(trans)
        else:
            f = lambda x: x
        self.tree.add_edge(self._node_index[self.name],
                           self._node_index[node_name],
                           f=f)
        self.depth = 1
        return self._node_index[self.name]
