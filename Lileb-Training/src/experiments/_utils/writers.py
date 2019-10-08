# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torch

from .plot import figure_to_image


def init_writer(_name, dirname=None, sacred_run=None, **config):
    if _name == 'sacred':
        return SacredWriter(dirname=dirname, sacred_run=sacred_run, **config)
    elif _name == 'tensorboard':
        from tensorboardX import SummaryWriter
        return SummaryWriter(log_dir=dirname, **config)
    elif _name == 'visdom':
        from tensorboardX.visdom_writer import VisdomWriter
        return VisdomWriter()
    raise NotImplementedError(_name)


def init_writers(*writer_kwargs, sacred_run=None, dirname=None):
    writers = []
    for kwargs in writer_kwargs:
        writer = init_writer(sacred_run=sacred_run, dirname=dirname, **kwargs)
        writers.append(writer)
    return WrappingWriters(writers)


class BaseWriter:
    def add_scalar(self, tag, scalar, step):
        raise NotImplementedError

    def add_image(self, tag, image, step):
        raise NotImplementedError

    def add_audio(self, tag, audio, step):
        raise NotImplementedError

    def add_text(self, tag, text, step):
        raise NotImplementedError

    def add_figure(self, **kwargs):
        pass

    def add_embedding(self, **kwargs):
        pass


class WrappingWriters(BaseWriter):
    def __init__(self, writers):
        self.writers = writers

    def add_scalar(self, tag, scalar, step):
        for w in self.writers:
            w.add_scalar(tag, scalar, step)

    def add_image(self, tag, image, step):
        for w in self.writers:
            w.add_image(tag, image, step)

    def add_audio(self, tag, audio, step):
        for w in self.writers:
            w.add_audio(tag, audio, step)

    def add_text(self, tag, text, step):
        for w in self.writers:
            w.add_text(tag, text, step)

    def add_figure(self, fig, text, step):
        for w in self.writers:
            w.add_figure(fig, text, step)

    def add_embedding(self, **kwargs):
        for w in self.writers:
            w.add_embedding(**kwargs)



class SacredWriter(BaseWriter):
    def __init__(self, sacred_run, dirname, save_info=False):
        self.sacred_run = sacred_run
        self.dirname = dirname
        self.save_info = save_info

    def add_scalar(self, tag, scalar, step):
        self.sacred_run.log_scalar(tag, scalar, step)

    def add_image(self, tag, image, step):

        if isinstance(image, torch.Tensor):
            image = image.mul(255).clamp(0, 255).byte().cpu().numpy()

        im = Image.fromarray(image.transpose(1, 2, 0))
        
        filename = os.path.join(self.dirname, f'{tag}_{step}.png')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        im.save(filename)
        self.sacred_run.add_artifact(filename, f'{tag}_{step}')

        if self.save_info:
            if 'image' not in self.sacred_run.info:
                self.sacred_run.info['image'] = {}
            if step not in self.sacred_run.info['image']:
                self.sacred_run.info['image'][step] = {}
            self.sacred_run.info['image'][step][tag] = filename

    def add_figure(self, tag, fig, step):
        self.add_image(tag, figure_to_image(fig), step)

