# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from src.modules.base import get_block_model
from src.modules.ll_model import LifelongLearningModel
from src.modules.utils import MultiHead


class ExperienceReplayLLModel(LifelongLearningModel):
    def __init__(self, input_dim, n_hidden, hidden_size, mem_size_per_class, base_model=get_block_model):
        super(ExperienceReplayLLModel, self).__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size

        self.n_classes = None
        self.mem = Memory(mem_size_per_class, self.input_dim)

        base_model = nn.Sequential(*base_model(self.input_dim, self.n_hidden, self.hidden_size))
        self.model = ERLearner(base_model, self.mem)

    def _new_model(self, n_classes, **kwargs):
        if self.n_classes is None:
            self.n_classes = n_classes
        assert n_classes == self.n_classes, 'SingleHead model can only be used with the same number of classes'
        if len(n_classes) != 1:
            raise NotImplementedError('ER not yet implemented for attributes')

        new_head = MultiHead(self.hidden_size, n_classes)
        self.model.add_head(new_head, n_classes)

        self.mem.add_classes(n_classes[0])
        return self.model


class ERLearner(nn.Module):
    def __init__(self, base_model, memory):
        super(ERLearner, self).__init__()
        self.base = base_model
        self.memory = memory

        self.heads = nn.ModuleList()
        self.n_classes = []

    @property
    def n_out(self):
        # heads = list(self.heads.values())
        assert all(h.n_out == self.heads[0].n_out for h in self.heads)
        return self.heads[0].n_out

    def loss(self, y_pred, y):
        return loss(y_pred, y)

    def forward(self, xs):
        """


        :param xs: Tuple containing the input features and the task id for each example.
        The format is (Bxn_features, B) format where le first element is the input data and the second contain the id
        of the task corresponding with each feature vector. The task id will be used to select the head corresponding to
        the task.
        :return:

        """
        x, t_idx = xs

        features = self.base(x)
        tasks_in_batch = sorted(t_idx.unique().tolist())
        n_classes = [self.n_classes[idx] for idx in tasks_in_batch]
        sizes = torch.tensor(n_classes).sum(0)
        y_hat = [torch.ones(x.size(0), size, device=features.device) * -float('inf') for size in sizes]

        #contains the offset for each attribute, this offset will be incremented after each task
        task_cols = [0] * len(n_classes[0])
        for i, t_id in enumerate(tasks_in_batch):
            mask = t_idx == t_id
            res = self.heads[t_id](features[mask])
            for j, (attr_y_hat, attr_res) in enumerate(zip(y_hat, res)):
                attr_y_hat[mask, task_cols[j]:task_cols[j]+attr_res.size(1)] = attr_res
                task_cols[j] += attr_res.size(1)
                assert n_classes[i][j] == attr_res.size(1)

        return y_hat

    def add_head(self, new_head, n_classes):
        self.heads.append(new_head)
        self.n_classes.append(n_classes)

    def prepare_batch_wrapper(self, func, task_id):
        def prepare_batch(batch, *args, **kwargs):
            x, y = batch
            if self.training:
                batch = self.memory.extend_batch(x, y, task_id, self.n_classes)
                self.memory.update_memory(x, y, task_id)
            else:
                batch = (x, torch.ones(x.size(0)).long() * task_id), y
            return func(batch, *args, **kwargs)
        return prepare_batch


class Memory(object):
    def __init__(self, mem_size_per_class, input_dim, total_n_classes=0):
        super(Memory, self).__init__()
        self.mem_size_per_class = mem_size_per_class
        self.total_n_classes = total_n_classes

        self.input_dim = input_dim
        self.output_dim = 1
        self.mem_size = self.mem_size_per_class * self.total_n_classes
        self.mem = (torch.empty(0, self.input_dim), torch.empty(0, self.output_dim).long(), torch.empty(0).long())

        self.n_examples_seen = 0

    @property
    def n_items(self):
        return self.mem[0].size(0)

    def add_classes(self, n_classes):
        self.total_n_classes += n_classes
        self.mem_size = self.mem_size_per_class * self.total_n_classes

    def extend_batch(self, x, y, task_id, n_classes):
        """
        return a bunch of (xs, ys, tid)
        :param x:
        :param y:
        :param task_id:
        :return:
        """

        cur_task_ids = torch.ones(x.size(0)).long() * task_id

        batch_size = x.size(0)
        n_mem_samples = min(self.n_items, batch_size)
        selected_idx = np.random.randint(self.n_items, size=n_mem_samples)

        samples = [mem[selected_idx] for mem in self.mem]

        ext_x = torch.cat([x, samples[0]])
        ext_y = torch.cat([y, samples[1]])
        task_ids = torch.cat([cur_task_ids, samples[2]])

        tasks_in_batch = sorted(task_ids.unique().tolist())
        n_classes = [n_classes[idx] for idx in tasks_in_batch]
        attr_offset = [0]*len(n_classes[0])
        for t_id, task_classes in zip(tasks_in_batch, n_classes):
            for i, attr_classes in enumerate(task_classes):
                ext_y[task_ids==t_id, i] += attr_offset[i]
                attr_offset[i] += attr_classes

        return (ext_x, task_ids), ext_y

    def update_memory(self, xs, ys, task_id):
        batch_size = xs.size(0)
        n_new_items = min(self.mem_size - self.n_items, batch_size)
        if n_new_items > 0:
            new_size = self.n_items + n_new_items
            self.mem[0].resize_(new_size, self.input_dim)
            self.mem[1].resize_(new_size, self.output_dim)
            self.mem[2].resize_(new_size)

            self.mem[0][-n_new_items:] = xs[:n_new_items]
            self.mem[1][-n_new_items:] = ys[:n_new_items]
            self.mem[2][-n_new_items:] = task_id
            self.n_examples_seen += n_new_items

        if n_new_items < batch_size:
            for x, y in zip(xs[n_new_items:].split(1), ys[n_new_items:].split(1)):
                i = np.random.randint(self.n_examples_seen)
                if i < self.mem_size:
                    self.mem[0][i] = x.squeeze(0)
                    self.mem[1][i] = y.squeeze(0)
                    self.mem[2][i] = task_id
                self.n_examples_seen += 1
