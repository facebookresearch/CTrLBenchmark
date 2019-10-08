# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

###
# From https://github.com/kuc2477/pytorch-ewc
# Credits to Ha Junsoo - [kuc2477](https://github.com/kuc2477)
###

from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F, init
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from src.modules.ll_model import LifelongLearningModel


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=(collate_fn or default_collate),
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )


def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)


class EWCLLModel(LifelongLearningModel):
    def __init__(self, input_dim, n_hidden, hidden_size, lamda, fisher_estimate_samples):
        super(EWCLLModel, self).__init__()
        self.model = None

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size

        self.lamda = lamda
        self.fisher_estimate_samples = fisher_estimate_samples

        self.n_model = 0

    def _new_model(self, n_classes, **kwargs):
        if self.model is None:
            self.model = MLP(self.input_dim, n_classes[0], self.hidden_size, self.n_hidden, lamda=self.lamda)
            old_loss = loss
            # new_loss = lambda y_hat, y: old_loss(y_hat, y) + self.model.ewc_loss()
            new_loss = lambda y_hat, y: F.cross_entropy(y_hat, y.squeeze()) + self.model.ewc_loss()
            self.model.loss = new_loss
            self.model.n_out = 1
            # xavier_initialize(self.model)

        self.n_model += 1
        return self.model

    def finish_task(self, dataset):
        fisher = self.model.estimate_fisher(dataset, self.fisher_estimate_samples)
        self.model.consolidate(fisher)


class MLP(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=400,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2,
                 lamda=40):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size
        self.lamda = lamda

        # Layers.
        self.layers = nn.ModuleList([
            # input
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
            # output
            nn.Linear(self.hidden_size, self.output_size)
        ])

    @property
    def name(self):
        return (
            'MLP'
            '-lambda{lamda}'
            '-in{input_size}-out{output_size}'
            '-h{hidden_size}x{hidden_layer_num}'
            '-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}'
        ).format(
            lamda=self.lamda,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return reduce(lambda x, l: l(x), self.layers, x)

    def estimate_fisher(self, dataset, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        data_loader = get_data_loader(dataset, batch_size)
        loglikelihoods = []
        for x, y in data_loader:
            x = x.view(x.size(0), -1)
            x = Variable(x).to(self.device())
            y = Variable(y).to(self.device())
            loglikelihoods.append(
                F.log_softmax(self(x), dim=1)[range(x.size(0)), y.squeeze()]
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(
            l, self.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return 0

    def device(self):
        return next(self.parameters()).device
