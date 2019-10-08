# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import torch.nn as nn
from ignite.metrics import Accuracy, Loss
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader

import visdom


def get_data_loaders(dataset, val_ratio):
    val_size = int(len(dataset) * val_ratio)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    return train_loader, val_loader


def train(model, dataset, target_train_acc):
    viz = visdom.Visdom()
    train_loader, val_loader = get_data_loaders(dataset, val_ratio=.2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    scheduler = StepLR(optimizer, step_size=200, gamma=1)

    loss = torch.nn.NLLLoss()
    wins = dict(acc=None, nll=None)

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': Accuracy(),
                                                'nll': Loss(loss)
                                            })
    #
    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(trainer):
    #     print('Epoch[{}] Loss: {:.2f}'.format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        nonlocal wins
        scheduler.step()

        evaluator.run(train_loader)
        metrics = evaluator.state.metrics

        if metrics['accuracy'] > target_train_acc:
            trainer.terminate()

        print('Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}'
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))
        wins['acc'] = viz.line(torch.tensor([metrics['accuracy']]),
                               X=torch.tensor([trainer.state.epoch]),
                               win=wins['acc'],
                               name='Train acc',
                               update='append' if wins['acc'] else None,
                               opts={'title': 'accuracies', 'showlegend': True})
        wins['nll'] = viz.line(torch.tensor([metrics['nll']]),
                               X=torch.tensor([trainer.state.epoch]),
                               win=wins['nll'],
                               name='Train loss',
                               update='append' if wins['nll'] else None,
                               opts={'title': 'nll', 'showlegend':True})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print('Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}'
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))
        wins['acc'] = viz.line(torch.tensor([metrics['accuracy']]),
                               X=torch.tensor([trainer.state.epoch]),
                               win=wins['acc'],
                               name='Val acc',
                               update='append' if wins['acc'] else None,)
        wins['nll'] = viz.line(torch.tensor([metrics['nll']]),
                               X=torch.tensor([trainer.state.epoch]),
                               win=wins['nll'],
                               name='Val loss',
                               update='append' if wins['nll'] else None)

    trainer.run(train_loader, max_epochs=300)
