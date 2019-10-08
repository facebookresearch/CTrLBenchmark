# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from collections import defaultdict

import ignite
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from torch.utils.data import DataLoader

from src.train.utils import create_supervised_evaluator, \
    create_supervised_trainer, StopAfterIterations, mytimeit

logger = logging.getLogger(__name__)


def get_attr_transform(attr_idx):
    def out_transform(out):
        y_pred, y = out

        if isinstance(y_pred, list):
            y_pred = y_pred[attr_idx]

        if y.dim() > 1:
            y = y[:, attr_idx]

        return y_pred, y

    return out_transform


def get_classic_dataloaders(datasets, batch_sizes):
    train_loader = DataLoader(datasets[0],
                              batch_size=batch_sizes[0],
                              shuffle=True)
    eval_loaders = [DataLoader(ds, batch_size=batch_sizes[1])
                    for ds in datasets]
    return train_loader, eval_loaders


def evaluate(model, dataset, task_id, batch_size, device, out_id=0):
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_classes = dataset.tensors[1][:, 0].unique().numel()
    out_transform = get_attr_transform(out_id)
    eval_metrics = {
        'accuracy': Accuracy(output_transform=out_transform),
        'confusion': ConfusionMatrix(num_classes=n_classes,
                                     output_transform=out_transform)
    }
    evaluator = create_supervised_evaluator(model, metrics=eval_metrics,
                                            device=device)
    evaluator._logger.setLevel(logging.WARNING)
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    return metrics['accuracy'], metrics['confusion']


def train(model, train_loader, eval_loaders, optimizer, loss_fn,
          n_it_max, patience, split_names, viz=None, device='cpu', name=None,
          log_steps=None, log_epoch=False, _run=None):
    """

    :param model:
    :param datasets: list containing the datasets corresponding to the different
            datasplits (train, val[, test])
    :param task_id:
    :param batch_sizes:
    :param optimizer:
    :param max_epoch:
    :param patience:
    :param log_interval:
    :param viz:
    :param device:
    :param name:
    :param log_steps:
    :param log_epoch:
    :param _run:
    :return:
    """
    if not log_steps and not log_epoch:
        logger.warning('/!\\ No logging during training /!\\')

    if log_steps is None:
        log_steps = []
    if log_epoch:
        log_steps.append(len(train_loader))

    trainer = create_supervised_trainer(model, optimizer, loss_fn,
                                        device=device)
    trainer._logger.setLevel(logging.WARNING)

    train_loss = RunningAverage(output_transform=lambda loss: loss,
                                epoch_bound=False)
    train_loss.attach(trainer, 'train_loss')

    StopAfterIterations([n_it_max]).attach(trainer)
    # epoch_pbar = ProgressBar(bar_format='{l_bar}{bar}{r_bar}', desc=name,
    #                          persist=True, disable=not (_run or viz))
    # epoch_pbar.attach(trainer, metric_names=['train_loss'])

    # training_pbar = ProgressBar(bar_format='{l_bar}{bar}{r_bar}', desc=name,
    #                             persist=True, disable=not (_run or viz))
    # training_pbar.attach(trainer, event_name=Events.EPOCH_COMPLETED,
    #                      closing_event_name=Events.COMPLETED)

    eval_metrics = {'nll': Loss(lambda y_pred, y: loss_fn(y_pred, y).mean())}
    for i in range(model.n_out):
        eval_metrics['accuracy_{}'.format(i)] = \
            Accuracy(output_transform=get_attr_transform(i))

    evaluator = create_supervised_evaluator(model, metrics=eval_metrics,
                                            device=device)
    all_metrics = defaultdict(dict)
    last_iteration = 0
    patience_counter = 0
    best_loss = float('inf')
    best_state_dict = None
    best_iter = -1

    def log_results(evaluator, data_loader, iteration, split_name):
        evaluator.run(data_loader)
        metrics = evaluator.state.metrics

        log_metrics = {}

        for metric_name, metric_val in metrics.items():
            log_name = '{} {}'.format(split_name, metric_name)
            if viz:
                viz.line([metric_val], X=[iteration], win=metric_name,
                         name=log_name,
                         update='append', opts={'title': metric_name,
                                                'showlegend': True,
                                                'width': 500})
            if _run:
                _run.log_scalar(log_name, metric_val, iteration)
            log_metrics[log_name] = metric_val
            all_metrics[log_name][iteration] = metric_val

        return log_metrics

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_event(trainer):
        iteration = trainer.state.iteration if trainer.state else 0
        nonlocal last_iteration, patience_counter, \
            best_state_dict, best_loss, best_iter

        if not log_steps or not \
                (iteration in log_steps or iteration % log_steps[-1] == 0):
            return
        all_metrics['training_epoch'][iteration] = iteration / len(train_loader)
        all_metrics['training_iterations'][iteration] = iteration
        if trainer.state and 'train_loss' in trainer.state.metrics:
            all_metrics['train_loss'][iteration] = trainer.state.metrics['train_loss']
        else:
            all_metrics['train_loss'][iteration] = float('nan')
        iter_this_step = iteration - last_iteration
        for d_loader, name in zip(eval_loaders, split_names):
            if name == 'Train':
                continue
            split_metrics = log_results(evaluator, d_loader, iteration, name)
            if name == 'Val' and patience > 0:
                if split_metrics['Val nll'] < best_loss:
                    best_loss = split_metrics['Val nll']
                    best_iter = iteration
                    patience_counter = 0
                    best_state_dict = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += iter_this_step
                    if patience_counter >= patience:
                        logger.info('#####')
                        logger.info('# Early stopping Run')
                        logger.info('#####')
                        trainer.terminate()
        last_iteration = iteration

    log_event(trainer)
    max_epoch = int(n_it_max / len(train_loader)) + 1
    trainer.run(train_loader, max_epochs=max_epoch)

    # all_metrics['mean_loss'] = all_metrics['Val nll']
    all_metrics['mean_loss'] = best_loss
    all_metrics['training_iteration'] = best_iter
    return trainer.state.iteration, all_metrics, best_state_dict
