# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import torch
from ignite.engine import Events, Engine
from ignite.utils import convert_tensor
from torch import nn
from torch.utils.data import ConcatDataset, TensorDataset

logger = logging.getLogger(__name__)


class StopAfterIterations(object):
    def __init__(self, log_iterations):
        """
        Should be attached to an Ignite.Engine.
        Will stop the training immediately after the `n_iterations_max`
        iteration of the given engine.
        """
        self.log_iterations = log_iterations
        self.iteration = 0

    def __call__(self, engine):
        self.iteration += 1
        if self.iteration in self.log_iterations \
                or self.iteration % self.log_iterations[-1] == 0:
            engine.terminate()

    def attach(self, engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y, *z = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
            *z)

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    """
    From Ignite
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        # z is optional (e.g. task ids)
        x, y, *z = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(*(x, *z))
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def create_supervised_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # z is optional (e.g. task ids)
            x, y, *z = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(*(x, *z))
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def get_multitask_dataset(tasks, min_samples_per_class=10):
    all_ds = []
    normal_splits_len = [None, None, None]
    elems_per_task = []

    for i, (paths, loss) in enumerate(tasks):
        cur_ds = []
        for j, split_path in enumerate(paths):
            x, y = torch.load(split_path)
            if normal_splits_len[j] is None:
                normal_splits_len[j] = x.size(0)
                elems_per_task.append(int(x.size(0)/len(tasks)))
            assert x.size(0) == normal_splits_len[j], 'All split should have ' \
                                                      'the same size'
            if j == 0:
                # Keep all elements for the train set
                n_elems = x.size(0)
            else:
                n_elems = elems_per_task[j]
                n_classes = y.unique().size(0)
                if n_elems < min_samples_per_class * n_classes:
                    logger.warning('Not enough sample, will select {} elems'
                                     ' for {} classes when requiring at '
                                     'least {} samples per class'
                                     .format(n_elems, n_classes, min_samples_per_class))
                    n_elems = min_samples_per_class * n_classes
                selected_idx = torch.randint(x.size(0), (n_elems,))
                x = x[selected_idx]
                y = y[selected_idx]
            z = torch.ones(n_elems, dtype=torch.long)*i
            cur_ds.append(TensorDataset(x, y, z))
        all_ds.append(cur_ds)

    ds = [ConcatDataset(split_datasets) for split_datasets in zip(*all_ds)]
    return ds


def mytimeit(f):

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('\n##########\nfunc:{} took: {:2.4f} sec\n##########\n'.format(f.__name__, te - ts))
        return result

    return timed


def set_dropout(model, dropout_p):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
