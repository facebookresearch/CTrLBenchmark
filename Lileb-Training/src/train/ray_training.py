# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy
from ray import tune
from torch.utils.data import DataLoader, TensorDataset, Sampler, \
    SubsetRandomSampler

from src.train.training import get_attr_transform, train, \
    get_classic_dataloaders
from src.train.utils import StopAfterIterations, create_supervised_trainer, \
    create_supervised_evaluator, get_multitask_dataset, mytimeit, set_dropout

logger = logging.getLogger(__name__)


class BaseTrainLLModel(tune.Trainable):
    def _setup(self, config):
        training_params = config['tp']
        hyper_params = config['hyper-params']
        self.name = training_params['name']
        self.loss_fn = training_params['loss_fn']
        self.device = training_params['device']
        self.n_it_max = training_params['n_it_max']
        self.split_names = training_params['split_names']

        self.model = torch.load(training_params['model_path'])
        self.datasets = self._load_datasets(training_params['data_path'],
                                            training_params['loss_fn'],
                                            training_params['past_tasks'])
        self.batch_sizes = training_params['batch_sizes']
        data_loaders = self._get_dataloaders(self.datasets, self.batch_sizes)
        self.train_loader, self.eval_loaders = data_loaders

        self.named_eval_loaders = OrderedDict(
            zip(self.split_names, self.eval_loaders))

        training_params['optim_func'].func.keywords['lr'] = hyper_params['lr']
        training_params['optim_func'].func.keywords['weight_decay'] = \
        hyper_params['wd']
        if 'dropout' in hyper_params:
            set_dropout(self.model, hyper_params['dropout'])

        # optim_func.keywords['momentum'] = config['momentum']
        self.optim = training_params['optim_func'](self.model.parameters())

        self.log_interval = training_params.get('log_interval', 30)
        self.log_steps = training_params['log_steps'].copy()
        if self.log_steps is None:
            self.log_steps = []
        if training_params['log_epoch']:
            self.log_steps.append(len(self.train_loader))
        self.n_iterations = 0
        self.n_epochs = 0
        self.n_steps = 0

        # For early stopping
        self.patience = training_params['patience']
        self.counter = 0
        self.best_score = None

        self.best_loss = float('inf')

        self.trainer = \
            create_supervised_trainer(self.model, self.optim,
                                      training_params['loss_fn'],
                                      device=self.device,
                                      output_transform=
                                      lambda x, y, y_pred, loss: (
                                          y_pred, y))
        self.trainer._logger.setLevel(logging.WARNING)

        l = Loss(lambda y_pred, y: self.loss_fn(y_pred, y).mean())
        l.attach(self.trainer, 'train_loss')
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                       l.completed, 'train_loss')
        StopAfterIterations(self.log_steps).attach(self.trainer)

        self.eval_metrics = {'nll': Loss(lambda y_pred, y:
                                         self.loss_fn(y_pred, y).mean())}
        for i in range(self.model.n_out):
            self.eval_metrics['accuracy_{}'.format(i)] = \
                Accuracy(output_transform=get_attr_transform(i))

        self.evaluator = \
            create_supervised_evaluator(self.model,
                                        metrics=self.eval_metrics,
                                        device=self.device)

        self.all_accuracies = defaultdict(dict)

    @staticmethod
    @abc.abstractmethod
    def _load_datasets(cur_data_path, cur_loss_fn, past_tasks):
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def _get_dataloaders(datasets, batch_sizes):
        return get_classic_dataloaders(datasets, batch_sizes)

    def log_perf(self):
        iteration = self.n_iterations
        val_dl = self.named_eval_loaders['Val']
        self.evaluator.run(val_dl)
        ray_metrics = {'mean_loss': self.evaluator.state.metrics['nll'],
                       'training_iteration': iteration}

        # we need to add dummy metrics because ray's logger will only report
        # metrics logged during the first step.
        splits = self.named_eval_loaders.keys()
        dummy_metrics = {'{} {}'.format(s, k): float('nan') for k in
                         self.eval_metrics.keys() for s in splits}
        ray_metrics.update(**dummy_metrics)
        if self.log_steps \
                and (iteration in self.log_steps
                     or iteration % self.log_steps[-1] == 0):

            ray_metrics.update(training_iteration=iteration)
            for d_loader, name in zip(self.eval_loaders, splits):
                split_metrics = self.log_results(self.evaluator, d_loader,
                                                 iteration, name,
                                                 self.all_accuracies[name])
                ray_metrics.update(**split_metrics)
        return ray_metrics

    def log_results(self, evaluator, data_loader, iteration, split_name,
                    all_accs):
        evaluator.run(data_loader)
        metrics = evaluator.state.metrics
        all_accs[iteration] = metrics['accuracy_0']

        log_metrics = {}

        for metric_name, metric_val in metrics.items():
            log_name = '{} {}'.format(split_name, metric_name)
            log_metrics[log_name] = metric_val

        if self.trainer.state and self.trainer.state.epoch % self.log_interval == 0:
            logger.info(
                '{} Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.2f}'
                    .format(split_name, self.trainer.state.epoch,
                            metrics['accuracy_0'], metrics['nll']))
        return log_metrics

    def _train(self):
        log_idx = min(self.n_steps, len(self.log_steps) - 1)
        last_log = 0 if log_idx == 0 else self.log_steps[log_idx - 1]
        if self.n_steps < len(self.log_steps):
            iter_this_step = self.log_steps[log_idx] - last_log
        else:
            iter_this_step = self.log_steps[-1]

        epochs_this_step = iter_this_step / len(self.train_loader)
        self.trainer.run(self.train_loader,
                         max_epochs=int(epochs_this_step) + 1)
        assert self.trainer.state.iteration == iter_this_step
        self.n_iterations += iter_this_step
        self.n_epochs += epochs_this_step
        self.n_steps += 1

        res = self.log_perf()
        res['training_epoch'] = self.n_epochs
        if 'mean_loss' in res and res['mean_loss'] < self.best_loss:
            self.best_loss = res['mean_loss']
            res.update(should_checkpoint=True)
            self.counter = 0
        else:
            self.counter += iter_this_step
            if 0 < self.patience <= self.counter:
                logger.info('#####')
                logger.info('# Early stopping Run')
                logger.info('#####')
                res.update(done=True)

        return res

    def _save(self, checkpoint_dir):
        assert self.name in checkpoint_dir
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class BaseOneShotTrainable(BaseTrainLLModel):
    def _train(self):
        t, accs, self.best_state_dict = train(self.model,
                                              train_loader=self.train_loader,
                                              eval_loaders=self.eval_loaders,
                                              optimizer=self.optim,
                                              loss_fn=self.loss_fn,
                                              n_it_max=self.n_it_max,
                                              patience=self.patience,
                                              split_names=self.split_names,
                                              device=self.device,
                                              name=self.name,
                                              log_steps=self.log_steps,
                                              log_epoch=False)
        keys = []
        res = {}
        for k, v in accs.items():
            if isinstance(v, dict):
                assert list(v.keys()) == list(
                    accs['training_iterations'].values())
                res[k] = list(v.values())
                keys.append(k)
            else:
                res[k] = v

        return {'unroll_columns': keys,
                'should_checkpoint': True,
                'done': True,
                **res}

    def _save(self, checkpoint_dir):
        assert self.name in checkpoint_dir
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.best_state_dict, checkpoint_path)
        return checkpoint_path


class TrainLLModel(BaseTrainLLModel):
    @staticmethod
    def _load_datasets(cur_data_path, cur_loss_fn, past_tasks):
        datasets = []
        for split_path in cur_data_path:
            x, y = torch.load(split_path)
            datasets.append(TensorDataset(x, y))
        return datasets


class OSTrainLLModel(BaseOneShotTrainable):
    @staticmethod
    def _load_datasets(cur_data_path, cur_loss_fn, past_tasks):
        datasets = []
        for split_path in cur_data_path:
            x, y = torch.load(split_path)
            datasets.append(TensorDataset(x, y))
        return datasets


class MultiTaskTrainable(abc.ABC):
    @staticmethod
    def _load_datasets(cur_data_path, cur_loss_fn, past_tasks):
        all_tasks = past_tasks + [[cur_data_path, cur_loss_fn]]
        return get_multitask_dataset(all_tasks)

    @staticmethod
    def _get_dataloaders(datasets, batch_sizes):
        train_batch_sampler = MTBatchSampler(datasets[0], batch_sizes[0], False, False)
        train_loader = DataLoader(datasets[0],
                                  batch_sampler=train_batch_sampler)

        eval_batch_samplers = [MTBatchSampler(ds, batch_sizes[1], False, False)
                               for ds in datasets]

        eval_loaders = [DataLoader(ds, batch_sampler=bs)
                        for ds, bs in zip(datasets, eval_batch_samplers)]
        return train_loader, eval_loaders


class SimpleMultiTaskTrainable(MultiTaskTrainable, BaseTrainLLModel):
    pass


class OSMultiTaskTrainable(MultiTaskTrainable, BaseOneShotTrainable):
    pass


class MTBatchSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, drop_last, patial_batches_ok):
        super(MTBatchSampler, self).__init__(concat_dataset)
        self.task_idx_bins = []
        offset = 0
        for ds in concat_dataset.datasets:
            self.task_idx_bins.append(np.arange(offset, offset + len(ds)))
            offset += len(ds)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.samplers = [SubsetRandomSampler(idx) for idx in self.task_idx_bins]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.partial_batches_ok = patial_batches_ok

    def __iter__(self):
        iters = [iter(sub_samp) for sub_samp in self.samplers]
        batch = []
        while iters:
            sampler_idx = torch.randint(len(iters), (1,)).item()
            sampler = iters[sampler_idx]
            for idx in sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    break
            if len(batch) == self.batch_size:
                # We have a full batch
                yield batch
                batch = []
            elif len(batch) > 0 and not self.drop_last:
                # We have a partial batch, meaning that the selected sampler is
                # finished.
                iters.remove(sampler)
                if self.partial_batches_ok or not iters:
                    yield batch
                    batch = []
            else:
                # We don't have a batch. Either the selected iterator was
                # empty or we have a partial batch but we are dropping it.
                iters.remove(sampler)

    def __len__(self):
        if self.drop_last:
            return sum(
                len(sampler) // self.batch_size for sampler in self.samplers)
        else:
            return sum(
                (len(sampler) + self.batch_size - 1) // self.batch_size for
                sampler in self.samplers)
