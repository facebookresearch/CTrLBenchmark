# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import csv
import logging
import os
import pickle
import time
from functools import partial

import ray
import torch
from ray import tune
from ray.tune import Experiment
from ray.tune.logger import Logger, JsonLogger
from ray.tune.result import EXPR_PROGRESS_FILE
from ray.tune.util import flatten_dict
from torch import nn

from src.modules.base import get_block_model
from src.train.ray_training import TrainLLModel, OSTrainLLModel
from src.train.training import train, get_classic_dataloaders
from src.utils.misc import get_env_url, rename_class
from src.utils.plotting import plot_res_dataframe

logger = logging.getLogger(__name__)


class LifelongLearningModel(nn.Module, abc.ABC):
    def __init__(self, n_hidden, n_convs, hidden_size, dropout_p, grid_params,
                 base_model=get_block_model, *args, **kwargs):
        super(LifelongLearningModel, self).__init__(*args, **kwargs)
        self.models = nn.ModuleList([])
        self.dropout_p = dropout_p
        self.grid_params = grid_params

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * n_hidden
        self.hidden_size = hidden_size
        self.n_convs = n_convs

        self.base_model_func = partial(base_model, dropout_p=dropout_p, n_convs=self.n_convs)

    def get_model(self, task_id, **task_infos):
        if task_id >= len(self.models):
            # this is a new task

            # New tasks should always give the x_dim and n_classes infos.
            assert 'x_dim' in task_infos and 'n_classes' in task_infos
            assert task_id == len(self.models)

            model = self._new_model(task_id=task_id, **task_infos)
            self.models.append(model)
        return self.models[task_id]

    @abc.abstractmethod
    def _new_model(self, **kwargs):
        raise NotImplementedError

    def get_search_space(self, smoke_test):
        # if smoke_test:
        #     lr = tune.grid_search([1e-3])
        #     wd = tune.grid_search([0])
        #     dropout = tune.grid_search([0., 0.5])
        # else:
        #     lr = tune.grid_search([1e-2, 1e-3, 5e-4, 1e-4])
        #     wd = tune.grid_search([0])
        #     dropout = tune.grid_search([0, 0.5])

        params = {k: tune.grid_search(v) for k, v in self.grid_params.items()}

        return params

    def finish_task(self, dataset):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def prepare_task(self, task, training_params):
        pass

    def train_model_on_task(self, task, task_viz, exp_dir, use_ray,
                            use_ray_logging, smoke_test, n_it_max, grace_period,
                            num_hp_samplings, local_mode, tune_register_lock,
                            resources, **training_params):
        logger.info("Training dashboard: {}".format(get_env_url(task_viz)))

        model = self.get_model(task_id=task.id)
        trainable = self.get_trainable(use_ray_logging=use_ray_logging)

        self.prepare_task(task, training_params)

        if use_ray:
            # Required to avoid collisions in Tune's global Registry:
            # https://github.com/ray-project/ray/blob/master/python/ray/tune/registry.py
            trainable = rename_class(trainable, training_params['name'])

            scheduler = None


            training_params['loss_fn'] = tune.function(
                training_params['loss_fn'])
            training_params['optim_func'] = tune.function(self.optim_func)
            training_params['n_it_max'] = n_it_max

            init_model_path = os.path.join(exp_dir, 'model_initializations')
            model_file_name = '{}_init.pth'.format(training_params['name'])
            model_path = os.path.join(init_model_path, model_file_name)
            torch.save(model, model_path)

            training_params['model_path'] = model_path
            config = {'hyper-params': self.get_search_space(smoke_test),
                      'tp': training_params}
            if use_ray_logging:
                stop_condition = {'training_iteration': n_it_max}
                loggers = None
            else:
                stop_condition = None
                loggers = [JsonLogger, MyCSVLogger]

            # We need to create the experiment using a lock here to avoid issues
            # with Tune's global registry, more specifically with the
            # `_to_flush` dict that may change during the iteration over it.
            # https://github.com/ray-project/ray/blob/e3c9f7e83a6007ded7ae7e99fcbe9fcaa371bad3/python/ray/tune/registry.py#L91-L93
            tune_register_lock.acquire()
            experiment = Experiment(
                name=training_params['name'],
                run=trainable,
                stop=stop_condition,
                config=config,
                resources_per_trial=resources,
                num_samples=num_hp_samplings,
                local_dir=exp_dir,
                loggers=loggers,
                keep_checkpoints_num=1,
                checkpoint_score_attr='min-mean_loss')
            tune_register_lock.release()

            analysis = tune.run(experiment,
                                scheduler=scheduler,
                                verbose=1,
                                raise_on_failed_trial=True,
                                # max_failures=-1,
                                # with_server=True,
                                # server_port=4321
                                )
            os.remove(model_path)
            logger.info("Training dashboard: {}".format(get_env_url(task_viz)))

            all_trials = {t.logdir: t for t in analysis.trials}
            best_logdir = analysis.get_best_logdir('mean_loss', 'min')
            best_trial = all_trials[best_logdir]

            # picked_metric = 'accuracy_0'
            # metric_names = {s: '{} {}'.format(s, picked_metric) for s in
            #                 ['Train', 'Val', 'Test']}

            logger.info('Best trial: {}'.format(best_trial))
            best_res = best_trial._checkpoint.last_result
            best_point = (best_res['training_iteration'], best_res['mean_loss'])

            y_keys = ['mean_loss' if use_ray_logging else 'Val nll', 'train_loss']
            epoch_key = 'training_epoch'
            it_key = 'training_iteration' if use_ray_logging else 'training_iterations'
            plot_res_dataframe(analysis, training_params['name'], best_point,
                               task_viz, epoch_key, it_key, y_keys)
            best_model = self.get_model(task_id=task.id)
            best_model.load_state_dict(torch.load(best_trial._checkpoint.value))

            t = best_trial._checkpoint.last_result['training_iteration']
        else:
            data_path = training_params.pop('data_path')
            past_tasks = training_params.pop('past_tasks')
            datasets = trainable._load_datasets(data_path,
                                                training_params['loss_fn'],
                                                past_tasks)
            train_loader, eval_loaders = get_classic_dataloaders(datasets,
                                                                 training_params.pop('batch_sizes'))
            optim = self.optim_func(model.parameters())

            t, accs, best_state_dict = train(model, train_loader, eval_loaders,
                                             optimizer=optim, viz=task_viz,
                                             n_it_max=n_it_max, **training_params)
        logger.info('Finishing task ...')
        t1 = time.time()
        self.finish_task(task.datasets[0])
        logger.info('done in {}s'.format(time.time() - t1))

        return t

    def get_trainable(self, use_ray_logging):
        if use_ray_logging:
            return TrainLLModel
        else:
            return OSTrainLLModel


class MyCSVLogger(Logger):
    def _init(self):
        """CSV outputted with Headers as first set of results."""
        # Note that we assume params.json was already created by JsonLogger
        progress_file = os.path.join(self.logdir, EXPR_PROGRESS_FILE)
        self._continuing = os.path.exists(progress_file)
        self._file = open(progress_file, "a")
        self._csv_out = None

    def on_result(self, result):
        tmp = result.copy()
        if "config" in tmp:
            del tmp["config"]

        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            if not self._continuing:
                self._csv_out.writeheader()
        columns_to_unroll = [tmp[col] for col in tmp['unroll_columns']]
        for i, row in enumerate(zip(*columns_to_unroll)):
            row = {k: v for k, v in zip(tmp['unroll_columns'], row)}
            if i == len(columns_to_unroll[0])-1:
                # Writing the additional information in the last row
                filtered_dict = {k: v for k, v in tmp.items() if k not in tmp['unroll_columns']}
                row.update(**filtered_dict)
            self._csv_out.writerow(row)
        self._file.flush()

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()
