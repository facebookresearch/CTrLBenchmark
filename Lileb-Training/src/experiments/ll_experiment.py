# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from threading import Lock

import numpy as np
import ray
import torch
import visdom
from tqdm import tqdm

from lileb.tasks.strategies.mixed_strategy import MixedStrategy
from lileb.tasks.strategies.more_data_strategy import MoreDataStrategy
from src.modules import MultitaskHeadLLModel, MultitaskLegLLModel
from src.train.training import evaluate
from src.utils import VISDOM_CONF_PATH, load_conf
from src.utils.misc import get_env_url
from src.utils.plotting import plot_heatmaps, plot_transfers, plot_speeds, \
    plot_speed_vs_tp, plot_accs, get_env_name, plot_tasks_env_urls, \
    plot_accs_data, plot_times, update_summary, update_acc_plots, \
    update_speed_plots, update_avg_acc

logger = logging.getLogger(__name__)


class LifelongLearningExperiment(object):
    def __init__(self, task_gen, ll_models, cuda, n_it_max, n_tasks, patience,
                 grace_period, num_hp_samplings, visdom_traces_folder,
                 batch_sizes, plot_tasks, log_steps, log_epoch, name,
                 task_save_folder, use_ray, use_ray_logging, redis_address,
                 use_threads, local_mode, smoke_test, sacred_run, log_dir,
                 norm_models, resources, seed):
        self.task_gen = task_gen
        self.sims = None
        self.sims_comp = None
        self.name = name
        self.smoke_test = smoke_test

        assert isinstance(ll_models, dict)
        if 'finetune-mt-head' in ll_models:
            assert 'multitask-head' in ll_models and \
                   isinstance(ll_models['multitask-head'],
                              MultitaskHeadLLModel), \
                'Fine tune should be associated with multitak LLModel'
            ll_models['finetune-mt-head'].set_source_model(
                ll_models['multitask-head'])
        if 'finetune-mt-leg' in ll_models:
            assert 'multitask-leg' in ll_models and \
                   isinstance(ll_models['multitask-leg'], MultitaskLegLLModel), \
                'Fine tune leg should be associated with multitak Leg LLModel'
            ll_models['finetune-mt-leg'].set_source_model(
                ll_models['multitask-leg'])
        self.ll_models = ll_models
        self.norm_models = norm_models

        keys = list(self.ll_models.keys())
        self.norm_models_idx = [keys.index(nm) for nm in self.norm_models]

        if cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.use_ray = use_ray
        self.redis_address = redis_address
        self.use_threads = use_threads
        self.local_mode = local_mode
        self.use_ray_logging = use_ray_logging

        self.n_it_max = n_it_max
        self.n_tasks = n_tasks
        self.patience = patience
        self.grace_period = grace_period
        self.num_hp_samplings = num_hp_samplings
        self.resources = resources

        self.plot_tasks = plot_tasks
        self.batch_sizes = batch_sizes
        if os.path.isfile(VISDOM_CONF_PATH):
            self.visdom_conf = load_conf(VISDOM_CONF_PATH)
        else:
            self.visdom_conf = None

        self.log_steps = log_steps
        self.log_epoch = log_epoch

        self.sacred_run = sacred_run
        self.seed = seed

        self.exp_name = get_env_name(sacred_run.config, sacred_run._id)
        self.exp_dir = os.path.join(log_dir, self.exp_name)
        init_model_path = os.path.join(self.exp_dir, 'model_initializations')
        if not os.path.isdir(init_model_path):
            os.makedirs(init_model_path)
        self.visdom_traces_folder = os.path.join(visdom_traces_folder,
                                                 self.exp_name)

        self.task_save_folder = os.path.join(task_save_folder, self.exp_name)
        main_env = get_env_name(sacred_run.config, sacred_run._id, main=True)
        trace_file = os.path.join(self.visdom_traces_folder, main_env)
        self.main_viz = visdom.Visdom(env=main_env,
                                      log_to_filename=trace_file,
                                      **self.visdom_conf)
        task_env = '{}_tasks'.format(self.exp_name)
        trace_file = '{}/{}'.format(self.visdom_traces_folder,
                                    task_env)
        self.task_env = visdom.Visdom(env=task_env,
                                      log_to_filename=trace_file,
                                      **self.visdom_conf)

        self.summary = {'model': list(self.ll_models.keys()),
                        'speed': [],
                        'accuracy': []}
        update_summary(self.summary, self.main_viz)

        self.sacred_run.info['transfers'] = defaultdict(dict)
        self.task_envs_str = defaultdict(list)

        self.plot_labels = defaultdict()
        self.plot_labels.default_factory = self.plot_labels.__len__

        self.tune_register_lock = Lock()
        self.eval_lock = Lock()

        # Init metrics
        self.metrics = defaultdict(lambda: [[] for _ in self.ll_models])
        self.training_times_it = [[] for _ in self.ll_models]
        self.training_times_s = [[] for _ in self.ll_models]
        self.all_perfs = [[] for _ in self.ll_models]
        self.all_perfs_normalized = [[] for _ in self.ll_models]
        self.ideal_potentials = [[] for _ in self.ll_models]
        self.current_potentials = [[] for _ in self.ll_models]

    def run(self):
        if self.task_gen.concept_pool.attribute_similarities is not None:
            attr_sim = self.task_gen.concept_pool.attribute_similarities
            self.main_viz.heatmap(attr_sim, opts={'title':
                                                      'Attribute similarities'})
        if self.plot_tasks:
            self.task_gen.concept_pool.draw_tree(viz=self.main_viz, title='Full tree')
            self.task_gen.concept_pool.draw_attrs(viz=self.main_viz)
            self.task_gen.concept_pool.plot_concepts(self.main_viz)

        self.init_tasks()
        self.init_sims()

        if self.use_ray:
            if self.redis_address and not self.smoke_test:
                ray.init(redis_address=self.redis_address)
            else:
                ray.init(object_store_memory=int(1e7), include_webui=True,
                         local_mode=self.local_mode, num_gpus=0)

        for i, task in enumerate(self.task_gen.task_pool):
            self.train_on_task(task)

            self.update_plots()

        return self.summary

    def init_tasks(self):
        for i in range(self.n_tasks):
            task_name = '{}_T{}'.format(self.task_gen.concept_pool.name, i)
            t = self.task_gen.add_task(task_name, self.task_save_folder)
            if self.plot_tasks:
                self.task_gen.concept_pool.draw_tree(
                    highlighted_concepts=t.src_concepts,
                    viz=self.task_env,
                    title=task_name)
                task_env = '{}_T{}'.format(self.exp_name, i)
                trace_file = '{}/{}'.format(self.visdom_traces_folder,
                                            task_env)
                task_viz = visdom.Visdom(env=task_env,
                                         log_to_filename=trace_file,
                                         **self.visdom_conf)
                task_viz.text('<pre>{}</pre>'.format(t), win='task_descr',
                              opts={'width': 800, 'height': 250})
                t.plot_task(task_viz, 'T{}'.format(i))

            logger.info('###')
            logger.info('Task {}:'.format(i))
            logger.info(t)
            for name, ll_model in self.ll_models.items():
                ll_model.get_model(task_id=i, x_dim=t.x_dim,
                                   n_classes=t.n_classes.tolist(),
                                   descriptor=task_name)

    def init_sims(self):
        if isinstance(self.task_gen.strat, MixedStrategy):
            components = ''
        else:
            components = 'xyz'
        task_similarities = self.task_gen.get_similarities(components)

        for comp, sim in task_similarities.items():
            self.sacred_run.info['{}_similarities'.format(comp)] = sim.tolist()
            matrix_names = 'P({}) Similarity matrix'.format(comp)
            self.main_viz.heatmap(sim, opts={'title': matrix_names})

            vary_across = sim.numel() != sim.sum()
            if vary_across:
                if self.sims is not None:
                    logger.warning('Tasks are varying over 2 components '
                                     '({} and {}), which isn\'t supposed '
                                     'to happen.'.format(self.sims_comp, comp))
                self.sims_comp = comp
                self.sims = sim

        if self.sims is None:
            logging.warning('/!\\ All the tasks are identical /!\\')
            # Just a placeholder since all tasks are the same
            self.sims = torch.ones(self.n_tasks, self.n_tasks)

    def update_plots(self):
        all_model_names = list(self.ll_models.keys())
        speeds = update_speed_plots(self.training_times_it, all_model_names,
                                    self.main_viz)
        accs = update_avg_acc(self.all_perfs, all_model_names, self.main_viz,
                              'Average Accuracies')
        if self.norm_models:
            update_avg_acc(self.all_perfs_normalized, all_model_names,
                           self.main_viz, 'Normalized Average Accuracies')
        self.summary.update(speed=speeds.tolist(), accuracy=accs.tolist())
        update_summary(self.summary, self.main_viz)
        plot_heatmaps(all_model_names, self.all_perfs, self.main_viz)

        plot_speeds(self.training_times_it, all_model_names, self.main_viz)
        plot_accs(self.all_perfs, all_model_names, self.main_viz,
                  'Learning Accuracies')
        if self.norm_models:
            plot_accs(self.all_perfs_normalized, all_model_names, self.main_viz,
                      'Normalized Learning Accuracies')
        plot_times(self.training_times_s, all_model_names, self.main_viz)
        if isinstance(self.task_gen.strat, MoreDataStrategy):
            plot_accs_data(self.all_perfs, all_model_names,
                           self.task_gen.strat.n_samples_per_task_per_class,
                           self.main_viz)
        plot_speed_vs_tp(self.training_times_it, self.ideal_potentials, 'Ideal',
                         all_model_names, self.main_viz)
        plot_speed_vs_tp(self.training_times_it, self.current_potentials,
                         'Current',
                         all_model_names, self.main_viz)
        self.save_traces()

    def execute_step(self, step_calls):
        if not self.use_threads:
            return [f() for f in step_calls]

        with ThreadPoolExecutor() as executor:
            training_futures = [executor.submit(f) for f in step_calls]
            return [future.result() for future in training_futures]

    def save_traces(self):
        logger.info('Archiving traces folder ...')
        with tempfile.TemporaryDirectory() as dir:
            archive_name = os.path.join(dir, '{}_traces'.format(self.exp_name))
            shutil.make_archive(archive_name, 'zip', self.visdom_traces_folder)
            self.sacred_run.add_artifact('{}.zip'.format(archive_name))

    def train_on_task(self, task):
        logger.info('###############')
        logger.info('## Task {}/{} ##'.format(task.id, self.n_tasks))
        logger.info('###############')

        training_calls = []
        all_train_viz = []
        main_env_url = get_env_url(self.main_viz)
        logger.info("General dashboard: {}".format(main_env_url))
        logger.info('Tasks: {}'.format(get_env_url(self.task_env)))
        for j, (name, ll_model) in enumerate(self.ll_models.items()):
            ###
            # Init
            ###
            training_name = '{}_{}-{}'.format(self.exp_name, name,
                                              task.name)
            log_folder = '{}/{}'.format(self.visdom_traces_folder,
                                        training_name)
            task_viz = visdom.Visdom(env=training_name,
                                     log_to_filename=log_folder,
                                     **self.visdom_conf)
            task_viz.text('<pre>{}</pre>'.format(task), win='task_descr',
                          opts={'width': 800, 'height': 250})
            self.task_envs_str[name].append(get_env_url(task_viz))
            all_train_viz.append(task_viz)

            task_names = [t.name for t in self.task_gen.task_pool]
            ideal_tp, current_tp = plot_transfers(self.all_perfs[j],
                                                  self.sims[task.id],
                                                  task_names,
                                                  task_viz)
            self.ideal_potentials[j].append(ideal_tp)
            self.current_potentials[j].append(current_tp)

            if self.plot_tasks:
                task.plot_task(task_viz, training_name)
            ###
            # Prepare a call to train on task & Evaluate on all tasks
            ###
            past_tasks = self.task_gen.task_pool[:task.id]
            params = dict(task=task, past_tasks=past_tasks,
                          task_viz=task_viz, learner=ll_model,
                          training_name=training_name)
            training_calls.append(partial(self.give_task_to_learner,
                                          **params))

        # Execute all the training calls
        plot_tasks_env_urls(self.task_envs_str, self.main_viz)

        training_results = self.execute_step(training_calls)

        ### Handle the results
        if self.norm_models:
            min = np.array(training_results[self.norm_models_idx[0]][1])
            max = np.array(training_results[self.norm_models_idx[1]][1])

        for j, ((train_time, all_tests, all_confs), train_viz, learner_name) in \
                enumerate(zip(training_results, all_train_viz,
                              self.ll_models.keys())):
            self.training_times_it[j].append(train_time['iterations'])
            self.training_times_s[j].append(train_time['seconds'])
            for key, val in train_time.items():
                self.metrics['Train time {}'.format(key)][j].append(val)

            if self.norm_models:
                norm_tests = np.array(all_tests) - min
                norm_tests = (norm_tests / (max - min)).tolist()
                self.all_perfs_normalized[j].append(norm_tests)

            self.all_perfs[j].append(all_tests)

            ###
            # Plot
            ###
            plot_heatmaps([learner_name], [self.all_perfs[j]], train_viz)
            categories = list(
                map(str, self.task_gen.task_pool[task.id].src_concepts))
            plot_heatmaps([learner_name], [all_confs[task.id]], train_viz,
                          title='Confusion matrix', width=600, height=600,
                          xlabel='Predicted category',
                          ylabel='Real category',
                          # rownames=categories,
                          # columnnames=categories
                          )
            update_acc_plots(self.all_perfs[j], learner_name, self.main_viz)
            self.sacred_run.info['transfers'][learner_name] = self.all_perfs[j]
            name = '{} Accuracies'.format(learner_name)
            self.sacred_run.log_scalar(name, all_tests)
            # plot_aucs(training_aucs[j], task_viz, self.main_viz)

    def give_task_to_learner(self, task, past_tasks, task_viz, learner,
                             training_name):
        ###
        # Train
        ###
        past_tasks_infos = [(t.save_path, t.loss_fn) for t in past_tasks]
        training_params = dict(batch_sizes=self.batch_sizes,
                               n_it_max=self.n_it_max,
                               # viz=task_viz,
                               past_tasks=past_tasks_infos,
                               data_path=task.save_path,
                               split_names=task.split_names,
                               loss_fn=task.loss_fn,
                               patience=self.patience,
                               grace_period=self.grace_period,
                               num_hp_samplings=self.num_hp_samplings,
                               device=self.device,
                               name=training_name,
                               log_steps=self.log_steps.copy(),
                               log_epoch=self.log_epoch)

        start_time = time.time()
        train_time_it = learner.train_model_on_task(task, task_viz,
                                                    exp_dir=self.exp_dir,
                                                    smoke_test=self.smoke_test,
                                                    use_ray=self.use_ray,
                                                    use_ray_logging=self.use_ray_logging,
                                                    local_mode=self.local_mode,
                                                    tune_register_lock=self.tune_register_lock,
                                                    resources=self.resources,
                                                    **training_params)
        train_time_s = time.time() - start_time

        ###
        # Eval
        ###
        try:
            self.eval_lock.acquire()
            all_tests, all_confs = evaluate_on_tasks(self.task_gen.task_pool,
                                                 learner, self.batch_sizes[1],
                                                 self.device)
        except Exception as e:
            raise e
        finally:
            self.eval_lock.release()

        train_time = dict(seconds=train_time_s, iterations=train_time_it)
        return train_time, all_tests, all_confs


def evaluate_on_tasks(tasks, ll_model, batch_size, device):
    all_tests = []
    all_confusions = []
    for t_id, task in enumerate(tqdm(tasks, desc='Evaluation on tasks',
                                     leave=False)):
        test_model = ll_model.get_model(task_id=t_id)
        test_dataset = task.datasets[2]
        test_acc, conf_mat = evaluate(test_model, test_dataset, t_id,
                                      batch_size, device)
        all_tests.append(test_acc)
        all_confusions.append(conf_mat)
    return all_tests, all_confusions
