# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from numbers import Number

import numpy as np
import pandas as pd
import torch
from json2html import json2html
from sklearn.metrics import auc
import plotly.graph_objs as go


def get_env_name(config, _id, main=False):
    var_domain = config['datasets']['task_gen']['strat']['domain']
    train_samples = config['datasets']['task_gen']['samples_per_class'][0]
    name = '{}-{}'.format(var_domain, train_samples)
    # name = '{}'.format(config['experiment']['name'])
    if _id is not None:
        name = '{}-{}'.format(name, _id)

    if main:
        name = '{}_main'.format(name)

    return name


def get_aucs(values, normalize, ks=None):
    """
    Get the AUCs for one task
    :param values: dict of format {'split_1' :{x1:y1, x2:y2, ...},
                                   'split_2' :{x1:y1, x2:y2, ...}}
    :param normalize: Will divide auc@k by k if True
    :return: {'split_1' :([x1, x2, ...], [auc@x1, auc@x2, ...]),
              'split_2' :([x1, x2, ...], [auc@x1, auc@x2, ...])}
    """
    res_at = {}
    for split_name, split_metrics in values.items():
        x, y = zip(*split_metrics.items())
        auc_at = ([], [])
        for i in range(1, len(x)):
            if ks and x[i] not in ks:
                continue
            auc_at[0].append(x[i])
            auc_at_i = auc(x[:i + 1], y[:i + 1])
            if normalize:
                auc_at_i /= x[i]
            auc_at[1].append(auc_at_i)
        res_at[split_name] = auc_at
    return res_at


def plot_transfers(all_perfs, similarities, task_names, task_viz,
                   agg_func=np.mean):
    """
    Plot the Final accuracy vs similarity scatter plot using two settings:
        - 'Ideal' where the accuracy used are the ones obtained after training
        on each task (diag of transfer matrix)
        - 'Current' where the accuracy used are the one on all task at current
        timestep.
    :param all_perfs: Nxt transfer matrix where N is the total number of tasks
    and t the number of tasks experienced so far
    :param similarities: array of size N containing the similarity between each
    task and the current one.
    :param task_names:
    :param task_viz:
    :return:
    """
    if not all_perfs:
        return 0, 0
    n_seen_tasks = len(all_perfs)

    prev_perf_after_training = [all_perfs[i][i] for i in range(n_seen_tasks)]
    prev_perf_now = all_perfs[-1][:n_seen_tasks]

    ideal = list(zip(similarities, prev_perf_after_training))
    dists_ideal = torch.tensor(ideal).norm(dim=1).tolist()

    current = list(zip(similarities, prev_perf_now))
    dists_current = torch.tensor(current).norm(dim=1).tolist()
    labels = torch.arange(n_seen_tasks) + 1

    opts = {'markersize': 5, 'legend': task_names, 'xtickmin': 0, 'xtickmax': 1,
            'ytickmin': 0, 'ytickmax': 1,
            'xlabel': 'similarity', 'width': 600}
    task_viz.scatter(ideal, labels,
                     opts={'title': 'Ideal Transfer', 'textlabels': dists_ideal,
                           'ylabel': 'acc after training', **opts})
    task_viz.scatter(current, labels, opts={'title': 'Current Transfer',
                                            'textlabels': dists_current,
                                            'ylabel': 'current acc', **opts})

    return agg_func(dists_ideal), agg_func(dists_current)


def plot_aucs(all_aucs, task_viz, main_viz):
    """
    Update the auc curves:
        - Draw k vs auc@k in the environment of this training for the current
        task
        - Draw k vs average auc@k on all tasks seen by this model so far (in
        the training and global envs).
    :param all_aucs: List containing the auc of a given model for all tasks.
    Each element of this list is in the format
    returned by `get_aucs`: {'split_1' :([x1, x2, ...], [auc@x1, auc@x2, ...]),
                             'split_2' :([x1, x2, ...], [auc@x1, auc@x2, ...]),
                             ...}
    :param task_viz: The Visdom environment of the concerned training (model
    and task).
    :param main_viz: The Visdom environment of the global experiment.
    :return:
    """
    ### Update AUC plots
    opts = {'legend': list(all_aucs[-1].keys()), 'markersize': 3,
            'xlabel': 'n iterations', 'ylabel': 'AuC'}

    # Current task
    all_points, labels = [], []
    last_auc = all_aucs[-1]
    for i, (split, (x, y)) in enumerate(last_auc.items()):
        all_points.extend(zip(x, y))
        labels.extend([i + 1] * len(x))
    task_viz.scatter(all_points, labels, win='task_auc',
                     opts={'title': 'Task AUCs', **opts})

    # All tasks so far
    split_aucs = defaultdict(list)
    for t_aucs in all_aucs:
        for split, (x, y) in t_aucs.items():
            assert x == all_aucs[0][split][
                0], 'All AUC of a split should be computed at the same points'
            split_aucs[split].append(y)

    ys = np.mean(list(split_aucs.values()), 1).reshape(-1)
    xs = all_aucs[0]['Train'][0]
    labels = np.repeat(range(len(split_aucs)), len(xs)) + 1
    xs = np.tile(xs, len(split_aucs))

    task_viz.scatter(np.array([xs, ys]).transpose(), labels, opts={
        'title': 'Average task AUCs {}'.format(len(all_aucs)),
        **opts})
    main_viz.scatter(np.array([xs, ys]).transpose(), labels, win='task_aucs',
                     opts={'title': 'Task AUCs {}'.format(len(all_aucs)),
                           **opts})


def plot_potential_speed(models_aucs, potentials, potential_type, main_viz,
                         model_names, plot_labels, splits):
    """
    Update the AUC vs transfer potential plots.
    :param models_aucs:
    :param potentials:
    :param potential_type:
    :param main_viz:
    :param model_names:
    :param plot_labels:
    :param splits:
    :return:
    """
    aucs_at_x = defaultdict(lambda: defaultdict(list))
    for model_auc, pot, model_name in zip(models_aucs, potentials, model_names):
        # model auc is [{split1:([x1...], [y_1...]), split2:([],[])},
        #               {split1:...}, ...]
        for split, (xs, ys) in model_auc.items():
            if splits and split not in splits:
                continue
            for i, (x, y) in enumerate(zip(xs, ys)):
                aucs_at_x[x]['values'].append((pot, y))
                trace_name = '{}_{}'.format(model_name, split)
                aucs_at_x[x]['labels'].append(plot_labels[trace_name])
                aucs_at_x[x]['legend'].append(trace_name)

    for x, d in aucs_at_x.items():
        opts = {'legend': d['legend'], 'markersize': 5,
                'xlabel': 'Transfer Potential', 'ylabel': 'AuC', 'width': 600,
                'height': 400}

        main_viz.scatter(d['values'], np.array(d['labels']) + 1,
                         win='{}_transaccbis{}'.format(potential_type, x),
                         update='append', opts={
                'title': '{} speed transfer@{}'.format(potential_type, x),
                **opts})

    return aucs_at_x


def plot_heatmaps(models, metrics, viz, **kwargs):
    """
    Plot the acc
    :param metrics: list of M matrix of size (N*t), where M is the number of
    models, N the total number of tasks and t
    the number of tasks seen so far.
    :param models: List of the names of the M models
    :param viz: The Visdom env in which the heatmaps will be drawn.
    :return:
    """
    kwargs['xlabel'] = kwargs.get('xlabel', 'Task seen')
    kwargs['ylabel'] = kwargs.get('ylabel', 'Task perf')

    for name, model_perfs in zip(models, metrics):
        opts = kwargs.copy()
        opts['title'] = opts.get('title', '{} transfer matrix'.format(name))
        if not torch.is_tensor(model_perfs):
            model_perfs = torch.tensor(model_perfs)
        viz.heatmap(model_perfs.t(), win='{}_heatmap'.format(name), opts=opts)


def plot_speeds(all_speeds, model_names, viz):
    task_id = len(all_speeds[0])

    new_points = []
    for model_speeds, mode_name in zip(all_speeds, model_names):
        new_points.append((len(model_speeds), model_speeds[-1]))

    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_points, labels, win='speeds',
                update='append' if task_id > 1 else None,
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'n task seen',
                      'ylabel': 'n iterations to converge', 'width': 600,
                      'height': 400, 'title': 'Training Durations'})


def plot_times(all_speeds, model_names, viz):
    new_points = []
    for model_speeds, mode_name in zip(all_speeds, model_names):
        new_points.append((len(model_speeds), model_speeds[-1]))

    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_points, labels, win='times',
                update='append' if len(all_speeds[0]) > 1 else None,
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'N task seen',
                      'ylabel': 'Time (s) to find the model', 'width': 600,
                      'height': 400, 'title': 'Time taken by the learner to '
                                              'give the model'})

def plot_times_bis(all_speeds, model_names, unit, viz):
    task_id = len(all_speeds[0])

    new_points = []
    for model_speeds in all_speeds:
        assert len(model_speeds) == task_id
        new_points.append((task_id, model_speeds[-1]))

    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_points, labels, win='times_{}'.format(unit),
                update='append' if task_id > 1 else None,
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'N task seen',
                      'ylabel': 'Time ({}) to find the model',
                      'width': 600,
                      'height': 400,
                      'title': 'Time ({}) taken by the learner.'.format(unit)})


def plot_accs(all_accs, model_names, viz, title):
    task_id = len(all_accs[0]) - 1
    last_task_accs = torch.tensor(all_accs)[:, -1, task_id]

    new_points = list(
        (task_id + 1, acc.item()) for acc in last_task_accs.unbind())
    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_points,
                labels,
                update='append' if task_id > 0 else None,
                win='acctasks{}'.format(title),
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'Task id',
                      'ylabel': 'Task test accuracy', 'width': 600,
                      'height': 400, 'title': title})


def plot_accs_data(all_accs, model_names, n_samples, viz):
    task_id = len(all_accs[0]) - 1
    last_task_accs = torch.tensor(all_accs)[:, -1, task_id]

    new_points = list(
        (n_samples[task_id], acc.item()) for acc in last_task_accs.unbind())
    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_points, labels, win='accdatatasks',
                update='append' if len(all_accs[0]) > 1 else None,
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'n samples per class',
                      'ylabel': 'Task test accuracy', 'width': 600,
                      'height': 400, 'title': 'Learning accuracies wrt data '
                                              'quantity'})


def plot_speed_vs_tp(training_times, potentials, potential_type, model_names,
                     viz):
    new_point = [[potential[-1], times[-1]] for times, potential in
                 zip(training_times, potentials)]
    labels = list(range(1, len(model_names) + 1))
    viz.scatter(new_point, labels,
                update='append' if len(training_times[0]) > 1 else None,
                win='{}_speeds_tp'.format(potential_type),
                opts={'legend': list(model_names), 'markersize': 5,
                      'xlabel': 'Transfer Potential',
                      'ylabel': 'Time to converge', 'width': 600,
                      'height': 400,
                      'title': '{} speed-TP'.format(potential_type)})


def plot_corr_coeffs(all_aucs, potentials, model_names, viz):
    all_potential_auc_at_k = defaultdict(lambda: defaultdict(list))
    for model_aucs, model_potentials, model_name in zip(all_aucs, potentials,
                                                        model_names):
        for task_aucs, potential in zip(model_aucs, model_potentials):
            for split, (xs, ys) in task_aucs.items():
                name = '{}-{}'.format(model_name, split)
                for k, auc_at_k in zip(xs, ys):
                    all_potential_auc_at_k[k][name].append(
                        (potential, auc_at_k))

    corr_coeffs = defaultdict(dict)

    for k, trace in all_potential_auc_at_k.items():
        for trace_name, trace_values in trace.items():
            corr = np.corrcoef(trace_values, rowvar=False)
            corr_coeffs[k][trace_name] = corr if isinstance(corr, Number) \
                                              else corr[0, 1]

    names = list(list(corr_coeffs.values())[0].keys())
    ks = ['@{}'.format(k) for k in corr_coeffs.keys()]
    corr_coeffs = [list(vals.values()) for vals in corr_coeffs.values()]
    viz.heatmap(corr_coeffs, opts={'columnnames': names, 'rownames': ks,
                                   'title': 'TP-AUC correlation after {}'
                                            'tasks'.format(len(all_aucs[0]))})

    return all_potential_auc_at_k


def plot_res_dataframe(analysis, plot_name, best_point, viz, epoch_key, it_key,
                       y_keys, width=1500, height=500):
    logdir_to_trial = {t.logdir: t for t in analysis.trials}
    res_df = pd.concat([t[y_keys] for t in
                        analysis.trial_dataframes.values()], axis=1)
    res_df.columns = ['{}-{}'.format(key, logdir_to_trial[logdir].experiment_tag)
                      for logdir in analysis.trial_dataframes.keys() for key in y_keys]

    longest_df = max(analysis.trial_dataframes.values(), key=len)
    x_epochs = longest_df[epoch_key]
    x_iterations = longest_df[it_key]

    _plot_df(x_epochs, res_df, viz, plot_name, width, height)
    _plot_df(x_iterations, res_df, viz, plot_name, width, height, best_point)


def _plot_df(x_series, df, viz, plot_name, width, height, best_point=None):
    assert len(x_series) == len(df)

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=x_series.values, y=df[col], name=col,
                                 line_width=1))
    fig.layout.title.text = plot_name
    fig.layout.xaxis.title.text = x_series.name
    fig.layout.yaxis.title.text = 'Validation loss'
    if best_point:
        fig.add_trace(go.Scatter(x=[best_point[0]],
                                 y=[best_point[1]],
                                 mode='markers',
                                 name='best checkpoint',
                                 marker_color='red',
                                 marker_size=5))
    win = viz.plotlyplot(fig)
    viz.update_window_opts(win=win, opts=dict(width=width, height=height,
                                              showlegend=True))


def plot_tasks_env_urls(urls, viz):
    all_models_urls = []
    for model, env_urls in urls.items():
        links = []
        for url in env_urls:
            links.append('<a href="{}">{}</a>'.format(url, url))
        model_urls = '<br/>'.join(links)
        all_models_urls.append('{}:<br/>{}'.format(model, model_urls))

    all_urls = '<br/><br/>'.join(all_models_urls)
    viz.text(all_urls, win='urls', opts={'width': 700})

def update_acc_plots(all_perfs, name, viz):
    n_tasks = len(all_perfs)
    current_accuracies = all_perfs[-1]
    accuracies_when_seen = [all_perfs[i][i] for i in range(n_tasks)]

    mean_perf_on_all_tasks = np.mean(current_accuracies)
    mean_perf_on_seen_tasks = np.mean(current_accuracies[:n_tasks])
    mean_perf_when_seen = np.mean(accuracies_when_seen)
    viz.line(Y=[[mean_perf_on_all_tasks],
                [mean_perf_on_seen_tasks],
                [mean_perf_when_seen]],
             X=[n_tasks],
             win='{}aggrac'.format(name),
             update='append' if n_tasks > 1 else None,
             opts={'title': '{} Average Test Accuracies'.format(name),
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Accuracy',
                   'width': 600,
                   'height': 400,
                   'legend': ['Current acc on all tasks',
                              'Current acc on seen tasks',
                              'Acc on tasks when seen']})


def update_avg_acc(all_accs, model_names, viz, title):
    n_tasks = len(all_accs[0])
    acc_when_seen = [[mod_acc[i][i] for i in range(n_tasks)] for mod_acc in
                     all_accs]
    mean_accs = np.mean(acc_when_seen, axis=1, keepdims=True)
    viz.line(Y=mean_accs,
             X=[n_tasks], win='mean_accs_{}'.format(title),
             update='append' if n_tasks > 1 else None,
             opts={'title': title,
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Avg Accuracy',
                   'width': 600,
                   'height': 400,
                   'legend': model_names})
    return mean_accs.squeeze(-1)


def update_speed_plots(all_speeds, model_names, viz):
    n_tasks = len(all_speeds[0])
    mean_speeds = np.mean(all_speeds, axis=1, keepdims=True)
    viz.line(Y=mean_speeds,
             X=[n_tasks],
             win='meanspeed',
             update='append' if n_tasks > 1 else None,
             opts={'title': 'Average Training Duration',
                   'xlabel': 'N tasks seen',
                   'ylabel': 'Average N iterations to converge',
                   'width': 600,
                   'height': 400,
                   'legend': model_names})
    return mean_speeds.squeeze(-1)


def update_summary(summary, viz):
    data = []
    for items in zip(*summary.values()):
        row = {}
        for k, val in zip(summary.keys(), items):
            row[k] = "{:4.3f}".format(val) if isinstance(val, float) else val
        data.append(row)

    html_summaty = json2html.convert(data,
                                     table_attributes="id=\"info-table\" "
                                                      "class=\"table "
                                                      "table-bordered"
                                                      " table-hover\"")
    viz.text(html_summaty, win='summary',
             opts={'title': 'Summary',
                   'width': 350,
                   'height': 350})

