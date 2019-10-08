# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch
import visdom
from torch import nn

from src.concepts.concept_tree import ConceptTree
from src.tasks.task_generator import TaskGenerator
from src.transformations.TransformationTree import TansformationTree
from training.train import train

logging.basicConfig(level=logging.INFO)

def get_arg_parser():
    parser = argparse.ArgumentParser()

    ### Concetps
    parser.add_argument('--idim', '--intrinsic_dim',  type=int, default=2)
    parser.add_argument('--c_levels', '--concept_levels', type=int, default=2)
    parser.add_argument('--c_degree', '--concept_degree', type=int, default=3)

    ### Transformations
    parser.add_argument('--adim', '--ambiant_dim',  type=int, default=50)

    ### Tasks
    parser.add_argument('--n_classes', type=int, default=3)

    ### Model
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)

    return parser

def get_model(args):
    layers = []
    last_size = args['adim']
    for i in range(args['n_hidden']):
        layers.extend([nn.Linear(last_size, args['adim']),
                       nn.ReLU()])
        last_size = args['adim']
    layers.extend([nn.Linear(last_size, args['n_classes']),
                   nn.LogSoftmax(-1)])
    return nn.Sequential(*layers)


def main(args):
    concept_pool = ConceptTree(args['c_levels'], args['c_degree'], args['idim'])
    trans_pool = TansformationTree(args['idim'], args['adim'], n_rotations=5)
    gtg = TaskGenerator(concept_pool, trans_pool)

    model = get_model(args)

    viz = visdom.Visdom()
    # latent_tasks = gtg.available_concepts.concepts_means()

    for i in range(5):
        gtg.add_task(args['adim'], args['n_classes'], 500)
        t = gtg.task_pool[-1]
        print(gtg)

    concept_means = [elt.intrinsic_distrib._mean for elt in gtg.available_concepts.get_leaf_concepts()]
    selected_means = [elt.intrinsic_distrib._mean for elt in t.concepts]

    viz.scatter(torch.stack(concept_means), opts={'markersize': 3})
    viz.scatter(torch.stack(selected_means), opts={'markersize': 3})

    t.plot_task(viz)

    # train(model, t.dataset, target_train_acc=0.9)




if __name__ == '__main__':
    parser = get_arg_parser()
    namespace = parser.parse_args()
    main(vars(namespace))
