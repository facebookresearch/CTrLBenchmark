# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict

import networkx as nx
import numpy as np
from ctrl.strategies.task_creation_strategy import TaskCreationStrategy


class SplitStrategy(TaskCreationStrategy):
    def __init__(self, reuse_attrs, with_replacement, traj=None,
                 first_level_weighting=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse_attrs = reuse_attrs
        self.with_replacement = with_replacement
        assert first_level_weighting in [None, 'class_uniform', 'ds_uniform']
        self.first_level_weighting = first_level_weighting
        self.traj = traj

        self.idx = 0
        self.concept_order = {}
        self.all_used_concepts = []

    def new_task(self, task_spec, concepts, transformations, previous_tasks):

        if self.with_replacement:
            old_concepts = set()
        else:
            old_concepts = {c for task in previous_tasks
                            for c in task.concepts}

        if self.traj:
            traj_step = self.traj[self.idx]
            if traj_step[0] in self.concept_order:
                assert all(itm in self.concept_order for itm in traj_step)
                new_concepts = [self.concept_order[id] for id in traj_step]
            else:
                assert not any(itm in self.concept_order for itm in traj_step)
                if isinstance(traj_step[0], str) and \
                        all([isinstance(step, int) for step in traj_step[1:]]):
                    branch = traj_step.pop(0)
                    nodes = None
                elif all(isinstance(step, str) for step  in traj_step):
                    branch = None
                    nodes = traj_step
                else:
                    branch = None
                    nodes = None
                new_concepts = concepts.get_compatible_concepts(len(traj_step),
                                                            old_concepts, True,
                                                            preferred_lca_dist=self.concepts_preferred_lca_dist,
                                                            max_lca_dist=self.concepts_max_lca_dist,
                                                            branch=branch, nodes=nodes)
                for id, concept in zip(traj_step, new_concepts):
                    self.concept_order[id] = concept
            # for id in traj_step:
            #     if id not in self.concept_order:
            #         new = concepts.get_compatible_concepts(
            #             1, self.all_used_concepts, True,
            #             preferred_lca_dist=self.concepts_preferred_lca_dist,
            #             max_lca_dist=self.concepts_max_lca_dist)[0]
            #         self.all_used_concepts.append(new)
            #         self.concept_order[id] = new
            #     new_concepts.append(self.concept_order[id])
        elif self.first_level_weighting is not None:
            tree = concepts.tree
            if self.first_level_weighting == 'class_uniform':
                classes_per_ds = defaultdict(int)
                for branch in tree.successors(concepts.root_node):
                    for node in nx.dfs_preorder_nodes(tree, branch):
                        if tree.out_degree(node) == 0:
                            classes_per_ds[branch] += 1

                # for k, v in classes_per_ds.items():
                #     classes_per_ds[k] = 1 / v
                n_tot = sum(classes_per_ds.values())
                for k, v in classes_per_ds.items():
                    classes_per_ds[k] = v / n_tot

                branches = list(classes_per_ds.keys())
                probas = list(classes_per_ds.values())
            else:
                branches = list(tree.successors(concepts.root_node))
                probas = [1/len(branches)]*len(branches)

            b = np.random.choice(a=branches, size=1, p=probas)[0].descriptor

            n_classes = len(task_spec.src_concepts)
            new_concepts = concepts.get_compatible_concepts(n_classes,
                                                            old_concepts, True,
                                                            preferred_lca_dist=self.concepts_preferred_lca_dist,
                                                            max_lca_dist=self.concepts_max_lca_dist,
                                                            branch=b)
        else:
            n_classes = len(task_spec.src_concepts)
            new_concepts = concepts.get_compatible_concepts(n_classes,
                                                            old_concepts, True,
                                                            preferred_lca_dist=self.concepts_preferred_lca_dist,
                                                                max_lca_dist=self.concepts_max_lca_dist)
        new_concepts = [(c,) for c in new_concepts]

        self.idx += 1
        task_spec.src_concepts = new_concepts
        if not self.reuse_attrs:
            task_spec.attributes = (task_spec.attributes[0], [])
        return task_spec
