# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from numbers import Number

import networkx as nx
import numpy as np
import torch
from torch.distributions import uniform

from src.concepts.concept import Concept

logger = logging.getLogger(__name__)


class ConceptTree(object):
    def __init__(self, n_levels, n_children, n_dims, low=-1000, high=1000, mean=0):

        self.n_levels = n_levels
        self.concepts = []

        self.tree = nx.DiGraph()
        self._build_tree(n_levels, n_children, n_dims, low, high, mean, 'A_')

    def _build_tree(self, n_levels, n_children, n_dims, low, high, mean, parent_node):
        if isinstance(low, Number):
            low = torch.ones(n_dims) * low
        if isinstance(high, Number):
            high = torch.ones(n_dims) * high

        cluster_means = uniform.Uniform(low + mean, high + mean).sample((n_children,))
        pref = '\t' * (self.n_levels - n_levels)
        for i, cluster_mean in enumerate(cluster_means):
            logger.debug(pref + 'New cluster centered on {}'.format(cluster_mean))
            node_name = '{}{}'.format(parent_node, i)
            self.tree.add_edge(parent_node, node_name)
            if n_levels > 0:
                self._build_tree(n_levels - 1, n_children, n_dims, low=low/10, high=high/10, mean=cluster_mean, parent_node=node_name)
                self.tree.add_node(node_name, cluster_mean=cluster_mean)
            else:
                concept = Concept(cluster_mean, np.eye(n_dims), node_name)
                self.tree.add_node(node_name, concept=concept)

    def _get_concept_node(self, N=None):
        """
        :param N: The number of concepts to select or None for a single Concept.
        :return: A list containing N atomic concepts (or a single Concept if N is None) selected at random uniformly.
        """
        all_concepts = self.get_leaf_nodes()
        return np.random.choice(all_concepts, N, replace=False)

    def get_concepts(self, N):
        """
        Get consistent set of concepts.
        This function first selects a Concept at random then selects N-1 compatible concepts.
        Compatibility is decided according to get_candidates function.
        :param N: The number of Concepts to get
        :return: A list of Concepts
        """
        first_node = self._get_concept_node()
        compatibles = self.get_compatible_concept([first_node], N - 1)
        nodes = np.append(compatibles, self.tree.nodes[first_node]['concept'])
        return nodes

    def get_compatible_concept(self, nodes, N=None):
        candidates = list(self.get_candidates(nodes))
        if not candidates:
            return None
        choice = np.random.choice(candidates, N, replace=False)
        return self._nodes_to_concepts(choice)

    def get_candidates(self, nodes):
        """
        Return a set of nodes compatible with the given nodes.
        For now a node is compatible is it has at least one sibling in the given list.
        :param nodes: List of nodes
        :return: A Set of nodes
        """
        if isinstance(nodes[0], Concept):
            nodes = self._concepts_to_nodes(nodes)
        nodes = set(nodes)
        candidates = set()
        for node in nodes:
            candidates.update(self.get_siblings(node))
        candidates.difference_update(nodes)
        return candidates

    def get_siblings(self, node):
        """
        Get all the direct siblings of a given node.
        """
        predecessors = list(self.tree.predecessors(node))
        assert len(predecessors) == 1
        parent = predecessors[0]
        all = list(self.tree.successors(parent))
        all.remove(node)
        return all



    def get_extended_siblings(self):
        """
        Get all the direct siblings of a given node and all their children.
        """

    def get_leaf_concepts(self):
        return [self.tree.node()[node]['concept'] for node, out_d in self.tree.out_degree() if out_d==0]

    def get_leaf_nodes(self):
        return [node for node, out_d in self.tree.out_degree() if out_d == 0]

    def _concepts_to_nodes(self, concepts):
        return [c.descriptor for c in concepts]

    def _nodes_to_concepts(self, nodes):
        if isinstance(nodes, list) or isinstance(nodes, set) or isinstance(nodes, np.ndarray):
            return [self.tree.nodes[n]['concept'] for n in nodes]
        else:
            return self.tree.nodes[nodes]['concept']
