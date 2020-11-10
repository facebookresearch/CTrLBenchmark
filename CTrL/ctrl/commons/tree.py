# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import random
from functools import lru_cache

import networkx as nx


class Tree(abc.ABC):
    """
    Abstract Tree structure containing basic attributes and methods.
    # """
    def __init__(self, name, seed=None):
        super().__init__()

        self.tree = nx.DiGraph()
        self.name = name

        self.leaf_nodes = set()
        self.all_nodes = set()

        self._seed = seed
        self.rnd = random.Random(self._seed)
        self.root_node = self.build_tree()

        self._shortest_path_lengths = None
        self._shortest_paths = None

    @property
    def shortest_path_lengths(self):
        if self._shortest_path_lengths is None:
            self._shortest_path_lengths = dict(
                nx.shortest_path_length(self.tree.to_undirected()))
        return self._shortest_path_lengths

    @property
    def shortest_paths(self):
        if self._shortest_paths is None:
            self._shortest_paths = dict(nx.shortest_path(self.tree))
        return self._shortest_paths

    @abc.abstractmethod
    def build_tree(self):
        raise NotImplementedError

    @lru_cache(1000)
    def lowest_common_ancestor(self, nodes):
        """
        Computes the LCA of a bunch of nodes.
        :param nodes: tuple of nodes
        :return: the Lowest Common Ancestor of the nodes.
        """
        # TODO change that using
        #  networkx::tree_all_pairs_lowest_common_ancestor
        cur_lca = nodes[0]
        for node in nodes[1:]:
            cur_lca = nx.lowest_common_ancestor(self.tree, cur_lca, node)
        return cur_lca

    def wu_palmer(self, a, b):
        """
        Compute the similarity between two nodes in the tree as

        sim(a, b) = 2 * depth(lcs(a, b)) / (depth(a) + depth(b))

        where lcs(a, b) is the Least Common Subsumer of two nodes (aka
        Lowest Common Ancestor).

        https://www.geeksforgeeks.org/nlp-wupalmer-wordnet-similarity/
        """

        lcs = self.lowest_common_ancestor((a, b))
        depth_lcs = self.shortest_path_lengths[self.root_node][lcs]
        depth_a = self.shortest_path_lengths[self.root_node][a]
        depth_b = self.shortest_path_lengths[self.root_node][b]

        return 2 * depth_lcs / (depth_a + depth_b)

    def get_compatible_nodes(self, N=None, exclude_nodes=None, leaf_only=False,
                             preferred_lca_dist=-1, max_lca_dist=-1,
                             force_nodes=None,):
        """
        Searches for N compatible nodes in the tree

        :param N: Int, Number of nodes to select
        :param exclude_nodes: Iterable, Nodes already selected, their parents and children will also be excluded from the search.
        :param leaf_only: Bool, the selection will be made only on leaf concepts if True.
        :return: A List of N compatible nodes.
        :raises ValueError: If the selection is not possible.
        """
        if exclude_nodes is None:
            exclude_nodes = set()
        else:
            exclude_nodes = set(exclude_nodes)
        if force_nodes is None:
            available_nodes = self.leaf_nodes if leaf_only else self.all_nodes
        else:
            available_nodes = force_nodes

        for node in exclude_nodes:
            children = nx.dfs_preorder_nodes(self.tree, node)
            parents = self.shortest_paths[self.root_node][node]
            available_nodes = available_nodes.difference(children, parents)

        if max_lca_dist == -1:
            max_lca_dist = self.depth

        cur_max_lca_dist = preferred_lca_dist
        while cur_max_lca_dist <= max_lca_dist:
            try:
                return self._get_compatible_nodes(N,
                                                  available_nodes,
                                                  cur_max_lca_dist)
            except ValueError:
                # The problem doesn't have any solution, we will try with a
                # broader search space if possible.
                cur_max_lca_dist += 1

        raise ValueError("Impossible to find new compatible Nodes")

    def _get_compatible_nodes(self, N, candidates, max_lca_dist):
        """
        Searches for a list of N compatible nodes from the list `candidates`.
        Two nodes are compatible if they don't have a parent/descendants
        relationship at any degress (i.e. there is no path going from A to B
        nor from B to A in the tree).
        A set of Nodes are compatible if every pair of nodes from the set is
        compatible.

        :param N: Number of compatible nodes to select.
        :param candidates: List or Set of nodes from which the selection will be made.
        :return: A list of compatible nodes of length N
        :raises ValueError: If no such combination exists.
        """

        # Make a copy of the list from which we will pick elements.
        cur_candidates = list(candidates)

        if N is None or N == 1:
            # We are done, we initialize the result with a randomly selected
            # element
            return [self.rnd.choice(cur_candidates)]

        # Before each trial, make sure that the problem can be solved
        while len(cur_candidates) >= N:
            cur_node = self.rnd.choice(cur_candidates)

            descendants = nx.dfs_preorder_nodes(self.tree, cur_node)
            ancestors = self.shortest_paths[self.root_node][cur_node]

            # Remove the ancestors and descendants nodes from the candidates
            # for the rest of the selection process
            new_candidates = candidates.difference(descendants, ancestors)
            if max_lca_dist != -1:
                new_candidates = self.filter_candidates(new_candidates,
                                                        cur_node,
                                                        max_lca_dist)

            try:
                # Try to solve the sub problem using the updated candidates
                other_nodes = self._get_compatible_nodes(N - 1, new_candidates,
                                                         max_lca_dist)
                # We have a solution for the sub-problem, we add the current
                # node to this solution
                other_nodes.append(cur_node)
                return other_nodes
            except (ValueError, IndexError):
                # There is no solution possible for the sub problem, we will
                # try another node at current level
                cur_candidates.remove(cur_node)

        # The problem doesn't have any solution
        raise ValueError("Impossible to find new compatible Nodes")

    def filter_candidates(self, candidates, selected, max_lca_dist):
        path_to_selected = self.shortest_paths[self.root_node][selected]
        max_lca = path_to_selected[-(max_lca_dist + 1)]
        max_lca_children = nx.dfs_preorder_nodes(self.tree, max_lca)
        return candidates.intersection(max_lca_children)
