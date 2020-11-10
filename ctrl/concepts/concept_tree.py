# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Trees of concepts
"""

import abc
import logging
from collections import defaultdict

import bs4
import networkx as nx
import numpy as np
import torch
from ctrl.commons.tree import Tree
from ctrl.commons.utils import default_colors
from ctrl.concepts.concept import ComposedConcept
from pydot import quote_if_necessary

logger = logging.getLogger(__name__)


class ConceptTree(Tree):
    def __init__(self, n_levels, n_children, n_samples_per_concept, n_attrs,
                 *args, **kwargs):
        """
        High level structure from which we can retrieve concepts while
        respecting some constraints. Could be an subtype of an abstract
        ConceptPool class if we want to represent the concepts using another
        data structure at some point. Currently we are only using Trees so
        an additional level of abstraction isn't required.

        :param n_levels:
        :param n_children:
        :param n_samples_per_concept: list of integers representing the number
            of samples per split for each leaf concept
        :param split_names: Should have the same length as
            `n_samples_per_concept`
        """
        if isinstance(n_children, list) and len(n_children) == 1:
            n_children = n_children * n_levels
        if isinstance(n_children, list):
            n_levels = len(n_children)
        self.n_levels = n_levels
        self.depth = n_levels
        self.n_children = n_children
        # self.concepts = []

        self.n_attrs = n_attrs
        self.attributes = []
        self.attribute_similarities = None

        super().__init__(*args, **kwargs)
        self.init_data(n_samples_per_concept=n_samples_per_concept)
        self.init_attributes(self.n_attrs)

    @abc.abstractmethod
    def build_tree(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_attributes(self, n_attrs):
        raise NotImplementedError

    def init_data(self, n_samples_per_concept):
        """
        Init all nodes but the root, starting from the leafs and
        climbing up the tree.

        :param n_samples_per_concept: Iterable containing the number of samples
            per split.
            For example [500, 200] will init the data for two splits, containing
            500 samples for the first one and 200 samples for the second.
        """
        for node in self.tree.successors(self.root_node):
            self._init_node(node, n_samples_per_concept)

    def _init_node(self, node, n_samples_per_leaf):
        successors = self.tree.successors(node)
        for succ in successors:
            self._init_node(succ, n_samples_per_leaf)

        node.init_samples(n_samples_per_leaf)

    def get_compatible_concepts(self, N=None, exclude_concepts=None,
                                leaf_only=False, preferred_lca_dist=-1,
                                max_lca_dist=-1, branch=None, nodes=None):
        """
        Searches for N compatible concepts in the pool
        :param N: Int, Number of concepts to select
        :param exclude_concepts: Iterable, Concepts already selected, their
        parents and children will also be excluded from the search.
        :param leaf_only: Bool, the selection will be made only on leaf concepts
        if True.
        :return: A List of N compatible concepts.
        :raises ValueError: If the selection is not possible.
        """
        force_nodes = None
        if branch is not None:
            for node in self.tree.successors(self.root_node):
                if node.descriptor == branch:
                    force_nodes = nx.descendants(self.tree, node)
                    if leaf_only:
                        force_nodes = self.leaf_nodes.intersection(force_nodes)
            assert force_nodes, 'Can\'t find branch {}'.format(branch)
        elif nodes:
            force_nodes = set()
            for c in self.all_concepts:
                for n in nodes:
                    if c.descriptor.endswith(n):
                        force_nodes.add(c)
            if len(nodes) != len(force_nodes):
                raise ValueError('Got concepts {} when nodes {} where asked.'
                                 .format(nodes, force_nodes))
        return self.get_compatible_nodes(N, exclude_concepts, leaf_only,
                                         preferred_lca_dist, max_lca_dist,
                                         force_nodes)

    def get_closest_categories(self, categories):
        res = None
        max_sim = None
        for i, c1 in enumerate(categories):
            for c2 in categories[i + 1:]:
                sim = self.get_pair_similarity(c1, c2)
                if max_sim is None or sim > max_sim:
                    res = [(c1, c2)]
                    max_sim = sim
                elif sim == max_sim:
                    res.append((c1, c2))
        return res

    def get_pair_similarity(self, cat1, cat2):
        """
        Compute the similarity between cat1 and cat2 as the minimum similarity
        between concepts composing cat1 and atomic concepts composing cat2.

        :param cat1: Iterable of concepts
        :param cat2: Iterable of concepts
        :return: The minimum wu_palmer similarity between concepts in cat1 and
            cat2.
        """
        cur_min = None
        for c1 in cat1:
            for c2 in cat2:
                sim = self.wu_palmer(c1, c2)
                if cur_min is None or sim < cur_min:
                    cur_min = sim

        return cur_min

    def merge_closest_categories(self, categories):
        pairs = self.get_closest_categories(categories)

        selected_pair = self.rnd.choice(pairs)

        categories.remove(selected_pair[0])
        categories.remove(selected_pair[1])

        categories.append(selected_pair[0] + selected_pair[1])
        return categories

    def get_widest_categories(self, categories):
        """
        Get the categories that covers the widest part of the tree. Computed
        using the lowest common ancestor of all the concepts contained in each
        category.

        :param categories: Iterable of iterables of concepts to select from
        :return: A list containing the categories covering the widest part of
            the tree.
        """
        lca = (self.lowest_common_ancestor(tuple(cat)) for cat in categories)
        depths = (self.shortest_path_lengths[self.root_node][node] for node in
                  lca)

        min_depth_categories = None
        min_depth = None

        for (cat, depth) in zip(categories, depths):
            if min_depth is None or depth < min_depth:
                min_depth_categories = [cat]
                min_depth = depth
            elif depth == min_depth:
                min_depth_categories.append(cat)

        return min_depth_categories

    def split_category(self, category, lca=None):
        """

        :param category:
        :param lca: The Lowest Common Ancestor of the concepts contained in this
            category. Will be computed if this parameter is left to None.
        :return:
        """
        if lca is None:
            lca = self.lowest_common_ancestor(tuple(category))

        if len(category) == 1:
            assert isinstance(category[0], ComposedConcept)
            assert category[0] == lca
            new_categories = [[n] for n in self.tree.successors(lca)]
        else:
            new_cat_dict = defaultdict(list)

            for concept in category:
                new_cat_lca = self.shortest_paths[lca][concept][1]
                new_cat_dict[new_cat_lca].append(concept)

            new_categories = list(new_cat_dict.values())
        return new_categories

    def categories_sim(self, cats1, cats2):
        """
        Only implemented for two pairs of categories, each category being
        composed by only 1 concept.
        :param cats1:
        :param cats2:
        :return:
        """

        assert len(cats1) == len(cats2)
        if not all(len(cat) == 1 for cat in cats1) \
                and all(len(cat) == 1 for cat in cats2):
            logger.warning('Can\'t compute sim between {} and {}'
                           .format(cats1, cats2))
            return float('nan')

        # Lets save some time if the categories are the same:
        if set(cats1) == set(cats2):
            return 1

        # First compute all pairwise similarities between categories
        sim_mat = torch.empty(len(cats1), len(cats2))
        for i, c1 in enumerate(cats1):
            for j, c2 in enumerate(cats2):
                sim_mat[i, j] = self.wu_palmer(c1[0], c2[0])

        # Then Greedily create the pairs
        n_categories = len(cats1)

        pairs = []
        for k in range(len(cats1)):
            a = torch.argmax(sim_mat).item()
            i = a // n_categories
            j = a % n_categories
            # the new pair is (cats1[i], cats2[j])
            pairs.append(sim_mat[i, j].item())
            sim_mat[i] = -1
            sim_mat[:, j] = -1

        return np.mean(pairs)

    def y_attributes_sim(self, y1, y2):
        use_cluster_id_1, attrs_1 = y1
        use_cluster_id_2, attrs_2 = y2

        if not use_cluster_id_1 == use_cluster_id_2:
            logger.warning('Only one tasks using cluster_id !')
            return float('nan')

        attrs_sims = []
        if use_cluster_id_1:
            attrs_sims.append(1)

        for attr_1, attr_2 in zip(attrs_1, attrs_2):
            attrs_sims.append(self.attribute_similarities[attr_1][attr_2])
        return np.mean(attrs_sims)

    def _concepts_to_nodes(self, concepts):
        logger.warning('_concepts_to_nodes is deprecated')
        return concepts

    def _nodes_to_concepts(self, nodes):
        logger.warning('_nodes_to_concepts is deprecated')
        return nodes

    @property
    def leaf_concepts(self):
        return self.leaf_nodes

    @property
    def all_concepts(self):
        return self.all_nodes

    @abc.abstractmethod
    def plot_concepts(self, viz):
        raise NotImplementedError

    def draw_tree(self, highlighted_concepts=None, tree=None, viz=None,
                  colors=None, title=None):
        """

        :param highlighted_concepts:
        :param tree:
        :return: A Pydot object
        """
        if tree is None:
            tree = self.tree

        if colors is None:
            colors = default_colors

        p = nx.drawing.nx_pydot.to_pydot(tree)

        if highlighted_concepts:
            if not isinstance(highlighted_concepts[0], (list, tuple)):
                highlighted_concepts = [highlighted_concepts]



            # Give a color to each group of nodes
            color_dict = {}
            for i, concept_list in enumerate(highlighted_concepts):
                highlighted_nodes = [c.descriptor for c in concept_list]
                for node in highlighted_nodes:
                    # quote_if_necessary is required to handle names in the same
                    # way as pydot
                    color_dict[quote_if_necessary(node)] = colors[
                        i % len(colors)]

            for node in p.get_nodes():
                if node.obj_dict['name'] in color_dict:
                    node.set_style('filled')
                    node.set_fillcolor(color_dict[node.obj_dict['name']])
        width = 18
        p.set_size('{}x1'.format(width))
        if viz:
            svg_bytes = p.create_svg()
            soup = bs4.BeautifulSoup(svg_bytes, 'xml')
            width = float(soup.svg.attrs['width'][:-2]) * 1.35
            height = float(soup.svg.attrs['height'][:-2]) * 1.4
            viz.svg(svgstr=str(svg_bytes),
                    opts=dict(title=title,
                              width=width,
                              height=height
                              ))
        return p

    def draw_attrs(self, viz):
        all_concepts = self.leaf_concepts
        for i, attr in enumerate(self.attributes):
            attr_positive_concepts = []
            for concept in all_concepts:
                if concept.attrs[i] == 1:
                    attr_positive_concepts.append(concept)
            self.draw_tree(highlighted_concepts=attr_positive_concepts, viz=viz,
                           title='attr {}'.format(i))

    def __str__(self):
        return type(self).__name__
