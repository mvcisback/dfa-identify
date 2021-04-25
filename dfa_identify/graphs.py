"""Module for model augmented prefix tree acceptor."""
from __future__ import annotations
from itertools import chain, combinations
from typing import Any

import attr
import networkx as nx
import funcy as fn


Label = bool
Word = list[Any]
Examples = dict[bool, list[Word]]
Node = Any
Nodes = set[Node]


def transition(tree: nx.DiGraph, node: Node, char: Any) -> Node:
    for node in tree.neighbors(node):
        if tree.nodes[node]['source'] == char:
            return node


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class APTA:
    """Augmented Prefix Tree Acceptor."""
    tree: nx.DiGraph

    @staticmethod
    def from_examples(examples: Examples) -> APTA:
        # Create prefix tree.
        tree, root = nx.prefix_tree(chain(*examples.values()))
        tree.remove_node(nx.generators.trees.NIL)  # <-- sink node added by nx.

        def access(word: Word) -> Node:
            node = root
            for char in word:  # Walk tree for node accessed by word.
                node = transition(tree, node, char)
            return node

        # Augment tree with node labels.
        for label, words in examples.items():
            for word in words:
                tree.nodes[access(word)]['label'] = label

        return APTA(tree)

    def consistency_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.tree.nodes)
        for pair in combinations(self.tree.nodes, 2):
            if not self.can_merge(graph, pair):
                graph.add_edge(*pair)

        return graph

    def can_merge(self, graph: nx.Graph, pair: Tuple[Node, Node]) -> bool:
        # TODO: joint DFS to make sure residual languages are compatible.
        # Note: Early terminate if reach two nodes known to be inconsistent.
        #      - edge in graph.edges
        succ = self.tree.neighbors
        nodes = self.tree.nodes

        stack = [pair]
        while stack:
            left, right = stack.pop()
            if (left, right) in graph.edges:
                return False  # Reached known distinguished nodes.

            left_lbl = nodes[left].get('label')
            right_lbl = nodes[right].get('label')
            if None not in {left_lbl, right_lbl} and left_lbl != right_lbl:
                return False  # Discovered distiguishing path.

            # Group neighbors by access token.
            succ_left = {nodes[n]['source'] for n in succ(left)}
            succ_right = {nodes[n]['source'] for n in succ(right)}
            merged = fn.merge(succ_left, succ_right)
            
            succ_pairs = [p for p in merged if len(p) > 1]  # Paths
            stack.extend(succ_pairs)

        return True
