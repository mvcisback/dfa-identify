"""Module for model augmented prefix tree acceptor and consistency graphs.

See Heule, "Exact DFA Identification Using SAT Solve" for details.
"""
from __future__ import annotations
from itertools import chain, combinations
from typing import Any, Iterable, Tuple

import attr
import networkx as nx
import funcy as fn
from bidict import bidict


Word = list[Any]
Node = Any


def transition(tree: nx.DiGraph, node: Node, char: Any) -> Node:
    for node in tree.neighbors(node):
        if tree.nodes[node]['source'] == char:
            return node


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class APTA:
    """Augmented Prefix Tree Acceptor."""
    tree: nx.DiGraph
    alphabet: bidict  # Mapping from token to int.
    accepting_nodes: set[Node]  # Accepting states in the DFA.
    rejecting_nodes: set[Node]  # Rejecting states in the DFA.
    ord_prefs: set[Tuple[Node, Node]]  # MemReP 'ordered' preferences (less_preferred_word, more_preferred_word)
    inc_prefs: set[Tuple[Node, Node]]  # MemReP 'incomparable' preferences (incomparable_word_1, incomparable_word_2)


    @property
    def nodes(self) -> Iterable[Node]:
        return self.tree.nodes

    @property
    def root(self) -> Node:
        return 0

    @property
    def accepting(self) -> set[Node]:
         return self.accepting_nodes

    @property
    def rejecting(self) -> set[Node]:
        return self.rejecting_nodes

    @property
    def ordered_preferences(self) -> set[Tuple[Node, Node]]:
        return self.ord_prefs

    @property
    def incomparable_preferences(self) -> set[Tuple[Node, Node]]:
        return self.inc_prefs


    @staticmethod
    def from_examples(accepting: list[Word], rejecting: list[Word],
                      ordered_preference_words: list[Tuple[Word, Word]] = None,
                      incomparable_preference_words: list[Tuple[Word, Word]] = None) -> APTA:
        """Return Augmented Prefix Tree Automata for accepting, rejecting."""
        # If preference tuples weren't provided, initialize them as empty lists
        if ordered_preference_words is None:
            ordered_preference_words = []
        if incomparable_preference_words is None:
            incomparable_preference_words = []
        # Create prefix tree.
        tree, root = nx.prefix_tree(chain(accepting, rejecting, list(chain(*ordered_preference_words)),
                                          list(chain(*incomparable_preference_words))))
        tree.remove_node(nx.generators.trees.NIL)  # <-- sink node added by nx.

        # Label nodes with integers. With root = 0.
        relabels = {n: i + 1 for i, n in enumerate(set(tree.nodes) - {root})}
        relabels[root] = 0
        nx.relabel_nodes(tree, relabels, copy=False)

        alphabet = {d['source'] for n, d in tree.nodes(data=True) if n != 0}
        if None in alphabet:
            raise ValueError("None not allowed in alphabet.")
        alphabet = bidict(enumerate(alphabet)).inv

        def access(word: Word) -> Node:
            node = relabels[root]
            for char in word:  # Walk tree for node accessed by word.
                node = transition(tree, node, char)
            return node


        # Augment this class with node labels.
        accepting_nodes, rejecting_nodes = set(), set()
        for label, words in [(True, accepting), (False, rejecting)]:
            for word in words:
                if label:
                    accepting_nodes.add(access(word))
                else:
                    rejecting_nodes.add(access(word))

        # Build the ordered preferences tuple set.
        ordered_preference_nodes = set()
        for (word_one, word_two) in ordered_preference_words:
            ordered_preference_nodes.add((access(word_one), access(word_two)))

        # Build the incomparable preference tuple set.
        incomparable_preference_nodes = set()
        for (word_one, word_two) in incomparable_preference_words:
            incomparable_preference_nodes.add((access(word_one), access(word_two)))

        return APTA(tree, alphabet, accepting_nodes, rejecting_nodes, ordered_preference_nodes,
                    incomparable_preference_nodes)

    def consistency_graph(self) -> nx.Graph:
        """Return consistency graph for APTA via repeated DFS."""
        graph = nx.Graph()
        graph.add_nodes_from(self.tree.nodes)
        for pair in combinations(self.tree.nodes, 2):
            # if inconsistency
            if not self._can_merge(graph, pair):
                graph.add_edge(*pair)
        return graph

    def _can_merge(self, graph: nx.Graph, pair: Tuple[Node, Node]) -> bool:
        succ = self.tree.neighbors
        nodes = self.tree.nodes

        stack, visited = [pair], set()
        #simulate prefix tree at different nodes and see if distinguishing behavior arises
        while stack:  # DFS for inconsistency in states.
            left, right = stack.pop()

            if (left, right) in visited:
                continue
            visited.add((left, right))

            if (left, right) in graph.edges:
                return False  # Reached known distinguished nodes.

            if left in self.accepting:
                left_lbl = True
            else:
                left_lbl = False if left in self.rejecting else None

            if right in self.accepting:
                right_lbl = True
            else:
                right_lbl = False if right in self.rejecting else None

            if None not in {left_lbl, right_lbl} and left_lbl != right_lbl:
                return False  # Discovered distiguishing path.

            # Group neighbors by access token.
            succ_left = {nodes[n]['source']: n for n in succ(left)}
            succ_right = {nodes[n]['source']: n for n in succ(right)}
            merged = list(fn.merge_with(set, succ_left, succ_right).values())

            # Interchange pair[0] and pair[1] is applicable.
            for p1, p2 in [pair, pair[::-1]]:
                merged.extend([(p | {p1}) - {p2} for p in merged if p2 in p])

            # Add un-reconciled successors to stack.
            stack.extend([p for p in merged if len(p) == 2])

        return True


__all__ = ['APTA', 'Node', 'Word']
