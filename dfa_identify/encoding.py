from __future__ import annotations

from itertools import product
from typing import Any, NamedTuple, Iterable, Tuple

import attr
import funcy as fn
import networkx as nx
from networkx.algorithms.approximation.clique import max_clique

from dfa_identify.graphs import APTA, Node


Nodes = Iterable[Node]
Clauses = Iterable[list[int]]


# =================== Codec : int <-> variable  ====================


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class ColorAcceptingVar:
    color: int
    true: bool


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class ColorNodeVar:
    color: int
    true: bool
    node: int


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class ParentRelationVar:
    parent_color: int
    node_color: int
    token: int
    true: bool


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class Codec:
    n_nodes: int
    n_colors: int
    n_tokens: int

    @staticmethod
    def from_apta(apta: APTA, n_colors: int = 0) -> Codec:
        return Codec(len(apta.nodes), n_colors, len(apta.alphabet))

    def color_accepting(self, color: int) -> int:  # get color var literal
        return 1 + color

    def color_node(self, node: int, color: int) -> int:  # get literal of node x with color y
        return 1 + self.n_colors * (1 + node) + color

    def parent_relation(self, token: Any, color1: int, color2: int) -> int:  # get literal of relation var between
        #  2 colors
        a = self.n_colors
        b = a**2
        c = 1 + self.n_colors * (1 + self.n_nodes)
        return color1 + a * color2 + b * token + c

    def decode(self, lit: int) -> Var:
        idx = abs(lit) - 1
        color1, true = idx % self.n_colors, lit > 0
        kind_idx = idx // self.n_colors
        if kind_idx == 0:
            return ColorAcceptingVar(color1, true)
        elif 1 <= kind_idx <= self.n_nodes:
            node = (idx - color1) // self.n_colors - 1
            return ColorNodeVar(color1, true, node)
        tmp = idx - self.n_colors * (1 + self.n_nodes)
        tmp //= self.n_colors
        color2 = tmp % self.n_colors
        token = tmp // self.n_colors
        return ParentRelationVar(color1, color2, token, true)
            


# ================= Clause Generator =====================


def dfa_id_encodings(apta: APTA) -> Iterable[Clauses]:
    cgraph = apta.consistency_graph()
    clique = max_clique(cgraph)

    for n_colors in range(len(clique), len(apta.nodes) + 1):
        codec = Codec.from_apta(apta, n_colors)
        yield codec, list(encode_dfa_id(apta, codec, clique, cgraph))


def encode_dfa_id(apta, codec, clique, cgraph):
    # Clauses from Table 1.                                      rows
    yield from onehot_color_clauses(codec)                     # 1, 5
    yield from partition_by_accepting_clauses(codec, apta)     # 2 will be adapted for preferences
    yield from colors_parent_rel_coupling_clauses(codec, apta) # 3, 7
    yield from onehot_parent_relation_clauses(codec)           # 4, 6
    yield from determination_conflicts(codec, cgraph)          # 8
    yield from symmetry_breaking(codec, clique)


def onehot_color_clauses(codec: Codec) -> Clauses:
    for n in range(codec.n_nodes):  # Each vertex has at least one color.
        yield [codec.color_node(n, c) for c in range(codec.n_colors)]

    for n in range(codec.n_nodes):  # Each vertex has at most one color.
        for i in range(codec.n_colors): # if it has one color, it can't have any others
            lit = codec.color_node(n, i)
            for j in range(i + 1, codec.n_colors):  # i < j
                yield [-lit, -codec.color_node(n, j)]


def tokensXcolors(codec: Codec):
    return product(range(codec.n_tokens), range(codec.n_colors))


def onehot_parent_relation_clauses(codec: Codec) -> Clauses:
    # Each parent relation must target at least one color.
    for token, i in tokensXcolors(codec):
        colors = range(codec.n_colors)
        yield [codec.parent_relation(token, i, j) for j in colors]

    # Each parent relation can target at most one color.
    for token, i in tokensXcolors(codec):
        for h in range(codec.n_colors):
            lit1 = codec.parent_relation(token, i, h)
            for j in range(h + 1, codec.n_colors):  # h < j
                yield [-lit1, -codec.parent_relation(token, i, j)]

#modify for preferences
def partition_by_accepting_clauses(codec: Codec, apta: APTA) -> Clauses:
    for c in range(codec.n_colors):
        lit = codec.color_accepting(c)
        yield from ([-codec.color_node(n, c), lit] for n in apta.accepting)
        yield from ([-codec.color_node(n, c), -lit] for n in apta.rejecting)
        # encode the ordering constraints on preferences (equation 6 in memreps)
        for c2 in range(codec.n_colors):
            lit2 = codec.color_accepting(c2)
            # acceptance on LHS leads to acceptance on RHS, rejection on RHS leads to rejection on LHS
            yield from ([-codec.color_node(np, c2), -codec.color_node(nl, c), -lit2, lit] for nl, np in apta.ordered_preferences)
            yield from ([-codec.color_node(np, c2), -codec.color_node(nl, c), -lit, lit2] for nl, np in apta.ordered_preferences)

            # encode the equality constraints on incomparable preferences
            # for either accepting or rejecting colors, these clauses should encode equality
            yield from ([-codec.color_node(np, c2), -lit, lit2] for nl, np in apta.incomparable_preferences)
            yield from ([-codec.color_node(np, c2), -lit2, lit] for nl, np in apta.incomparable_preferences)
            yield from ([-codec.color_node(nl, c), -lit, lit2] for nl, np in apta.incomparable_preferences)
            yield from ([-codec.color_node(nl, c), -lit2, lit] for nl, np in apta.incomparable_preferences)



            #yield from ([-codec.color_node(nl, c), lit, codec.color_node(np, c2), -lit2] for nl, np in apta.incomparable_preferences)



# couples transitions
def colors_parent_rel_coupling_clauses(codec: Codec, apta: APTA) -> Clauses:
    colors = range(codec.n_colors)
    rev_tree = apta.tree.reverse()
    non_root_nodes = set(apta.nodes) - {0}     # Root doesn't have a parent.
    for node, i, j in product(non_root_nodes, colors, colors):
        parent, *_ = rev_tree.neighbors(node)  # only have 1 parent.
        token = apta.alphabet[apta.nodes[node]['source']]

        parent_color = codec.color_node(parent, i)
        node_color = codec.color_node(node, j)
        parent_rel = codec.parent_relation(token, i, j)

        # Parent relation and node color coupled throuh parent color.
        yield [-parent_color, -node_color, parent_rel]  # 3
        yield [-parent_color, node_color, -parent_rel]  # 7


def determination_conflicts(codec: Codec, cgraph: nx.Graph) -> Clauses:
    colors = range(codec.n_colors)
    for (n1, n2), c in product(cgraph.edges, colors):
        yield [-codec.color_node(n1, c), -codec.color_node(n2, c)]


def symmetry_breaking(codec: Codec, clique: Nodes) -> Clauses:
    for node, color in enumerate(clique):
        yield [codec.color_node(node, color)]


__all__ = ['Codec', 'dfa_id_encodings']
