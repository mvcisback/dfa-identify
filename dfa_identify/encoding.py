from __future__ import annotations

from itertools import product
from typing import Any, NamedTuple, Iterable

import attr

from dfa_identify.graphs import APTA, Node


class Var(NamedTuple):
    color: int
    kind: str
    idx: int
    true: bool = True


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class Codec:
    n_nodes: int
    n_colors: int
    n_tokens: int

    @staticmethod
    def from_apta(apta: APTA, n_colors: int = 0) -> Codec:
        return Codec(len(apta.nodes), n_colors, len(apta.alphabet))

    def decode(self, lit: int) -> Var:
        idx = abs(lit) - 1
        kind_idx = idx // self.n_colors        
        if kind_idx == 0:
            kind = "color_accepting"
        elif 1 <= kind_idx <= self.n_nodes:
            kind = "color_node"
        else:
            kind = "parent_relation"
        return Var(idx % self.n_colors, kind, idx, lit > 0)

    def color_accepting(self, color: int) -> int:
        return 1 + color

    def color_node(self, node: int, color: int) -> int:
        return 1 + self.n_colors * (1 + node) + color

    def parent_relation(self, token: Any, color1: int, color2: int) -> int:
        a = self.n_colors
        b = a**2
        c = 1 + self.n_colors * (1 + self.n_nodes)
        return color1 + a * color2 + b * token + c


# ================= Clause Generator =====================


Clauses = Iterable[list[int]]


def onehot_color_clauses(codec: Codec) -> Clauses:
    for n in range(codec.n_nodes):  # Each vertex has at least one color.
        yield [codec.color_node(n, c) for c in range(codec.n_colors)]

    for n in range(codec.n_nodes):  # Each vertex has at most one color.
        for i in range(codec.n_colors):
            lit = codec.color_node(n, i)
            for j in range(i + 1, codec.n_colors):  # i < j
                yield [-lit, -codec.color_node(n, i)]


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


def partition_by_accepting_clauses(apta: APTA, codec: Codec) -> Clauses:
    for c in range(codec.n_colors):
        lit = codec.color_accepting(c)
        yield from ([-codec.color_node(n, c), lit] for n in apta.accepting)
        yield from ([-codec.color_node(n, c), -lit] for n in apta.rejecting)


def color_sets_parent_rel_clauses(apta, codec):
    colors = range(codec.n_colors)
    rev_tree = apta.tree.reverse()
    non_root_nodes = set(apta.nodes) - {0}     # Root doesn't have a parent.
    for node, i, j in product(non_root_nodes, colors, colors):
        parent, *_ = rev_tree.neighbors(node)  # only have 1 parent.
        token = apta.alphabet[apta.nodes[node]['source']]
        yield [
            codec.parent_relation(token, i, j),
            -codec.color_node(parent, i),
            -codec.color_node(node, j),
        ]


def encode_dfa_id(apta: APTA, n_colors: int = 1):
    codec = Codec.from_apta(apta, n_colors)

    # Clauses from Table 1.                                    rows
    yield from onehot_color_clauses(codec)                   # 1, 5
    yield from partition_by_accepting_clauses(apta, codec)   # 2
    yield from color_sets_parent_rel_clauses(apta, codec)    # 3
    yield from onehot_parent_relation_clauses(codec)         # 4, 6
