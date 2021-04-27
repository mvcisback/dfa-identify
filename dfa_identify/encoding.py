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
        yield [codec.color_node(n, c) for c in range(code.n_colors)]

    for n in range(codec.n_nodes):  # Each vertex has at most one color.
        for i in range(codec.n_colors):
            lit1 = codec.color_node(n, i)
            for j in range(i + 1, codec.n_colors):
                lit2 = codec.color_node(n, i)
                yield [-lit1, -lit2]


def accepting_partition_clauses(apta: APTA, codec: Codec) -> Clauses:
    pass


def encode_dfa_id(apta: APTA, n_colors: int = 1):
    codec = Codec(len(apta.nodes), n_colors)

    yield from onehot_color_clauses(codec)
