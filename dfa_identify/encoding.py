from __future__ import annotations

import inspect
from itertools import product, groupby
from functools import wraps
from typing import Any, Callable, Iterable, Literal, Optional, Union

import attr
from dfa import dict2dfa, DFA
import funcy as fn
import networkx as nx
from networkx.algorithms.approximation.clique import max_clique
from functools import partial

from dfa_identify.graphs import APTA, Node

Nodes = Iterable[Node]
Clauses = Iterable[list[int]]
Encodings = Iterable[Clauses]
SymMode = Optional[Literal['bfs', 'clique']]
Bounds = tuple[Optional[int], Optional[int]]


# =================== Codec : int <-> variable  ====================


def encoder(offset):
    def _encoder(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            bound = sig.bind_partial(self, *args, **kwargs)
            for key, val in bound.arguments.items():
                if key.startswith('color'):
                    assert 0 <= val < self.n_colors
                elif key.startswith('node'):
                    assert 0 <= val < self.n_nodes
                elif key.startswith('token'):
                    assert 0 <= val < self.n_tokens

            base = self.offsets[offset]
            return func(self, *args, **kwargs) + base
        return wrapper
    return _encoder


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class AuxillaryVar:
    idx: int


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


Var = Union[ColorAcceptingVar, ColorNodeVar, ParentRelationVar, AuxillaryVar]


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class Codec:
    n_nodes: int
    n_colors: int
    n_tokens: int
    sym_mode: SymMode
    apta: APTA = None
    extra_clauses: ExtraClauseGenerator = lambda *_: ()

    # Internal
    counts: list = None
    offsets: list = None

    def __attrs_post_init__(self):
        object.__setattr__(self, "counts", (
            self.n_colors,                                    # z
            self.n_colors * self.n_nodes,                     # x
            self.n_tokens * self.n_colors * self.n_colors,    # y
            (self.n_colors * (self.n_colors - 1)) // 2,       # p
            (self.n_colors * (self.n_colors - 1)) // 2,       # t
            (self.n_colors - 1) * self.n_tokens,              # m
        ))
        offsets = [0] + fn.lsums(self.counts)
        object.__setattr__(self, "offsets", tuple(offsets))

    @staticmethod
    def from_apta(apta: APTA,
                  n_colors: int = 0,
                  sym_mode: SymMode = None,
                  extra_clauses: ExtraClauseGenerator=lambda *_: ()) -> Codec:

        return Codec(n_nodes=len(apta.nodes),
                     n_colors=n_colors,
                     n_tokens=len(apta.alphabet),
                     sym_mode=sym_mode,
                     apta=apta,
                     extra_clauses=extra_clauses)

    @encoder(offset=0)
    def color_accepting(self, color: int) -> int:
        """ Literature refers to these variables as z """
        return 1 + color

    @encoder(offset=1)
    def color_node(self, node: int, color: int) -> int:
        """ Literature refers to these variables as x """
        return 1 + self.n_colors * node + color

    @encoder(offset=2)
    def parent_relation(self, token: Any, color1: int, color2: int) -> int:
        """ Literature refers to these variables as y """
        a = self.n_colors
        b = a**2
        return 1 + color1 + a * color2 + b * token

    # --------------------- BFS Sym_Mode Only ---------------------------
    @encoder(offset=3)
    def enumeration_parent(self, color1: int, color2: int) -> int:
        """ Literature refers to these variables as p
        Note: here we use p_{i,j} rather than p_{j,i} """
        assert (color1 < color2), "color1 must be smaller than color2"
        return 1 + (((color2) * (color2 - 1)) // 2) + color1

    @encoder(offset=4)
    def transition_relation(self, color1: int, color2: int) -> int:
        """ Literature refers to these variables as t """
        assert (color1 < color2), "color1 must be smaller than color2"
        return 1 + (((color2) * (color2 - 1)) // 2) + color1

    @encoder(offset=5)
    def enumeration_label(self, token: Any, color: int) -> int:
        """ Literature refers to these variables as m """
        assert color > 0
        return 1 + self.n_tokens * (color - 1) + token

    # -------------------------------------------------------------------

    def decode(self, lit: int) -> Var:
        idx = abs(lit) - 1
        color1, true = idx % self.n_colors, lit > 0
        if idx < self.offsets[1]:
            return ColorAcceptingVar(color1, true)
        elif idx < self.offsets[2]:
            node = (idx - color1) // self.n_colors - 1
            return ColorNodeVar(color1, true, node)
        elif idx < self.offsets[3]:
            tmp = idx - self.n_colors * (1 + self.n_nodes)
            tmp //= self.n_colors
            color2 = tmp % self.n_colors
            token = tmp // self.n_colors
            return ParentRelationVar(color1, color2, token, true)

        return AuxillaryVar(idx)

    def interpret_model(self, model: list[int]) -> DFA:
        # Wrap old API for backwards compat.
        return self.extract_dfa(model)

    def extract_dfa(self, model: list[int]) -> DFA:
        # Fill in don't cares in model.
        decoded = fn.lmap(self.decode, model)
        var_groups = groupby(decoded, type)

        group1 = next(var_groups)
        assert group1[0] == ColorAcceptingVar
        accepting = {v.color for v in group1[1] if v.true}

        group2 = next(var_groups)
        assert group2[0] == ColorNodeVar

        node2color = {}
        for var in group2[1]:
            if not var.true:
                continue
            assert var.node not in node2color
            node2color[var.node] = var.color

        group3 = next(var_groups)
        assert group3[0] == ParentRelationVar
        dfa_dict = {}
        token2char = self.apta.alphabet.inv.get if self.apta else lambda _, x: x
        for var in group3[1]:
            if not var.true:
                continue
            default = (var.parent_color in accepting, {})
            (_, char2node) = dfa_dict.setdefault(var.parent_color, default)
            char = token2char(var.token, var.token)
            assert char not in char2node
            char2node[char] = var.node_color
        dfa_ = dict2dfa(dfa_dict, start=node2color[0])

        return DFA(start=dfa_.start,
                   inputs=dfa_.inputs,
                   outputs=dfa_.outputs,
                   label=dfa_._label,
                   transition=dfa_._transition)

    @property
    def non_stutter_lits(self):
        # Compute parent relation variables that don't stutter.
        lits = []
        for lit in range(1 + self.offsets[2], self.offsets[3] + 1):
            par_rel = self.decode(lit)
            assert isinstance(par_rel, ParentRelationVar)
            if par_rel.node_color == par_rel.parent_color:
                continue
            lits.append(lit)
        return lits

    @property
    def max_id(self):
        return self.offsets[-1]

    def couple_labeling_clauses(self):
        yield from partition_by_accepting_clauses(self, self.apta)

    def clauses(self, cgraph=None, clique=None):
        yield from encode_dfa_id(self.apta, self, cgraph=cgraph, clique=clique)
        yield from self.extra_clauses(self.apta, self)

# ================= Clause Generator =====================

ExtraClauseGenerator = Callable[[APTA, Codec], Clauses]


def onehot_color_clauses(codec: Codec) -> Clauses:
    for n in range(codec.n_nodes):  # Each vertex has at least one color.
        yield [codec.color_node(n, c) for c in range(codec.n_colors)]

    for n in range(codec.n_nodes):  # Each vertex has at most one color.
        for i in range(codec.n_colors):
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


def partition_by_accepting_clauses(codec: Codec, apta: APTA) -> Clauses:
    for c in range(codec.n_colors):
        lit = codec.color_accepting(c)
        yield from ([-codec.color_node(n, c), lit] for n in apta.accepting)
        yield from ([-codec.color_node(n, c), -lit] for n in apta.rejecting)


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
    for color, node in enumerate(clique):
        yield [codec.color_node(node, color)]


def symmetry_breaking_common(codec: Codec) -> Clauses:
    """
    Symmetry breaking clauses for both DFS and BFS
    See Ulyantsev 2016.
    """
    # Ensures start vertex is 0 - not listed in Ulyantsev
    yield [codec.color_node(0, 0)]

    for color2 in range(codec.n_colors):
        if color2 > 0:
            yield [
                codec.enumeration_parent(color1, color2)
                for color1 in range(color2)
            ]  # 4
        for color1 in range(color2):
            p = codec.enumeration_parent(color1, color2)
            t = codec.transition_relation(color1, color2)
            m = partial(codec.enumeration_label, color=color2)
            y = partial(codec.parent_relation, color1=color1, color2=color2)

            yield [-t] + [y(token) for token in range(codec.n_tokens)]  # 1
            yield [t, -p]  # 3

            for token2 in range(codec.n_tokens):
                yield [t, -y(token2)]  # 2
                yield [-p, -m(token2), y(token2)]  # 5
                yield [-y(token2), -p, m(token2)] + \
                    [y(token1) for token1 in range(token2)]  # 7

                for token1 in range(token2):
                    yield [-p, -m(token2), -y(token1)]  # 6


def symmetry_breaking_bfs(codec: Codec) -> Clauses:
    """
    Symmetry breaking clauses for BFS
    See Ulyantsev 2016.
    """
    for color2 in range(codec.n_colors):
        t_2 = partial(codec.transition_relation, color2=color2)
        for color1 in range(color2):
            p12 = codec.enumeration_parent(color1, color2)
            t12 = codec.transition_relation(color1, color2)

            yield from [[-p12, -t_2(color3)] for color3 in range(color1)]  # 12
            yield [-t12, p12] + [t_2(color3) for color3 in range(color1)]  # 13

            if color2 + 1 >= codec.n_colors:
                continue

            for color3 in range(color1):  # 14
                yield [-p12, -codec.enumeration_parent(color3, color2 + 1)]

            for token2 in range(codec.n_tokens):
                for token1 in range(token2):
                    yield [
                        -p12,
                        -codec.enumeration_parent(color1, color2 + 1),
                        -codec.enumeration_label(token2, color2),
                        -codec.enumeration_label(token1, color2 + 1),
                    ]  # 15


def encode_dfa_id(apta,
                  codec,
                  cgraph=None,
                  clique=None):
    # Clauses from Table 1.                                      rows
    yield from onehot_color_clauses(codec)                      # 1, 5
    yield from codec.couple_labeling_clauses()                  # 2
    yield from colors_parent_rel_coupling_clauses(codec, apta)  # 3, 7
    yield from onehot_parent_relation_clauses(codec)            # 4, 6

    if cgraph:
        # Disabled when conflict graph not generated, e.g.,
        # DFA decompositions.
        yield from determination_conflicts(codec, cgraph)       # 8

    if clique and codec.sym_mode == "clique":
        yield from symmetry_breaking(codec, clique)
    elif codec.sym_mode == "bfs":
        yield from symmetry_breaking_common(codec)
        yield from symmetry_breaking_bfs(codec)


__all__ = ['Codec', 'Bounds', 'ExtraClauseGenerator']
