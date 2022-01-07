from __future__ import annotations

import inspect
from itertools import product
from functools import wraps
from typing import Any, Callable, Iterable, Literal, Optional, Union

import attr
import funcy as fn
import networkx as nx
from networkx.algorithms.approximation.clique import max_clique
from functools import partial

from dfa_identify.graphs import APTA, Node

Nodes = Iterable[Node]
Clauses = Iterable[list[int]]
Encodings = Iterable[Clauses]
SymMode = Optional[Literal['bfs', 'clique']]


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

@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class AcceptingIncVar:
    observation_number: int
    true: bool

@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class RejectingIncVar:
    observation_number: int
    true: bool

@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class OrderedIncVar:
    observation_number: int
    true: bool

@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class EquivalentIncVar:
    observation_number: int
    true: bool

Var = Union[ColorAcceptingVar, ColorNodeVar, ParentRelationVar, AuxillaryVar,
            AcceptingIncVar, RejectingIncVar, OrderedIncVar, EquivalentIncVar]


@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class Codec:
    n_nodes: int
    n_colors: int
    n_tokens: int
    n_accepting: int
    n_rejecting: int
    n_ordered: int
    n_equivalent: int
    error_variables: Iterable[int]
    sym_mode: SymMode

    def __attrs_post_init__(self):
        object.__setattr__(self, "counts", (
            self.n_colors,                                    # z
            self.n_colors * self.n_nodes,                     # x
            self.n_tokens * self.n_colors * self.n_colors,    # y
            self.n_accepting,                                 # a
            self.n_rejecting,                                 # r
            self.n_ordered,                                   # o
            self.n_equivalent,                                # e
            (self.n_colors * (self.n_colors - 1)) // 2,       # p
            (self.n_colors * (self.n_colors - 1)) // 2,       # t
            (self.n_colors - 1) * self.n_tokens,              # m
        ))
        object.__setattr__(self, "offsets", tuple([0] + fn.lsums(self.counts)))

    @staticmethod
    def from_apta(apta: APTA,
                  n_colors: int = 0,
                  sym_mode: SymMode = None) -> Codec:
        error_vars = list(range(len(apta.nodes) + n_colors + len(apta.alphabet) + 1,
                                len(apta.accepting) + len(apta.rejecting) + len(apta.ordered_preferences) + len(apta.equivalent_preferences) + 1))
        return Codec(len(apta.nodes), n_colors, len(apta.alphabet), len(apta.accepting),
                     len(apta.rejecting), len(apta.ordered_preferences),
                     len(apta.equivalent_preferences), error_vars, sym_mode)

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

    # -------------------------------------------------------------------

    @encoder(offset=3)
    def accepting_inconsistency_var(self, accepting_idx: int) -> int:
        return 1 + accepting_idx

    @encoder(offset=4)
    def rejecting_inconsistency_var(self, rejecting_idx: int) -> int:
        return 1 + rejecting_idx

    @encoder(offset=5)
    def ordered_inconsistency_var(self, ordered_idx: int) -> int:
        return 1 + ordered_idx

    @encoder(offset=6)
    def equivalent_inconsistency_var(self, equivalent_idx: int) -> int:
        return 1 + equivalent_idx
    # -------------------------------------------------------------------

    # --------------------- BFS Sym_Mode Only ---------------------------
    @encoder(offset=7)
    def enumeration_parent(self, color1: int, color2: int) -> int:
        """ Literature refers to these variables as p
        Note: here we use p_{i,j} rather than p_{j,i} """
        assert (color1 < color2), "color1 must be smaller than color2"
        return 1 + (((color2) * (color2 - 1)) // 2) + color1

    @encoder(offset=8)
    def transition_relation(self, color1: int, color2: int) -> int:
        """ Literature refers to these variables as t """
        assert (color1 < color2), "color1 must be smaller than color2"
        return 1 + (((color2) * (color2 - 1)) // 2) + color1

    @encoder(offset=9)
    def enumeration_label(self, token: Any, color: int) -> int:
        """ Literature refers to these variables as m """
        assert color > 0
        return 1 + self.n_tokens * (color - 1) + token



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
        elif idx < self.offsets[4]:
            return AcceptingIncVar(idx, true)
        elif idx < self.offsets[5]:
            return RejectingIncVar(idx, true)
        elif idx < self.offsets[6]:
            return OrderedIncVar(idx, true)
        elif idx < self.offsets[7]:
            return EquivalentIncVar(idx, true)

        return AuxillaryVar(idx)


# ================= Clause Generator =====================

Bounds = tuple[Optional[int], Optional[int]]
ExtraClauseGenerator = Callable[[APTA, Codec], Clauses]


def dfa_id_encodings(
        apta: APTA,
        sym_mode: SymMode = None,
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None)
        ) -> Encodings:
    """Iterator of codecs and clauses for DFAs of increasing size."""
    cgraph = apta.consistency_graph()
    clique = max_clique(cgraph)
    max_needed = len(apta.nodes)

    low, high = bounds

    # Tighten lower bound.
    if low is None:
        low = 1
    low = max(low, len(clique))

    if (low > max_needed) and ((high is None) or (low < high)):
        high = low  # Will find something at low if one exists.
    elif high is None:
        high = max_needed
    else:
        high = min(high, max_needed)

    if high < low:
        raise ValueError('Empty bound range!')

    for n_colors in range(low, high + 1):
        codec = Codec.from_apta(apta, n_colors, sym_mode=sym_mode)

        clauses = list(encode_dfa_id(apta, codec, cgraph, clique))
        clauses.extend(list(extra_clauses(apta, codec)))

        yield codec, clauses


def encode_dfa_id(apta, codec, cgraph, clique=None):
    # Clauses from Table 1.                                      rows
    yield from onehot_color_clauses(codec)                      # 1, 5
    yield from partition_by_accepting_clauses(codec, apta)      # 2
    yield from colors_parent_rel_coupling_clauses(codec, apta)  # 3, 7
    yield from onehot_parent_relation_clauses(codec)            # 4, 6
    yield from determination_conflicts(codec, cgraph)           # 8
    if codec.sym_mode == "clique":
        yield from symmetry_breaking(codec, clique)
    elif codec.sym_mode == "bfs":
        yield from symmetry_breaking_common(codec)
        yield from symmetry_breaking_bfs(codec)
    yield from preference_clauses(codec, apta)


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
        yield from ([codec.accepting_inconsistency_var(idx),
                     -codec.color_node(n, c), lit] for idx, n in enumerate(apta.accepting))
        yield from ([codec.rejecting_inconsistency_var(idx),
                     -codec.color_node(n, c), -lit] for idx, n in enumerate(apta.rejecting))

def preference_clauses(codec: Codec, apta: APTA) -> Clauses:
    for c in range(codec.n_colors):
        less_pref_color_lit = codec.color_accepting(c)
        # encode the ordering constraints on preferences (equation 6 in memreps)
        for c2 in range(codec.n_colors):
            more_pref_color_lit = codec.color_accepting(c2)
            # acceptance on LHS leads to acceptance on RHS, rejection on RHS leads to rejection on LHS
            yield from ([codec.ordered_inconsistency_var(idx), -codec.color_node(np, c2), -codec.color_node(nl, c),
                         more_pref_color_lit, -less_pref_color_lit]
                        for idx, (nl, np) in enumerate(apta.ordered_preferences))

            # encode the equality constraints on incomparable preferences
            # for either accepting or rejecting colors, these clauses should encode equality
            yield from ([codec.equivalent_inconsistency_var(idx), -codec.color_node(np, c2), -codec.color_node(nl, c),
                         -less_pref_color_lit, more_pref_color_lit]
                        for idx, (nl, np) in enumerate(apta.equivalent_preferences))
            yield from ([codec.equivalent_inconsistency_var(idx), -codec.color_node(np, c2), -codec.color_node(nl, c),
                         less_pref_color_lit, -more_pref_color_lit]
                        for idx, (nl, np) in enumerate(apta.equivalent_preferences))

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


__all__ = ['Codec', 'dfa_id_encodings', 'Bounds', 'ExtraClauseGenerator']
