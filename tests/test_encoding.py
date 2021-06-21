from itertools import product

from dfa_identify.graphs import APTA
from dfa_identify.encoding import Codec, dfa_id_encodings
from dfa_identify.encoding import (
    ColorAcceptingVar,
    ColorNodeVar,
    ParentRelationVar
)


def kind(var):
    if isinstance(var, ColorAcceptingVar):
        return 'color_accepting'
    elif isinstance(var, ColorNodeVar):
        return 'color_node'
    elif isinstance(var, ParentRelationVar):
        return 'parent_relation'


def test_codec():
    codec = Codec(n_nodes=10, n_colors=3, n_tokens=4)
    assert kind(codec.decode(1)) == 'color_accepting'
    assert kind(codec.decode(3)) == 'color_accepting'
    assert kind(codec.decode(4)) == 'color_node'
    assert kind(codec.decode(33)) == 'color_node'
    assert kind(codec.decode(34)) == 'parent_relation'

    colors = range(codec.n_colors)
    nodes = range(codec.n_nodes)
    tokens = range(codec.n_tokens)

    # Test bijection on color accepting vars.
    lits = set()
    for c in colors:
        lit = codec.color_accepting(c)
        lits.add(lit)
        var = codec.decode(lit)
        assert kind(var) == 'color_accepting'
        assert var.color == c
    assert len(lits) == 3

    # Test bijection on color nodes.
    lits = set()
    for n, c in product(nodes, colors):
        lit = codec.color_node(n, c)
        lits.add(lit)
        var = codec.decode(lit)
        assert kind(var) == 'color_node'
        assert var.color == c
        assert var.node == n
    assert len(lits) == 3 * 10  # Check bijection.

    # Test bijection on parent relation vars.
    lits = set()
    for t, c1, c2 in product(tokens, colors, colors):
        lit = codec.parent_relation(t, c1, c2)
        lits.add(lit)
        var = codec.decode(lit)
        assert kind(var) == 'parent_relation'
        assert var.parent_color == c1
        assert var.node_color == c2
        assert var.token == t

    assert len(lits) == 9 * 4  # Check bijection.
    assert max(lits) == 3 + 3 * 10 + 9 * 4


def test_encode_dfa_id():
    apta = APTA.from_examples(
        accepting=['a', 'abaa', 'bb'],
        rejecting=['abb', 'b'],
    )

    encodings = dfa_id_encodings(apta)
    clauses1 = next(encodings)[1]
    clauses2 = next(encodings)[1]
    clauses3 = next(encodings)[1]
    assert 1 < (len(clauses2) / len(clauses1)) < 3
    assert 1 < (len(clauses3) / len(clauses2)) < 3
