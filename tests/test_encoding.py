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
    codec = Codec(n_nodes=10, n_colors=3, n_tokens=4, symm_mode="bfs")
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

def test_symm_break():
    codec = Codec(n_nodes=10, n_colors=3, n_tokens=4, symm_mode="bfs")
    assert kind(codec.decode(1)) == 'color_accepting'
    assert kind(codec.decode(3)) == 'color_accepting'
    assert kind(codec.decode(4)) == 'color_node'
    assert kind(codec.decode(33)) == 'color_node'
    assert kind(codec.decode(34)) == 'parent_relation'

    p = [codec.enumeration_parent(i,j) for i,j in product(range(codec.n_colors), range(codec.n_colors)) if i < j]
    t = [codec.transition_relation(i,j) for i,j in product(range(codec.n_colors), range(codec.n_colors)) if i < j]
    m = [codec.enumeration_label(l,i) for l,i in product(range(codec.n_tokens), range(codec.n_colors))]
    assert len(p) == 3
    assert len(t) == 3
    assert len(m) == 12
    for i in range(70, 72):
        assert i in p
    for i in range(73, 75):
        assert i in t
    for i in range(76, 87):
        assert i in m


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

def check_validation(func, *args):
    try:
        func(*args)
        assert False, "error not caught!"
    except AssertionError as msg:
        if msg.args[0].startswith("error not caught!"):
            raise msg
        else:
            print(msg)

def test_codec_errors():
    """check that codec performs checks on token and colors being within range"""
    codec = Codec(n_nodes=10, n_colors=3, n_tokens=4, symm_mode="bfs")
    check_validation(codec.color_accepting, -1)
    check_validation(codec.color_accepting, 3)
    check_validation(codec.color_node, -1, 2)
    check_validation(codec.color_node, 10, 0)
    check_validation(codec.color_node, 9, 3)
    check_validation(codec.parent_relation, -1, 2, 2)
    check_validation(codec.parent_relation, 4, 1, 2)
    check_validation(codec.enumeration_parent, 2, 2)
    check_validation(codec.enumeration_parent, 1, 3)
    check_validation(codec.transition_relation, 2, 2)
    check_validation(codec.transition_relation, 1, 3)
    check_validation(codec.enumeration_label, -1, 2)
    check_validation(codec.enumeration_label, 4, 1)
    
    
    