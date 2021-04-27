from itertools import product

from dfa_identify.graphs import APTA
from dfa_identify.encoding import Codec, dfa_id_encodings


def test_codec():
    codec = Codec(n_nodes=10, n_colors=3, n_tokens=4)
    assert codec.decode(1).kind == 'color_accepting'
    assert codec.decode(3).kind == 'color_accepting'
    assert codec.decode(4).kind == 'color_node'
    assert codec.decode(33).kind == 'color_node'
    assert codec.decode(34).kind == 'parent_relation'

    colors = range(codec.n_colors)
    nodes = range(codec.n_nodes)
    tokens = range(codec.n_tokens)

    lits = set()
    for c in colors:
        lit = codec.color_accepting(c)
        lits.add(lit)
        assert codec.decode(lit).kind == 'color_accepting'
    assert len(lits) == 3  # Check bijection.

    lits = set()
    for n, c in product(nodes, colors):
        lit = codec.color_node(n, c)
        lits.add(lit)
        assert codec.decode(lit).kind == 'color_node'
    assert len(lits) == 3 * 10  # Check bijection.

    lits = set()
    for t, c1, c2 in product(tokens, colors, colors):
        lit = codec.parent_relation(t, c1, c2)
        lits.add(lit)
        assert codec.decode(lit).kind == 'parent_relation'
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
