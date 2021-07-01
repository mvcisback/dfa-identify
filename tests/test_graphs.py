from dfa_identify.graphs import APTA

def test_fig1():
    """Example from fig1 in Heule 2010."""
    apta = APTA.from_examples(
        accepting=['a', 'abaa', 'bb'],
        rejecting=['abb', 'b'],
    )
    assert apta.alphabet.keys() == {'a', 'b'}

    assert len(apta.tree.nodes) == 8
    assert len(apta.tree.edges) == 7

    assert len(apta.accepting) == 3
    assert len(apta.rejecting) == 2

    graph = apta.consistency_graph()
    assert len(graph.nodes) == 8
    assert len(graph.edges) == 10
