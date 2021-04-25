from collections import Counter

from dfa_identify.graphs import APTA


def test_fig1():
    """Example from fig1 in Heule 2010."""
    apta = APTA.from_examples({
        True: ['a', 'abaa', 'bb'],
        False: ['abb', 'b'],
    })

    assert len(apta.tree.nodes) == 8
    assert len(apta.tree.edges) == 7

    nodes = apta.tree.nodes
    label_counts = Counter(data.get('label') for _, data in nodes(data=True))

    assert label_counts[True] == 3
    assert label_counts[False] == 2
    assert label_counts[None] == 3

    # graph = apta.consistency_graph()
    # assert len(graph.nodes) == 8
    # assert len(graph.edges) == 9
