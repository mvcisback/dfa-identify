import pytest

import dfa
from more_itertools import take

from dfa_identify import find_dfa, find_dfas


def test_identify():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']

    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)

    assert len(my_dfa.states()) == 3

    accepting = [[0], [0, 'z', 0, 0], ['z', 'z']]
    rejecting = [[0, 'z', 'z'], ['z']]

    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting, bounds=(3, 10))

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)

    assert len(my_dfa.states()) == 3


def test_identify_repeatedly():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']

    for i in range(3):
        my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

        # check that dfa found matches
        for x in accepting:
            assert my_dfa.label(x)

        for x in rejecting:
            assert not my_dfa.label(x)

        # check that minimal dfa is found
        assert len(my_dfa.states()) == 3


def test_unique():
    for sym_mode in ['bfs', 'clique']:
        dfas = list(find_dfas(
            accepting=['a', 'ab'],
            rejecting=['', 'b', 'aa'],
            sym_mode=sym_mode,
        ))
        assert len(dfas) == 1


def test_enumerate():
    for sym_mode in ['bfs', 'clique']:
        dfas = list(find_dfas(
            accepting=['a'],
            rejecting=['', 'b'],
            sym_mode=sym_mode,
        ))
        assert len(dfas) == 4


def test_overlapping_examples():
    pos = [[False, True]]
    neg = [[False], [False, True]]
    my_dfa = find_dfa(accepting=pos, rejecting=neg)
    assert my_dfa is None


def test_identify_ns_edges():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']

    my_dfa = find_dfa(
        accepting=accepting,
        rejecting=rejecting,
        order_by_stutter=True,
    )

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)

    assert len(my_dfa.states()) == 3

    my_dfa = find_dfa(
        order_by_stutter=True,
        accepting=[
            ('yellow',),
            ('yellow', 'yellow'),
        ],
        rejecting=[
            (), ('red',), ('red', 'red'),
            ('red', 'yellow'), ('yellow', 'red'),
            ('yellow', 'yellow', 'red'),
        ]
    )
    assert len(my_dfa.states()) == 3
    graph, _ = dfa.dfa2dict(my_dfa)
    count = 0
    for s1, (_, transitions) in graph.items():
        count += sum(s1 != s2 for s2 in transitions.values())
    assert count == 3

    my_dfa = find_dfa(
        order_by_stutter=True,
        accepting=[
            (0, 0, 0, 1)
        ],
        rejecting=[
            (0, 0, 0, 0),
        ]
    )


def test_order_by_stutter():
    examples = [
        (['x'], []),
        ([], ['y']),
        (['a'], ['', 'b']),
    ]

    for accepting, rejecting in examples:
        unordered = list(find_dfas(
            accepting=accepting,
            rejecting=rejecting,
            order_by_stutter=False,
        ))

        ordered = list(find_dfas(
            accepting=accepting,
            rejecting=rejecting,
            order_by_stutter=True,
        ))

        assert len(ordered) == len(unordered)

        def non_stutter_count(x):
            graph, _ = dfa.dfa2dict(x)
            count = 0
            for s1, (_, transitions) in graph.items():
                count += sum(s1 != s2 for s2 in transitions.values())
            return count

        ordered_counts = list(map(non_stutter_count, ordered))
        unordered_counts = list(map(non_stutter_count, unordered))

        assert set(ordered_counts) == set(unordered_counts)
        assert ordered == sorted(ordered, key=non_stutter_count)

def test_identify_preferences():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']

    ordered_preference_words = [("bb", "aba"), ("ab", "b"), ("b", "a")]
    equiv_preference_words = [("abb", "abbb")]
    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting, ordered_preference_words=ordered_preference_words
                      ,equivalent_preference_words=equiv_preference_words)
    true_accepting = ['a', 'abaa', 'bb', "aba"]
    true_rejecting = ["abb", "b", "abbb", "ab"]
    for x in true_accepting:
        assert my_dfa.label(x)

    for x in true_rejecting:
        assert not my_dfa.label(x)


def test_empty_examples():
    with pytest.raises(ValueError):
        next(find_dfas(accepting=[], rejecting=[]))

    dfas = take(4, find_dfas(accepting=[], rejecting=[], alphabet={'x', 'y'}))
    assert len(dfas) == 2
    for i, dfa in enumerate(dfas):
        assert dfa.label(()) != (i & 1)
