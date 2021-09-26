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
