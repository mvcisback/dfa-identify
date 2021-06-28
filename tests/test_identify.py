from dfa_identify import find_dfa


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

    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

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
