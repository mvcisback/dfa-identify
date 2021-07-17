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

    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)



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

def test_enumerate_preferences():
    for sym_mode in ['bfs', 'clique']:
        dfas = list(find_dfas(
            accepting=['a'],
            rejecting=['', 'b'],
            ordered_preference_words=[("ab", "aa")],
            incomparable_preference_words=[("baa", "aaa")],
            sym_mode=sym_mode,
        ))
        assert len(dfas) == 2

def test_identify_preferences():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']

    ordered_preference_words = [("bb", "aba"), ("ab", "b"), ("b", "a")]
    incomparable_preference_words = [("abb", "abbb")]
    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting, ordered_preference_words=ordered_preference_words
                      ,incomparable_preference_words=incomparable_preference_words)

    true_accepting = ['a', 'abaa', 'bb', "aba"]
    true_rejecting = ["abb", "b", "abbb", "ab"]
    for x in true_accepting:
        assert my_dfa.label(x)

    for x in true_rejecting:
        assert not my_dfa.label(x)

