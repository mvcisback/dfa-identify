from dfa_identify import find_dfa


def test_identify():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']
    
    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)


    accepting = [[0], [0, 'z', 0, 0], ['z', 'z']]
    rejecting = [[0, 'z', 'z'], ['z']]
    
    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)

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

