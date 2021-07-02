from dfa_identify import find_dfa
from dfa_identify.distinguish_utils import get_augmented_set

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

def test_identify_incomparable():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']

    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    # now, let's try to synthesize an incomparable DFA
    augmented_accepting, augmented_rejecting = get_augmented_set(my_dfa, accepting, rejecting)
    incomparable_dfa = find_dfa(accepting=accepting, rejecting=rejecting,
                                augmented_original_accepting=augmented_accepting,
                                augmented_original_rejecting=augmented_rejecting)
    for x in accepting:
        assert my_dfa.label(x)
        assert incomparable_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)
        assert not my_dfa.label(x)

    toggled_acc_flag = False
    for xa in augmented_accepting:
        assert my_dfa.label(xa)
        if not incomparable_dfa.label(xa):
            toggled_acc_flag = True
    assert toggled_acc_flag

    toggled_rej_flag = False
    for xa in augmented_rejecting:
        assert not my_dfa.label(xa)
        if incomparable_dfa.label(xa):
            toggled_rej_flag = True
    assert toggled_rej_flag

