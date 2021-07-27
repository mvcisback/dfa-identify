from dfa_identify import find_dfa, find_dfas
from dfa_identify.memreps import run_memreps_naive, equivalence_oracle_memreps
from dfa.utils import find_equiv_counterexample, find_subset_counterexample
from dfa import dict2dfa

def test_memreps_naive():
    transition_dict = {1: (True, {'b': 1, 'a': 0}),
                       0: (False, {'a' : 1, 'b': 0})}

    test_system1 = dict2dfa(transition_dict, 1)

    def pref_fxn(word1, word2):
        trace1 = test_system1.trace(word1)
        trace2 = test_system1.trace(word2)
        t1count, t2count = 0, 0
        for state in trace1:
            if state == 0:
                t1count += 1
        for state in trace2:
            if state == 0:
                t2count += 1
        if t1count == t2count:
            return "incomparable"
        elif t1count > t2count:
            return word1, word2
        else:
            return word2, word1

    def membership_fxn(word):
        trace1 = test_system1.trace(word)
        tcount = 0
        for state in trace1:
            if state == 0:
                tcount += 1
        return tcount < 3

    def simple_query_scoring(query, label):
        if label == "preference":
            return 2
        else:
            return 1

    accepting = ['b', 'aa', 'a', 'baab', 'bab', 'bb']
    rejecting = ['aaaaa', 'abb', 'bbabb', 'aaab']

    resulting_dfa = run_memreps_naive(accepting, rejecting, 100, 30,
                                      pref_fxn, membership_fxn, simple_query_scoring)

    aug_accept = ['b', 'aa', 'a', 'ab', 'bbb', 'baab', 'bab']
    aug_reject = ['aaaaa', 'abb', 'bbabb', 'aaab', 'bbbaaaaaaa']

    for x in aug_accept:
        assert resulting_dfa.label(x)

    for x in aug_reject:
        assert not resulting_dfa.label(x)

def test_equivalence_memreps():
    transition_dict = {0: (True, {'a': 1, 'b': 0}),
                       1: (True, {'a' : 4, 'b': 2}),
                       2: (True, {'a' : 5, 'b': 3}),
                       3: (False, {'a' : 3, 'b': 3}),
                       4: (True, {'a' : 2, 'b': 4}),
                       5: (True, {'a' : 3, 'b': 5})}

    true_dfa = dict2dfa(transition_dict, 0)

    def pref_fxn(word1, word2):
        if true_dfa.label(word1) == true_dfa.label(word2):
            return "incomparable"
        elif true_dfa.label(word1):
            return word2, word1
        else:
            return word1, word2

    def membership_fxn(word):
        return true_dfa.label(word)

    def simple_query_scoring(query, label):
        if label == "preference":
            return 2
        else:
            return 1

    def equivalence_fxn(candidate):
        return find_equiv_counterexample(candidate, true_dfa)

    accepting = ['b', 'aa', 'a']
    rejecting = ['aaaaa', 'abb']

    resulting_dfa = equivalence_oracle_memreps(equivalence_fxn, accepting, rejecting,
                                               100, 5, pref_fxn, membership_fxn,
                                               simple_query_scoring, num_equiv_iters=20)
    assert find_equiv_counterexample(true_dfa, resulting_dfa) is None
