from dfa_identify import find_dfa, find_dfas
from dfa_identify.memreps import run_memreps_naive
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
            if state == "0":
                t1count += 1
        for state in trace2:
            if state == "0":
                t2count += 1
        if t1count == t2count:
            return "incomparable"
        elif t1count > t2count:
            return t1count, t2count
        else:
            return t2count, t1count

    def membership_fxn(word):
        trace1 = test_system1.trace(word)
        tcount = 0
        for state in trace1:
            if state == "0":
                tcount += 1
        return tcount < 3

    def simple_query_scoring(query):
        if type(query) == tuple:
            return 2
        else:
            return 1

    accepting = ['b', 'aa', 'a']
    rejecting = ['aaaaa', 'abb']

    resulting_dfa = run_memreps_naive(accepting, rejecting, 15, 5,
                                      pref_fxn, membership_fxn, simple_query_scoring)

    aug_accept = ['b', 'aa', 'a', 'ab', 'bbb', 'baab', 'bab']
    aug_reject = ['aaaaa', 'abb', 'bbabb', 'aaab', 'bbbaaaaaaa']

    for x in aug_accept:
        assert resulting_dfa.label(x)

    for x in aug_reject:
        assert not resulting_dfa.label(x)
