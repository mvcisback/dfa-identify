import dfa
from dfa_identify import find_dfa_active


def test_parity():
    def oracle(word):
        return sum(word) % 2 == 0

    lang = find_dfa_active(alphabet=[0, 1],
                           oracle=oracle,
                           n_queries=20)
    assert lang == dfa.DFA(
        inputs=[0,1],
        label=lambda s: s,
        transition=lambda s, c: s ^ bool(c),
        start=True
    )

    lang2 = find_dfa_active(alphabet=[0, 1],
                           positive=[(0,), (0,0)],
                           negative=[(1,), (1,0)],
                           oracle=oracle,
                           n_queries=20)
    assert lang == lang2
