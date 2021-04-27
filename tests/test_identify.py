from dfa_identify.identify import find_dfa


def test_identify():
    model = find_dfa(
        accepting=['a', 'abaa', 'bb'],
        rejecting=['abb', 'b'],
    )
