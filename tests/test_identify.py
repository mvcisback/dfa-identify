from dfa_identify import find_dfa
from dfa.draw import write_dot
import os
import unittest

class TestIdentify(unittest.TestCase):
    
    outdir = os.path.join(os.path.split(os.path.realpath(__file__))[0],"outputs")

    def test_identify(self):
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

        write_dot(my_dfa, os.path.join(self.outdir,"test1.dot"))

    def test_identify_repeatedly(self):

        accepting = ['a', 'abaa', 'bb']
        rejecting = ['abb', 'b']
        
        for i in range(200):
            my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)
            
            # check that dfa found matches
            for x in accepting:
                assert my_dfa.label(x)

            for x in rejecting:
                assert not my_dfa.label(x)

            # check that minimal dfa is found
            assert len(my_dfa.states()) == 3

if __name__ == '__main__':
    unittest.main()