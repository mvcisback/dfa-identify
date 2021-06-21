from dfa_identify import find_dfa
from dfa_identify.encoding import SymmBreak
from dfa.draw import write_dot
import os
import unittest
import attr

@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class IdProblem:
    accepting: list[str]
    rejecting: list[str]
    size: int # min DFA size


class TestIdentify(unittest.TestCase):
    
    outdir = os.path.join(os.path.split(os.path.realpath(__file__))[0],"outputs")

    def solve_and_check(self, problem: IdProblem, symm_mode: SymmBreak = SymmBreak.CLIQUE):
        accepting = problem.accepting
        rejecting = problem.rejecting
        
        my_dfa = find_dfa(accepting=accepting, rejecting=rejecting, symm_mode = symm_mode)

        # check that identified DFA fits observations
        for x in accepting:
            assert my_dfa.label(x)

        for x in rejecting:
            assert not my_dfa.label(x)

        # check that returned DFA is minimal
        # assert len(my_dfa.states()) == problem.size
        return my_dfa

    def test_identify(self):

        symm = SymmBreak.BFS

        accepting = ['a', 'abaa', 'bb']
        rejecting = ['abb', 'b']
        problem1 = IdProblem(accepting, rejecting, 3)
        self.solve_and_check(problem1, symm_mode = symm)

        accepting = [[0], [0, 'z', 0, 0], ['z', 'z']]
        rejecting = [[0, 'z', 'z'], ['z']]
        problem2 = IdProblem(accepting, rejecting, 3)
        my_dfa = self.solve_and_check(problem2, symm_mode = symm)

        write_dot(my_dfa, os.path.join(self.outdir,"test1.dot"))

        accepting = ['ab', 'b', 'ba', 'bbb']
        rejecting = ['abbb', 'baba']
        problem3 = IdProblem(accepting, rejecting, 3)
        my_dfa = self.solve_and_check(problem3, symm_mode = symm)

        write_dot(my_dfa, os.path.join(self.outdir,"test2.dot"))
    
    def test_identify_2(self):

        symm = SymmBreak.BFS

        accepting = ['aaaab', 'aaaabaaaab', 'aaaabaaaabaaaab']
        rejecting = ['aaab', 'bb', 'aaaabb', 'aaaaaa', 'abba', 'ab', 'b', 'aab','a','aabb','ba', 'aba', 'bab','aaaaa','aa','aaa','aaaa','baab','bbaaab','bbaab','bbb']
        problem4 = IdProblem(accepting, rejecting, 5)
        my_dfa = self.solve_and_check(problem4, symm_mode = symm)

        write_dot(my_dfa, os.path.join(self.outdir,"test3.dot"))

    def test_identify_repeatedly(self):

        symm = SymmBreak.NONE

        accepting = ['a', 'abaa', 'bb']
        rejecting = ['abb', 'b']
        problem1 = IdProblem(accepting, rejecting, 3)
        for i in range(200):
            self.solve_and_check(problem1, symm_mode = symm)

        accepting = ['ab', 'b', 'ba', 'bbb']
        rejecting = ['abbb', 'baba']
        problem2 = IdProblem(accepting, rejecting, 3)
        for i in range(200):
            self.solve_and_check(problem2, symm_mode = symm)

        accepting = ['aaaab', 'aaaabaaaab', 'aaaabaaaabaaaab']
        rejecting = ['aaab', 'bb', 'aaaabb', 'aaaaaa', 'abba', 'ab', 'b', 'aab','a','aabb','ba', 'aba', 'bab','aaaaa','aa','aaa','aaaa','baab','bbaaab','bbaab','bbb']
        problem4 = IdProblem(accepting, rejecting, 5)
        for i in range(100):
            self.solve_and_check(problem4, symm_mode = symm)

if __name__ == '__main__':
    unittest.main()