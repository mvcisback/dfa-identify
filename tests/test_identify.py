from dfa_identify import find_dfa
from dfa_identify.encoding import SymmBreak
from dfa.draw import write_dot
import numpy as np
import time
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
    

    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']
    problem1 = IdProblem(accepting, rejecting, 3)

    accepting = [[0], [0, 'z', 0, 0], ['z', 'z']]
    rejecting = [[0, 'z', 'z'], ['z']]
    problem2 = IdProblem(accepting, rejecting, 3)
    
    accepting = ['ab', 'b', 'ba', 'bbb']
    rejecting = ['abbb', 'baba']
    problem3 = IdProblem(accepting, rejecting, 3)

    accepting = ['aaaab', 'aaaabaaaab', 'aaaabaaaabaaaab']
    rejecting = ['aaab', 'bb', 'aaaabb', 'aaaaaa', 'abba', 'ab', 'b', 'aab','a','aabb','ba', 'aba', 'bab','aaaaa','aa','aaa','aaaa','baab','bbaaab','bbaab','bbb']
    problem4 = IdProblem(accepting, rejecting, 5)

    accepting = ['aaaa','aabb','abab','abba','baab','baba','bbaa','bbbb','aaa','b']
    rejecting = ['aaab','aaba','abaa','abbb','baaa','babb','bbab','bbba']
    problem5 = IdProblem(accepting, rejecting, 4)

    def rand_test_set(self, alphabet, f, set_sz, p=0.8):
        positives = set()
        negatives = set()
        while (len(positives) + len(negatives)) < set_sz:
            x = ""
            while np.random.rand() < p:
                x += str(np.random.choice(alphabet))
            if f(x):
                positives.add(x)
            else:
                negatives.add(x)
        return list(positives), list(negatives)

    def generate_problem(self, n, input_sz, bypass = False):
        def g(x):
            ones = [a for a in x if a == '1']
            return 1 if len(ones) % n == 0 else 0
        accepting, rejecting = self.rand_test_set([0,1], g, input_sz)
        if bypass:
            return IdProblem(accepting, rejecting, 0)
        problem = IdProblem(accepting, rejecting, n)
        return problem

    def solve_and_check(self, problem: IdProblem, symm_mode: SymmBreak = SymmBreak.CLIQUE):
        accepting = problem.accepting
        rejecting = problem.rejecting
        
        my_dfa = find_dfa(accepting=accepting, rejecting=rejecting, symm_mode = symm_mode)

        # check that identified DFA fits observations
        for x in accepting:
            assert my_dfa.label(x)

        for x in rejecting:
            assert not my_dfa.label(x)

        # check that returned DFA is minimal (check bypassed if problem.size = 0)
        assert (problem.size == 0) or len(my_dfa.states()) == problem.size
        return my_dfa

    def test_identify(self):

        symm = SymmBreak.BFS

        self.solve_and_check(self.problem1, symm_mode = symm)

        my_dfa = self.solve_and_check(self.problem2, symm_mode = symm)
        write_dot(my_dfa, os.path.join(self.outdir,"test1.dot"))

        my_dfa = self.solve_and_check(self.problem3, symm_mode = symm)
        write_dot(my_dfa, os.path.join(self.outdir,"test2.dot"))

        my_dfa = self.solve_and_check(self.problem5, symm_mode = symm)
        write_dot(my_dfa, os.path.join(self.outdir,"test3.dot"))
    
    def test_identify_2(self):

        input_sz = 200
        
        symm = SymmBreak.NONE
        start = 0
        end = 5
        step = 1
        problems = {}

        for n in range(start, end, step):
            problems[n] = self.generate_problem(8, input_sz, bypass=True)
            # problems[n] = self.generate_problem(n, input_sz, bypass=True)

        print("Running tests for no symmetry breaking...")
        for n in range(start, end, step):

            t0 = time.time()
            dfa1 = self.solve_and_check(problems[n], symm_mode=symm)
            elapsed = time.time() - t0
            
            print("solved n = {} with no symmetry-breaking in {:.4g}s".format(n,elapsed))
            write_dot(dfa1, os.path.join(self.outdir,"longtest_n-{}_symm-{}.dot".format(n,symm)))
        
        symm = SymmBreak.CLIQUE
        print("Running tests for clique-based symmetry breaking...")
        for n in range(start, end, step):

            t0 = time.time()
            dfa1 = self.solve_and_check(problems[n], symm_mode=symm)
            elapsed = time.time() - t0
            
            print("solved n = {} with clique-based symmetry-breaking in {:.4g}s".format(n,elapsed))
            write_dot(dfa1, os.path.join(self.outdir,"longtest_n-{}_symm-{}.dot".format(n,symm)))
        
        symm = SymmBreak.BFS
        print("Running tests for BFS symmetry breaking...")
        for n in range(start, end, step):
            
            t0 = time.time()
            dfa1 = self.solve_and_check(problems[n], symm_mode=symm)
            elapsed = time.time() - t0
            
            print("solved n = {} with BFS symmetry-breaking in {:.4g}s".format(n,elapsed))
            write_dot(dfa1, os.path.join(self.outdir,"longtest_n-{}_symm-{}.dot".format(n,symm)))

    def test_identify_repeatedly(self):

        symm = SymmBreak.BFS

        for i in range(50):
            self.solve_and_check(self.problem1, symm_mode = symm)

        for i in range(50):
            self.solve_and_check(self.problem3, symm_mode = symm)

        for i in range(50):
            self.solve_and_check(self.problem4, symm_mode = symm)

        for i in range(50):
            self.solve_and_check(self.problem5, symm_mode = symm)

if __name__ == '__main__':
    unittest.main()