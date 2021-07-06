from dfa_identify import find_dfa
from dfa.draw import write_dot
from typing import Optional, Literal
import numpy as np
import os
import attr
from datetime import datetime

from pysat.solvers import Glucose4
from dfa_identify.identify import extract_dfa
from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import dfa_id_encodings

@attr.s(auto_detect=True, auto_attribs=True, frozen=True)
class IdProblem:
    accepting: list[str]
    rejecting: list[str]
    size: int # min DFA size

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

def rand_test_set(alphabet, f, set_sz, p=0.8):
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

def generate_problem(n, input_sz, bypass = False):
    def g(x):
        ones = [a for a in x if a == '1']
        return 1 if len(ones) % n == 0 else 0
    accepting, rejecting = rand_test_set([0,1], g, input_sz)
    if bypass:
        return IdProblem(accepting, rejecting, 0)
    problem = IdProblem(accepting, rejecting, n)
    return problem

def solve_and_check(problem: IdProblem, symm_mode: Optional[Literal["clique", "bfs"]] = "clique"):
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

def solve_and_check_exhaustively(problem: IdProblem, symm_mode: Optional[Literal["clique", "bfs"]] = "clique"):
    accepting = problem.accepting
    rejecting = problem.rejecting
    solutions = []
    
    apta = APTA.from_examples(accepting=accepting, rejecting=rejecting)
    for codec, clauses in dfa_id_encodings(apta, symm_mode = symm_mode):
        print("{} \tSolving SAT problem for \tcolors = {} \tclauses = {}".format(datetime.now().strftime("%y%m%d-%H:%M:%S"),codec.n_colors, len(clauses)))
        with Glucose4() as solver:
            # print("clauses:",[clause for clause in clauses if (62 in clause)])
            for clause in clauses:
                solver.add_clause(clause)

            while solver.solve():
                model = solver.get_model()
                solutions.append(extract_dfa(codec, apta, model))
                # print(model)
                solver.add_clause([-lit for lit in model])

            if len(solutions) > 0:
                break

    # check that identified DFA fits observations
    for my_dfa in solutions:
        for x in accepting:
            assert my_dfa.label(x)

        for x in rejecting:
            assert not my_dfa.label(x)

        # check that returned DFA is minimal (check bypassed if problem.size = 0)
        assert (problem.size == 0) or len(my_dfa.states()) == problem.size
    return solutions

def test_identify():

    symm = "bfs"

    solve_and_check(problem1, symm_mode = symm)

    my_dfa = solve_and_check(problem2, symm_mode = symm)
    write_dot(my_dfa, os.path.join(outdir,"test1.dot"))

    my_dfa = solve_and_check(problem3, symm_mode = symm)
    write_dot(my_dfa, os.path.join(outdir,"test2.dot"))

    my_dfa = solve_and_check(problem5, symm_mode = symm)
    write_dot(my_dfa, os.path.join(outdir,"test3.dot"))

def test_identify_exhaustively():
    
    symm = "bfs"

    print("Performing exhaustive search")
    solutions = solve_and_check_exhaustively(problem2, symm_mode = symm)
    for i, my_dfa in enumerate(solutions):
        write_dot(my_dfa, os.path.join(outdir,"problem2_soln{}.dot".format(i)))
    print("{} solutions found for problem2, using BFS symmetry-breaking.".format(len(solutions)))
    assert len(solutions) == 15

    solutions = solve_and_check_exhaustively(problem3, symm_mode = symm)
    for i, my_dfa in enumerate(solutions):
        write_dot(my_dfa, os.path.join(outdir,"problem3_soln{}.dot".format(i)))
    print("{} solutions found for problem3, using BFS symmetry-breaking.".format(len(solutions)))
    assert len(solutions) == 1

    problem6 = generate_problem(8, 500)
    solutions = solve_and_check_exhaustively(problem6, symm_mode = symm)
    for i, my_dfa in enumerate(solutions):
        write_dot(my_dfa, os.path.join(outdir,"problem6_soln{}.dot".format(i)))
    print("{} solutions found for problem6, using BFS symmetry-breaking.".format(len(solutions)))
    assert len(solutions) == 1

    # problem7 = generate_problem(10, 500)
    # solutions = solve_and_check_exhaustively(problem7, symm_mode = symm)
    # for i, my_dfa in enumerate(solutions):
    #     write_dot(my_dfa, os.path.join(outdir,"problem7_soln{}.dot".format(i)))
    # print("{} solutions found for problem7, using BFS symmetry-breaking.".format(len(solutions)))
    # assert len(solutions) == 1

def test_identify_repeatedly():

    symm = "bfs"

    for i in range(10):
        solve_and_check(problem1, symm_mode = symm)

    for i in range(10):
        solve_and_check(problem3, symm_mode = symm)

    for i in range(10):
        solve_and_check(problem4, symm_mode = symm)

    for i in range(10):
        solve_and_check(problem5, symm_mode = symm)

if __name__ == "__main__":
    test_identify()
    test_identify_exhaustively()