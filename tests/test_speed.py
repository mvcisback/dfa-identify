from dfa_identify import find_dfa
from tests.test_identify import generate_problem, solve_and_check
from dfa.draw import write_dot
import numpy as np
import time
import os

outdir = os.path.join(os.path.split(os.path.realpath(__file__))[0],"outputs")

def test_identify_2():

    input_sz = 300
    
    symm = None
    start = 3
    end = 8
    step = 1
    problems = {}

    # for n in range(start, end, step):
    #     # problems[n] = self.generate_problem(8, input_sz, bypass=True)
    #     problems[n] = generate_problem(n, input_sz, bypass=True)

    # print("Running tests for no symmetry breaking...")
    # for n in range(start, end, step):

    #     t0 = time.time()
    #     dfa1 = solve_and_check(problems[n], symm_mode=symm)
    #     elapsed = time.time() - t0
        
    #     print("solved n = {} with no symmetry-breaking in {:.4g}s".format(n,elapsed))
    #     write_dot(dfa1, os.path.join(outdir,"longtest_n-{}_symm-{}.dot".format(n,symm)))
    
    # symm = "clique"
    # print("Running tests for clique-based symmetry breaking...")
    # for n in range(start, end, step):

    #     t0 = time.time()
    #     dfa1 = solve_and_check(problems[n], symm_mode=symm)
    #     elapsed = time.time() - t0
        
    #     print("solved n = {} with clique-based symmetry-breaking in {:.4g}s".format(n,elapsed))
    #     write_dot(dfa1, os.path.join(outdir,"longtest_n-{}_symm-{}.dot".format(n,symm)))
    
    # symm = "bfs"
    # print("Running tests for BFS symmetry breaking...")
    # for n in range(start, end, step):
        
    #     t0 = time.time()
    #     dfa1 = solve_and_check(problems[n], symm_mode=symm)
    #     elapsed = time.time() - t0
        
    #     print("solved n = {} with BFS symmetry-breaking in {:.4g}s".format(n,elapsed))
    #     write_dot(dfa1, os.path.join(outdir,"longtest_n-{}_symm-{}.dot".format(n,symm)))

if __name__ == "__main__":
    test_identify_2()