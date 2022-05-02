from decompose import find_dfa_decompositions, enumerate_pareto_frontier
from dfa_identify import find_dfas
from dfa import dict2dfa, DFA, draw
import itertools
import time
import signal

TIMEOUT_SECONDS = 10 * 60 # Timeout after 10 minutes

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()

def interleave(strings, upper_bound=None):
    temps = [strings[0]]
    for l in strings[1:-1]:
        for temp in list(temps):
            interleave_helper(temp, l, '', 0, 0, temps, upper_bound)
    temps = list(filter(lambda x: len(x) == sum(len(l) for l in strings[:-1]), temps))
    results = []
    for temp in temps:
        interleave_helper(temp, strings[-1], '', 0, 0, results, upper_bound)
    return results

def interleave_helper(s, t, res, i, j, lis, upper_bound=None):
    if i == len(s) and j == len(t):
        lis.append(res)
        return
    if i < len(s):
        if upper_bound is not None and len(lis) < upper_bound:
            interleave_helper(s, t, res + s[i], i + 1, j, lis, upper_bound)
        elif upper_bound is None:
            interleave_helper(s, t, res + s[i], i + 1, j, lis, upper_bound)
    if j < len(t):
        if upper_bound is not None and len(lis) < upper_bound:
            interleave_helper(s, t, res + t[j], i, j + 1, lis, upper_bound)
        elif upper_bound is None:
            interleave_helper(s, t, res + t[j], i, j + 1, lis, upper_bound)

def generate_examples(n_tasks: int,
                      n_subtasks: int,
                      accepting_upper_bound: int,
                      rejecting_upper_bound: int,
                      alphabet: str):
    assert n_tasks > 0
    assert n_subtasks > 1
    assert accepting_upper_bound > 0
    assert rejecting_upper_bound > 0
    assert n_tasks * n_subtasks <= len(alphabet)
    accepting = []
    rejecting = []
    alphabet_head = 0
    task_symbols = []
    for _ in range(n_tasks):
        task_symbol = ''
        for _ in range(n_subtasks):
            task_symbol += alphabet[alphabet_head]
            alphabet_head += 1
        task_symbols.append(task_symbol)
    for trace in interleave(task_symbols, accepting_upper_bound):
        accepting.append(trace)
    rejecting_count = 0
    for j in itertools.permutations(''.join(task_symbols)):
        trace = ''.join(j)
        if trace not in accepting:
            rejecting.append(trace)
            rejecting_count += 1
            if rejecting_count >= rejecting_upper_bound:
                break
    return accepting, rejecting

def construct_monolithic_dfa(accepting, rejecting, write_to_dot_file=False):
    start_time = time.time()
    my_dfas_gen = find_dfas(accepting, rejecting, order_by_stutter=True)
    my_dfa = next(my_dfas_gen)
    end_time = time.time()
    assert all(my_dfa.label(x) for x in accepting)
    assert all(not my_dfa.label(x) for x in rejecting)
    if write_to_dot_file:
        draw.write_dot(my_dfa, "temp.dot")
    return end_time - start_time

def construct_dfa_decompositions(n_dfas, accepting, rejecting, write_to_dot_file=False):
    start_time = time.time()
    my_dfas_gen = enumerate_pareto_frontier(accepting, rejecting, n_dfas, order_by_stutter=True)
    my_dfas = next(my_dfas_gen)
    end_time = time.time()
    for my_dfa in my_dfas:
        assert all(my_dfa.label(x) for x in accepting)
    for x in rejecting:
        assert any(not my_dfa.label(x) for my_dfa in my_dfas)
    if write_to_dot_file:
        count = 0
        for my_dfa in my_dfas:
            draw.write_dot(my_dfa, "temp" + str(count) + ".dot")
            count += 1
    return end_time - start_time

baseline = {}
this_work = {}

signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(0)

for bound in range(10, 101, 10):
    for n_syms in range(2, 5):
        for n_dfas in range(1, 11):
            try:
                accepting, rejecting = generate_examples(n_dfas, n_syms, bound, bound, "abcdefghijklmnopqrstuvwxyz0123456789@#$%")
                signal.alarm(TIMEOUT_SECONDS)
                try:
                    result = construct_monolithic_dfa(accepting, rejecting)
                except TimeOutException:
                    result = None
                signal.alarm(0)
                baseline[(bound, n_syms, n_dfas)] = result
                print('Baseline', (bound, n_syms, n_dfas), baseline[(bound, n_syms, n_dfas)])
                signal.alarm(TIMEOUT_SECONDS)
                try:
                    result = construct_dfa_decompositions(n_dfas, accepting, rejecting)
                except TimeOutException:
                    result = None
                signal.alarm(0)
                this_work[(bound, n_syms, n_dfas)] = result
                print('This Work', (bound, n_syms, n_dfas), this_work[(bound, n_syms, n_dfas)])
            except:
                signal.alarm(0)
                pass
    f = open('baseline_bound_' + str(bound) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in baseline:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(baseline[result]) + '\n')
    f.close()

    f = open('this_work_bound_' + str(bound) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in this_work:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(this_work[result]) + '\n')
    f.close()

# n_dfas = 4
# n_syms = 2
# bound = 100

# accepting, rejecting = generate_examples(n_dfas, n_syms, bound, bound, "abcdefghijklmnopqrstuvwxyz0123456789@#$%")

# print(accepting)
# print(rejecting)

# construct_monolithic_dfa(accepting, rejecting, True)
# construct_dfa_decompositions(n_dfas, accepting, rejecting, True)
