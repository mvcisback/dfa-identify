from decompose import find_dfa_decompositions, enumerate_pareto_frontier
from dfa_identify import find_dfas
from dfa import dict2dfa, DFA, draw
import itertools
import time
import signal
import math
import random

TIMEOUT_SECONDS = 10 * 60 # Timeout after 10 minutes

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeOutException()

def generate_examples(n_tasks: int,
                      n_subtasks: int,
                      bound: int):

    assert n_tasks > 1
    assert n_subtasks > 1
    assert bound % 2 == 0 and bound > 0

    symbol_base = 'x'
    symbol_counter = 0
    tasks_symbols = []
    tasks_symbols_flatten = []

    for _ in range(n_tasks):
        task_symbols = []
        for _ in range(n_subtasks):
            task_symbols.append(symbol_base + str(symbol_counter))
            symbol_counter += 1
        tasks_symbols.append(task_symbols)
        tasks_symbols_flatten.extend(task_symbols)

    positive_examples = []
    while len(positive_examples) < bound // 2:
        positive_example_heads = [0] * n_tasks
        trace = []
        while not all(symbol in trace for symbol in tasks_symbols_flatten):
            random_head_idx = random.randint(0, n_tasks - 1)
            trace.append(tasks_symbols[random_head_idx][positive_example_heads[random_head_idx]])
            positive_example_heads[random_head_idx] += 1
            positive_example_heads[random_head_idx] %= n_subtasks
        if trace not in positive_examples:
            positive_examples.append(trace)

    negative_examples = []
    while len(negative_examples) < bound // 2:
        negative_example_heads = [n_subtasks - 1] * n_tasks
        trace = []
        while not all(symbol in trace for symbol in tasks_symbols_flatten):
            random_head_idx = random.randint(0, n_tasks - 1)
            trace.append(tasks_symbols[random_head_idx][negative_example_heads[random_head_idx]])
            negative_example_heads[random_head_idx] += n_subtasks - 1
            negative_example_heads[random_head_idx] %= n_subtasks
        if trace not in negative_examples:
            negative_examples.append(trace)

    assert len(positive_examples) == bound // 2 and len(negative_examples) == bound // 2

    return positive_examples, negative_examples

def get_next_solution_and_check(generator, accepting, rejecting, is_monolithic):
    try:
        start_time = time.time()
        result = next(generator)
        end_time = time.time()
    except:
        return None
    if is_monolithic:
        assert all(result.label(x) for x in accepting)
        assert all(not result.label(x) for x in rejecting)
        # draw.write_dot(result, "temp.dot")
        # input('Done1')
    else:
        for my_dfa in result:
            assert all(my_dfa.label(x) for x in accepting)
            # draw.write_dot(my_dfa, "temp.dot")
            # input('Done2')
        for x in rejecting:
            assert any(not my_dfa.label(x) for my_dfa in result)
    return end_time - start_time

def exp_vary_dfas(seed, n_syms, n_dfas_lower, n_dfas_upper, bound):
    baseline = {}
    this_work = {}

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(0)

    for n_dfas in range(n_dfas_lower, n_dfas_upper + 1):
        try:
            accepting, rejecting = generate_examples(n_dfas, n_syms, bound)
            baseline_generator = find_dfas(accepting, rejecting, order_by_stutter=True)
            signal.alarm(TIMEOUT_SECONDS)
            try:
                result = get_next_solution_and_check(baseline_generator, accepting, rejecting, True)
            except TimeOutException:
                result = None
            signal.alarm(0)
            baseline[(bound, n_syms, n_dfas)] = result
            print('Baseline', (bound, n_syms, n_dfas), baseline[(bound, n_syms, n_dfas)])
            this_work_generator = enumerate_pareto_frontier(accepting, rejecting, n_dfas, order_by_stutter=True)
            signal.alarm(TIMEOUT_SECONDS)
            try:
                result = get_next_solution_and_check(this_work_generator, accepting, rejecting, False)
            except TimeOutException:
                result = None
            signal.alarm(0)
            this_work[(bound, n_syms, n_dfas)] = result
            print('This Work', (bound, n_syms, n_dfas), this_work[(bound, n_syms, n_dfas)])
        except TimeOutException:
            pass
        signal.alarm(0)

    f = open('seed' + str(seed) + '_exp_vary_dfas_baseline_' + str(n_syms) + '_' + str(n_dfas_lower) + '_' + str(n_dfas_upper) + '_' + str(bound) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in baseline:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(baseline[result]) + '\n')
    f.close()

    f = open('seed' + str(seed) + '_exp_vary_dfas_this_work_' + str(n_syms) + '_' + str(n_dfas_lower) + '_' + str(n_dfas_upper) + '_' + str(bound) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in this_work:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(this_work[result]) + '\n')
    f.close()

def exp_vary_examples(seed, n_syms, n_dfas, bound_lower, bound_upper, step=10):
    baseline = {}
    this_work = {}

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(0)

    for bound in range(bound_lower, bound_upper + 1, step):
        try:
            accepting, rejecting = generate_examples(n_dfas, n_syms, bound)
            baseline_generator = find_dfas(accepting, rejecting, order_by_stutter=True)
            signal.alarm(TIMEOUT_SECONDS)
            try:
                result = get_next_solution_and_check(baseline_generator, accepting, rejecting, True)
            except TimeOutException:
                result = None
            signal.alarm(0)
            baseline[(bound, n_syms, n_dfas)] = result
            print('Baseline', (bound, n_syms, n_dfas), baseline[(bound, n_syms, n_dfas)])
            this_work_generator = enumerate_pareto_frontier(accepting, rejecting, n_dfas, order_by_stutter=True)
            signal.alarm(TIMEOUT_SECONDS)
            try:
                result = get_next_solution_and_check(this_work_generator, accepting, rejecting, False)
            except TimeOutException:
                result = None
            signal.alarm(0)
            this_work[(bound, n_syms, n_dfas)] = result
            print('This Work', (bound, n_syms, n_dfas), this_work[(bound, n_syms, n_dfas)])
        except TimeOutException:
            pass
        signal.alarm(0)

    f = open('seed' + str(seed) + '_exp_vary_examples_baseline_' + str(n_syms) + '_' + str(n_dfas) + '_' + str(bound_lower) + '_' + str(bound_upper) + '_' + str(step) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in baseline:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(baseline[result]) + '\n')
    f.close()

    f = open('seed' + str(seed) + '_exp_vary_examples_this_work_' + str(n_syms) + '_' + str(n_dfas) + '_' + str(bound_lower) + '_' + str(bound_upper) + '_' + str(step) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in this_work:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(this_work[result]) + '\n')
    f.close()

def exp_vary_solutions(seed, n_syms, n_dfas, bound, solutions=100):
    baseline = {}
    this_work = {}

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(0)

    accepting, rejecting = generate_examples(n_dfas, n_syms, bound)
    baseline_generator = find_dfas(accepting, rejecting, order_by_stutter=True)
    this_work_generator = enumerate_pareto_frontier(accepting, rejecting, n_dfas, order_by_stutter=True)
    baseline_result = 0
    this_work_result = 0
    for solution in range(1, solutions + 1):
        try:
            signal.alarm(TIMEOUT_SECONDS)
            try:
                baseline_result += get_next_solution_and_check(baseline_generator, accepting, rejecting, True)
            except TimeOutException:
                break
            signal.alarm(0)
            baseline[(bound, n_syms, n_dfas)] = baseline_result
            print('Baseline', (bound, n_syms, n_dfas, solution), baseline[(bound, n_syms, n_dfas)])
            signal.alarm(TIMEOUT_SECONDS)
            try:
                this_work_result += get_next_solution_and_check(this_work_generator, accepting, rejecting, False)
            except TimeOutException:
                break
            signal.alarm(0)
            this_work[(bound, n_syms, n_dfas)] = this_work_result
            print('This Work', (bound, n_syms, n_dfas, solution), this_work[(bound, n_syms, n_dfas)])
        except TimeOutException:
            pass
        signal.alarm(0)

    f = open('seed' + str(seed) + '_exp_vary_solutions_baseline_' + str(n_syms) + '_' + str(n_dfas) + '_' + str(bound) + '_' + str(solutions) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in baseline:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(baseline[result]) + '\n')
    f.close()

    f = open('seed' + str(seed) + '_exp_vary_solutions_this_work_' + str(n_syms) + '_' + str(n_dfas) + '_' + str(bound) + '_' + str(solutions) + '_results.csv', 'w+')
    f.write('bound,n_syms,n_dfas,time\n')
    for result in this_work:
        f.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(this_work[result]) + '\n')
    f.close()

if __name__ == '__main__':

    for seed in range(10):

        random.seed(seed)

        print('Running for seed', seed)

        exp_vary_dfas(seed, 2, 2, 10, 10)
        exp_vary_dfas(seed, 4, 2, 10, 10)
        exp_vary_dfas(seed, 8, 2, 10, 10)

        exp_vary_dfas(seed, 2, 2, 10, 100)
        exp_vary_dfas(seed, 4, 2, 10, 100)
        exp_vary_dfas(seed, 8, 2, 10, 100)

        exp_vary_examples(seed, 2, 2, 10, 100)
        exp_vary_examples(seed, 2, 4, 10, 100)
        exp_vary_examples(seed, 2, 8, 10, 100)

        exp_vary_examples(seed, 4, 2, 10, 100)
        exp_vary_examples(seed, 4, 4, 10, 100)
        exp_vary_examples(seed, 4, 8, 10, 100)

        exp_vary_examples(seed, 8, 2, 10, 100)
        exp_vary_examples(seed, 8, 4, 10, 100)
        exp_vary_examples(seed, 8, 8, 10, 100)

        exp_vary_solutions(seed, 4, 2, 10, 100)
        exp_vary_solutions(seed, 4, 4, 10, 100)
        exp_vary_solutions(seed, 4, 8, 10, 100)

        exp_vary_solutions(seed, 8, 2, 10, 100)
        exp_vary_solutions(seed, 8, 4, 10, 100)
        exp_vary_solutions(seed, 8, 8, 10, 100)

