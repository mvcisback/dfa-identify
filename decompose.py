from dfa_identify.encoding import dfa_id_encodings, Codec, SymMode
from pysat.solvers import Glucose4
from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import Bounds, ExtraClauseGenerator, Clauses
from dfa_identify.identify import extract_dfa, find_dfas
from typing import Optional, Iterable
from dfa import dict2dfa, DFA, draw
from dfa.utils import find_equiv_counterexample, minimize

from pysat.card import CardEnc
from dfa_identify.encoding import ParentRelationVar

import itertools
from collections import deque

from more_itertools import interleave_longest

def order_models_by_stutter(
        solver_fact,
        codecs: list[Codec],
        offset_list: list[int],
        clauses: list[list[int]],
        model: list[int],
) -> Iterable[DFA]:
    top_id = codecs[-1].offsets[-1] + offset_list[-1]

    # Compute parent relation variables that don't stutter.
    codecs_lits = []
    for codec, offset in zip(codecs, offset_list):
        lits = []
        for lit in range(1 + codec.offsets[2], codec.offsets[3] + 1):
            par_rel = codec.decode(lit)
            assert isinstance(par_rel, ParentRelationVar)
            if par_rel.node_color == par_rel.parent_color:
                continue
            lits.append(lit + offset if lit >= 0 else lit - offset)
        codecs_lits.append(lits)

    # Binary search for min non-stutter using cardinality constraints.

    def non_stutter_count(model) -> int:
        return [sum(model[x - 1] > 0 for x in lits) for lits in codecs_lits]

    def find_models(bounds: list[int], make_formula):
        formulas = [make_formula(lits=lits, bound=bound, top_id=top_id) for lits, bound in zip(codecs_lits, bounds)]

        with solver_fact(bootstrap_with=clauses) as solver:
            for formula in formulas:
                solver.append_formula(formula, no_return=True)
            if not solver.solve():
                return
            yield from solver.enum_models()

    def find_increment_index(los, his, normalize=True):
        sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
        if normalize:
            diffs = [(hi - lo) / hi if hi != 0 else sign(hi - lo) * float('inf') for lo, hi in zip(los, his)]
        else:
            diffs = [(hi - lo) for lo, hi in zip(los, his)]
        max_diff = max(diffs)
        max_diff_index = diffs.index(max_diff)
        return max_diff_index

    candidate_bounds = non_stutter_count(model) # Candidate upper bound.
    his = candidate_bounds # Also upper bounds lower bound.
    los = [codec.n_colors - 1 for codec in codecs] # Each node needs to be visited.
    while any([lo < hi for lo, hi in zip(los, his)]):
        mids = [(lo + hi) // 2 for lo, hi in zip(los, his)]
        models = find_models(mids, CardEnc.atmost)
        witness = next(models, None)
        if witness is not None:
            his = non_stutter_count(witness)
            assert all([hi <= mid for hi, mid in zip(his, mids)])
        else:
            increment_index = find_increment_index(los, his)
            los[increment_index] = mids[increment_index] + 1

    # Incrementally emit models with less stutter.
    naive_bounds = [len(lits) for lits in codecs_lits]
    bounds = los
    while any([bound <= naive_bound for bound, naive_bound in zip(bounds, naive_bounds)]):
        if any([bound > candidate_bound for bound, candidate_bound in zip(bounds, candidate_bounds)]):
            witness = next(find_models(bounds, CardEnc.atmost), None)
            if witness is None:
                break
            candidate_bounds = non_stutter_count(witness)
        yield from find_models(bounds, CardEnc.atmost) # For some reason CardEnc.equal does not work here
        increment_index = find_increment_index(bounds, naive_bounds)
        bounds[increment_index] += 1

def partition_by_rejecting_clauses(codec: Codec, apta: APTA) -> Clauses:
    for c in range(codec.n_colors):
        lit = codec.color_accepting(c)
        yield from ([-codec.color_node(n, c), -lit] for n in apta.rejecting)

def get_max_var(clauses):
    return max([abs(l) for clause in clauses for l in clause])

def offset_clauses(clauses, offset_amount):
    return [[l + offset_amount if l >= 0 else l - offset_amount for l in clause] for clause in clauses]

def offset_encodings(encodings_list):
    first_codec, new_clauses_list = encodings_list[0]
    codec_list = [first_codec]
    offset_list = [0]
    for codec, clauses in encodings_list[1:]:
        codec_list.append(codec)
        offset_amount = get_max_var(new_clauses_list)
        new_clauses = offset_clauses(clauses, offset_amount)
        new_clauses_list.extend(new_clauses)
        offset_list.append(offset_amount)

    return codec_list, offset_list, new_clauses_list

def remove_rejecting_clauses(encodings, apta):
    for codec, clauses in encodings:
        rejecting_clauses = list(partition_by_rejecting_clauses(codec, apta))
        new_clauses = []
        for clause in clauses:
            if clause not in rejecting_clauses:
                new_clauses.append(clause)
        yield codec, new_clauses

def add_new_rejecting_clause(clauses, codecs, offset_list, apta):
    color_sizes = [range(codec.n_colors) for codec in codecs]
    for v in apta.rejecting:
        for prod in itertools.product(*color_sizes):
            new_rejecting_clause = []
            for j,c in enumerate(prod):
                # j indexs the dfa, c is the color for that dfa
                codec = codecs[j]
                offset_amount = offset_list[j]
                lit = codec.color_accepting(c)
                clause = [-codec.color_node(v, c), -lit]
                offset_clause = offset_clauses([clause], offset_amount)[0]
                new_rejecting_clause.extend(offset_clause)
            clauses.append(new_rejecting_clause)
    return clauses

def extract_dfas(codecs, offset_list, apta, m):
    offset_list = list(offset_list)
    offset_list.append(get_max_var([m]))
    m_partition = []
    offset_counter = 0
    for i in range(len(offset_list)-1):
        sub_m = m[offset_list[i]:offset_list[i+1]]
        offset = offset_list[i]
        m_partition.append([j - offset if j >= 0 else j + offset for j in sub_m])

    sub_dfas = []
    for sub_m, codec in zip(m_partition, codecs):
        sub_dfas.append(extract_dfa(codec, apta, sub_m))

    return sub_dfas

def enumerate_pareto_frontier(
        accepting: list[Word],
        rejecting: list[Word],
        num_dfas: int,
        min_dfa_sizes: int = 2,
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        order_by_stutter: bool = False,
        alphabet: frozenset = None,
        allow_unminimized: bool = False,
) -> Iterable[DFA]:

    min_dfa_size = [min_dfa_sizes]*num_dfas
    size_q = deque()
    size_q.append(min_dfa_size)
    pareto_frontier = {}

    while size_q: # while not empty
        dfa_sizes = size_q.popleft()
        dominated = not all(any(new_size < front_size for new_size,front_size in zip(new_dfa_sizes,frontier_sizes)) for frontier_sizes in pareto_frontier.keys())
        if dominated:
            continue
        dfa_gen = find_dfa_decompositions(accepting,rejecting,num_dfas,dfa_sizes,solver_fact,sym_mode,extra_clauses,\
                order_by_stutter,alphabet,allow_unminimized)
        try:
            # yield the dfa from this generator
            next_dfa = next(dfa_gen)
            pareto_frontier[tuple(dfa_sizes)] = dfa_gen
            yield next_dfa
        except StopIteration:
            # add children to queue
            for i in range(num_dfas):
                new_dfa_sizes = list(dfa_sizes)
                new_dfa_sizes[i] += 1
                not_dominated = all(any(new_size < front_size for new_size,front_size in zip(new_dfa_sizes,frontier_sizes)) for frontier_sizes in pareto_frontier.keys())
                nondecreasing = all(new_dfa_sizes[i] <= new_dfa_sizes[i+1] for i in range(len(new_dfa_sizes) - 1))
                not_in_queue = new_dfa_sizes not in size_q

                # we want to avoid making symmetric solves, so only append the sizes that are ordered in increasing size
                if not_dominated and nondecreasing and not_in_queue:
                    size_q.append(new_dfa_sizes)

    # interleave generators on the pareto frontier until exhausted
    for dfa_product in interleave_longest(*pareto_frontier.values()):
        yield dfa_product

def find_dfa_decompositions(
        accepting: list[Word],
        rejecting: list[Word],
        num_dfas: int,
        dfa_sizes: list[int],
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        order_by_stutter: bool = False,
        alphabet: frozenset = None,
        allow_unminimized: bool = False,
) -> Iterable[DFA]:
    """Finds all minimal dfa that are consistent with the labeled examples.

    Here "minimal" means that a no DFA with smaller size is consistent with
    the data. Thus, all returns DFAs are the same size.

    Inputs:
      - accepting: A sequence of "words" to be accepted.
      - rejecting: A sequence of "words" to be rejected.
      - solver: A py-sat API compatible object for solving CNF SAT queries.
      - bounds: DFA size range (inclusive) to restrict search to, e.g.,
                - (None, 10): DFA can have as most 10 states.
                - (2, None): DFA must have at least 2 states.
                - (2, 10):  DFA must have between 2 and 10 states.
                - (None, None): No constraints (default).
      - sym_mode: Which symmetry breaking strategy to employ.
      - extra_clauses: Optional user defined additional clauses to add
          for a given codec (encoding of size k DFA).
      - order_by_stutter: Order DFA by number of self loop transitions.
      - alphabet: Optionally specify the alphabet the DFA should be over.
      - allow_unminimized: Continue after all minimized (equiv
          states merges) have been enumerated.

    Returns:
      An iterable of all minimal DFA consistent with accepting and rejecting.
    """
    # Convert to hashable words.
    accepting = list(map(tuple, accepting))
    rejecting = list(map(tuple, rejecting))

    if set(accepting) & set(rejecting):
        return
    elif len(accepting) == len(rejecting) == 0:
        if not alphabet:
            raise ValueError('Need examples or an alphabet!')

        # Conjecture empty string label and interleave dfas.
        kwargs = {
            'solver_fact': solver_fact, 'sym_mode': sym_mode,
            'extra_clauses': extra_clauses, 'num_dfas': num_dfas, 'dfa_sizes': dfa_sizes,
            'order_by_stutter': order_by_stutter, 'alphabet': alphabet,
            'allow_unminimized': allow_unminimized,
        }
        dfas_pos = find_dfa_decompositions(accepting=[()], rejecting=[  ], **kwargs)
        dfas_neg = find_dfa_decompositions(accepting=[  ], rejecting=[()], **kwargs)
        yield from interleave_longest(dfas_pos, dfas_neg)
        return 

    apta = APTA.from_examples(
        accepting=accepting, rejecting=rejecting, alphabet=alphabet
    )
    encodings_list = []
    for dfa_size in dfa_sizes:
        bounds = (dfa_size, dfa_size)
        encodings = dfa_id_encodings(
            apta=apta, sym_mode=sym_mode,
            extra_clauses=extra_clauses, bounds=bounds)
        encodings = remove_rejecting_clauses(encodings, apta)
        first_encoding = list(next(encodings))
        encodings_list.append(first_encoding)

    codecs, offset_list, clauses = offset_encodings(encodings_list)

    clauses = add_new_rejecting_clause(clauses, codecs, offset_list, apta)

    with solver_fact(bootstrap_with=clauses) as solver:
        if not solver.solve():
            return
        if not order_by_stutter:
            models = solver.enum_models()
            yield from (extract_dfas(codecs, offset_list, apta, m) for m in models)
            if allow_unminimized:
                return
            return

        model = solver.get_model()  # Save for analysis below.

    # Search for maximally stuttering DFAs.
    models = order_models_by_stutter(solver_fact, codecs, offset_list, clauses, model)
    yield from (extract_dfas(codecs, offset_list, apta, m) for m in models)
    if allow_unminimized:
        return
    return

if __name__=="__main__":

    accepting = ['abcd', 'acbd', 'acdb', 'cdab', 'cadb', 'cabd']
    rejecting = []

    for i in range(5):
        for j in itertools.product('abcd', repeat=i):
            trace = ''.join(j)
            if trace not in accepting:
                rejecting.append(trace)
    # accepting = ['y', 'yy', 'oy', 'boy']
    # rejecting = ['r', 'b', 'o', 'or', 'br', 'yr', 'rr', 'by']
    num_dfas = 2
    dfa_sizes = [3, 3]
        
    # my_dfas_gen = find_dfa_decompositions(accepting, rejecting, num_dfas, dfa_sizes, order_by_stutter=True)
    my_dfas_gen = enumerate_pareto_frontier(accepting, rejecting, num_dfas, order_by_stutter=True)
    for my_dfas in my_dfas_gen:
        print(my_dfas)
        count = 0
        for my_dfa in my_dfas:
            draw.write_dot(my_dfa, "temp" + str(count) + ".dot")
            assert all(my_dfa.label(x) for x in accepting)
            count += 1
        for x in rejecting:
            assert any(not my_dfa.label(x) for my_dfa in my_dfas)

    # my_dfa_gen = find_dfas(accepting, rejecting)

    # for my_dfa in my_dfa_gen:
    #     print(my_dfa)
    #     draw.write_dot(my_dfa, "temp.dot")
    #     assert all(my_dfa.label(x) for x in accepting)
    #     assert all(not my_dfa.label(x) for x in rejecting)
    #     input()
