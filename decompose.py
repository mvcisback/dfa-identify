from dfa_identify.encoding import dfa_id_encodings, Codec, SymMode
from pysat.solvers import Glucose4
from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import Bounds, ExtraClauseGenerator, Clauses
from dfa_identify.identify import extract_dfa
from typing import Optional, Iterable
from dfa import dict2dfa, DFA 
from dfa.utils import find_equiv_counterexample

import itertools

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
    new_rejecting_clauses = {}
    for codec, offset in zip(codecs, offset_list):
        rejecting_clauses = list(partition_by_rejecting_clauses(codec, apta))
        rejecting_clauses = offset_clauses(rejecting_clauses, offset)
        for i, rejecting_clause in enumerate(rejecting_clauses):
            if i >= len(new_rejecting_clauses):
                new_rejecting_clauses.append([])
            new_rejecting_clauses[i].extend(rejecting_clause)

    clauses.extend(new_rejecting_clauses)
    return clauses

def add_new_rejecting_clause(clauses, codecs, offset_list, apta):
    # new_rejecting_clauses = {}
    # for sub_dfa, codec, offset in enumerate(zip(codecs, offset_list)):
    #     for c in range(codec.n_colors):
    #         lit = codec.color_accepting(c)
    #         for v in apta.rejecting:
    #             # (codec, v, c) : clause
    #             clause = [-codec.color_node(v, c), -lit]

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


        # rejecting_clauses = list(partition_by_rejecting_clauses(codec, apta))
        # rejecting_clauses = offset_clauses(rejecting_clauses, offset)
        # for i, rejecting_clause in enumerate(rejecting_clauses):
        #     if i >= len(new_rejecting_clauses):
        #         new_rejecting_clauses.append([])
        #     new_rejecting_clauses[i].extend(rejecting_clause)

    # clauses.extend(new_rejecting_clauses)

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
            'extra_clauses': extra_clauses, 'bounds': (None, None),
            'order_by_stutter': order_by_stutter, 'alphabet': alphabet,
            'allow_unminimized': allow_unminimized,
        }
        dfas_pos = find_dfas(accepting=[()], rejecting=[  ], **kwargs)
        dfas_neg = find_dfas(accepting=[  ], rejecting=[()], **kwargs)
        yield from roundrobin(dfas_pos, dfas_neg)
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
    models = order_models_by_stutter(solver_fact, codec, clauses, model)
    yield from (extract_dfa(codec, apta, m) for m in models)
    if allow_unminimized:
        return
    return

if __name__=="__main__":

    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']
    num_dfas = 3
    dfa_sizes = [4,4,4]
        
    my_dfas_gen = find_dfa_decompositions(accepting, rejecting, num_dfas, dfa_sizes)

    for my_dfas in my_dfas_gen:
        print(my_dfas)
        for my_dfa in my_dfas:
            assert all(my_dfa.label(x) for x in accepting)
        for my_dfa in my_dfas:
            print([my_dfa.label(x) for x in rejecting])
        for x in rejecting:
            assert any(not my_dfa.label(x) for my_dfa in my_dfas)
