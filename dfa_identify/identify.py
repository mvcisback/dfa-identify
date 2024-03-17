from __future__ import annotations

from collections import deque
from functools import partial
from typing import Optional, Iterable

import funcy as fn
from dfa import DFA
from networkx.algorithms.approximation.clique import max_clique
from pysat.solvers import Glucose4
from pysat.card import CardEnc
from more_itertools import roundrobin

from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import Codec, SymMode
from dfa_identify.encoding import Bounds, ExtraClauseGenerator
from dfa_identify.encoding import (
    ColorAcceptingVar,
    ColorNodeVar,
    ParentRelationVar
)


def find_dfas(
        accepting: list[Word],
        rejecting: list[Word],
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
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
    models = find_models(accepting=accepting,
                         rejecting=rejecting,
                         solver_fact=solver_fact,
                         sym_mode=sym_mode,
                         extra_clauses=extra_clauses,
                         bounds=bounds,
                         order_by_stutter=order_by_stutter,
                         alphabet=alphabet,
                         allow_unminimized=allow_unminimized)
    yield from (codec.interpret_model(m) for codec, m in models)


def find_models(
        accepting: list[Word],
        rejecting: list[Word],
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
        order_by_stutter: bool = False,
        alphabet: frozenset = None,
        allow_unminimized: bool = False,
        n_dfas: int = 1,
) -> Iterable[tuple[Codec, list[int]]]:
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
            'extra_clauses': extra_clauses, 'bounds': bounds,
            'order_by_stutter': order_by_stutter, 'alphabet': alphabet,
            'allow_unminimized': allow_unminimized,
        }
        models_pos = find_models(accepting=[()], rejecting=[  ], **kwargs)
        models_neg = find_models(accepting=[  ], rejecting=[()], **kwargs)
        yield from roundrobin(models_pos, models_neg)
        return 

    apta = APTA.from_examples(
        accepting=accepting, rejecting=rejecting, alphabet=alphabet
    )

    cgraph = apta.consistency_graph()
    clique = max_clique(cgraph)

    low, high = bounds
    if (low is not None) and (high is not None) and (high < low):
        raise ValueError('Empty bound range!')

    # Tighten lower bound.
    if low is None: low = 1
    low = max(low, len(clique))

    gen_models = partial(_gen_models,
                         apta=apta,
                         cgraph=cgraph,
                         clique=clique,
                         sym_mode=sym_mode,
                         solver_fact=solver_fact,
                         extra_clauses=extra_clauses,
                         order_by_stutter=order_by_stutter)

    yield from pareto_search(gen_models,
                             num_dfas=1,
                             min_size=low,
                             max_size=high,
                             upperbound=len(apta.nodes),
                             allow_unminimized=allow_unminimized)


def _gen_models(sizes, apta, cgraph, clique, sym_mode,
                extra_clauses, solver_fact, order_by_stutter):
    n_colors = sizes[0]
    codec = Codec.from_apta(apta,
                            n_colors,
                            sym_mode=sym_mode,
                            extra_clauses=extra_clauses)

    clauses = list(codec.clauses(cgraph, clique))

    with solver_fact(bootstrap_with=clauses) as solver:
        if not solver.solve():
            return
        if not order_by_stutter:
            models = solver.enum_models()
            yield from ((codec, m) for m in models)
            return

        model = solver.get_model()  # Save for analysis below.

    # Search for maximally stuttering DFAs.
    models = order_models_by_stutter(solver_fact, codec, clauses, model)
    yield from ((codec, m) for m in models)


def find_dfa(
        accepting: list[Word],
        rejecting: list[Word],
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
        order_by_stutter: bool = False,
        alphabet: frozenset = None,
) -> Optional[DFA]:
    """Finds a minimal dfa that is consistent with the labeled examples.

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

    Returns:
      Either a DFA consistent with accepting and rejecting or None
      indicating that no DFA exists.
    """
    all_dfas = find_dfas(
        accepting, rejecting, solver_fact, sym_mode, extra_clauses, bounds,
        order_by_stutter, alphabet
    )
    return next(all_dfas, None)


def order_models_by_stutter(
        solver_fact,
        codec: Codec,
        clauses: list[list[int]],
        model: list[int],
) -> Iterable[DFA]:
    # Compute the maximum id used in codec or by extra clauses.
    top_id = max(map(max, clauses))
    top_id = max(codec.max_id, top_id)

    # Compute parent relation variables that don't stutter.
    lits = codec.non_stutter_lits

    # Binary search for min non-stutter using cardinality constraints.

    def non_stutter_count(model) -> int:
        return sum(model[x - 1] > 0 for x in lits)

    def find_models(bound: int, make_formula):
        formula = make_formula(lits=lits, bound=bound, top_id=top_id)

        with solver_fact(bootstrap_with=clauses) as solver:
            solver.append_formula(formula, no_return=True)
            if not solver.solve():
                return
            yield from solver.enum_models()

    candidate_bound = non_stutter_count(model)  # Candidate upper bound.
    hi = candidate_bound     # Also upper bounds lower bound.
    lo = 0
    if hasattr(codec, 'n_colors'):
        lo = codec.n_colors - 1  # Each node needs to be visited.
    elif hasattr(codec, 'codecs'):
        lo = sum(c.n_colors - 1 for c in codec.codecs)
    while lo < hi:
        mid = (lo + hi) // 2
        models = find_models(mid, CardEnc.atmost)
        witness = next(models, None)
        if witness is not None:
            hi = non_stutter_count(witness)
            assert hi <= mid
        else:
            lo = mid + 1

    # Incrementally emit models with less stutter.
    naive_bound = len(lits)
    for bound in range(lo, naive_bound + 1):
        if bound > candidate_bound:
            witness = next(find_models(bound, CardEnc.atmost), None)
            if witness is None:
                break
            candidate_bound = non_stutter_count(witness)

        yield from find_models(bound, CardEnc.equals)


def pareto_search(gen_models,
                  num_dfas: int,
                  min_size: int = 1,
                  max_size: int | None = None,
                  upperbound: int | None = None,
                  allow_unminimized: bool = False) -> Encodings:
    if max_size is None: max_size = float('inf')
    if upperbound is None: upperbound = float('inf')

    # Initial frontier with smallest dfa sizes.
    sizes = num_dfas * (min_size,)
    frontier = deque([(sizes, gen_models(sizes), False)])

    while frontier:  # while not empty.
        sizes, models, emitted_solutions = frontier.popleft()
        model = next(models, None)

        if model is not None:
            yield model
            frontier.append((sizes, models, True))
            continue  # Round robin frontier.

        if emitted_solutions and not allow_unminimized:
            continue

        # Replace exhausted node with successors.
        for i in range(num_dfas):
            new_sizes = list(sizes)
            new_sizes[i] += 1

            if new_sizes[i] > max_size:
                continue  # Terminate due to user defined max size.

            if (not emitted_solutions) and (new_sizes[i] > upperbound):
                continue  # Early terminate search for solutions.

            if any(s1 > s2 for s1, s2 in fn.pairwise(new_sizes)):
                continue  # Require increasing order.

            if any(all(s1 <= s2 for s1, s2 in zip(new_sizes, prev_sizes))
                   for prev_sizes, *_ in frontier):
                continue  # (weakly) Dominated by size tuple on frontier.

            frontier.append((new_sizes, gen_models(new_sizes), False))


__all__ = ['DFA', 'find_dfas', 'find_dfa']
