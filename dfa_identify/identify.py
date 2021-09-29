from itertools import groupby
from typing import Optional, Iterable

from dfa import dict2dfa, DFA
from pysat.solvers import Glucose4
from pysat.card import CardEnc

from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import dfa_id_encodings, Codec, SymMode, Encodings
from dfa_identify.encoding import Bounds, ExtraClauseGenerator
from dfa_identify.encoding import (
    ColorAcceptingVar,
    ColorNodeVar,
    ParentRelationVar
)


def max_stuttering_dfas(
        solver_fact,
        codec: Codec,
        clauses: list[list[int]],
        model: list[int],
) -> Iterable[DFA]:
    # Compute parent relation variables that don't stutter.
    lits = []
    for lit in range(1 + codec.offsets[2], codec.offsets[3] + 1):
        par_rel = codec.decode(lit)
        assert isinstance(par_rel, ParentRelationVar)
        if par_rel.node_color == par_rel.parent_color:
            continue
        lits.append(lit)

    top_id = codec.offsets[-1]

    def find_models(bound: int):
        card_formula = CardEnc.atmost(lits=lits, bound=bound, top_id=top_id)

        with solver_fact(bootstrap_with=clauses) as solver:
            if solver.supports_atmost():
                solver.add_atmost(lits, bound)
            else:
                solver.append_formula(card_formula, no_return=False)
            if not solver.solve():
                return
            yield from solver.enum_models()

    def non_stutter_count(model) -> int:
        return sum(model[x] > 0 for x in lits)

    hi = non_stutter_count(model)
    lo = codec.n_colors - 1

    # Binary search using cardinality constraints
    while lo < hi:
        mid = (lo + hi) //  2
        models = find_models(mid) 
        model = next(models, None)
        if model is not None:
            hi = non_stutter_count(model)
            assert hi <= mid
        else:
            lo = mid + 1
    assert next(find_models(lo), None) is not None
    yield from find_models(lo)


def extract_dfa(codec: Codec, apta: APTA, model: list[int]) -> DFA:
    # Fill in don't cares in model.
    decoded = map(codec.decode, model)
    decoded = list(decoded)
    var_groups = groupby(decoded, type)

    group1 = next(var_groups)
    assert group1[0] == ColorAcceptingVar
    accepting = {v.color for v in group1[1] if v.true}

    group2 = next(var_groups)
    assert group2[0] == ColorNodeVar

    node2color = {}
    for var in group2[1]:
        if not var.true:
            continue
        assert var.node not in node2color
        node2color[var.node] = var.color

        if var.color in accepting:
            assert apta.tree.nodes[var.node].get('label', True)

    group3 = next(var_groups)
    assert group3[0] == ParentRelationVar
    dfa_dict = {}
    token2char = apta.alphabet.inv
    for var in group3[1]:
        if not var.true:
            continue
        default = (var.parent_color in accepting, {})
        (_, char2node) = dfa_dict.setdefault(var.parent_color, default)
        char = token2char[var.token]
        assert char not in char2node
        char2node[char] = var.node_color
    return dict2dfa(dfa_dict, start=node2color[0])


def find_dfas(
        accepting: list[Word],
        rejecting: list[Word],
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
        minimum_ns_edges: bool = False
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

    Returns:
      An iterable of all minimal DFA consistent with accepting and rejecting.
    """
    apta = APTA.from_examples(accepting=accepting, rejecting=rejecting)
    encodings = dfa_id_encodings(
        apta=apta, sym_mode=sym_mode,
        extra_clauses=extra_clauses, bounds=bounds)

    for codec, clauses in encodings:
        with solver_fact(bootstrap_with=clauses) as solver:
            if not solver.solve():
                continue
            if not minimum_ns_edges:
                models = solver.enum_models()
                yield from (extract_dfa(codec, apta, m) for m in models)
                return

            model = solver.get_model()  # Save for analysis below.

        # Search for maximally stuttering DFAs.
        assert minimum_ns_edges
        models = max_stuttering_dfas(solver_fact, codec, clauses, model)
        yield from (extract_dfa(codec, apta, m) for m in models)


def find_dfa(
        accepting: list[Word],
        rejecting: list[Word],
        solver_fact=Glucose4,
        sym_mode: SymMode = "bfs",
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
        minimum_ns_edges: bool = False
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

    Returns:
      Either a DFA consistent with accepting and rejecting or None
      indicating that no DFA exists.
    """
    all_dfas = find_dfas(
        accepting, rejecting, solver_fact, sym_mode, extra_clauses, bounds,
        minimum_ns_edges 
    )
    return next(all_dfas, None)


__all__ = ['find_dfas', 'find_dfa', 'extract_dfa']
