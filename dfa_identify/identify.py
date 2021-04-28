import random
from itertools import groupby
from typing import Optional

import funcy as fn
from pysat.solvers import Glucose4

from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import dfa_id_encodings, Codec
from dfa_identify.encoding import (
    ColorAcceptingVar,
    ColorNodeVar,
    ParentRelationVar
)


def extract_dfa(codec: Codec, apta: APTA, model: list[int]):
    # Fill in don't cares in model.
    n_tokens = len(apta.alphabet)

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

    breakpoint()

    # TODO: check accepting coloring consistent with apta.

    return

    root_color_var = min(nodes, key=lambda v: v.node)  # positive ids.
    assert root_color_var.node == 0  # Should be recover root = 0.
    start = root_color_var.color

    dfa_dict = {}
    for parent_relation in transitions:
        pass


def find_dfa(
        accepting: list[Word], 
        rejecting: list[Word],
        solver=Glucose4, 
):
    apta = APTA.from_examples(accepting=accepting, rejecting=rejecting)
    for codec, clauses in dfa_id_encodings(apta):
        with Glucose4() as solver:
            for clause in clauses:
                solver.add_clause(clause)

            if solver.solve():
                model = solver.get_model()
                return extract_dfa(codec, apta, model)
                return model
    raise ValueError('No DFA exists')
