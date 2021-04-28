import random
from itertools import groupby
from typing import Optional

import funcy as fn
from dfa import dict2dfa, DFA
from pysat.solvers import Glucose4

from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import dfa_id_encodings, Codec
from dfa_identify.encoding import (
    ColorAcceptingVar,
    ColorNodeVar,
    ParentRelationVar
)


def extract_dfa(codec: Codec, apta: APTA, model: list[int]) -> DFA:
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


def find_dfa(
        accepting: list[Word], 
        rejecting: list[Word],
        solver=Glucose4, 
) -> Optional[DFA]:
    apta = APTA.from_examples(accepting=accepting, rejecting=rejecting)
    for codec, clauses in dfa_id_encodings(apta):
        with Glucose4() as solver:
            for clause in clauses:
                solver.add_clause(clause)

            if solver.solve():
                model = solver.get_model()
                return extract_dfa(codec, apta, model)
                return model
