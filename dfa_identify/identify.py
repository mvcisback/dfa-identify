from typing import Optional

import funcy as fn
from pysat.solvers import Glucose3

from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import dfa_id_encodings


def find_dfa(
        accepting: list[Word], 
        rejecting: list[Word],
        solver=Glucose3, 
):
    apta = APTA.from_examples(accepting=accepting, rejecting=rejecting)
    for codec, clauses in dfa_id_encodings(apta):
        for clause in clauses:
            solver = Glucose3()
            solver.add_clause(clause)

        if solver.solve():
            model = solver.get_model()
            return model
