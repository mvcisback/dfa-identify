from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from functools import partial
from typing import Sequence, Iterable

import funcy as fn
from more_itertools import roundrobin
from pysat.solvers import Glucose4

from dfa_identify.identify import APTA, pareto_search, order_models_by_stutter
from dfa_identify.encoding import Codec, ExtraClauseGenerator, SymMode 


def sgn(x):
    return 1 if x > 0 else -1


def offset_lits(lits: Iterable[int], offset: int) -> Sequence[int]:
    return [sgn(x) * (abs(x) + offset) for x in lits]


def offset_clauses(clauses: Iterable[Sequence[int]],
                   offset: int) -> Iterable[Sequence[int]]:
    return [offset_lits(clause, offset) for clause in clauses]


@dataclass
class ConjunctiveCodec:
    codecs: Sequence[Codec]
    extra_clauses: ExtraClauseGenerator = lambda *_: ()

    # Internal
    offsets: int = None

    def __post_init__(self):
        offsets = [0] + fn.lsums([c.max_id for c in self.codecs])
        object.__setattr__(self, "offsets", tuple(offsets))

    @property
    def apta(self) -> APTA:
        return self.codecs[0].apta

    @staticmethod
    def from_apta(apta: APTA,
                  dfa_sizes: Sequence[int],
                  extra_clauses: ExtraClauseGenerator=lambda *_: (),
                  sym_mode: SymMode = None) -> ConjunctiveCodec:
        codecs = [Codec(n_nodes=len(apta.nodes),
                        n_colors=s,
                        n_tokens=len(apta.alphabet),
                        sym_mode=sym_mode,
                        apta=apta) for s in dfa_sizes]

        # HACK: Disable coupling of accepting / rejecting in Codec.
        noop = fn.constantly(())
        for codec in codecs:
            object.__setattr__(codec, "couple_labeling_clauses", noop)

        return ConjunctiveCodec(tuple(codecs), extra_clauses)

    def interpret_model(self, model: list[int]) -> Sequence[DFA]:
        # Break up model by codec.
        models = (model[s:e] for s, e in fn.pairwise(self.offsets))
        models = [offset_lits(m, -x) for m, x in zip(models, self.offsets)]

        if len(models) != len(self.codecs):
            raise ValueError("model and codec mismatch.")

        dfas = (c.interpret_model(m) for c, m in zip(self.codecs, models))
        return tuple(dfas)

    @property
    def non_stutter_lits(self) -> int:
        return fn.lcat(c.non_stutter_lits for c in self.codecs)

    @property
    def max_id(self):
        return sum(c.max_id for c in self.codecs)

    def clauses(self):
        # Offset sub codec clauses.
        for offset, codec in zip(self.offsets, self.codecs):
            yield from offset_clauses(codec.clauses(), offset)
        
        # All DFAs must accept when example accepted.
        for n in self.apta.accepting:
            for offset, codec in zip(self.offsets, self.codecs):
                for c in range(codec.n_colors):
                    accept_lit = codec.color_accepting(c)
                    color_lit = codec.color_node(n, c)
                    clause = [-color_lit, accept_lit]
                    yield offset_lits(clause, offset)

        # At least one DFA must reject when example rejected.
        for n in self.apta.rejecting:
            for colors in product(*(range(c.n_colors) for c in self.codecs)):
                clause = []
                for offset, codec, c in zip(self.offsets, self.codecs, colors):
                    accept_lit = codec.color_accepting(c)
                    color_lit = codec.color_node(n, c)
                    subclause = [-color_lit, -accept_lit]
                    clause.extend(offset_lits(subclause, offset))
                yield clause

        yield from self.extra_clauses(self.apta, self)


def find_conjunction_of_dfas(
        accepting: list[Word],
        rejecting: list[Word],
        n_dfas: int,
        solver_fact=Glucose4,
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
        order_by_stutter: bool = False,
        alphabet: frozenset = None,
        allow_unminimized: bool = False,
) -> Iterable[DFA]:
    """Finds all conjuncive dfa combinations that are consistent with the
    labeled examples.

    Here "conjunctive" means that the language is the intersection of
    the dfas languages. This corresponds to all dfas needing accept accepting
    words and at least one rejecting a rejecting word.

    Inputs:
      - accepting: A sequence of "words" to be accepted.
      - rejecting: A sequence of "words" to be rejected.
      - n_dfas: Number of DFAs to search conjunction over.
      - solver: A py-sat API compatible object for solving CNF SAT queries.
      - bounds: DFA size range (inclusive) to restrict search to, e.g.,
                - (None, 10): DFA can have as most 10 states.
                - (2, None): DFA must have at least 2 states.
                - (2, 10):  DFA must have between 2 and 10 states.
                - (None, None): No constraints (default).
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
                         n_dfas=n_dfas,
                         solver_fact=solver_fact,
                         extra_clauses=extra_clauses,
                         bounds=bounds,
                         order_by_stutter=order_by_stutter,
                         alphabet=alphabet,
                         allow_unminimized=allow_unminimized)
    yield from (codec.interpret_model(m) for codec, m in models)



def find_models(
        accepting: list[Word],
        rejecting: list[Word],
        n_dfas: int,
        solver_fact=Glucose4,
        extra_clauses: ExtraClauseGenerator = lambda *_: (),
        bounds: Bounds = (None, None),
        order_by_stutter: bool = False,
        alphabet: frozenset = None,
        allow_unminimized: bool = False,
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
            'n_dfas': num_dfas,
        }
        models_pos = find_models(accepting=[()], rejecting=[  ], **kwargs)
        models_neg = find_models(accepting=[  ], rejecting=[()], **kwargs)
        yield from roundrobin(models_pos, models_neg)
        return 

    apta = APTA.from_examples(accepting=accepting,
                              rejecting=rejecting,
                              alphabet=alphabet)

    low, high = bounds
    if (low is not None) and (high is not None) and (high < low):
        raise ValueError('Empty bound range!')

    # Tighten lower bound.
    if low is None: low = 1

    gen_models = partial(_gen_models,
                         apta=apta,
                         solver_fact=solver_fact,
                         extra_clauses=extra_clauses,
                         order_by_stutter=order_by_stutter)

    yield from pareto_search(gen_models,
                             num_dfas=n_dfas,
                             min_size=low,
                             max_size=high,
                             upperbound=len(apta.nodes),
                             allow_unminimized=allow_unminimized)


def _gen_models(sizes, apta, extra_clauses, solver_fact, order_by_stutter):
    codec = ConjunctiveCodec.from_apta(apta,
                                       sizes,
                                       sym_mode="bfs",
                                       extra_clauses=extra_clauses)

    clauses = list(codec.clauses())

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


__all__ = ['find_conjunction_of_dfas']
