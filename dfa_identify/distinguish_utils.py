import random
from itertools import groupby
from typing import Optional, Tuple

import funcy as fn
from lstar.learn import learn_dfa
from dfa import DFA
from dfa.utils import find_equiv_counterexample

from dfa_identify.graphs import Word, APTA
from dfa_identify.encoding import dfa_id_encodings, Codec
from dfa_identify.encoding import (
    ColorAcceptingVar,
    ColorNodeVar,
    ParentRelationVar
)
from dfa_identify.identify import find_dfa

def get_augmented_set(original_dfa: DFA, current_accepting: set,
                      current_rejecting: set) -> Tuple[set, set]:
    '''
    Creates sets that are disjoint from the current accepting and rejecting
    '''
    augmented_accepting, augmented_rejecting = set(), set()

    def label_wrapper(word):
        if original_dfa.label(word):
            if "".join(word) not in current_accepting:
                augmented_accepting.add("".join(word))
            return True
        else:
            if "".join(word) not in current_rejecting:
                augmented_rejecting.add("".join(word))
            return False

    def equiv_wrapper(dfa_candidate):
        return find_equiv_counterexample(dfa_candidate, original_dfa)
    learn_dfa(inputs=original_dfa.inputs, label=label_wrapper,
              find_counter_example=equiv_wrapper, outputs=[True, False])
    return augmented_accepting, augmented_rejecting