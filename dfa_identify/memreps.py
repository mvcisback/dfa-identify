import random
import math
from itertools import groupby
import attr
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from typing import Any, Optional, Tuple, Callable, List, Dict
from copy import copy
from dfa import DFA
from dfa.utils import find_equiv_counterexample, find_subset_counterexample, find_word

from dfa_identify.graphs import Word, APTA
from dfa_identify.identify import find_dfa, find_dfas, find_different_dfas
from dfa_identify.encoding import dfa_id_encodings, Codec, SymMode

'''
Given a set of specifications, we want to find words that maximize the information gain
amongst all of the specifications provided.
For now, we've provided an implementation for k=2, with a generalization to k>2 to be done.
'''
def select_words_from_specs(all_specs: List[DFA]):
    if len(all_specs) > 2:
        raise NotImplementedError("Word selection for k > 2 not yet implemented!")
    spec1, spec2 = all_specs[0], all_specs[1]
    spec12 = (spec1 ^ spec2)  # take the symmetric difference of the two specs
    membership_word = find_word(spec12)
    if membership_word is None:
        raise AssertionError("No distinguishing words available for 2 equivalent specifications")
    preference_word = find_word(~spec12)  # find a word that isn't in the symmetric difference
    return membership_word, preference_word

def normalize(values):
    return np.array(values) / np.sum(values)

'''
contextual bandits. starting with a simple implementation with no actual bandits logic
encoded
'''
class ContextualExpertBandits:

    def __init__(self,
                 experts: List[Callable],
                 eta: float,
                 reward_func: Callable,
                 gamma: float):
        self.experts = experts
        self.eta = eta
        self.gamma = gamma
        self.expert_weights = np.tile([1.0/len(experts)], len(experts))

    def selection(self, arms):
        arm_weights = np.zeros(len(arms))
        # get the weights from all the experts
        for idx in range(len(self.experts)):
            expert_scores = self.experts[idx](arms)
            arm_weights += self.expert_weights[idx] * expert_scores
        return arms[np.random.choice(len(arms), p=normalize(arm_weights))]

    def update(self, arm_idx, reward):
        #TODO: implement update function and update weights
        pass

def expert_func(all_arms):
    #TODO: replace this placeholder with actual experts
    return np.array([0.5, 0.5])

def reward_func(m_specs):
    #TODO: replace this placeholder with actual reward function
    return 1

'''
Main implementation of the MemRePs algorithm.
Hyperparameters such as the number of total iterations, the number
of queries that should be successfully asked per iteration, and the 
symmetry breaking mode have been included, as well as parameters for
initial accepting, rejecting, and preference sets.
'''
def run_memreps_with_data(
        accepting: List[Word],
        rejecting: List[Word],
        max_num_iters: int,
        preference_func: Callable[[Word, Word], Any],
        membership_func: Callable[[Word], Any],
        preference_weight: float = 0.5,
        membership_weight: float = 0.5,
        k_specs: int = 2,
        m_specs: int = 10,
        strong_memrep: bool = True,
        ordered_preference_words: list[Tuple[Word, Word]] = None,
        incomparable_preference_words: list[Tuple[Word, Word]] = None,
        sym_mode: SymMode = "clique"
) -> Tuple[Optional[DFA], Tuple]:
    if ordered_preference_words is None:
        ordered_preference_words = []
    if incomparable_preference_words is None:
        incomparable_preference_words = []

    # first, find a starter spec consistent with the initial data
    current_spec = find_dfa(accepting, rejecting, ordered_preference_words,
                            incomparable_preference_words, sym_mode=sym_mode)

    # establish contextual bandits
    bandits = ContextualExpertBandits([expert_func], 1, reward_func, 1)
    for _ in range(max_num_iters):  # outer loop
        preference_queries = set([])
        membership_queries = set([])
        # find k different specs from the current. If none exist, we have a unique spec
        candidate_spec_gen = find_different_dfas(current_spec, accepting, rejecting,
                                                 ordered_preference_words,
                                                 incomparable_preference_words, sym_mode=sym_mode)
        candidate_spec = next(candidate_spec_gen, None)
        if candidate_spec is None:
            # no other minimal DFAs are consistent with this spec
            return current_spec, (accepting, rejecting,
                                  ordered_preference_words,
                                  incomparable_preference_words)
        num_candidates_synthesized = 0
        kspec_list = [current_spec]
        mspec_list = [current_spec]
        total_num_specs = max(k_specs, m_specs)
        while num_candidates_synthesized < total_num_specs and candidate_spec is not None:
            while len(kspec_list) < k_specs:
                kspec_list.append(candidate_spec)
            while len(mspec_list) < m_specs:
                mspec_list.append(candidate_spec)
            num_candidates_synthesized += 1
            candidate_spec = next(candidate_spec_gen, None)
        # generate the words for either a membership or preference query
        membership_word, preference_word = select_words_from_specs(kspec_list)
        # use the contextual bandits algorithm to determine which query to ask
        # the two arms are the membership query and preference query respectively
        query = bandits.selection([('membership', membership_word),
                                   ('preference', (membership_word, preference_word))])
        # ask the query and update our data with the response
        query_type, current_query = query
        if query_type == "preference":  # tagged earlier
            response = preference_func(current_query[0], current_query[1])
            if response == "incomparable":
                if strong_memrep:
                    incomparable_preference_words.append(current_query)
            elif response != "unknown":
                ordered_preference_words.append(response)
        else:  # we're in a membership query
            response = membership_func(current_query)
            if response != "unknown":
                accepting.append(current_query) if response else rejecting.append(current_query)
        # now, resynthesize a spec that is consistent with the new information
        current_spec = find_dfa(accepting, rejecting, ordered_preference_words,
                                incomparable_preference_words, sym_mode=sym_mode)
    print("Elapsed maximum number of iterations. Returning current consistent specification.")
    return current_spec, (accepting, rejecting, ordered_preference_words, incomparable_preference_words)


'''
MemRePs wrapper that takes in an equivalence function (oracle) and finds an equivalent
automata, if possible within the allotted resources.
'''
def equivalence_oracle_memreps(
        equivalence_func: Callable[[DFA], Optional[Word]],
        accepting: List[Word],
        rejecting: List[Word],
        max_num_iters: int,
        preference_func: Callable[[Word, Word], Any],
        membership_func: Callable[[Word], Any],
        preference_weight: float = 0.5,
        membership_weight: float = 0.5,
        k_specs: int = 2,
        m_specs: int = 10,
        strong_memrep: bool = True,
        ordered_preference_words: list[Tuple[Word, Word]] = None,
        incomparable_preference_words: list[Tuple[Word, Word]] = None,
        sym_mode: SymMode = "clique",
        num_equiv_iters: int = 10
) -> Optional[DFA]:
    for itr in range(num_equiv_iters):  # max number of equivalence checks
        candidate_dfa = run_memreps_naive(accepting, rejecting, max_num_iters,
                                          preference_func, membership_func,
                                          preference_weight=preference_weight,
                                          membership_weight=membership_weight,
                                          k_specs=k_specs, m_specs=m_specs,
                                          strong_memrep=strong_memrep,
                                          ordered_preference_words=ordered_preference_words,
                                          incomparable_preference_words=incomparable_preference_words,
                                          sym_mode=sym_mode)
        if candidate_dfa is None:
            print("Error: no DFA available that is consistent with data.")
            return None
        # check: did we find the true DFA?
        counterexample = equivalence_func(candidate_dfa)
        if counterexample is not None:
            # the counterexample is incorrectly labeled by our candidate
            # so add it to the opposite set
            rejecting.append(counterexample) if candidate_dfa.label(counterexample)\
                else accepting.append(counterexample)
        else:
            return candidate_dfa
    # we weren't able to find an equivalent DFA in the number of iterations allowed
    print("Maximum allotted iterations have elapsed without finding equivalent DFA.")
    return None

def pac_memreps(
        epsilon: float,
        delta: float,
        sampling_func: Callable[[DFA], Optional[Word]],
        accepting: List[Word],
        rejecting: List[Word],
        max_num_iters: int,
        preference_func: Callable[[Word, Word], Any],
        membership_func: Callable[[Word], Any],
        preference_weight: float = 0.5,
        membership_weight: float = 0.5,
        k_specs: int = 2,
        m_specs: int = 10,
        strong_memrep: bool = True,
        ordered_preference_words: list[Tuple[Word, Word]] = None,
        incomparable_preference_words: list[Tuple[Word, Word]] = None,
        sym_mode: SymMode = "clique",
        max_pac_iters: int = 10
) -> Optional[DFA]:
    for itr in range(max_pac_iters):  # max number of equivalence checks
        candidate_dfa = run_memreps_naive(accepting, rejecting, max_num_iters,
                                          preference_func, membership_func,
                                          preference_weight=preference_weight,
                                          membership_weight=membership_weight,
                                          k_specs=k_specs, m_specs=m_specs,
                                          strong_memrep=strong_memrep,
                                          ordered_preference_words=ordered_preference_words,
                                          incomparable_preference_words=incomparable_preference_words,
                                          sym_mode=sym_mode)
        if candidate_dfa is None:
            print("Error: no DFA available that is consistent with data.")
            return None
        # compute N in order to perform the PAC check on the candidate DFA
        num_samples = int(np.log(delta / 2) / (-1 * 2 * epsilon ** 2))
        counterexample = None
        for i in range(num_samples):
            counterexample = sampling_func(candidate_dfa)
            if counterexample is not None:  # we are violating the pac guarantee
                rejecting.append(counterexample) if candidate_dfa.label(counterexample) \
                    else accepting.append(counterexample)
                break
        if counterexample is None:  # we got through all of the samples without breaking
            return candidate_dfa
    # we weren't able to find an equivalent DFA in the number of iterations allowed
    print("Maximum allotted iterations have elapsed without finding pac DFA.")
    return None


'''
MemRePs wrapper that returns only the DFA, and not the augmented datasets.
'''
def run_memreps_naive(
        accepting: List[Word],
        rejecting: List[Word],
        max_num_iters: int,
        preference_func: Callable[[Word, Word], Any],
        membership_func: Callable[[Word], Any],
        preference_weight: float = 0.5,
        membership_weight: float = 0.5,
        k_specs: int = 2,
        m_specs: int = 10,
        strong_memrep: bool = True,
        ordered_preference_words: list[Tuple[Word, Word]] = None,
        incomparable_preference_words: list[Tuple[Word, Word]] = None,
        sym_mode: SymMode = "clique"
) -> Optional[DFA]:
    candidate_dfa, data = run_memreps_with_data(accepting, rejecting, max_num_iters,
                                                preference_func, membership_func,
                                                preference_weight=preference_weight,
                                                membership_weight=membership_weight,
                                                k_specs=k_specs, m_specs=m_specs,
                                                strong_memrep=strong_memrep,
                                                ordered_preference_words=ordered_preference_words,
                                                incomparable_preference_words=incomparable_preference_words,
                                                sym_mode=sym_mode)
    return candidate_dfa




