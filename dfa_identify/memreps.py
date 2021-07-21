import random
from itertools import groupby
import attr
import numpy as np
from scipy.special import softmax
from typing import Any, Optional, Tuple, Callable
from copy import copy
from dfa import DFA
from dfa.utils import find_equiv_counterexample, find_subset_counterexample

from dfa_identify.graphs import Word, APTA
from dfa_identify.identify import find_dfa, find_dfas, find_different_dfas
from dfa_identify.encoding import dfa_id_encodings, Codec, SymMode

@attr.s(auto_detect=True, auto_attribs=True)
class QuerySet:
    all_queries: np.array
    query_scores: np.array
    query_probabilities: np.array

    @staticmethod
    def construct_set(all_queries, scoring_function):
        all_queries = list(all_queries)
        query_scores = np.zeros(len(all_queries))
        for query_idx in range(len(all_queries)):
            query_scores[query_idx] = scoring_function(all_queries[query_idx])
        qset = QuerySet(all_queries, query_scores, [])
        qset.renormalize()
        return qset

    def sample_without_replacement(self):
        if len(self.all_queries) == 0:
            return None
        selected_idx = np.random.choice(len(self.all_queries), 1, p=self.query_probabilities).item()
        item = copy(self.all_queries[selected_idx])
        self.remove_item(selected_idx)
        return item

    def remove_item(self, idx):
        self.all_queries.pop(idx)
        self.query_scores = np.delete(self.query_scores, idx)
        if len(self.query_scores) > 0:
            self.renormalize()

    def renormalize(self):
        #  use softmax to compute probabilities
        self.query_probabilities = softmax(self.query_scores)


'''
Main implementation of the MemRePs algorithm.
Hyperparameters such as the number of total iterations, the number
of queries that should be successfully asked per iteration, and the 
symmetry breaking mode have been included, as well as parameters for
initial accepting, rejecting, and preference sets.
'''
def run_memreps_with_data(
        accepting: list[Word],
        rejecting: list[Word],
        max_num_iters: int,
        num_candidates_per_iter: int,
        preference_func,
        membership_func,
        query_scoring_func,
        query_batch_size: int = 1,
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
    itr = 0


    while itr < max_num_iters:  # outer loop
        all_queries = set([])
        # find k different specs from the current. If none exist, we have a unique spec
        candidate_spec_gen = find_different_dfas(current_spec, accepting, rejecting, ordered_preference_words,
                           incomparable_preference_words, sym_mode=sym_mode)
        candidate_spec = next(candidate_spec_gen, None)
        if candidate_spec is None:
            # no other minimal DFAs are consistent with this spec
            return current_spec, (accepting, rejecting,
                                  ordered_preference_words,
                                  incomparable_preference_words)
        num_candidates_synthesized = 0

        def get_queries(cand_spec, orig_spec):
            cex1 = find_subset_counterexample(cand_spec, orig_spec)  # find subset specs, if any
            if cex1 is not None:
                all_queries.add(("membership", cex1))
                cex2 = find_subset_counterexample(orig_spec, cand_spec)
                if cex2 is not None:
                    all_queries.add(("membership", cex2))
                    all_queries.add(("preference", (cex1, cex2)))
            else:
                cex2 = find_subset_counterexample(orig_spec, cand_spec)
                all_queries.add(("membership", cex2))

        while num_candidates_synthesized < num_candidates_per_iter and candidate_spec is not None:
            get_queries(candidate_spec, current_spec)
            num_candidates_synthesized += 1
            candidate_spec = next(candidate_spec_gen, None)
        # with the queries accumulated, order them and sample from them
        ordered_query_set = QuerySet.construct_set(all_queries, query_scoring_func)
        successful_queries = 0
        while successful_queries < query_batch_size:
            query = ordered_query_set.sample_without_replacement()
            #breakpoint()
            if query is None:
                break
            query_type, current_query = query
            if query_type == "preference":  # tagged earlier
                word1, word2 = current_query
                #breakpoint()
                response = preference_func(word1, word2)
                if response == "incomparable":
                    if strong_memrep:
                        successful_queries += 1
                        incomparable_preference_words.append(current_query)
                elif response != "unknown":
                    successful_queries += 1
                    ordered_preference_words.append(response)
            else:  # we're in a membership query
                response = membership_func(current_query)
                if response != "unknown":
                    successful_queries += 1
                    accepting.append(current_query) if response else rejecting.append(current_query)
            #breakpoint()
        # now, resynthesize a spec that is consistent with the new information
        current_spec = find_dfa(accepting, rejecting, ordered_preference_words,
                                incomparable_preference_words, sym_mode=sym_mode)
        itr += 1
    return current_spec, (accepting, rejecting, ordered_preference_words, incomparable_preference_words)


'''
MemRePs wrapper that takes in an equivalence function (oracle) and finds an equivalent
automata, if possible within the allotted resources.
'''
def equivalence_oracle_memreps(
        equivalence_func: Callable[[DFA], Optional[Word]],
        accepting: list[Word],
        rejecting: list[Word],
        max_num_iters: int,
        num_candidates_per_iter: int,
        preference_func: Callable[[Word, Word], Any],
        membership_func: Callable[[Word], Any],
        query_scoring_func,
        query_batch_size: int = 1,
        strong_memrep: bool = True,
        ordered_preference_words: list[Tuple[Word, Word]] = None,
        incomparable_preference_words: list[Tuple[Word, Word]] = None,
        sym_mode: SymMode = "clique",
        num_equiv_iters: int = 10
) -> Optional[DFA]:
    for itr in range(num_equiv_iters):  # max number of equivalence checks
        candidate_dfa = run_memreps_naive(accepting, rejecting, max_num_iters,
                                          num_candidates_per_iter, preference_func,
                                          membership_func, query_scoring_func,
                                          query_batch_size=query_batch_size, strong_memrep=strong_memrep,
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


'''
MemRePs wrapper that returns only the DFA, and not the augmented datasets.
'''
def run_memreps_naive(
        accepting: list[Word],
        rejecting: list[Word],
        max_num_iters: int,
        num_candidates_per_iter: int,
        preference_func,
        membership_func,
        query_scoring_func,
        query_batch_size: int = 1,
        strong_memrep: bool = True,
        ordered_preference_words: list[Tuple[Word, Word]] = None,
        incomparable_preference_words: list[Tuple[Word, Word]] = None,
        sym_mode: SymMode = "clique"
) -> Optional[DFA]:
    candidate_dfa, data = run_memreps_with_data(accepting, rejecting, max_num_iters,
                                                num_candidates_per_iter, preference_func,
                                                membership_func, query_scoring_func,
                                                query_batch_size=query_batch_size, strong_memrep=strong_memrep,
                                                ordered_preference_words=ordered_preference_words,
                                                incomparable_preference_words=incomparable_preference_words,
                                                sym_mode=sym_mode)
    return candidate_dfa








