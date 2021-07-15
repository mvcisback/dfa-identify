import random
from itertools import groupby
import attr
import numpy as np
from scipy.special import softmax
from typing import Any, Optional, Tuple
from copy import copy

import funcy as fn
from lstar.learn import learn_dfa
from dfa import DFA
from dfa.utils import find_equiv_counterexample, find_subset_counterexample

from dfa_identify.graphs import Word, APTA
from dfa_identify.identify import find_dfa, find_different_dfas
from dfa_identify.encoding import dfa_id_encodings, Codec

@attr.s(auto_detect=True)
class QuerySet:
    all_queries: list[Any]
    query_scores: np.array
    query_probabilities: np.array

    @staticmethod
    def construct_set(all_queries, scoring_function):
        query_scores = np.zeros(10)
        for query_idx in range(len(all_queries)):
            query_scores[query_idx] = scoring_function(all_queries[query_idx])
        qset = QuerySet(all_queries, query_scores, [])
        qset.renormalize()
        return qset

    def sample_without_replacement(self):
        if len(self.all_queries) == 0:
            return None
        selected_idx = np.random.choice(len(self.all_queries), 1, p=self.query_probabilities)
        item = copy(self.all_queries[selected_idx])
        self.remove_item(selected_idx)
        return item

    def remove_item(self, idx):
        self.all_queries = np.delete(self.all_queries, idx)
        self.query_scores = np.delete(self.query_scores, idx)
        self.renormalize()

    def renormalize(self):
        #  use softmax to compute probabilities
        self.query_probabilities = softmax(self.query_scores)

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
) -> Optional[DFA]:
    # first, find a starter spec consistent with the initial data
    current_spec = find_dfa(accepting, rejecting, ordered_preference_words,
                            incomparable_preference_words)
    iter = 0


    while iter < max_num_iters:  # outer loop
        all_queries = []
        # find k different specs from the current. If none exist, we have a unique spec
        candidate_spec_gen = find_different_dfas(current_spec, accepting, rejecting, ordered_preference_words,
                           incomparable_preference_words)
        candidate_spec = next(candidate_spec_gen, None)
        if candidate_spec is None:
            return current_spec  # no other minimal DFAs are consistent with this spec
        num_candidates_synthesized = 0

        def get_queries(cand_spec, orig_spec):
            cex1 = find_subset_counterexample(cand_spec, orig_spec)  # find subset specs, if any
            if cex1 is not None:
                all_queries.append(cex1)
                cex2 = find_subset_counterexample(orig_spec, cand_spec)
                if cex2 is not None:
                    all_queries.append(cex2)
                    all_queries.append((cex1, cex2))
            else:
                cex2 = find_subset_counterexample(orig_spec, cand_spec)
                all_queries.append(cex2)

        while num_candidates_synthesized < num_candidates_per_iter and candidate_spec is not None:
            get_queries(candidate_spec, current_spec)
            num_candidates_synthesized += 1
            candidate_spec = next(candidate_spec_gen, None)
        # with the queries accumulated, order them and sample from them
        ordered_query_set = QuerySet.construct_set(all_queries, query_scoring_func)
        successful_queries = 0
        while successful_queries < query_batch_size:
            current_query = ordered_query_set.sample_without_replacement()
            if current_query is None:
                break
            if type(current_query) == tuple:  # hacky, but tells us if we're in a preference query
                response = preference_func(current_query)
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

        # now, resynthesize a spec that is consistent with the new information
        current_spec = find_dfa(accepting, rejecting, ordered_preference_words,
                                incomparable_preference_words)

    return current_spec








