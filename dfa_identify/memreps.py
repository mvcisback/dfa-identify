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
from dfa.utils import find_equiv_counterexample, find_subset_counterexample

from dfa_identify.graphs import Word, APTA
from dfa_identify.identify import find_dfa, find_dfas, find_different_dfas
from dfa_identify.encoding import dfa_id_encodings, Codec, SymMode


@attr.s(auto_detect=True, auto_attribs=True)
class QuerySet:
    membership_queries: List
    preference_queries: List
    info_scores: Dict
    user_scores: Dict
    scoring_weight: float
    all_queries: List = []
    query_probabilities: np.array = []

    @staticmethod
    def construct_set(membership_queries, preference_queries,
                      scoring_function, scoring_weight, candidate_specs):
        info_scores = {"membership": [], "preference": []}
        user_scores = {"membership": [], "preference": []}
        for label, queries in [("membership", membership_queries), ("preference", preference_queries)]:
            for query in queries:
                info_scores[label].append(evaluate_query_information(query, label, candidate_specs))
                user_scores[label].append(scoring_function(query, label))
        qset = QuerySet(membership_queries, preference_queries, info_scores, user_scores, scoring_weight)
        qset.renormalize()
        return qset

    def sample_without_replacement(self):
        if len(self.membership_queries) + len(self.preference_queries) == 0:
            return None
        selected_idx = np.random.choice(len(self.all_queries), 1, p=self.query_probabilities).item()
        item = copy(self.all_queries[selected_idx])
        if selected_idx >= len(self.membership_queries):
            label = "preference"
            remove_idx = selected_idx - len(self.membership_queries)
        else:
            label = "membership"
            remove_idx = selected_idx - 1
        self.remove_item(label, remove_idx)
        return label, item

    def remove_item(self, label, idx):
        self.info_scores[label].pop(idx)
        self.user_scores[label].pop(idx)
        if label == "membership":
            self.membership_queries.pop(idx)
        else:
            self.preference_queries.pop(idx)
        if len(self.membership_queries) + len(self.preference_queries) > 0:
            self.renormalize()

    def renormalize(self):
        user_normalized = normalize(self.user_scores["membership"] + self.user_scores["preference"])
        info_normalized = np.concatenate((normalize(self.info_scores["membership"]),
                                             normalize(self.info_scores["preference"])))
        self.all_queries = self.membership_queries + self.preference_queries
        #  use softmax to compute probabilities
        self.query_probabilities = softmax(self.scoring_weight * user_normalized +
                                           (1-self.scoring_weight) * info_normalized)

def normalize(values):
    return np.array(values) / np.sum(values)

'''
Determine the entropy of a query based on responses from a set of candidates.
'''
def evaluate_query_information(query, label, candidate_specs):
    if label == "preference":
        counts = {(True, True): 0,
                  (True, False): 0,
                  (False, True): 0,
                  (False, False): 0}
        word1, word2 = query
        for candidate in candidate_specs:
            counts[(candidate.label(word1), candidate.label(word2))] += 1
        return entropy(list(counts.values()))
    else:  # we're in the membership case
        counts = {True: 0, False: 0}
        for candidate in candidate_specs:
            counts[candidate.label(query)] += 1
        return entropy(list(counts.values()))

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
        num_candidates_per_iter: int,
        preference_func: Callable[[Word, Word], Any],
        membership_func: Callable[[Word], Any],
        query_scoring_func: Callable[[Word], Any],
        query_scoring_weight: float = 0.5,
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
        all_candidate_specs = [current_spec]
        def get_queries(cand_spec, orig_spec):
            cex1 = find_subset_counterexample(cand_spec, orig_spec)  # find subset specs, if any
            if cex1 is not None:
                membership_queries.add(cex1)
                cex2 = find_subset_counterexample(orig_spec, cand_spec)
                if cex2 is not None:
                    membership_queries.add(cex2)
                    preference_queries.add((cex1, cex2))
            else:
                cex2 = find_subset_counterexample(orig_spec, cand_spec)
                membership_queries.add(cex2)

        while num_candidates_synthesized < num_candidates_per_iter and candidate_spec is not None:
            all_candidate_specs.append(candidate_spec)
            get_queries(candidate_spec, current_spec)
            num_candidates_synthesized += 1
            candidate_spec = next(candidate_spec_gen, None)
        # with the queries accumulated, order them and sample from them
        ordered_query_set = QuerySet.construct_set(list(membership_queries),
                                                   list(preference_queries), query_scoring_func,
                                                   query_scoring_weight, all_candidate_specs)
        successful_queries = 0
        while successful_queries < query_batch_size:
            query = ordered_query_set.sample_without_replacement()
            if query is None:
                break
            query_type, current_query = query
            if query_type == "preference":  # tagged earlier
                word1, word2 = current_query
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

def pac_memreps(
        epsilon: float,
        delta: float,
        sampling_func: Callable[[DFA], Optional[Word]],
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
        max_pac_iters: int = 10
) -> Optional[DFA]:
    for itr in range(max_pac_iters):  # max number of equivalence checks
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




