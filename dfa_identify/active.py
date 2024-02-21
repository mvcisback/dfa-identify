from functools import cache
from itertools import combinations_with_replacement

import funcy as fn
from dfa_identify import find_dfas


def all_words(alphabet):
    """Enumerates all words in the alphabet."""
    n = 0
    while True:
        yield from combinations_with_replacement(alphabet, n)
        n += 1


def distinguishing_query(positive, negative, alphabet):
    """Return a string that seperates the two smallest (consistent) DFAs."""
    candidates = find_dfas(positive, negative, alphabet=alphabet)
    lang1 = next(candidates)

    # DFAs may represent the same language. Filter those out.
    candidates = (c for c in candidates if lang1 != c)
    lang2 = next(candidates, None)

    # Try to find a seperating word.
    if (lang1 is not None) and (lang2 is not None):
        return tuple((lang1 ^ lang2).find_word(True))

    # TODO: due to  a bug in dfa-identify allow_unminimized doesn't always work
    # so we need to come up with a word that is not in positive/negative but is
    # not constrained by positive / negative.
    constrained = set(positive) | set(negative)
    return fn.first(w for w in all_words(alphabet) if w not in constrained)


def find_dfas_active(alphabet, oracle, n_queries,
                     positive=(), negative=(), **kwargs):
    """
    Returns an iterator of DFAs consistent with passively and actively
    collected examples.

    SAT based version space learner that actively queries the oracle
    for a string that distinguishes two "minimal" DFAs, where
    minimal is lexicographically ordered in (#states, #edges).

    - positive, negative: set of accepting and rejecting words.
    - oracle: Callable taking in a word and returning {True, False, None}.
              If None, then no constraint is added.
    - n_queries: Number of queries to ask the oracle.
    - additional kwargs are passed to find_dfas.
    """
    positive, negative = list(positive), list(negative)

    # 1. Ask membership queries that distiguish remaining candidates.
    for _ in range(n_queries):
        word = distinguishing_query(positive, negative, alphabet)

        label = oracle(word)
        if label is True:    positive.append(word)
        elif label is False: negative.append(word)
        else: assert label is None  # idk case.

    # 2. Return minimal consistent DFA.
    kwargs.setdefault('order_by_stutter', True)
    kwargs.setdefault('allow_unminimized', True)
    yield from find_dfas(positive, negative, alphabet=alphabet, **kwargs)


def find_dfa_active(alphabet, oracle, n_queries,
                    positive=(), negative=(), **kwargs):
    """
    Returns minimal DFA consistent with passively and actively collected
    examples.

    SAT based version space learner that actively queries the oracle
    for a string that distinguishes two "minimal" DFAs, where
    minimal is lexicographically ordered in (#states, #edges).

    - positive, negative: set of accepting and rejecting words.
    - oracle: Callable taking in a word and returning {True, False, None}.
              If None, then no constraint is added.
    - n_queries: Number of queries to ask the oracle.
    - additional kwargs are passed to find_dfas.
    """
    dfas = find_dfas_active(positive=positive,
                            negative=negative,
                            alphabet=alphabet,
                            oracle=oracle,
                            n_queries=n_queries,
                            **kwargs)
    return fn.first(dfas)


__all__ = ['find_dfas_active',
           'find_dfa_active',
           'all_words',
           'distinguishing_query']
