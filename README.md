# dfa-identify
Python library for identifying (learning) minimal DFAs from labeled examples
by reduction to SAT.

[![Build Status](https://cloud.drone.io/api/badges/mvcisback/dfa-identify/status.svg)](https://cloud.drone.io/mvcisback/dfa-identify)
[![PyPI version](https://badge.fury.io/py/dfa-identify.svg)](https://badge.fury.io/py/dfa-identify)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Encoding](#encoding)
- [Goals and related libraries](#goals-and-related-libraries)

# Installation

If you just need to use `dfa`, you can just run:

`$ pip install dfa`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Usage

`dfa_identify` is centered around the `find_dfa` and `find_dfas` function. Both take in
sequences of accepting and rejecting "words", where are word is a
sequence of arbitrary python objects. 

1. `find_dfas` returns all minimally sized (no `DFA`s exist of size
smaller) consistent with the given labeled data.

2. `find_dfa` returns an arbitrary (first) minimally sized `DFA`.

The returned `DFA` object is from the [dfa](https://github.com/mvcisback/dfa) library.

```python
from dfa_identify import find_dfa


accepting = ['a', 'abaa', 'bb']
rejecting = ['abb', 'b']
    
my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

assert all(my_dfa.label(x) for x in accepting)
assert all(not my_dfa.label(x) for x in rejecting)
```

Because words are sequences of arbitrary python objects, the
identification problem, with `a` ↦ 0 and `b` ↦ 1, is given below:


```python
accepting = [[0], [0, 'z', 0, 0], ['z', 'z']]
rejecting = [[0, 'z', 'z'], ['z']]

my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)
```

# Active learning

There are also active variants of `find_dfa` and `find_dfas` called
`find_dfa_active` and `find_dfas_active` resp. 

An example from the unit tests:

```python
import dfa
from dfa_identify import find_dfa_active


def oracle(word):
    return sum(word) % 2 == 0

lang = find_dfa_active(alphabet=[0, 1],
                       oracle=oracle,
                       n_queries=20)
assert lang == dfa.DFA(
    inputs=[0,1],
    label=lambda s: s,
    transition=lambda s, c: s ^ bool(c),
    start=True
)

# Can also send in positive and negative examples:
lang = find_dfa_active(alphabet=[0, 1],
                       positive=[(0,), (0,0)],
                       negative=[(1,), (1,0)],
                       oracle=oracle,
                       n_queries=20)

```

# Learning Decomposed DFAs

The is also support for learning decomposed DFAs following,
[Learning Deterministic Finite Automata Decompositions
 from Examples and Demonstrations. FMCAD`22](
 https://doi.org/10.34727/2022/isbn.978-3-85448-053-2_39).
These are tuples of DFAs whose labels are combined to determine
the acceptence / rejection of string, e.g., via conjunction or
disjunction.

Similar to learning a monolithic dfa, this functionality can
be accessed using `find_decomposed_dfas`. For example:

```python
from dfa_identify import find_decomposed_dfas
accepting = ['y', 'yy', 'gy', 'bgy', 'bbgy', 'bggy']
rejecting = ['', 'r', 'ry', 'by', 'yr', 'gr', 'rr', 'rry', 'rygy']

# --------------------------------------
# 1. Learn a disjunctive decomposition.
# --------------------------------------
gen_dfas = find_decomposed_dfas(accepting=accepting,
                                rejecting=rejecting,
                                n_dfas=2,
                                order_by_stutter=True,
                                decompose_via="disjunction")
dfas = next(gen_dfas)

monolithic = dfas[0] | dfas[1]  # Build DFA that is the union of languages.
assert all(monolithic.label(w) for w in accepting)
assert not any(monolithic.label(w) for w in rejecting)

# Each dfa must reject a rejecting string.
assert all(all(~d.label(w) for d in dfas) for w in rejecting)
# At least one dfa must accept an accepting string.
assert all(any(d.label(w) for d in dfas) for w in accepting)

# --------------------------------------
# 2. Learn a conjunctive decomposition.
# --------------------------------------
gen_dfas = find_decomposed_dfas(accepting=accepting,
                                rejecting=rejecting,
                                n_dfas=2,
                                order_by_stutter=True,
                                decompose_via="conjunction")
dfas = next(gen_dfas)

monolithic = dfas[0] & dfas[1]  # Build DFA that is the union of languages.
assert all(monolithic.label(w) for w in accepting)
assert not any(monolithic.label(w) for w in rejecting)

```

# Minimality

There are two forms of "minimality" supported by `dfa-identify`.

1. By default, dfa-identify returns DFAs that have the minimum
   number of states required to seperate the accepting and
   rejecting set.
2. If the `order_by_stutter` flag is set to `True`, then the
   `find_dfas` (lazily) orders the DFAs so that the number of
   self loops (stuttering transitions) appearing the DFAs decreases.
   `find_dfa` thus returns a DFA with the most number of self loops
   given the minimal number of states.

# Encoding

This library currently uses the encodings outlined in [Heule, Marijn JH, and Sicco Verwer. "Exact DFA identification using SAT solvers." International Colloquium on Grammatical Inference. Springer, Berlin, Heidelberg, 2010.](https://link.springer.com/chapter/10.1007/978-3-642-15488-1_7) and [Ulyantsev, Vladimir, Ilya Zakirzyanov, and Anatoly Shalyto. "Symmetry Breaking Predicates for SAT-based DFA Identification."](https://arxiv.org/abs/1602.05028).

The key difference is in the use of the symmetry breaking clauses. Two kinds are exposed.

1. clique (Heule 2010): Partially breaks symmetries by analyzing
   conflict graph.
2. bfs (Ulyantsev 2016): Breaks all symmetries so that each model corresponds to a unique DFA.

# Goals and related libraries

There are many other python libraries that 
perform DFA and other automata inference.

1. [DFA-Inductor-py](https://github.com/ctlab/DFA-Inductor-py) - State of the art passive inference via reduction to SAT (as of 2019).
2. [z3gi](https://gitlab.science.ru.nl/rick/z3gi): Uses SMT backed passive learning algorithm.
3. [lstar](https://pypi.org/project/lstar/): Active learning algorithm based L* derivative.

The primary goal of this library is to loosely track the state of the art in passive SAT based inference while providing a simple implementation and API.
