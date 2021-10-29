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
