# dfa-identify
Python library for identifying DFAs from labeled examples by reduction to SAT.

[![Build Status](https://cloud.drone.io/api/badges/mvcisback/dfa_identify/status.svg)](https://cloud.drone.io/mvcisback/dfa_identify)
[![Docs](https://img.shields.io/badge/API-link-color)](https://mvcisback.github.io/dfa_identify)
[![codecov](https://codecov.io/gh/mvcisback/dfa-identify/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/dfa_identify)
[![PyPI version](https://badge.fury.io/py/dfa_identify.svg)](https://badge.fury.io/py/dfa_identify)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Installation

If you just need to use `dfa`, you can just run:

`$ pip install dfa`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Usage

`dfa_identify` is centered around the `find_dfa` function. It takes in
sequences of accepting and rejecting "words", where are word is a
sequence of arbitrary python objects. 

The returned `DFA` object is from the [dfa](https://github.com/mvcisback/dfa) library.


```python
from dfa_identify import find_dfa


def test_identify():
    accepting = ['a', 'abaa', 'bb']
    rejecting = ['abb', 'b']
    
    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)
```

Because words are sequences of arbitrary python objects, the
identification problem, with `a` ↦ 0 and `b` ↦ 1, is given below:


```python
    accepting = [[0], [0, 'z', 0, 0], ['z', 'z']]
    rejecting = [[0, 'z', 'z'], ['z']]
    
    my_dfa = find_dfa(accepting=accepting, rejecting=rejecting)

    for x in accepting:
        assert my_dfa.label(x)

    for x in rejecting:
        assert not my_dfa.label(x)
```

# Encoding

This library currently uses the encoding outlined in [Heule, Marijn JH, and Sicco Verwer. "Exact DFA identification using SAT solvers." International Colloquium on Grammatical Inference. Springer, Berlin, Heidelberg, 2010.](https://link.springer.com/chapter/10.1007/978-3-642-15488-1_7)
