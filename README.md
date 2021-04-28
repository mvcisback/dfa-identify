# dfa-identify
Python library for identifying (learning) DFAs from labeled examples
by reduction to SAT.


[![Build Status](https://cloud.drone.io/api/badges/mvcisback/dfa-identify/status.svg)](https://cloud.drone.io/mvcisback/dfa-identify)
[![Docs](https://img.shields.io/badge/API-link-color)](https://mvcisback.github.io/dfa-identify)
[![PyPI version](https://badge.fury.io/py/dfa_identify.svg)](https://badge.fury.io/py/dfa_identify)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Encoding](#encoding)

<!-- markdown-toc end -->


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

# Encoding

This library currently uses the encoding outlined in [Heule, Marijn JH, and Sicco Verwer. "Exact DFA identification using SAT solvers." International Colloquium on Grammatical Inference. Springer, Berlin, Heidelberg, 2010.](https://link.springer.com/chapter/10.1007/978-3-642-15488-1_7)
