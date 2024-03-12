from itertools import product

import attr

from dfa_identify.encoding import Codec


def enforce_chain(_, codec):
    """Enforce that the resulting DFA is a chain.

       0 -> 1 -> ... -> n

    """
    return enforce_reach_avoid_seq(_, codec, with_avoid=False)

def enforce_reach_avoid_seq(_, codec, *, with_avoid=True):
    """Enforce that the resulting DFA is sequence of reach avoid problems.

    In particular, the graph of the DFA is chain with an optional failing state
    for not avoiding the active "avoid set" (can be empty).

       0 -> 1 -> ... -> n - 1
        \   |     /
         \  |    /
          \ |   /
            n (failing sink)
    """
    if codec.n_colors == 1:
        return              # One of the trivial DFAs. No extra constraints.

    colors, tokens = range(codec.n_colors), range(codec.n_tokens)
    last = codec.n_colors - 1

    for token, c1, c2 in product(tokens, colors, colors):
        if (c1 == c2) or (c1 + 1 == c2): 
            continue  # Can stutter or move forward.
        if with_avoid and (c2 == last):
            continue  # Optionally support transitions to the last state. 
        yield [-codec.parent_relation(token, c1, c2)]

    # Force exactly one accepting state at end of chain.
    if (with_avoid and codec.n_colors == 2):
        # One being accepting makes the other rejecting.
        l0, l1 = codec.color_accepting(0), codec.color_accepting(1)
        yield [-l0, l1]
        yield [-l1, l0]
    else:
        # "last" state in the chain is accepting by convention.
        accepting_color = last - 1 if with_avoid else last
        rejecting_colors = filter(lambda c: c != accepting_color, colors)
        yield from ([-codec.color_accepting(c)] for c in rejecting_colors)
        yield [codec.color_accepting(accepting_color)]

# ================ Invariant Representation Class ====================

def depth_node(codec: Codec, depth: int, color: int) -> int:
    offset = codec.max_id + 1
    return offset + depth * codec.n_colors + color

"""
Restriction for invariants over Σⁿ where Σ = alphabet and n is the
tokens per observation.

For example {0, 1}ⁿ may encode the state of a system and the monitor
encodes staying in a safe set.

The DFA wraps a Quasi Reduced Binary Decision Diagram (QDD) where:
    1. The initial state is the final accepting node.
    2. Each interior node is associated with a depth.
    3. All interior (not max depth) states transition to states of increasing depth.
    4. The unique accepting (initial) state transitions to the root.
    5. The final rejecting state is a sink.
"""
@attr.frozen
class EnforceInvariant:
    tokens_per_state: int

    def __call__(self, _, codec):
        max_depth = self.tokens_per_state

        # 1. All interior nodes are rejecting and we start at max depth.
        yield [depth_node(codec, max_depth - 1, 0)]  # Start at end of tree.
        for c in range(1, codec.n_colors):
            yield [-codec.color_accepting(c)]        # Only start state is accepting.

        # 2. Map colors to depths.
        # Based on dfa_identify.encoding.onehot_depth_clauses.
        for c in range(codec.n_colors):  # Each color has at least one depth.
            yield [depth_node(codec, d, c) for d in range(max_depth)]

        for d in range(max_depth):  # Each depth has at least one color.
            yield [depth_node(codec, d, c) for c in range(codec.n_colors)]

        for c in range(codec.n_colors):  # Each color has at most one depth.
            for i in range(max_depth):
                lit = depth_node(codec, i, c)
                for j in range(i + 1, max_depth):  # i < j
                    yield [-lit, -depth_node(codec, j, c)]

        transitions = product(range(codec.n_colors), 
                              range(codec.n_tokens),
                              range(codec.n_colors))

        for c1, token, c2 in transitions:
            # 3. All but last depth advance by one.
            for d in range(max_depth - 1):
                yield [-codec.parent_relation(token, c1, c2),
                       -depth_node(codec, d, c1),
                        depth_node(codec, d + 1, c2)]

            # 4. Final accepting states transitions to root.
            yield [-codec.parent_relation(token, c1, c2),
                   -depth_node(codec, max_depth - 1, c1),
                   -codec.color_accepting(c1),
                    depth_node(codec, 0, c2)]

            # 5. Final rejecting state is a sink.
            if c1 == c2:
                yield [-depth_node(codec, max_depth - 1, c1),
                       codec.color_accepting(c1),
                       codec.parent_relation(token, c1, c2)]
