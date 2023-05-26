from itertools import product


def enforce_chain(_, codec):
    """Enforce that the resulting DFA is a chain.

       0 -> 1 -> ... -> n

    """
    return enforce_reach_avoid_seq(_, codec, with_avoid=False)

def enforce_reach_avoid_seq(_, codec, *, with_avoid=True):
    """Enforce that the resulting DFA is sequence of reach avoid problems.

    In particular, the graph of the DFA is chain with an optional failing state
    for not avoiding the active "avoid set" (can be empty).

       0 -> 1 -> ... -> n
        \   |     /
         \  |    /
          \ |   /
            fail
    """
    colors, tokens = range(codec.n_colors), range(codec.n_tokens)

    for token, c1, c2 in product(tokens, colors, colors):
        if (c1 == c2) or (c1 + 1 == c2): 
            continue  # Can stutter or move forward.
        if with_avoid and (c2 + 1 == codec.n_colors):
            continue  # Optionally support transitions to the last state. 
        yield [-codec.parent_relation(token, c1, c2)]
