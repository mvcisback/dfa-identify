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
