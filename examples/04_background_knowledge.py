"""
Add background knowledge
========================

:class:`~causalbase.search.GES` and :class:`~causalbase.search.XGES` take
pairs of variables that must be adjacent (``required``) or must not be
(``forbidden``), by name or by column index. Knowledge fixes adjacency, not
direction: a search over equivalence classes still orients edges from the
data.
"""

from causalbase import GES
from causalbase.datasets import load_asia

bundle = load_asia()
frame = bundle.data.to_pandas()


def neighbours(model, node):
    """Return the variables joined to `node` by an edge of either kind."""
    adjacency = model.adjacency_
    return [v for v in adjacency.columns if adjacency.loc[node, v] or adjacency.loc[v, node]]


# %%
# In asia, ``either`` is tuberculosis or lung cancer, and GES joins it to
# both of its causes and to its two effects.

free = GES().fit(frame)
neighbours(free, "either")

# %%
# Forbidding ``lung -- either`` keeps the two apart whatever the data say.
# The search then explains the dependence through other edges, such as a
# direct one between ``lung`` and ``xray``.

constrained = GES(forbidden=[("lung", "either")]).fit(frame)
neighbours(constrained, "either"), neighbours(constrained, "lung")

# %%
# The constraint costs score: under the same BDeu, the free network is
# better.

print(f"Free {free.score_:.1f}, with the forbidden pair {constrained.score_:.1f}")
