"""Random problems for property tests, and the best DAG of a score by brute force."""

from __future__ import annotations

from typing import Any

import numpy as np

from causalbase.data import ContinuousDataset, DiscreteDataset
from causalbase.graph import DAG


def random_discrete(p: int, n: int, seed: int) -> DiscreteDataset:
    """Data from a random DAG over a fixed order, with noisy CPTs, so the optimum is not trivial."""
    rng = np.random.default_rng(seed)
    cards = rng.integers(2, 4, size=p)
    X = np.zeros((n, p), dtype=np.int64)
    for v in range(p):
        parents = [u for u in range(v) if rng.random() < 0.5]
        key = np.zeros(n, dtype=np.int64)
        for u in parents:
            key = key * cards[u] + X[:, u]
        q = int(np.prod([cards[u] for u in parents])) if parents else 1
        cpt = rng.dirichlet(np.full(cards[v], 0.6), size=q)
        u01 = rng.random(n)
        X[:, v] = np.minimum((u01[:, None] > np.cumsum(cpt[key], axis=1)).sum(axis=1), cards[v] - 1)
    perm = rng.permutation(p)  # the generating order is not the variable order
    X, cards = X[:, perm], cards[perm]
    return DiscreteDataset(
        X, cards, [f"x{i}" for i in range(p)], [[str(a) for a in range(c)] for c in cards]
    )


def random_gaussian(p: int, n: int, seed: int) -> ContinuousDataset:
    """Data from a random linear Gaussian DAG, with the variables shuffled out of its order."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, p))
    for v in range(p):
        weights = rng.uniform(0.3, 1.0, size=v) * rng.choice([-1, 1], size=v)
        weights[rng.random(v) >= 0.5] = 0.0
        X[:, v] = X[:, :v] @ weights + rng.normal(scale=rng.uniform(0.5, 1.5), size=n)
    X = X[:, rng.permutation(p)]
    return ContinuousDataset(X, [f"x{i}" for i in range(p)])


def random_dag(p: int, density: float, seed: int) -> DAG:
    """A random DAG over a random order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(p)
    edges = [
        (int(order[i]), int(order[j])) for j in range(p) for i in range(j) if rng.random() < density
    ]
    return DAG.from_edges(p, edges)


def _acyclic(masks: list[int]) -> bool:
    left = set(range(len(masks)))
    while left:
        roots = [v for v in left if not any((masks[v] >> u) & 1 for u in left)]
        if not roots:
            return False
        left -= set(roots)
    return True


def best_dag_score(score: Any) -> float:
    """Best score over every assignment of parent sets that forms a DAG."""
    p = int(score.n_vars)
    table = np.full((p, 1 << p), -np.inf)
    for v in range(p):
        for mask in range(1 << p):
            if not (mask >> v) & 1:
                s = score.local_score(v, [u for u in range(p) if (mask >> u) & 1])
                table[v, mask] = s if np.isfinite(s) else -np.inf
    options = [[mask for mask in range(1 << p) if not (mask >> v) & 1] for v in range(p)]
    totals = np.zeros([len(o) for o in options])
    for v in range(p):
        shape = [1] * p
        shape[v] = len(options[v])
        totals = totals + table[v, options[v]].reshape(shape)
    flat = totals.ravel()
    for i in np.argsort(-flat, kind="stable"):
        choice = np.unravel_index(i, totals.shape)
        if _acyclic([options[v][choice[v]] for v in range(p)]):
            return float(flat[i])
    raise AssertionError("no DAG found")
