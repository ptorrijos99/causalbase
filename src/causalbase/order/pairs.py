"""Pair additions: a polish for parents that only help together.

Greedy per-node searches (hill climbing over an order, BOSS's grow-shrink, GES's
single-edge inserts) add one parent at a time, so they never find a family whose
members have no marginal effect on the node, as in a noisy XOR or a logic gate (andes
has several). :func:`pair_polish` climbs a DAG by adding parents to each node in turn:
the best single addition while one improves the local score, and when none does, the
best pair among the ``n_candidates`` candidates with the largest single-addition gains. Only
non-descendants are added, so the graph stays acyclic, and the score never drops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from causalbase.graph import DAG
from causalbase.scores._compiled import engine_of

if TYPE_CHECKING:
    from causalbase.scores.bdeu import BDeuScore

__all__ = ["pair_polish"]


def _clean(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), -np.inf, x)


def _descendants(children: list[set[int]], v: int) -> set[int]:
    seen, stack = {v}, [v]
    while stack:
        for y in children[stack.pop()]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen


def pair_polish(score: BDeuScore, dag: DAG, n_candidates: int = 10) -> tuple[DAG, float, int]:
    """Add parents (singly, or in pairs when no single one helps) until no node improves.

    Parameters
    ----------
    score : BDeuScore
        The score, with either prior; families are scored with its compiled
        engine.
    dag : DAG
        Starting graph; its arcs are kept.
    n_candidates : int, default=10
        Pairs are drawn from this many best single candidates; 0 turns pairs off, leaving a greedy
        single-addition climb.

    Returns
    -------
    tuple
        The DAG, the score gained and the number of pair moves applied.
    """
    engine = engine_of(score)
    n = len(dag.topological_order())
    parents = [set(dag.parents(v)) for v in range(n)]
    children = [set(dag.children(v)) for v in range(n)]
    gained, pairs = 0.0, 0
    changed = True
    while changed:
        changed = False
        for v in range(n):
            while True:
                desc = _descendants(children, v)
                cands = [u for u in range(n) if u not in desc and u not in parents[v]]
                if not cands:
                    break
                base = float(engine.local_score(v, frozenset(parents[v])))
                singles = (
                    _clean(
                        np.asarray(
                            engine.local_scores_add(v, sorted(parents[v]), cands), dtype=float
                        )
                    )
                    - base
                )
                k = int(np.argmax(singles))
                if singles[k] > 1e-9:
                    add, gain = [cands[k]], float(singles[k])
                else:
                    if n_candidates < 2:
                        break
                    best = np.argsort(-singles, kind="stable")[:n_candidates]
                    pool = [cands[i] for i in best]
                    gain, add = 1e-9, []
                    for i, u in enumerate(pool[:-1]):
                        rest = pool[i + 1 :]
                        vals = (
                            _clean(
                                np.asarray(
                                    engine.local_scores_add(v, sorted(parents[v] | {u}), rest),
                                    dtype=float,
                                )
                            )
                            - base
                        )
                        j = int(np.argmax(vals))
                        if vals[j] > gain:
                            gain, add = float(vals[j]), [u, rest[j]]
                    if not add:
                        break
                    pairs += 1
                for u in add:
                    parents[v].add(u)
                    children[u].add(v)
                gained += gain
                changed = True
    return DAG.from_edges(n, [(u, v) for v in range(n) for u in parents[v]]), gained, pairs
