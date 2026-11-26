"""Iterated local search over orders: insertion search, then perturb the best order and repeat.

Each iteration perturbs the best order found so far with ``n_swaps`` random
transpositions (``k = ceil(ln p)`` by default, the strength FLOP uses and the
one that worked best here), runs the insertion search to a local optimum, and
keeps the result when it is at least as good, so plateaus are crossed. The
search stops at a wall-clock limit (checked inside the local search too) or
after a number of iterations.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causalbase._validation import check_count, resolve_seed
from causalbase.order.evaluator import OrderScorer
from causalbase.order.insertion import InsertionSearch

__all__ = ["ILSResult", "IteratedLocalSearch"]


@dataclass
class ILSResult:
    """Best order and score, and the anytime trace ``(seconds, iteration, best score)``.

    Iteration 0 is the first local search from the start order. ``improvements``
    holds one entry per strict improvement after it: the iteration, the gain, the
    transpositions applied (pairs of variables) and the nodes whose parent set
    changed with respect to the previous best.
    """

    order: list[int]
    score: float
    iterations: int
    accepted: int
    trace: list[tuple[float, int, float]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)


class IteratedLocalSearch:
    """Iterated insertion search with random-transposition kicks.

    Parameters
    ----------
    scorer : OrderScorer
        Order evaluator.
    n_swaps : int or None, default=None
        Transpositions per perturbation; ``ceil(ln p)`` when None. Kept as
        ``n_swaps`` with that default resolved.
    random_state : int or None, default=None
        Seeds every random choice (visiting order and perturbations); None
        draws a fresh seed.
    kick : {"swaps", "topo", "mixed"}, default="mixed"
        ``swaps``: ``k`` random transpositions. ``topo``: a random topological
        order of the network the evaluator learns from the order, the same DAG
        reached through another order (the evaluator depends on the order, not
        only on the DAG). ``mixed``: either, with equal probability.
    """

    def __init__(
        self,
        scorer: OrderScorer,
        n_swaps: int | None = None,
        random_state: int | None = None,
        kick: str = "mixed",
    ) -> None:
        if kick not in ("swaps", "topo", "mixed"):
            raise ValueError(f"unknown kick {kick!r}")
        check_count("n_swaps", n_swaps, allow_none=True)
        self.scorer = scorer
        self.search = InsertionSearch(scorer)
        self.n_swaps = (
            n_swaps if n_swaps is not None else max(1, math.ceil(math.log(scorer.n_vars)))
        )
        self.kick = kick
        self.rng = np.random.default_rng(resolve_seed(random_state))

    def perturb(self, order: Sequence[int]) -> list[int]:
        """Return ``order`` perturbed by the configured kick.

        For transpositions, the transposed pairs of variables are left in
        ``last_swaps`` (empty after a topological kick).
        """
        if self.kick == "topo" or (self.kick == "mixed" and self.rng.random() < 0.5):
            self.last_swaps: list[list[int]] = []
            return self._random_topological(order)
        out = list(order)
        self.last_swaps = []
        for _ in range(self.n_swaps):
            i, j = self.rng.choice(len(out), 2, replace=False)
            self.last_swaps.append([int(out[i]), int(out[j])])
            out[i], out[j] = out[j], out[i]
        return out

    def _random_topological(self, order: Sequence[int]) -> list[int]:
        """Run Kahn's algorithm on the order's network, drawing a ready node uniformly each step."""
        parents = self.scorer.score(order).parents
        n = len(parents)
        children: list[list[int]] = [[] for _ in range(n)]
        missing = [len(ps) for ps in parents]
        for v, ps in enumerate(parents):
            for u in ps:
                children[u].append(v)
        ready = [v for v in range(n) if missing[v] == 0]
        out = []
        while ready:
            v = ready.pop(int(self.rng.integers(len(ready))))
            out.append(v)
            for w in children[v]:
                missing[w] -= 1
                if missing[w] == 0:
                    ready.append(w)
        return out

    def run(
        self,
        start: Sequence[int],
        time_limit: float | None = None,
        max_iterations: int | None = None,
    ) -> ILSResult:
        """Search from ``start`` until a time or iteration limit.

        The search stops after ``time_limit`` seconds or ``max_iterations`` perturbations.
        """
        t0 = time.perf_counter()
        deadline = None if time_limit is None else t0 + time_limit
        best, best_score = self.search.run(start, self.rng, deadline=deadline)
        trace = [(time.perf_counter() - t0, 0, best_score)]
        improvements: list[dict[str, Any]] = []

        iterations = accepted = 0
        while (max_iterations is None or iterations < max_iterations) and (
            deadline is None or time.perf_counter() < deadline
        ):
            candidate, score = self.search.run(self.perturb(best), self.rng, deadline=deadline)
            iterations += 1
            if score >= best_score - 1e-9:
                accepted += 1
                if score > best_score + 1e-9:
                    trace.append((time.perf_counter() - t0, iterations, score))
                    before = self.scorer.score(best).parents
                    after = self.scorer.score(candidate).parents
                    improvements.append(
                        {
                            "iteration": iterations,
                            "gain": score - best_score,
                            "swaps": self.last_swaps,
                            "changed": [v for v in range(len(after)) if after[v] != before[v]],
                        }
                    )
                best, best_score = candidate, score
        return ILSResult(
            best, self.scorer.score(best).total, iterations, accepted, trace, improvements
        )
