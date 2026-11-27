"""What every search returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from causalbase.graph.cpdag import dag_to_cpdag

if TYPE_CHECKING:
    from causalbase.graph import DAG, PDAG

__all__ = ["SearchResult"]


@dataclass
class SearchResult:
    """The network a search found, its score, and what it took to find it.

    Returned by every search: :func:`~causalbase.search.ges`,
    :func:`~causalbase.search.xges`, :func:`~causalbase.search.hill_climbing`,
    :func:`~causalbase.order.order_search`, :func:`~causalbase.order.flop` and
    :func:`~causalbase.exact.exact_search`.

    Attributes
    ----------
    dag : DAG
        The best network found.
    score : float
        Its score, as the search computed it.
    n_iter : int
        Iterations of the search; each function says what one is.
    trace : list of (float, int, float)
        ``(seconds, iteration, best score)`` at the start and each time the best
        score improved.
    info : dict
        What else the search reports, by name; each function lists its keys.
    cpdag : PDAG
        The Markov equivalence class of `dag`, which GES and XGES learn
        directly.
    """

    dag: DAG
    score: float
    n_iter: int
    trace: list[tuple[float, int, float]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def cpdag(self) -> PDAG:
        """PDAG: The Markov equivalence class of `dag`, as its CPDAG."""
        return dag_to_cpdag(self.dag)
