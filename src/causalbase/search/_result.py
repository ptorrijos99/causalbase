"""What every search but GES and XGES returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causalbase.graph import DAG

__all__ = ["SearchResult"]


@dataclass
class SearchResult:
    """The network a search found, its score, and what it took to find it.

    Returned by :func:`~causalbase.search.hill_climbing`,
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
    """

    dag: DAG
    score: float
    n_iter: int
    trace: list[tuple[float, int, float]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
