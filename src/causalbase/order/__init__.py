"""Structure learning by search over variable orders.

:func:`order_search` is an iterated local search over orders whose local
searches end with GES or XGES started from the order's network, and
:func:`flop` is FLOP (Wienoebst, Henckel and Weichwald, 2026) on the same
score; :class:`OrderSearch` and :class:`FLOP` wrap them as estimators. Both
start, by default, from a topological order of the network GES learns.

The building blocks are here too, for those who want to combine them
differently: :class:`OrderScorer` scores an order by the best network
consistent with it (a greedy climb per node, exact and incremental),
:class:`InsertionSearch` climbs over orders by insertion moves,
:class:`IteratedLocalSearch` iterates it with random kicks, and
:func:`pair_polish` adds parents that only help together.
"""

from causalbase.order.estimator import FLOP, OrderSearch
from causalbase.order.evaluator import OrderResult, OrderScorer
from causalbase.order.flop import flop
from causalbase.order.ils import ILSResult, IteratedLocalSearch
from causalbase.order.insertion import InsertionSearch
from causalbase.order.pairs import pair_polish
from causalbase.order.polished import order_search
from causalbase.search import SearchResult

__all__ = [
    "FLOP",
    "ILSResult",
    "InsertionSearch",
    "IteratedLocalSearch",
    "OrderResult",
    "OrderScorer",
    "OrderSearch",
    "SearchResult",
    "flop",
    "order_search",
    "pair_polish",
]
