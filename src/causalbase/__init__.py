"""Bayesian network structure learning and causal discovery.

Six estimators cover the common cases:

* :class:`GES` learns a causal structure, and :class:`XGES` does so with a
  search that removes edges as it goes.
* :class:`OrderSearch` learns a Bayesian network by iterated local search
  over variable orders, polished by GES or XGES, and :class:`FLOP` with
  FLOP's search over orders.
* :class:`HillClimbing` climbs over DAGs, optionally as a tabu search with
  restarts, and :class:`ExactSearch` finds the network of highest score.

Each takes any of the four scores by name (``criterion``): BDeu or BIC on
discrete data, the Gaussian BIC or BGe on continuous data. They share their
interface and follow scikit-learn's conventions: hyperparameters go to the
constructor, ``fit`` learns, and results end in an underscore. The function
behind each, which takes a score object and returns a
:class:`~causalbase.search.SearchResult`, is in its subpackage. Federated
learning over data held by several sites lives in the ``fedcausal`` package,
which builds on this one.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causalbase.exact import ExactSearch
    from causalbase.graph import DAG, PDAG
    from causalbase.order import FLOP, OrderSearch
    from causalbase.search import GES, XGES, HillClimbing

__version__ = "0.1.0"

__all__ = [
    "DAG",
    "FLOP",
    "GES",
    "PDAG",
    "XGES",
    "ExactSearch",
    "HillClimbing",
    "OrderSearch",
    "__version__",
]

# Imported on first use, so that a subpackage (causalbase.graph, say) loads
# without the others.
_HOMES = {
    "DAG": "causalbase.graph",
    "PDAG": "causalbase.graph",
    "GES": "causalbase.search",
    "XGES": "causalbase.search",
    "HillClimbing": "causalbase.search",
    "OrderSearch": "causalbase.order",
    "FLOP": "causalbase.order",
    "ExactSearch": "causalbase.exact",
}


def __getattr__(name: str) -> Any:
    if name in _HOMES:
        value = getattr(importlib.import_module(_HOMES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'causalbase' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
