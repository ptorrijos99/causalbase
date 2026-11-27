"""Bayesian network structure learning and causal discovery.

Seven estimators cover the common cases:

* :class:`GES` learns a causal structure from discrete or continuous data, and
  :class:`XGES` does so with a search that removes edges as it goes.
* :class:`OrderSearch` learns a Bayesian network from discrete data by
  iterated local search over variable orders, polished by GES or XGES, and
  :class:`FLOP` with FLOP's search over orders.
* :class:`HillClimbing` climbs over DAGs, optionally as a tabu search with
  restarts, and :class:`ExactSearch` finds the network of highest score.
* :class:`RCD` learns a linear non-Gaussian model from continuous data through
  its higher-order cumulants, with HC-LiNGAM and HC among its settings.

OrderSearch, FLOP, HillClimbing and ExactSearch score with the compiled engine of
:class:`~causalbase.scores.BDeuScore`, with its default Bernoulli structure
prior or Tetrad's (``prior="tetrad"``); the functions behind them are
:func:`~causalbase.order.order_search`, :func:`~causalbase.order.flop`,
:func:`~causalbase.search.hill_climbing` and :func:`~causalbase.exact.exact_search`.
Each estimator follows scikit-learn's conventions: hyperparameters go to the
constructor, ``fit`` learns, and results end in an underscore. Federated
learning over data held by several sites lives in the ``fedcausal`` package,
which builds on this one.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causalbase.exact import ExactSearch
    from causalbase.graph import DAG, PDAG
    from causalbase.lingam import RCD
    from causalbase.order import FLOP, OrderSearch
    from causalbase.search import GES, XGES, HillClimbing

__version__ = "0.1.0"

__all__ = [
    "DAG",
    "FLOP",
    "GES",
    "PDAG",
    "RCD",
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
    "RCD": "causalbase.lingam",
}


def __getattr__(name: str) -> Any:
    if name in _HOMES:
        value = getattr(importlib.import_module(_HOMES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'causalbase' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
