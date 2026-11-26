"""Exceptions raised by causalbase.

A cycle in a :class:`~causalbase.graph.DAG` and reading an estimator's results
before ``fit`` derive from :class:`CausalBaseError`, so callers can catch them
with one ``except``. An invalid argument or input raises :class:`ValueError`
(or :class:`ConfigurationError`, which is one), and a wrong kind of input
:class:`TypeError`.
"""

from __future__ import annotations

__all__ = [
    "CausalBaseError",
    "ConfigurationError",
    "CycleError",
    "NotFittedError",
]


class CausalBaseError(Exception):
    """Base class for every error causalbase raises deliberately."""


class ConfigurationError(CausalBaseError, ValueError):
    """An argument or setting is invalid.

    Also a :class:`ValueError`, so ``except ValueError`` catches it too.
    """


class CycleError(CausalBaseError, ValueError):
    """An edge would close a directed cycle, or a graph that must be acyclic is not.

    Also a :class:`ValueError`. Importable from :mod:`causalbase.graph` too.
    """


class NotFittedError(CausalBaseError, AttributeError):
    """An estimator's result was read before ``fit`` was called.

    Also an :class:`AttributeError`, so ``hasattr`` and scikit-learn's
    checks treat an unfitted result as missing.
    """
