"""Checks of the arguments the searches and estimators share, and the seed behind `random_state`."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

__all__ = ["check_count", "check_n_jobs", "check_positive", "check_time_limit", "resolve_seed"]


def resolve_seed(random_state: int | None) -> int:
    """Return the integer seed `random_state` stands for.

    An integer is used as it is, so that ``random_state=s`` seeds every
    generator exactly as the searches' former ``seed=s`` did; None draws a
    fresh seed, so that each call differs.

    Raises
    ------
    ValueError
        If `random_state` is neither an integer nor None.
    """
    if random_state is None:
        return int(np.random.default_rng().integers(2**31))
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise ValueError(f"random_state must be an integer or None, got {random_state!r}")
    return int(random_state)


def check_count(name: str, value: object, *, allow_none: bool = False) -> None:
    """Raise ValueError unless `value` is a non-negative integer (or None, if allowed)."""
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a non-negative integer{suffix}, got {value!r}")


def check_positive(name: str, value: object, *, allow_zero: bool = False) -> None:
    """Raise ValueError unless `value` is a real number above zero (or equal to it, if allowed)."""
    ok = isinstance(value, Real) and not isinstance(value, bool)
    if ok:
        number = float(value)  # type: ignore[arg-type]
        ok = number >= 0 if allow_zero else number > 0
    if not ok:
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {bound} number, got {value!r}")


def check_time_limit(time_limit: object) -> None:
    """Raise ValueError unless `time_limit` is None or a positive number of seconds."""
    if time_limit is not None:
        check_positive("time_limit", time_limit)


def check_n_jobs(n_jobs: object) -> None:
    """Raise ValueError unless `n_jobs` is a positive integer."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer, got {n_jobs!r}")
