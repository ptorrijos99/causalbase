"""How many threads the compiled kernels run on, as every ``n_jobs`` in the package means it."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numba

from causalbase._validation import check_n_jobs

__all__ = ["n_threads", "numba_threads"]


def n_threads(n_jobs: int) -> int:
    """Return the threads `n_jobs` stands for: -1 is every thread numba has, others are capped.

    numba refuses more threads than it started with (``NUMBA_NUM_THREADS``,
    the number of cores by default), so a larger `n_jobs` gets that many.

    Raises
    ------
    ValueError
        If `n_jobs` is neither a positive integer nor -1.
    """
    check_n_jobs(n_jobs)
    available = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
    return available if n_jobs == -1 else min(int(n_jobs), available)


@contextmanager
def numba_threads(threads: int) -> Iterator[None]:
    """Run the compiled kernels inside on `threads` threads, restoring the count after."""
    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        yield
    finally:
        numba.set_num_threads(previous)
