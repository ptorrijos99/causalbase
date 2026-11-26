"""Every example in a docstring runs and prints what it shows."""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import causalbase


def _modules() -> list[str]:
    names = []
    for info in pkgutil.walk_packages(causalbase.__path__, "causalbase."):
        names.append(info.name)
    return sorted(names)


def test_docstring_examples_run() -> None:
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=flags)
    for name in _modules():
        for test in finder.find(importlib.import_module(name), name):
            runner.run(test)
    result = runner.summarize(verbose=False)
    assert result.failed == 0
    assert result.attempted >= 6  # guard the guard: the finder found the examples
