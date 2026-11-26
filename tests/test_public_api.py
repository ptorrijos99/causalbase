"""Every name in an ``__all__`` exists, and every import the docs show works."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SUBPACKAGES = [
    "causalbase",
    "causalbase.data",
    "causalbase.exact",
    "causalbase.exceptions",
    "causalbase.graph",
    "causalbase.io",
    "causalbase.metrics",
    "causalbase.order",
    "causalbase.scores",
    "causalbase.search",
]

DOC_FILES = (
    list(REPO.glob("*.md"))
    + list((REPO / "docs").glob("*.md"))
    + list((REPO / "examples").rglob("*.md"))
)
DOC_IMPORTS = sorted(
    {
        (m.group(1), name.strip())
        for f in DOC_FILES
        for m in re.finditer(
            r"^from (causalbase[\w.]*) import ([^\n]+)$", f.read_text(encoding="utf-8"), re.M
        )
        for name in m.group(2).split(",")
        if name.strip()
    }
)


def test_every_export_resolves() -> None:
    """Nothing in an ``__all__`` has been renamed out from under it."""
    missing = [
        f"{name}.{export}"
        for name in SUBPACKAGES
        for export in getattr(importlib.import_module(name), "__all__", [])
        if not hasattr(importlib.import_module(name), export)
    ]
    assert not missing


def test_documented_imports_work() -> None:
    """Every ``from causalbase... import ...`` shown in the docs actually runs."""
    broken = [
        f"{module}.{name}"
        for module, name in DOC_IMPORTS
        if not hasattr(importlib.import_module(module), name)
    ]
    assert not broken
    assert len(DOC_IMPORTS) > 5  # guard the guard: the scrape found the imports
