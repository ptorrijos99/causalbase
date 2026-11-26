"""causalbase stands alone: it never imports the federated package built on it.

``fedcausal`` depends on causalbase, never the other way round, and nothing here
needs Flower.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "causalbase"
FORBIDDEN = ("fedcausal", "flwr", "ray")


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_causalbase_never_imports_the_federated_package_or_flower() -> None:
    """Not even under TYPE_CHECKING."""
    offenders = {
        p.name: sorted(i)
        for p in _source_files()
        if (i := {n for n in _imported_modules(p) if n.split(".")[0] in FORBIDDEN})
    }
    assert not offenders
    assert len(_source_files()) > 20  # the walk found the package


def test_learning_loads_nothing_federated() -> None:
    """In a fresh interpreter, a GES fit loads neither fedcausal nor Flower."""
    code = textwrap.dedent(
        """
        import sys
        from causalbase import GES
        from causalbase.datasets import load_cancer
        from causalbase.metrics import shd

        bundle = load_cancer()
        shd(bundle.truth, GES().fit(bundle.data).dag_)
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("fedcausal", "flwr", "ray"))
        assert not loaded, loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
