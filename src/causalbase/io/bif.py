"""Reader for BIF (``.bif``) discrete Bayesian networks.

BIF is the text format of the Bayesian Network Repository and of bnRep, as R's
bnlearn writes it::

    variable tub {
      type discrete [ 2 ] { yes, no };
    }
    probability ( tub | asia ) {
      (yes) 0.05, 0.95;
      (no) 0.01, 0.99;
    }

A conditional table is given one row per configuration of the parents, each
row headed by the parents' states in the order the ``probability`` line lists
them, and ``default`` gives the row of every configuration not listed. A node
without parents gives its distribution as ``table``. Writers disagree on the
order of a conditional table given whole as ``table``, so that form is
refused. ``property`` entries and ``//`` and ``/* */`` comments are ignored.
"""

from __future__ import annotations

import gzip
import re
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from causalbase.graph import DAG
from causalbase.io.xbif import DiscreteBN

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["load_bif", "parse_bif"]

_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_TOKENS = re.compile(r"[{}()\[\];,|]|[^\s{}()\[\];,|]+")


class _Tokens:
    """The tokens of a BIF text, read one at a time."""

    def __init__(self, text: str, source: str) -> None:
        self._items = _TOKENS.findall(_COMMENTS.sub(" ", text))
        self._pos = 0
        self.source = source

    def peek(self) -> str | None:
        return self._items[self._pos] if self._pos < len(self._items) else None

    def next(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError(f"{self.source}: unexpected end of file")
        self._pos += 1
        return token

    def expect(self, token: str) -> None:
        got = self.next()
        if got != token:
            raise ValueError(f"{self.source}: expected {token!r}, got {got!r}")

    def until(self, stop: str) -> list[str]:
        """Return the tokens up to `stop`, without the commas, and consume `stop`."""
        out = []
        while (token := self.next()) != stop:
            if token != ",":
                out.append(token)
        return out

    def skip_block(self) -> None:
        """Consume a ``{ ... }`` block, nested ones included."""
        self.expect("{")
        depth = 1
        while depth:
            token = self.next()
            depth += token == "{"
            depth -= token == "}"


def _numbers(tokens: list[str], source: str, what: str) -> list[float]:
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"{source}: {what} holds something that is not a number") from exc


def parse_bif(text: str, source: str = "<string>") -> DiscreteBN:
    """Parse the text of a BIF file into a `DiscreteBN`.

    Parameters
    ----------
    text : str
        The contents of a ``.bif`` file.
    source : str, default="<string>"
        Where the text comes from, for the error messages.

    Returns
    -------
    DiscreteBN
        The network, its variables in the order the file declares them.

    Raises
    ------
    ValueError
        If the text is not BIF, a table names an unknown variable or state,
        has the wrong size or leaves a configuration out, or a row does not
        sum to 1.

    Examples
    --------
    >>> from causalbase.io import parse_bif
    >>> bn = parse_bif('''
    ... variable rain { type discrete [ 2 ] { yes, no }; }
    ... variable wet { type discrete [ 2 ] { yes, no }; }
    ... probability ( rain ) { table 0.2, 0.8; }
    ... probability ( wet | rain ) { (yes) 0.9, 0.1; (no) 0.1, 0.9; }
    ... ''')
    >>> bn.names, bn.dag.edges()
    (['rain', 'wet'], [(0, 1)])
    """
    tokens = _Tokens(text, source)
    names: list[str] = []
    outcomes: list[list[str]] = []
    index: dict[str, int] = {}
    blocks: list[tuple[str, list[str], list[tuple[str, list[str], list[str]]]]] = []

    while (keyword := tokens.peek()) is not None:
        tokens.next()
        if keyword == "network":
            tokens.next()
            tokens.skip_block()
        elif keyword == "variable":
            name = tokens.next()
            tokens.expect("{")
            states: list[str] | None = None
            while (token := tokens.next()) != "}":
                if token == "type":
                    if tokens.next() != "discrete":
                        raise ValueError(f"{source}: {name!r} is not a discrete variable")
                    tokens.expect("[")
                    size = int(tokens.next())
                    tokens.expect("]")
                    tokens.expect("{")
                    states = tokens.until("}")
                    tokens.expect(";")
                    if len(states) != size:
                        raise ValueError(
                            f"{source}: {name!r} declares {size} states and lists {len(states)}"
                        )
                elif token == "property":
                    tokens.until(";")
                else:
                    raise ValueError(f"{source}: unexpected {token!r} in variable {name!r}")
            if not states:
                raise ValueError(f"{source}: variable {name!r} has no states")
            if name in index:
                raise ValueError(f"{source}: variable {name!r} is declared twice")
            index[name] = len(names)
            names.append(name)
            outcomes.append(states)
        elif keyword == "probability":
            tokens.expect("(")
            head = tokens.until(")")
            child, parents = head[0], [t for t in head[1:] if t != "|"]
            tokens.expect("{")
            entries: list[tuple[str, list[str], list[str]]] = []
            while (token := tokens.next()) != "}":
                if token == "(":
                    config = tokens.until(")")
                    entries.append(("row", config, tokens.until(";")))
                elif token in ("table", "default"):
                    entries.append((token, [], tokens.until(";")))
                elif token == "property":
                    tokens.until(";")
                else:
                    raise ValueError(f"{source}: unexpected {token!r} in the table of {child!r}")
            blocks.append((child, parents, entries))
        else:
            raise ValueError(f"{source}: unexpected {keyword!r} at the top level")

    n = len(names)
    parent_order: list[tuple[int, ...]] = [() for _ in range(n)]
    cpts: list[npt.NDArray[np.float64] | None] = [None] * n
    for child, parents, entries in blocks:
        for var in (child, *parents):
            if var not in index:
                raise ValueError(f"{source}: the table of {child!r} names unknown {var!r}")
        j = index[child]
        if cpts[j] is not None:
            raise ValueError(f"{source}: {child!r} has two tables")
        pa = tuple(index[p] for p in parents)
        r = len(outcomes[j])
        configs = list(product(*(outcomes[p] for p in pa)))
        row_of = {config: i for i, config in enumerate(configs)}
        table = np.full((len(configs), r), np.nan)
        for kind, config, values in entries:
            numbers = _numbers(values, source, f"the table of {child!r}")
            if kind == "table":
                if pa:
                    raise ValueError(
                        f"{source}: the table of {child!r} is given whole with 'table'; "
                        "give one row per configuration of its parents instead"
                    )
                if len(numbers) != table.size:
                    raise ValueError(
                        f"{source}: the table of {child!r} has {len(numbers)} entries, "
                        f"expected {table.size}"
                    )
                table[:] = np.reshape(numbers, table.shape)
                continue
            if len(numbers) != r:
                raise ValueError(
                    f"{source}: a row of {child!r} has {len(numbers)} entries, expected {r}"
                )
            if kind == "default":
                table[np.isnan(table[:, 0])] = numbers
            else:
                key = tuple(config)
                if key not in row_of:
                    raise ValueError(
                        f"{source}: {child!r} has a row for {key}, which is not a "
                        f"configuration of {parents}"
                    )
                table[row_of[key]] = numbers
        if np.isnan(table).any():
            raise ValueError(f"{source}: the table of {child!r} leaves configurations out")
        if not np.allclose(table.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{source}: a row of the table of {child!r} does not sum to 1")
        parent_order[j] = pa
        cpts[j] = table

    missing = [names[j] for j in range(n) if cpts[j] is None]
    if missing:
        raise ValueError(f"{source}: no table for {missing}")
    edges = [(p, j) for j in range(n) for p in parent_order[j]]
    return DiscreteBN(
        dag=DAG.from_edges(n, edges),
        names=names,
        outcomes=outcomes,
        parent_order=parent_order,
        cpts=[cpt for cpt in cpts if cpt is not None],
    )


def load_bif(path: str | Path) -> DiscreteBN:
    """Read a BIF (``.bif`` or ``.bif.gz``) file into a `DiscreteBN`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file; a name ending in ``.gz`` is decompressed.

    Returns
    -------
    DiscreteBN
        The network, its variables in the order the file declares them.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file is not a valid BIF network (see :func:`parse_bif`).
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = path.read_text(encoding="utf-8")
    return parse_bif(text, str(path))
