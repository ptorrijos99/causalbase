"""Reader for XMLBIF (``.xbif``) discrete Bayesian networks.

Each ``<TABLE>`` is read as shape ``(q, r)`` in row-major order: ``r`` is the
variable's number of outcomes, and its parents' configurations run with the
first ``<GIVEN>`` varying slowest.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causalbase.graph import DAG

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

__all__ = ["DiscreteBN", "load_xbif"]


@dataclass
class DiscreteBN:
    """A discrete Bayesian network loaded from an XMLBIF file.

    Attributes
    ----------
    dag : DAG
        Directed acyclic graph over the variables, indexed in the same
        order as `names`.
    names : list[str]
        Variable names, in the order they were declared in the source
        file. `dag` node `j` corresponds to `names[j]`.
    outcomes : list[list[str]]
        Outcome labels per variable, in declaration order.
    parent_order : list[tuple[int, ...]]
        Parents per variable, in ``<GIVEN>`` order, which is the order that
        indexes the rows of ``cpts[j]``.
    cpts : list[np.ndarray]
        ``cpts[j]`` has one row per parent configuration (first parent varying
        slowest) and one column per outcome of variable `j`. Rows sum to 1.
    """

    dag: DAG
    names: list[str]
    outcomes: list[list[str]]
    parent_order: list[tuple[int, ...]]
    cpts: list[npt.NDArray[np.float64]]


def _text(elem: ET.Element, tag: str) -> str:
    """Return the stripped text of `elem`'s first `tag` child, or raise `ValueError`."""
    child = elem.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"expected a <{tag}> element with text inside <{elem.tag}>")
    return child.text.strip()


def load_xbif(path: str | Path) -> DiscreteBN:
    """Parse an XMLBIF (``.xbif``) file into a `DiscreteBN`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.xbif`` file.

    Returns
    -------
    DiscreteBN
        The parsed network.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If an element is missing, a definition names an unknown variable, or
        a table has the wrong size or rows that do not sum to 1.
    """
    path_str = str(path)
    try:
        tree = ET.parse(path_str)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FileNotFoundError(f"XBIF file not found: {path_str}") from exc

    root = tree.getroot()
    network = root.find("NETWORK")
    if network is None:
        raise ValueError(f"{path_str}: no <NETWORK> element found")

    names: list[str] = []
    outcomes: list[list[str]] = []
    name_to_index: dict[str, int] = {}
    for var_elem in network.findall("VARIABLE"):
        name = _text(var_elem, "NAME")
        var_outcomes = [
            oc.text.strip() for oc in var_elem.findall("OUTCOME") if oc.text is not None
        ]
        if not var_outcomes:
            raise ValueError(f"{path_str}: variable {name!r} declares no <OUTCOME> entries")
        name_to_index[name] = len(names)
        names.append(name)
        outcomes.append(var_outcomes)

    num_nodes = len(names)
    parent_order: list[tuple[int, ...]] = [() for _ in range(num_nodes)]
    cpts: list[npt.NDArray[np.float64] | None] = [None for _ in range(num_nodes)]
    edges: list[tuple[int, int]] = []

    for def_elem in network.findall("DEFINITION"):
        for_name = _text(def_elem, "FOR")
        if for_name not in name_to_index:
            raise ValueError(
                f"{path_str}: <DEFINITION><FOR>{for_name}</FOR> references an undeclared variable"
            )
        j = name_to_index[for_name]

        given_names = [g.text.strip() for g in def_elem.findall("GIVEN") if g.text is not None]
        parents: list[int] = []
        for g_name in given_names:
            if g_name not in name_to_index:
                raise ValueError(
                    f"{path_str}: <DEFINITION> for {for_name!r} has <GIVEN>{g_name}</GIVEN>, "
                    "an undeclared variable"
                )
            parents.append(name_to_index[g_name])
        parent_order[j] = tuple(parents)
        for p in parents:
            edges.append((p, j))

        table_elem = def_elem.find("TABLE")
        if table_elem is None or table_elem.text is None:
            raise ValueError(f"{path_str}: <DEFINITION> for {for_name!r} has no <TABLE> entries")
        flat = [float(tok) for tok in table_elem.text.split()]

        r = len(outcomes[j])
        q = 1
        for p in parents:
            q *= len(outcomes[p])

        if len(flat) != q * r:
            raise ValueError(
                f"{path_str}: <TABLE> for {for_name!r} has {len(flat)} entries, "
                f"expected q*r = {q}*{r} = {q * r}"
            )

        table = np.array(flat, dtype=np.float64).reshape((q, r))
        row_sums = table.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"{path_str}: <TABLE> for {for_name!r} does not reshape to rows summing to 1 "
                f"under shape (q={q}, r={r}) with parents {[names[p] for p in parents]} "
                f"in declared order; row sums were {row_sums.tolist()} -- table layout is "
                "ambiguous or the file is corrupt"
            )
        cpts[j] = table

    missing = [names[j] for j, cpt in enumerate(cpts) if cpt is None]
    if missing:
        raise ValueError(f"{path_str}: no <DEFINITION> found for variable(s): {missing}")

    dag = DAG.from_edges(num_nodes, edges)
    resolved_cpts: list[npt.NDArray[np.float64]] = [cpt for cpt in cpts if cpt is not None]
    return DiscreteBN(
        dag=dag,
        names=names,
        outcomes=outcomes,
        parent_order=parent_order,
        cpts=resolved_cpts,
    )
