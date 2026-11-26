"""Everything worth keeping about a learned network.

No run then has to be repeated for a metric.

:func:`graph_record` stores the arcs themselves next to the usual scores and
distances; any other metric can be computed later from the arcs.
"""

from __future__ import annotations

from typing import Any

from causalbase.graph import DAG
from causalbase.graph.cpdag import dag_to_cpdag
from causalbase.metrics.structural import shd, shd_skeleton, smhd

__all__ = ["graph_record"]


def graph_record(true: DAG, dag: DAG, score: Any) -> dict[str, Any]:
    """Arcs, score and distances to the true network of a learned DAG.

    Returns a JSON-serialisable dict with ``edges`` (``[parent, child]`` pairs),
    ``n_edges``, ``bdeu`` (``score`` summed over the families of ``dag``),
    ``smhd`` (moral graphs), ``shd_dag`` (arcs added, removed or reversed),
    ``shd_cpdag`` (between the equivalence classes) and ``shd_skeleton``.
    """
    edges = [[int(u), int(v)] for v in range(dag.num_nodes) for u in sorted(dag.parents(v))]
    return {
        "edges": edges,
        "n_edges": len(edges),
        "bdeu": score.score_dag(dag),
        "smhd": int(smhd(true, dag)),
        "shd_dag": int(shd(true, dag)),
        "shd_cpdag": int(shd(dag_to_cpdag(true), dag_to_cpdag(dag))),
        "shd_skeleton": int(shd_skeleton(true, dag)),
    }
