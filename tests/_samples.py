"""The samples the order-search tests were written against.

``fixtures/samples/<net>.csv.gz`` is sample 0 of each network in the IPMU 2024
experiments (5000 rows), and ``<net>.xbif`` its network where a test needs the
true graph. Pinned values in the tests come from these samples.
"""

from __future__ import annotations

from pathlib import Path

from causalbase.data import DiscreteDataset
from causalbase.graph import DAG
from causalbase.io import load_xbif

SAMPLES = Path(__file__).resolve().parent / "fixtures" / "samples"


def load_cell(net: str, i: int = 0) -> tuple[DiscreteDataset, DAG | None]:
    """Sample ``i`` (only 0 ships) of ``net`` and its true DAG indexed like the columns.

    The DAG is None for the networks whose definition is not shipped.
    """
    if i != 0:
        raise ValueError("only sample 0 ships with the tests")
    ds = DiscreteDataset.from_csv(SAMPLES / f"{net}.csv.gz")
    path = SAMPLES / f"{net}.xbif"
    if not path.exists():
        return ds, None
    bn = load_xbif(path)
    idx = [ds.names.index(name) for name in bn.names]
    edges = [(idx[u], idx[v]) for v in range(len(bn.names)) for u in bn.dag.parents(v)]
    return ds, DAG.from_edges(ds.n_vars, edges)
