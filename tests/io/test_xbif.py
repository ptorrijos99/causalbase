"""Reading XMLBIF networks and sampling from them, checked on asia."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causalbase.io import forward_sample, load_xbif

ASIA = (
    Path(__file__).resolve().parents[2] / "src" / "causalbase" / "datasets" / "data" / "asia.xbif"
)


def test_asia_loads_with_its_declared_orders_and_tables() -> None:
    bn = load_xbif(ASIA)
    idx = {name: i for i, name in enumerate(bn.names)}
    assert set(idx) == {"asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"}
    assert bn.dag.num_edges == 8
    assert bn.dag.has_edge(idx["smoke"], idx["lung"])
    # Outcomes keep the file's order ("yes" then "no"), not a sorted one.
    assert all(outcomes == ["yes", "no"] for outcomes in bn.outcomes)
    assert all(np.allclose(cpt.sum(axis=1), 1.0) for cpt in bn.cpts)
    # <TABLE>0.1 0.9 0.01 0.99</TABLE>: a row per state of smoke.
    np.testing.assert_allclose(bn.cpts[idx["lung"]], [[0.1, 0.9], [0.01, 0.99]])
    # dysp is given bronc then either, the first parent slowest. The (yes, no)
    # and (no, yes) rows differ, so a transposed layout would fail.
    assert bn.parent_order[idx["dysp"]] == (idx["bronc"], idx["either"])
    dysp = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9]]
    np.testing.assert_allclose(bn.cpts[idx["dysp"]], dysp)
    with pytest.raises(FileNotFoundError):
        load_xbif(ASIA.with_name("missing.xbif"))


def test_forward_sampling_is_reproducible_and_follows_the_tables() -> None:
    bn = load_xbif(ASIA)
    ds = forward_sample(bn, 50_000, seed=0)
    assert np.array_equal(ds.values, forward_sample(bn, 50_000, seed=0).values)
    assert not np.array_equal(ds.values[:500], forward_sample(bn, 500, seed=1).values)
    assert all(list(cats) == sorted(cats) for cats in ds.categories)
    frame = ds.to_pandas()
    assert (frame["asia"] == "yes").mean() == pytest.approx(0.01, abs=0.003)
    smokers = frame[frame["smoke"] == "yes"]
    assert (smokers["lung"] == "yes").mean() == pytest.approx(0.1, abs=0.01)
    # either is a deterministic OR of lung and tub.
    either = (frame["lung"] == "yes") | (frame["tub"] == "yes")
    assert ((frame["either"] == "yes") == either).all()
