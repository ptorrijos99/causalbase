"""Reading BIF networks, checked against the XMLBIF reader on the bundled networks."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from _bif import to_bif
from causalbase.datasets import list_networks
from causalbase.io import DiscreteBN, load_bif, load_xbif, parse_bif

DATA = Path(__file__).resolve().parents[2] / "src" / "causalbase" / "datasets" / "data"


def by_name(bn: DiscreteBN) -> dict[str, tuple[list[str], tuple[str, ...], np.ndarray]]:
    return {
        name: (outcomes, tuple(bn.names[p] for p in parents), cpt)
        for name, outcomes, parents, cpt in zip(
            bn.names, bn.outcomes, bn.parent_order, bn.cpts, strict=True
        )
    }


@pytest.mark.parametrize("name", list(list_networks()))
def test_a_network_written_as_bif_reads_back_as_the_same_network(name: str) -> None:
    # Rows in a random order exercise the mapping from a configuration to its row.
    xbif = load_xbif(DATA / f"{name}.xbif")
    bif = parse_bif(to_bif(xbif, np.random.default_rng(len(name))))
    expected, got = by_name(xbif), by_name(bif)
    assert expected.keys() == got.keys()
    for var, (outcomes, parents, cpt) in expected.items():
        assert got[var][:2] == (outcomes, parents)
        np.testing.assert_allclose(got[var][2], cpt, rtol=0, atol=1e-15)
    assert sorted(bif.dag.edges()) == sorted(
        (bif.names.index(xbif.names[u]), bif.names.index(xbif.names[v]))
        for u, v in xbif.dag.edges()
    )


def test_files_comments_properties_and_default_rows(tmp_path: Path) -> None:
    text = """// asia's first two variables
    network "toy" { property author me ; }
    variable asia { type discrete [ 2 ] { yes, no }; property position = (1, 2) ; }
    variable tub { type discrete [ 2 ] { yes, no }; }
    /* a comment
       over two lines */
    probability ( asia ) { table 0.01, 0.99; }
    probability ( tub | asia ) { (yes) 0.05, 0.95; default 0.01, 0.99; }
    """
    plain, packed = tmp_path / "toy.bif", tmp_path / "toy.bif.gz"
    plain.write_text(text, encoding="utf-8")
    with gzip.open(packed, "wt", encoding="utf-8") as fh:
        fh.write(text)
    for bn in (load_bif(plain), load_bif(packed)):
        assert bn.names == ["asia", "tub"] and bn.dag.edges() == [(0, 1)]
        np.testing.assert_allclose(bn.cpts[1], [[0.05, 0.95], [0.01, 0.99]])
    with pytest.raises(FileNotFoundError):
        load_bif(tmp_path / "missing.bif")


HEAD = "variable a { type discrete [ 2 ] { x, y }; } variable b { type discrete [ 2 ] { x, y }; }\n"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("probability ( a ) { table 0.5, 0.5; }", "no table for"),
        ("probability ( a ) { table 0.5, 0.6; } probability ( b ) { table 1, 0; }", "sum to 1"),
        ("probability ( a ) { table 1, 0; } probability ( b | a ) { (x) 1, 0; }", "leaves"),
        ("probability ( a ) { table 1, 0; } probability ( b | a ) { (z) 1, 0; }", "configuration"),
        ("probability ( a ) { table 1, 0; } probability ( b | c ) { (x) 1, 0; }", "unknown"),
        ("probability ( a ) { table 1, 0; } probability ( b | a ) { table 1, 0, 0, 1; }", "whole"),
        ("probability ( a ) { table 1, 0, 0; } probability ( b ) { table 1, 0; }", "entries"),
        ("probability ( a ) { table 1, zero; } probability ( b ) { table 1, 0; }", "number"),
        ("variable a { type continuous; }", "not a discrete"),
    ],
)
def test_malformed_networks_are_refused_with_a_reason(body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_bif(HEAD + body)
