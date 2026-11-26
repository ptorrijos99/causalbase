"""The structural metrics against their definitions over edge sets."""

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalbase.graph import DAG, PDAG
from causalbase.metrics.structural import EdgeScores, edge_scores, shd, shd_skeleton, smhd


@st.composite
def _pairs(draw: st.DrawFn) -> tuple[DAG, DAG]:
    n = draw(st.integers(0, 7))

    def dag() -> DAG:
        order = draw(st.permutations(range(n)))
        pairs = itertools.combinations(range(n), 2)
        return DAG.from_edges(n, [(order[i], order[j]) for i, j in pairs if draw(st.booleans())])

    return dag(), dag()


def _skeleton(dag: DAG) -> set[frozenset[int]]:
    return {frozenset(e) for e in dag.edges()}


def _moral(dag: DAG) -> set[frozenset[int]]:
    parents = (dag.parents(v) for v in range(dag.num_nodes))
    married = {frozenset(p) for ps in parents for p in itertools.combinations(ps, 2)}
    return _skeleton(dag) | married


def _prf(hits: int, n_est: int, n_true: int) -> tuple[float, float, float]:
    if n_est == n_true == 0:
        return 1.0, 1.0, 1.0
    if 0 in (n_est, n_true) or hits == 0:
        return 0.0, 0.0, 0.0
    p, r = hits / n_est, hits / n_true
    return p, r, 2 * p * r / (p + r)


@given(_pairs())
@settings(max_examples=150, deadline=None)
def test_metrics_follow_their_definitions(pair: tuple[DAG, DAG]) -> None:
    g, h = pair
    adjacent_in_one = _skeleton(g) ^ _skeleton(h)
    reversed_ = {frozenset(e) for e in g.edges() if h.has_edge(e[1], e[0])}
    assert shd(g, h) == shd(h, g) == len(adjacent_in_one) + len(reversed_)
    assert shd_skeleton(g, h) == len(adjacent_in_one)
    assert smhd(g, h) == smhd(h, g) == len(_moral(g) ^ _moral(h))
    directed = len(set(g.edges()) & set(h.edges()))
    skeleton = len(_skeleton(g) & _skeleton(h))
    expected = _prf(directed, h.num_edges, g.num_edges) + _prf(skeleton, h.num_edges, g.num_edges)
    assert edge_scores(g, h) == pytest.approx(EdgeScores(*expected))


def test_shd_compares_cpdags_mark_by_mark() -> None:
    chain = PDAG(3)
    chain.add_undirected(0, 1)
    chain.add_undirected(1, 2)
    collider = PDAG.from_dag(DAG.from_edges(3, [(0, 1), (2, 1)]))
    assert shd(chain, chain) == 0
    assert shd(chain, collider) == 2
    assert shd(chain, DAG.from_edges(3, [(0, 1)])) == 2
    with pytest.raises(ValueError, match="node count"):
        shd(chain, DAG(4))
