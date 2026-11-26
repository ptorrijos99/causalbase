import numpy as np
import pandas as pd
import pytest

from _samples import load_cell
from causalbase.data import DiscreteDataset
from causalbase.graph import DAG
from causalbase.order.pairs import pair_polish
from causalbase.scores import BDeuScore


@pytest.fixture(scope="module")
def xor():
    """C = A xor B (5 % noise), D independent: C has no marginal dependence on A or on B."""
    rng = np.random.default_rng(0)
    n = 3000
    a, b, d = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 2, n)
    c = (a ^ b) ^ (rng.random(n) < 0.05)
    df = pd.DataFrame({"A": a, "B": b, "C": c.astype(int), "D": d}).astype(str)
    return DiscreteDataset.from_pandas(df)


def test_single_parents_do_not_help_the_xor_node(xor):
    fast = BDeuScore(xor, prior="tetrad")
    empty = fast.local_score(2, frozenset())
    assert fast.local_score(2, frozenset({0})) < empty
    assert fast.local_score(2, frozenset({1})) < empty
    assert fast.local_score(2, frozenset({0, 1})) > empty


def test_pair_polish_joins_the_xor_triple(xor):
    # A, B and C are symmetric (each is the xor of the other two), so any of them may become
    # the child.
    fast = BDeuScore(xor, prior="tetrad")
    dag, gain, pairs = pair_polish(fast, DAG.from_edges(4, []))
    triple = {0, 1, 2}
    assert any(set(dag.parents(x)) == triple - {x} for x in triple)
    assert set(dag.parents(3)) == set() and set(dag.children(3)) == set()
    assert pairs >= 1
    assert gain == pytest.approx(
        fast.score_dag(dag) - fast.score_dag(DAG.from_edges(4, [])), abs=1e-6
    )


def test_without_pairs_the_xor_node_stays_empty(xor):
    fast = BDeuScore(xor, prior="tetrad")
    dag, _, pairs = pair_polish(fast, DAG.from_edges(4, []), n_candidates=0)
    assert pairs == 0
    assert set(dag.parents(2)) == set()


def test_pair_polish_never_lowers_the_score_and_keeps_a_dag():
    ds, _ = load_cell("alarm", 0)
    fast = BDeuScore(ds, prior="tetrad")
    start = DAG.from_edges(ds.n_vars, [(0, 1), (1, 2), (5, 7)])
    dag, gain, _ = pair_polish(fast, start)
    assert isinstance(dag, DAG) and len(dag.topological_order()) == ds.n_vars
    assert gain >= 0
    assert fast.score_dag(dag) == pytest.approx(fast.score_dag(start) + gain, rel=1e-12)
    for u, v in start.edges():
        assert u in dag.parents(v)  # only additions


def test_the_result_is_a_local_optimum_for_single_additions():
    ds, _ = load_cell("child", 0)
    fast = BDeuScore(ds, prior="tetrad")
    dag, _, _ = pair_polish(fast, DAG.from_edges(ds.n_vars, []))
    n = ds.n_vars
    for v in range(n):
        desc = {v}
        stack = [v]
        while stack:
            x = stack.pop()
            for y in dag.children(x):
                if y not in desc:
                    desc.add(y)
                    stack.append(y)
        ps = frozenset(dag.parents(v))
        base = fast.local_score(v, ps)
        for u in range(n):
            if u not in desc and u not in ps:
                assert fast.local_score(v, ps | {u}) <= base + 1e-9
