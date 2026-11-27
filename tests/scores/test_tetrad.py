"""BDeuScore with Tetrad 7.6.5's structure prior: ``BDeuScore(data, prior="tetrad")``."""

import math

import numpy as np
import pytest

from _samples import load_cell
from causalbase.graph import DAG
from causalbase.scores import BDeuScore, make_score
from causalbase.scores._compiled import CompiledBDeu
from causalbase.scores._tetrad import TetradBDeuReference
from causalbase.search import ges, xges

# The Tetrad score of the DAG the retired ges_dag learned on the alarm sample (45 arcs).
GES_ALARM_SCORE = -54062.23678185102


def _tetrad_reference(ds, node, parents, ess, structure_prior):
    """BDeu by counting rows in a dict, plus Tetrad 7.6.5's structure prior with N = rows."""
    values = np.asarray(ds.values)
    cards = ds.cardinalities
    r = int(cards[node])
    q = math.prod(int(cards[p]) for p in parents)
    counts = {}
    for row in values:
        counts.setdefault(tuple(row[list(parents)]), np.zeros(r))[row[node]] += 1
    a_j, a_jk = ess / q, ess / (q * r)
    total = 0.0
    for n_k in counts.values():
        total += math.lgamma(a_j) - math.lgamma(a_j + n_k.sum())
        total += sum(math.lgamma(a_jk + c) - math.lgamma(a_jk) for c in n_k)
    if structure_prior:
        vm = ds.n_rows - 1
        k = len(parents)
        total += k * math.log(structure_prior / vm) + (vm - k) * math.log(1 - structure_prior / vm)
    return total


@pytest.mark.parametrize("structure_prior", [1.0, 0.0])
def test_tetrad_bdeu_reference_is_the_ipmu_score(structure_prior):
    ds, _ = load_cell("alarm", 0)
    score = TetradBDeuReference(ds, structure_prior=structure_prior)
    for node, parents in [(0, ()), (3, (1,)), (5, (2, 7)), (12, (4, 9, 30))]:
        expected = _tetrad_reference(ds, node, parents, 1.0, structure_prior)
        assert score.local_score(node, frozenset(parents)) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("expected_parents", [1.0, 0.0, None, 2.5])
def test_values_are_the_compiled_engines_bit_for_bit(expected_parents):
    ds, true = load_cell("alarm", 0)
    score = BDeuScore(ds, ess=2.0, expected_parents=expected_parents, prior="tetrad")
    engine = CompiledBDeu(ds, 2.0, expected_parents, prior="tetrad")
    assert isinstance(score, BDeuScore)
    assert score.prior == "tetrad" and score.ess == 2.0
    assert score.expected_parents == expected_parents
    rng = np.random.default_rng(0)
    for _ in range(40):
        node = int(rng.integers(ds.n_vars))
        others = [v for v in range(ds.n_vars) if v != node]
        parents = sorted(
            int(p) for p in rng.choice(others, size=int(rng.integers(0, 4)), replace=False)
        )
        cands = [v for v in others if v not in parents]
        assert score.local_score(node, parents) == engine.local_score(node, parents)
        assert list(score.local_scores_with(node, frozenset(parents), cands)) == list(
            engine.local_scores_add(node, parents, cands)
        )
    assert score.score_dag(true) == engine.score_dag(true)
    assert score.local_score(5, (2, 7)) == pytest.approx(
        _tetrad_reference(ds, 5, (2, 7), 2.0, expected_parents), rel=1e-10
    )


def test_no_prior_is_zero_or_none():
    ds, true = load_cell("asia", 0)
    zero = BDeuScore(ds, expected_parents=0.0, prior="tetrad")
    none = BDeuScore(ds, expected_parents=None, prior="tetrad")
    assert zero.score_dag(true) == none.score_dag(true)


def test_counts_every_family_its_engine_scores():
    ds, _ = load_cell("alarm", 0)
    score = BDeuScore(ds, prior="tetrad")
    assert score.n_families == 0
    score.local_score(3, frozenset({1}))
    score.local_scores_with(3, frozenset({1}), [0, 2, 4, 5])
    assert score.n_families == 5
    assert score._engine() is score._engine()
    assert score._engine().n_families == 5


def test_search_view_has_the_same_structure_differences():
    # GES and XGES search the Bernoulli BDeu with Tetrad's p; it differs from Tetrad's value
    # by a constant per family.
    ds, _ = load_cell("alarm", 0)
    score = BDeuScore(ds, ess=5.0, expected_parents=2.0, prior="tetrad")
    view = score._search_view()
    assert type(view) is BDeuScore and view.prior == "bernoulli" and view.data is ds
    assert score._search_view() is not view  # a new one, with an empty cache, per call
    reference = TetradBDeuReference(ds, ess=5.0, structure_prior=2.0).bdeu
    offset = score.local_score(0, ()) - view.local_score(0, ())
    for node, parents in [(0, ()), (3, (1,)), (5, (2, 7)), (12, (4, 9, 30))]:
        assert view.local_score(node, parents) == reference.local_score(node, parents)
        assert score.local_score(node, parents) - view.local_score(node, parents) == (
            pytest.approx(offset, rel=1e-9)
        )


def test_search_view_without_the_structure_prior_is_plain_bdeu():
    ds, _ = load_cell("asia", 0)
    view = BDeuScore(ds, ess=2.0, expected_parents=0.0, prior="tetrad")._search_view()
    assert view.expected_parents is None
    assert view.local_score(3, (1,)) == BDeuScore(ds, ess=2.0, expected_parents=None).local_score(
        3, (1,)
    )


def test_a_bernoulli_score_is_its_own_search_view():
    ds, _ = load_cell("asia", 0)
    score = BDeuScore(ds)
    assert score.prior == "bernoulli" and score._search_view() is score


@pytest.mark.parametrize("net", ["asia", "child", "alarm"])
def test_ges_and_xges_learn_what_they_learned_before(net):
    # The retired ges_dag and xges_dag ran on TetradBDeu(data).bdeu.
    ds, _ = load_cell(net, 0)
    score = BDeuScore(ds, prior="tetrad")
    result = ges(score)
    # The search runs on the view; the engine scores only the networks the result reports.
    assert score.n_families <= (len(result.trace) + 1) * ds.n_vars
    assert result.cpdag == ges(TetradBDeuReference(ds).bdeu).cpdag
    for extended in (False, True):
        assert (
            xges(score, extended=extended).cpdag
            == xges(TetradBDeuReference(ds).bdeu, extended=extended).cpdag
        )


def test_ges_gives_the_pinned_alarm_network():
    # Recorded with the retired ges_dag on the alarm sample (TetradBDeu objective).
    ds, _ = load_cell("alarm", 0)
    score = BDeuScore(ds, prior="tetrad")
    dag = ges(score).dag
    assert isinstance(dag, DAG) and dag.num_edges == 45
    assert score.score_dag(dag) == pytest.approx(GES_ALARM_SCORE, rel=1e-12)


def test_extended_xges_never_ends_below_xges0():
    ds, _ = load_cell("insurance", 0)
    score = BDeuScore(ds, prior="tetrad")
    x0 = xges(score, extended=False).dag
    x = xges(score).dag
    assert isinstance(x, DAG) and isinstance(x0, DAG)
    assert score.score_dag(x) >= score.score_dag(x0) - 1e-9


def test_make_score_builds_either_prior():
    ds, true = load_cell("asia", 0)
    tetrad = make_score("bdeu", ds, ess=2.0, expected_parents=1.5, prior="tetrad")
    assert tetrad.score_dag(true) == BDeuScore(
        ds, ess=2.0, expected_parents=1.5, prior="tetrad"
    ).score_dag(true)
    assert make_score("bdeu", ds).score_dag(true) == BDeuScore(ds).score_dag(true)
    with pytest.raises(ValueError, match="prior"):
        make_score("bic", ds, prior="tetrad")
    with pytest.raises(ValueError, match="prior"):
        make_score("bdeu", ds, prior="uniform")


def test_refuses_bad_settings():
    ds, _ = load_cell("asia", 0)
    with pytest.raises(ValueError, match="prior"):
        BDeuScore(ds, prior="uniform")
    with pytest.raises(ValueError, match="expected_parents"):
        BDeuScore(ds, expected_parents=-1.0, prior="tetrad")
    with pytest.raises(ValueError, match="expected_parents"):
        BDeuScore(ds, expected_parents=float(ds.n_rows - 1), prior="tetrad")
    with pytest.raises(ValueError, match="expected_parents"):
        BDeuScore(ds, expected_parents=0.0)  # the Bernoulli prior needs a positive value
    with pytest.raises(ValueError, match="ess"):
        BDeuScore(ds, ess=0.0, prior="tetrad")
    with pytest.raises(ValueError, match="n_jobs"):
        BDeuScore(ds, n_jobs=0)


def test_tetrads_score_is_no_longer_a_class_of_its_own():
    import causalbase.scores

    assert not hasattr(causalbase.scores, "TetradBDeuScore")
    assert "TetradBDeuScore" not in causalbase.scores.__all__
