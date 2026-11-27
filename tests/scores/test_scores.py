"""The scores against independent definitions, and the batch kernels against single families."""

from __future__ import annotations

import math

import numpy as np
import pytest

from causalbase.data import ContinuousDataset, DiscreteDataset
from causalbase.graph import DAG
from causalbase.scores import BDeuScore, BICScore, GaussianBICScore, log_likelihood, make_score
from causalbase.scores._kernels import (
    extension_buffers,
    family_extensions,
    family_score,
    value_bits,
)


def _discrete(n_rows: int, cards: list[int], seed: int) -> DiscreteDataset:
    rng = np.random.default_rng(seed)
    values = np.column_stack([rng.integers(0, c, size=n_rows) for c in cards])
    names = [f"x{i}" for i in range(len(cards))]
    return DiscreteDataset(values, cards, names, [[str(k) for k in range(c)] for c in cards])


def _continuous(n_rows: int, seed: int) -> ContinuousDataset:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n_rows)
    x1 = 0.8 * x0 + rng.normal(size=n_rows)
    x2 = -0.5 * x1 + 0.3 * x0 + rng.normal(size=n_rows)
    return ContinuousDataset(np.column_stack([x0, x1, x2, rng.normal(size=n_rows)]), list("abcd"))


def _counts(data: DiscreteDataset, node: int, parents: tuple[int, ...]) -> list[np.ndarray]:
    r = int(data.cardinalities[node])
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for row in data.values:
        counts.setdefault(tuple(row[list(parents)]), np.zeros(r))[row[node]] += 1
    return list(counts.values())


def _bdeu(data: DiscreteDataset, node: int, parents: tuple[int, ...], ess: float) -> float:
    q = math.prod(int(data.cardinalities[p]) for p in parents)
    a_j, a_jk = ess / q, ess / (q * int(data.cardinalities[node]))
    return sum(
        math.lgamma(a_j)
        - math.lgamma(a_j + n.sum())
        + sum(math.lgamma(a_jk + c) - math.lgamma(a_jk) for c in n)
        for n in _counts(data, node, parents)
    )


def test_bdeu_matches_the_definition_even_past_any_integer_code() -> None:
    data = _discrete(300, [3] * 50, seed=1)
    # 3**45 parent configurations overflow any direct encoding; they must be relabelled.
    for parents in [(), (1,), (1, 3), (1, 2, 3, 4), tuple(range(1, 13)), tuple(range(1, 46))]:
        for ess in (1.0, 10.0):
            score = BDeuScore(data, ess=ess, expected_parents=None)
            assert score.local_score(0, parents) == pytest.approx(
                _bdeu(data, 0, parents, ess), rel=1e-10
            )


def test_bic_scores_match_their_definitions() -> None:
    data = _discrete(500, [3, 2, 4, 3, 2], seed=0)
    plain = BICScore(data, expected_parents=None)
    for parents in [(), (1,), (1, 3)]:
        r, q = int(data.cardinalities[0]), math.prod(int(data.cardinalities[p]) for p in parents)
        counts = _counts(data, 0, parents)
        lik = sum(float(n @ np.log(np.where(n > 0, n / n.sum(), 1))) for n in counts)
        expected = lik - math.log(500) / 2 * q * (r - 1)
        assert plain.local_score(0, parents) == pytest.approx(expected, rel=1e-10)
    continuous = _continuous(2000, seed=1)
    x = continuous.values
    design = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])
    residual = x[:, 2] - design @ np.linalg.lstsq(design, x[:, 2], rcond=None)[0]
    lik = -len(x) / 2 * (math.log(2 * math.pi * residual @ residual / len(x)) + 1)
    gaussian = GaussianBICScore(continuous, expected_parents=None)
    assert gaussian.local_score(2, {0, 1}) == pytest.approx(
        lik - math.log(len(x)) / 2 * 4, rel=1e-10
    )


def test_the_prior_and_the_penalty_charge_per_parent() -> None:
    data = _discrete(300, [2] * 6, seed=4)
    plain, sparse = BDeuScore(data, expected_parents=None), BDeuScore(data)
    per_parent = math.log(1 / 5) - math.log(4 / 5)  # six variables, one expected parent
    for parents in [(), (1,), (3, 1, 2)]:
        gap = sparse.local_score(0, parents) - plain.local_score(0, parents)
        assert gap == pytest.approx(len(parents) * per_parent)
    assert plain.local_score(0, (3, 1, 2)) == plain.local_score(0, (1, 2, 3))
    dag = DAG.from_edges(6, [(0, 1), (1, 3), (2, 3)])
    assert sparse.score_dag(dag) == pytest.approx(
        sum(sparse.local_score(v, dag.parents(v)) for v in range(6))
    )
    continuous = _continuous(300, seed=4)
    plain_g = GaussianBICScore(continuous, expected_parents=None)
    harsher = [
        GaussianBICScore(continuous, penalty_discount=2.0, expected_parents=None),
        GaussianBICScore(continuous),
    ]
    assert all(h.local_score(2, {0, 1}) < plain_g.local_score(2, {0, 1}) for h in harsher)


def test_a_batch_of_extensions_equals_each_family_alone_bit_for_bit() -> None:
    """GES compares batch and single scores directly, so a last-bit difference changes ties.

    Families too wide for the dense table and narrow ones alike, with a deterministic
    column for many empty cells, under BDeu and BIC.
    """
    cards = [3, 21, 13, 8, 2, 21, 5, 17, 2, 11]
    data = _discrete(250, cards, seed=7)
    data.values[:, 4] = (data.values[:, 1] + data.values[:, 2]) % 2
    for node, parents in [(0, ()), (4, (1, 2)), (3, (1, 5, 7)), (9, (0, 1, 5, 7))]:
        extra = [x for x in range(len(cards)) if x != node and x not in parents]
        for make in (BDeuScore, BICScore):
            together = make(data).local_scores_with(node, frozenset(parents), extra)
            assert list(together) == [make(data).local_score(node, {*parents, x}) for x in extra]
        families = [tuple(sorted({*parents, x})) for x in extra]
        alone = [BDeuScore(data, expected_parents=None).local_score(node, f) for f in families]
        assert alone == pytest.approx([_bdeu(data, node, f, 1.0) for f in families], rel=1e-10)
    continuous = _continuous(400, seed=3)
    together = GaussianBICScore(continuous).local_scores_with(0, frozenset({1}), [2, 3])
    alone = [GaussianBICScore(continuous).local_score(0, {1, x}) for x in (2, 3)]
    np.testing.assert_allclose(together, alone, rtol=1e-10)


def test_reused_buffers_leave_nothing_behind() -> None:
    """One set of buffers serves every family of a chunk, call after call: a family
    counted by bitsets must leave nothing behind for one too wide for the dense table,
    and terms cached for one family's priors must not reach a family with others."""
    rng = np.random.default_rng(11)
    cards = np.array([3, 2, 120, 2, 97], dtype=np.int64)
    columns = [rng.integers(0, c, size=60) for c in cards]
    values = np.asfortranarray(np.column_stack(columns).astype(np.int32))
    bits, offsets = value_bits(values, cards)
    none = np.zeros(0, dtype=np.int64)
    scratch = np.empty(240, dtype=np.int64)
    buffers = extension_buffers(60, cards, len(scratch), 1)
    for node in (0, 1, 3, 0):
        extra = np.array([x for x in range(5) if x != node], dtype=np.int64)
        together = family_extensions(
            values, cards, node, none, extra, 1.0, len(scratch), bits, offsets, *buffers
        )
        one = [np.array([x], dtype=np.int64) for x in extra]
        assert list(together) == [family_score(values, cards, node, p, 1.0, scratch) for p in one]


def test_held_out_likelihood() -> None:
    train, test = _continuous(3000, seed=6), _continuous(1000, seed=7)
    truth = DAG.from_edges(4, [(0, 1), (0, 2), (1, 2)])
    assert log_likelihood(truth, train, test) > log_likelihood(DAG(4), train, test)
    # The discrete one is a Bayesian predictive likelihood: equal across a class.
    data = _discrete(600, [3, 3, 3], seed=8)
    train_d, test_d = data.subset(np.arange(400)), data.subset(np.arange(400, 600))
    forward, backward = DAG.from_edges(3, [(0, 1), (1, 2)]), DAG.from_edges(3, [(2, 1), (1, 0)])
    assert log_likelihood(forward, train_d, test_d) == pytest.approx(
        log_likelihood(backward, train_d, test_d), rel=1e-10
    )
    # np.cov reads a lone row as one variable; the Gaussian score must not.
    for n_rows in (0, 1):
        score = GaussianBICScore(ContinuousDataset(np.ones((n_rows, 3)), ["a", "b", "c"]))
        assert np.isfinite(score.local_scores_with(0, frozenset(), [1, 2])).all()


def test_scores_refuse_bad_settings_and_data() -> None:
    discrete, continuous = _discrete(10, [2, 2], seed=0), _continuous(10, seed=0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ess"):
            BDeuScore(discrete, ess=bad)
        with pytest.raises(ValueError, match="penalty_discount"):
            BICScore(discrete, penalty_discount=bad)
        with pytest.raises(ValueError, match="penalty_discount"):
            GaussianBICScore(continuous, penalty_discount=bad)
    for bad in (0.0, float("nan")):
        with pytest.raises(ValueError, match="expected_parents"):
            BDeuScore(discrete, expected_parents=bad)
    with pytest.raises(ValueError, match=r"no levels.*'a'"):
        BDeuScore(DiscreteDataset(np.zeros((0, 2)), [0, 2], ["a", "b"], [[], ["x", "y"]]))
    defaults = (make_score("bdeu", discrete), BICScore(discrete), GaussianBICScore(continuous))
    assert all(s.expected_parents == 1.0 for s in defaults)
    with pytest.raises(ValueError, match="score must be one of"):
        make_score("aic", discrete)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ContinuousDataset"):
        make_score("gaussian_bic", discrete)


def test_a_gaussian_parent_the_others_determine_explains_nothing_more() -> None:
    """The residual is the least-squares one, and the parent still pays its penalty."""
    rng = np.random.default_rng(5)
    for seed in range(5):
        x = _continuous(300, seed).values
        weights = rng.normal(size=2)
        values = np.column_stack([x, x[:, 0] * weights[0] + x[:, 1] * weights[1]])
        data = ContinuousDataset(values, ["a", "b", "c", "d", "ab"])
        score = GaussianBICScore(data, expected_parents=None)
        design = np.column_stack([np.ones(300), values[:, [0, 1, 4]]])
        residual = values[:, 2] - design @ np.linalg.lstsq(design, values[:, 2], rcond=None)[0]
        lik = -300 / 2 * (math.log(2 * math.pi * (residual @ residual) / 300) + 1)
        expected = lik - math.log(300) / 2 * (3 + 2)
        assert score.local_score(2, {0, 1, 4}) == pytest.approx(expected, rel=1e-9)
        together = score.local_scores_with(2, frozenset({0, 1}), [4])
        assert together[0] == pytest.approx(expected, rel=1e-9)
