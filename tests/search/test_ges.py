"""GES and XGES: what they recover, and that they stop at a true local optimum."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from causalbase import GES, XGES
from causalbase.data import as_continuous, as_discrete
from causalbase.datasets import load_alarm, load_asia, load_network
from causalbase.graph import DAG, PDAG
from causalbase.graph.cpdag import dag_to_cpdag
from causalbase.metrics import shd
from causalbase.scores import BDeuScore, FamilyScore, make_score
from causalbase.search import Knowledge, ges, xges
from causalbase.search import _operators as ops

# The score GES reaches on each bundled sample, and the SHD of what it learns. A
# change in a score means GES now stops at another local optimum; a change in an
# SHD alone, another structure of the same score. Check that either is intended,
# and say so in the changelog. Scores are compared to 1e-9, not bit for bit.
LEARNED = {
    ("asia", "bdeu"): (-11292.382734195426, 2),
    ("child", "bdeu"): (-62114.38162544265, 2),
    ("insurance", "bdeu"): (-67131.24624087544, 17),
    ("alarm", "bdeu"): (-53644.30126961614, 10),
    ("asia", "bic"): (-11303.722075470887, 2),
    ("alarm", "bic"): (-54594.88083963548, 12),
}

# The same for XGES with BDeu, keyed by `extended`: False is XGES-0.
LEARNED_XGES = {
    ("asia", False): (-11292.382734195426, 2),
    ("asia", True): (-11292.382734195426, 2),
    ("child", False): (-62114.38162544265, 2),
    ("child", True): (-62114.38162544265, 2),
    ("insurance", False): (-67131.24624087544, 17),
    ("insurance", True): (-67097.97708892381, 13),
    ("alarm", False): (-53504.66821786846, 4),
    ("alarm", True): (-53504.66821786846, 4),
}


def test_learns_the_same_structures_as_before() -> None:
    for (name, criterion), (score, distance) in LEARNED.items():
        bundle = load_network(name)
        model = GES(criterion=criterion).fit(bundle.data)
        assert model.score_ == pytest.approx(score, rel=1e-9), (name, criterion)
        assert shd(bundle.truth, model.dag_) == distance, (name, criterion)
    for (name, extended), (score, distance) in LEARNED_XGES.items():
        bundle = load_network(name)
        model = XGES(extended=extended).fit(bundle.data)
        assert model.score_ == pytest.approx(score, rel=1e-9), (name, extended)
        assert shd(bundle.truth, model.dag_) == distance, (name, extended)


def test_the_estimator_starts_from_a_given_dag() -> None:
    bundle = load_alarm()
    model = GES().fit(bundle.data, initial_graph=bundle.truth)
    assert model.score_ >= BDeuScore(bundle.data).score_dag(bundle.truth)
    assert model.cpdag_ == dag_to_cpdag(model.dag_)
    with pytest.raises(ValueError, match="nodes"):
        GES().fit(bundle.data, initial_graph=DAG(3))


def test_only_candidate_pairs_gain_edges() -> None:
    data = load_asia().data
    allowed = frozenset({(0, 1), (1, 0), (2, 3), (3, 2)})
    cpdag = ges(BDeuScore(data), candidates=allowed)
    edges = cpdag.directed_edges() + cpdag.undirected_edges()
    assert {frozenset(e) for e in edges} <= {frozenset(p) for p in allowed}


def test_the_searches_take_the_number_of_variables_from_the_score() -> None:
    data = load_asia().data
    score = BDeuScore(data)
    assert ges(score).num_nodes == xges(score).num_nodes == data.n_vars
    with pytest.raises(ValueError, match="nodes"):
        ges(score, DAG(3))


def test_the_searches_take_tetrads_score_through_its_view() -> None:
    from causalbase.scores import BDeuScore

    data = load_asia().data
    score = BDeuScore(data, prior="tetrad")
    for search in (ges, xges):
        assert search(score) == search(score._search_view())
        assert search(score).num_nodes == score.n_vars


def test_the_estimator_takes_frames_and_arrays_and_names_its_result() -> None:
    frame = load_asia().data.to_pandas()
    by_frame = GES().fit(frame)
    by_array = GES().fit(frame.to_numpy())

    assert by_frame.cpdag_ == by_array.cpdag_
    assert list(by_frame.feature_names_in_) == list(frame.columns)
    assert list(by_array.feature_names_in_) == [f"x{j}" for j in range(frame.shape[1])]
    adjacency = by_frame.adjacency_
    assert isinstance(adjacency, pd.DataFrame)
    assert adjacency.to_numpy().sum() == 2 * len(by_frame.cpdag_.undirected_edges()) + len(
        by_frame.cpdag_.directed_edges()
    )
    assert by_frame.score_ == pytest.approx(BDeuScore(load_asia().data).score_dag(by_frame.dag_))


def test_required_and_forbidden_adjacencies_hold_whatever_the_data_say() -> None:
    bundle = load_asia()
    names = bundle.data.names
    learned = {frozenset(e) for e in GES().fit(bundle.data).dag_.edges()}
    present = next(iter(learned))
    absent = next(
        frozenset((a, b))
        for a in range(len(names))
        for b in range(a + 1, len(names))
        if frozenset((a, b)) not in learned
    )
    for cls in (GES, XGES):
        model = cls(
            required=[tuple(names[i] for i in absent)],
            forbidden=[tuple(sorted(present))],
        ).fit(bundle.data)
        skeleton = {frozenset(e) for e in model.dag_.edges()}
        assert absent in skeleton, cls
        assert present not in skeleton, cls


def test_knowledge_rejects_contradictions_and_unknown_names() -> None:
    from causalbase.search import Knowledge

    with pytest.raises(ValueError, match="both required and forbidden"):
        Knowledge.from_pairs([("a", "b")], [("b", "a")], names=["a", "b"])
    with pytest.raises(ValueError, match="unknown variable"):
        Knowledge.from_pairs([("a", "z")], names=["a", "b"])
    with pytest.raises(ValueError, match="no variable at index"):
        Knowledge.from_pairs(forbidden=[(0, 2)], names=["a", "b"])
    with pytest.raises(ValueError, match="neither"):
        Knowledge.from_pairs([(0, 1.7)], names=["a", "b"])
    with pytest.raises(ValueError, match="criterion must be one of"):
        GES(criterion="aic").fit(load_asia().data)  # type: ignore[arg-type]


def test_max_parents_caps_every_family() -> None:
    for cls in (GES, XGES):
        model = cls(max_parents=1).fit(load_alarm().data)
        assert max(len(model.dag_.parents(v)) for v in range(model.n_features_in_)) == 1


def _linear_gaussian(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = rng.normal(size=n)
    d = 0.7 * b - 0.8 * c + rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def test_gaussian_bic_finds_the_collider_of_continuous_data() -> None:
    for cls in (GES, XGES):
        adjacency = cls(criterion="gaussian_bic").fit(_linear_gaussian(5000, 0)).adjacency_
        assert adjacency.loc["b", "d"] == 1 and adjacency.loc["d", "b"] == 0
        assert adjacency.loc["c", "d"] == 1 and adjacency.loc["d", "c"] == 0
        assert adjacency.loc["a", "b"] == 1 and adjacency.loc["b", "a"] == 1
        assert adjacency.loc["a", "c"] == 0


def test_score_is_the_held_out_likelihood_per_row_and_ranks_structures() -> None:
    frame = load_asia().data.to_pandas()
    train, test = frame.iloc[:3000], frame.iloc[3000:]
    learned = GES().fit(train)
    empty = GES(forbidden=[(a, b) for a in frame for b in frame if a < b]).fit(train)
    assert learned.dag_.num_edges > 0
    assert empty.dag_.num_edges == 0
    assert learned.score(test) > empty.score(test)
    assert -10 < learned.score(test) < 0
    with pytest.raises(ValueError, match="outside its levels"):
        learned.score(test.assign(asia="maybe"))
    model_selection = pytest.importorskip("sklearn.model_selection")
    search = model_selection.GridSearchCV(GES(), {"max_parents": [0, 2]}, cv=2)
    search.fit(load_asia().data.to_pandas())
    assert search.best_params_ == {"max_parents": 2}


def _problems(count: int) -> list[tuple[FamilyScore, DAG]]:
    """Small dense random problems: linear Gaussian data, as XGES's authors simulate it.

    Dense graphs and few rows are where the searches go wrong in the ways the
    tests look for. Every third problem is discretised into terciles and scored
    by BDeu.
    """
    rng = np.random.default_rng(1)
    problems = []
    for i in range(count):
        n_vars = 8
        order = rng.permutation(n_vars)
        weights = np.zeros((n_vars, n_vars))
        for a in range(n_vars):
            for b in range(a + 1, n_vars):
                if rng.random() < 0.85:
                    weights[order[a], order[b]] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
        x = np.zeros((300, n_vars))
        for v in order:
            weights[:, v] /= max(np.abs(weights[:, v]).sum(), 1.0)
            x[:, v] = x @ weights[:, v] + rng.normal(scale=rng.uniform(0.5, 1.0), size=300)
        edges = [(int(u), int(v)) for u, v in zip(*np.nonzero(weights), strict=True)]
        truth = DAG.from_edges(n_vars, edges)
        if i % 3:
            score = make_score("gaussian_bic", as_continuous(x), expected_parents=None)
        else:
            terciles = np.argsort(np.argsort(x, axis=0), axis=0) * 3 // len(x)
            score = make_score("bdeu", as_discrete(terciles))
        problems.append((score, truth))
    return problems


def _value(score: FamilyScore, graph: DAG | PDAG) -> float:
    return score.score_dag(graph if isinstance(graph, DAG) else graph.to_dag())


def _improving(score: FamilyScore, cpdag: PDAG, kinds: str) -> list[tuple[str, int, int]]:
    """Every operator of the given kinds (I, D, R) that would still improve the score."""
    found = []
    for y in range(cpdag.num_nodes):
        if "I" in kinds:
            found += [("I", x, y) for x in ops.inserts_into(score, cpdag, y, None)]
        if "D" in kinds:
            found += [("D", x, y) for x in ops.deletes_into(score, cpdag, y)]
        if "R" in kinds:
            found += [("R", x, y) for x in ops.reverses_into(score, cpdag, y, None)]
    return found


def test_every_search_stops_at_a_local_optimum_and_never_below_its_start() -> None:
    for score, truth in _problems(30):
        assert _improving(score, ges(score, backward=False), "I") == []
        assert _improving(score, ges(score), "D") == []
        xges0, full = xges(score, extended=False), xges(score)
        assert _improving(score, xges0, "IDR") == []
        assert _improving(score, full, "IDR") == []
        assert _value(score, full) >= _value(score, xges0) - 1e-6
        for extended in (False, True):
            from_truth = xges(score, truth, extended=extended)
            assert _value(score, from_truth) >= _value(score, truth) - 1e-6
        assert _value(score, ges(score, truth)) >= _value(score, truth) - 1e-6


def test_every_search_keeps_to_knowledge_the_parent_cap_and_the_screen() -> None:
    rng = np.random.default_rng(2)
    for score, truth in _problems(20):
        n = truth.num_nodes
        pairs = [frozenset((a, b)) for a in range(n) for b in range(a + 1, n)]
        required, forbidden = (pairs[i] for i in rng.choice(len(pairs), 2, replace=False))
        knowledge = Knowledge(frozenset({required}), frozenset({forbidden}))
        for extended in (None, False, True):
            options = dict(knowledge=knowledge, max_parents=2, one_edge_faithfulness=True)
            if extended is None:
                cpdag = ges(score, **options)  # type: ignore[arg-type]
            else:
                cpdag = xges(score, extended=extended, **options)  # type: ignore[arg-type]
            dag = cpdag.to_dag()
            skeleton = {frozenset(e) for e in dag.edges()}
            assert required in skeleton and forbidden not in skeleton
            assert max(len(dag.parents(v)) for v in range(n)) <= 2
            for u, v in (tuple(e) for e in skeleton - {required}):
                alone = max(
                    score.local_score(v, {u}) - score.local_score(v, set()),
                    score.local_score(u, {v}) - score.local_score(u, set()),
                )
                assert alone > 0, (extended, u, v)


def _subsets(items: list[int], cap: int | None) -> list[frozenset[int]]:
    largest = len(items) if cap is None else min(cap, len(items))
    return [frozenset(c) for k in range(largest + 1) for c in combinations(items, k)]


def _reached(pdag: PDAG, y: int, block: frozenset[int], skip: tuple[int, int]) -> set[int]:
    """Nodes a semi-directed path from `y` reaches without passing `block` or using `skip`."""
    reached, stack = {y}, [y]
    while stack:
        v = stack.pop()
        for w in (pdag.children(v) | pdag.undirected_neighbors(v)) - reached:
            if (v, w) == skip:
                continue
            reached.add(w)
            if w not in block:
                stack.append(w)
    return reached


def _is_clique(pdag: PDAG, nodes: frozenset[int]) -> bool:
    return all(b in pdag.adjacent(a) for a in nodes for b in nodes if a < b)


def _keep(best: dict[int, ops.Candidate], x: int, delta: float, scale: float, subset) -> None:  # type: ignore[no-untyped-def]
    if delta > 1e-10 * max(scale, 1.0) and (x not in best or delta > best[x][0]):
        best[x] = (delta, subset)


def _by_definition(
    score: FamilyScore, pdag: PDAG, y: int, kind: str, cap: int | None
) -> dict[int, ops.Candidate]:
    """The best improving operator of `kind` into `y` per `x`, checked one at a time."""
    best: dict[int, ops.Candidate] = {}
    s = score.local_score
    ne, pa = pdag.undirected_neighbors(y), pdag.parents(y)
    if kind == "insert":
        for x in sorted(set(range(pdag.num_nodes)) - pdag.adjacent(y) - {y}):
            na = ne & pdag.adjacent(x)
            for t in _subsets(sorted(ne - na), cap):
                cond = na | t
                if _is_clique(pdag, cond) and x not in _reached(pdag, y, cond, (-1, -1)):
                    a, b = s(y, pa | cond | {x}), s(y, pa | cond)
                    _keep(best, x, a - b, max(abs(a), abs(b)), t)
    elif kind == "delete":
        for x in sorted(pa | ne):
            na = ne & pdag.adjacent(x)
            for h in _subsets(sorted(na), None):
                if _is_clique(pdag, na - h):
                    base = (pa - {x}) | (na - h)
                    a, b = s(y, base), s(y, base | {x})
                    _keep(best, x, a - b, max(abs(a), abs(b)), h)
    else:
        for x in sorted(pdag.children(y)):
            na, pa_x = ne & pdag.adjacent(x), pdag.parents(x)
            for t in _subsets(sorted(ne - pdag.adjacent(x)), cap):
                cond = na | t
                block = cond | pdag.undirected_neighbors(x)
                if _is_clique(pdag, cond) and x not in _reached(pdag, y, block, (y, x)):
                    values = (s(y, pa | cond | {x}), s(y, pa | cond), s(x, pa_x - {y}), s(x, pa_x))
                    delta = values[0] - values[1] + values[2] - values[3]
                    _keep(best, x, delta, max(map(abs, values)), t)
    return best


def _partial_cpdags() -> list[tuple[FamilyScore, PDAG]]:
    """Graphs halfway to the truth, with some edges turned: undirected edges and wrong arrows."""
    rng = np.random.default_rng(3)
    bundle = load_alarm()
    cases = [(BDeuScore(bundle.data), bundle.truth)] + _problems(20)
    graphs = []
    for score, truth in cases:
        for _ in range(2):
            dag = DAG(truth.num_nodes)
            for u, v in sorted(truth.edges()):
                if rng.random() < 0.7:
                    a, b = (v, u) if rng.random() < 0.3 else (u, v)
                    if not dag.has_path(b, a):
                        dag.add_edge(a, b)
            graphs.append((score, dag_to_cpdag(dag)))
    return graphs


def test_the_operators_are_what_their_definitions_give() -> None:
    finders = {
        "insert": lambda sc, g, y, cap: ops.inserts_into(sc, g, y, cap),
        "delete": lambda sc, g, y, cap: ops.deletes_into(sc, g, y),
        "reverse": lambda sc, g, y, cap: ops.reverses_into(sc, g, y, cap),
    }
    for score, cpdag in _partial_cpdags():
        for y in range(cpdag.num_nodes):
            for kind, find in finders.items():
                for cap in (1, None):
                    found = find(score, cpdag, y, cap)
                    expected = _by_definition(score, cpdag, y, kind, cap)
                    assert found.keys() == expected.keys(), (kind, y, cap)
                    for x, (delta, subset) in found.items():
                        assert delta == pytest.approx(expected[x][0], rel=1e-9)
                        assert subset == expected[x][1], (kind, x, y, cap)


def test_every_operator_offered_changes_the_score_by_what_it_claims() -> None:
    """Applied, each operator gives a class whose score differs by exactly its improvement."""
    appliers = {
        "insert": ops.apply_insert,
        "delete": ops.apply_delete,
        "reverse": ops.apply_reverse,
    }
    offered = dict.fromkeys(appliers, 0)
    for score, cpdag in _partial_cpdags():
        before = _value(score, cpdag)
        for y in range(cpdag.num_nodes):
            for kind, apply in appliers.items():
                for x, (delta, subset) in _by_definition(score, cpdag, y, kind, None).items():
                    after = cpdag.copy()
                    apply(after, x, y, subset)
                    assert _value(score, after) - before == pytest.approx(delta, rel=1e-6, abs=1e-6)
                    offered[kind] += 1
    assert min(offered.values()) > 0
