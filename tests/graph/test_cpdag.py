"""DAG-to-CPDAG conversion against the definition of Markov equivalence."""

from __future__ import annotations

import itertools

import hypothesis.strategies as st
from hypothesis import given, settings

from causalbase.graph import DAG, PDAG
from causalbase.graph.cpdag import apply_meek_rules, dag_to_cpdag


@st.composite
def _dags(draw: st.DrawFn, max_nodes: int = 6) -> DAG:
    n = draw(st.integers(2, max_nodes))
    order = draw(st.permutations(range(n)))
    edges = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if draw(st.booleans())]
    return DAG.from_edges(n, edges)


def _skeleton(dag: DAG) -> set[frozenset[int]]:
    return {frozenset(e) for e in dag.edges()}


def _v_structures(dag: DAG) -> set[tuple[int, frozenset[int]]]:
    out = set()
    for child in range(dag.num_nodes):
        for a, b in itertools.combinations(sorted(dag.parents(child)), 2):
            if b not in dag.adjacent(a):
                out.add((child, frozenset((a, b))))
    return out


@given(_dags(), _dags())
@settings(max_examples=200, deadline=None)
def test_two_dags_share_a_cpdag_exactly_when_they_are_markov_equivalent(g: DAG, h: DAG) -> None:
    """Verma & Pearl: equivalent iff same skeleton and same v-structures."""
    if g.num_nodes != h.num_nodes:
        return
    equivalent = _skeleton(g) == _skeleton(h) and _v_structures(g) == _v_structures(h)
    assert (dag_to_cpdag(g) == dag_to_cpdag(h)) == equivalent


@given(_dags(max_nodes=8))
@settings(max_examples=200, deadline=None)
def test_a_cpdag_extends_back_to_a_member_of_its_class(dag: DAG) -> None:
    cpdag = dag_to_cpdag(dag)
    member = cpdag.to_dag()
    assert _skeleton(member) == _skeleton(dag)
    assert _v_structures(member) == _v_structures(dag)
    assert dag_to_cpdag(member) == cpdag


def test_compelled_edges_stay_directed() -> None:
    # a -> c <- b is a v-structure, and c -> d is then compelled (Meek's first rule).
    cpdag = dag_to_cpdag(DAG.from_edges(4, [(0, 2), (1, 2), (2, 3)]))
    assert cpdag.directed_edges() == [(0, 2), (1, 2), (2, 3)]
    assert cpdag.undirected_edges() == []


def test_a_chain_is_entirely_reversible() -> None:
    cpdag = dag_to_cpdag(DAG.from_edges(3, [(0, 1), (1, 2)]))
    assert cpdag.directed_edges() == []
    assert cpdag.undirected_edges() == [(0, 1), (1, 2)]


def test_a_pdag_with_no_extension_is_rejected() -> None:
    # a - b - c - d - a: a chordless 4-cycle has no extension without a new collider.
    pdag = PDAG(4)
    for u, v in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        pdag.add_undirected(u, v)
    try:
        pdag.to_dag()
    except ValueError as exc:
        assert "extension" in str(exc)
    else:
        raise AssertionError("expected no consistent extension")


def test_meek_rules_orient_what_a_collider_compels() -> None:
    # a -> b - c with a and c not adjacent: Meek's first rule gives b -> c.
    pdag = PDAG(3)
    pdag.add_directed(0, 1)
    pdag.add_undirected(1, 2)
    apply_meek_rules(pdag)
    assert pdag.directed_edges() == [(0, 1), (1, 2)]


def test_meek_rules_can_be_kept_from_closing_a_cycle() -> None:
    # a -> b - c, c -> d -> b: the first rule would orient b -> c and close
    # b -> c -> d -> b, which only a PDAG with no consistent extension allows.
    # Guarded, the second rule orients the edge the other way instead.
    pdag = PDAG(4)
    pdag.add_directed(0, 1)
    pdag.add_undirected(1, 2)
    pdag.add_directed(2, 3)
    pdag.add_directed(3, 1)
    guarded = pdag.copy()
    apply_meek_rules(guarded, prevent_cycles=True)
    assert guarded.has_directed(2, 1)
    apply_meek_rules(pdag)
    assert pdag.has_directed(1, 2)
