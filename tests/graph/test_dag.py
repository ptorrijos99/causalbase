"""DAG: edits keep it acyclic, and malformed input is refused."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalbase.exceptions import CausalBaseError
from causalbase.graph import DAG, CycleError


@given(st.integers(2, 12), st.data())
@settings(max_examples=50, deadline=None)
def test_random_edits_keep_the_graph_acyclic_and_consistent(n: int, data) -> None:
    """Against a plain edge set: every accessor agrees, and a refused edge closed a cycle."""
    dag, edges = DAG(n), set()
    for _ in range(4 * n):
        u, v = data.draw(st.integers(0, n - 1)), data.draw(st.integers(0, n - 1))
        if (u, v) in edges and data.draw(st.booleans()):
            dag.remove_edge(u, v)
            edges.discard((u, v))
            continue
        before = dag.edges()
        try:
            dag.add_edge(u, v)
            edges.add((u, v))
        except CycleError:
            assert u == v or dag.has_path(v, u)
            assert dag.edges() == before
    assert dag.edges() == sorted(edges)
    assert dag.num_edges == len(edges)
    assert DAG.from_amat(dag.adjacency()) == dag
    for v in range(n):
        assert dag.parents(v) == {a for a, b in edges if b == v}
        assert dag.children(v) == {b for a, b in edges if a == v}
    position = {x: i for i, x in enumerate(dag.topological_order())}
    assert all(position[a] < position[b] for a, b in edges)
    clone = dag.copy()
    for edge in clone.edges():
        clone.remove_edge(*edge)
    assert dag.edges() == sorted(edges)


def test_ancestors_and_paths() -> None:
    dag = DAG.from_edges(4, [(0, 1), (1, 2), (3, 2)])
    assert dag.ancestors(2) == {0, 1, 3}
    assert dag.has_path(0, 2)
    assert not dag.has_path(2, 0)
    assert not dag.has_path(0, 0)


def test_malformed_input_is_refused() -> None:
    with pytest.raises(KeyError):
        DAG(3).remove_edge(0, 1)
    with pytest.raises(CycleError):
        DAG(2).add_edge(1, 1)
    with pytest.raises(CausalBaseError):  # CycleError is one, and a ValueError
        DAG.from_edges(2, [(0, 1), (1, 0)])
    with pytest.raises(CycleError):
        DAG.from_amat(np.array([[0, 1], [1, 0]]))
    # A negative index used to wrap round to the last node.
    with pytest.raises(IndexError):
        DAG.from_edges(3, [(-1, 0)])
    with pytest.raises(IndexError):
        DAG(3).add_edge(0, 3)
    with pytest.raises(ValueError):
        DAG(-1)
    with pytest.raises(ValueError, match="square"):
        DAG.from_amat(np.zeros((2, 3)))
