from _samples import load_cell
from causalbase.graph import DAG
from causalbase.metrics.record import graph_record
from causalbase.scores import BDeuScore


def test_record_of_the_true_network_is_at_distance_zero():
    ds, true = load_cell("alarm", 0)
    score = BDeuScore(ds, prior="tetrad")
    rec = graph_record(true, true, score)
    assert rec["smhd"] == rec["shd_dag"] == rec["shd_cpdag"] == rec["shd_skeleton"] == 0
    assert rec["bdeu"] == score.score_dag(true)
    assert rec["n_edges"] == len(rec["edges"])
    assert (
        DAG.from_edges(ds.n_vars, [tuple(e) for e in rec["edges"]]).adjacency().tolist()
        == true.adjacency().tolist()
    )


def test_a_reversal_counts_in_the_dag_but_not_always_in_the_cpdag():
    ds, true = load_cell("alarm", 0)
    edges = [(u, v) for v in range(ds.n_vars) for u in true.parents(v)]
    # A covered edge (same parents apart from each other) can be reversed within the class.
    for u, v in edges:
        if set(true.parents(v)) - {u} == set(true.parents(u)):
            flipped = DAG.from_edges(
                ds.n_vars, [(v, u) if (a, b) == (u, v) else (a, b) for a, b in edges]
            )
            break
    rec = graph_record(true, flipped, BDeuScore(ds, prior="tetrad"))
    assert rec["shd_dag"] == 1
    assert rec["shd_cpdag"] == 0
    assert rec["shd_skeleton"] == 0
