"""
Compare the searches on one network
===================================

Every estimator on BDeu reports ``score_``, the score of the network it
learned under the same prior, so the searches can be compared directly, and
against the network the data were sampled from.
"""

import time

from causalbase import FLOP, GES, XGES, HillClimbing, OrderSearch
from causalbase.datasets import load_insurance
from causalbase.graph import dag_to_cpdag
from causalbase.metrics import shd

bundle = load_insurance()
truth = dag_to_cpdag(bundle.truth)
print(f"{bundle.data.n_vars} variables, {bundle.data.n_rows} rows")

# %%
# The order searches run for ``max_iter`` kicks here, so that the example
# gives the same result every time; in practice ``time_limit`` is the usual
# budget. The SHD is computed between CPDAGs, which does not depend on the
# member of the equivalence class a search returns.

estimators = {
    "GES": GES(),
    "XGES": XGES(),
    "Hill climbing": HillClimbing(),
    "Tabu search": HillClimbing(tabu_length=100, max_tabu_moves=100),
    "OrderSearch": OrderSearch(time_limit=None, max_iter=10, random_state=0),
    "FLOP": FLOP(time_limit=None, max_iter=10, random_state=0),
}
for name, model in estimators.items():
    start = time.perf_counter()
    model.fit(bundle.data)
    seconds = time.perf_counter() - start
    print(
        f"{name:14s} BDeu {model.score_:10.1f}   SHD {shd(truth, model.cpdag_):3d}"
        f"   {seconds:5.2f} s"
    )

# %%
# ``trace_`` records, for the searches that iterate, when the best score
# improved: ``(seconds, iteration, score)``.

for seconds, iteration, score in estimators["OrderSearch"].trace_:
    print(f"{seconds:6.2f} s, kick {iteration:2d}: {score:.1f}")
