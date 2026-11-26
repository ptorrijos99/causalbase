"""
Learn a structure from discrete data
====================================

:class:`~causalbase.search.GES` learns the Markov equivalence class of a causal
structure from discrete observations. It takes a DataFrame, an array or a
:class:`~causalbase.data.DiscreteDataset`, and every distinct value of a column
is one level.
"""

from causalbase import GES
from causalbase.datasets import load_asia
from causalbase.metrics import edge_scores, shd

bundle = load_asia()
frame = bundle.data.to_pandas()
frame.head()

# %%
# Fitting follows scikit-learn: hyperparameters go to the constructor, and
# the results end in an underscore. ``cpdag_`` is the equivalence class;
# ``adjacency_`` is the same graph with the variable names.

model = GES().fit(frame)
model.adjacency_

# %%
# A directed edge is one every DAG of the class agrees on. An undirected edge
# is one the data cannot orient.

names = model.feature_names_in_
for u, v in model.cpdag_.directed_edges():
    print(f"{names[u]} -> {names[v]}")
for u, v in model.cpdag_.undirected_edges():
    print(f"{names[u]} -- {names[v]}")

# %%
# ``dag_`` is one member of the class, which is what the metrics compare.

scores = edge_scores(bundle.truth, model.dag_)
print(f"SHD {shd(bundle.truth, model.dag_)}")
print(f"Skeleton precision {scores.precision_skel:.2f}, recall {scores.recall_skel:.2f}")
