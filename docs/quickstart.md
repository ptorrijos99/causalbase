# Quickstart

## Learn a structure from one table

```python
import pandas as pd

from causalbase import GES

frame = pd.read_csv("survey.csv", dtype=str)
model = GES().fit(frame)
model.adjacency_
```

Every distinct value of a column is one level, so numbers are read as labels too. `fit` learns a CPDAG, the equivalence class of DAGs the data cannot tell apart. In `adjacency_`, `A[i, j] = 1` with `A[j, i] = 0` is a compelled edge `i -> j`, one every DAG of the class shares; both set is an edge whose direction the data does not decide. `cpdag_` is the same graph over column indices, and `dag_` is one DAG of the class.

## Measure it against a known network

```python
from causalbase.datasets import load_asia
from causalbase.metrics import edge_scores, shd

bundle = load_asia()  # data sampled from a known network
model = GES().fit(bundle.data)
shd(bundle.truth, model.dag_)  # structural Hamming distance
edge_scores(bundle.truth, model.dag_).f1_skel
```

`bundle.truth` is already indexed like the data's columns. To compare against a network of your own, load it with `causalbase.io.load_xbif` and re-index it onto the data's column order first.

## Tune like any scikit-learn estimator

```python
model = GES(expected_parents=0.5, ess=10)
model.set_params(max_t_size=2).get_params()
```

`sklearn.base.clone` works on every estimator. `GES.score(X)` is the mean log likelihood of held-out rows, so a parameter search can choose the hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

frame = load_asia().data.to_pandas()
search = GridSearchCV(GES(), {"expected_parents": [0.5, 1.0, 2.0]}, cv=3).fit(frame)
search.best_params_
```

Continuous data takes `GES(criterion="gaussian_bic")`.

## Next

[Algorithms](algorithms.md) explains every option.
