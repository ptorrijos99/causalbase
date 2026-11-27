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

Continuous data takes `criterion="gaussian_bic"` or `criterion="bge"`, in this estimator and in every other.

## Measure it against a known network

```python
from causalbase.datasets import load_asia
from causalbase.metrics import edge_scores, shd

bundle = load_asia()  # data sampled from a known network
model = GES().fit(bundle.data)
shd(bundle.truth, model.dag_)  # structural Hamming distance
edge_scores(bundle.truth, model.dag_).f1_skel
```

`bundle.truth` is already indexed like the data's columns. To compare against a network of your own, load it with `causalbase.io.load_xbif` and re-index it onto the data's column order first. Since `dag_` is one member of an equivalence class, comparing CPDAGs, `shd(dag_to_cpdag(bundle.truth), model.cpdag_)`, does not depend on which member the search returned.

## Other searches

`XGES` takes the options of `GES` and searches the same space with deletions and reversals as well. The searches over orders, over DAGs and the exact search take categorical data and return one DAG:

```python
from causalbase import ExactSearch, HillClimbing, OrderSearch

OrderSearch(time_limit=30, random_state=0).fit(bundle.data).score_
HillClimbing(tabu_length=100, max_tabu_moves=100).fit(bundle.data).score_
ExactSearch().fit(bundle.data).score_  # the optimum, for small networks
```

`score_` is the score of the learned network (BDeu unless `criterion` says otherwise), with the structure prior, so it compares the searches directly. `OrderSearch` and `FLOP` run until `time_limit` seconds have passed or `max_iter` kicks are done, and `trace_` records when the best score improved.

## Tune like any scikit-learn estimator

```python
model = GES(expected_parents=0.5, ess=10)
model.set_params(max_t_size=2).get_params()
```

`sklearn.base.clone` works on every estimator. `score(X)` is the mean log likelihood of held-out rows, so a parameter search can choose the hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

frame = load_asia().data.to_pandas()
search = GridSearchCV(GES(), {"expected_parents": [0.5, 1.0, 2.0]}, cv=3).fit(frame)
search.best_params_
```

## The functions behind the estimators

Each estimator other than GES and XGES calls a function that takes a score object and returns a `SearchResult`. Calling it directly lets one score serve several searches, and lets a search start where another ended:

```python
from causalbase.order import order_search
from causalbase.scores import BDeuScore
from causalbase.search import hill_climbing, ges

score = BDeuScore(bundle.data, prior="tetrad")
start = ges(score).to_dag()
climbed = hill_climbing(score, start)
result = order_search(score, max_iter=10, random_state=0)
result.dag, result.score, result.n_iter, result.info
```

## Next

[Algorithms](algorithms.md) explains every option, and [Bring your own data](bring-your-own-data.md) what the estimators accept.
