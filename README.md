# causalbase

Causal structure learning from data held in one place.

| Estimator | Data |
|---|---|
| `GES` | Discrete (BDeu, BIC) or continuous (Gaussian BIC) |
| `XGES` | The same, with a search that also removes and reverses edges as it goes |
| `OrderSearch` | Discrete (BDeu, with the Bernoulli or Tetrad's structure prior), by iterated local search over variable orders |
| `FLOP` | The same score, by FLOP's search over variable orders |
| `HillClimbing` | The same score, by hill climbing or tabu search over DAGs |
| `ExactSearch` | The same score: the best network, or the best one under a parent limit |

All follow scikit-learn's conventions: hyperparameters go to the constructor, `fit` learns, and results end in an underscore. `GES` scores categorical data by BDeu or BIC and continuous data by a Gaussian BIC, and its `score(X)`, the held-out log likelihood, lets scikit-learn's parameter searches tune it.

```bash
pip install causalbase
```

```python
from causalbase import GES
from causalbase.datasets import load_alarm

data = load_alarm().data  # 37 variables, 5000 rows, bundled

model = GES().fit(data)  # also takes a DataFrame or an array
model.adjacency_  # the learned CPDAG, with variable names
```

## Search over variable orders

`OrderSearch` searches the space of variable orders under BDeu. An order is scored by the best network consistent with it (an exact, incremental greedy climb per node, restarted without the first parent each climb chose); an iterated local search moves through orders by insertions, and each local search ends with GES or XGES started from the order's network, so that arcs the order rules out can still be added.

```python
from causalbase import OrderSearch
from causalbase.datasets import load_alarm

model = OrderSearch(polish="xges", time_limit=60).fit(load_alarm().data)
model.adjacency_, model.score_
```

Every estimator on this score has a function behind it, which takes the score and returns a `SearchResult` (`dag`, `score`, `n_iter`, `trace`, `info`): `order_search` and `flop` in `causalbase.order`, `hill_climbing` in `causalbase.search`, and `exact_search` in `causalbase.exact`, which runs dynamic programming or, under a parent limit, integer programming (the ILP needs `pip install "causalbase[exact]"`). The score is a `BDeuScore`, which these functions evaluate with a compiled engine. Its structure prior is the Bernoulli prior by default; `BDeuScore(data, prior="tetrad")` gives Tetrad 7.6.5's prior instead (`k log p + (m - k) log(1 - p)` for `k` parents, with `p = expected_parents / m` and `m = n_rows - 1`), with Tetrad's absolute values. Every search, GES and XGES included, takes either score directly, and so does every estimator (`prior="tetrad"`).

```python
from causalbase.datasets import load_alarm
from causalbase.order import order_search
from causalbase.scores import BDeuScore
from causalbase.search import ges

score = BDeuScore(load_alarm().data, prior="tetrad")
start = ges(score).to_dag()  # what order_search starts from by default
result = order_search(score, polish="xges", max_iter=20, random_state=0)
result.dag, result.score, result.info["accepted"]
```

The building blocks of the order searches (`OrderScorer`, `InsertionSearch`, `IteratedLocalSearch`, `pair_polish`) are in `causalbase.order` too, for those who want to combine them differently.

Learning from data spread across sites that cannot share it is the job of [fedcausal](https://github.com/ptorrijos99/fedcausal), which builds on this package.

## Speed

GES runs in compiled code where it matters (counting, path checks, CPDAG completion) and re-scores only the part of the graph each step changes. On a laptop it learns `andes` (223 variables, 5000 rows) in under half a second and `munin` (1041 variables) in about 4 s, where causal-learn does not finish `andes` in half an hour. The [benchmarks](docs/benchmarks.md) have the full ladder.

## Documentation

The user guide, the API reference and a gallery of examples that run with no setup build with Sphinx:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## License

MIT. Pablo Torrijos Arenas, Universidad de Castilla-La Mancha.
