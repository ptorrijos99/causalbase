# causalbase

[![CI](https://github.com/ptorrijos99/causalbase/actions/workflows/ci.yml/badge.svg)](https://github.com/ptorrijos99/causalbase/actions/workflows/ci.yml)
[![Docs](https://github.com/ptorrijos99/causalbase/actions/workflows/docs.yml/badge.svg)](https://ptorrijos99.github.io/causalbase/)
[![PyPI](https://img.shields.io/pypi/v/causalbase.svg)](https://pypi.org/project/causalbase/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ptorrijos99/causalbase/blob/main/LICENSE)

Bayesian network structure learning and causal discovery in Python. causalbase learns, from a table of observations, the network of dependencies between its variables: the Markov equivalence class that greedy equivalence search returns, or a single DAG from a search over orders, over DAGs or an exact one. The scores and the graph routines run in compiled Numba code, so GES learns a network of a thousand variables in seconds.

| Estimator | Data | Search |
|---|---|---|
| `GES` | Discrete (BDeu, BIC) or continuous (Gaussian BIC) | Greedy equivalence search (Chickering, 2002) |
| `XGES` | The same | Extremely greedy equivalence search (Nazaret and Blei, 2024), which also deletes and reverses edges as it goes |
| `OrderSearch` | Discrete (BDeu) | Iterated local search over variable orders, each local search ended by GES or XGES |
| `FLOP` | Discrete (BDeu) | FLOP's search over orders (Wienöbst, Henckel and Weichwald, 2026) |
| `HillClimbing` | Discrete (BDeu) | Hill climbing or tabu search over DAGs, with restarts |
| `ExactSearch` | Discrete (BDeu) | The network of highest score, by dynamic programming, or the best one under a parent limit, by integer programming |

Every estimator follows scikit-learn's conventions (hyperparameters in the constructor, `fit` to learn, results ending in an underscore) without depending on it, so `clone`, `GridSearchCV` and pipelines work, and `score(X)` is the log likelihood of held-out rows. BDeu comes with two structure priors, a Bernoulli prior on each possible parent and Tetrad 7.6.5's, whose scores it reproduces.

## Installation

```bash
pip install causalbase
```

causalbase needs Python 3.10 or later, NumPy, pandas and Numba. The integer program of `ExactSearch` under a parent limit also needs SCIP, through PySCIPOpt:

```bash
pip install "causalbase[exact]"
```

The kernels are compiled the first time they are used and cached on disk, so the first fit in a new environment takes a few seconds longer than the next ones.

## Quick start

```python
from causalbase import GES
from causalbase.datasets import load_alarm
from causalbase.metrics import shd

bundle = load_alarm()  # 37 variables, 5000 rows, bundled with the package

model = GES().fit(bundle.data)  # also takes a DataFrame or an array
model.adjacency_  # the learned CPDAG, with the variable names
shd(bundle.truth, model.dag_)  # structural Hamming distance to the true network
```

Every distinct value of a column is a level, so a numeric column is read as categories. Continuous data takes `GES(criterion="gaussian_bic")`.

The searches over orders and over DAGs return one network:

```python
from causalbase import OrderSearch

model = OrderSearch(polish="xges", time_limit=60, random_state=0).fit(bundle.data)
model.dag_, model.score_
```

Behind each estimator on BDeu there is a function that takes the score and returns a `SearchResult` (`dag`, `score`, `n_iter`, `trace`, `info`), for those who want to choose the starting point, the limits or the score object themselves:

```python
from causalbase.order import order_search
from causalbase.scores import BDeuScore

score = BDeuScore(bundle.data, prior="tetrad")
result = order_search(score, polish="xges", max_iter=20, random_state=0)
result.dag, result.score, result.info["accepted"]
```

`causalbase.order` also has the pieces the order searches are built from (`OrderScorer`, `InsertionSearch`, `IteratedLocalSearch`, `pair_polish`), and `causalbase.exact.exact_search`, `causalbase.order.flop` and `causalbase.search.hill_climbing` are the functions behind the other estimators.

## Speed

GES counts in compiled code, checks paths and completes the CPDAG over neighbour lists, and re-scores after each step only the part of the graph the step changed. On a laptop it learns `andes` (223 variables, 5000 rows) in under half a second and `munin` (1041 variables) in about 4 s; causal-learn's GES takes 53 s on `alarm` and does not finish `andes` in half an hour. The [benchmarks](https://ptorrijos99.github.io/causalbase/benchmarks.html) have the full tables, for every estimator.

## Documentation

- [Installation and quick start](https://ptorrijos99.github.io/causalbase/quickstart.html)
- [Algorithms](https://ptorrijos99.github.io/causalbase/algorithms.html): what each search does and what its options change.
- [API reference](https://ptorrijos99.github.io/causalbase/api.html)
- [Examples](https://ptorrijos99.github.io/causalbase/auto_examples/index.html), which run on the bundled networks.
- [Validation](https://ptorrijos99.github.io/causalbase/validation.html): what the test suite checks.

Learning from data spread across sites that cannot share it is the job of [fedcausal](https://github.com/ptorrijos99/fedcausal), which is built on causalbase.

## Development

```bash
git clone https://github.com/ptorrijos99/causalbase
cd causalbase
pip install -e ".[dev]"
pytest
```

[CONTRIBUTING.md](https://github.com/ptorrijos99/causalbase/blob/main/CONTRIBUTING.md) has the checks a pull request must pass and how the tests are written; the documentation builds with `pip install -e ".[docs]"` and `sphinx-build -b html docs docs/_build/html`.

## Citation

If you use causalbase in a scientific publication, please cite:

```bibtex
@software{causalbase,
  author = {Torrijos, Pablo and G{\'a}mez, Jos{\'e} A. and Puerta, Jos{\'e} M.},
  title = {causalbase: Bayesian network structure learning and causal discovery},
  year = {2026},
  url = {https://github.com/ptorrijos99/causalbase}
}
```

The same information is in [`CITATION.cff`](https://github.com/ptorrijos99/causalbase/blob/main/CITATION.cff), which GitHub shows as "Cite this repository".

## References

- Chickering, D. M. (1995). A transformational characterization of equivalent Bayesian network structures. *UAI '95*, 87-98.
- Chickering, D. M. (2002). Optimal structure identification with greedy search. *Journal of Machine Learning Research*, 3, 507-554.
- Silander, T., & Myllymäki, P. (2006). A simple approach for finding the globally optimal Bayesian network structure. *UAI '06*.
- Cussens, J. (2011). Bayesian network learning with cutting planes. *UAI '11*.
- Bartlett, M., & Cussens, J. (2017). Integer linear programming for the Bayesian network structure learning problem. *Artificial Intelligence*, 244, 258-271.
- Ramsey, J., Glymour, M., Sanchez-Romero, R., & Glymour, C. (2017). A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models. *International Journal of Data Science and Analytics*, 3(2), 121-129.
- Nazaret, A., & Blei, D. (2024). Extremely greedy equivalence search. *UAI 2024*, PMLR 244, 2716-2745.
- Wienöbst, M., Henckel, L., & Weichwald, S. (2026). Embracing discrete search: A reasonable approach to causal structure learning. *ICLR 2026*.

## License

MIT. See [LICENSE](https://github.com/ptorrijos99/causalbase/blob/main/LICENSE), and [NOTICE](https://github.com/ptorrijos99/causalbase/blob/main/NOTICE) for the bundled networks.
