# causalbase

[Source](https://github.com/ptorrijos99/causalbase) | [Issues](https://github.com/ptorrijos99/causalbase/issues) | [Changelog](https://github.com/ptorrijos99/causalbase/blob/main/CHANGELOG.md) | [Citing](citing.md)

causalbase learns the structure of Bayesian networks from data. It has seven estimators, all following scikit-learn's conventions:

| Estimator | Data | Returns |
|---|---|---|
| {class}`~causalbase.search.GES` | Discrete (BDeu, BIC) or continuous (Gaussian BIC, BGe) | The equivalence class, by greedy equivalence search |
| {class}`~causalbase.search.XGES` | The same | The equivalence class, by a search that also deletes and reverses edges |
| {class}`~causalbase.order.OrderSearch` | The same | A DAG, by iterated local search over variable orders |
| {class}`~causalbase.order.FLOP` | The same | A DAG, by FLOP's search over orders |
| {class}`~causalbase.search.HillClimbing` | The same | A DAG, by hill climbing or tabu search |
| {class}`~causalbase.exact.ExactSearch` | The same | The DAG of highest score, or the best one under a parent limit |
| {class}`~causalbase.lingam.RCD` | Continuous, linear with non-Gaussian noise | A causal order and its coefficients, from higher-order cumulants |

```python
from causalbase import GES
from causalbase.datasets import load_alarm

model = GES().fit(load_alarm().data)
model.adjacency_
```

Learning across sites that cannot share their data is the job of [fedcausal](https://github.com/ptorrijos99/fedcausal), which is built on causalbase.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: quickstart
:link-type: doc

Installation and a first structure, measured against the true network.
:::

:::{grid-item-card} Algorithms
:link: algorithms
:link-type: doc

What each search does and what its options change.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

Parameters, attributes and methods of every class and function.
:::

:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc

Scripts that run on the bundled networks.
:::
::::

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: User guide

bring-your-own-data
algorithms
auto_examples/index
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api
benchmarks
validation
citing
Changelog <https://github.com/ptorrijos99/causalbase/blob/main/CHANGELOG.md>
Contributing <https://github.com/ptorrijos99/causalbase/blob/main/CONTRIBUTING.md>
```
