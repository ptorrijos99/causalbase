# causalbase

Causal structure learning from data held in one place.

| Estimator | Data |
|---|---|
| {class}`~causalbase.search.GES` | Discrete or continuous |
| {class}`~causalbase.search.XGES` | Discrete or continuous, with a search that also removes and reverses edges |

```python
from causalbase import GES
from causalbase.datasets import load_alarm

model = GES().fit(load_alarm().data)
model.adjacency_
```

Learning across sites that cannot share their data is the job of `fedcausal`, which builds on this package.

```{toctree}
:maxdepth: 1
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 1
:caption: User guide

bring-your-own-data
algorithms
```

```{toctree}
:maxdepth: 1
:caption: Examples

auto_examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
benchmarks
validation
```

How to contribute: [CONTRIBUTING.md](https://github.com/ptorrijos99/causalbase/blob/main/CONTRIBUTING.md).
