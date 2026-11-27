# Install

causalbase needs Python 3.10 or later. pip installs NumPy, pandas and Numba with it, which is all the estimators and the bundled networks need.

## From PyPI

```bash
pip install causalbase
```

The integer program that `ExactSearch` runs under a parent limit needs SCIP, through PySCIPOpt, which the `exact` extra installs:

```bash
pip install "causalbase[exact]"
```

scikit-learn is not a dependency. The estimators follow its conventions on their own, so `clone`, `GridSearchCV` and pipelines work once it is installed.

## From GitHub

The development version, from the `main` branch:

```bash
pip install git+https://github.com/ptorrijos99/causalbase
```

## From source

To work on the code, clone the repository and install it in editable mode with the tools of the tests, the linter and the type checker:

```bash
git clone https://github.com/ptorrijos99/causalbase
cd causalbase
pip install -e ".[dev]"
pytest
```

`pip install -e ".[docs]"` adds what the documentation needs, and `sphinx-build -b html docs docs/_build/html` builds it.

## First use

The scores, the graph routines and the searches run in Numba kernels, compiled the first time each is used and cached on disk next to the package. The first fit in a new environment, or after an upgrade, therefore takes a few seconds longer than the next ones, and later sessions load the compiled code instead of compiling it again.

## Check it worked

```python
from causalbase import GES
from causalbase.datasets import load_asia

print(GES().fit(load_asia().data).adjacency_)
```
