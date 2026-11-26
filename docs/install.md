# Install

```bash
pip install causalbase
```

Python 3.10 or newer. The install pulls in numpy, pandas and numba, and that is all the estimators and the bundled networks need. The compiled kernels are built on first use and cached next to the package, so the first fit in a fresh environment takes a few seconds longer than the rest.

The development version installs from GitHub:

```bash
pip install git+https://github.com/ptorrijos99/causalbase
```

## Check it worked

```python
from causalbase import GES
from causalbase.datasets import load_asia

print(GES().fit(load_asia().data).adjacency_)
```
