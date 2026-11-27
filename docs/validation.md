# Validation

What the test suite checks, so you can decide how much to trust it.

```bash
pip install -e ".[dev]"
pytest
```

No downloads and no environment variables: everything the tests need ships with the package or with the tests. CI runs the suite on Linux and Windows with Python 3.10 and 3.13, and once more with the oldest dependencies `pyproject.toml` allows.

| Layer | How it is checked |
|---|---|
| Scores | BDeu and BIC against independent count-and-`lgamma` implementations, including parent sets with more configurations than any integer code can hold, and the Gaussian BIC against a regression; a batch of families scores the same, bit for bit, as each family alone. The compiled engine of the order searches gives BDeu's values bit for bit, whatever the counting kernel, the number of threads or the size of its cache, and Tetrad's prior gives the values of the reference implementation used in MCTS-BN |
| Graph kernels | The compiled CPDAG completion, consistent extension and path search against the dense reference implementations they replaced, over random graphs |
| GES and XGES | Learn, on four bundled samples, the score and SHD recorded in the tests, so any change in what they learn shows. On thirty small dense random problems, Gaussian and discretised, the forward phase of GES ends with no improving Insert and the backward one with no improving Delete, XGES-0 and XGES end with no improving Insert, Delete or Reverse, XGES scores at least what XGES-0 does, no search ends below the graph it starts from, and required and forbidden pairs, the parent cap and the one-edge-faithfulness screen all hold. Insert, Delete and Reverse give, for every child, what their definitions give, capped or not, and each one offered changes the score of the class by exactly the improvement it claims |
| Order searches | The order scorer's incremental answers equal a climb from scratch, whatever was asked before, and the score of an order is the sum of its families; the insertion search makes, bit for bit, the moves of a sweep that asks the scorer at every position; the iterated searches never end below their start, are reproducible with a seed and stop at their time limit; `OrderSearch` reports the score of the DAG it returns and beats GES polished alone; the pair polish finds an XOR family that single additions miss |
| Hill climbing | Move for move what a naive search over all arc changes does, ending at a local optimum; tabu search never ends below the plain climb, and restarts keep the best |
| Exact search | Equals a brute force over every DAG of small problems; the integer program agrees with dynamic programming; the pruned candidates are the full enumeration's; the optimum scores at least the true network and GES's; more than 64 variables under a parent limit are refused |
| CPDAG completion | Two random DAGs share a CPDAG exactly when they have the same skeleton and v-structures (Verma & Pearl), and every CPDAG extends back into its own class |
| Networks | The BIF reader gives, for each bundled network written as BIF with its rows shuffled, the network the XMLBIF reader gives, and refuses malformed files with the reason; a fetched network equals the bundled one served the way Zenodo serves it, comes from the cache after the first call, and a corrupt, failed or truncated download leaves nothing behind. A test that downloads from Zenodo runs with `CAUSALBASE_TEST_DOWNLOADS=1` |
| Metrics | SHD, SMHD and the edge scores against their definitions over edge sets |
| Estimators | Hyperparameters round-trip through `get_params`, `set_params` and `sklearn.base.clone`; an empty table or an out-of-range setting is refused with an error that names it; `score(X)` is the mean held-out log likelihood |
| Layout | causalbase never imports fedcausal or Flower, not even for type hints, and a fit loads neither |
| Documentation | Every `from causalbase... import ...` shown in the docs runs, and every example in a docstring prints what it shows |

Search results are compared through scores and equivalence classes rather than edge lists where it matters. A search over equivalence classes returns one member of a Markov equivalence class, and the choice among its members is arbitrary; BDeu and the CPDAG are the same for every member.
