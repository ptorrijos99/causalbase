# Contributing

Bug reports, fixes and new methods are welcome. Use the [issue tracker](https://github.com/ptorrijos99/causalbase/issues) for bugs and proposals, with a minimal script that reproduces a bug; report security problems privately, as [SECURITY.md](SECURITY.md) says. Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Getting set up

```bash
git clone https://github.com/ptorrijos99/causalbase
cd causalbase
pip install -e ".[dev]"
pytest
```

The suite is self-contained. No downloads, no environment variables, no JVM.

## Before you open a pull request

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

CI runs the same three on Linux and Windows, on Python 3.10 and 3.13, and the suite once more with the oldest dependencies `pyproject.toml` allows and no extras.

## What the tests are for

A test states something a user could observe: a result, an error, a guarantee in the docs. It should fail only when that stops being true. So:

- Assert behaviour through the public API, not private helpers, cache contents, memory layout or the order a loop happens to visit things.
- Compare floating-point results to a tolerance, never bit for bit. The last few bits move between platforms, BLAS builds and libm versions.
- Do not pin tie-breaks or values recorded from one run. If two answers are equally correct, the test must accept both. The one exception is deliberate: `LEARNED` in `tests/search/test_ges.py` records the score and SHD of what GES learns from the bundled samples, so that any change in what they learn is seen. When one fails, decide whether the change is intended, update the table and say so in the changelog.
- Prefer one test that checks a property over many random inputs, or loops over cases, to many tests that each check one example; a parametrisation over dozens of cases is a loop.
- Do not add a strict xfail to record a known difference. Assert the property that makes the difference acceptable.
- Prefer checking against an independent definition (a brute-force score, Verma and Pearl's characterisation of equivalence) over checking against recorded output.

The whole suite runs in under a minute.

## Performance changes

Two things make a performance claim believable here, and wall-clock is neither of them on its own.

Compare timings within one process, back to back, after a warm-up fit so that numba's compilation is not counted, and take the minimum of several runs. A synced folder or a busy machine moves wall-clock by tens of percent, which swamps most optimisations worth making.

A change to the search must keep each phase of GES at a local optimum of its operator, and XGES at one of Insert, Delete and Reverse together; `tests/search/test_ges.py` checks both on random problems.

## Adding a method

A search goes in `causalbase.search` (over graphs or equivalence classes) or `causalbase.order` (over variable orders), a score in `causalbase.scores`. Nothing here may import `fedcausal` or `flwr`, not even under `TYPE_CHECKING`; `tests/test_layout.py` checks it. Federated methods belong in `fedcausal`.

## Style

Ruff handles formatting and import order, so do not argue with it.

Docstrings follow the numpydoc convention. Full parameter blocks on the public API, one line when the signature already says everything. Explain why something is the way it is when that is not obvious; do not restate what the code says.

Development history belongs in commit messages and pull requests, not in source comments. If you need to record why a decision was reversed, the commit is the place.

## Releasing

1. Set `__version__` in `src/causalbase/__init__.py`, and `version` and `date-released` in `CITATION.cff`.
2. Give the release its section in `CHANGELOG.md`, dated.
3. Commit, tag `vX.Y.Z` and push the tag.
4. Publish a GitHub release from the tag. `publish.yml` builds the package, checks that the tag matches `__version__` and uploads it to PyPI by trusted publishing; `docs.yml` publishes the documentation.
5. Archive the release on Zenodo with the metadata of `.zenodo.json`, as a new version of the existing record, so that the concept DOI keeps resolving to the latest release.
