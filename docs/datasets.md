# Benchmark networks

Learning a structure from data sampled from a known network is how a method is measured, since the learned graph can be compared with the true one. causalbase gives such data in two ways.

## Bundled with the package

Five networks of bnlearn's [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/) ship with causalbase, each with one sample of 5000 rows, so that the examples and the tests run without a connection:

```python
from causalbase.datasets import list_networks, load_network

list_networks()  # {'cancer': 5, 'asia': 8, 'child': 20, 'insurance': 27, 'alarm': 37}
bundle = load_network("alarm")
bundle.data, bundle.truth, bundle.network
```

`truth` is the true DAG, already indexed like the columns of `data`, and `network` holds the conditional probability tables, from which `causalbase.io.forward_sample` draws more rows.

## Downloaded from Zenodo

`fetch_network` gives 100 samples of 5000 rows of each of 226 networks: the 24 discrete networks of bnlearn's repository and the 202 of [bnRep](https://github.com/manueleleonelli/bnRep), a collection of networks from the literature.

```python
from causalbase.datasets import fetch_network, list_remote_networks

networks = list_remote_networks()  # name -> number of variables
bundle = fetch_network("pathfinder", sample=3)
bundle = fetch_network("bnrep_earthquake", sample=0)
```

bnRep's names carry the prefix `bnrep_`, because some clash with bnlearn's: `earthquake` is bnlearn's network of 5 variables, `bnrep_earthquake` bnRep's of 40. `list_remote_networks("bnlearn")` and `list_remote_networks("bnRep")` list each repository alone.

The samples are those of two Zenodo records, [10.5281/zenodo.14917795](https://doi.org/10.5281/zenodo.14917795) for bnlearn and [10.5281/zenodo.23561524](https://doi.org/10.5281/zenodo.23561524) for bnRep, and causalbase pins one version of each, so a network always gives the same rows. bnRep's files carry their networks in BIF; bnlearn's networks come from the [BayesFL](https://github.com/ptorrijos99/BayesFL) repository, at a fixed commit. Every download is checked against its md5 or SHA-256 before it is kept.

### The cache

The first call for a network downloads its file and keeps it; later calls, of that sample or of any other, read the cache and need no connection. The cache is the folder `get_data_home()` returns: the `data_home` argument if given, else the `CAUSALBASE_DATA` environment variable, else `causalbase` inside `XDG_CACHE_HOME` (`~/.cache` by default). Deleting the folder, or a file in it, only makes the next call download again.

Zenodo serves whole files, so the first sample of a network brings all 100: from a few hundred kB for the small networks to 200 MB for `munin`. bnRep keeps its 147 networks of up to 20 variables in four files grouped by size (1 to 5 variables, 6 to 10, 11 to 15 and 16 to 20), so the first of them downloads the whole group.

## Networks of your own

`causalbase.io.load_bif` reads a network in BIF, the text format of both repositories, and `causalbase.io.load_xbif` one in XMLBIF; both give the same `DiscreteBN`. See [Bring your own data](bring-your-own-data.md) for tables of your own.
