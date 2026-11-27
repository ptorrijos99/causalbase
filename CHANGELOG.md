# Changelog

All notable changes to causalbase are listed here. The project follows [semantic versioning](https://semver.org/); before 1.0.0, a minor release may change the API, and the changelog says how.

## Unreleased

### Memory

- `OrderScorer` used to keep a score for every variable of every parent set it visited, so its memory grew with the length of the search (about 12 GB on munin, with 1041 variables, after 30 minutes). It now keeps, for each parent set, only the toggles that improve on it, which are the only ones a climb, a replay or `breaks` can take, and one answer object per result shared by every predecessor set that reaches it. Searches and their results are unchanged.
- `cache_mb`, in `OrderScorer`, `order_search`, `flop`, `OrderSearch` and `FLOP`: a memory budget for the caches of local scores and visited parent sets, 2048 MiB by default except in `OrderScorer` (unbounded). When a query leaves them above it, every cache drops its oldest quarter. Results do not change, since a dropped entry is computed again when needed; a tight budget only costs time. `OrderScorer.cache_nbytes` is the estimate the budget is checked against, and `flop` reports the scores it dropped in `info["cache_evicted"]`.

## 0.1.0 (2026-11-27)

First release. causalbase is the part of [fedcausal](https://github.com/ptorrijos99/fedcausal) that learns from data held in one place, made a package of its own; fedcausal 1.0.0 keeps the federated methods and depends on it. The history of the code up to the split is in fedcausal's changelog.

### Estimators

- `GES` and `XGES`, greedy equivalence search (Chickering, 2002) and extremely greedy equivalence search (Nazaret and Blei, 2024), on BDeu, BIC or Gaussian BIC, with required and forbidden adjacencies, a parent cap and the one-edge-faithfulness screen of fGES. `XGES(extended=False)` is XGES-0.
- `OrderSearch`: iterated local search over variable orders, scoring each order by the best network consistent with it and ending each local search with GES or XGES started from that network, so that arcs the order rules out can still be added.
- `FLOP`: the search over orders of Wienöbst, Henckel and Weichwald (2026), on BDeu.
- `HillClimbing`: hill climbing or tabu search over DAGs, with random restarts.
- `ExactSearch`: the network of highest score by dynamic programming over subsets (Silander and Myllymäki, 2006), or the best one under a parent limit by integer programming with SCIP (the `exact` extra), which needs at most 64 variables.
- All of them follow scikit-learn's parameter protocol without depending on it, have `dag_`, `cpdag_`, `adjacency_`, `score_`, `n_iter_` and `trace_` after `fit`, and score held-out rows by their log likelihood with `score(X)`.

### Functions and building blocks

- The functions behind the estimators on BDeu, each returning a `SearchResult(dag, score, n_iter, trace, info)`: `causalbase.order.order_search` and `flop`, `causalbase.search.hill_climbing` and `causalbase.exact.exact_search`. They take `random_state`, `max_iter`, `n_restarts`, `time_limit` and `n_jobs` with scikit-learn's meaning, and the order searches start by default from a topological order of the network GES learns on the same score.
- `causalbase.order.OrderScorer` (the best network consistent with an order, by an exact and incremental greedy climb per node), `InsertionSearch`, `IteratedLocalSearch` and `pair_polish` (parents that only help together).
- `ges(score, initial_graph=None, ...)` and `xges(...)` run either phase alone, from any starting graph, restricted to candidate pairs if given.

### Scores

- `BDeuScore(data, *, ess=1.0, expected_parents=1.0, prior="bernoulli", n_jobs=1)`. The default Bernoulli structure prior gives each possible parent the probability `expected_parents / (n_vars - 1)`; `prior="tetrad"` is Tetrad 7.6.5's prior, `k log p + (m - k) log(1 - p)` with `p = expected_parents / m` and `m = n_rows - 1`, with Tetrad's absolute values. Every search takes either prior, and `make_score("bdeu", data, prior=...)` builds both.
- `BICScore` and `GaussianBICScore`, and `log_likelihood(dag, train, test)`, the held-out likelihood behind `score(X)`.
- A compiled engine scores the families of the order searches, hill climbing, FLOP and exact search, with the same values as `BDeuScore`.

### Data, graphs and metrics

- `causalbase.data`: `DiscreteDataset` and `ContinuousDataset`, and one encoding shared by several tables (`as_discrete_sites`).
- `causalbase.graph`: `DAG`, `PDAG`, the CPDAG of a DAG, Meek's rules, moralization, triangulation, maximal cliques and treewidth.
- `causalbase.metrics`: SHD, SMHD, the precision and recall of edges and skeletons, and `graph_record`, everything worth storing about a learned network.
- `causalbase.io`: XMLBIF networks and forward sampling. `causalbase.datasets`: five networks of the Bayesian Network Repository with a sample of 5000 rows each.
- The base exception is `CausalBaseError`; fedcausal's `FedCausalError` is an alias of it.
