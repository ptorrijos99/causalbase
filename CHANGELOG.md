# Changelog

## 0.1.0 (unreleased)

The centralised part of fedcausal 0.3.0, as a package of its own. fedcausal 1.0.0 keeps only the federated methods and depends on this one; the history of everything here up to the split is in fedcausal's changelog.

### Added

- `causalbase.data`, `graph`, `scores`, `search`, `metrics`, `io` and `datasets`, with the API and results they had as `fedcausal.*` 1.0.0.
- Four estimators on the compiled BDeu, all with `fit`, `dag_`, `cpdag_`, `adjacency_`, `score_`, `n_iter_`, `trace_` and a held-out `score(X)`: `OrderSearch` (iterated local search over variable orders polished by GES or XGES), `FLOP` (Wienöbst, Henckel and Weichwald, 2026), `HillClimbing` (hill climbing or tabu search over DAGs, with restarts) and `ExactSearch` (dynamic programming over subsets, or integer programming under a parent limit with the `exact` extra, pyscipopt). They take the score's `ess`, `expected_parents` and `prior` (the Bernoulli prior by default, `"tetrad"` for Tetrad's), as `GES` and `XGES` now do.
- The functions behind them, each returning a `SearchResult(dag, score, n_iter, trace, info)`: `causalbase.order.order_search` and `flop`, `causalbase.search.hill_climbing` and `causalbase.exact.exact_search`. They take scikit-learn's `random_state`, `max_iter`, `n_restarts`, `time_limit` and `n_jobs`; an integer `random_state` seeds exactly as the prototypes' `seed` did, so their results are reproduced bit for bit. `order_search` and `flop` start by default from a topological order of GES's network on the same score.
- `BDeuScore(data, *, ess=1.0, expected_parents=1.0, prior="bernoulli", n_jobs=1)`: `prior="tetrad"` is Tetrad 7.6.5's structure prior, `k log p + (m - k) log(1 - p)` per family with `p = expected_parents / m` and `m = n_rows - 1`, with Tetrad's absolute values (those of MCTS-BN); the default Bernoulli prior is unchanged. Both priors carry `n_jobs` and `n_families` (families scored by the compiled engine behind the order searches, hill climbing, FLOP and exact search, which now take either prior). `ges` and `xges` take a `BDeuScore(prior="tetrad")` directly and search it through the Bernoulli BDeu with the same `p`, whose differences between families are the same. `make_score("bdeu", data, prior=...)` builds either.
- The building blocks of the order searches in `causalbase.order`: `OrderScorer` (the best network consistent with an order, by an exact incremental greedy climb per node, optionally with `n_restarts`), `InsertionSearch`, `IteratedLocalSearch` and `pair_polish` (parents that only help together).
- `causalbase.metrics.graph_record`, everything worth storing about a learned network.

### Changed

- `ges(score, initial_graph=None, ...)` and `xges(...)` take the number of variables from `score.n_vars`, which the `Score` protocol now includes, instead of a second positional argument.
- The base exception is `CausalBaseError` (fedcausal's `FedCausalError` is an alias of it, so existing `except` clauses keep working).
