# Changelog

All notable changes to causalbase are listed here. The project follows [semantic versioning](https://semver.org/); before 1.0.0, a minor release may change the API, and the changelog says how.

## 0.2.0 (unreleased)

Every search takes every score, every score runs on compiled kernels, and the six estimators and their functions share one interface, which breaks the API of 0.1.0 as listed below. BGe is new, and RCD and the other cumulant methods for linear non-Gaussian models move here from fedcausal.

### Breaking changes

- `ges` and `xges` return a `SearchResult`, as the other searches do, instead of a `PDAG`: the learned class is `result.cpdag`, one DAG of it `result.dag`. `SearchResult` gained `cpdag` for every search, and `n_iter`, `trace` and `info` are now filled in by GES (Inserts and Deletes applied) and XGES (operators applied).
- Every estimator takes the same score parameters first, in the order `criterion`, `ess`, `expected_parents`, `prior`, `penalty_discount`, `alpha_w`, and `n_jobs` last. `OrderSearch`, `FLOP`, `HillClimbing` and `ExactSearch` gained `criterion`, `penalty_discount` and `alpha_w`; `GES` and `XGES` gained `alpha_w` and `n_jobs`. A positional first argument, which used to be `ess`, is now the criterion, and `OrderSearch` takes `max_iter` before `time_limit`, as `FLOP` does.
- Every estimator's `fit` is `fit(X, y=None, initial_graph=None)`, and every one keeps `dag_`, `cpdag_`, `score_`, `n_iter_`, `trace_` and `adjacency_`. `GES` and `XGES` gained `n_iter_` and `trace_`, and the others gained `initial_graph` (the first order of the order searches, and the first solution of `ExactSearch`'s integer program).
- `n_jobs` belongs to the score. `BICScore`, `GaussianBICScore` and `BGeScore` take it as `BDeuScore` did, and `exact_search` no longer has its own; it uses the score's. `n_jobs=-1` means every thread numba has, everywhere. The discrete scores used to count GES's batches on every thread numba had whatever `n_jobs` said; they now use `n_jobs`, 1 by default, which on the bundled networks is as fast.
- `exact_search` no longer refuses discrete data of 32768 rows or more, or with more than 255 levels: it scores such data through the score's own batches, more slowly.
- The Gaussian BIC drops a parent that the others determine exactly instead of fitting it by least squares, which gives the same residual; its values may differ from 0.1.0's in the last digits.

### Scores

- `BGeScore(data, *, alpha_mu=1.0, alpha_w=None, expected_parents=1.0, n_jobs=1)`, the Bayesian marginal likelihood of a linear Gaussian model under a normal-Wishart prior (Geiger and Heckerman, 1994; Kuipers, Moffa and Heckerman, 2014), with the prior mean at the sample means and `alpha_w` at `n_vars + 2` by default, as in bnlearn. Markov equivalent DAGs score the same. The estimators and `make_score` take it as `criterion="bge"`, with `ess` as `alpha_mu`.
- The Gaussian BIC and BGe run on compiled kernels that factor the parents' block of the covariance (or scale) matrix once per batch of candidates, with a Cholesky decomposition that drops linearly dependent parents. FLOP on the Gaussian BIC over 30 variables runs about ten times faster than in 0.1.0.
- `make_score` takes every parameter of every score, `alpha_w` and `n_jobs` included, and builds the score of every estimator.
- `FamilyScore.local_scores_add(node, parents, candidates)`, the batch the order searches, hill climbing and FLOP call; with it every `FamilyScore`, including one of one's own, works with every search. `FamilyScore` takes `n_jobs`.
- `CONTINUOUS_SCORES` in `causalbase.scores` names the scores that take a `ContinuousDataset`.

### Searches

- Every search function takes its score first and its starting point second (a graph for `ges`, `xges` and `hill_climbing`, an order for `order_search`, `flop` and `exact_search`, whose `initial_order` was keyword-only).
- `order_search`, `flop`, `hill_climbing`, `pair_polish`, `OrderScorer`, `PolishedILS` and `exact_search` take any `FamilyScore`. `PolishedILS.ess` and `PolishedILS.data` are `None` for a score without them.
- `exact_search` walks the parent sets of every score with compiled kernels: the counting kernels of BDeu also score the discrete BIC, and new kernels score the Gaussian BIC and BGe by one elimination step per set, about a tenth of a microsecond each, so 20 Gaussian variables take a second or two. The discrete BIC is pruned with the bound of de Campos and Ji (2011) and the Gaussian BIC with the likelihood of the family of every other variable; BGe is not pruned. A score of one's own is walked in Python.
- The estimators live with their functions: `GES`, `XGES` and `HillClimbing` in `causalbase.search.estimator`, `OrderSearch` and `FLOP` in `causalbase.order.estimator`, `ExactSearch` in `causalbase.exact.estimator`. Their public import paths do not change.

### Memory

- `OrderScorer` used to keep a score for every variable of every parent set it visited, so its memory grew with the length of the search (about 12 GB on munin, with 1041 variables, after 30 minutes). It now keeps, for each parent set, only the toggles that improve on it, which are the only ones a climb, a replay or `breaks` can take, and one answer object per result shared by every predecessor set that reaches it. Searches and their results are unchanged.
- `cache_mb`, in `OrderScorer`, `order_search`, `flop`, `OrderSearch` and `FLOP`: a memory budget for the caches of local scores and visited parent sets, 2048 MiB by default except in `OrderScorer` (unbounded). When a query leaves them above it, every cache drops its oldest quarter. Results do not change, since a dropped entry is computed again when needed; a tight budget only costs time. `OrderScorer.cache_nbytes` is the estimate the budget is checked against, and `flop` reports the scores it dropped in `info["cache_evicted"]`.

### Linear non-Gaussian models

- `causalbase.lingam`, the cumulant methods for linear non-Gaussian models, moved from fedcausal so that the centralised ones live with the other methods that learn from one table. fedcausal 1.0.0 imports them from here and keeps its federated names.
- `RCD`, also importable from `causalbase`: the causal order from the third- or fourth-order cumulants of the data, then the coefficients by adaptive lasso or OLS, with fitted `causal_order_`, `coef_`, `dag_` and `adjacency_`. Its options are those of fedcausal's `FedRCD`; HC-LiNGAM is `RCD(method="sort")` and HC `RCD(method="exact")`, and both reproduce the orders and coefficients of the original implementation.
- The functions behind it, on the statistics rather than the data: `moment_sums` and `cumulants_from_sums`, which merge the sums of any number of blocks of rows exactly; `rcd` on the cumulants, `rcd_exact` on a table and `exact_order` with the deflation left to the caller; the tau discriminators `tau_matrix_3rd` and `tau_matrix_4th`; and `estimate_b_from_cov` and `estimate_b_lasso_from_cov` for the coefficients on a known order.

## 0.1.0 (2026-11-27)

First release. causalbase is the part of [fedcausal](https://github.com/ptorrijos99/fedcausal) that learns from data held in one place, made a package of its own; fedcausal 1.0.0 keeps the federated methods and depends on it.

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
