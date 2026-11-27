# Changelog

All notable changes to causalbase are listed here. The project follows [semantic versioning](https://semver.org/); before 1.0.0, a minor release may change the API, and the changelog says how.

## 0.2.0 (unreleased)

Every search now takes every score, and BGe is new.

### Scores

- `BGeScore(data, *, alpha_mu=1.0, alpha_w=None, expected_parents=1.0)`, the Bayesian marginal likelihood of a linear Gaussian model under a normal-Wishart prior (Geiger and Heckerman, 1994; Kuipers, Moffa and Heckerman, 2014), with the prior mean at the sample means and `alpha_w` at `n_vars + 2` by default, as in bnlearn. Markov equivalent DAGs score the same. `make_score("bge", data, ess=...)` builds it with `ess` as `alpha_mu`, and every estimator takes `criterion="bge"`.
- `FamilyScore.local_scores_add(node, parents, candidates)`, the batch method the order searches, hill climbing and FLOP call; it is `local_scores_with` under that name, so that every `FamilyScore`, including one of one's own, works with them.
- `CONTINUOUS_SCORES` in `causalbase.scores` names the scores that take a `ContinuousDataset`.

### Estimators and searches

- `OrderSearch`, `FLOP`, `HillClimbing` and `ExactSearch` take `criterion` (`"bdeu"`, `"bic"`, `"gaussian_bic"` or `"bge"`) as their first parameter, as `GES` and `XGES` do, and `penalty_discount` after `prior`. A positional first argument, which used to be `ess`, is now the criterion. BDeu stays the default and runs on the same compiled engine, with the same results.
- `order_search`, `flop`, `hill_climbing`, `pair_polish`, `OrderScorer`, `PolishedILS` and `exact_search` take any `FamilyScore`. `PolishedILS.ess` and `PolishedILS.data` are `None` for a score without them.
- `exact_search` scores the parent sets of the scores other than BDeu in Python, pruning the discrete BIC with the bound of de Campos and Ji (2011) and the Gaussian BIC with the likelihood of the family of every other variable. BGe is not pruned.
- `GES` and `XGES` take `criterion="bge"`, and `prior="tetrad"` is refused with any criterion other than `"bdeu"` in every estimator.

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
