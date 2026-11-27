# Algorithms and options

What each estimator does, what its options change, and how it is made fast. The measurements behind the defaults are in [Benchmarks](benchmarks.md).

GES and XGES search the space of Markov equivalence classes and return a CPDAG. The other four search over variable orders or over DAGs, return one DAG, and score categorical data with BDeu only; `cpdag_` is then the class of that DAG.

## Scores and structure priors

All the searches maximise a decomposable score, a sum of one term per node that depends on the node and its parents (a family). On categorical data the default is BDeu, the log marginal likelihood of the data under a Dirichlet prior of equivalent sample size `ess` spread uniformly over the cells of each family. `BICScore` is the penalised log likelihood, and `GaussianBICScore` the BIC of a linear Gaussian model, computed from the covariance matrix.

A structure prior is added to each family. The default Bernoulli prior makes each of the `n_vars - 1` other variables a parent with probability `expected_parents / (n_vars - 1)`, so a family with `k` parents gains `k log(p / (1 - p))`. `prior="tetrad"` is the prior of Tetrad 7.6.5's BDeu, `k log p + (m - k) log(1 - p)` with `m = n_rows - 1`, which counts rows instead of variables and keeps the constant, so `score_` is the value Tetrad reports for the same network. With `expected_parents=None` the prior is uniform and the score is plain BDeu or BIC.

```python
from causalbase.scores import BDeuScore

score = BDeuScore(data, ess=1.0, expected_parents=1.0, prior="tetrad")
score.local_score(3, (0, 5))  # node 3 with parents 0 and 5
score.score_dag(dag)
```

The score objects cache the families they compute, so one object should serve a whole search; they are not safe to share between threads.

## GES

Greedy Equivalence Search (Chickering, 2002) moves between Markov equivalence classes. A forward phase applies the Insert that most improves the score until none does; a backward phase does the same with Delete. The result is a CPDAG: directed edges are compelled, undirected ones are those the data cannot orient.

```python
GES(
    criterion="bdeu",
    ess=1.0,
    expected_parents=1.0,
    prior="bernoulli",
    penalty_discount=1.0,
    max_parents=None,
    max_t_size=None,
    required=None,
    forbidden=None,
    one_edge_faithfulness=False,
)
```

| Option | Default | What it does |
|---|---|---|
| `criterion` | `"bdeu"` | The score. `"bdeu"` and `"bic"` take categorical data; `"gaussian_bic"` takes continuous data and scores a linear Gaussian model from its covariance matrix. |
| `ess` | `1.0` | Equivalent sample size of the BDeu score. Larger values favour denser graphs. |
| `expected_parents` | `1.0` | Structure prior, added to any criterion: each of the `m = n_vars - 1` possible parents of a node is present with probability `expected_parents / m`, capped at one half. `None` is the uniform prior, plain BDeu or plain BIC. |
| `prior` | `"bernoulli"` | Which structure prior. `"tetrad"`, for BDeu only, is Tetrad 7.6.5's, `k log p + (m - k) log(1 - p)` with `p = expected_parents / m` and `m = n_rows - 1`: the same Bernoulli prior with the number of rows in place of the number of variables, so `score_` is Tetrad's value. GES searches it through the Bernoulli BDeu with that `p`, which ranks the families of a node the same way. |
| `penalty_discount` | `1.0` | Multiplier of the BIC penalty, `log(n) / 2` per free parameter. |
| `max_parents` | `None` | Most parents an Insert may give a node. |
| `max_t_size` | `None` | Largest set of neighbours an Insert orients into its head. `None` tries every set that can be valid, as Chickering defines the operator. |
| `one_edge_faithfulness` | `False` | Join two variables only if the score finds them dependent on their own, as fGES does (Ramsey et al., 2017). |

**The structure prior.** Over twelve networks from 5 to 223 variables, three samples each, one expected parent lowers the summed SHD against the true networks from 659 to 523. The gain grows with the network: `win95pts` drops from 80 to 40, `andes` from 104 to 60. Below about 30 variables the two priors give the same structures.

**The subsets of an Insert.** An Insert orients into its head a set `T` of the head's undirected neighbours, and it is valid only if `T`, together with the neighbours the new parent already shares with the head, forms a clique. The search therefore grows `T` one node at a time and drops any set that stops being a clique, together with all its supersets, so the uncapped scan, which is the operator as Chickering defines it, costs about what a cap of two costs on every benchmark network and learns the same structures. A cap still bounds the work when the neighbours of a node form a large clique.

**One-edge faithfulness.** fGES speeds GES up by never joining two variables that the score finds independent on their own, a check made once, before the search. On continuous data from sparse graphs this pays: over 1000 variables with one edge per variable on average it is five times faster and closer to the truth. On the discrete benchmark networks it saves no time and loses edges whose dependence only shows given other parents (`link` goes from an SHD of 679 to 816), so it is off by default.

`fit(X, initial_graph=dag)` starts the search from a DAG instead of the empty graph.

**Model selection.** `score(X)` is the mean log likelihood per row of held-out data under the learned DAG, its parameters learned from the training rows: the Bayesian predictive likelihood with a BDeu prior of equivalent sample size `ess` on categorical data, which is the same for every DAG of the equivalence class, and the maximum-likelihood linear Gaussian model on continuous data. scikit-learn's parameter searches use it, so `GridSearchCV(GES(), {"expected_parents": [0.5, 1, 2]}).fit(frame)` picks the prior on held-out rows. `causalbase.scores.log_likelihood(dag, train, test)` is the same number for any DAG.

**Background knowledge.** `GES(required=[("smoking", "cancer")], forbidden=[("age", "sex")])` keeps the first pair adjacent and never joins the second; pairs are given by name or column index. Knowledge fixes adjacency, not direction: a search over equivalence classes decides directions from the data. The functional form, `causalbase.search.ges(score, initial_graph=None, ...)`, also takes `candidates`, a set of ordered pairs to restrict the search to, and runs either phase alone.

### How it is made fast

* **Scoring.** BDeu and BIC are computed by compiled kernels. The families an Insert compares share their parents, so the rows are grouped once by the configuration of those parents, labelled densely in order of first occurrence so that any number of parents is exact, and each candidate parent is then counted without sorting: with bitsets when the table is small, in one pass over the rows otherwise, the candidates in parallel. The score is summed over the distinct cell counts, so `log Gamma` is evaluated once per count value, and a family scores the same, bit for bit, whether it is counted in a batch or alone. Families are cached.
* **Evaluating operators.** Operators are evaluated a child at a time. Every candidate parent with the same neighbours in common with the child shares its candidate subsets, so one traversal of the graph decides the path condition for all of them. The subsets are grown as cliques, so the ones that cannot be valid are never visited, and a restriction to candidate pairs (the screen of `one_edge_faithfulness`, say) removes the other parents before anything is counted.
* **Re-scoring.** After a step only the children whose neighbourhood changed, or one of whose undirected neighbours did, are re-scored. Validity also depends on paths elsewhere in the graph, so the best operator is re-checked before it is applied, and when nothing improves every child is re-scored once more before the phase stops. Each phase therefore ends where no operator of its own kind improves the score: no Insert after the forward phase, no Delete after the backward one. An Insert may improve again once the backward phase has removed edges; XGES, below, is the search that ends where no operator of any kind does.
* **Completion.** After each step the graph is completed to its CPDAG with a compiled Dor-Tarsi extension and Chickering's (1995) edge labelling, both over neighbour and parent lists, and only the pairs whose marks changed are written back.

## XGES

Extremely Greedy Equivalence Search (Nazaret and Blei, 2024) moves through the same space as GES with the same score and the same operators, plus a Reverse, which turns a directed edge round. What changes is the order. GES adds edges until none helps and only then removes them, so an edge added early, when the graph was still empty, can keep out the edges that would have explained the data better, and the backward phase cannot undo that. XGES-0 applies at every step the best Delete that improves the score; if there is none, the best Reverse; and only then the best Insert. XGES then forces out each edge of that result in turn, bars the search from putting it back in the same direction, runs XGES-0 again from there, and keeps the result if it scores higher, until no forced deletion pays.

```python
XGES(
    criterion="bdeu",
    ess=1.0,
    expected_parents=1.0,
    prior="bernoulli",
    penalty_discount=1.0,
    max_parents=None,
    max_t_size=None,
    required=None,
    forbidden=None,
    one_edge_faithfulness=False,
    extended=True,
)
```

The options are those of `GES`; `extended=False` runs XGES-0 alone. Both keep GES's guarantee: with a consistent score and enough data, they return the true equivalence class.

XGES-0 takes up to two thirds more time than GES on the discrete networks and up to twice as long on continuous data, since every step re-scores the Deletes and Reverses of the children it touches as well as the Inserts; XGES takes from a few times to some forty times GES's time on the discrete networks, and more on dense continuous graphs. On continuous data from random graphs both are much closer to the truth than GES, and XGES recovers the true class almost exactly up to three edges per variable. On the discrete networks XGES-0 gains on `alarm`, `win95pts` and `munin` and stays within a few edges of GES elsewhere, and XGES gains most on `insurance` and `win95pts` but loses on `pathfinder`, where a higher BDeu score goes with a structure further from the true one. The figures are in [Benchmarks](benchmarks.md).

A forced deletion starts from what it changed, since the rest of the graph is already at XGES-0's optimum, and an Insert or Reverse can only become valid where the deletion cut a path, so each trial re-scores the children from which a changed node can be reached instead of every child. An accepted trial is followed by a full XGES-0 pass. As in the authors' code, a bar stays once its deletion is accepted (their paper lifts it after the trial); because a bar can then hold back an improving operator, the search ends with XGES-0 free of bars and starts over if that improves, so the result is a local optimum for Delete, Reverse and Insert alike.

## OrderSearch

A variable order fixes, for every node, the set of variables that may be its parents: those before it. Given an order, the nodes no longer constrain each other, so the best network consistent with it is found node by node, and the search moves through orders instead of through graphs, where every move must keep the graph acyclic. `OrderSearch` is an iterated local search over orders whose local searches end with GES or XGES.

```python
OrderSearch(
    ess=1.0,
    expected_parents=1.0,
    prior="bernoulli",
    n_restarts=2,
    polish="ges",
    n_pair_candidates=0,
    init="ges",
    time_limit=60.0,
    max_iter=None,
    random_state=None,
    n_jobs=1,
)
```

**Scoring an order.** For each node, `OrderScorer` climbs greedily among the node's predecessors, adding or removing the parent of largest gain while the gain is positive, as in MCTS-BN. The climb is deterministic, so its result depends only on the predecessor set, and when that set changes the scorer replays the stored path instead of climbing again: a candidate that was never chosen can be removed without changing anything, and a new one changes the path only if it beats a step that was taken. The answer is the one a climb from scratch gives. With `n_restarts`, each node climbs again without the first parent the previous climb chose, and keeps the best family.

**The local search.** A move takes one variable and reinserts it at the position of the order that scores best, the neighbourhood of OBS and BOSS. All positions are scored in one sweep in which the variable jumps over its neighbours one at a time; a jump changes only the two nodes involved, and most jumps provably change neither, so the scorer is asked only about the rest. The search moves until no reinsertion improves the order. Then GES (or XGES, with `polish="xges"`) starts from the order's network and adds the arcs the order rules out, which several coordinated reinsertions would otherwise be needed to reach.

**The iterated search.** It starts from a topological order of the network GES (or XGES, `init="xges"`) learns on the same score; that first network is learned before `time_limit` starts counting. Each iteration kicks the best network found so far, half the time to a random topological order of it and otherwise to that order with `ceil(ln p)` random transpositions, runs a local search from there, and moves to the result if it scores at least as well, so plateaus are crossed. `max_iter` counts kicks; at least one of `max_iter` and `time_limit` must be set. With an integer `random_state` and `max_iter` alone the result is reproducible.

**Parents that only help together.** A greedy climb adds one parent at a time, so it never finds a family whose parents have no effect on their own, such as a noisy XOR. With `n_pair_candidates=k`, each local search also tries, for every node where no single parent helps, the best pair among the `k` candidates of largest single gain (`causalbase.order.pair_polish`), keeping the graph acyclic.

## FLOP

FLOP (Wienöbst, Henckel and Weichwald, 2026) is another search over orders, here on BDeu instead of the linear Gaussian score of the original. It differs from `OrderSearch` in the scorer and the moves. For each node it runs a non-greedy grow-shrink, warm-started from the node's previous parents: candidates are visited in random order and any that improves is added, then any parent whose removal improves is removed. Its local search reinserts every variable at its best position, visiting them in their current order, and its iterated search kicks the best order with `round(ln p)` random swaps, rebuilds the parent sets from scratch, and moves only to a strictly better order. It does not end its local searches with GES.

```python
FLOP(ess=1.0, expected_parents=1.0, prior="bernoulli", init="ges",
     max_iter=None, time_limit=60.0, random_state=None, n_jobs=1)
```

The implementation follows `flopsearch` 0.3.0. `n_iter_` counts kicks, and the function behind it, `causalbase.order.flop`, also returns the best order and the parent sets in `info`.

## HillClimbing

The classic score-based search over DAGs, as bnlearn's `hc` and `tabu`. From the empty graph, or from `initial_graph` in `causalbase.search.hill_climbing`, it applies the valid addition, deletion or reversal of an arc with the largest gain until none improves. Ties go to the move with the smallest index, so the search is deterministic. A move changes at most two families, so only their gains are scored again.

```python
HillClimbing(ess=1.0, expected_parents=1.0, prior="bernoulli", tabu_length=0,
             max_tabu_moves=0, n_restarts=0, perturbation=1, time_limit=None,
             random_state=None, n_jobs=1)
```

| Option | Default | What it does |
|---|---|---|
| `tabu_length` | `0` | With a positive value, tabu search: the best move whose DAG is not among the last `tabu_length` visited is applied even if it loses score. |
| `max_tabu_moves` | `0` | Tabu search stops after this many moves without improving its best DAG. |
| `n_restarts` | `0` | Climbs after the first, each from the best DAG so far perturbed by `perturbation` random moves. |
| `time_limit` | `None` | Seconds for the whole run. |

A hill climb over DAGs stops at the first network no single arc change improves, which on the bundled networks is often well below what GES reaches; tabu search and restarts close part of the gap (see [Benchmarks](benchmarks.md)).

## ExactSearch

The network of highest score, found by dynamic programming over subsets of variables (Silander and Myllymäki, 2006). For every node the score of every possible parent set is computed, and a set is kept only if it beats all its subsets. The best network over a set `W` of variables is then the best, over the variable `v` of `W` that comes last, of the best network over `W - v` plus the best parent set of `v` inside `W - v`. Most of the time goes into scoring the `2^(p-1)` parent sets of each node, and the search skips the supersets that a bound on BDeu proves cannot be useful.

```python
ExactSearch(ess=1.0, expected_parents=1.0, prior="bernoulli",
            max_parents=None, time_limit=None, n_jobs=1)
```

Time and memory grow as `p 2^p`. The 20 variables of `child` take about eight minutes on one thread, most of it scoring parent sets, which `n_jobs` spreads over threads; 27 variables need about 6 GB of memory. The data must have fewer than 32768 rows and at most 255 levels per variable.

With `max_parents`, the search is an integer program in the style of GOBNILP (Cussens, 2011; Bartlett and Cussens, 2017), solved with SCIP through PySCIPOpt (`pip install "causalbase[exact]"`): one binary variable per candidate parent set of at most `max_parents` variables, exactly one chosen per node, and acyclicity imposed by cluster constraints, added when a solution violates them. SCIP starts from the best network consistent with the order GES learns. The result is optimal among the DAGs whose families have at most `max_parents` parents, and the global optimum only if no larger family would have been a candidate, which is not checked. Parent sets are 64-bit masks, so this search takes at most 64 variables. With `time_limit` it returns the best network SCIP found in time, and the function behind it, `causalbase.exact.exact_search`, says in `info["status"]` whether it was proved optimal.
