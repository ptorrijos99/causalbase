# Benchmarks

Measured on one laptop with the default settings, 5000 rows per network, from the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/). Structures are exact; timings are not, so read them as orders of magnitude.

## How fast, and how close to the truth

| Network | Variables | Seconds | Edges | True edges | SHD | SMHD |
|---|---:|---:|---:|---:|---:|---:|
| cancer | 5 | 0.01 | 3 | 4 | 1 | 2 |
| asia | 8 | 0.02 | 7 | 8 | 2 | 1 |
| child | 20 | 0.08 | 25 | 25 | 2 | 0 |
| insurance | 27 | 0.10 | 44 | 52 | 17 | 19 |
| water | 32 | 0.07 | 31 | 66 | 50 | 88 |
| mildew | 35 | 0.09 | 33 | 46 | 22 | 40 |
| alarm | 37 | 0.10 | 47 | 46 | 10 | 11 |
| hailfinder | 56 | 0.15 | 62 | 66 | 58 | 72 |
| hepar2 | 70 | 0.12 | 71 | 123 | 64 | 73 |
| win95pts | 76 | 0.16 | 109 | 112 | 38 | 50 |
| pathfinder | 109 | 0.34 | 157 | 195 | 203 | 245 |
| andes | 223 | 0.38 | 286 | 338 | 60 | 147 |
| diabetes | 413 | 2.48 | 583 | 602 | 309 | 251 |
| pigs | 441 | 0.88 | 592 | 592 | 0 | 0 |
| link | 724 | 1.89 | 855 | 1125 | 679 | 987 |
| munin | 1041 | 4.06 | 994 | 1397 | 1234 | 1603 |

Runtime follows how dense and interconnected a network is more than its size: `pigs` has more variables than `diabetes` and finishes faster. For comparison, causal-learn's GES on the same data takes 53 s on `alarm` and does not finish `andes` in half an hour.

## GES, XGES-0 and XGES

The same data, the mean over three samples of each network (one for `munin` and `link`), with the SHD between the learned and the true DAG:

| Network | Variables | GES SHD | XGES-0 SHD | XGES SHD | GES (s) | XGES-0 (s) | XGES (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| child | 20 | 2.0 | 2.0 | 2.0 | 0.02 | 0.03 | 0.18 |
| insurance | 27 | 21.7 | 21.7 | 12.0 | 0.03 | 0.04 | 0.60 |
| water | 32 | 45.0 | 45.0 | 41.3 | 0.02 | 0.03 | 0.22 |
| mildew | 35 | 22.0 | 22.0 | 22.0 | 0.03 | 0.04 | 0.10 |
| alarm | 37 | 10.0 | 4.0 | 4.0 | 0.03 | 0.04 | 0.18 |
| hailfinder | 56 | 55.3 | 55.3 | 58.3 | 0.05 | 0.06 | 0.33 |
| hepar2 | 70 | 62.7 | 62.7 | 61.3 | 0.04 | 0.05 | 0.42 |
| win95pts | 76 | 37.7 | 26.7 | 19.3 | 0.06 | 0.08 | 0.77 |
| pathfinder | 109 | 202.0 | 202.7 | 245.0 | 0.12 | 0.15 | 4.74 |
| andes | 223 | 59.7 | 59.7 | 61.0 | 0.14 | 0.16 | 4.31 |
| link | 724 | 679 | 689 | | 0.67 | 1.10 | |
| munin | 1041 | 1234 | 1219 | | 1.60 | 1.91 | |

XGES-0 moves clearly closer to the truth on `alarm` and `win95pts`, and stays within a few edges of GES elsewhere, for up to two thirds more time. XGES gains most on `insurance` and `win95pts`; on `pathfinder` it reaches a higher score with a structure further from the true one, whose near-deterministic tables the BDeu score rewards with extra parents.

On linear Gaussian data from random graphs (the setting of Nazaret and Blei: Erdős–Rényi DAGs with `rho` edges per variable on average, weights L1-normalised per child, BIC with a penalty of 2, 10,000 rows, five graphs per row), GES adds far more edges than the true graph has and does not remove them, and XGES recovers the true equivalence class almost exactly up to three edges per variable. The SHD here is between CPDAGs:

| Variables | rho | True edges | GES SHD | XGES-0 SHD | XGES SHD | GES (s) | XGES-0 (s) | XGES (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 2 | 46 | 17.0 | 7.8 | 1.6 | 0.02 | 0.03 | 0.36 |
| 25 | 3 | 73 | 43.8 | 34.6 | 1.8 | 0.02 | 0.04 | 1.16 |
| 25 | 4 | 99 | 61.2 | 43.6 | 18.2 | 0.03 | 0.06 | 1.40 |
| 50 | 2 | 96 | 34.8 | 13.0 | 2.4 | 0.04 | 0.05 | 1.28 |
| 50 | 3 | 148 | 57.6 | 41.0 | 2.2 | 0.05 | 0.10 | 3.30 |
| 50 | 4 | 205 | 80.2 | 39.8 | 13.6 | 0.06 | 0.12 | 5.06 |

## Against earlier releases

The same data and the same structure prior through 0.1.0 and 0.2.0, on the same machine:

| Network | Variables | 0.1.0 (s) | 0.2.0 (s) | Speedup |
|---|---:|---:|---:|---:|
| alarm | 37 | 0.89 | 0.05 | 18x |
| hailfinder | 56 | 2.26 | 0.10 | 23x |
| win95pts | 76 | 3.59 | 0.15 | 24x |
| pathfinder | 109 | 7.14 | 0.33 | 22x |
| andes | 223 | 24.8 | 0.72 | 34x |
| pigs | 441 | 113.5 | 2.71 | 42x |

Both versions learn structures with the same number of edges and the same BDeu score on every network in the table. The prior was 0.1.0's default, one parent expected per `n_rows - 1` candidates, which 0.2.0 expresses as `expected_parents=(n_vars - 1) / (n_rows - 1)`.

1.0.0 counts families without sorting rows and completes the CPDAG over neighbour lists. With the same prior, measured back to back:

| Network | Variables | 0.2.0 (s) | 1.0.0 (s) | Speedup |
|---|---:|---:|---:|---:|
| pathfinder | 109 | 0.62 | 0.42 | 1.5x |
| diabetes | 413 | 7.9 | 2.8 | 2.8x |
| link | 724 | 17.3 | 1.6 | 11x |
| munin | 1041 | 41.3 | 4.1 | 10x |

Both learn the same CPDAG on all four. With the default prior a few networks (`win95pts` and `munin` in the first table) end at a different structure of the same score, up to rounding, because ties between equally good operators break differently now that the score sums its terms in another order.
