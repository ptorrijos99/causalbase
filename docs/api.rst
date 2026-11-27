.. _api:

#############
API reference
#############

Everything listed here is public. Modules, functions, methods and attributes
whose names start with an underscore, and names a module does not list in its
``__all__``, are implementation details and may change without notice.

Stability
=========

causalbase follows `semantic versioning <https://semver.org/>`_ from 1.0.0;
before that, a minor release may change the API, and the changelog says how.
From 1.0.0, within a major version, the names below keep their signatures and
meaning. What a minor release may still change is the exact structure learned
when two operators or two orders score the same up to rounding, since a
faster kernel may sum the same terms in another order; such changes are listed
in the changelog. The score of a given structure on given data does not change
beyond rounding.

The score objects cache what they compute and are not safe to share between
threads; use one per thread, or processes.

Estimators
==========

Every estimator is importable from ``causalbase`` itself. Hyperparameters go
to the constructor, :meth:`fit` learns, and the results end in an underscore.
The four on BDeu run a function of their subpackage, listed below, which
returns a :class:`~causalbase.search.SearchResult`.

.. currentmodule:: causalbase

.. autosummary::
   :toctree: generated/
   :template: estimator.rst

   search.GES
   search.XGES
   order.OrderSearch
   order.FLOP
   search.HillClimbing
   exact.ExactSearch
   lingam.RCD

Search functions
================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   search.ges
   search.xges
   order.order_search
   order.flop
   search.hill_climbing
   exact.exact_search

.. autosummary::
   :toctree: generated/
   :template: class.rst

   search.SearchResult
   search.Knowledge

Building blocks of the order searches
=====================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   order.OrderScorer
   order.OrderResult
   order.InsertionSearch
   order.IteratedLocalSearch
   order.ILSResult

.. autosummary::
   :toctree: generated/
   :template: function.rst

   order.pair_polish

Cumulant methods
================

The functions behind :class:`~causalbase.lingam.RCD`. Each takes the
statistics, not the data, except :func:`~causalbase.lingam.rcd_exact`.
``Cumulants`` and ``MomentSums`` are the dictionaries they pass around,
``Pruning`` and ``LambdaSelection`` the values of the estimator's options of
the same names.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lingam.moment_sums
   lingam.cumulants_from_sums
   lingam.rcd
   lingam.rcd_exact
   lingam.exact_order
   lingam.tau_matrix_3rd
   lingam.tau_matrix_4th
   lingam.estimate_b_from_cov
   lingam.estimate_b_lasso_from_cov

.. autosummary::
   :toctree: generated/
   :template: class.rst

   lingam.RCDResult

Scores
======

``Score`` is the interface every search reads; a new score implements it,
usually by subclassing :class:`~causalbase.scores.FamilyScore`, which adds the
cache, the batching and the structure prior. ``Prior`` names the structure
priors of BDeu (``"bernoulli"`` and ``"tetrad"``) and ``ScoreName`` the
scores ``make_score`` builds.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   scores.Score
   scores.FamilyScore
   scores.BDeuScore
   scores.BICScore
   scores.GaussianBICScore

.. autosummary::
   :toctree: generated/
   :template: function.rst

   scores.make_score
   scores.log_likelihood

Data
====

.. autosummary::
   :toctree: generated/
   :template: class.rst

   data.DiscreteDataset
   data.ContinuousDataset

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data.as_discrete
   data.as_discrete_sites
   data.as_continuous
   data.as_continuous_sites

Bundled networks
================

Five networks of the `Bayesian Network Repository
<https://www.bnlearn.com/bnrepository/>`_ ship with the package, each with one
sample of 5000 rows, so that the examples and the tests need no downloads.
Each loader returns a :class:`~causalbase.datasets.NetworkBundle` whose true
structure is already indexed like the columns of its data.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.load_network
   datasets.list_networks
   datasets.load_cancer
   datasets.load_asia
   datasets.load_child
   datasets.load_insurance
   datasets.load_alarm

.. autosummary::
   :toctree: generated/
   :template: class.rst

   datasets.NetworkBundle

Graphs
======

.. autosummary::
   :toctree: generated/
   :template: class.rst

   graph.DAG
   graph.PDAG

.. autosummary::
   :toctree: generated/
   :template: function.rst

   graph.dag_to_cpdag
   graph.moralize
   graph.mcs_order
   graph.fill_in
   graph.maximal_cliques
   graph.treewidth

Metrics
=======

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.shd
   metrics.shd_skeleton
   metrics.smhd
   metrics.smhd_graphs
   metrics.edge_scores
   metrics.graph_record

.. autosummary::
   :toctree: generated/
   :template: class.rst

   metrics.EdgeScores

Networks and sampling
=====================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   io.load_xbif
   io.forward_sample

.. autosummary::
   :toctree: generated/
   :template: class.rst

   io.DiscreteBN

Exceptions
==========

A cycle in a DAG and reading an estimator's results before ``fit`` raise a
:class:`~causalbase.exceptions.CausalBaseError`, which also derives from the
built-in exception a caller would expect. An invalid argument or input raises
``ValueError`` (or ``ConfigurationError``, which is one), and the wrong kind
of input ``TypeError``.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   exceptions.CausalBaseError
   exceptions.ConfigurationError
   exceptions.CycleError
   exceptions.NotFittedError
