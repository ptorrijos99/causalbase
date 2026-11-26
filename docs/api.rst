API reference
=============

Everything below is public. Anything not listed here is an implementation
detail and may change without notice: modules, functions, methods and
attributes whose names start with an underscore, and names a module does not
list in its ``__all__``.

The estimators, :class:`~causalbase.search.GES`,
:class:`~causalbase.search.XGES`, :class:`~causalbase.search.HillClimbing`,
:class:`~causalbase.order.OrderSearch`, :class:`~causalbase.order.FLOP` and
:class:`~causalbase.exact.ExactSearch`, are also importable from
``causalbase`` itself. Each of the last four runs a function of the same
subpackage (:func:`~causalbase.search.hill_climbing`,
:func:`~causalbase.order.order_search`, :func:`~causalbase.order.flop`,
:func:`~causalbase.exact.exact_search`) on a
:class:`~causalbase.scores.BDeuScore`, scored with its compiled engine, and
each function returns a :class:`~causalbase.search.SearchResult`. GES and
XGES take any score, ``BDeuScore(prior="tetrad")`` included, and return a
CPDAG. Every estimator takes the score's ``ess``, ``expected_parents`` and
``prior`` (``"bernoulli"`` or ``"tetrad"``). All take ``random_state``, ``max_iter``, ``n_restarts``, ``time_limit`` and
``n_jobs`` with scikit-learn's meaning where they apply.

Stability
---------

causalbase follows `Semantic Versioning <https://semver.org/>`_ from 1.0.0;
before that, minor releases may change the API. From 1.0.0,
within a major version, the names below keep their signatures and meaning,
and code written against one release runs on the next. What may change in a
minor release is the exact structure learned when two operators or two
orders score the same up to rounding, since a faster kernel may sum the same
terms in another order; such changes are listed in the changelog. Scores of a
given structure on given data do not change beyond rounding.

The score objects cache what they compute and are not safe to share between
threads; use one per thread, or processes.

Data
----

.. automodule:: causalbase.datasets

.. automodule:: causalbase.data

Scores and search
-----------------

.. automodule:: causalbase.scores

.. automodule:: causalbase.search

.. automodule:: causalbase.order

.. automodule:: causalbase.exact

Graphs and metrics
------------------

.. automodule:: causalbase.graph

.. automodule:: causalbase.metrics

.. automodule:: causalbase.io

Exceptions
----------

.. automodule:: causalbase.exceptions
