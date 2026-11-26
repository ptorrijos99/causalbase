"""Small Bayesian networks bundled with the package, for examples and tests.

Five networks from the public `Bayesian Network Repository
<https://www.bnlearn.com/bnrepository/>`_ ship inside ``causalbase``, each as a
network definition and one dataset of 5000 rows sampled from it. They are here
so that the examples, the documentation, and the test suite run with no
downloads and no environment variables.

    >>> from causalbase.datasets import load_asia
    >>> bundle = load_asia()
    >>> bundle.data.n_vars
    8

Each loader returns a :class:`NetworkBundle` carrying the sampled data, the
true structure, and the network itself. The true structure comes back already
re-indexed onto the dataset's column order: the ``.xbif`` file declares its own
variable order, which need not match the CSV's, and comparing without
re-indexing silently compares the wrong nodes.

For anything larger than ``alarm``, bring your own data (see
``DiscreteDataset.from_csv``) or sample from a network you load yourself with
:func:`causalbase.io.load_xbif`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import TYPE_CHECKING

from causalbase.data import DiscreteDataset
from causalbase.graph import DAG
from causalbase.io import load_xbif

if TYPE_CHECKING:
    from causalbase.io import DiscreteBN

__all__ = [
    "NetworkBundle",
    "list_networks",
    "load_alarm",
    "load_asia",
    "load_cancer",
    "load_child",
    "load_insurance",
    "load_network",
]

#: The bundled networks, smallest first, with their variable counts.
_NETWORKS: dict[str, int] = {
    "cancer": 5,
    "asia": 8,
    "child": 20,
    "insurance": 27,
    "alarm": 37,
}


@dataclass(frozen=True)
class NetworkBundle:
    """A bundled network, its true structure, and data sampled from it.

    Attributes
    ----------
    name : str
        The network's name, e.g. ``"asia"``.
    data : DiscreteDataset
        5000 rows sampled from `network`.
    truth : DAG
        The true structure, re-indexed onto ``data.names`` so that it can be
        compared against a learned graph directly.
    network : DiscreteBN
        The network itself, including its conditional probability tables.
        Pass it to :func:`causalbase.io.forward_sample` to draw more rows.
    """

    name: str
    data: DiscreteDataset
    truth: DAG
    network: DiscreteBN


def list_networks() -> dict[str, int]:
    """Return the bundled networks and their variable counts, smallest first.

    Returns
    -------
    dict[str, int]
        Network name to number of variables.

    Examples
    --------
    >>> from causalbase.datasets import list_networks
    >>> list_networks()
    {'cancer': 5, 'asia': 8, 'child': 20, 'insurance': 27, 'alarm': 37}
    """
    return dict(_NETWORKS)


def load_network(name: str) -> NetworkBundle:
    """Load one of the bundled networks by name.

    Parameters
    ----------
    name : str
        One of the keys of :func:`list_networks`.

    Returns
    -------
    NetworkBundle
        The data, the true structure, and the network.

    Raises
    ------
    ValueError
        If `name` is not one of the bundled networks.

    Examples
    --------
    >>> from causalbase.datasets import load_network
    >>> load_network("cancer").data.n_vars
    5
    """
    if name not in _NETWORKS:
        available = ", ".join(_NETWORKS)
        raise ValueError(f"unknown network {name!r}; bundled networks are: {available}")

    root = files(__package__) / "data"

    # `as_file` is a no-op for an unpacked wheel and extracts to a temporary
    # file if the package is imported from a zip, so both loaders below get a
    # real path either way. `from_csv` reads the `.gz` directly.
    with as_file(root / f"{name}.0.csv.gz") as csv_path:
        data = DiscreteDataset.from_csv(csv_path)
    with as_file(root / f"{name}.xbif") as xbif_path:
        network = load_xbif(xbif_path)

    # A level the sample happens not to contain is still a level of the
    # network, so the encoding takes its levels from the network rather than
    # from the data, as forward_sample does.
    outcomes = dict(zip(network.names, network.outcomes, strict=True))
    levels = {var: sorted(levels) for var, levels in outcomes.items()}
    data = DiscreteDataset.from_pandas(data.to_pandas(), [levels[var] for var in data.names])

    # The .xbif declares its own variable order, which need not match the CSV's.
    index = {var: i for i, var in enumerate(data.names)}
    truth = DAG.from_edges(
        data.n_vars,
        [(index[network.names[u]], index[network.names[v]]) for u, v in network.dag.edges()],
    )
    return NetworkBundle(name=name, data=data, truth=truth, network=network)


def load_cancer() -> NetworkBundle:
    """Load `cancer`: 5 variables, 5000 rows. The smallest bundled network."""
    return load_network("cancer")


def load_asia() -> NetworkBundle:
    """Load `asia`: 8 variables, 5000 rows. The usual starting point."""
    return load_network("asia")


def load_child() -> NetworkBundle:
    """Load `child`: 20 variables, 5000 rows."""
    return load_network("child")


def load_insurance() -> NetworkBundle:
    """Load `insurance`: 27 variables, 5000 rows."""
    return load_network("insurance")


def load_alarm() -> NetworkBundle:
    """Load `alarm`: 37 variables, 5000 rows. The largest bundled network."""
    return load_network("alarm")
