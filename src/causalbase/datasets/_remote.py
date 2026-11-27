"""Networks downloaded from Zenodo on first use and kept in a local cache.

The samples are those of two Zenodo records: 100 samples of 5000 rows of each
of the 24 discrete networks of bnlearn's Bayesian Network Repository, and of
each of the 202 discrete networks of bnRep. ``remote.json`` pins the version
of each record, the file that holds every network and its md5, so that a
download is checked and a network always gives the same rows. bnRep's zip
files carry their networks in BIF; bnlearn's networks come from the BIF files
of the BayesFL repository, at a pinned commit and checked by their SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from causalbase.data import DiscreteDataset
from causalbase.io.bif import parse_bif
from causalbase.io.xbif import DiscreteBN

__all__ = ["fetch", "get_data_home", "list_remote_networks"]

N_SAMPLES = 100
ZENODO = "https://zenodo.org/api/records/{record}/files/{file}/content"
BAYESFL = (
    "https://raw.githubusercontent.com/ptorrijos99/BayesFL/{commit}/res/networks/BIF/{name}.bif"
)

# Replaced by the tests, which serve the files from a local folder.
_urlopen = urllib.request.urlopen


@cache
def _registry() -> dict[str, Any]:
    text = (files(__package__) / "remote.json").read_text(encoding="utf-8")
    registry: dict[str, Any] = json.loads(text)
    return registry


def list_remote_networks(source: str | None = None) -> dict[str, int]:
    """Return the networks :func:`fetch_network` can download, with their variable counts.

    Parameters
    ----------
    source : {"bnlearn", "bnRep"} or None, default=None
        Only the networks of one repository; None lists both. bnRep's names
        carry the prefix ``bnrep_``, since some clash with bnlearn's.

    Returns
    -------
    dict[str, int]
        Network name to number of variables, bnlearn's first, each
        repository from the smallest network to the largest.

    Raises
    ------
    ValueError
        If `source` is not one of the two repositories.

    Examples
    --------
    >>> from causalbase.datasets import list_remote_networks
    >>> networks = list_remote_networks()
    >>> len(networks), networks["pathfinder"], networks["bnrep_earthquake"]
    (226, 109, 40)
    """
    if source not in (None, "bnlearn", "bnRep"):
        raise ValueError(f"source must be 'bnlearn', 'bnRep' or None, got {source!r}")
    return {
        name: entry["n_vars"]
        for name, entry in _registry()["networks"].items()
        if source is None or entry["source"] == source
    }


def get_data_home(data_home: str | os.PathLike[str] | None = None) -> Path:
    """Return the folder that caches the networks :func:`fetch_network` downloads.

    Parameters
    ----------
    data_home : str or path-like, optional
        The folder to use. By default the ``CAUSALBASE_DATA`` environment
        variable, and without it ``causalbase`` inside ``XDG_CACHE_HOME``
        (``~/.cache`` when that is not set either).

    Returns
    -------
    pathlib.Path
        The folder, which need not exist yet.
    """
    path = data_home if data_home is not None else os.environ.get("CAUSALBASE_DATA")
    if path is None:
        path = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "causalbase"
    return Path(path).expanduser()


def _download(url: str, dest: Path, algorithm: str, digest: str) -> None:
    """Download `url` to `dest`, checking its digest, so that `dest` is complete or absent."""
    from causalbase import __version__

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": f"causalbase/{__version__}"})
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    tmp = Path(tmp_name)
    try:
        hasher = hashlib.new(algorithm)
        with os.fdopen(fd, "wb") as out, _urlopen(request, timeout=60) as response:
            while chunk := response.read(1 << 20):
                hasher.update(chunk)
                out.write(chunk)
        if hasher.hexdigest() != digest:
            raise OSError(
                f"the download of {url} is corrupt: its {algorithm} is {hasher.hexdigest()}, "
                f"expected {digest}; try again"
            )
        os.replace(tmp, dest)
    except urllib.error.URLError as exc:
        raise OSError(f"could not download {url}: {exc.reason}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _cached(dest: Path, url: str, algorithm: str, digest: str, size: int | None = None) -> Path:
    if not dest.exists() or (size is not None and dest.stat().st_size != size):
        _download(url, dest, algorithm, digest)
    return dest


def fetch(
    name: str, sample: int, data_home: str | os.PathLike[str] | None
) -> tuple[DiscreteDataset, DiscreteBN]:
    """Return sample `sample` of network `name` and the network, downloading what is missing."""
    networks = _registry()["networks"]
    if name not in networks:
        raise ValueError(
            f"unknown network {name!r}; list_remote_networks() gives the {len(networks)} available"
        )
    if isinstance(sample, bool) or not isinstance(sample, int) or not 0 <= sample < N_SAMPLES:
        raise ValueError(f"sample must be an integer from 0 to {N_SAMPLES - 1}, got {sample!r}")
    entry = networks[name]
    home = get_data_home(data_home)
    archive = _cached(
        home / "zenodo" / entry["record"] / entry["file"],
        ZENODO.format(record=entry["record"], file=entry["file"]),
        "md5",
        entry["md5"],
        entry["size"],
    )
    with zipfile.ZipFile(archive) as zf:
        members = {member.rsplit("/", 1)[-1]: member for member in zf.namelist()}
        csv = members.get(f"{name}.{sample}.csv")
        if csv is None:
            raise OSError(f"{archive} holds no {name}.{sample}.csv; delete it and fetch again")
        with zf.open(csv) as fh:
            data = DiscreteDataset.from_csv(fh)
        if entry["source"] == "bnRep":
            with zf.open(members[f"{name}.bif"]) as fh:
                text = fh.read().decode("utf-8")
            return data, parse_bif(text, f"{archive.name}/{name}.bif")
    commit = _registry()["bayesfl_commit"]
    bif = _cached(
        home / "bayesfl" / commit[:12] / f"{name}.bif",
        BAYESFL.format(commit=commit, name=name),
        "sha256",
        entry["bif_sha256"],
    )
    return data, parse_bif(bif.read_text(encoding="utf-8"), str(bif))
