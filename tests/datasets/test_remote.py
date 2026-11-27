"""Downloading networks from Zenodo, against files served from memory."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _bif import to_bif
from causalbase.datasets import _remote as remote
from causalbase.datasets import (
    fetch_network,
    get_data_home,
    list_remote_networks,
    load_network,
)
from causalbase.io import load_xbif

DATA = Path(__file__).resolve().parents[2] / "src" / "causalbase" / "datasets" / "data"


def asia_csv() -> bytes:
    return gzip.decompress((DATA / "asia.0.csv.gz").read_bytes())


def asia_bif() -> bytes:
    bn = load_xbif(DATA / "asia.xbif")
    return to_bif(bn, np.random.default_rng(0)).encode()


def zipped(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buffer.getvalue()


class Server:
    """Answers the downloads from a dict of URL to bytes, and counts them."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[str] = []

    def __call__(self, request: Any, timeout: float) -> io.BytesIO:
        url = request.full_url
        self.requests.append(url)
        if url not in self.files:
            raise urllib.error.URLError("no route to host")
        return io.BytesIO(self.files[url])


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Server:
    """Serve asia as bnlearn serves it (CSVs in a zip, the BIF apart) and as bnRep does."""
    bnlearn_zip = zipped({"asia/asia.0.csv": asia_csv(), "asia/asia.1.csv": asia_csv()})
    bnrep_zip = zipped(
        {"bnrep_asia/bnrep_asia.5.csv": asia_csv(), "bnrep_asia/bnrep_asia.bif": asia_bif()}
    )
    registry = {
        "bayesfl_commit": "0123456789abcdef",
        "networks": {
            "asia": {
                "source": "bnlearn",
                "record": "1",
                "file": "asia.zip",
                "md5": hashlib.md5(bnlearn_zip).hexdigest(),
                "size": len(bnlearn_zip),
                "n_vars": 8,
                "bif_sha256": hashlib.sha256(asia_bif()).hexdigest(),
            },
            "bnrep_asia": {
                "source": "bnRep",
                "record": "2",
                "file": "bnrep_small_06-10_nodes.zip",
                "md5": hashlib.md5(bnrep_zip).hexdigest(),
                "size": len(bnrep_zip),
                "n_vars": 8,
            },
        },
    }
    srv = Server(
        {
            remote.ZENODO.format(record="1", file="asia.zip"): bnlearn_zip,
            remote.ZENODO.format(record="2", file="bnrep_small_06-10_nodes.zip"): bnrep_zip,
            remote.BAYESFL.format(commit="0123456789abcdef", name="asia"): asia_bif(),
        }
    )
    monkeypatch.setattr(remote, "_registry", lambda: registry)
    monkeypatch.setattr(remote, "_urlopen", srv)
    return srv


def leftovers(home: Path) -> list[Path]:
    return [p for p in home.rglob("*") if p.suffix == ".part"]


@pytest.mark.parametrize(("name", "sample"), [("asia", 1), ("bnrep_asia", 5)])
def test_a_fetched_network_is_the_bundled_one_and_is_read_from_the_cache_after(
    server: Server, tmp_path: Path, name: str, sample: int
) -> None:
    bundled = load_network("asia")
    fetched = fetch_network(name, sample, data_home=tmp_path)
    assert fetched.name == name
    assert fetched.data.names == bundled.data.names
    assert fetched.data.categories == bundled.data.categories
    assert np.array_equal(fetched.data.values, bundled.data.values)
    assert sorted(fetched.truth.edges()) == sorted(bundled.truth.edges())
    n_requests = len(server.requests)
    fetch_network(name, sample, data_home=tmp_path)
    assert len(server.requests) == n_requests
    assert not leftovers(tmp_path)


def test_a_corrupt_download_leaves_nothing_behind(server: Server, tmp_path: Path) -> None:
    url = remote.ZENODO.format(record="1", file="asia.zip")
    server.files[url] = server.files[url][:-10] + b"0123456789"
    with pytest.raises(OSError, match="corrupt"):
        fetch_network("asia", data_home=tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_a_failed_download_says_which_and_leaves_nothing_behind(
    server: Server, tmp_path: Path
) -> None:
    del server.files[remote.BAYESFL.format(commit="0123456789abcdef", name="asia")]
    with pytest.raises(OSError, match="could not download"):
        fetch_network("asia", data_home=tmp_path)
    assert not leftovers(tmp_path)


def test_a_truncated_file_in_the_cache_is_downloaded_again(server: Server, tmp_path: Path) -> None:
    fetch_network("asia", data_home=tmp_path)
    archive = tmp_path / "zenodo" / "1" / "asia.zip"
    archive.write_bytes(archive.read_bytes()[:100])
    fetch_network("asia", data_home=tmp_path)
    assert zipfile.is_zipfile(archive)


def test_unknown_networks_and_samples_are_refused(server: Server, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown network"):
        fetch_network("asai", data_home=tmp_path)
    for sample in (-1, 100, 1.0, True):
        with pytest.raises(ValueError, match="sample"):
            fetch_network("asia", sample, data_home=tmp_path)  # type: ignore[arg-type]
    assert not server.requests


def test_the_cache_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CAUSALBASE_DATA", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert get_data_home() == tmp_path / "xdg" / "causalbase"
    monkeypatch.setenv("CAUSALBASE_DATA", str(tmp_path / "env"))
    assert get_data_home() == tmp_path / "env"
    assert get_data_home(tmp_path / "given") == tmp_path / "given"


def test_the_registry_lists_every_network_once_with_its_checksum() -> None:
    every = list_remote_networks()
    bnlearn, bnrep = list_remote_networks("bnlearn"), list_remote_networks("bnRep")
    assert len(bnlearn) == 24 and len(bnrep) == 202
    assert every == {**bnlearn, **bnrep}
    assert all(name.startswith("bnrep_") for name in bnrep)
    assert {"asia": 8, "pathfinder": 109, "munin": 1041}.items() <= bnlearn.items()
    networks = remote._registry()["networks"]
    for name, entry in networks.items():
        assert len(entry["md5"]) == 32 and entry["size"] > 0
        assert entry["source"] == "bnRep" or len(entry["bif_sha256"]) == 64
        if entry["source"] == "bnRep" and entry["n_vars"] <= 20:
            low = (entry["n_vars"] - 1) // 5 * 5 + 1
            assert entry["file"] == f"bnrep_small_{low:02d}-{low + 4:02d}_nodes.zip", name
    with pytest.raises(ValueError, match="source"):
        list_remote_networks("bnrepo")


@pytest.mark.skipif(
    not os.environ.get("CAUSALBASE_TEST_DOWNLOADS"),
    reason="downloads from Zenodo and GitHub; set CAUSALBASE_TEST_DOWNLOADS=1 to run",
)
def test_real_downloads_give_the_networks_of_both_repositories(tmp_path: Path) -> None:
    for name, sample in (("asia", 99), ("bnrep_titanic", 0)):
        bundle = fetch_network(name, sample, data_home=tmp_path)
        assert bundle.data.n_rows == 5000
        assert bundle.data.n_vars == list_remote_networks()[name]
        assert bundle.truth.num_nodes == bundle.data.n_vars
