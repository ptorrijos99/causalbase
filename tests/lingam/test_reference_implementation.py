"""causalbase.lingam reproduces the original FedLiNGAM and HC-LiNGAM implementations.

The fixtures come from those implementations; see tests/fixtures/lingam/README.md.
If this fails, fix the code: never re-record the fixtures from it.
"""

import json
from pathlib import Path

import numpy as np

from causalbase.lingam import cumulants_from_sums, moment_sums, rcd, rcd_exact

FIXTURE = Path(__file__).parent.parent / "fixtures" / "lingam" / "rcd_baseline.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
# The cumulants belong to the problem, not the method, so one record per problem.
PROBLEMS = list({case["name"]: case for case in CASES}.values())

# Each recorded method as a setting of the solvers, all with OLS coefficients
# (pruning=None), as the originals estimated them. FedISHC is left out: the
# original divides Cum(x_j, x_j, x_s) by Cum(x_s, x_s, x_s), which estimates the
# square of the coefficient, and rcd(method="ratio") uses the coefficient.
SETTINGS = {
    "fedrcd": {},
    "fedrcd_noniterative": {"method": "sort"},
    "ishc": {"order": 3},
    "fedhc": {"method": "sort", "order": 3},
}


def make_lingam_data(seed: int, n_samples: int, n_vars: int) -> np.ndarray:
    """Regenerate a recorded problem, drawing in exactly the recorder's order."""
    rng = np.random.default_rng(seed)
    b_true = np.zeros((n_vars, n_vars))
    for child in range(1, n_vars):
        for parent in range(child):
            if rng.random() < 0.5:
                b_true[child, parent] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
    noise = rng.laplace(0.0, 1.0, size=(n_samples, n_vars))
    x = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        x[:, j] = x @ b_true[j] + noise[:, j]
    return x


def _data(case: dict) -> np.ndarray:
    return make_lingam_data(case["seed"], case["n_samples"], case["n_vars"])


def _blocks(case: dict) -> list[np.ndarray]:
    """The rows as the recording split them between sites, here blocks of one table."""
    return list(np.array_split(_data(case), case["n_clients"], axis=0))


def test_solvers_reproduce_the_original() -> None:
    for case in CASES:
        method = case["method"]
        if method == "fedishc":
            continue
        if method in ("hc", "fedrcd_exact"):
            # Deflating every site's rows by the same coefficients deflates the table.
            result = rcd_exact(_data(case), pruning=None)
        elif method == "hc_lingam":
            result = rcd(
                cumulants_from_sums([moment_sums(_data(case))]), method="sort", pruning=None
            )
        else:
            stats = cumulants_from_sums([moment_sums(b, order=4) for b in _blocks(case)])
            result = rcd(stats, pruning=None, **SETTINGS[method])  # type: ignore[arg-type]

        assert list(result.causal_order) == case["causal_order"], case["name"]
        if case["B"] is None:
            # The original returns no B for fedhc.
            continue
        np.testing.assert_allclose(
            result.coef, np.array(case["B"]), rtol=1e-9, atol=1e-9, err_msg=case["name"]
        )


def test_cumulants_match_the_recording() -> None:
    """The cumulants match, not just the order they lead to.

    B is estimated from the covariance alone, so a small error in the higher
    cumulants could pass the other test unless it flipped the order.
    """
    for case in PROBLEMS:
        stats = cumulants_from_sums([moment_sums(b, order=4) for b in _blocks(case)])
        for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"):
            np.testing.assert_allclose(
                stats[key], np.array(case["cumulants"][key]), rtol=1e-12, atol=1e-12
            )
