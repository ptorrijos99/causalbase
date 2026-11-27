"""Record the original FedLiNGAM implementation's outputs as the regression fixture.

Run against the original repository, never against this package, or the test
would compare the package with itself. Usage::

    python tests/fixtures/lingam/record_baseline.py --fedlingam /path/to/FedLiNGAM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def make_lingam_data(seed: int, n_samples: int, n_vars: int) -> np.ndarray:
    """Generate a linear non-Gaussian SEM in a fixed causal order 0..n_vars-1."""
    rng = np.random.default_rng(seed)
    b_true = np.zeros((n_vars, n_vars))
    for child in range(1, n_vars):
        for parent in range(child):
            if rng.random() < 0.5:
                b_true[child, parent] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
    # Laplace noise: non-Gaussian, which is what makes the order identifiable.
    noise = rng.laplace(0.0, 1.0, size=(n_samples, n_vars))
    x = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        x[:, j] = x @ b_true[j] + noise[:, j]
    return x


def split(x: np.ndarray, n_clients: int) -> list[np.ndarray]:
    """Contiguous horizontal split, matching the experiments' partitioning."""
    return [part for part in np.array_split(x, n_clients, axis=0)]


CASES = [
    {"name": "p5_n2000_c3", "seed": 0, "n_samples": 2000, "n_vars": 5, "n_clients": 3},
    {"name": "p8_n5000_c4", "seed": 1, "n_samples": 5000, "n_vars": 8, "n_clients": 4},
    {"name": "p10_n3000_c2", "seed": 2, "n_samples": 3000, "n_vars": 10, "n_clients": 2},
]


def main() -> None:
    """Record FedRCD baselines from the original FedLiNGAM checkout to a JSON fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedlingam", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/lingam/rcd_baseline.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.fedlingam))
    from src.common.cumulants import (  # noqa: PLC0415
        aggregate_and_compute_cumulants,
        compute_local_raw_stats,
    )
    from src.fedishc.core import hc as fedhc_orig  # noqa: PLC0415
    from src.fedishc.core import ishc as fedishc_orig  # noqa: PLC0415
    from src.fedrcd.core import (  # noqa: PLC0415
        fedrcd,
        fedrcd_exact,
        fedrcd_noniterative,
    )
    from src.fedrcd.core import ishc as ishc_beta_orig  # noqa: PLC0415
    from src.hclingam.hclingam import HCLiNGAM  # noqa: PLC0415

    # Two different functions named `ishc`, aliased above to keep them apart.

    records = []
    for case in CASES:
        x = make_lingam_data(case["seed"], case["n_samples"], case["n_vars"])
        clients = split(x, case["n_clients"])
        raw = [compute_local_raw_stats(part, order=4) for part in clients]
        stats = aggregate_and_compute_cumulants(raw)

        # The same for every method of a case; see test_cumulants_match_the_recording.
        cumulants = {
            key: np.asarray(stats[key], dtype=float).tolist()
            for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22")
        }

        # Federated baselines: consume aggregated stats, never raw data.
        for method, order_and_b in [
            ("fedrcd", fedrcd(stats)),
            ("fedrcd_noniterative", fedrcd_noniterative(stats)),
            ("ishc", ishc_beta_orig(stats)),
            ("fedrcd_exact", fedrcd_exact(clients)),
            ("fedishc", fedishc_orig(stats)),
            # The original `hc()` returns None for B.
            ("fedhc", fedhc_orig(stats)),
        ]:
            causal_order, b_est = order_and_b
            records.append(
                {
                    **case,
                    "method": method,
                    "causal_order": [int(v) for v in causal_order],
                    "B": None if b_est is None else np.asarray(b_est, dtype=float).tolist(),
                    "cumulants": cumulants,
                }
            )

        # Centralised baselines: consume the pooled data matrix directly,
        # not the per-client stats -- there is no federated aggregation step.
        for method, hclingam_method in [("hc_lingam", "hc-lingam"), ("hc", "hc")]:
            model = HCLiNGAM(method=hclingam_method).fit(x)
            records.append(
                {
                    **case,
                    "method": method,
                    "causal_order": [int(v) for v in model.causal_order_],
                    "B": np.asarray(model.adjacency_matrix_, dtype=float).tolist(),
                    "cumulants": cumulants,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": records}, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
