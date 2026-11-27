"""Writing a network as BIF, for the tests of the BIF reader and of the downloads."""

from __future__ import annotations

from itertools import product

import numpy as np

from causalbase.io import DiscreteBN


def to_bif(bn: DiscreteBN, rng: np.random.Generator) -> str:
    """Write `bn` as BIF, its declarations and conditional rows in a random order."""
    lines = ["network unknown {", "}"]
    for j in rng.permutation(len(bn.names)):
        states = ", ".join(bn.outcomes[j])
        lines += [
            f"variable {bn.names[j]} {{",
            f"  type discrete [ {len(bn.outcomes[j])} ] {{ {states} }};",
            "}",
        ]
    for j in rng.permutation(len(bn.names)):
        parents = [bn.names[p] for p in bn.parent_order[j]]
        cpt = bn.cpts[j]
        if not parents:
            lines += [
                f"probability ( {bn.names[j]} ) {{",
                "  table " + ", ".join(repr(float(v)) for v in cpt[0]) + ";",
                "}",
            ]
            continue
        lines.append(f"probability ( {bn.names[j]} | {', '.join(parents)} ) {{")
        configs = list(product(*(bn.outcomes[p] for p in bn.parent_order[j])))
        for i in rng.permutation(len(configs)):
            lines.append(
                f"  ({', '.join(configs[i])}) " + ", ".join(repr(float(v)) for v in cpt[i]) + ";"
            )
        lines.append("}")
    return "\n".join(lines) + "\n"
