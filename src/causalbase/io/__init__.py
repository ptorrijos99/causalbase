"""Reading Bayesian networks and sampling data from them."""

from __future__ import annotations

from causalbase.io.sampling import forward_sample
from causalbase.io.xbif import DiscreteBN, load_xbif

__all__ = ["DiscreteBN", "forward_sample", "load_xbif"]
