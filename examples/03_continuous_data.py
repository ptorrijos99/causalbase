"""
Learn from continuous data
==========================

With ``criterion="gaussian_bic"`` or ``criterion="bge"``, every estimator
scores a linear Gaussian model of each node given its parents; here
:class:`~causalbase.search.GES` and :class:`~causalbase.search.XGES` do. The
data below come from a known linear model, so the answer can be checked.
"""

import numpy as np
import pandas as pd

from causalbase import GES, XGES

rng = np.random.default_rng(0)
n = 5000
smoking = rng.normal(size=n)
pollution = rng.normal(size=n)
cancer = 0.8 * smoking + 0.6 * pollution + rng.normal(scale=0.5, size=n)
cough = 0.7 * cancer + rng.normal(scale=0.5, size=n)
xray = 0.9 * cancer + rng.normal(scale=0.5, size=n)
frame = pd.DataFrame(
    {"smoking": smoking, "pollution": pollution, "cancer": cancer, "cough": cough, "xray": xray}
)

# %%
# ``smoking -> cancer <- pollution`` is a v-structure, so both of its edges
# are compelled, and so are the edges out of ``cancer``, which the
# v-structure orients.

model = GES(criterion="gaussian_bic").fit(frame)
model.adjacency_

# %%
# XGES learns the same class here; it differs from GES on denser graphs,
# where GES's early insertions keep out better edges.

(XGES(criterion="gaussian_bic").fit(frame).adjacency_ == model.adjacency_).all().all()

# %%
# ``score(X)`` is the mean log likelihood per row of held-out data under the
# learned structure, with the linear Gaussian model fitted on the training
# rows.

train, test = frame.iloc[:4000], frame.iloc[4000:]
GES(criterion="gaussian_bic").fit(train).score(test)
