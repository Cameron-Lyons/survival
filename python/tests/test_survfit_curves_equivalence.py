"""survfit's faster paths against the code they replaced.

Each test takes the previous path, or keeps a copy of the previous implementation, and
checks the new one against it: the clustered Kaplan-Meier variance from running sums
agrees with the dense loop over every cluster to rounding.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
core = survival._survival
_survfit = r._survfit


def _identical(a, b):
    """Equal values, NaN where the other is NaN, and the same sign of every zero."""

    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return type(a) is type(b) and a == b


def _clustered_km_data(last_status, n=3000, seed=6):
    """Hundreds of clusters of mixed sizes, ties, zero weights and a last time at which
    everyone still at risk has ``last_status``, so that the curve (with ``reverse``, the
    censoring curve for a status of 0) ends at zero."""

    rng = np.random.default_rng(seed)
    time = np.round(rng.exponential(1.0, n), 1) + 0.1
    status = rng.integers(0, 2, n)
    last = rng.random(n) < 0.02
    time[last], status[last] = time.max() + 1.0, last_status
    cluster = np.where(rng.random(n) < 0.1, 0, rng.integers(1, n // 4, n))
    weights = np.where(rng.random(n) < 0.05, 0.0, rng.uniform(0.5, 2.0, n))
    return pd.DataFrame(
        {
            "time": time,
            "status": status,
            "cl": cluster,
            "w": weights,
            "g": np.array(["a", "b"])[rng.integers(0, 2, n)],
        }
    )


@pytest.mark.parametrize(
    "options",
    [{"weights": "w"}, {}, {"weights": "w", "reverse": True}, {"weights": "w", "stype": 2}],
)
def test_clustered_robust_variance_matches_the_dense_cluster_loop(options):
    """Many clusters take the running-sum variance; ``influence=True`` keeps the dense loop
    that updates every cluster at every time.  The two agree to rounding, and exactly where
    the curve reaches zero, so the confidence limits there are the same zero or NaN."""

    data = _clustered_km_data(last_status=0 if options.get("reverse") else 1)
    fast = r.survfit("Surv(time, status) ~ g", data=data, cluster="cl", **options)
    dense = r.survfit("Surv(time, status) ~ g", data=data, cluster="cl", influence=True, **options)
    assert _identical(fast.surv, dense.surv)
    for field in ("std_err", "std_chaz", "lower", "upper"):
        actual, expected = np.array(getattr(fast, field)), np.array(getattr(dense, field))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14, err_msg=field)
    zero = np.array(fast.surv) == 0.0
    assert zero.any() == ("stype" not in options)
    assert _identical(np.array(fast.std_err)[zero].tolist(), np.array(dense.std_err)[zero].tolist())
