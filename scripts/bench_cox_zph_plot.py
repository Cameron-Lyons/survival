"""Measure shared Cox diagnostic smoothing against separate per-term calls."""

from __future__ import annotations

import argparse
import json
import platform
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np
from survival.regression import cox_zph_smooth


def smooth_separately(x, y):
    return [cox_zph_smooth(x, y[:, [j]], [1.0]) for j in range(y.shape[1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rows", type=int, nargs="+", default=[1000, 10000, 100000])
    args = parser.parse_args()
    if min(args.rows) < 2 or args.repeats < 1:
        parser.error("rows must be at least 2 and repeats must be positive")
    results = []
    for n in args.rows:
        x = np.linspace(0, 1, n)
        for p in (1, 5, 20):
            y = np.sin(x[:, None] * np.arange(1, p + 1))
            for missing in (False, True):
                if missing:
                    y[::3, 1::2] = np.nan
                variance = np.ones(p)
                shared = cox_zph_smooth(x, y, variance)
                separate = smooth_separately(x, y)
                np.testing.assert_allclose(shared.y, np.column_stack([r.y for r in separate]))
                np.testing.assert_allclose(
                    shared.std_err, np.column_stack([r.std_err for r in separate])
                )
                times = {}
                for name, call in (
                    ("shared", partial(cox_zph_smooth, x, y, variance)),
                    ("separate", partial(smooth_separately, x, y)),
                ):
                    samples = []
                    for _ in range(args.repeats):
                        start = perf_counter()
                        call()
                        samples.append(1000 * (perf_counter() - start))
                    times[name + "_ms"] = median(samples)
                results.append({"rows": n, "terms": p, "two_masks": missing, **times})
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
