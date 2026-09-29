#!/usr/bin/env python3
"""Compare prepared KM calls and serialized result sizes."""

import argparse
import json
import pickle
import statistics
import time

import numpy as np
from survival import r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--curves", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.curves, args.repeats) < 1:
        parser.error("rows, curves and repeats must be positive")
    rng = np.random.default_rng(1973)
    y = r.Surv(rng.exponential(size=args.rows), rng.binomial(1, 0.7, args.rows))
    x = r.strata(rng.integers(args.curves, size=args.rows), shortlabel=True)
    results = []
    full = None
    for name, function in (
        ("full", lambda: r.survfit(y, group=x, timefix=False)),
        ("bare", lambda: r.survfitKM(x, y)),
    ):
        durations = []
        for _ in range(args.repeats + 1):
            fit = None
            start = time.perf_counter()
            fit = function()
            durations.append(time.perf_counter() - start)
        if full is None:
            full = fit
        else:
            for field in ("time", "n_risk", "surv", "cumhaz", "std_err", "lower", "upper"):
                np.testing.assert_allclose(
                    getattr(fit, field), getattr(full, field), rtol=1e-12, atol=1e-14
                )
        results.append(
            {
                "mode": name,
                "rows": args.rows,
                "curves": args.curves,
                "median_ms": statistics.median(durations[1:]) * 1000,
                "serialized_bytes": len(pickle.dumps(fit, protocol=5)),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
