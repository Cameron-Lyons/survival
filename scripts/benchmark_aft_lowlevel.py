#!/usr/bin/env python3
"""Compare AFT fitting and retained result sizes with prepared native data."""

import argparse
import json
import pickle
import statistics
import time

import numpy as np
from survival import regression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(1973)
    x = rng.normal(size=(args.rows, args.columns))
    x[:, 0] = 1
    y = x @ np.linspace(0.1, 0.6, args.columns) + rng.normal(size=args.rows)
    status = rng.binomial(1, 0.7, args.rows).astype(np.int32)
    data = regression.SurvregData(y, status, x)
    distribution = regression.SurvregDistribution("gaussian")
    results = []
    full = None
    for name, function in (("full", regression.survreg_fit), ("bare", regression.survreg_fit_raw)):
        durations = []
        for _ in range(args.repeats + 1):
            start = time.perf_counter()
            fit = function(data, distribution)
            durations.append(time.perf_counter() - start)
        if full is None:
            full = fit
        else:
            np.testing.assert_allclose(fit.coefficients, full.coefficients, atol=1e-12)
            np.testing.assert_allclose(fit.var, full.variance_matrix, atol=1e-12)
            np.testing.assert_allclose(
                fit.loglik, [full.intercept_only_log_likelihood, full.log_likelihood], atol=1e-12
            )
        results.append(
            {
                "mode": name,
                "rows": args.rows,
                "columns": args.columns,
                "median_ms": statistics.median(durations[1:]) * 1000,
                "serialized_bytes": len(pickle.dumps(fit, protocol=5)),
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
