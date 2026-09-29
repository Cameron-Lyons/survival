#!/usr/bin/env python3
"""Compare full and bare Cox binding calls on the same inputs and optimizer."""

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
    time_values = rng.integers(1, 1000, args.rows).astype(float)
    status = rng.binomial(1, 0.7, args.rows).astype(np.int32)
    results = []
    reference = None
    for name, function, kwargs in (
        ("full", regression.coxph_fit, {}),
        ("bare_with_residuals", regression.coxph_fit_raw, {"resid": True}),
        ("bare_without_residuals", regression.coxph_fit_raw, {"resid": False}),
    ):
        durations = []
        for _ in range(args.repeats + 1):
            start = time.perf_counter()
            fit = function(time_values, status, x, nocenter=[], **kwargs)
            durations.append(time.perf_counter() - start)
        if reference is None:
            reference = fit
        np.testing.assert_allclose(fit.coefficients, reference.coefficients, atol=1e-12)
        np.testing.assert_allclose(fit.var, reference.var, atol=1e-12)
        np.testing.assert_allclose(fit.loglik, reference.loglik, atol=1e-12)
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
