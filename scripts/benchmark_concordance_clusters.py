#!/usr/bin/env python3
"""Complete clustered concordance calls, excluding fitting and input setup.

An optional --baseline names a previous _concordance.py source file. It is
loaded within survival.r so imports resolve to the same current shared modules.
Only cases with equivalent correct baseline results are timed.
"""

import argparse
import gc
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from survival import r


def baseline_module(path):
    name = "survival.r._concordance_benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.rows < 30 or args.repeats < 1:
        parser.error("at least 30 rows and one repeat are required")
    rng = np.random.default_rng(915)
    data = {
        "time": rng.exponential(size=args.rows),
        "status": rng.binomial(1, 0.7, size=args.rows),
        "x": rng.normal(size=args.rows),
        "z": rng.normal(size=args.rows),
        "id": np.arange(args.rows) % 100,
    }
    first = r.coxph("Surv(time,status)~x+cluster(id)", data)
    second = r.coxph("Surv(time,status)~z+cluster(id)", data)
    renamed = r.coxph("Surv(time,status)~z+cluster(id)", {**data, "id": (data["id"] + 7) % 100})
    baseline = None if args.baseline is None else baseline_module(args.baseline)
    calls = {
        "single": {"current": lambda: r.concordance(first)},
        "joint": {"current": lambda: r.concordance(first, second)},
        "explicit_joint": {"current": lambda: r.concordance(first, second, cluster=data["id"])},
        "renamed_joint": {"current": lambda: r.concordance(first, renamed)},
    }
    if baseline is not None:
        calls["single"]["previous"] = lambda: baseline.concordance(first)
        calls["joint"]["previous"] = lambda: baseline.concordance(first, second)
        calls["explicit_joint"]["previous"] = lambda: baseline.concordance(
            first, second, cluster=data["id"]
        )
    expected_joint = r.concordance(first, second)
    results = {}
    for name, variants in calls.items():
        expected = r.concordance(first) if name == "single" else expected_joint
        for function in variants.values():
            actual = function()
            if actual.count != expected.count:
                raise AssertionError(f"{name}: concordance counts differ")
            np.testing.assert_allclose(actual.concordance, expected.concordance, atol=1e-14)
            np.testing.assert_allclose(actual.var, expected.var, atol=1e-14)
            np.testing.assert_allclose(actual.cvar, expected.cvar, atol=1e-14)
            for _ in range(3):
                function()
        samples = {key: [] for key in variants}
        for iteration in range(args.repeats):
            order = list(variants) if iteration % 2 == 0 else list(reversed(variants))
            for key in order:
                gc.collect()
                start = time.perf_counter()
                value = variants[key]()
                samples[key].append((time.perf_counter() - start) * 1000)
                del value
        results[name] = {
            key: {
                "median_ms": statistics.median(values),
                "range_ms": [min(values), max(values)],
                "samples_ms": values,
            }
            for key, values in samples.items()
        }
    print(
        json.dumps(
            {
                "rows": args.rows,
                "repeats": args.repeats,
                "warmups": 3,
                "python": sys.version,
                "scope": "Complete Python calls; fitting, input setup and explicit GC excluded",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
