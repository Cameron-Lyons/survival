#!/usr/bin/env python3
"""Complete expected-survival calls; optional old facade and isolated peak RSS."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from survival import r_api as r  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--memory-variant", choices=["current", "previous"])
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    rng = np.random.default_rng(318)
    train = {
        "time": rng.integers(1, 1001, 2000),
        "status": rng.binomial(1, 0.7, 2000),
        "x": rng.normal(size=2000),
    }
    fit = r.coxph("Surv(time,status) ~ x", train)
    data = {
        "time": rng.uniform(50, 1000, args.rows),
        "x": rng.normal(size=args.rows),
        "group": np.arange(args.rows) % 4,
        "weight": rng.uniform(0.5, 2, args.rows),
    }
    variants = {"current": r.survexp}
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location(
            "survival.r._survexp_previous", args.baseline_source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        variants["previous"] = module.survexp
    if args.memory_variant and args.memory_variant not in variants:
        parser.error("previous memory variant requires --baseline-source")

    def call(function, method):
        return function("time ~ group", data, ratetable=fit, weights="weight", method=method)

    if args.memory_variant:
        # One fresh process per variant, including imports/training/common setup.
        call(variants[args.memory_variant], "ederer")
        print(
            json.dumps({"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        )
        return
    memory = {}
    for name in variants:
        command = [sys.executable, __file__, "--rows", str(args.rows), "--memory-variant", name]
        if args.baseline_source:
            command.extend(["--baseline-source", str(args.baseline_source)])
        memory[name] = [
            json.loads(subprocess.check_output(command, text=True))["peak_rss_mib"]  # noqa: S603 -- this script, no shell
            for _ in range(3)
        ]
    results = {}
    for method in ["ederer", "hakulinen", "conditional"]:
        expected = call(r.survexp, method)
        for function in variants.values():
            actual = call(function, method)
            for field in ["time", "surv", "n_risk"]:
                np.testing.assert_allclose(
                    getattr(actual, field), getattr(expected, field), rtol=1e-12
                )
        del actual, expected
        for _ in range(2):
            for function in variants.values():
                call(function, method)
        elapsed = {name: [] for name in variants}
        for sample in range(args.repeats):
            names = list(variants) if sample % 2 == 0 else list(reversed(variants))
            for name in names:
                gc.collect()
                started = time.perf_counter()
                output = call(variants[name], method)
                elapsed[name].append((time.perf_counter() - started) * 1000)
                del output
        results[method] = {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in elapsed.items()
        }
    print(
        json.dumps(
            {
                "python": sys.version,
                "rows": args.rows,
                "training_rows": 2000,
                "repeats": args.repeats,
                "results": results,
                "peak_rss_mib": memory,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
