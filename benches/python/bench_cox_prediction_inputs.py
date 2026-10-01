#!/usr/bin/env python3
"""Measure complete native Cox prediction calls on a reusable population matrix.

Run on a release build with PYTHONPATH=python. Use --extension to compare a saved
previous extension in a separate process. Fitting, input construction, and baseline
cache warmup are excluded; buffer conversion, validation, and output materialization
are timed. Output consists of JSON measurements and NPZ arrays for cross-build checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np


def _extension(path):
    if path is None:
        from survival import _survival

        return _survival
    spec = importlib.util.spec_from_file_location("_survival", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls(core, rows, columns, order):
    rng = np.random.default_rng(918 + columns)
    n = 1000
    time_values = rng.integers(1, 101, size=n).astype(float)
    status = (np.arange(n) % 3 != 0).astype(np.int32)
    fit = core.coxph_fit(
        time_values,
        status,
        rng.normal(size=(n, columns)),
        strata=np.where(np.arange(n) % 2, 17, -3),
        weights=rng.uniform(0.5, 2, size=n),
        offset=rng.normal(0, 0.1, n),
    )
    x = np.asarray(rng.normal(size=(rows, columns)), order=order)
    strata = np.where(np.arange(rows) % 2, 17, -3)
    offset = rng.normal(0, 0.1, rows)
    followup = rng.integers(1, 101, size=rows).astype(float)
    query = np.array([10.0, 30.0, 60.0, 90.0])
    groups = (np.arange(rows) % 3).astype(np.int32)
    weights = np.ones(rows)
    return {
        "survival_at": lambda: fit.predict_survival_at(
            query, newdata=x, new_strata=strata, new_offset=offset
        ),
        "expected": lambda: np.asarray(
            fit.predict(
                "expected", newdata=x, new_strata=strata, new_offset=offset, new_time=followup
            ).fit
        ),
        "cohort": lambda: np.asarray(
            fit.expected_survival(
                x, groups, weights, new_strata=strata, new_offset=offset, times=query
            ).surv
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--columns", type=int, nargs="+", default=[3, 32])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--order", choices=["C", "F"], default="C")
    parser.add_argument("--output", type=Path, required=True, help="output filename prefix")
    parser.add_argument("--compare", type=Path, help="NPZ outputs from another build")
    parser.add_argument("--extension", type=Path, help="saved native extension to load instead")
    args = parser.parse_args()
    if args.rows <= 0 or args.repeat <= 0 or any(p <= 0 for p in args.columns):
        parser.error("rows, repeat, and columns must be positive")
    core = _extension(args.extension)
    measurements, arrays = [], {}
    for columns in args.columns:
        calls = _calls(core, args.rows, columns, args.order)
        for name, function in calls.items():
            arrays[f"{columns}_{name}"] = function()
            for _ in range(2):
                function()
        samples = {name: [] for name in calls}
        for repeat in range(args.repeat):
            order = list(calls.items())
            if repeat % 2:
                order.reverse()
            for name, function in order:
                start = time.perf_counter()
                function()
                samples[name].append((time.perf_counter() - start) * 1000)
        measurements.append(
            {
                "rows": args.rows,
                "columns": columns,
                "order": args.order,
                "milliseconds": {
                    name: {"median": statistics.median(v), "min": min(v), "max": max(v)}
                    for name, v in samples.items()
                },
            }
        )
    if args.compare:
        with np.load(args.compare) as reference:
            if set(reference.files) != set(arrays):
                raise ValueError("comparison arrays must cover the same columns and routines")
            for key, value in arrays.items():
                np.testing.assert_allclose(value, reference[key], rtol=5e-14, atol=1e-15)
    np.savez(str(args.output) + ".npz", **arrays)
    # ru_maxrss is bytes on macOS, KiB on Linux. This includes interpreter,
    # NumPy, training data, input buffers, caches, temporary storage and results.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024**2 if platform.system() == "Darwin" else 1024)
    Path(str(args.output) + ".json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "extension": core.__file__,
                "samples": args.repeat,
                "warmups": 3,
                "scope": "complete native Python calls; fitting and input creation excluded",
                "peak_process_rss_mib": rss_mib,
                "results": measurements,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
