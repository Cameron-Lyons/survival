#!/usr/bin/env python3
"""Measure complete concordance matrix and formula calls on a release build.

Input construction, result checks and warmups are excluded from measurements.
Numeric array layouts must return exactly the same outputs as row lists.
--compare verifies complete result fingerprints before comparing timings.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from survival import _survival, r


def _fingerprint(result):
    return hashlib.sha256(json.dumps(asdict(result), sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--columns", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.rows < 30 or args.repeat <= 0 or any(p <= 0 for p in args.columns):
        parser.error("at least 30 rows and positive repeat and column counts are required")
    baseline = None if args.compare is None else json.loads(args.compare.read_text())
    if baseline is not None and (baseline["rows"], baseline["columns"]) != (
        args.rows,
        args.columns,
    ):
        parser.error("comparison rows and columns must match this run")

    row = np.arange(args.rows)
    response = r.Surv(1.0 + ((row * 4999) % args.rows) // 4, row % 4 != 0)
    weights = 0.5 + (row % 7) / 4
    results = {}
    for columns in args.columns:
        values = np.column_stack(
            [((row * (37 + j * 8) + j * 11) % 127).astype(float) for j in range(columns)]
        )
        inputs = {
            "lists": values.tolist(),
            "contiguous": values,
            "fortran": np.asfortranarray(values),
            "strided": np.repeat(values, 2, axis=0)[::2],
        }
        data = {
            "time": np.asarray(response.time),
            "status": np.asarray(response.event),
            **{f"x{column + 1}": values[:, column] for column in range(columns)},
        }
        formula = "Surv(time, status) ~ " + " + ".join(f"x{i + 1}" for i in range(columns))
        for std_err in (False, True):
            calls = {
                name: partial(r.concordancefit, response, matrix, weights=weights, std_err=std_err)
                for name, matrix in inputs.items()
            }
            if std_err:
                calls["formula"] = partial(r.concordance, formula, data=data, weights=weights)
            fingerprints = {}
            samples = {name: [] for name in calls}
            for name, function in calls.items():
                for _ in range(3):
                    result = function()
                fingerprints[name] = _fingerprint(result)
            if len({fingerprints[name] for name in inputs}) != 1:
                raise AssertionError("array and row-list results differ")
            for repeat in range(args.repeat):
                order = list(calls.items())
                if repeat % 2:
                    order.reverse()
                for name, function in order:
                    del result
                    begin = time.perf_counter()
                    result = function()
                    samples[name].append((time.perf_counter() - begin) * 1000)
            if _fingerprint(result) != fingerprints[name]:
                raise AssertionError("repeated call changed its result")
            for name, timings in samples.items():
                key = f"columns={columns},std_err={std_err},layout={name}"
                case = {
                    "median_ms": statistics.median(timings),
                    "times_ms": timings,
                    "result_sha256": fingerprints[name],
                }
                if baseline is not None:
                    previous = baseline["results"][key]
                    if case["result_sha256"] != previous["result_sha256"]:
                        raise AssertionError(f"{key}: result differs from baseline")
                    case["previous_median_ms"] = previous["median_ms"]
                    case["speedup"] = previous["median_ms"] / case["median_ms"]
                results[key] = case
    extension = Path(_survival.__file__)
    facade = Path(inspect.getfile(r.concordancefit))
    report = {
        "rows": args.rows,
        "columns": args.columns,
        "repeats": args.repeat,
        "warmups": 3,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "facade_path": str(facade),
        "facade_sha256": hashlib.sha256(facade.read_bytes()).hexdigest(),
        "scope": "complete calls; input construction and result disposal excluded",
        "results": results,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
