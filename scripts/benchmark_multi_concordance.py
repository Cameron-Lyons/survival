#!/usr/bin/env python3
"""Measure native concordance calls with one or several predictor columns.

Input construction, three warmups and disposal of the previous result are
outside the measurements. Argument handling, the Rust fit and result creation
are included. A saved baseline verifies numerical equivalence before reporting
the before/after medians; separate process runs do not alternate versions.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival, core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.rows < 30 or args.columns < 2 or args.repeats < 1:
        parser.error("at least 30 rows, two columns and one repeat are required")
    baseline = None if args.baseline is None else json.loads(args.baseline.read_text())
    if baseline is not None and (baseline["rows"], baseline["columns"]) != (
        args.rows,
        args.columns,
    ):
        parser.error("baseline rows and columns must match this run")

    n = args.rows
    # Unsorted outcomes with event-time and predictor ties; fractional weights.
    times = [1.0 + ((i * 4999) % n) // 4 for i in range(n)]
    status = [int(i % 4 != 0) for i in range(n)]
    weights = core.Weights([0.5 + (i % 7) / 4 for i in range(n)])
    data = core.SurvivalData(times, status)
    results = {}
    for columns in (1, args.columns):
        matrix = core.CovariateMatrix(
            [float((i * (37 + j * 8) + j * 11) % 127) for i in range(n) for j in range(columns)],
            n,
            columns,
        )
        for std_err in (False, True):
            for timewt in ("n", "S"):
                for _ in range(3):
                    result = core.concordancefit(
                        data, matrix, weights=weights, timewt=timewt, std_err=std_err
                    )
                measured = []
                for _ in range(args.repeats):
                    del result
                    begin = time.perf_counter()
                    result = core.concordancefit(
                        data, matrix, weights=weights, timewt=timewt, std_err=std_err
                    )
                    measured.append((time.perf_counter() - begin) * 1000)
                key = f"columns={columns},std_err={std_err},timewt={timewt}"
                case = {
                    "median_ms": statistics.median(measured),
                    "times_ms": measured,
                    "concordance": result.concordance,
                    "var": result.var,
                    "cvar": result.cvar,
                }
                if baseline is not None:
                    previous = baseline["results"][key]
                    for field in ("concordance", "var", "cvar"):
                        if case[field] is None:
                            if previous[field] is not None:
                                raise AssertionError(f"{key}: {field} differs")
                        else:
                            np.testing.assert_allclose(
                                case[field], previous[field], rtol=1e-12, atol=1e-14
                            )
                    case["previous_median_ms"] = previous["median_ms"]
                    case["speedup"] = previous["median_ms"] / case["median_ms"]
                results[key] = case
    extension = Path(_survival.__file__)
    report = {
        "rows": n,
        "columns": args.columns,
        "repeats": args.repeats,
        "warmups": 3,
        "extension_path": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "scope": "Native Python calls; input construction and result disposal excluded",
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
