#!/usr/bin/env python3
"""Measure complete AFT formula preparation and fits on a release extension.

Use --row-design-reference to exercise the original row-list preparation, then
--compare to verify identical outputs with the array path on the same extension.
Input creation, warmups and output-property reads are excluded from timings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
from survival import _survival, r

aft = importlib.import_module("survival.r._survreg")
formula_helpers = importlib.import_module("survival.r._formula")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--columns", type=int, nargs="+", default=[3, 8])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--row-design-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.rows < 30 or args.repeat < 1 or any(p < 1 for p in args.columns):
        parser.error("at least 30 rows and positive columns/repeat are required")
    if args.row_design_reference:
        aft._design_array_from_spec = formula_helpers._design_rows_from_spec
    reports, outputs = [], {}
    for columns in args.columns:
        rng = np.random.default_rng(3821 + columns)
        x = rng.normal(size=(args.rows, columns))
        event = rng.exponential(size=args.rows) / np.exp(x @ np.linspace(-0.3, 0.3, columns))
        censor = rng.exponential(scale=2.0, size=args.rows)
        data = {f"x{i + 1}": x[:, i] for i in range(columns)}
        data.update(time=np.minimum(event, censor) + 0.01, status=(event <= censor).astype(int))
        formula = "Surv(time, status) ~ " + " + ".join(f"x{i + 1}" for i in range(columns))

        calls = {
            "prepare": partial(
                aft._formula_frame,
                formula,
                data,
                weights=None,
                subset=None,
                na_action="omit",
                offset=None,
                cluster=None,
                keep_model=False,
            ),
            "fit": partial(r.survreg, formula, data),
        }
        for name, call in calls.items():
            for _ in range(3):
                result = call()
            if name == "prepare":
                outputs[f"{columns}_design"] = np.asarray(result.x)
            else:
                for field in ("coefficients", "var", "loglik", "linear_predictors", "scale"):
                    outputs[f"{columns}_{field}"] = np.asarray(getattr(result, field))
                outputs[f"{columns}_residuals"] = np.asarray(r.residuals(result))
                outputs[f"{columns}_prediction"] = np.asarray(r.predict(result))
        samples = {name: [] for name in calls}
        for repeat in range(args.repeat):
            order = list(calls.items())
            if repeat % 2:
                order.reverse()
            for name, call in order:
                del result
                started = time.perf_counter()
                result = call()
                samples[name].append((time.perf_counter() - started) * 1000)
                if result is None:
                    raise AssertionError(f"{name} returned no result")
        reports.append(
            {
                "rows": args.rows,
                "predictors": columns,
                "milliseconds": {
                    name: {"median": statistics.median(values), "samples": values}
                    for name, values in samples.items()
                },
            }
        )
    if args.compare is not None:
        with np.load(args.compare) as expected:
            if set(expected.files) != set(outputs):
                raise ValueError("reference must cover the same predictor counts and fields")
            for name, values in outputs.items():
                np.testing.assert_array_equal(values, expected[name], err_msg=name)
    np.savez(str(args.output) + ".npz", **outputs)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "extension": _survival.__file__,
        "extension_sha256": hashlib.sha256(Path(_survival.__file__).read_bytes()).hexdigest(),
        "row_design_reference": args.row_design_reference,
        "warmups": 3,
        "repeats": args.repeat,
        "scope": "complete calls; prebuilt inputs and previous result disposal excluded",
        "results": reports,
    }
    Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
