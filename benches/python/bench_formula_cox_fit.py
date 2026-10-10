#!/usr/bin/env python3
"""Measure complete formula and native Cox fits against a saved package.

Select the package with PYTHONPATH and run each build in a separate process.
Input creation, warmup and post-fit snapshots are excluded from the timings.
Use --compare to require exactly matching fitted and post-fit numerical outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival, coxph


def _calls(rows, columns):
    rng = np.random.default_rng(71293 + columns)
    x = rng.normal(size=(rows, columns))
    time_values = rng.integers(1, 501, rows).astype(float)
    status = (np.arange(rows) % 3 != 0).astype(np.int32)
    data = {"time": time_values, "status": status}
    data.update({f"x{i}": x[:, i] for i in range(columns)})
    formula = "Surv(time, status) ~ " + " + ".join(f"x{i}" for i in range(columns))
    interaction_formula = formula + " + x0:x1 + x1:x2 + x0:x1:x2"
    interactions = np.column_stack(
        (x, x[:, 0] * x[:, 1], x[:, 1] * x[:, 2], (x[:, 0] * x[:, 1]) * x[:, 2])
    )
    return {
        "formula_numeric": lambda: coxph(formula, data).fit,
        "formula_interactions": lambda: coxph(interaction_formula, data).fit,
        "native_numeric": lambda: _survival.coxph_fit(time_values, status, x),
        "native_interactions": lambda: _survival.coxph_fit(time_values, status, interactions),
    }


def _snapshot(fit):
    fields = (
        "coefficients",
        "var",
        "loglik",
        "score",
        "wald_test",
        "iter",
        "flag",
        "means",
        "linear_predictors",
        "residuals",
        "first",
        "x",
    )
    snapshot = {field: np.asarray(getattr(fit, field)) for field in fields}
    for kind in ("lp", "risk", "expected"):
        result = fit.predict(kind, se_fit=True)
        snapshot[f"predict_{kind}"] = np.asarray(result.fit)
        snapshot[f"predict_{kind}_se"] = np.asarray(result.se_fit)
    terms = fit.predict_terms(se_fit=True)
    snapshot["predict_terms"] = np.asarray(terms.fit)
    snapshot["predict_terms_se"] = np.asarray(terms.se_fit)
    snapshot["predict_terms_constant"] = np.asarray(terms.constant)
    hazard = fit.basehaz()
    for field in ("hazard", "time", "strata"):
        value = getattr(hazard, field)
        snapshot[f"basehaz_{field}"] = np.asarray(value if value is not None else [])
    snapshot["basehaz_strata_present"] = np.asarray(hazard.strata is not None)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--columns", type=int, nargs="+", default=[3, 16])
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True, help="output filename prefix")
    parser.add_argument("--compare", type=Path, help="NPZ snapshots from another build")
    args = parser.parse_args()
    if args.rows <= 0 or args.repeat <= 0 or any(p < 3 for p in args.columns):
        parser.error("rows and repeat must be positive and columns must be at least three")
    if len(set(args.columns)) != len(args.columns):
        parser.error("columns must be distinct")
    measurements, snapshots = [], {}
    for columns in args.columns:
        calls = _calls(args.rows, columns)
        for name, function in calls.items():
            for _ in range(3):
                fit = function()
            for field, value in _snapshot(fit).items():
                snapshots[f"{columns}_{name}_{field}"] = value
            del fit
        samples = {name: [] for name in calls}
        for repeat in range(args.repeat):
            ordered = list(calls.items())
            if repeat % 2:
                ordered.reverse()
            for name, function in ordered:
                start = time.perf_counter()
                fit = function()
                samples[name].append((time.perf_counter() - start) * 1000)
                del fit
        measurements.append(
            {
                "rows": args.rows,
                "columns": columns,
                "milliseconds": {
                    name: {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "samples": values,
                    }
                    for name, values in samples.items()
                },
            }
        )
    if args.compare:
        with np.load(args.compare) as reference:
            if set(reference.files) != set(snapshots):
                raise ValueError("snapshots must cover the same columns, routines and fields")
            for key, value in snapshots.items():
                np.testing.assert_array_equal(value, reference[key], err_msg=key)
    np.savez(str(args.output) + ".npz", **snapshots)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(str(args.output) + ".json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "extension": _survival.__file__,
                "samples": args.repeat,
                "warmups": 3,
                "scope": (
                    "complete Cox fits; input creation, result destruction "
                    "and post-fit snapshots excluded"
                ),
                "peak_process_rss_mib": rss / (1024**2 if platform.system() == "Darwin" else 1024),
                "results": measurements,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
