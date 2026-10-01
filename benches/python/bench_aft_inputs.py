#!/usr/bin/env python3
"""Measure AFT construction and fitting calls, including input validation.

Run a release build with PYTHONPATH=python. Use --extension to load a saved
previous build in a separate process, then --compare to verify numerical outputs.
Random input creation and copying result properties are excluded from timings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
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
    rng = np.random.default_rng(1973 + columns)
    x = np.asarray(rng.normal(size=(rows, columns)), order=order)
    x[:, 0] = 1
    y = x @ np.linspace(0.1, 0.6, columns) + rng.normal(size=rows)
    status = rng.binomial(1, 0.7, rows).astype(np.int32)
    weights = rng.uniform(0.5, 2, rows)
    offset = rng.normal(0, 0.1, rows)

    def construct():
        return core.SurvregData(y, status, x, weights=weights, offset=offset)

    data = construct()
    distribution = core.SurvregDistribution("gaussian")
    penalized = {
        "penalties": [core.CoxPenalty.ridge(theta=1, scale=False)],
        "pcols": [list(range(1, columns))],
        "assign": [[0], list(range(1, columns))],
    }
    calls = {"construct": construct}
    for name in ("survreg_fit", "survreg_fit_raw", "survpenal_fit", "survpenal_fit_raw"):
        function = getattr(core, name)
        options = penalized if "survpenal" in name else {}
        calls[name] = lambda fn=function, kw=options: fn(data, distribution, **kw)
        calls[f"construct+{name}"] = lambda fn=function, kw=options: fn(
            construct(), distribution, **kw
        )
    return calls


def _snapshot(fit, name):
    result = {
        field: np.asarray(getattr(fit, field))
        for field in ("coefficients", "icoef", "linear_predictors", "score")
    }
    if name.endswith("survreg_fit"):
        result["var"] = np.asarray(fit.variance_matrix)
        result["loglik"] = np.asarray([fit.intercept_only_log_likelihood, fit.log_likelihood])
        result["iter"] = np.asarray(fit.iterations)
    else:
        for field in ("var", "loglik", "iter"):
            result[field] = np.asarray(getattr(fit, field))
    if "survpenal" in name:
        for field in ("var2", "df", "penalty"):
            result[field] = np.asarray(getattr(fit, field))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--columns", type=int, nargs="+", default=[3, 16])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--order", choices=["C", "F"], default="C")
    parser.add_argument("--output", type=Path, required=True, help="output filename prefix")
    parser.add_argument("--compare", type=Path, help="NPZ outputs from another build")
    parser.add_argument("--extension", type=Path, help="saved native extension to load instead")
    args = parser.parse_args()
    if args.rows <= 0 or args.repeat <= 0 or any(p < 2 for p in args.columns):
        parser.error("rows and repeat must be positive; columns must include intercept and slope")
    core = _extension(args.extension)
    measurements, arrays = [], {}
    for columns in args.columns:
        calls = _calls(core, args.rows, columns, args.order)
        for name, function in calls.items():
            for _ in range(2):
                fit = function()
            if name != "construct":
                for field, value in _snapshot(fit, name).items():
                    arrays[f"{columns}_{name}_{field}"] = value
            del fit
        samples = {name: [] for name in calls}
        for repeat in range(args.repeat):
            ordered = list(calls.items())
            if repeat % 2:
                ordered.reverse()
            for name, function in ordered:
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
    Path(str(args.output) + ".json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "extension": core.__file__,
                "samples": args.repeat,
                "warmups": 2,
                "scope": "complete native calls; random input creation and result getters excluded",
                "results": measurements,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
