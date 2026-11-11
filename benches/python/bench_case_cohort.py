#!/usr/bin/env python3
"""Compare complete case-cohort fits with the inputs used by the Rust benchmarks.

Use --extension to load a saved release in a separate process. Inputs are prepared
outside the timed region. --compare verifies fitted coefficients, covariance,
phase-two variance, residuals and Borgan score contributions before saving results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np


def extension(path):
    if path is None:
        from survival import _survival

        return _survival
    spec = importlib.util.spec_from_file_location("_survival", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calls(core, rows):
    i = np.arange(rows)
    j = np.arange(4)
    stop = (i + 1.0) * 0.25
    status = (i % 4 != 0).astype(np.int32)
    subcohort = (i % 5 != 0).astype(np.int32)
    status[subcohort == 0] = 1
    x = (i[:, None] % 17 - 8) * 0.03 + (j % 5 - 2) * 0.1 + (i[:, None] * (j + 3) % 11) * 0.01
    ids = i.tolist()
    strata = ((i // 4) % 2).tolist()
    return {
        "Prentice": lambda: core.cch_fit(stop, status, x, subcohort, ids, rows * 4),
        "I.Borgan": lambda: core.cch_borgan_fit(
            stop, status, x, subcohort, ids, strata, [rows * 2, rows * 2]
        ),
    }


def snapshot(result):
    values = {
        name: np.asarray(getattr(result, name))
        for name in ("coefficients", "var", "naive_var", "phase2var")
    }
    if result.sc is not None:
        values["sc"] = np.asarray(result.sc)
    values["martingale"] = np.asarray(result.fit.residuals)
    values["loglik"] = np.asarray(result.fit.loglik)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[100, 1000, 5000])
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or any(n < 4 for n in args.rows):
        parser.error("repeat must be positive and each row count must be at least four")
    core = extension(args.extension)
    measurements, arrays = [], {}
    for rows in args.rows:
        for name, function in calls(core, rows).items():
            samples = []
            for repeat in range(args.repeat + 3):
                start = time.perf_counter()
                result = function()
                elapsed = (time.perf_counter() - start) * 1000
                if repeat >= 3:
                    samples.append(elapsed)
                if repeat == args.repeat + 2:
                    arrays.update({f"{rows}_{name}_{k}": v for k, v in snapshot(result).items()})
                del result
            measurements.append(
                {
                    "rows": rows,
                    "method": name,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
    if args.compare:
        with np.load(args.compare) as reference:
            if set(reference.files) != set(arrays):
                raise ValueError("comparison must cover the same methods, row counts and outputs")
            for key, value in arrays.items():
                np.testing.assert_allclose(
                    value, reference[key], rtol=1e-10, atol=1e-11, err_msg=key
                )
    np.savez(str(args.output) + ".npz", **arrays)
    Path(str(args.output) + ".json").write_text(
        json.dumps(
            {
                "extension": core.__file__,
                "repeats": args.repeat,
                "warmups": 3,
                "results": measurements,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
