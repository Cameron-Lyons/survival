#!/usr/bin/env python3
"""Measure complete Turnbull fits and compare every curve field to a saved build.

Run --extension /path/to/previous/_survival.so --output before, then
--compare before.npz --output after. Input creation is outside the timings.
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


def inputs(rows, overlapping):
    i = np.arange(rows)
    lower = 2.0 * i
    upper = lower + (11.0 if overlapping else 1.0)
    status = np.full(rows, 3, dtype=int)
    if overlapping:
        status[i % 13 == 0] = 1
        status[i % 17 == 0] = 0
        status[i % 19 == 0] = 2
    return lower.tolist(), upper.tolist(), status.tolist(), (0.5 + (i % 7) * 0.25).tolist()


def snapshot(result):
    curve = result.curves[0]
    return {
        field: np.asarray(getattr(curve, field))
        for field in (
            "time",
            "n_risk",
            "n_event",
            "n_censor",
            "surv",
            "std_err",
            "lower",
            "upper",
            "iterations",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[100, 1000, 5000])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or any(n < 2 for n in args.rows):
        parser.error("repeat must be positive and row counts must be at least two")
    core = extension(args.extension)
    measurements, arrays = [], {}
    for rows in args.rows:
        for name, overlapping in (("disjoint", False), ("overlapping", True)):
            prepared = inputs(rows, overlapping)
            samples = []
            for repeat in range(args.repeat + 2):
                start = time.perf_counter()
                result = core.turnbull(*prepared, timefix=False)
                elapsed = (time.perf_counter() - start) * 1000
                if repeat >= 2:
                    samples.append(elapsed)
                if repeat == args.repeat + 1:
                    arrays.update({f"{rows}_{name}_{k}": v for k, v in snapshot(result).items()})
                del result
            measurements.append(
                {
                    "rows": rows,
                    "case": name,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
    if args.compare:
        with np.load(args.compare) as reference:
            if set(reference.files) != set(arrays):
                raise ValueError("comparison must cover the same cases, row counts and outputs")
            for key, value in arrays.items():
                np.testing.assert_allclose(
                    value, reference[key], rtol=1e-12, atol=1e-12, err_msg=key
                )
    np.savez(str(args.output) + ".npz", **arrays)
    Path(str(args.output) + ".json").write_text(
        json.dumps({"extension": core.__file__, "warmups": 2, "results": measurements}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
