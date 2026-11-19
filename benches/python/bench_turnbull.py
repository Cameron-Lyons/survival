#!/usr/bin/env python3
"""Measure complete Turnbull fits and compare estimates and counts to a saved build.

Run --extension /path/to/previous/_survival.so --output before, then
--compare before.npz --output after. Input creation is outside the timings.
Use --cases disjoint overlapping to compare with older two-case snapshots.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

CASES = ("disjoint", "overlapping", "full_span", "wide", "mixed_width")


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


def inputs(rows, case):
    if case in {"full_span", "wide", "mixed_width"}:
        return anchored_inputs(rows, case)
    i = np.arange(rows)
    lower = 2.0 * i
    overlapping = case == "overlapping"
    upper = lower + (11.0 if overlapping else 1.0)
    status = np.full(rows, 3, dtype=int)
    if overlapping:
        status[i % 13 == 0] = 1
        status[i % 17 == 0] = 0
        status[i % 19 == 0] = 2
    return lower.tolist(), upper.tolist(), status.tolist(), (0.5 + (i % 7) * 0.25).tolist()


def anchored_inputs(rows, case):
    jumps = rows // 2
    times = np.arange(1, jumps + 1)
    intervals = rows - jumps
    rng = np.random.default_rng(8134)
    if case == "full_span":
        start = np.zeros(intervals, dtype=int)
        width = np.full(intervals, jumps)
        weights = np.full(rows, 0.75)
    else:
        width = rng.integers(max(1, jumps // 4), max(1, jumps * 3 // 4) + 1, size=intervals)
        if case == "mixed_width":
            # Many inspection intervals cover only a handful of exact events.
            for part, bound in [(width[::3], 8), (width[1::3], int(np.sqrt(jumps)))]:
                part[:] = rng.integers(1, min(jumps, bound) + 1, size=part.size)
        start = rng.integers(0, jumps - width + 1)
        weights = 0.75 + (np.arange(rows) % 7) / 16
    lower = np.concatenate((times, start)).astype(float)
    upper = np.concatenate((times, start + width)).astype(float)
    status = np.repeat([1, 3], [jumps, intervals])
    return lower.tolist(), upper.tolist(), status.tolist(), weights.tolist()


def check_full_span(curve, rows):
    # The intervals cover the whole grid and add a constant to the log
    # likelihood. Equal-weight exact events therefore put equal mass at each jump.
    jumps = rows // 2
    time = np.arange(1, jumps + 1)
    events = 0.75 * rows / jumps
    np.testing.assert_array_equal(curve["time"], time)
    np.testing.assert_array_equal(curve["n_censor"], np.zeros(jumps))
    np.testing.assert_allclose(curve["surv"], (jumps - time) / jumps, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(curve["n_event"], events, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(curve["n_risk"], (jumps - time + 1) * events, rtol=1e-11)


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
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
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
        for name in args.cases:
            prepared = inputs(rows, name)
            samples = []
            for repeat in range(args.repeat + 2):
                start = time.perf_counter()
                result = core.turnbull(*prepared, timefix=False)
                elapsed = (time.perf_counter() - start) * 1000
                if repeat >= 2:
                    samples.append(elapsed)
                if repeat == args.repeat + 1:
                    curve = snapshot(result)
                    if name == "full_span":
                        check_full_span(curve, rows)
                    arrays.update({f"{rows}_{name}_{k}": v for k, v in curve.items()})
                del result
            measurements.append(
                {
                    "rows": rows,
                    "case": name,
                    "times": len(curve["time"]),
                    "iterations": int(curve["iterations"]),
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
    if args.compare:
        with np.load(args.compare) as reference:
            if set(reference.files) != set(arrays):
                raise ValueError("comparison must cover the same cases, row counts and outputs")
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, reference[key], err_msg=key)
    np.savez(str(args.output) + ".npz", **arrays)
    Path(str(args.output) + ".json").write_text(
        json.dumps(
            {
                "extension": core.__file__,
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
                "warmups": 2,
                "results": measurements,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
