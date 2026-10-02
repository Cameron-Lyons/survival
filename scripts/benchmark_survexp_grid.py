#!/usr/bin/env python3
"""Time native or formula rate-table expected-survival calls on dense output grids."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from survival import population, r  # noqa: E402


def measure(
    table, *, rows: int, unique_times: int, duplicates: int, repeats: int, api: str
) -> dict:
    positions = np.zeros((rows, 1), dtype=np.float64)
    data = {"age": positions[:, 0]}
    times = np.repeat(np.linspace(0.0, 1000.0, unique_times), duplicates)
    expected = np.exp(-1e-5 * times)
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = (
            population.survexp(table, positions, times=times)
            if api == "native"
            else r.survexp("~ 1", data, ratetable=table, times=times)
        )
        samples.append((time.perf_counter() - start) * 1000.0)
        arrays = (
            result.to_arrays()
            if api == "native"
            else {
                "time": np.asarray(result.time),
                "surv": np.asarray(result.surv)[:, None],
                "n_risk": np.asarray(result.n_risk)[:, None],
            }
        )
        np.testing.assert_array_equal(arrays["time"], times)
        np.testing.assert_allclose(arrays["surv"][:, 0], expected, rtol=5e-12, atol=1e-14)
        np.testing.assert_array_equal(arrays["n_risk"], np.full((len(times), 1), rows))
        if duplicates > 1:
            values = arrays["surv"].reshape(unique_times, duplicates)
            np.testing.assert_array_equal(values, np.repeat(values[:, :1], duplicates, axis=1))
    return {
        "rows": rows,
        "unique_times": unique_times,
        "requested_times": len(times),
        "duplicates": duplicates,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "last_survival": float(arrays["surv"][-1, 0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 4000, 16000, 32000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--api", choices=["native", "formula"], default="native")
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        parser.error("repeats must be positive and sizes must be at least two")
    table = population.RateTable([1], ["age"], [["0"]], [[0.0]], [2], [1e-5])
    results = [
        measure(
            table,
            rows=1,
            unique_times=size,
            duplicates=duplicates,
            repeats=args.repeats,
            api=args.api,
        )
        for size in args.sizes
        for duplicates in [1, 2]
    ]
    controls = [
        measure(
            table,
            rows=rows,
            unique_times=size,
            duplicates=1,
            repeats=args.repeats,
            api=args.api,
        )
        for rows, size in [(1, 25), (1, 100), (100, 25), (1000, 25)]
    ]
    print(json.dumps({"api": args.api, "dense_grids": results, "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
