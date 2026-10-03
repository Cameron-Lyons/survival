#!/usr/bin/env python3
"""Measure complete native Aalen-Johansen variance and initial-state preparation calls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from survival import _survival, surv_analysis  # noqa: E402


def measure(arguments: dict, repeats: int, check) -> dict:
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = surv_analysis.survfitaj(**arguments)
        samples.append((time.perf_counter() - start) * 1000.0)
        check(result)
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def independent_input(rows: int, weighted: bool = False) -> dict:
    arguments = {
        "time": np.arange(1, rows + 1, dtype=float),
        "state": np.resize(np.array([1, 2, 0], dtype=np.int32), rows),
        "states": ["a", "b"],
        "timefix": False,
    }
    if weighted:
        arguments["weights"] = 0.5 + (np.arange(rows) % 7) / 4.0
        # Leave weighted survival positive, so this workload measures the
        # independent sweep without the absorbed-curve compatibility fallback.
        arguments["state"][-1] = 0
    return arguments


def independent_expectation(
    state: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = len(state)
    weights = np.ones(rows) if weights is None else weights
    remaining = np.cumsum(weights[::-1])[::-1]
    observed = sorted(set(state) - {0})
    hazard_column = {destination: column for column, destination in enumerate(observed)}
    probability = np.array([1.0, 0.0, 0.0])
    hazard = np.zeros(len(observed))
    pstate = np.empty((rows, 3))
    cumhaz = np.empty((rows, len(observed)))
    for index, destination in enumerate(state):
        if destination:
            increment = weights[index] / remaining[index]
            incidence = probability[0] * increment
            probability[0] -= incidence
            probability[destination] += incidence
            hazard[hazard_column[destination]] += increment
        pstate[index] = probability
        cumhaz[index] = hazard
    risk = np.zeros((rows, 3))
    risk[:, 0] = remaining
    return pstate, cumhaz, risk


def check_independent(
    result, arguments: dict, expected: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    np.testing.assert_array_equal(result.time, arguments["time"])
    np.testing.assert_allclose(result.pstate, expected[0], rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(result.cumhaz, expected[1], rtol=2e-12, atol=1e-14)
    np.testing.assert_array_equal(result.n_risk, expected[2])
    if arguments["se_fit"]:
        np.testing.assert_equal(np.asarray(result.std_err).shape, expected[0].shape)
        np.testing.assert_equal(np.asarray(result.std_auc).shape, expected[0].shape)
        np.testing.assert_equal(np.asarray(result.std_chaz).shape, expected[1].shape)
        np.testing.assert_equal(np.isfinite(result.std_err).all(), True)
        np.testing.assert_equal(np.isfinite(result.std_auc).all(), True)
        np.testing.assert_equal(np.isfinite(result.std_chaz).all(), True)
    else:
        for name in ["std_err", "std_chaz", "std_auc"]:
            np.testing.assert_equal(getattr(result, name), None)


def variance_case(rows: int, se_fit: bool, repeats: int, weighted: bool = False) -> dict:
    arguments = {**independent_input(rows, weighted), "se_fit": se_fit}
    expected = independent_expectation(arguments["state"], arguments.get("weights"))
    result = measure(arguments, repeats, lambda fit: check_independent(fit, arguments, expected))
    return {"rows": rows, "se_fit": se_fit, "weighted": weighted, **result}


def preparation_case(rows: int, fixed_p0: bool, counting: bool, repeats: int) -> dict:
    arguments = {
        "time": np.tile([1.0, 2.0], rows // 2),
        "state": np.ones(rows, dtype=np.int32),
        "states": ["dead"],
        "istate": ["a", "b"] * (rows // 2),
        "strata": np.repeat(np.arange(rows // 2, dtype=np.int32), 2),
        "p0": [0.5, 0.5, 0.0] if fixed_p0 else None,
        "se_fit": False,
        "timefix": False,
    }
    if counting:
        arguments["start"] = np.zeros(rows)
        arguments["id"] = list(range(rows))
    pstate = np.tile([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], (rows // 2, 1))
    hazard = np.tile([[1.0, 0.0], [1.0, 1.0]], (rows // 2, 1))
    risk = np.tile([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], (rows // 2, 1))

    def check(fit) -> None:
        np.testing.assert_array_equal(fit.time, arguments["time"])
        np.testing.assert_array_equal(fit.pstate, pstate)
        np.testing.assert_array_equal(fit.cumhaz, hazard)
        np.testing.assert_array_equal(fit.n_risk, risk)
        np.testing.assert_array_equal(fit.p0, [[0.5, 0.5, 0.0]] * (rows // 2))

    result = measure(arguments, repeats, check)
    return {
        "rows": rows,
        "curves": rows // 2,
        "fixed_p0": fixed_p0,
        "counting": counting,
        **result,
    }


def variance_control(rows: int, clusters: int | None, repeats: int, weighted: bool = False) -> dict:
    arguments = {**independent_input(rows, weighted), "se_fit": True}
    if clusters is not None:
        arguments["cluster"] = (np.arange(rows) % clusters).tolist()
    # Asking for influence retains the general per-cluster reference kernel.
    reference = surv_analysis.survfitaj(**arguments, influence=True)
    expected = independent_expectation(arguments["state"], arguments.get("weights"))

    def check(fit) -> None:
        check_independent(fit, arguments, expected)
        for name in ["std_err", "std_chaz", "std_auc"]:
            np.testing.assert_allclose(getattr(fit, name), getattr(reference, name), rtol=2e-12)

    result = measure(arguments, repeats, check)
    return {"rows": rows, "clusters": clusters or rows, "weighted": weighted, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 4000, 8000])
    parser.add_argument("--prep-sizes", type=int, nargs="+", default=[1000, 4000, 8000, 16000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--weighted-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        parser.error("repeats must be positive and sizes must be at least two")
    if any(size < 2 or size % 2 for size in args.prep_sizes):
        parser.error("prep-sizes must be even and at least two")
    result = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "extension_sha256": hashlib.sha256(Path(_survival.__file__).read_bytes()).hexdigest(),
            "repeats": args.repeats,
        },
        "variance": [
            variance_case(rows, se_fit, args.repeats, weighted)
            for rows in args.sizes
            for weighted in ([True] if args.weighted_only else [False, True])
            for se_fit in [False, True]
        ],
        "preparation": []
        if args.weighted_only
        else [
            preparation_case(rows, fixed_p0, counting, args.repeats)
            for rows in args.prep_sizes
            for fixed_p0 in [False, True]
            for counting in [False, True]
        ],
        "controls": (
            [variance_control(100, None, args.repeats, True)]
            if args.weighted_only
            else [
                variance_control(100, None, args.repeats),
                variance_control(1000, 50, args.repeats),
                variance_control(100, None, args.repeats, True),
            ]
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
