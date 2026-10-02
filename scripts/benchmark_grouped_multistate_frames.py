"""Benchmark complete grouped multi-state table conversion, excluding fitting.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_grouped_multistate_frames.py.
For a previous implementation, save its _models.py to a file and pass
--baseline-source /tmp/previous_models.py. The benchmark extracts only that
revision's grouped-table function; its helpers use the current input contract.
Actual multi-state fits provide the curve snapshots and ragged group time grids.
Every table column is checked before timing, including state and group ordering.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import r_api as r
from survival.r import _models


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def previous_function(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_grouped_survfit_frame"
    )
    namespace = dict(vars(_models))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- explicitly supplied repository revision
    return namespace[function.name]


def fitted_curves(states, groups, rows):
    labels = [f"event{i + 1}" for i in range(states)]
    data = {"time": [], "event": [], "group": []}
    for group in range(groups):
        # Different time counts exercise contiguous blocks with unequal widths.
        count = rows + group % 3
        data["time"].extend(range(1, count + 1))
        data["event"].extend(labels[i % states] for i in range(count))
        data["group"].extend([f"group{group + 1}"] * count)
    data["event"] = r._r_factor(data["event"], ["censor", *labels])
    data["group"] = r._r_factor(data["group"], [f"group{i + 1}" for i in range(groups)])
    fit = r.survfit("Surv(time,event) ~ group", data, se_fit=False)
    return r._survfit_strata_curves(fit)


def compare(actual, expected):
    if list(actual) != list(expected):
        raise AssertionError("table columns differ")
    for name, values in actual.items():
        if name in {"strata", "state"}:
            if values != expected[name]:
                raise AssertionError(f"{name} ordering differs")
        else:
            np.testing.assert_allclose(values, expected[name], rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=positive_int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--groups", type=positive_int, default=16)
    parser.add_argument("--rows", type=positive_int, default=200)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    if args.groups < 2:
        parser.error("groups must be at least two for grouped-table conversion")
    variants = {"current": r.as_data_frame}
    if args.baseline_source is not None:
        variants["previous"] = previous_function(args.baseline_source)
    results = []
    for states in args.states:
        curves = fitted_curves(states, args.groups, args.rows)
        expected = r.as_data_frame(curves)
        output_rows = len(expected["time"])
        for function in variants.values():
            compare(function(curves), expected)
        del expected
        samples = {name: [] for name in variants}
        for sample in range(args.repeats):
            names = list(variants) if sample % 2 == 0 else list(reversed(variants))
            for name in names:
                gc.collect()
                start = time.perf_counter_ns()
                output = variants[name](curves)
                samples[name].append((time.perf_counter_ns() - start) / 1_000_000)
                del output
        results.append(
            {
                "event_states": states,
                "state_columns": states + 1,
                "groups": args.groups,
                "min_times_per_group": args.rows,
                "output_rows": output_rows,
                "variants": {
                    name: {"median_ms": statistics.median(values), "samples_ms": values}
                    for name, values in samples.items()
                },
            }
        )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
