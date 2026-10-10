#!/usr/bin/env python3
"""Complete survival-response repetition calls and traced Python allocations.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_surv_repetition.py.
Use --baseline-source /tmp/previous-surv-vector.py for a saved _surv_vector.py.
Response/control creation is excluded; conversion, repetition and construction
of the returned response are included. NumPy object-array allocations are traced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from survival import r


def _baseline(path: Path) -> Any:
    name = "survival.r._repetition_benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.rep_surv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("rows must be at least two, repeat positive and warmup nonnegative")
    functions = {"after": r.rep_surv}
    if args.baseline_source is not None:
        functions["before"] = _baseline(args.baseline_source)
    starts = np.arange(args.rows, dtype=float)
    response = r.Surv(starts, starts + 1, np.arange(args.rows) % 2)
    cases = {
        "scalar-times": {"times": 3},
        "scalar-each": {"each": 3},
        "vector-times": {"times": (np.arange(args.rows) % 3).tolist()},
        "each-vector-times": {"each": 2, "times": (np.arange(args.rows * 2) % 3).tolist()},
        "length-cycle": {"each": 3, "length_out": args.rows * 7 + 1},
        "length-prefix": {"each": 3, "length_out": 5},
    }
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "input_rows": args.rows,
        "repeat": args.repeat,
        "cases": {},
    }
    for name, options in cases.items():
        results = {key: function(response, **options) for key, function in functions.items()}
        if any(not result.equals(results["after"]) for result in results.values()):
            raise AssertionError(f"baseline differs for {name}")
        output_rows = len(results["after"])
        del results
        for function in functions.values():
            for _ in range(args.warmup):
                function(response, **options)
        samples: dict[str, list[float]] = {key: [] for key in functions}
        for iteration in range(args.repeat):
            order = list(functions)
            if iteration % 2:
                order.reverse()
            for key in order:
                start = time.perf_counter()
                functions[key](response, **options)
                samples[key].append((time.perf_counter() - start) * 1000)
        case: dict[str, Any] = {"output_rows": output_rows}
        for key, function in functions.items():
            tracemalloc.start()
            result = function(response, **options)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del result
            case[key] = {
                "median_ms": statistics.median(samples[key]),
                "min_ms": min(samples[key]),
                "max_ms": max(samples[key]),
                "peak_traced_bytes": peak,
            }
        report["cases"][name] = case
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
