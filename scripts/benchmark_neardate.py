#!/usr/bin/env python3
"""Complete neardate calls for numeric array/list inputs, with optional source comparison.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_neardate.py.
Use --baseline-source /tmp/previous-data-prep.py for a saved _data_prep.py.
Input creation is excluded; conversion, native matching and output assembly are timed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from survival import r


def _baseline(path: Path) -> Any:
    name = "survival.r._neardate_benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.neardate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("rows must be at least two, repeat positive and warmup nonnegative")
    functions = {"after": r.neardate}
    if args.baseline_source is not None:
        functions["before"] = _baseline(args.baseline_source)
    rng = np.random.default_rng(123)
    ids = rng.integers(0, 100, args.rows).tolist()
    query = rng.uniform(0, 10_000, args.rows)
    reference = rng.uniform(0, 10_000, args.rows)
    missing_query = query.tolist()
    missing_reference = reference.tolist()
    missing_query[::101] = [None] * len(missing_query[::101])
    missing_reference[::103] = [None] * len(missing_reference[::103])
    inputs = {
        "array": (query, reference),
        "list": (query.tolist(), reference.tolist()),
        "missing-list": (missing_query, missing_reference),
    }
    report: dict[str, Any] = {"rows": args.rows, "repeat": args.repeat, "cases": {}}
    for layout, dates in inputs.items():
        for best in ("after", "prior"):
            case = f"{layout}-{best}"
            results = {name: fn(ids, ids, *dates, best=best) for name, fn in functions.items()}
            if any(result != results["after"] for result in results.values()):
                raise AssertionError(f"baseline differs for {case}")
            for fn in functions.values():
                for _ in range(args.warmup):
                    fn(ids, ids, *dates, best=best)
            samples: dict[str, list[float]] = {name: [] for name in functions}
            # Alternate order to avoid always favoring the same implementation.
            for repeat in range(args.repeat):
                order = list(functions)
                if repeat % 2:
                    order.reverse()
                for name in order:
                    start = time.perf_counter()
                    functions[name](ids, ids, *dates, best=best)
                    samples[name].append((time.perf_counter() - start) * 1000)
            report["cases"][case] = {
                name: {
                    "median_ms": statistics.median(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }
                for name, times in samples.items()
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
