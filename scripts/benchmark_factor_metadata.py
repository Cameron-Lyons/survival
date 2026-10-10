#!/usr/bin/env python3
"""Compare complete ordinary-array strata calls with a saved factor helper.

Input creation is excluded; factor extraction/coding, native grouping, returned
factor construction and all result fields are included. Payload hashing is
outside timing. Baseline/current order alternates within each process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from survival.r import _coerce, _surv


def source_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def complete_call(module: Any, inputs: Any) -> dict[str, Any]:
    response = module.strata(inputs, shortlabel=True)
    return {field: getattr(response, field) for field in ("codes", "levels", "labels", "counts")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 500_000])
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    if min(args.rows) < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("rows must be at least two, repeat positive and warmup nonnegative")
    old_factor = source_module(args.baseline_source, "survival.r._factor_benchmark_baseline")
    old_boundary = source_module(Path(_surv.__file__), "survival.r._strata_benchmark_baseline")
    old_boundary._factor = old_factor._factor
    modules = {"after": _surv, "before": old_boundary}
    if args.reverse_order:
        modules = dict(reversed(list(modules.items())))
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "initial_order": list(modules),
        "result_fields_timed": True,
        "payload_hashing_timed": False,
        "source_hashes": {
            label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for label, module in {
                "boundary": _surv,
                "after_factor": _coerce,
                "before_factor": old_factor,
            }.items()
        },
        "cases": [],
    }
    for rows in args.rows:
        codes = np.arange(rows, dtype=np.int64) % 100
        strided = np.empty(rows * 2, dtype=float)
        strided[::2] = codes
        inputs = {
            "int64": codes,
            "float64": codes.astype(float),
            "float64-strided": strided[::2],
            "list": codes.tolist(),
        }
        for layout, values in inputs.items():
            results = {label: complete_call(module, values) for label, module in modules.items()}
            if any(result != results["after"] for result in results.values()):
                raise AssertionError(f"baseline differs: {layout}/{rows}")
            payload_hash = hashlib.sha256(
                json.dumps(results["after"], sort_keys=True).encode()
            ).hexdigest()
            payload = results["after"]
            output = {
                "levels": payload["levels"],
                "counts": payload["counts"],
                "first_codes": payload["codes"][:3],
                "last_codes": payload["codes"][-3:],
                "first_labels": payload["labels"][:3],
                "last_labels": payload["labels"][-3:],
            }
            del payload
            del results
            for module in modules.values():
                for _ in range(args.warmup):
                    complete_call(module, values)
            samples: dict[str, list[float]] = {label: [] for label in modules}
            for iteration in range(args.repeat):
                order = list(modules)
                if iteration % 2:
                    order.reverse()
                for label in order:
                    start = time.perf_counter()
                    result = complete_call(modules[label], values)
                    samples[label].append((time.perf_counter() - start) * 1000)
                    del result
            report["cases"].append(
                {
                    "input": layout,
                    "rows": rows,
                    "output": output,
                    "payload_sha256": payload_hash,
                    "measurements": {
                        label: {
                            "median_ms": statistics.median(times),
                            "min_ms": min(times),
                            "max_ms": max(times),
                            "samples_ms": times,
                        }
                        for label, times in samples.items()
                    },
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
