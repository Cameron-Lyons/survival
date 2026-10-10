#!/usr/bin/env python3
"""Time complete Surv/Surv2 construction and numeric matrix extraction.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_surv_inputs.py
--baseline-source /tmp/previous-surv.py. Input arrays/lists are built outside
timing; constructor validation, status coding, owned response construction,
matrix extraction and metadata access are timed. Payload hashing is untimed.
"""

from __future__ import annotations

import argparse
import gc
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
from survival.r import _surv


def source_module(path: Path) -> Any:
    name = "survival.r._surv_input_benchmark_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def complete_call(module: Any, constructor: str, inputs: tuple[Any, ...]) -> dict[str, Any]:
    response = getattr(module, constructor)(*inputs)
    return {
        "matrix": response.as_matrix(),
        "type": getattr(response, "type", None),
        "states": response.states,
        "clabel": response.clabel,
        "repeated": getattr(response, "repeated", None),
        "rows": len(response),
        "time": response.time,
        "status": response.status,
    }


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value for key, value in payload.items() if key not in {"matrix", "time", "status"}
    }
    matrix = np.asarray(payload["matrix"], dtype="<f8")
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    digest.update(matrix.tobytes())
    return {
        **metadata,
        "shape": list(matrix.shape),
        "first_rows": payload["matrix"][:3],
        "last_rows": payload["matrix"][-3:],
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 500_000])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--disable-gc", action="store_true")
    args = parser.parse_args()
    if min(args.rows) < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("rows must be at least two, repeat positive and warmup nonnegative")
    if args.disable_gc:
        gc.disable()
    modules = {"after": _surv}
    if args.baseline_source is not None:
        modules["before"] = source_module(args.baseline_source)
    if args.reverse_order:
        modules = dict(reversed(list(modules.items())))
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "initial_order": list(modules),
        "cyclic_gc_enabled": gc.isenabled(),
        "matrix_and_metadata_timed": True,
        "payload_hashing_timed": False,
        "sources": {
            label: {
                "path": module.__file__,
                "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            }
            for label, module in modules.items()
        },
        "cases": [],
    }
    for rows in args.rows:
        starts = np.arange(rows, dtype=float)
        times = starts + 1
        codes = np.arange(rows) % 2
        strided = np.empty(rows * 2, dtype=float)
        strided[::2] = codes + 1
        events = {
            "float64": codes.astype(float) + 1,
            "float64-strided": strided[::2],
            "logical": codes.astype(bool),
            "list": codes.tolist(),
        }
        for constructor, prefix in (
            ("Surv", (times,)),
            ("Surv-counting", (starts, times)),
            ("Surv2", (times,)),
        ):
            actual_constructor = "Surv" if constructor == "Surv-counting" else constructor
            for layout, event in events.items():
                inputs = (*prefix, event)
                payloads = {
                    label: summarize_payload(complete_call(module, actual_constructor, inputs))
                    for label, module in modules.items()
                }
                if any(payload != payloads["after"] for payload in payloads.values()):
                    raise AssertionError(f"baseline differs: {constructor}/{layout}/{rows}")
                for module in modules.values():
                    for _ in range(args.warmup):
                        complete_call(module, actual_constructor, inputs)
                samples: dict[str, list[float]] = {label: [] for label in modules}
                for iteration in range(args.repeat):
                    order = list(modules)
                    if iteration % 2:
                        order.reverse()
                    for label in order:
                        start = time.perf_counter()
                        result = complete_call(modules[label], actual_constructor, inputs)
                        samples[label].append((time.perf_counter() - start) * 1000)
                        del result
                report["cases"].append(
                    {
                        "constructor": constructor,
                        "input": layout,
                        "rows": rows,
                        "payload": payloads["after"],
                        "measurements": {
                            label: {
                                "median_ms": statistics.median(values),
                                "min_ms": min(values),
                                "max_ms": max(values),
                                "samples_ms": values,
                            }
                            for label, values in samples.items()
                        },
                    }
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
