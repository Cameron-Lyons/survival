#!/usr/bin/env python3
"""Time complete tmerge calls and compare full frames with a saved Python adapter."""

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
from survival.r import _data_prep


def baseline(path: Path) -> Any:
    name = "survival.r._tmerge_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(module: Any, frame: Any) -> dict[str, Any]:
    return {
        "columns": frame.columns,
        "tname": frame.tname,
        "tevent": frame.tevent,
        "tdcvar": frame.tdcvar,
        "tcount": frame.tcount,
        "summary": module.summary_tmerge(frame),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--unused-columns", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    if args.rows < 4 or args.unused_columns < 0 or args.repeat < 1:
        parser.error("rows >= 4, unused-columns >= 0, and repeat >= 1 are required")
    modules = {"after": _data_prep}
    if args.baseline_source is not None:
        modules["before"] = baseline(args.baseline_source)
    subjects = max(1, args.rows // 4)
    base = {"id": np.arange(subjects), "time": np.full(subjects, 8.0), "event": np.ones(subjects)}
    initial = _data_prep.tmerge(base, base, id="id", death=_data_prep.event("time", "event"))
    rows = np.arange(args.rows)
    updates = {
        "id": rows % subjects,
        "time": np.floor(rows / subjects) + 1.0,
        "value": np.sin(rows),
        "event": (rows % 3 == 0).astype(np.int32),
    }
    wide = {
        **updates,
        **{f"unused_{column}": np.cos(rows + column) for column in range(args.unused_columns)},
    }

    def initial_call(module: Any) -> dict[str, Any]:
        return snapshot(
            module, module.tmerge(base, base, id="id", death=module.event("time", "event"))
        )

    def update_call(module: Any, data: Any) -> dict[str, Any]:
        frame = module.tmerge(
            initial,
            data,
            id="id",
            lab=module.tdc("time", "value"),
            count=module.cumtdc("time"),
            infection=module.event("time", "event"),
        )
        return snapshot(module, frame)

    workloads = {
        "initial_shared_frames": lambda module: initial_call(module),
        "updates_narrow": lambda module: update_call(module, updates),
        "updates_wide": lambda module: update_call(module, wide),
    }
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rows": args.rows,
        "subjects": subjects,
        "unused_columns": args.unused_columns,
        "repeat": args.repeat,
        "measurements": {},
    }
    for name, workload in workloads.items():
        payloads = {
            label: json.dumps(workload(module), sort_keys=True) for label, module in modules.items()
        }
        if len(set(payloads.values())) != 1:
            raise AssertionError(f"complete before/after frame differs for {name}")
        samples = {label: [] for label in modules}
        for iteration in range(args.repeat):
            order = list(modules) if iteration % 2 else list(reversed(modules))
            for label in order:
                started = time.perf_counter()
                frame = workload(modules[label])
                samples[label].append((time.perf_counter() - started) * 1000)
                del frame
        result["measurements"][name] = {
            "frame_sha256": hashlib.sha256(payloads["after"].encode()).hexdigest(),
            "timings_ms": {
                label: {"median": statistics.median(values), "min": min(values), "max": max(values)}
                for label, values in samples.items()
            },
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
