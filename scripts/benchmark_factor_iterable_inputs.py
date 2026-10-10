#!/usr/bin/env python3
"""Compare complete public aggregation/concordance calls with saved boundaries.

Create reusable inputs outside timing. Each sample includes factor preparation,
the native calculation, result construction and every result field/getter. Hash
the complete result outside timing and require identical before/after payloads.
Alternate saved/current Python boundaries in one process with one extension.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
import types
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from survival import _survival as native
from survival.r import _coerce, _concordance, _surv, _survfit


def source_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load saved source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def copied_function(function: Any, **replacements: Any) -> Any:
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    copy = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    return copy


def aggregate_call(function: Any, curves: Any, groups: Any) -> dict[str, Any]:
    result = function(curves, by=groups)
    newdata = result.newdata
    return {
        "surv": result.surv,
        "pstate": result.pstate,
        "newdata": None if newdata is None else {"names": newdata.names, "labels": newdata.labels},
    }


def concordance_call(function: Any, response: Any, predictor: Any, groups: Any) -> dict[str, Any]:
    result = function(response, predictor, cluster=groups, influence=1, ranks=False)
    return {field.name: getattr(result, field.name) for field in dataclasses.fields(result)}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(calls: dict[str, Any], repeat: int, warmup: int, reverse_order: bool) -> dict[str, Any]:
    order = list(calls)
    if reverse_order:
        order.reverse()
    payloads = {label: call() for label, call in calls.items()}
    encoded = {label: json.dumps(payload, sort_keys=True) for label, payload in payloads.items()}
    if any(payload != encoded["after"] for payload in encoded.values()):
        raise AssertionError("saved/current complete result payloads differ")
    payload = payloads["after"]
    payload_hash = hashlib.sha256(encoded["after"].encode()).hexdigest()
    del payloads
    del encoded
    for label in order:
        for _ in range(warmup):
            calls[label]()
    samples: dict[str, list[float]] = {label: [] for label in calls}
    for iteration in range(repeat):
        sample_order = order if iteration % 2 == 0 else list(reversed(order))
        for label in sample_order:
            start = time.perf_counter()
            result = calls[label]()
            samples[label].append((time.perf_counter() - start) * 1000)
            del result
    return {
        "output": payload,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-factor-source", type=Path, required=True)
    parser.add_argument("--baseline-concordance-source", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 500_000])
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--disable-gc", action="store_true")
    args = parser.parse_args()
    if min(args.rows) < 2 or args.repeat < 1 or args.warmup < 0:
        parser.error("rows must be at least two, repeat positive and warmup nonnegative")
    old_factor = source_module(
        args.baseline_factor_source, "survival.r._iterable_factor_benchmark_before"
    )
    old_concordance = source_module(
        args.baseline_concordance_source, "survival.r._iterable_concordance_benchmark_before"
    )
    old_concordance._factor = old_factor._factor
    old_concordance._r_factor_levels = old_factor._r_factor_levels
    old_grouping = copied_function(_survfit._grouping_factors, _factor=old_factor._factor)
    old_aggregate = copied_function(_survfit.aggregate_survfit, _grouping_factors=old_grouping)
    if args.disable_gc:
        gc.disable()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "initial_order": ["after", "before"] if args.reverse_order else ["before", "after"],
        "gc_enabled": gc.isenabled(),
        "result_fields_timed": True,
        "payload_hashing_timed": False,
        "baseline_reconstruction": {
            "aggregate": "current public body with saved factor helper in grouping adapter",
            "concordance": "saved public module with saved factor helper",
            "native": "one shared extension for both Python versions",
        },
        "source_hashes": {
            "aggregate_boundary": sha256(_survfit.__file__),
            "before_factor": sha256(args.baseline_factor_source),
            "after_factor": sha256(_coerce.__file__),
            "before_concordance": sha256(args.baseline_concordance_source),
            "after_concordance": sha256(_concordance.__file__),
            "extension": sha256(native.__file__),
        },
        "cases": [],
    }
    for rows in args.rows:
        codes = np.arange(rows, dtype=np.int64) % 100
        labels = np.asarray([f"group{code:03d}" for code in range(100)])[codes]
        ordinary = {
            "character-list": labels.tolist(),
            "character-unicode-array": labels,
            "character-object-array": labels.astype(object),
            "numeric-array-control": codes,
        }
        values = np.arange(rows, dtype=np.float64)
        curves = types.SimpleNamespace(
            surv=[(0.9 - (values % 17) / 100).tolist(), (0.7 - (values % 19) / 100).tolist()]
        )
        response = _surv.Surv(values + 1, (codes % 3 != 0).astype(float))
        predictor = ((values * 17) % 311).astype(float)
        declared = _coerce._r_factor(
            ordinary["character-list"], tuple(sorted(set(labels), reverse=True))
        )
        for operation, inputs in (
            ("aggregate_survfit", ordinary),
            (
                "concordancefit",
                {
                    "character-list": ordinary["character-list"],
                    "numeric-array-control": ordinary["numeric-array-control"],
                    "declared-factor-control": declared,
                },
            ),
        ):
            for layout, groups in inputs.items():
                if operation == "aggregate_survfit":
                    calls = {
                        "before": partial(aggregate_call, old_aggregate, curves, groups),
                        "after": partial(
                            aggregate_call, _survfit.aggregate_survfit, curves, groups
                        ),
                    }
                else:
                    calls = {
                        "before": partial(
                            concordance_call,
                            old_concordance.concordancefit,
                            response,
                            predictor,
                            groups,
                        ),
                        "after": partial(
                            concordance_call,
                            _concordance.concordancefit,
                            response,
                            predictor,
                            groups,
                        ),
                    }
                result = measure(calls, args.repeat, args.warmup, args.reverse_order)
                report["cases"].append(
                    {"operation": operation, "input": layout, "rows": rows, **result}
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
