"""Measure complete quantile calls on stacked curve vectors and NumPy layouts.

PYTHONPATH=python .venv/bin/python scripts/benchmark_curve_quantile_boundary.py
Add --nprobs 3 1001 for many probabilities, or --prepared-fit for the binding
that receives an already constructed fit. --extension loads a saved library
from the same Python ABI. Inputs and prepared fits are built outside timing;
the default vector calls include native extraction, ownership, construction,
validation, curve inversion, result construction and every result-list getter.
Result hashes are computed after timing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from survival import _survival


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def input_layout(values: np.ndarray, layout: str) -> list[float] | np.ndarray:
    """Keep logical values identical while changing storage and strides."""

    if layout == "lists":
        return values.tolist()
    if layout == "strided":
        return np.repeat(values, 2)[::2]
    if layout == "reversed":
        return values[::-1].copy()[::-1]
    return values.copy()


def result_payload(result: Any) -> dict[str, Any]:
    return {
        "probs": result.probs,
        "quantile": result.quantile,
        "lower": result.lower,
        "upper": result.upper,
    }


def measure(call: Callable[[], Any], warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        result_payload(call())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        payload = result_payload(call())
        samples.append((time.perf_counter_ns() - started) / 1e6)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "median_ms": statistics.median(samples),
        "range_ms": [min(samples), max(samples)],
        "samples_ms": samples,
        "result_sha256": hashlib.sha256(encoded).hexdigest(),
        **payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[100000, 500000])
    parser.add_argument("--nprobs", type=positive_int, nargs="+", default=[3])
    parser.add_argument("--repeats", type=positive_int, default=9)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--prepared-fit", action="store_true")
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    core = _survival
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must name a loadable extension library")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    extension = Path(core.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    for n in args.n:
        times = np.arange(1, n + 1, dtype=np.float64)
        surv = 1 - times / (n + 1)
        lower = surv**1.3
        upper = surv**0.7
        prepared = (
            core.SurvfitKMResult.from_stacked(
                times, times[::-1], np.ones(n), surv, [n], lower=lower, upper=upper
            )
            if args.prepared_fit
            else None
        )
        for layout in ("lists", "contiguous", "strided", "reversed"):
            curve_inputs = tuple(
                input_layout(values, layout) for values in (times, surv, lower, upper)
            )
            for nprobs in args.nprobs:
                values = (
                    np.array([0.25, 0.5, 0.75]) if nprobs == 3 else np.linspace(0.01, 0.99, nprobs)
                )
                probabilities = input_layout(values, layout)
                # Omitting probs exercises the entry point's default quartiles.
                boundary_probs = None if nprobs == 3 else probabilities
                for conf_int in (False, True):
                    boundary_call = partial(
                        core.quantile_survfit_curves,
                        *curve_inputs,
                        probs=boundary_probs,
                        conf_int=conf_int,
                    )
                    results.append(
                        {
                            "entry_point": "quantile_survfit_curves",
                            "n": n,
                            "layout": layout,
                            "nprobs": nprobs,
                            "default_probs": nprobs == 3,
                            "conf_int": conf_int,
                            **measure(boundary_call, args.warmup, args.repeats),
                        }
                    )
                    if prepared is not None:
                        prepared_call = partial(
                            core.quantile_survfit, prepared, probabilities, conf_int=conf_int
                        )
                        results.append(
                            {
                                "entry_point": "quantile_survfit",
                                "n": n,
                                "layout": layout,
                                "nprobs": nprobs,
                                "default_probs": False,
                                "conf_int": conf_int,
                                **measure(prepared_call, args.warmup, args.repeats),
                            }
                        )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "extension": str(extension),
                "extension_sha256": digest,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "input_preparation_timed": False,
                "prepared_fit_construction_timed": False,
                "result_getters_timed": True,
                "result_hashing_timed": False,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
