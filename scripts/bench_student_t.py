"""Benchmark Student-t helpers and their AFT fitting/residual consumers.

Select a complete Python package and native extension via PYTHONPATH:
    PYTHONPATH=python .venv/bin/python scripts/bench_student_t.py --n 10000

Calls include input conversion and output construction; input preparation and
result disposal are excluded. Quantile inputs stay away from extreme tails so
older implementations can complete the same workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import survival
from survival import _survival as native


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def measure(call, repeats: int):
    result = call()
    samples = []
    for _ in range(repeats):
        del result
        before = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - before) / 1_000_000)
    return samples, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, default=10000)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    values = np.linspace(-8, 8, args.n).tolist()
    probabilities = np.linspace(0.001, 0.999, args.n).tolist()
    mean = [0.0] * args.n
    scale = [1.0] * args.n
    extension = Path(native.__file__).resolve()
    with extension.open("rb") as source:
        extension_hash = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "label": args.label,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "extension_sha256": extension_hash,
        "package_path": str(Path(survival.__file__).resolve()),
        "repeats": args.repeats,
        "warmups": 1,
        "results": [],
    }

    def record(operation, call, n, df):
        samples, result = measure(call, args.repeats)
        report["results"].append(
            {
                "operation": operation,
                "n": n,
                "df": df,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )
        return result

    for df in [1.0, 4.0, 7.0, 1000.0, 1e8]:
        for kind, inputs in [
            ("density", values),
            ("distribution", values),
            ("quantile", probabilities),
        ]:
            record(
                kind,
                partial(native.survreg_distribution, inputs, mean, scale, "t", kind, df),
                args.n,
                df,
            )
    x = np.linspace(-1, 1, 1000)
    time = (10 + 0.3 * x + np.sin(np.arange(1000) * 0.43)).tolist()
    status = (np.arange(1000) % 4 != 0).astype(int).tolist()
    fit = record(
        "fit",
        partial(
            survival.regression.survreg,
            time=time,
            status=status,
            covariates=np.column_stack([np.ones(1000), x]).tolist(),
            distribution="t",
            distribution_parameter=7.0,
            max_iter=40,
        ),
        1000,
        7.0,
    )
    report["fit"] = {
        "coefficients": fit.coefficients,
        "scale": fit.scale,
        "log_likelihood": fit.log_likelihood,
        "convergence_flag": fit.convergence_flag,
        "iterations": fit.iterations,
    }
    record(
        "residual_matrix",
        partial(
            survival.survreg_residual_matrix,
            time,
            status,
            fit.linear_predictors,
            fit.scale,
            "t",
            distribution_parameter=7.0,
        ),
        1000,
        7.0,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
