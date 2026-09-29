"""Benchmark Cox predictions and Brier scores, including the Python/Rust boundary.

Run the same command against each release build:
    PYTHONPATH=python .venv/bin/python scripts/bench_cox_predictions.py

Fitting, input construction and a warmup call are excluded. The default sizes
also work with older builds that materialize every subject's full curve.
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
from survival import CoxPHEstimator, _survival
from survival.r import brier


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[100, 1000, 3000])
    parser.add_argument("--times", type=positive_int, nargs="+", default=[4, 64])
    parser.add_argument("--repeats", type=positive_int, default=5)
    args = parser.parse_args()
    extension = Path(_survival.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    for n in args.n:
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(n, 3))
        followup = rng.exponential(size=n) * np.exp(-0.3 * x[:, 0])
        status = (rng.random(n) < 0.7).astype(np.int32)
        estimator = CoxPHEstimator().fit(x, np.column_stack([followup, status]))
        for n_times in args.times:
            times = np.linspace(0.0, float(np.quantile(followup, 0.9)), n_times)
            # Plain lists keep the kernel benchmark comparable with older bindings.
            time_list, status_list, at = followup.tolist(), status.tolist(), times.tolist()
            phat = (1.0 - np.exp(-times[:, None] * np.exp(x[:, 0])[None, :])).tolist()
            calls = {
                "cox_survival": partial(estimator.predict_survival_function, x, times),
                "brier": partial(brier, estimator.model_, times=at),
                "brier_kernel": partial(_survival.brier, time_list, status_list, at, phat),
            }
            for operation, call in calls.items():
                result = call()
                samples = []
                for _ in range(args.repeats):
                    del result
                    before = time.perf_counter_ns()
                    result = call()
                    samples.append((time.perf_counter_ns() - before) / 1_000_000)
                results.append(
                    {
                        "operation": operation,
                        "n": n,
                        "n_times": n_times,
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                    }
                )
                del result
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "extension": str(extension),
                "extension_sha256": digest,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
