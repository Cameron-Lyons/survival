"""Benchmark public survival-curve calls, including Python/Rust conversion.

Run against each release build to compare changes:
    PYTHONPATH=python .venv/bin/python scripts/bench_survival_curves.py

Input construction, warmup and disposal of the previous result are excluded.
Fractional weights exercise the default robust variance of Kaplan-Meier fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival, surv_analysis


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[1000, 10000, 30000])
    parser.add_argument("--repeats", type=positive_int, default=5)
    args = parser.parse_args()
    extension = Path(_survival.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    for n in args.n:
        rows = np.arange(n)
        times = (rows + 1).astype(float)
        status = (rows % 3 != 0).astype(np.int32)
        weights = 0.5 + (rows % 7) / 4
        for estimator in ["survfitkm", "nelson_aalen"]:
            fit = getattr(surv_analysis, estimator)
            for weighted in [False, True]:
                kwargs = {"weights": weights} if weighted else {}
                result = fit(times, status, **kwargs)
                samples = []
                for _ in range(args.repeats):
                    del result
                    before = time.perf_counter_ns()
                    result = fit(times, status, **kwargs)
                    samples.append((time.perf_counter_ns() - before) / 1_000_000)
                results.append(
                    {
                        "estimator": estimator,
                        "n": n,
                        "weighted": weighted,
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                        "output_rows": len(result.time),
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
