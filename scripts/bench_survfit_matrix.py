"""Benchmark joining three transition curves, including Python result conversion.

PYTHONPATH=python .venv/bin/python scripts/bench_survfit_matrix.py
Curve fitting and a warmup call are excluded from the measurements.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival, r


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=positive_int, default=5)
    args = parser.parse_args()
    extension = Path(_survival.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    for n in args.n:
        rows = np.arange(n)
        curves = [
            r.survfit(r.Surv(1.0 + rows + k * 0.1, (rows % 3 != k).astype(int)), se_fit=False)
            for k in range(3)
        ]
        matrix = [[None, curves[0], curves[1]], [None, None, curves[2]], [None, None, None]]
        for method in ("discrete", "matexp"):
            result = r.survfit(matrix, method=method)
            samples = []
            for _ in range(args.repeats):
                del result
                before = time.perf_counter_ns()
                result = r.survfit(matrix, method=method)
                samples.append((time.perf_counter_ns() - before) / 1e6)
            results.append(
                {
                    "method": method,
                    "n_per_transition": n,
                    "output_times": len(result.time),
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
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
