"""Measure Surv row operations without response-construction time.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_surv_vectors.py.
"""

from __future__ import annotations

import argparse
import json
import platform
from statistics import median
from time import perf_counter

from survival import Surv, duplicated_surv, unique_surv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive")
    results = []
    for size in args.sizes:
        for distinct in (True, False):
            period = size if distinct else max(1, size // 10)
            response = Surv([float(row % period) for row in range(size)], [1] * size)
            for operation in (duplicated_surv, unique_surv):
                result = operation(response)
                expected_unique = period
                count = len(result) if operation is unique_surv else size - sum(result)
                if count != expected_unique:
                    raise RuntimeError("unexpected distinct-row count")
                samples = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    operation(response)
                    samples.append(perf_counter() - start)
                results.append(
                    {
                        "rows": size,
                        "distinct": distinct,
                        "operation": operation.__name__,
                        "median_ms": median(samples) * 1000,
                    }
                )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
