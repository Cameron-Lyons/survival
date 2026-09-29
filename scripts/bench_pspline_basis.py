"""Measure P-spline basis construction with fixed prediction boundaries.

PYTHONPATH=python .venv/bin/python scripts/bench_pspline_basis.py
Run from the same Python environment before and after changes to compare
the Rust basis evaluation and Python result/penalty construction together.
"""

import argparse
import json
import platform
import statistics
import time

from survival import r


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nterm", type=positive_int, nargs="+", default=[10, 100, 300])
    parser.add_argument("--nrows", type=positive_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=9)
    args = parser.parse_args()
    values = [9 * i / max(args.nrows - 1, 1) for i in range(args.nrows)]
    results = []
    for nterm in args.nterm:
        kwargs = {"nterm": nterm, "boundary_knots": (0, 9), "penalty": False}
        r.pspline(values, **kwargs)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            result = r.pspline(values, **kwargs)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        results.append(
            {
                "nterm": nterm,
                "nrows": args.nrows,
                "ncols": result.n_cols,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )
    print(
        json.dumps(
            {"python": platform.python_version(), "repeats": args.repeats, "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
