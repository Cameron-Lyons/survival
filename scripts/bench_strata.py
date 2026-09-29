"""Measure formula-frame construction with plain and interacting strata.

Run with PYTHONPATH=python .venv/bin/python scripts/bench_strata.py.
Uses NumPy input and includes formula evaluation, grouping, and matrix construction.
"""

from functools import partial
from statistics import median
from timeit import repeat

import numpy as np
from survival.r._fit import _model_frame


def main():
    n = 100_000
    data = {
        "time": np.arange(1.0, n + 1),
        "status": np.tile([1, 1, 0, 1], n // 4),
        "age": np.tile(np.arange(40.0, 80.0), n // 40),
        "sex": np.tile([1, 2], n // 2),
    }
    for rhs in ("age + strata(sex)", "age * strata(sex)", "age * strata(sex == 1)"):
        build = partial(_model_frame, f"Surv(time, status) ~ {rhs}", data)
        try:
            build()
        except ValueError as error:
            print(f"{rhs}: {error}")
            continue
        samples = repeat(build, number=3, repeat=5)
        print(f"{rhs}: {median(samples) * 1000 / 3:.2f} ms ({n:,} rows)")


if __name__ == "__main__":
    main()
