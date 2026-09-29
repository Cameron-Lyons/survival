"""Measure owned AFT data construction, including matrix conversion and validation.

Run with PYTHONPATH=python .venv/bin/python scripts/bench_matrix_input.py.
The input arrays/lists are prepared before timing; no model is fitted.
"""

from statistics import median
from time import perf_counter

import numpy as np
from survival.regression import SurvregData


def run(n, p):
    time = np.arange(1.0, n + 1)
    status = np.ones(n, dtype=np.int32)
    x = np.arange(n * p, dtype=float).reshape(n, p) / (n * p)
    layouts = {
        "list": x.tolist(),
        "tuple": tuple(map(tuple, x.tolist())),
        "numpy": x,
        "fortran": np.asfortranarray(x),
        "strided": np.repeat(x, 2, axis=1)[:, ::2],
    }
    for name, values in layouts.items():
        elapsed = []
        for _ in range(11):
            started = perf_counter()
            data = SurvregData(time, status, values)
            elapsed.append(perf_counter() - started)
            del data
        print(f"{n:>7,} x {p:<2} {name:>7}: {median(elapsed) * 1000:.3f} ms", flush=True)


if __name__ == "__main__":
    for n, p in ((1000, 4), (100000, 4), (100000, 16)):
        run(n, p)
