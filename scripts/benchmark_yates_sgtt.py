"""Measure Python argument conversion and native SGTT with several array layouts.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_yates_sgtt.py.
"""

from statistics import median
from time import perf_counter

import numpy as np
from survival.validation import yates_sgtt


def run(n):
    index = np.arange(n)
    a, b = index % 2, (index // 2) % 2
    x = np.column_stack(
        [np.ones(n), a, 1 - a, b, 1 - b, a * b, (1 - a) * b, a * (1 - b), (1 - a) * (1 - b)]
    )
    inputs = {
        "list": x.tolist(),
        "numpy": x,
        "fortran": np.asfortranarray(x),
        "strided": np.repeat(x, 2, axis=1)[:, ::2],
    }
    for layout, values in inputs.items():
        elapsed = []
        for _ in range(9):
            started = perf_counter()
            yates_sgtt(
                values,
                [0, 1, 1, 2, 2, 3, 3, 3, 3],
                [[3], [3], []],
                [2, 3, 4, 5],
                np.eye(4),
                [0, 1, 2, 3],
                [(1, "a"), (2, "b")],
            )
            elapsed.append(perf_counter() - started)
        print(f"{n:>7,} rows {layout:>7}: {median(elapsed) * 1000:.3f} ms", flush=True)


if __name__ == "__main__":
    for n in (1_000, 10_000, 100_000):
        run(n)
