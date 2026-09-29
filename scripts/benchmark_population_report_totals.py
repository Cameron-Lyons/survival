#!/usr/bin/env python3
"""Compare person-years report totals with flattening an existing grouped table.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_population_report_totals.py.
Input storage is allocated before measuring peak temporary memory.
"""

import argparse
import json
import math
import statistics
import time
import tracemalloc

from survival.r._population_print import _total
from survival.r._pyears import _flatten


def measure(function, repeats):
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - started)
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"median_ms": statistics.median(durations) * 1000, "peak_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    results = []
    for size in args.sizes:
        values = [[(i + j) % 5 + 0.2 for j in range(size)] for i in range(size)]
        total, direct = measure(lambda values=values: _total(values), args.repeats)
        flattened, flat = measure(
            lambda values=values, size=size: math.fsum(_flatten(values, [size, size])), args.repeats
        )
        if not math.isclose(total, flattened, rel_tol=1e-14):
            raise RuntimeError("total reducers disagree")
        results.append({"shape": [size, size], "direct": direct, "flatten": flat})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
