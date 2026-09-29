#!/usr/bin/env python3
"""Compare bounded and complete rate-table report rendering.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_array_reports.py.
Both paths retain every rate. Measurements exclude input table construction.
"""

import argparse
import json
import statistics
import time
import tracemalloc

from survival import r


def measure(table, limit, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        report = r.print_ratetable(table, max_print=limit)
        times.append(time.perf_counter() - started)
    tracemalloc.start()
    report = r.print_ratetable(table, max_print=limit)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "median_ms": statistics.median(times) * 1000,
        "peak_bytes": peak,
        "displayed": report.displayed,
        "retained": len(report.rates),
        "lines": len(report.lines),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    tables = {
        "survexp.us": r.survexp_us(),
        "custom": r.RateTable(
            [128, 4, 128],
            ["age", "group", "year"],
            [[str(i) for i in range(n)] for n in [128, 4, 128]],
            [None] * 3,
            [1] * 3,
            [i * 1e-9 for i in range(128 * 4 * 128)],
        ),
    }
    results = {}
    for name, table in tables.items():
        results[name] = {
            "bounded": measure(table, 6, args.repeats),
            "complete": measure(table, 2147483646, args.repeats),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
