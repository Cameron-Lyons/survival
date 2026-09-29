#!/usr/bin/env python3
"""Measure the native rate-table matcher on repeated categorical observations.

Run before and after rebuilding the extension; table/input construction is excluded.
"""

import argparse
import json
import statistics
import time

from survival import population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    results = []
    for count in (2, 64, 1024):
        levels = [f"level{i:04d}" for i in range(count)]
        table = population.RateTable([count], ["group"], [levels], [None], [1], [0.01] * count)
        labels = [levels[i % count].upper() for i in range(args.rows)]
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            matched = population.match_ratetable(table, ["group"], [labels])
            durations.append(time.perf_counter() - start)
        positions = matched.r
        if len(positions) != args.rows or positions[-1] != [float((args.rows - 1) % count + 1)]:
            raise RuntimeError("matched positions are incorrect")
        results.append(
            {"levels": count, "rows": args.rows, "median_ms": statistics.median(durations) * 1000}
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
