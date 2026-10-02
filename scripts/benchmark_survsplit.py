"""Benchmark complete Python interval splitting with cuts outside follow-up."""

import argparse
import gc
import json
import statistics
import time

from survival.data_prep import survsplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--cuts", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.cuts, args.repeats) < 1:
        parser.error("rows, cuts and repeats must be positive")

    times = [1.0] * args.rows
    status = [0.0] * args.rows
    results = {}
    for case, cuts in (
        ("cuts_before_followup", [float(i - args.cuts) for i in range(args.cuts)]),
        ("cuts_after_followup", [float(i + 2) for i in range(args.cuts)]),
    ):
        result = survsplit(times, status, cuts, timefix=False)
        episode = args.cuts if case == "cuts_before_followup" else 0
        if (
            len(result.row) != args.rows
            or result.start != [0.0] * args.rows
            or result.end != times
            or result.interval != [episode] * args.rows
        ):
            raise RuntimeError(f"survsplit returned unexpected intervals for {case}")
        del result
        elapsed = []
        for _ in range(args.repeats):
            gc.collect()
            started = time.perf_counter()
            result = survsplit(times, status, cuts, timefix=False)
            elapsed.append(1000 * (time.perf_counter() - started))
            del result
        results[case] = {
            "median_ms": statistics.median(elapsed),
            "samples_ms": elapsed,
        }
    print(json.dumps({"rows": args.rows, "cuts": args.cuts, "results": results}, indent=2))


if __name__ == "__main__":
    main()
