"""Measure compact report preparation versus full summaries, excluding fitting.

The full summary also returns event-time arrays. It is a useful reference for
the time and Python allocation cost avoided when only the compact table is wanted.
"""

import argparse
import gc
import json
import platform
import tracemalloc
from statistics import median
from time import perf_counter

import numpy as np
from survival import r


def measure(function, repeats):
    function()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        durations.append(1000 * (perf_counter() - start))
    gc.collect()
    tracemalloc.start()
    result = function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"ms": median(durations), "python_peak_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if min([*args.n, args.repeats]) < 1:
        parser.error("sizes and repeats must be positive")
    results = []
    for n in args.n:
        time = np.arange(1, n + 1, dtype=float)
        status = (time % 3 != 0).astype(int)
        fit = r.survfit(r.Surv(time, status))
        results.append(
            {
                "n": n,
                "compact_report": measure(
                    lambda fit=fit: r.print_survfit(fit, rmean="common"), args.repeats
                ),
                "full_summary": measure(lambda fit=fit: r.summary_survfit(fit), args.repeats),
            }
        )
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
