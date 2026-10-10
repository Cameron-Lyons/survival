"""Measure native and formula curve summaries on sparse and dense time grids.

Fit preparation is excluded. Run against a release extension before and after
changes; native measurements include checked Python input extraction and the
returned Rust result, while formula measurements also include the summary table.
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np
from survival import r, surv_analysis


def measure(function, repeats):
    function()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        durations.append(1000 * (perf_counter() - start))
    return median(durations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=int, default=[10000, 100000, 300000])
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if min([*args.n, args.repeats]) < 1:
        parser.error("sizes and repeats must be positive")
    results = []
    for n in args.n:
        time = np.arange(1, n + 1, dtype=float)
        fit = r.survfit(r.Surv(time, (time % 3 != 0).astype(int)))
        for queries in (10, n):
            values = np.linspace(0, n + 1, queries)
            for layout in ("list", "numpy"):
                times = values.tolist() if layout == "list" else values
                results.append(
                    {
                        "n": n,
                        "queries": queries,
                        "layout": layout,
                        "native_ms": measure(
                            lambda fit=fit, times=times: surv_analysis.summary_survfit(
                                fit.engine, times=times, extend=True
                            ),
                            args.repeats,
                        ),
                        "formula_ms": measure(
                            lambda fit=fit, times=times: r.summary_survfit(
                                fit, times=times, extend=True
                            ),
                            args.repeats,
                        ),
                    }
                )
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
