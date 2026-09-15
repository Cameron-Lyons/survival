#!/usr/bin/env python3
"""Time the core kernels against R's `survival` on the same synthetic data.

Usage (from the repo root, with the extension built into the active venv):

    PYTHONPATH=python python benches/python/bench_vs_r.py [--sizes 1000,10000,100000]
        [--rscript /path/to/Rscript] [--repeat 3] [--csv out.csv]

The Rust kernels are called through their low-level Python bindings so that
only the kernel (plus the list -> Vec conversion at the boundary) is timed;
R runs the same fits through `coxph`, `survfit`, `survdiff`, `concordance`
and `survreg` on identical CSV files. Rscript is looked up in ``--rscript``,
``$SURVIVAL_RSCRIPT`` and ``$PATH``; without it only the Python column is
reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

R_SCRIPT = r"""
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
path <- args[1]; repeat_n <- as.integer(args[2])
d <- read.csv(path)
d$x1 <- d$x1; d$x2 <- d$x2; d$x3 <- d$x3
best <- function(expr) {
  times <- numeric(repeat_n)
  for (i in seq_len(repeat_n)) {
    t0 <- proc.time()[["elapsed"]]; force(expr()); times[i] <- proc.time()[["elapsed"]] - t0
  }
  min(times)
}
out <- list(
  coxph = best(function() coxph(Surv(time, status) ~ x1 + x2 + x3, data = d, ties = "efron")),
  survfit = best(function() survfit(Surv(time, status) ~ 1, data = d)),
  survfit_strata = best(function() survfit(Surv(time, status) ~ group, data = d)),
  survdiff = best(function() survdiff(Surv(time, status) ~ group, data = d)),
  concordance = best(function() concordance(Surv(time, status) ~ x1, data = d)),
  survreg = best(function() survreg(Surv(time, status) ~ x1 + x2 + x3, data = d, dist = "weibull"))
)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA))
"""


def make_data(n: int, seed: int) -> dict[str, list]:
    rng = random.Random(seed)  # noqa: S311 - synthetic benchmark data
    x1 = [rng.gauss(0.0, 1.0) for _ in range(n)]
    x2 = [rng.gauss(0.0, 1.0) for _ in range(n)]
    x3 = [float(rng.random() < 0.5) for _ in range(n)]
    group = [rng.randrange(3) for _ in range(n)]
    time_ = []
    status = []
    for a, b, c in zip(x1, x2, x3, strict=True):
        rate = math.exp(0.5 * a - 0.3 * b + 0.2 * c)
        event = -math.log(rng.random()) / rate
        censor = -math.log(rng.random()) / 0.5
        t = min(event, censor)
        # a coarse grid so ties are common, as in real data
        time_.append(round(t, 2) + 0.01)
        status.append(int(event <= censor))
    return {"time": time_, "status": status, "x1": x1, "x2": x2, "x3": x3, "group": group}


def best_of(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def python_timings(data: dict[str, list], repeat: int) -> dict[str, float]:
    from survival import _survival as core

    n = len(data["time"])
    rows = [[a, b, c] for a, b, c in zip(data["x1"], data["x2"], data["x3"], strict=True)]
    flat = [v for row in rows for v in row]
    surv = core.SurvivalData(data["time"], data["status"])
    x1 = core.CovariateMatrix(data["x1"], n, 1)
    survreg_data = core.SurvregData(data["time"], data["status"], [[1.0, *row] for row in rows])
    weibull = core.SurvregDistribution("weibull")
    del flat
    return {
        "coxph": best_of(
            lambda: core.coxph_fit(data["time"], data["status"], rows, method="efron"), repeat
        ),
        "survfit": best_of(lambda: core.survfitkm(data["time"], data["status"]), repeat),
        "survfit_strata": best_of(
            lambda: core.survfitkm(data["time"], data["status"], strata=data["group"]), repeat
        ),
        "survdiff": best_of(
            lambda: core.survdiff(data["time"], data["status"], data["group"]), repeat
        ),
        "concordance": best_of(lambda: core.concordancefit(surv, x1), repeat),
        "survreg": best_of(lambda: core.survreg_fit(survreg_data, weibull), repeat),
    }


def r_timings(rscript: str, csv_path: Path, repeat: int) -> dict[str, float] | None:
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(R_SCRIPT)
        script = handle.name
    try:
        proc = subprocess.run(  # noqa: S603 - fixed interpreter, generated script
            [rscript, "--vanilla", script, str(csv_path), str(repeat)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(script)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


def find_rscript(explicit: str | None) -> str | None:
    for candidate in (explicit, os.environ.get("SURVIVAL_RSCRIPT"), shutil.which("Rscript")):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rscript", default=None)
    parser.add_argument("--csv", default=None, help="also write the table as CSV")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    rscript = find_rscript(args.rscript)
    sizes = [int(s) for s in args.sizes.split(",") if s]
    rows_out = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in sizes:
            data = make_data(n, args.seed)
            py = python_timings(data, args.repeat)
            r = None
            if rscript:
                path = Path(tmp) / f"bench_{n}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(list(data))
                    writer.writerows(zip(*data.values(), strict=True))
                r = r_timings(rscript, path, args.repeat)
            for kernel, seconds in py.items():
                r_seconds = r.get(kernel) if r else None
                rows_out.append(
                    {
                        "n": n,
                        "kernel": kernel,
                        "python_s": seconds,
                        "r_s": r_seconds,
                        "speedup": (r_seconds / seconds) if r_seconds and seconds > 0 else None,
                    }
                )

    header = f"{'n':>8} {'kernel':<15} {'python (s)':>11} {'R (s)':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for row in rows_out:
        r_txt = f"{row['r_s']:.4f}" if row["r_s"] is not None else "-"
        s_txt = f"{row['speedup']:.1f}x" if row["speedup"] is not None else "-"
        print(f"{row['n']:>8} {row['kernel']:<15} {row['python_s']:>11.4f} {r_txt:>9} {s_txt:>8}")
    speedups = [row["speedup"] for row in rows_out if row["speedup"]]
    if speedups:
        print(f"\ngeometric-mean speedup vs R: {statistics.geometric_mean(speedups):.1f}x")
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
