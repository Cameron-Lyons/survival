#!/usr/bin/env python3
"""Time survival against R's `survival` on the same synthetic data, layer by layer.

Usage (from the repo root, with the extension built into the active venv):

    PYTHONPATH=python python benches/python/bench_vs_r.py [--sizes 1000,10000,100000]
        [--rscript /path/to/Rscript] [--repeat 3] [--csv out.csv]

Each routine is timed at two layers, and each layer is compared only with the
matching R layer:

``formula``
    ``survival.r.<fn>(formula, DataFrame)`` against R's ``<fn>(formula, data)``:
    the whole user-facing call (model frame, fit and the extras R computes,
    such as coxph's concordance and residuals).  This is what users pay.

``kernel``
    The ``survival._survival`` binding on NumPy arrays against the R entry
    point that runs the same computation without a formula:

    ==============  =====================================  ==================================
    routine         Python                                 R
    ==============  =====================================  ==================================
    coxph           ``coxph_fit(time, status, x)``         ``coxph.fit`` + ``concordancefit``
    survfit         ``survfitkm(time, status)``            ``survfitKM``
    survfit_strata  ``survfitkm(..., strata=group)``       ``survfitKM``
    survdiff        ``survdiff(time, status, group)``      ``survdiff.fit``
    concordance     ``concordancefit(SurvivalData, x1)``   ``concordancefit``
    survreg         ``survreg_fit(SurvregData, weibull)``  ``survreg.fit`` (extreme, log time)
    ==============  =====================================  ==================================

    ``coxph_fit`` always returns the concordance of its linear predictor, so
    R's side adds the ``concordancefit`` call that ``coxph()`` makes.  The
    Python side builds the binding's typed inputs inside the timed call; R's
    ``Surv`` object and design matrix are built beforehand, as the NumPy
    arrays are.

The ratio column is R's time over Python's within a layer (above 1: Python is
faster).  Rscript is looked up in ``--rscript``, ``$SURVIVAL_RSCRIPT`` and
``$PATH``; without it only the Python columns are reported.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROUTINES = ("coxph", "survfit", "survfit_strata", "survdiff", "concordance", "survreg")

R_SCRIPT = r"""
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
path <- args[1]; repeat_n <- as.integer(args[2])
d <- read.csv(path)
best <- function(layer, routine, expr) {
  times <- numeric(repeat_n)
  for (i in seq_len(repeat_n)) {
    t0 <- Sys.time(); force(expr()); times[i] <- as.double(Sys.time() - t0, units = "secs")
  }
  cat(layer, routine, format(min(times), digits = 17), "\n")
}
best("formula", "coxph",
     function() coxph(Surv(time, status) ~ x1 + x2 + x3, data = d, ties = "efron"))
best("formula", "survfit", function() survfit(Surv(time, status) ~ 1, data = d))
best("formula", "survfit_strata", function() survfit(Surv(time, status) ~ group, data = d))
best("formula", "survdiff", function() survdiff(Surv(time, status) ~ group, data = d))
best("formula", "concordance", function() concordance(Surv(time, status) ~ x1, data = d))
best("formula", "survreg",
     function() survreg(Surv(time, status) ~ x1 + x2 + x3, data = d, dist = "weibull"))

n <- nrow(d)
y <- Surv(d$time, d$status)
logy <- Surv(log(d$time), d$status)
x <- as.matrix(d[c("x1", "x2", "x3")])
x_intercept <- cbind(1, x)
one <- factor(rep(1, n))
group <- factor(d$group)
best("kernel", "coxph", function() {
  fit <- coxph.fit(x, y, strata = NULL, offset = NULL, init = NULL,
                   control = coxph.control(), weights = NULL, method = "efron",
                   rownames = NULL, nocenter = c(-1, 0, 1))
  concordancefit(y, fit$linear.predictors, reverse = TRUE, timefix = FALSE)
})
best("kernel", "survfit", function() survfitKM(one, y))
best("kernel", "survfit_strata", function() survfitKM(group, y))
best("kernel", "survdiff", function() survival:::survdiff.fit(y, d$group))
best("kernel", "concordance", function() concordancefit(y, d$x1))
best("kernel", "survreg",
     function() survreg.fit(x_intercept, logy, weights = NULL, offset = NULL, init = NULL,
                            controlvals = survreg.control(),
                            dist = survreg.distributions$extreme))
"""


def make_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = (rng.random(n) < 0.5).astype(float)
    group = rng.integers(0, 3, size=n)
    event = rng.exponential(size=n) / np.exp(0.5 * x1 - 0.3 * x2 + 0.2 * x3)
    censor = rng.exponential(scale=2.0, size=n)
    # a coarse grid so ties are common, as in real data
    time_ = np.round(np.minimum(event, censor), 2) + 0.01
    status = (event <= censor).astype(np.int64)
    return pd.DataFrame(
        {"time": time_, "status": status, "x1": x1, "x2": x2, "x3": x3, "group": group}
    )


def best_of(fn: Callable[[], object], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def formula_timings(data: pd.DataFrame, repeat: int) -> dict[str, float]:
    from survival import r

    calls: dict[str, Callable[[], object]] = {
        "coxph": lambda: r.coxph("Surv(time, status) ~ x1 + x2 + x3", data, ties="efron"),
        "survfit": lambda: r.survfit("Surv(time, status) ~ 1", data),
        "survfit_strata": lambda: r.survfit("Surv(time, status) ~ group", data),
        "survdiff": lambda: r.survdiff("Surv(time, status) ~ group", data),
        "concordance": lambda: r.concordance("Surv(time, status) ~ x1", data),
        "survreg": lambda: r.survreg("Surv(time, status) ~ x1 + x2 + x3", data, dist="weibull"),
    }
    return {name: best_of(calls[name], repeat) for name in ROUTINES}


def kernel_timings(data: pd.DataFrame, repeat: int) -> dict[str, float]:
    from survival import _survival as core

    n = len(data)
    time_ = data["time"].to_numpy()
    status = data["status"].to_numpy()
    group = data["group"].to_numpy()
    x = data[["x1", "x2", "x3"]].to_numpy()
    x_intercept = np.column_stack([np.ones(n), x])
    x1 = data["x1"].to_numpy()
    weibull = core.SurvregDistribution("weibull")
    calls: dict[str, Callable[[], object]] = {
        "coxph": lambda: core.coxph_fit(time_, status, x, method="efron"),
        "survfit": lambda: core.survfitkm(time_, status),
        "survfit_strata": lambda: core.survfitkm(time_, status, strata=group),
        "survdiff": lambda: core.survdiff(time_, status, group),
        "concordance": lambda: core.concordancefit(
            core.SurvivalData(time_, status), core.CovariateMatrix(x1, n, 1)
        ),
        "survreg": lambda: core.survreg_fit(core.SurvregData(time_, status, x_intercept), weibull),
    }
    return {name: best_of(calls[name], repeat) for name in ROUTINES}


def r_timings(rscript: str, csv_path: Path, repeat: int) -> dict[str, dict[str, float]] | None:
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
    timings: dict[str, dict[str, float]] = {"formula": {}, "kernel": {}}
    for line in proc.stdout.splitlines():
        layer, routine, seconds = line.split()
        timings[layer][routine] = float(seconds)
    return timings


def find_rscript(explicit: str | None) -> str | None:
    for candidate in (explicit, os.environ.get("SURVIVAL_RSCRIPT"), shutil.which("Rscript")):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
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
            python = {
                "formula": formula_timings(data, args.repeat),
                "kernel": kernel_timings(data, args.repeat),
            }
            r = None
            if rscript:
                path = Path(tmp) / f"bench_{n}.csv"
                data.to_csv(path, index=False, float_format="%.17g")
                r = r_timings(rscript, path, args.repeat)
            for layer, timings in python.items():
                for routine, seconds in timings.items():
                    r_seconds = r[layer].get(routine) if r else None
                    rows_out.append(
                        {
                            "n": n,
                            "layer": layer,
                            "routine": routine,
                            "python_s": seconds,
                            "r_s": r_seconds,
                            "r_over_python": (
                                r_seconds / seconds if r_seconds and seconds > 0 else None
                            ),
                        }
                    )

    header = (
        f"{'n':>8} {'layer':<8} {'routine':<15} {'python (s)':>11} {'R (s)':>9} {'R/python':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows_out:
        r_txt = f"{row['r_s']:.4f}" if row["r_s"] is not None else "-"
        ratio = row["r_over_python"]
        ratio_txt = f"{ratio:.2f}" if ratio is not None else "-"
        print(
            f"{row['n']:>8} {row['layer']:<8} {row['routine']:<15} "
            f"{row['python_s']:>11.4f} {r_txt:>9} {ratio_txt:>9}"
        )
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
