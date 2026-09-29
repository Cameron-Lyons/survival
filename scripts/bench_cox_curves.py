"""Compare full Cox survival-curve calls across release builds.

Run with PYTHONPATH=python .venv/bin/python scripts/bench_cox_curves.py.
Fitting, input preparation, baseline warmup and result checks are not timed.
The output includes raw timings and a numerical digest for each workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
from survival import _survival, regression


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, default=20_000)
    parser.add_argument("--columns", type=positive_int, default=12)
    parser.add_argument("--new-rows", type=positive_int, default=8)
    parser.add_argument("--intervals", type=positive_int, default=1000)
    parser.add_argument("--subjects", type=positive_int, default=1000)
    parser.add_argument("--repeats", type=positive_int, default=7)
    args = parser.parse_args()
    if args.n < args.columns + 2:
        parser.error("--n must be at least --columns + 2")
    rng = np.random.default_rng(20260929)
    times = np.arange(1, args.n + 1, dtype=float)
    x = rng.normal(size=(args.n, args.columns))
    status = (np.arange(args.n) % 3 != 0).astype(np.int32)
    # One evaluation at beta=0 isolates prediction performance from convergence.
    fit = regression.coxph_fit(times, status, x, method="efron", iter_max=0)
    ordinary = rng.normal(size=(args.new_rows, args.columns))
    boundaries = np.linspace(0, args.n, args.intervals + 1)
    changing = rng.normal(size=(args.intervals, args.columns))
    # Interleave two short intervals for every subject, in first-appearance order.
    subject_start = rng.integers(0, args.n - 2, size=args.subjects).astype(float)
    short_start = np.concatenate([subject_start, subject_start + 1])
    short_x = rng.normal(size=(args.subjects * 2, args.columns))
    ids = np.tile(np.arange(args.subjects, dtype=np.int32)[::-1], 2)
    calls = {
        "ordinary": partial(fit.survfit, ordinary),
        "one_changing_subject": partial(
            fit.survfit_individual,
            changing,
            boundaries[:-1],
            boundaries[1:],
            np.zeros(args.intervals, dtype=np.int32),
        ),
        "many_short_subjects": partial(
            fit.survfit_individual, short_x, short_start, short_start + 1, ids
        ),
    }
    results = []
    for name, call in calls.items():
        for se_fit in (False, True):
            result = call(se_fit=se_fit)
            samples = []
            for _ in range(args.repeats):
                del result
                before = time.perf_counter_ns()
                result = call(se_fit=se_fit)
                samples.append((time.perf_counter_ns() - before) / 1_000_000)
            digest = hashlib.sha256()
            for curve in result:
                for field in ("time", "n_risk", "n_event", "n_censor", "surv", "cumhaz"):
                    digest.update(np.asarray(getattr(curve, field), dtype="<f8").tobytes())
                if se_fit:
                    digest.update(np.asarray(curve.std_err, dtype="<f8").tobytes())
            results.append(
                {
                    "workload": name,
                    "se_fit": se_fit,
                    "curves": len(result),
                    "time_rows": sum(len(curve.time) for curve in result),
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "result_sha256": digest.hexdigest(),
                }
            )
            del result
    extension = Path(_survival.__file__).resolve()
    with extension.open("rb") as source:
        digest_hex = hashlib.file_digest(source, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "extension_sha256": digest_hex,
                "parameters": vars(args),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
