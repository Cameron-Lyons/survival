"""Check and time complete calls to the native aggregate.survfit port.

Run with the Python extension installed, for example:
    .venv/bin/python scripts/benchmark_aggregate_survfit.py --columns 100000
"""

import argparse
import json
import platform
import statistics
import time

import numpy as np
from survival.surv_analysis import GroupingFactor, aggregate_survfit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=100000)
    parser.add_argument("--times", type=int, default=20)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    if min(args.columns, args.times, args.groups, args.repeat) < 1:
        parser.error("columns, times, groups and repeat must be positive")
    if args.groups > args.columns:
        parser.error("groups must be no greater than columns")

    t = np.arange(args.times)[:, None]
    j = np.arange(args.columns)[None, :]
    surv = ((j * 7919 + t * 101) % 100003) / 100003.0
    pstate = np.stack(
        [((j * 7919 + t * 101 + s * 37) % 100003) / 100003.0 for s in range(3)],
        axis=2,
    )
    codes = (np.arange(args.columns) % args.groups).tolist()
    by = [GroupingFactor(codes, [str(g) for g in range(args.groups)])]
    expected_surv = np.stack(
        [
            np.median(surv[:, np.arange(args.columns) % args.groups == g], axis=1)
            for g in range(args.groups)
        ],
        axis=1,
    )
    expected_pstate = np.stack(
        [
            np.median(pstate[:, np.arange(args.columns) % args.groups == g], axis=1)
            for g in range(args.groups)
        ],
        axis=1,
    )
    pstate_list = pstate.tolist()
    pstate_fortran = np.asfortranarray(pstate)
    cases = [
        (
            "median",
            lambda: aggregate_survfit(surv=surv, fun="median"),
            "surv",
            np.median(surv, axis=1)[:, None],
        ),
        (
            "grouped_median",
            lambda: aggregate_survfit(surv=surv, by=by, fun="median"),
            "surv",
            expected_surv,
        ),
        (
            "grouped_pstate_median",
            lambda: aggregate_survfit(pstate=pstate_list, by=by, fun="median"),
            "pstate",
            expected_pstate,
        ),
        (
            "grouped_pstate_numpy_median",
            lambda: aggregate_survfit(pstate=pstate, by=by, fun="median"),
            "pstate",
            expected_pstate,
        ),
        (
            "grouped_pstate_fortran_median",
            lambda: aggregate_survfit(pstate=pstate_fortran, by=by, fun="median"),
            "pstate",
            expected_pstate,
        ),
    ]
    results = []
    for name, call, component, expected in cases:
        np.testing.assert_allclose(getattr(call(), component), expected, rtol=0, atol=1e-14)
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = call()
            samples.append(time.perf_counter() - start)
            del result
        results.append(
            {"case": name, "median_seconds": statistics.median(samples), "samples_seconds": samples}
        )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "inputs": vars(args),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
