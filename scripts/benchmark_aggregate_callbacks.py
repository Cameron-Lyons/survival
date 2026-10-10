"""Check complete native and formula-facing aggregation calls before timing them."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np
from survival import r
from survival.surv_analysis import GroupingFactor, aggregate_survfit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=40000)
    parser.add_argument("--times", type=int, default=8)
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
    pstate = np.stack([surv, 1.0 - surv], axis=2)
    codes = np.arange(args.columns) % args.groups
    by = [GroupingFactor(codes.tolist(), [str(g) for g in range(args.groups)])]
    curves = SimpleNamespace(surv=surv, pstate=pstate)
    results = []
    for name, reducer in (("mean", np.mean), ("sum", np.sum)):
        expected_surv = np.stack(
            [reducer(surv[:, codes == g], axis=1) for g in range(args.groups)], axis=1
        )
        expected_pstate = np.stack(
            [reducer(pstate[:, codes == g], axis=1) for g in range(args.groups)], axis=1
        )
        for surface in ("native", "facade"):
            for callback in (False, True):
                fun = reducer if callback else name

                def call(*, selected_surface=surface, selected_fun=fun):
                    if selected_surface == "native":
                        return aggregate_survfit(surv=surv, pstate=pstate, by=by, fun=selected_fun)
                    return r.aggregate_survfit(curves, by=codes, FUN=selected_fun)

                checked = call()
                np.testing.assert_allclose(checked.surv, expected_surv, rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose(checked.pstate, expected_pstate, rtol=1e-13, atol=1e-14)
                expected_labels = [[str(g)] for g in range(args.groups)]
                if checked.newdata is None or checked.newdata.labels != expected_labels:
                    raise AssertionError("group labels do not match the output data margin")
                samples = []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    result = call()
                    samples.append(time.perf_counter() - start)
                    del result
                results.append(
                    {
                        "case": f"{surface}_{name}_{'callback' if callback else 'builtin'}",
                        "median_seconds": statistics.median(samples),
                        "samples_seconds": samples,
                    }
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
