"""Complete person-years calls for equivalent Surv, cbind and numeric matrix responses."""

import argparse
import gc
import json
import statistics
import time
from functools import partial

import numpy as np
from survival import r_api as r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    rng = np.random.default_rng(420)
    data = {
        "time": rng.uniform(1, 100, args.rows),
        "event": rng.integers(0, 2, args.rows),
        "group": np.arange(args.rows) % 10,
        "weight": rng.uniform(0.5, 2, args.rows),
    }
    data["Y"] = np.column_stack((data["time"], data["event"]))
    formulas = {
        "surv": "Surv(time, event) ~ group",
        "cbind": "cbind(time, event) ~ group",
        "matrix": "Y ~ group",
    }
    results = {}
    for selection in ("all", "subset_missing"):
        if selection == "subset_missing":
            data["time"][::11] = np.nan
            data["Y"][::11, 0] = np.nan
        subset = None if selection == "all" else np.arange(0, args.rows, 2)
        calls = {
            name: partial(r.pyears, formula, data, weights="weight", subset=subset)
            for name, formula in formulas.items()
        }
        reference = calls["surv"]()
        for call in calls.values():
            actual = call()
            for field in ("pyears", "event", "n", "offtable"):
                np.testing.assert_allclose(
                    getattr(actual, field), getattr(reference, field), rtol=1e-12
                )
            if actual.na_action != reference.na_action:
                raise ValueError("missing-row metadata differs")
            for _ in range(3):
                call()
        elapsed = {name: [] for name in calls}
        for sample in range(args.repeats):
            order = list(calls) if sample % 2 == 0 else list(reversed(calls))
            for name in order:
                gc.collect()
                started = time.perf_counter()
                result = calls[name]()
                elapsed[name].append(1000 * (time.perf_counter() - started))
                del result
        results[selection] = {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in elapsed.items()
        }
    print(
        json.dumps(
            {"rows": args.rows, "repeats": args.repeats, "warmup_calls": 3, "results": results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
