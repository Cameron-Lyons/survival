"""Complete Python Fine–Gray formula calls; setup and explicit GC are excluded."""

import argparse
import gc
import importlib.util
import json
import statistics
import sys
import time
from functools import partial

import numpy as np
from survival import r_api as r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline-source", help="previous repository _finegray.py to compare")
    args = parser.parse_args()
    if args.rows < 100 or args.repeats < 1:
        parser.error("rows must be at least 100 and repeats must be positive")
    rng = np.random.default_rng(812)
    data = {
        "time": rng.integers(1, 501, args.rows),
        "event": rng.integers(0, 3, args.rows),
        "x": rng.normal(size=args.rows),
        "group": np.arange(args.rows) % 100,
    }
    functions = {"current": r.finegray}
    if args.baseline_source:
        name = "survival.r._finegray_baseline"
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        if spec is None or spec.loader is None:
            parser.error("cannot load baseline source")
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[name] = baseline
        spec.loader.exec_module(baseline)
        functions["baseline"] = baseline.finegray
    results = {}
    for kind, rhs in [("right", "x"), ("strata", "x + strata(group)")]:
        calls = {
            name: partial(function, f"Surv(time, event, type='mstate') ~ {rhs}", data)
            for name, function in functions.items()
        }
        result = calls["current"]()
        if len(result["fgstart"]) < args.rows or not np.all(np.isfinite(result["fgwt"])):
            raise RuntimeError("invalid Fine–Gray expansion")
        rows = len(result["fgstart"])
        for name, call in calls.items():
            if name != "current":
                expected = call()
                for column in result:
                    np.testing.assert_allclose(result[column], expected[column], rtol=1e-12)
                del expected
            for _ in range(2):
                call()
        del result
        elapsed = {name: [] for name in calls}
        for sample in range(args.repeats):
            order = list(calls) if sample % 2 == 0 else list(reversed(calls))
            for name in order:
                gc.collect()
                started = time.perf_counter()
                result = calls[name]()
                elapsed[name].append(1000 * (time.perf_counter() - started))
                del result
        results[kind] = {
            "expanded_rows": rows,
            "timing": {
                name: {"median_ms": statistics.median(values), "samples_ms": values}
                for name, values in elapsed.items()
            },
        }
    print(json.dumps({"rows": args.rows, "repeats": args.repeats, "results": results}, indent=2))


if __name__ == "__main__":
    main()
