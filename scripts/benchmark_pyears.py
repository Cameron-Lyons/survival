"""Complete Python person-years calls with optional previous facade comparison."""

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
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--baseline-source", help="previous repository _pyears.py to compare")
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    functions = {"current": r.pyears}
    if args.baseline_source:
        name = "survival.r._pyears_baseline"
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        if spec is None or spec.loader is None:
            parser.error("cannot load baseline source")
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[name] = baseline
        spec.loader.exec_module(baseline)
        functions["baseline"] = baseline.pyears
    rng = np.random.default_rng(915)
    data = {
        "time": rng.uniform(1, 1000, args.rows),
        "event": rng.integers(0, 2, args.rows),
        "sex": rng.integers(1, 3, args.rows),
        "group": np.arange(args.rows) % 10,
        "age": rng.uniform(40, 80, args.rows) * 365.25,
    }
    results = {}
    for kind, formula in [
        ("fixed", "Surv(time, event) ~ group + sex"),
        ("tcut", "Surv(time, event) ~ group + tcut(age, c(0,50,60,70,100)*365.25)"),
        ("direct", None),
    ]:
        calls = {
            name: partial(function, time=data["time"], event=data["event"], group=data["group"])
            if formula is None
            else partial(function, formula, data)
            for name, function in functions.items()
        }
        result = calls["current"]()
        for name, call in calls.items():
            if name != "current":
                expected = call()
                for field in ("pyears", "n", "event", "offtable"):
                    np.testing.assert_allclose(
                        getattr(result, field), getattr(expected, field), rtol=1e-12
                    )
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
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in elapsed.items()
        }
    print(json.dumps({"rows": args.rows, "repeats": args.repeats, "results": results}, indent=2))


if __name__ == "__main__":
    main()
