"""Compare bounded Aalen influence reduction with conversion and a squared cube."""

import argparse
import json
import platform
import tracemalloc
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np
from survival._aalen_plot import _influence_variance


def full_cube(values):
    cube = np.asarray(values, dtype=float)
    return np.sum(cube * cube, axis=0).T


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        call()
        samples.append(1000 * (perf_counter() - started))
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"ms": median(samples), "peak_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--times", type=int, default=1000)
    parser.add_argument("--terms", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.groups, args.times, args.terms, args.repeats) < 1:
        parser.error("sizes and repeats must be positive")
    results = []
    for groups in args.groups:
        cube = np.random.default_rng(123).normal(size=(groups, args.terms, args.times))
        for layout in ("array", "list"):
            values = cube if layout == "array" else cube.tolist()
            bounded = partial(
                _influence_variance,
                values,
                args.terms,
                args.times,
                args.times,
                list(range(args.terms)),
            )
            baseline = partial(full_cube, values)
            np.testing.assert_allclose(bounded(), baseline(), rtol=1e-12)
            results.append(
                {
                    "groups": groups,
                    "terms": args.terms,
                    "times": args.times,
                    "layout": layout,
                    "bounded": measure(bounded, args.repeats),
                    "full_cube": measure(baseline, args.repeats),
                }
            )
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
