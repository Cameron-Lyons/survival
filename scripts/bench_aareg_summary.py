"""Measure bounded robust Aalen covariance against whole-cube conversion."""

import argparse
import json
import platform
import tracemalloc
from functools import partial
from statistics import median
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from survival.r._aareg import _summary_influence_covariance


def whole_cube(values, weights):
    weighted = np.einsum("gpt,pt->gp", np.asarray(values, dtype=float), weights)
    return weighted.T @ weighted


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append(1000 * (perf_counter() - start))
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"ms": median(samples), "peak_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, nargs="+", default=[100, 2000])
    parser.add_argument("--times", type=int, default=400)
    parser.add_argument("--terms", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(*args.groups, args.times, args.terms, args.repeats) < 1:
        parser.error("sizes and repeats must be positive")
    results = []
    times = list(range(args.times))
    weights = np.linspace(0.5, 2, args.times * args.terms).reshape(args.times, args.terms)
    weight_list = weights.tolist()
    for groups in args.groups:
        cube = np.random.default_rng(123).normal(size=(groups, args.terms, args.times))
        for layout in ("array", "list"):
            values = cube if layout == "array" else cube.tolist()
            bounded = partial(
                _summary_influence_covariance, SimpleNamespace(dfbeta=values), times, weight_list
            )
            baseline = partial(whole_cube, values, weights.T)
            np.testing.assert_allclose(bounded(), baseline(), rtol=1e-12, atol=1e-9)
            results.append(
                {
                    "groups": groups,
                    "terms": args.terms,
                    "times": args.times,
                    "layout": layout,
                    "bounded": measure(bounded, args.repeats),
                    "whole_cube": measure(baseline, args.repeats),
                }
            )
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
