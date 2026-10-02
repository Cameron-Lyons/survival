"""Measure complete shared formula-frame preparation, excluding model fitting.

Run with PYTHONPATH=python .venv/bin/python scripts/bench_formula_model_frame.py.
Numeric NumPy input and its list equivalent must agree before each timing.
"""

import argparse
import json
import platform
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np
from survival.r._fit import _model_frame


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append(1000 * (perf_counter() - start))
    return median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--terms", type=int, nargs="+", default=[2, 16])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(*args.sizes, *args.terms, args.repeats) < 1:
        parser.error("sizes, terms and repeats must be positive")
    results = []
    for size in args.sizes:
        for terms in args.terms:
            rng = np.random.default_rng(123)
            arrays = {
                "time": rng.exponential(10, size=size),
                "status": rng.integers(0, 2, size=size),
                **{f"x{i}": rng.normal(size=size) for i in range(terms)},
            }
            lists = {name: values.tolist() for name, values in arrays.items()}
            formula = "Surv(time, status) ~ " + " + ".join(f"x{i}" for i in range(terms))
            prepare_array = partial(_model_frame, formula, arrays)
            prepare_list = partial(_model_frame, formula, lists)
            array_frame, list_frame = prepare_array(), prepare_list()
            np.testing.assert_array_equal(array_frame.x, list_frame.x)
            if any(
                (
                    array_frame.y.time != list_frame.y.time,
                    array_frame.y.event != list_frame.y.event,
                    array_frame.y.type != list_frame.y.type,
                    array_frame.names != list_frame.names,
                    array_frame.assign != list_frame.assign,
                    array_frame.na_action != list_frame.na_action,
                )
            ):
                raise AssertionError("array and list formula frames differ")
            del array_frame, list_frame
            results.append(
                {
                    "rows": size,
                    "terms": terms,
                    "array_ms": measure(prepare_array, args.repeats),
                    "list_ms": measure(prepare_list, args.repeats),
                }
            )
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
