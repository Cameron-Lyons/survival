#!/usr/bin/env python3
"""Time complete carry-forward calls after checking every output row.

Use --baseline-source with an earlier r/_data_prep.py to compare its Python
wrapper against the current one in the same process and native extension.
Input construction and validation of the numerical outputs are excluded from
timing; input materialization, native sorting and copying results are included.
"""

import argparse
import ast
import gc
import importlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import r

_data_prep = importlib.import_module("survival.r._data_prep")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def previous_lvcf(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "lvcf"
    )
    namespace = dict(vars(_data_prep))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- explicitly supplied repository revision
    return namespace[function.name]


def workload(rows, shuffled, first):
    ordinal = np.arange(rows)
    position = ordinal % 8
    values = np.array([None, None, 1, None, 0, None, 1, None], dtype=object)[position]
    expected = np.array([0 if first else None, 0 if first else None, 1, 1, 0, 0, 1, 1])[position]
    order = np.random.default_rng(7341).permutation(rows) if shuffled else ordinal
    return (
        (ordinal // 8)[order].tolist(),
        values[order].tolist(),
        position[order].tolist(),
        expected[order],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--samples", type=positive_int, default=7)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    previous = None if args.baseline_source is None else previous_lvcf(args.baseline_source)
    results = []
    for rows in args.rows:
        for shuffled in (False, True):
            for first in (True, False):
                ids, values, times, expected = workload(rows, shuffled, first)
                variants = {"current": r.lvcf}
                if previous is not None:
                    variants["previous"] = previous
                for function in variants.values():
                    np.testing.assert_array_equal(function(ids, values, times, first), expected)
                samples = {name: [] for name in variants}
                for sample in range(args.samples):
                    names = list(variants) if sample % 2 == 0 else list(reversed(variants))
                    for name in names:
                        gc.collect()
                        start = time.perf_counter_ns()
                        result = variants[name](ids, values, times, first)
                        samples[name].append((time.perf_counter_ns() - start) / 1_000_000)
                        del result
                results.append(
                    {
                        "rows": rows,
                        "shuffled": shuffled,
                        "first": first,
                        "variants": {
                            name: {"median_ms": statistics.median(values), "samples_ms": values}
                            for name, values in samples.items()
                        },
                    }
                )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "samples": args.samples,
                "baseline_source": None
                if args.baseline_source is None
                else str(args.baseline_source),
                "outputs_checked": True,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
