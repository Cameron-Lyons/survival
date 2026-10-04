"""Measure formula frames and fitter preparation without fitting a model.

Run ``.venv/bin/python scripts/benchmark_typed_formulas.py --output result.json``.
An archived Python source tree can be measured with ``--python-path PATH --legacy``
when it shares the same native extension. Use ``--cpu N`` on Linux to pin both runs.
Reported wall and process CPU times exclude setup and semantic checks.
"""

import argparse
import gc
import importlib
import json
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

CASES = {
    "numpy_additive": "x + z",
    "logical_comparison": "x + I(z > 0)",
    "numeric_wrapper": "x + I(z^2)",
    "boolean_and": "x + I(a & b)",
    "nested_identity_offset": "x + I(identity(I(a | b))) + offset(identity(I(a & b)))",
}


def expected_column(name, data):
    if name == "numpy_additive":
        return data["z"]
    if name == "numeric_wrapper":
        return data["z"] ** 2
    if name == "logical_comparison":
        return np.where(np.isnan(data["z"]), np.nan, data["z"] > 0)
    logical = data["a"] & data["b"] if name == "boolean_and" else data["a"] | data["b"]
    return logical.astype(float)


def build(formula_module, formula, data, action="na.omit"):
    frame = formula_module.model_frame(formula, data, na_action=action)
    design = formula_module._fit_formula_design(frame.data, frame.spec, frame.terms, frame.n)
    matrix = formula_module._design_array_from_spec(
        frame.data, design, frame.n, allow_missing=action == "na.pass"
    )
    return frame, design, matrix


def check_semantics(module, name, formula, data):
    frame, design, matrix = build(module, formula, data)
    expected = np.column_stack((data["x"], expected_column(name, data)))
    np.testing.assert_allclose(matrix, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(np.isnan(matrix), np.isnan(expected))
    variables = module._model_variables(frame)
    logical = name in {"logical_comparison", "boolean_and", "nested_identity_offset"}
    for index, (_label, values) in enumerate(variables):
        kind = getattr(values, "kind", None)
        if kind is not None:
            np.testing.assert_equal(kind, "logical" if logical and index else "numeric")
    small = {key: value[:5].copy() for key, value in data.items()}
    small["x"][1] = np.nan
    small["z"][2] = np.nan
    missing_expected = np.column_stack((small["x"], expected_column(name, small)))
    if name in {"boolean_and", "nested_identity_offset"}:
        small["a"] = np.array([True, None, False, None, True], dtype=object)
        small["b"] = np.array([True, False, True, True, None], dtype=object)
        missing_expected[:, 1] = (
            [1, 0, 0, np.nan, np.nan] if name == "boolean_and" else [1, np.nan, 1, 1, 1]
        )
    missing_frame, _design, missing_matrix = build(module, formula, small, "na.pass")
    np.testing.assert_allclose(missing_matrix, missing_expected, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(missing_matrix), np.isnan(missing_expected))
    if name == "nested_identity_offset":
        np.testing.assert_array_equal(frame.offset, data["a"] & data["b"])
        np.testing.assert_allclose(missing_frame.offset, [1, 0, 0, np.nan, np.nan], equal_nan=True)
    return (
        frame,
        design,
        {
            "variables": [
                {"name": label, "kind": getattr(value, "kind", None)} for label, value in variables
            ],
            "columns": [
                column
                for term in design.covariates
                for column in module._design_term_output_names(term)
            ],
            "matrix_shape": list(matrix.shape),
        },
    )


def measure(operation, repetitions):
    operation()
    wall, cpu = [], []
    for _ in range(repetitions):
        gc.collect()
        gc.disable()
        try:
            started, cpu_started = time.perf_counter(), time.process_time()
            value = operation()
            cpu.append(time.process_time() - cpu_started)
            wall.append(time.perf_counter() - started)
        finally:
            gc.enable()
        del value
    return {
        "median_seconds": statistics.median(wall),
        "cpu_median_seconds": statistics.median(cpu),
        "samples_seconds": wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument(
        "--python-path", type=Path, default=Path(__file__).resolve().parents[1] / "python"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Measure only paths supported before typed expressions",
    )
    args = parser.parse_args()
    if args.rows < 5 or args.repeat < 1:
        parser.error("--rows must be at least 5 and --repeat positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    sys.path.insert(0, str(args.python_path.resolve()))
    module = importlib.import_module("survival.r._formula")
    native = importlib.import_module("survival._survival")
    index = np.arange(args.rows)
    data = {
        "time": (index + 1).astype(float),
        "status": (index % 3 != 0).astype(np.int8),
        "x": np.sin(index * 0.013),
        "z": np.cos(index * 0.031),
        "a": index % 2 == 0,
        "b": index % 3 == 0,
    }
    results = {
        "module": module.__file__,
        "native": native.__file__,
        "n": args.rows,
        "samples": args.repeat,
        "cases": {},
    }
    for name, rhs in CASES.items():
        if args.legacy and name in {"boolean_and", "nested_identity_offset"}:
            continue
        formula = "Surv(time,status) ~ " + rhs
        frame, design, summary = check_semantics(module, name, formula, data)
        operations = {
            "model_frame": partial(module.model_frame, formula, data),
            "frame_design_matrix": lambda formula=formula: build(module, formula, data)[2],
            "fit_design_only": partial(
                module._fit_formula_design, frame.data, frame.spec, frame.terms, frame.n
            ),
            "retained_frame_only": partial(
                module._formula_model_frame, frame.data, frame.response, design
            ),
        }
        summary.update(
            {label: measure(operation, args.repeat) for label, operation in operations.items()}
        )
        results["cases"][name] = summary
    text = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
