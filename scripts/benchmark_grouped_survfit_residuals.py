#!/usr/bin/env python3
"""Complete formula residual/pseudo calls with many strata and a balanced control.

Fitting the facade object is excluded; each measured call reevaluates its formula
frame, refits the native curve, computes residuals/pseudo values and materializes
all rows. Analytic full-array checks run before timing. --baseline-source extracts
the old Python pseudo function only, so both variants share the installed native
extension. --compare verifies every stored output against a previous native run.
"""

import argparse
import ast
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival as native
from survival import r_api as r
from survival.r import _survfit_residuals


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def previous_pseudo(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pseudo"
    )
    namespace = dict(vars(_survfit_residuals))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- explicitly supplied repository revision
    return namespace[function.name]


def workload(rows, group_mode, multistate):
    pair = np.arange(rows) // 2
    codes = pair if group_mode == "many" else pair % 64
    group_count = int(codes.max()) + 1
    status = np.tile([1, 0], rows // 2)
    data = {"time": np.tile([1.0, 2.0], rows // 2), "event": status, "group": codes}
    if multistate:
        data["event"] = r._r_factor(
            ["event" if value else "censor" for value in status], ["censor", "event"]
        )
    fit = r.survfit("Surv(time,event) ~ group", data, se_fit=False, timefix=False)
    # A stratum with m event/censor pairs has event residual -1/(4m),
    # censor residual +1/(4m), and pseudo values zero/one respectively.
    counts = np.bincount(codes)
    residual = np.where(status, -0.5, 0.5) / counts[codes]
    pseudo = (1 - status).astype(float)
    if multistate:
        residual = np.stack((residual, -residual), axis=1)[:, :, None]
        pseudo = np.stack((pseudo, 1 - pseudo), axis=1)
    else:
        residual = residual[:, None]
    return fit, residual, pseudo, codes, group_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, nargs="+", default=[1000, 4000, 16000, 32000])
    parser.add_argument("--samples", type=positive_int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--mode", default="current")
    args = parser.parse_args()
    if any(rows % 2 for rows in args.rows):
        parser.error("rows must be even (one event/censor pair per two input rows)")
    native_digest = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    baseline = None if args.baseline_source is None else previous_pseudo(args.baseline_source)
    previous = None
    if args.compare is not None:
        with np.load(args.compare.with_suffix(".npz")) as archive:
            previous = {name: archive[name] for name in archive.files}
    outputs, results = {}, []
    for rows in args.rows:
        for group_mode in ("many", "fixed64"):
            for multistate in (False, True):
                kind = "aj" if multistate else "km"
                fit, residual, pseudo, codes, groups = workload(rows, group_mode, multistate)
                calls = {
                    "residual": lambda fit=fit: r.survfit_residuals(fit, times=[1.5]),
                    "pseudo": lambda fit=fit: r.pseudo(fit, times=[1.5]),
                }
                if baseline is not None:
                    calls["previous_python_pseudo"] = lambda fit=fit: baseline(fit, times=[1.5])
                expected = {
                    "residual": residual,
                    "pseudo": pseudo,
                    "previous_python_pseudo": pseudo,
                }
                for operation, call in calls.items():
                    result = call()
                    values = np.asarray(result.resid if operation == "residual" else result)
                    np.testing.assert_allclose(values, expected[operation], rtol=2e-12, atol=2e-13)
                    if operation == "residual":
                        np.testing.assert_array_equal(result.id, np.arange(1, rows + 1))
                        np.testing.assert_array_equal(
                            result.curve, codes + 1 if groups > 1 else None
                        )
                        np.testing.assert_array_equal(result.time, [1.5])
                        np.testing.assert_array_equal(
                            result.columns, ["(s0)", "event"] if multistate else None
                        )
                    key = f"{kind}_{group_mode}_{rows}_{operation}"
                    if previous is not None:
                        np.testing.assert_array_equal(values, previous[key])
                    outputs[key] = values
                samples = {name: [] for name in calls}
                for sample in range(args.samples):
                    names = list(calls) if sample % 2 == 0 else list(reversed(calls))
                    for name in names:
                        gc.collect()
                        start = time.perf_counter_ns()
                        result = calls[name]()
                        samples[name].append((time.perf_counter_ns() - start) / 1_000_000)
                        del result
                results.append(
                    {
                        "kind": kind,
                        "rows": rows,
                        "group_mode": group_mode,
                        "groups": groups,
                        "variants": {
                            name: {"median_ms": statistics.median(times), "samples_ms": times}
                            for name, times in samples.items()
                        },
                    }
                )
    if previous is not None and outputs.keys() != previous.keys():
        raise AssertionError("baseline output keys differ from the current workload")
    np.savez(args.output.with_suffix(".npz"), **outputs)
    report = {
        "mode": args.mode,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native_extension_sha256": native_digest,
        "samples": args.samples,
        "baseline_source": None if args.baseline_source is None else str(args.baseline_source),
        "whole_outputs_equal": None if previous is None else True,
        "scope": __doc__,
        "results": results,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
