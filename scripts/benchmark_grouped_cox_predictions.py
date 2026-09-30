#!/usr/bin/env python3
"""Complete Python terms predictions, output comparison and process peak memory."""

import argparse
import gc
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from survival import r


def peak_rss_mib():
    units = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--mode", choices=["baseline", "current"], default="current")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(719)
    x = rng.normal(size=(2000, args.columns))
    data = {f"x{j}": x[:, j] for j in range(args.columns)}
    data["time"] = rng.exponential(size=len(x))
    data["status"] = rng.integers(0, 2, size=len(x))
    fit = r.coxph("Surv(time,status) ~ " + "+".join(f"x{j}" for j in range(args.columns)), data)
    newx = rng.normal(size=(args.rows, args.columns))
    newdata = {f"x{j}": newx[:, j] for j in range(args.columns)}
    group = np.arange(args.rows, dtype=np.int32) % 257
    setup_peak = peak_rss_mib()

    def call():
        return r.predict(fit, newdata, type="terms", se_fit=True, collapse=group)

    result = call()
    actual = {"fit": np.asarray(result.fit), "se_fit": np.asarray(result.se_fit)}
    equal = None
    if args.compare is not None:
        with np.load(args.compare.with_suffix(".npz")) as reference:
            for key, value in actual.items():
                np.testing.assert_allclose(value, reference[key], rtol=2e-13, atol=1e-12)
            equal = all(np.array_equal(value, reference[key]) for key, value in actual.items())
    np.savez(args.output.with_suffix(".npz"), **actual)
    for _ in range(3):
        call()
    samples = []
    for _ in range(args.samples):
        gc.collect()
        start = time.perf_counter()
        call()
        samples.append(1000 * (time.perf_counter() - start))
    output = {
        "mode": args.mode,
        "rows": args.rows,
        "columns": args.columns,
        "groups": 257,
        "training_rows": 2000,
        "warmups": 3,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "range_ms": [min(samples), max(samples)],
        "setup_peak_rss_mib": setup_peak,
        "process_peak_rss_mib": peak_rss_mib(),
        "whole_outputs_equal": equal,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scope": (
            "Complete public Python new-data terms prediction with errors and grouping, "
            "including formula evaluation, encoding, conversion, native calculation and "
            "result materialization. Fitting, input creation and explicit GC excluded "
            "from timing. Peak RSS includes the complete process, fitting and inputs; "
            "it does not isolate temporary allocations."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
