"""Benchmark built-in AFT fits through the Python extension.

PYTHONPATH=python .venv/bin/python scripts/bench_survreg.py
Input conversion and a warmup fit are excluded. Each fit includes the
intercept-only initialization, optimization and result construction.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import survival


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=positive_int, default=11)
    parser.add_argument(
        "--distributions", nargs="+", default=["lognormal", "weibull", "loglogistic"]
    )
    parser.add_argument(
        "--extension", type=Path, help="load a saved extension build for comparison"
    )
    args = parser.parse_args()
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must name a loadable extension library")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["survival._survival"] = module
        survival._survival = module
    from survival import _survival, regression

    extension = Path(_survival.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    for n in args.n:
        rng = np.random.default_rng(492)
        x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        times = np.exp(1 + x[:, 1] * 0.2 + rng.normal(size=n) * 0.6)
        status = (rng.random(n) > 0.2).astype(np.int32)
        data = regression.SurvregData(times, status, x)
        for name in args.distributions:
            distribution = regression.SurvregDistribution(name)
            fit = regression.survreg_fit(data, distribution)
            samples = []
            for _ in range(args.repeats):
                before = time.perf_counter_ns()
                fit = regression.survreg_fit(data, distribution)
                samples.append((time.perf_counter_ns() - before) / 1e6)
            if not fit.converged:
                raise RuntimeError(f"{name}, n={n}: fit did not converge")
            results.append(
                {
                    "distribution": name,
                    "n": n,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "coefficients": fit.coefficients,
                    "log_likelihood": fit.log_likelihood,
                }
            )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "extension": str(extension),
                "extension_sha256": digest,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
