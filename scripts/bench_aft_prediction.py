"""Benchmark AFT prediction after fitting and constructing inputs.

Select each complete Python package and its native extension via PYTHONPATH:
    PYTHONPATH=python .venv/bin/python scripts/bench_aft_prediction.py --n 10000 100000

Native/Python calls include argument conversion and returned matrices. NumPy
covariate multiplication is included for sklearn. Fitting, input construction,
and result disposal are excluded. Student-t native quantiles require the fixed
prediction implementation; the default families also run on older packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import survival
from survival import _survival as native
from survival.sklearn_compat import AFTEstimator


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def measure(call, repeats: int):
    result = call()
    samples = []
    for _ in range(repeats):
        del result
        before = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - before) / 1_000_000)
    return samples, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=positive_int, default=[10000, 100000])
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--quantiles", type=positive_int, default=9)
    parser.add_argument("--distributions", nargs="+", default=["weibull", "gaussian"])
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    probabilities = [(j + 1) / (args.quantiles + 1) for j in range(args.quantiles)]
    extension_path = Path(native.__file__).resolve()
    with extension_path.open("rb") as source:
        extension_hash = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "label": args.label,
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "package_path": str(Path(survival.__file__).resolve()),
        "extension_path": str(extension_path),
        "extension_sha256": extension_hash,
        "repeats": args.repeats,
        "warmups": 1,
        "probabilities": probabilities,
        "timing": (
            "prediction call including conversions and result construction; "
            "excluding fitting, input construction, and result disposal"
        ),
        "results": [],
    }
    training_x = np.array([[(j % 13) / 6 - 1] for j in range(104)])
    training_time = np.exp(1 + 0.2 * training_x[:, 0] + 0.1 * np.sin(np.arange(104)))
    training_y = np.column_stack([training_time, np.ones(104)])
    for distribution in args.distributions:
        estimator = AFTEstimator(distribution=distribution).fit(training_x, training_y)
        fitted = estimator.model_
        for n in args.n:
            x = np.array([[(j % 31) / 15 - 1] for j in range(n)])
            rows = np.column_stack([np.ones(n), x]).tolist()
            calls = {
                "native_quantile": partial(fitted.predict_quantile, rows, probabilities),
                "python_quantile": partial(
                    survival.predict, fitted, rows, type="quantile", p=probabilities
                ),
                "python_quantile_se": partial(
                    survival.predict, fitted, rows, type="quantile", p=probabilities, se_fit=True
                ),
                "sklearn_response": partial(estimator.predict, x),
                "sklearn_quantile": partial(estimator.predict_quantile, x, q=0.9),
            }
            for operation, call in calls.items():
                samples, result = measure(call, args.repeats)
                values = getattr(result, "predictions", result)
                report["results"].append(
                    {
                        "distribution": distribution,
                        "n": n,
                        "operation": operation,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "samples_ms": samples,
                        "first_prediction": np.asarray(values[0]).tolist(),
                        "last_prediction": np.asarray(values[-1]).tolist(),
                    }
                )
                del result, values
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
