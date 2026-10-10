"""Compare complete native DPQR calls with a saved predecessor extension.

Run against matching release feature sets on an otherwise idle machine:
    PYTHONPATH=python python scripts/benchmark_survreg_dpqr.py \\
        --baseline-extension /path/to/predecessor/_survival.so
The benchmark validates outputs before timing and includes argument conversion,
callback work and returned Python lists. Inputs and distributions are prepared
before timing. JSON output retains every alternating timing sample.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival as current


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(f"{name}._survival", path)
    if spec is None or spec.loader is None:
        raise ValueError("baseline-extension must be a native library for this Python ABI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def callback_density(z):
    z = np.asarray(z)
    probability = 1 / (1 + np.exp(-z))
    density = probability * (1 - probability)
    score = 1 - 2 * probability
    return np.column_stack((probability, 1 - probability, density, score, score**2 - 2 * density))


def callback_quantile(p):
    p = np.asarray(p)
    return np.log(p / (1 - p))


def callback_init(y, weights):
    return [0.0, 1.0]


def callback_deviance(y, scale):
    return {"center": y[:, 0], "loglik": np.zeros(len(y))}


parser = argparse.ArgumentParser()
parser.add_argument("--baseline-extension", type=Path, required=True)
parser.add_argument("--candidate-extension", type=Path)
parser.add_argument("--repeats", type=int, default=11)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("repeats must be positive")
engines = {
    "before": load_extension("dpqr_before", args.baseline_extension),
    "after": current
    if args.candidate_extension is None
    else load_extension("dpqr_after", args.candidate_extension),
}
cpus = os.sched_getaffinity(0)
os.sched_setaffinity(0, {min(cpus)})
results = []
for n in (1000, 100000):
    for family in ("gaussian", "weibull", "logistic", "callback"):
        distributions = {
            name: engine.SurvregDistribution.from_callbacks(
                name="logistic callback",
                init=callback_init,
                density=callback_density,
                deviance=callback_deviance,
                quantile=callback_quantile,
                variance=lambda: math.pi**2 / 3,
            )
            if family == "callback"
            else engine.SurvregDistribution(family)
            for name, engine in engines.items()
        }
        for operation in ("density", "cdf", "quantile"):
            x = np.linspace(0.001, 0.999, n) if operation == "quantile" else np.linspace(0.2, 3, n)
            for vector in (False, True):
                mean = np.linspace(-0.2, 0.3, n) if vector else np.array([0.1])
                scale = np.linspace(0.7, 1.4, n) if vector else np.array([1.2])
                method = {
                    "density": "pdf_values",
                    "cdf": "cdf_values",
                    "quantile": "quantile_values",
                }[operation]
                calls = {
                    name: lambda dist=dist, method=method, x=x, mean=mean, scale=scale: getattr(
                        dist, method
                    )(x, mean, scale)
                    for name, dist in distributions.items()
                }
                np.testing.assert_allclose(calls["before"](), calls["after"](), rtol=2e-14, atol=0)
                samples = {name: [] for name in engines}
                for pair in range(args.repeats):
                    for name in tuple(engines) if pair % 2 == 0 else tuple(reversed(engines)):
                        start = time.perf_counter_ns()
                        value = calls[name]()
                        samples[name].append((time.perf_counter_ns() - start) / 1e6)
                        del value
                row = {
                    "n": n,
                    "family": family,
                    "operation": operation,
                    "vector_mean_scale": vector,
                }
                row.update({name: statistics.median(values) for name, values in samples.items()})
                row["ratio"] = row["after"] / row["before"]
                row["samples"] = samples
                results.append(row)
print(
    json.dumps(
        {
            "cpu": min(cpus),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "extensions": {
                name: {
                    "path": engine.__file__,
                    "sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
                }
                for name, engine in engines.items()
            },
            "results": results,
        },
        indent=2,
    )
)
