"""Measure native survival-curve quantiles, excluding curve construction.

PYTHONPATH=python .venv/bin/python scripts/bench_surv_quantiles.py
Use --extension to compare a saved release library from the same Python ABI.
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
    parser.add_argument("--repeats", type=positive_int, default=21)
    parser.add_argument("--nprobs", type=positive_int, nargs="+", default=[3, 1001])
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must name a loadable extension library")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["survival._survival"] = module
        survival._survival = module
    from survival import _survival as core

    extension = Path(core.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    results = []
    for n in args.n:
        time_values = np.arange(1, n + 1, dtype=float)
        for plateau in (False, True):
            index = (time_values // 4) * 4 if plateau else time_values
            surv = 1 - index / (n + 1)
            fit = core.SurvfitKMResult.from_stacked(
                time_values,
                time_values[::-1],
                np.ones(n),
                surv,
                [n],
                lower=surv**1.3,
                upper=surv**0.7,
            )
            for nprobs in args.nprobs:
                probs = (
                    [0.25, 0.5, 0.75] if nprobs == 3 else np.linspace(0.01, 0.99, nprobs).tolist()
                )
                for conf_int in (False, True):
                    result = core.quantile_survfit(fit, probs, conf_int=conf_int)
                    samples = []
                    for _ in range(args.repeats):
                        start = time.perf_counter_ns()
                        result = core.quantile_survfit(fit, probs, conf_int=conf_int)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    results.append(
                        {
                            "n": n,
                            "nprobs": nprobs,
                            "plateau": plateau,
                            "conf_int": conf_int,
                            "median_ms": statistics.median(samples),
                            "samples_ms": samples,
                            "quantile": result.quantile,
                            "lower": result.lower,
                            "upper": result.upper,
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
