"""Measure complete confidence-band calls, including input and output conversion.

Run before and after a release build. --extension compares the native boundary
with a saved library of the same Python ABI; facade calls use the installed code.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from survival import _survival, r_api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300_000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    if min(args.n, args.repeats) < 1:
        parser.error("size and repeats must be positive")
    core = _survival
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must be a loadable extension library")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    p = np.linspace(0.99, 0.01, args.n)
    se = np.full(args.n, 0.1)
    selow = np.full(args.n, 0.15)
    interfaces = [("native", core.survfit_confint)]
    if args.extension is None:
        interfaces.append(("r_facade", r_api.survfit_confint))
    results = []
    for name, function in interfaces:
        for conf_type in ("plain", "log", "log-log"):

            def call(function=function, conf_type=conf_type):
                bands = function(p, se, conf_type=conf_type, selow=selow)
                return bands.lower, bands.upper

            call()
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                lower, upper = call()
                samples.append((time.perf_counter_ns() - started) / 1e6)
            digest = hashlib.sha256()
            for values in (lower, upper):
                digest.update(np.asarray(values, dtype="<f8").tobytes())
            results.append(
                {
                    "interface": name,
                    "conf_type": conf_type,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "result_sha256": digest.hexdigest(),
                }
            )
    extension = Path(core.__file__).resolve()
    with extension.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "n": args.n,
                "repeats": args.repeats,
                "extension": str(extension),
                "extension_sha256": digest,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
