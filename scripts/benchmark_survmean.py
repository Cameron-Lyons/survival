"""Measure survival-summary tables without fitting or event-row extraction.

Run before and after a release build. --extension loads a saved native library
from the same Python ABI for an independent predecessor measurement.
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
from survival import _survival


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=int, default=[10000, 100000, 300000])
    parser.add_argument("--curves", nargs="+", type=int, default=[1, 100])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    if min([*args.n, *args.curves, args.repeats]) < 1:
        parser.error("sizes and repeats must be positive")
    core = _survival
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must be a loadable extension library")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    fields = (
        "records",
        "n_max",
        "n_start",
        "events",
        "rmean",
        "se_rmean",
        "median",
        "lower",
        "upper",
        "end_time",
    )
    results = []
    for n in args.n:
        for curves in args.curves:
            if n < curves:
                continue
            lengths = [n // curves + (i < n % curves) for i in range(curves)]
            times = np.concatenate([np.arange(1, size + 1, dtype=float) for size in lengths])
            risks = np.concatenate([np.arange(size, 0, -1, dtype=float) for size in lengths])
            surv = np.concatenate([1 - np.arange(1, size + 1) / (size + 1) for size in lengths])
            fit = core.SurvfitKMResult.from_stacked(
                times,
                risks,
                np.ones(n),
                surv,
                lengths,
                strata=lengths if curves > 1 else None,
                lower=surv**1.3,
                upper=surv**0.7,
            )
            for scale in (1.0, 2.5):
                for rmean in ("none", "common", str(max(1.0, max(lengths) / 10))):
                    core.survmean(fit, scale=scale, rmean=rmean)
                    samples = []
                    for _ in range(args.repeats):
                        start = time.perf_counter_ns()
                        result = core.survmean(fit, scale=scale, rmean=rmean)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    results.append(
                        {
                            "n": n,
                            "curves": curves,
                            "scale": scale,
                            "rmean": rmean,
                            "median_ms": statistics.median(samples),
                            "samples_ms": samples,
                            "result": {name: getattr(result, name) for name in fields},
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
