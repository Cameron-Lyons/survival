#!/usr/bin/env python3
"""Measure complete native G-rho calls; JSON retains every output for comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
from survival import _survival


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 100_000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 100:
        parser.error("repeats must be positive and sizes must be at least 100")
    core = _survival
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must be a loadable native library for this Python ABI")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    results = []
    for n in args.sizes:
        index = np.arange(n)
        status = (index % 7 != 0).astype(np.int32)
        group = (index % 3).astype(np.int32)
        for tied in [False, True]:
            times = ((index * 137) % (997 if tied else n) + 1).astype(float)
            for counting in [False, True]:
                start = times * ((index * 37 % 101) / 103.0) if counting else None
                for grouped in [False, True]:
                    strata = (index % 5).astype(np.int32) if grouped else None
                    for rho in [0.0, 1.0]:
                        call = partial(
                            core.survdiff,
                            times,
                            status,
                            group,
                            start=start,
                            strata=strata,
                            rho=rho,
                            timefix=False,
                        )

                        call()
                        samples = []
                        for _ in range(args.repeats):
                            started = time.perf_counter_ns()
                            result = call()
                            samples.append((time.perf_counter_ns() - started) / 1e6)
                        results.append(
                            {
                                "n": n,
                                "tied": tied,
                                "counting": counting,
                                "grouped": grouped,
                                "rho": rho,
                                "median_ms": statistics.median(samples),
                                "samples_ms": samples,
                                "result": {
                                    field: getattr(result, field)
                                    for field in (
                                        "n",
                                        "obs",
                                        "exp",
                                        "var",
                                        "chisq",
                                        "pvalue",
                                        "df",
                                        "strata",
                                        "group_codes",
                                    )
                                },
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
                "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
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
