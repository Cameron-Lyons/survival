"""Measure RMST comparisons over many interleaved groups before and after a build.

Use --extension with a saved native library built for the same Python ABI to
compare a predecessor. JSON includes every returned value to check equality.
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
    parser.add_argument("--n", type=int, default=600_000)
    parser.add_argument("--groups", nargs="+", type=int, default=[2, 32, 128])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    if min(args.n, args.repeats) < 1 or min(args.groups) < 2 or max(args.groups) > args.n:
        parser.error(
            "sizes and repeats must be positive, with at least two groups and one row per group"
        )
    core = _survival
    if args.extension is not None:
        spec = importlib.util.spec_from_file_location("survival._survival", args.extension)
        if spec is None or spec.loader is None:
            parser.error("--extension must be a loadable extension library")
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    index = np.arange(args.n)
    times = ((index * 137) % 997 + 1).astype(float)
    status = ((index % 7) != 0).astype(np.int32)
    group_fields = ("group", "n", "events", "rmean", "se_rmean", "lower", "upper")
    comparison_fields = (
        "tau",
        "difference",
        "difference_se",
        "difference_lower",
        "difference_upper",
        "difference_p_value",
        "chisq",
        "df",
        "p_value",
    )
    results = []
    for n_groups in args.groups:
        groups = (index % n_groups).astype(np.int32)
        core.rmst_comparison(times, status, groups, 900.0)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            result = core.rmst_comparison(times, status, groups, 900.0)
            samples.append((time.perf_counter_ns() - started) / 1e6)
        results.append(
            {
                "n": args.n,
                "groups": n_groups,
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "result": {
                    **{field: getattr(result, field) for field in comparison_fields},
                    "groups": [
                        {field: getattr(group, field) for field in group_fields}
                        for group in result.groups
                    ],
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
