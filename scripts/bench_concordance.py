"""Time native concordance summaries and influences with mixed event/censor ties.

Run against the installed extension, for example:
    .venv/bin/python scripts/bench_concordance.py --n 1000 10000 100000 --repeats 5

Input construction and output disposal are outside each timed call. Python to
Rust input conversion and native result construction are included.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--timewt", choices=["n", "S", "I"], default="n")
    parser.add_argument(
        "--operations",
        "--operation",
        choices=["summary", "influence"],
        nargs="+",
        default=["summary", "influence"],
    )
    parser.add_argument(
        "--responses",
        "--response",
        choices=["right", "counting"],
        nargs="+",
        default=["right", "counting"],
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--extension", type=Path, help="Load a previously built native extension")
    return parser.parse_args()


def load_extension(path: Path | None):
    if path is None:
        from survival import _survival

        return _survival
    spec = importlib.util.spec_from_file_location("_survival", path.resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_inputs(n: int) -> tuple[list[float], list[float], list[int], list[float], list[float]]:
    # For the standard sizes, multiplication by 37 permutes row indices. Eight
    # rows share each stop time, with censoring independent of the stop order.
    stop = [float(1 + ((row * 37) % n) // 8) for row in range(n)]
    start = [float((row * 17 + 3) % int(end)) for row, end in enumerate(stop)]
    status = [int(row % 3 != 0) for row in range(n)]
    risk = [((row * 53 + row // 7) % 257) / 32.0 for row in range(n)]
    weights = [0.0 if row % 29 == 0 else 0.5 + (row * 7 % 13) / 4.0 for row in range(n)]
    return start, stop, status, risk, weights


def measure(call: Callable[[], object], repeats: int) -> tuple[list[float], object]:
    # Warm up once before collecting samples, including native code paths and
    # allocator reuse. Keep the last result for an untimed result summary.
    result = call()
    samples = []
    for _ in range(repeats):
        del result
        before = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - before) / 1_000_000.0)
    return samples, result


def main() -> None:
    args = parse_args()
    native = load_extension(args.extension)
    extension_path = Path(native.__file__).resolve()
    with extension_path.open("rb") as extension_file:
        extension_sha256 = hashlib.file_digest(extension_file, "sha256").hexdigest()
    report = {
        "label": args.label,
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "extension_path": str(extension_path),
        "extension_sha256": extension_sha256,
        "repeats": args.repeats,
        "warmups": 1,
        "timewt": args.timewt,
        "timing": "native wrapper call, including input conversion and result construction",
        "results": [],
    }
    for n in args.n:
        start, stop, status, risk, weights = generate_inputs(n)
        for response in args.responses:
            for operation in args.operations:
                name = f"{'counting_' if response == 'counting' else ''}concordance_{operation}"
                if operation == "influence":
                    name += "_rows"
                function = getattr(native, name)
                call_args = (stop, status, risk, weights, args.timewt)
                if response == "counting":
                    call_args = (start, *call_args, False)
                samples, result = measure(partial(function, *call_args), args.repeats)
                if operation == "summary":
                    result_summary = {
                        key: result[key] for key in ["concordance", "concordant", "comparable"]
                    }
                else:
                    result_summary = {"n_rows": len(result[0]), "variance": result[2]}
                report["results"].append(
                    {
                        "response": response,
                        "operation": operation,
                        "n": n,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "samples_ms": samples,
                        "result": result_summary,
                    }
                )
                del result
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
