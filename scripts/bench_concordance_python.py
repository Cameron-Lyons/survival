"""Time Python concordance with a preconstructed Surv response.

Run against the installed package:
    .venv/bin/python scripts/bench_concordance_python.py --n 10000 100000 --repeats 7

Select a baseline Python package and its extension together with PYTHONPATH:
    PYTHONPATH=/path/to/baseline/python .venv/bin/python scripts/bench_concordance_python.py

Input construction, Surv construction, and output disposal are outside each timed
call. Facade validation, native conversion, and result construction are included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import survival
from bench_concordance import generate_inputs, measure, positive_int
from survival import _survival, r_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--timewt", choices=["n", "S", "I"], default="n")
    parser.add_argument(
        "--responses",
        "--response",
        choices=["right", "counting"],
        nargs="+",
        default=["right", "counting"],
    )
    parser.add_argument(
        "--timefix",
        choices=["default", "true", "false"],
        default="default",
        help="Use the package default or explicitly control near-tie adjudication",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def result_number(value: float | None) -> float | str | None:
    # Tiny inputs can have no comparable pairs. Preserve undefined results in
    # valid JSON rather than failing after the timing samples have completed.
    return value if value is None or math.isfinite(value) else str(value)


def main() -> None:
    args = parse_args()
    extension_path = Path(_survival.__file__).resolve()
    facade_path = Path(r_api.__file__).resolve()
    report = {
        "label": args.label,
        "started_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "survival": survival.__version__,
        "package_path": str(Path(survival.__file__).resolve()),
        "facade_path": str(facade_path),
        "facade_sha256": file_sha256(facade_path),
        "extension_path": str(extension_path),
        "extension_sha256": file_sha256(extension_path),
        "repeats": args.repeats,
        "warmups": 1,
        "timewt": args.timewt,
        "timefix": args.timefix,
        "timing": (
            "Python concordance with preconstructed Surv; includes facade validation, "
            "native conversion, and result construction; excludes input construction, "
            "Surv construction, and output disposal"
        ),
        "results": [],
    }
    for n in args.n:
        start, stop, status, risk, weights = generate_inputs(n)
        for response_type in args.responses:
            response = (
                survival.Surv(start, stop, status)
                if response_type == "counting"
                else survival.Surv(stop, status)
            )
            for diagnostics in [False, True]:
                options = {"ranks": True, "influence": 3} if diagnostics else {}
                if args.timefix != "default":
                    options["timefix"] = args.timefix == "true"
                call = partial(
                    survival.concordance,
                    response,
                    risk_scores=risk,
                    weights=weights,
                    timewt=args.timewt,
                    **options,
                )
                samples, result = measure(call, args.repeats)
                report["results"].append(
                    {
                        "response": response_type,
                        "operation": "ranks_influence" if diagnostics else "default",
                        "n": n,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "samples_ms": samples,
                        "result": {
                            "concordance": result_number(result.concordance),
                            "variance": result_number(result.variance),
                        },
                    }
                )
                del result
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
