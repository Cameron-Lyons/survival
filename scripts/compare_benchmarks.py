"""Compare the measured Divan tables from two successful benchmark runs.

Capture stdout without merging stderr: Divan's timer diagnostics are written
while the first timing row is being printed. Timing changes are advisory on
shared CI runners; missing or malformed measurements are errors.
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_HEADER = re.compile(
    r"^(.+?)\s+fastest\s+│\s+slowest\s+│\s+median\s+│\s+mean\s+│\s+samples\s+│\s+iters\s*$"
)
_NODE = re.compile(r"^((?:│  |   )*)(?:├─ |╰─ )(.*)$")
_DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+(ps|ns|µs|us|ms|s|m|h|d)")
_FIRST_DURATION = re.compile(r"^(.*?)\s+([0-9]+(?:\.[0-9]+)?\s+(?:ps|ns|µs|us|ms|s|m|h|d))\s*$")
_NANOSECONDS = {
    "ps": 0.001,
    "ns": 1.0,
    "µs": 1_000.0,
    "us": 1_000.0,
    "ms": 1e6,
    "s": 1e9,
    "m": 60e9,
    "h": 3600e9,
    "d": 86400e9,
}


@dataclass(frozen=True)
class Measurement:
    median_ns: float
    samples: int
    iterations: int


def _duration_ns(value: str) -> float:
    match = _DURATION.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid duration {value!r}")
    duration = float(match[1]) * _NANOSECONDS[match[2]]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be finite and positive: {value!r}")
    return duration


def parse_report(report: str) -> dict[str, Measurement]:
    """Read full benchmark identities, including suites, groups and arguments."""
    measurements: dict[str, Measurement] = {}
    parents: list[str] = []
    suites: dict[str, int] = {}

    for line_number, line in enumerate(report.splitlines(), 1):
        header = _HEADER.fullmatch(line)
        if header:
            suite = header[1].strip()
            if suite in suites:
                raise ValueError(f"line {line_number}: duplicate suite {suite!r}")
            suites[suite] = 0
            parents = [suite]
            continue
        if not line.strip():
            parents = []
            continue
        if not parents:
            # Cargo status messages, if redirected to stdout, are outside tables.
            continue

        node = _NODE.fullmatch(line)
        if node is None:
            # Divan may print throughput and allocation statistics underneath a
            # measured leaf. They have no branch marker and no sampling counts.
            # A damaged timing row must not be mistaken for an annotation.
            annotation = line.rsplit("│", 5)
            if (
                len(annotation) == 6
                and any(column.strip() for column in annotation[:4])
                and not any(column.strip() for column in annotation[4:])
            ):
                continue
            raise ValueError(f"line {line_number}: unexpected content in benchmark table")
        depth = len(node[1]) // 3 + 1
        if depth > len(parents):
            raise ValueError(f"line {line_number}: benchmark tree skips a parent")
        columns = node[2].rsplit("│", 5)
        if len(columns) != 6:
            raise ValueError(f"line {line_number}: missing timing columns")
        columns = [column.strip() for column in columns]
        first = _FIRST_DURATION.fullmatch(columns[0])
        if first is None:
            name = columns[0]
            if name.endswith("(ignored)") and not any(columns[1:]):
                continue
            if not name or any(columns[1:]):
                raise ValueError(f"line {line_number}: malformed timing row")
            parents = [*parents[:depth], name]
            continue

        name, fastest = first.groups()
        if not name.strip():
            raise ValueError(f"line {line_number}: benchmark name is empty")
        identity = "::".join([*parents[:depth], name.strip()])
        try:
            fastest_ns, slowest_ns, median_ns, mean_ns = (
                _duration_ns(value) for value in [fastest, *columns[1:4]]
            )
            samples, iterations = (int(value) for value in columns[4:])
            if samples <= 0 or iterations < samples:
                raise ValueError("sample count must be positive and iterations >= samples")
            if not fastest_ns <= median_ns <= slowest_ns:
                raise ValueError("median must be between fastest and slowest")
            if not fastest_ns <= mean_ns <= slowest_ns:
                raise ValueError("mean must be between fastest and slowest")
        except ValueError as error:
            raise ValueError(f"line {line_number} ({identity}): {error}") from error
        if identity in measurements:
            raise ValueError(f"line {line_number}: duplicate benchmark {identity!r}")
        measurements[identity] = Measurement(median_ns, samples, iterations)
        suites[parents[0]] += 1
        # Measured leaves can have later siblings; they never become parents.
        parents = parents[:depth]

    if not measurements:
        raise ValueError("report contains no measured benchmarks (smoke output is insufficient)")
    empty_suites = [suite for suite, count in suites.items() if not count]
    if empty_suites:
        raise ValueError(f"suites have no measured benchmarks: {', '.join(empty_suites)}")
    return measurements


def _code(value: str) -> str:
    return f"<code>{html.escape(value).replace('|', '&#124;')}</code>"


def _format_duration(nanoseconds: float) -> str:
    for unit, scale in [("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1.0)]:
        if nanoseconds >= scale:
            return f"{nanoseconds / scale:.4g} {unit}"
    return f"{nanoseconds * 1e3:.4g} ps"


def compare_reports(base: dict[str, Measurement], current: dict[str, Measurement]) -> str:
    common = base.keys() & current.keys()
    if not common:
        raise ValueError("base and current reports have no benchmark cases in common")
    added = sorted(current.keys() - base.keys())
    removed = sorted(base.keys() - current.keys())
    changes = {
        identity: 100.0 * (current[identity].median_ns / base[identity].median_ns - 1.0)
        for identity in common
    }
    lines = [
        "## Benchmark Results",
        "",
        f"Compared {len(common)} cases; {len(added)} added and {len(removed)} removed.",
        "",
        "Median time changes are advisory on shared CI runners. Positive changes mean slower; "
        "the benchmark smoke tests, execution, and report validation must all pass.",
        "",
        "| Benchmark | Base median | PR median | Change | Samples (base / PR) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for identity in sorted(common, key=lambda name: (-changes[name], name)):
        before, after = base[identity], current[identity]
        lines.append(
            f"| {_code(identity)} | {_format_duration(before.median_ns)} | "
            f"{_format_duration(after.median_ns)} | {changes[identity]:+.1f}% | "
            f"{before.samples} / {after.samples} |"
        )
    for title, identities in [("Added cases", added), ("Removed cases", removed)]:
        if identities:
            lines.extend(["", f"### {title}", ""])
            lines.extend(f"- {_code(identity)}" for identity in identities)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Base revision's Divan stdout")
    parser.add_argument("current", type=Path, help="PR revision's Divan stdout")
    parser.add_argument("--output", type=Path, required=True, help="Markdown comparison report")
    args = parser.parse_args(argv)
    try:
        base = parse_report(args.base.read_text(encoding="utf-8"))
        current = parse_report(args.current.read_text(encoding="utf-8"))
        args.output.write_text(compare_reports(base, current), encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"benchmark comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Compared benchmark reports: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
