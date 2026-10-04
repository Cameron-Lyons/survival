from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def coverage_threshold(value: str) -> float:
    try:
        percent = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("minimum coverage must be a number") from error
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise argparse.ArgumentTypeError("minimum coverage must be finite and between 0 and 100")
    return percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a Cobertura-style coverage report is below a minimum line coverage percentage."
        )
    )
    parser.add_argument("report", type=Path, help="Path to cobertura.xml")
    parser.add_argument(
        "--min-percent",
        type=coverage_threshold,
        required=True,
        help="Minimum acceptable line coverage percentage, for example 30",
    )
    return parser.parse_args()


def extract_line_rate(root: ET.Element) -> float:
    if root.tag.rsplit("}", 1)[-1] != "coverage":
        raise ValueError("expected a Cobertura coverage element")

    raw_line_rate = root.attrib.get("line-rate")
    line_rate = None
    if raw_line_rate is not None:
        line_rate = float(raw_line_rate)
        if not math.isfinite(line_rate) or not 0.0 <= line_rate <= 1.0:
            raise ValueError("line-rate must be finite and between 0 and 1")

    raw_lines_covered = root.attrib.get("lines-covered")
    raw_lines_valid = root.attrib.get("lines-valid")
    if raw_lines_covered is not None or raw_lines_valid is not None:
        if raw_lines_covered is None or raw_lines_valid is None:
            raise ValueError("lines-covered and lines-valid must be supplied together")
        covered = int(raw_lines_covered)
        valid = int(raw_lines_valid)
        if valid <= 0:
            raise ValueError("coverage report has no valid lines")
        if not 0 <= covered <= valid:
            raise ValueError("lines-covered must be between 0 and lines-valid")
        # Counts retain their precision when the producer rounds line-rate.
        return covered / valid

    if line_rate is not None:
        return line_rate

    raise ValueError("could not determine line coverage from report")


def main() -> int:
    args = parse_args()
    if not args.report.is_file():
        print(f"coverage report not found: {args.report}", file=sys.stderr)
        return 1

    try:
        root = ET.parse(args.report).getroot()
        line_rate = extract_line_rate(root)
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"invalid coverage report: {error}", file=sys.stderr)
        return 1
    percent = line_rate * 100.0

    print(f"Line coverage: {percent:.2f}%")
    print(f"Minimum required: {args.min_percent:.2f}%")

    if percent + 1e-9 < args.min_percent:
        print(
            f"coverage check failed: {percent:.2f}% is below {args.min_percent:.2f}%",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
