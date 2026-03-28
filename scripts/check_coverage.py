#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if a Cobertura-style coverage report is below a minimum line coverage percentage."
    )
    parser.add_argument("report", type=Path, help="Path to cobertura.xml")
    parser.add_argument(
        "--min-percent",
        type=float,
        required=True,
        help="Minimum acceptable line coverage percentage, for example 30",
    )
    return parser.parse_args()


def extract_line_rate(root: ET.Element) -> float:
    raw_line_rate = root.attrib.get("line-rate")
    if raw_line_rate is not None:
        return float(raw_line_rate)

    raw_lines_covered = root.attrib.get("lines-covered")
    raw_lines_valid = root.attrib.get("lines-valid")
    if raw_lines_covered is not None and raw_lines_valid is not None:
        covered = float(raw_lines_covered)
        valid = float(raw_lines_valid)
        if valid <= 0.0:
            raise ValueError("coverage report has no valid lines")
        return covered / valid

    raise ValueError("could not determine line coverage from report")


def main() -> int:
    args = parse_args()
    if not args.report.is_file():
        print(f"coverage report not found: {args.report}", file=sys.stderr)
        return 1

    root = ET.parse(args.report).getroot()
    line_rate = extract_line_rate(root)
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
