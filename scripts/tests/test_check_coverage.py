"""Exercise the coverage gate with valid and malformed producer output."""

# This suite runs with the standard library alone, before building the extension.
# ruff: noqa: PT009, PT027

from __future__ import annotations

import argparse
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import check_coverage


class CoverageReportTests(unittest.TestCase):
    def test_accepts_rate_only_reports(self) -> None:
        for value in ("0", "0.6", "1"):
            with self.subTest(value=value):
                root = ET.Element("coverage", {"line-rate": value})
                self.assertEqual(check_coverage.extract_line_rate(root), float(value))

    def test_accepts_counts_and_preserves_their_precision(self) -> None:
        for extra in ({}, {"line-rate": "0.67"}):
            with self.subTest(extra=extra):
                root = ET.Element("coverage", {"lines-covered": "2", "lines-valid": "3", **extra})
                self.assertEqual(check_coverage.extract_line_rate(root), 2 / 3)

    def test_rejects_nonfinite_and_out_of_range_rates(self) -> None:
        for value in ("NaN", "inf", "-inf", "-0.01", "1.01", "bad", ""):
            with self.subTest(value=value):
                root = ET.Element("coverage", {"line-rate": value})
                with self.assertRaises(ValueError):
                    check_coverage.extract_line_rate(root)

    def test_rejects_empty_invalid_or_incomplete_counts_even_with_a_rate(self) -> None:
        invalid = [
            {"lines-covered": "0", "lines-valid": "0"},
            {"lines-covered": "-1", "lines-valid": "10"},
            {"lines-covered": "11", "lines-valid": "10"},
            {"lines-covered": "1", "lines-valid": "-10"},
            {"lines-covered": "NaN", "lines-valid": "10"},
            {"lines-covered": "1", "lines-valid": "inf"},
            {"lines-covered": "1.5", "lines-valid": "10"},
            {"lines-covered": "1"},
            {"lines-valid": "10"},
        ]
        for counts in invalid:
            with self.subTest(counts=counts):
                root = ET.Element("coverage", {"line-rate": "1", **counts})
                with self.assertRaises(ValueError):
                    check_coverage.extract_line_rate(root)

    def test_rejects_reports_without_coverage(self) -> None:
        for root in (ET.Element("coverage"), ET.Element("other", {"line-rate": "1"})):
            with self.subTest(tag=root.tag), self.assertRaises(ValueError):
                check_coverage.extract_line_rate(root)

    def test_threshold_is_a_finite_percentage(self) -> None:
        for value in ("0", "60", "100"):
            with self.subTest(value=value):
                self.assertEqual(check_coverage.coverage_threshold(value), float(value))
        for value in ("NaN", "inf", "-inf", "-1", "101", "bad"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                check_coverage.coverage_threshold(value)

    def test_cli_fails_closed_and_reports_errors(self) -> None:
        cases = [
            ('<coverage line-rate="0.6"/>', 0, "Line coverage: 60.00%"),
            ('<coverage line-rate="0.599"/>', 1, "coverage check failed"),
            ('<coverage line-rate="NaN"/>', 1, "invalid coverage report"),
            ('<coverage line-rate="inf"/>', 1, "invalid coverage report"),
            ('<coverage line-rate="1.2"/>', 1, "invalid coverage report"),
            ('<coverage line-rate="1" lines-valid="0" lines-covered="0"/>', 1, "no valid lines"),
            ('<coverage line-rate="1">', 1, "invalid coverage report"),
            (None, 1, "coverage report not found"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "coverage.xml"
            for xml, expected_status, message in cases:
                with self.subTest(xml=xml):
                    if xml is None:
                        report.unlink(missing_ok=True)
                    else:
                        report.write_text(xml, encoding="utf-8")
                    stdout, stderr = io.StringIO(), io.StringIO()
                    args = ["check_coverage", str(report), "--min-percent", "60"]
                    with (
                        patch("sys.argv", args),
                        redirect_stdout(stdout),
                        redirect_stderr(stderr),
                    ):
                        status = check_coverage.main()
                    self.assertEqual(status, expected_status)
                    self.assertIn(message, stdout.getvalue() + stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
