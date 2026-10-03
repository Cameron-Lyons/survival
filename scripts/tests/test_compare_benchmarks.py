# The CI helper suite deliberately uses unittest so it needs no built extension
# or third-party test dependencies; subprocess commands invoke this repository's
# helper with the same Python interpreter.
# ruff: noqa: PT009, PT027, S603
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "compare_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("compare_benchmarks", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
compare_benchmarks = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare_benchmarks
SPEC.loader.exec_module(compare_benchmarks)

# Format verified against the repository's Divan 0.1.21, including nested
# argument leaves and several suites in the stdout of one `cargo bench` run.
REPORT = """survival_benchmarks  fastest │ slowest │ median │ mean │ samples │ iters
├─ single_curve      4 ns │ 6 ns │ 5 ns │ 5 ns │ 100 │ 51200
╰─ curves                │      │      │      │     │
   ├─ fitted_curve       │      │      │      │     │
   │  ├─ 1          1 µs │ 3 µs │ 2 µs │ 2 µs │ 100 │ 12800
   │  ╰─ 10         2 ms │ 4 ms │ 3 ms │ 3 ms │ 100 │ 3200
   ╰─ weighted_curve 1 s │ 3 s  │ 2 s  │ 2 s  │ 100 │ 100

concordance_benchmarks fastest │ slowest │ median │ mean │ samples │ iters
╰─ counts            1 ns │ 3 ns │ 2 ns │ 2 ns │ 100 │ 100
"""


class ParseReportTests(unittest.TestCase):
    def test_preserves_suite_group_and_argument_identity(self):
        result = compare_benchmarks.parse_report(REPORT)
        self.assertEqual(len(result), 5)
        self.assertEqual(result["survival_benchmarks::single_curve"].median_ns, 5)
        self.assertEqual(result["survival_benchmarks::curves::fitted_curve::1"].median_ns, 2_000)
        self.assertEqual(result["survival_benchmarks::curves::fitted_curve::10"].median_ns, 3e6)
        self.assertEqual(result["survival_benchmarks::curves::weighted_curve"].median_ns, 2e9)
        self.assertEqual(result["concordance_benchmarks::counts"].iterations, 100)

    def test_supported_duration_units(self):
        self.assertEqual(compare_benchmarks._duration_ns("250 ps"), 0.25)
        self.assertEqual(compare_benchmarks._duration_ns("2.5 us"), 2_500)
        self.assertEqual(compare_benchmarks._duration_ns("1 m"), 60e9)
        self.assertEqual(compare_benchmarks._duration_ns("1 h"), 3600e9)
        self.assertEqual(compare_benchmarks._duration_ns("1 d"), 86400e9)

    def test_ignores_cargo_messages_outside_tables(self):
        result = compare_benchmarks.parse_report("Finished bench profile\n" + REPORT)
        self.assertEqual(len(result), 5)

    def test_ignores_extra_throughput_and_allocation_rows(self):
        annotated = REPORT.replace(
            "╰─ curves",
            "│                  100 MB/s │ 100 MB/s │ 100 MB/s │ 100 MB/s │ │\n"
            "│                  max alloc: │ │ │ │ │\n"
            "│                  16 B │ 16 B │ 16 B │ 16 B │ │\n╰─ curves",
        )
        self.assertEqual(
            compare_benchmarks.parse_report(annotated), compare_benchmarks.parse_report(REPORT)
        )

    def test_rejects_empty_smoke_and_unmeasured_suite(self):
        for report in ["", "survival_benchmarks\n╰─ single_curve\n", REPORT.split("├─")[0]]:
            with self.subTest(report=report), self.assertRaises(ValueError):
                compare_benchmarks.parse_report(report)

    def test_rejects_missing_columns_and_corrupt_statistics(self):
        corruptions = [
            REPORT.replace("4 ns │ 6 ns │ 5 ns │ 5 ns │ 100 │ 51200", "4 ns │ 6 ns"),
            REPORT.replace("4 ns", "NaN ns"),
            REPORT.replace("4 ns", "0 ns"),
            REPORT.replace("6 ns", "1 ns"),
            REPORT.replace("100 │ 51200", "0 │ 51200"),
            REPORT.replace("100 │ 51200", "100 │ 1"),
            REPORT.replace("100 │ 51200", "unknown │ 51200"),
        ]
        for report in corruptions:
            with self.subTest(report=report), self.assertRaises(ValueError):
                compare_benchmarks.parse_report(report)

    def test_rejects_duplicate_cases_suites_and_missing_parents(self):
        corruptions = [
            REPORT.replace("   │  ╰─ 10", "   │  ╰─ 1"),
            REPORT + REPORT,
            REPORT.replace("├─ single_curve", "   ├─ single_curve"),
        ]
        for report in corruptions:
            with self.subTest(report=report), self.assertRaises(ValueError):
                compare_benchmarks.parse_report(report)

    def test_rejects_stderr_diagnostics_interleaved_with_timing_rows(self):
        merged = REPORT.replace("4 ns", "Timer precision: 21 ns\n4 ns")
        with self.assertRaises(ValueError):
            compare_benchmarks.parse_report(merged)

    def test_rejects_timing_row_with_missing_branch_marker(self):
        with self.assertRaises(ValueError):
            compare_benchmarks.parse_report(REPORT.replace("├─ single_curve", "   single_curve"))


class ComparisonTests(unittest.TestCase):
    def test_reports_actual_percentage_changes_and_case_additions_removals(self):
        base = compare_benchmarks.parse_report(REPORT)
        current = dict(base)
        current["survival_benchmarks::single_curve"] = compare_benchmarks.Measurement(10, 80, 800)
        current["new_case"] = current.pop("concordance_benchmarks::counts")
        result = compare_benchmarks.compare_reports(base, current)
        self.assertIn("Compared 4 cases; 1 added and 1 removed.", result)
        self.assertIn("5 ns | 10 ns | +100.0% | 100 / 80", result)
        self.assertIn("### Added cases\n\n- <code>new_case</code>", result)
        self.assertIn("### Removed cases\n\n- <code>concordance_benchmarks::counts</code>", result)

    def test_rejects_unrelated_reports(self):
        measured = compare_benchmarks.Measurement(1, 100, 100)
        with self.assertRaisesRegex(ValueError, "no benchmark cases in common"):
            compare_benchmarks.compare_reports({"base": measured}, {"current": measured})

    def test_escapes_benchmark_names_for_markdown_tables(self):
        self.assertEqual(compare_benchmarks._code("<T>|`x`"), "<code>&lt;T&gt;&#124;`x`</code>")

    def test_cli_writes_report_and_fails_on_missing_or_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, current, output = (root / name for name in ["base.txt", "pr.txt", "report.md"])
            base.write_text(REPORT)
            current.write_text(REPORT)
            command = [
                sys.executable,
                str(SCRIPT),
                str(base),
                str(current),
                "--output",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Compared 5 cases", output.read_text())
            for malformed in [True, False]:
                if malformed:
                    current.write_text("no measured cases")
                else:
                    current.unlink()
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn("benchmark comparison failed:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
