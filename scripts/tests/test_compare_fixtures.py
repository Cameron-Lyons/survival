"""Keep logical fixture values distinct from numbers in the stock-R gate."""

# The CI helper suite runs with the standard library alone.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "test/r/compare_fixtures.py"
SPEC = importlib.util.spec_from_file_location("compare_fixtures", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
compare_fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_fixtures)


class FixtureComparisonTests(unittest.TestCase):
    def test_rejects_logical_numeric_type_changes_in_either_direction(self):
        pairs = [(True, 1), (True, 1.0), (False, 0), (False, 0.0)]
        for old, new in [*pairs, *((new, old) for old, new in pairs)]:
            with self.subTest(old=old, new=new):
                differences = []
                compare_fixtures._walk(old, new, ".logical", 1.0, 1.0, differences)
                self.assertEqual(differences, [(".logical", old, new)])

    def test_preserves_exact_boolean_values_and_numeric_tolerances(self):
        for old, new in [(True, True), (False, False), (1, 1.0), (2.0, 2.000000001)]:
            with self.subTest(old=old, new=new):
                differences = []
                compare_fixtures._walk(old, new, ".value", 1e-9, 1e-12, differences)
                self.assertEqual(differences, [])
        differences = []
        compare_fixtures._walk(True, False, ".logical", 1.0, 1.0, differences)
        self.assertEqual(differences, [(".logical", True, False)])

    def test_directory_gate_reports_nested_logical_regressions_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "committed", root / "regenerated"
            old.mkdir()
            new.mkdir()
            reference = {
                "cases": [{"name": "logical_expression", "expected": {"values": [True, False]}}]
            }
            (old / "logical.json").write_text(json.dumps(reference), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                changed = {
                    "cases": [{"name": "logical_expression", "expected": {"values": [1, 0]}}]
                }
                (new / "logical.json").write_text(json.dumps(changed), encoding="utf-8")
                status = compare_fixtures.main(
                    [str(old), str(new), "--rtol", "1e-9", "--atol", "1e-12"]
                )
            self.assertEqual(status, 1)
            self.assertIn("2 differing values", output.getvalue())
            self.assertIn(
                "case 'logical_expression'.expected.values[0]: True -> 1", output.getvalue()
            )
            self.assertIn(
                "case 'logical_expression'.expected.values[1]: False -> 0", output.getvalue()
            )

            (new / "logical.json").write_text(json.dumps(reference), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = compare_fixtures.main([str(old), str(new)])
            self.assertEqual(status, 0)
            self.assertIn("1 fixture files match", output.getvalue())


if __name__ == "__main__":
    unittest.main()
