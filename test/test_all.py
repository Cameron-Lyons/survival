#!/usr/bin/env python3

import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "target", "wheels"))


def run_test_file(test_file: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Running {test_file}")
    print(f"{'=' * 60}")

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True, timeout=60
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"{test_file} timed out")
        return False
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main() -> int:
    test_dir: str = os.path.dirname(__file__)
    this_file: str = os.path.abspath(__file__)
    all_test_files: list[str] = sorted(glob.glob(os.path.join(test_dir, "test_*.py")))

    test_files: list[str] = [f for f in all_test_files if os.path.abspath(f) != this_file]

    if not test_files:
        print("No test files found!")
        return 1

    print(f"Found {len(test_files)} test file(s)")

    passed: int = 0
    failed: int = 0

    for test_file in test_files:
        if run_test_file(test_file):
            passed += 1
            print(f"{os.path.basename(test_file)} passed")
        else:
            failed += 1
            print(f"{os.path.basename(test_file)} failed")

    print(f"\n{'=' * 60}")
    print(f"Summary: {passed} passed, {failed} failed out of {len(test_files)} tests")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
