#!/usr/bin/env python3
"""Compare two directories of R fixtures value by value.

The fixture files are single-line JSON, so ``git diff`` on a regenerated copy
prints one unreadable 1 MB line per file.  This walks both trees instead and
reports every leaf that differs, with the fixture case and field it belongs
to, so a regeneration on another machine can be diagnosed from a CI log:

    python test/r/compare_fixtures.py committed/ test/r/fixtures

Numbers compare within ``--rtol``/``--atol`` (both 0 by default, i.e. exact);
everything else must match exactly.  The exit status is 1 when anything
differs, or when a file is present on one side only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

NON_FINITE = {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}

# Values R itself does not reproduce across machines (or across GitHub's runner
# pool): differences under these ``(file, case, field prefix)`` keys are reported
# but do not fail the comparison.  Each carries the reason.
UNSTABLE: dict[tuple[str, str, str], str] = {
    ("aareg.json", "veteran_karno_celltype", "expected.coefficient"): (
        "late-time increments sit on a rank decision of the least-squares step"
    ),
    ("coxph.json", "lung_age_sex_init_iter0", "expected.concordance"): (
        "init makes age 71/sex 2 and age 46/sex 1 mathematically tied; whether the "
        "linear predictors are bit-equal depends on the platform's rounding"
    ),
    ("coxph.json", "lung_age_sex_init_iter0", "expected.summary.concordance"): (
        "as expected.concordance"
    ),
    ("coxph.json", "lung_age_sex_cluster_inst", "expected.concordance.cvar"): (
        "clustered concordance variance moves with last-ulp ties in the linear predictor"
    ),
    ("coxph.json", "lung_factor_ph_ecog", "expected.concordance.cvar"): "as above",
    ("coxph.json", "lung_interaction", "expected.concordance.cvar"): "as above",
    ("survreg.json", "lung_weibull_factor_ph_ecog", "expected.concordance.cvar"): "as above",
    ("coxph_penalized.json", "lung_pspline_karno_df3_nterm6", "expected.concordance"): (
        "tied linear predictors decided by floating-point noise"
    ),
    ("coxph_penalized.json", "lung_pspline_age_df0_aic", "expected.penalty"): (
        "the AIC-optimal penalty stops on an iteration count that rounding can shift"
    ),
}


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return NON_FINITE.get(value)
    return None


def _same_number(old: float, new: float, rtol: float, atol: float) -> bool:
    if math.isnan(old) or math.isnan(new):
        return math.isnan(old) and math.isnan(new)
    if math.isinf(old) or math.isinf(new):
        return old == new
    return abs(old - new) <= atol + rtol * abs(old)


def _walk(old: object, new: object, path: str, rtol: float, atol: float, out: list[tuple]):
    old_num, new_num = _as_number(old), _as_number(new)
    if old_num is not None and new_num is not None:
        if not _same_number(old_num, new_num, rtol, atol):
            out.append((path, old, new))
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                out.append((f"{path}.{key}", old.get(key, "<missing>"), new.get(key, "<missing>")))
            else:
                _walk(old[key], new[key], f"{path}.{key}", rtol, atol, out)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append((f"{path}#len", len(old), len(new)))
        for index, (o, n) in enumerate(zip(old, new, strict=False)):
            _walk(o, n, f"{path}[{index}]", rtol, atol, out)
        return
    if old != new:
        out.append((path, old, new))


def _case_paths(doc: object) -> dict[str, str]:
    """Map ``cases[i]`` to the case name so reports say which case changed."""
    if not isinstance(doc, dict) or not isinstance(doc.get("cases"), list):
        return {}
    return {
        f".cases[{index}]": str(case.get("name", index))
        for index, case in enumerate(doc["cases"])
        if isinstance(case, dict)
    }


def _describe(path: str, names: dict[str, str]) -> str:
    for prefix, name in names.items():
        if path.startswith(prefix):
            return f"case {name!r}{path[len(prefix) :]}"
    return path.lstrip(".")


def _relative(old: object, new: object) -> str:
    old_num, new_num = _as_number(old), _as_number(new)
    if old_num is None or new_num is None:
        return ""
    if not math.isfinite(old_num) or not math.isfinite(new_num):
        return ""
    scale = max(abs(old_num), abs(new_num))
    if scale == 0:
        return ""
    return f" (rel {abs(old_num - new_num) / scale:.3g})"


def _unstable(file_name: str, path: str, names: dict[str, str]) -> bool:
    for prefix, case in names.items():
        if path.startswith(prefix):
            field = path[len(prefix) :].lstrip(".")
            return any(
                key[0] == file_name and key[1] == case and field.startswith(key[2])
                for key in UNSTABLE
            )
    return False


def compare_file(old_path: Path, new_path: Path, rtol: float, atol: float, limit: int) -> int:
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    diffs: list[tuple] = []
    _walk(old, new, "", rtol, atol, diffs)
    if not diffs:
        return 0
    names = _case_paths(old)
    unstable = [d for d in diffs if _unstable(old_path.name, d[0], names)]
    diffs = [d for d in diffs if not _unstable(old_path.name, d[0], names)]
    if unstable:
        cases = sorted({_describe(path, names).split(".")[0] for path, _, _ in unstable})
        print(
            f"{old_path.name}: {len(unstable)} unstable values differ "
            f"in {len(cases)} case(s) (ignored)"
        )
    if not diffs:
        return 0
    cases = sorted({_describe(path, names).split(".")[0] for path, _, _ in diffs})
    print(f"{old_path.name}: {len(diffs)} differing values in {len(cases)} case(s)")
    for path, o, n in diffs[:limit]:
        print(f"  {_describe(path, names)}: {o!r} -> {n!r}{_relative(o, n)}")
    if len(diffs) > limit:
        print(f"  ... {len(diffs) - limit} more")
    return len(diffs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old", type=Path, help="directory with the committed fixtures")
    parser.add_argument("new", type=Path, help="directory with the regenerated fixtures")
    parser.add_argument("--rtol", type=float, default=0.0, help="relative tolerance for numbers")
    parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for numbers")
    parser.add_argument("--limit", type=int, default=40, help="differences to print per file")
    args = parser.parse_args(argv)

    old_files = {p.name for p in args.old.glob("*.json")}
    new_files = {p.name for p in args.new.glob("*.json")}
    status = 0
    for name in sorted(old_files ^ new_files):
        side = "committed" if name in old_files else "regenerated"
        print(f"{name}: only in the {side} fixtures")
        status = 1
    for name in sorted(old_files & new_files):
        if compare_file(args.old / name, args.new / name, args.rtol, args.atol, args.limit):
            status = 1
    if status == 0:
        print(f"{len(old_files & new_files)} fixture files match")
    return status


if __name__ == "__main__":
    sys.exit(main())
