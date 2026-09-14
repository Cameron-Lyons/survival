#!/usr/bin/env python3
"""Rebuild a KNOWN_FAILURES burndown list from an R_FIXTURES_COLLECT file.

Both fixture suites append one JSON line ``{"id", "kind", "message"}`` per
check when ``R_FIXTURES_COLLECT`` names a file (see test/r/README.md).  This
script turns such a file into the ``KNOWN_FAILURES`` literal of one suite:

    python test/r/burndown.py py.jsonl python --summary
    python test/r/burndown.py py.jsonl python --write   # edits test_r_fixtures.py
    python test/r/burndown.py rs.jsonl rust --write     # edits r_fixtures.rs

Without ``--write`` the literal is printed.  Entries are keyed by
``topic/case/aspect``, collapsed to ``topic/case`` when every checked aspect
of a case fails for the same category.  Reasons are shortened (5 significant
digits, 72 characters) so the lines stay readable; the collected file keeps
the full messages.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGETS = {
    "python": REPO / "python" / "tests" / "test_r_fixtures.py",
    "rust": REPO / "src" / "tests" / "r_fixtures.rs",
}

DATASET_MISMATCH = re.compile(
    r"column '.*' not found in data|no Python loader for dataset|no bundled CSV"
)
# API errors that really mean "this feature does not exist in the Python API".
MISSING_FEATURE = re.compile(
    r"unsupported formula term|only supported for|not supported|unexpected keyword argument|"
    r"requires a survfit result with a stored model frame|formula response must be Surv|"
    r"anova requires fitted Cox model|survSplit response must be a Surv object|"
    r"currently supports"
)


def category(record: dict) -> str | None:
    kind, message = record["kind"], record["message"]
    if kind == "pass":
        return None
    if DATASET_MISMATCH.search(message):
        return "dataset mismatch"
    if kind == "missing feature" or (kind == "error" and MISSING_FEATURE.search(message)):
        return "missing feature"
    if kind == "mismatch":
        return "mismatch"
    return "error"


def _round_number(match: re.Match) -> str:
    text = match.group(0)
    try:
        value = float(text)
    except ValueError:
        return text
    return f"{value:.5g}"


def reason(record: dict) -> str:
    message = record["message"].replace("unsupported: ", "")
    message = re.sub(r"\s+", " ", message).strip()
    message = re.sub(r" \((rtol|atol)=[^)]*\)", "", message)
    message = re.sub(r"\[[a-z_]+\.py:\d+\]", "", message).strip()
    message = re.sub(r"-?\d+\.\d+(e[-+]?\d+)?", _round_number, message)
    if len(message) > 72:
        message = message[:69] + "..."
    return f"{category(record)}: {message}"


def load_records(path: Path) -> dict[str, dict]:
    """Latest record per id wins (a run may append several times)."""

    by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_id[record["id"]] = record
    return by_id


def build_entries(by_id: dict[str, dict]) -> dict[str, str]:
    checks_per_case: dict[tuple[str, str], list] = collections.defaultdict(list)
    for test_id, record in by_id.items():
        topic, case, aspect = test_id.split("/", 2)
        checks_per_case[(topic, case)].append((aspect, record))
    entries: dict[str, str] = {}
    for (topic, case), items in sorted(checks_per_case.items()):
        failed = [(aspect, rec) for aspect, rec in items if category(rec) is not None]
        if not failed:
            continue
        categories = {category(rec) for _, rec in failed}
        if len(failed) == len(items) and len(categories) == 1:
            messages = collections.Counter(reason(rec) for _, rec in failed)
            entries[f"{topic}/{case}"] = messages.most_common(1)[0][0]
        else:
            for aspect, rec in failed:
                entries[f"{topic}/{case}/{aspect}"] = reason(rec)
    return entries


def print_summary(by_id: dict[str, dict], entries: dict[str, str]) -> None:
    failing = {tid: rec for tid, rec in by_id.items() if category(rec) is not None}
    totals = collections.Counter(category(rec) for rec in failing.values())
    print(f"{len(failing)} failing checks of {len(by_id)}; {len(entries)} KNOWN_FAILURES entries")
    for cat, count in totals.most_common():
        print(f"  {cat}: {count}")
    per_topic: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for tid, rec in failing.items():
        per_topic[tid.split("/")[0]][category(rec)] += 1
    for topic in sorted({tid.split("/")[0] for tid in by_id}):
        checked = sum(1 for tid in by_id if tid.startswith(topic + "/"))
        detail = " ".join(f"{k}={v}" for k, v in per_topic[topic].most_common())
        print(f"  {topic:22s} checked={checked:4d} {detail}")


def wrap_literal(text: str, width: int) -> list[str]:
    """Split text into pieces so each JSON-quoted piece fits in width."""

    pieces: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else current + " " + word
        if len(json.dumps(candidate)) > width and current:
            pieces.append(current + " ")
            current = word
        else:
            current = candidate
    pieces.append(current)
    return pieces


def render(entries: dict[str, str], flavour: str) -> str:
    if flavour == "python":
        lines = ["KNOWN_FAILURES: dict[str, str] = {"]
        for key, value in entries.items():
            line = f"    {json.dumps(key)}: {json.dumps(value)},"
            if len(line) <= 100:
                lines.append(line)
                continue
            lines.append(f"    {json.dumps(key)}: (")
            lines.extend(f"        {json.dumps(piece)}" for piece in wrap_literal(value, 88))
            lines.append("    ),")
        lines.append("}")
    else:
        lines = ["const KNOWN_FAILURES: &[(&str, &str)] = &["]
        lines.extend(
            f"    ({json.dumps(key)}, {json.dumps(value)})," for key, value in entries.items()
        )
        lines.append("];")
    return "\n".join(lines) + "\n"


BLOCK = {
    "python": re.compile(r"^KNOWN_FAILURES: dict\[str, str\] = \{\n.*?^\}\n", re.S | re.M),
    "rust": re.compile(r"^const KNOWN_FAILURES: &\[\(&str, &str\)\] = &\[\n.*?^\];\n", re.S | re.M),
}


def write_block(flavour: str, literal: str) -> Path:
    target = TARGETS[flavour]
    source = target.read_text(encoding="utf-8")
    pattern = BLOCK[flavour]
    if len(pattern.findall(source)) != 1:
        raise SystemExit(f"cannot find a single KNOWN_FAILURES block in {target}")
    target.write_text(pattern.sub(lambda _match: literal, source, count=1), encoding="utf-8")
    return target


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("collect", type=Path, help="R_FIXTURES_COLLECT JSONL file")
    parser.add_argument("flavour", choices=sorted(TARGETS))
    parser.add_argument("--summary", action="store_true", help="print per-topic counts")
    parser.add_argument("--write", action="store_true", help="replace the suite's list in place")
    args = parser.parse_args(argv)

    by_id = load_records(args.collect)
    entries = build_entries(by_id)
    if args.summary:
        print_summary(by_id, entries)
    literal = render(entries, args.flavour)
    if args.write:
        target = write_block(args.flavour, literal)
        print(f"wrote {len(entries)} entries to {target}")
        if args.flavour == "rust":
            print("run `cargo fmt` to reflow the long lines")
    elif not args.summary:
        sys.stdout.write(literal)


if __name__ == "__main__":
    main(sys.argv[1:])
