#!/usr/bin/env python3
"""Measure penalized-model column grouping, with equivalent outputs checked first."""

import argparse
import json
import platform
import statistics
import time

from survival.r._survpenal import assign_list


def repeated_scans(assign, term_labels, strata_term):
    """The previous grouping loop; inputs follow ordinary model-matrix column order."""
    labels, columns = [], []
    if 0 in assign:
        labels.append("(Intercept)")
        columns.append([j for j, term in enumerate(assign) if term == 0])
    for term_index, label in enumerate(term_labels, start=1):
        term_columns = [j for j, term in enumerate(assign) if term == term_index]
        if term_index != strata_term and term_columns:
            labels.append(label)
            columns.append(term_columns)
    return labels, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--columns-per-term", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.terms) < 2 or args.columns_per_term < 1 or args.repeats < 1:
        parser.error("need at least two terms and positive column/repetition counts")
    results = []
    for count in args.terms:
        labels = [f"term{i}" for i in range(count)]
        assign = [0] + [i for i in range(1, count + 1) for _ in range(args.columns_per_term)]
        inputs = (assign, labels, count // 2)
        if assign_list(*inputs) != repeated_scans(*inputs):
            raise RuntimeError("column groups differ from the previous implementation")
        timings = {}
        for name, function in (("repeated_scans", repeated_scans), ("single_pass", assign_list)):
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                result = function(*inputs)
                samples.append((time.perf_counter_ns() - started) / 1e6)
                del result
            timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
        results.append({"terms": count, "columns": len(assign), "timings": timings})
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "columns_per_term": args.columns_per_term,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
