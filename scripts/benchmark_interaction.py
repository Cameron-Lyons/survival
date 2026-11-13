"""Measure time and peak allocations for sparse interaction(..., drop=TRUE) terms.

Run with PYTHONPATH=python .venv/bin/python scripts/benchmark_interaction.py.
The diagonal observations use only one combination per row, while the full
Cartesian product would contain levels ** factors combinations.
"""

import argparse
import json
import platform
import tracemalloc
from statistics import median
from time import perf_counter

from survival import r
from survival.r._formula import _expression_values


def measure(level_count, factor_count, lex_order, repeats):
    levels = [f"level{i:08d}" for i in range(level_count)]
    data = {f"x{i}": r.Factor(levels, levels) for i in range(factor_count)}
    expression = (
        f"interaction({', '.join(data)}, drop = TRUE, "
        f"lex.order = {'TRUE' if lex_order else 'FALSE'})"
    )

    def evaluate():
        return _expression_values(data, expression, level_count)

    expected = [".".join([level] * factor_count) for level in levels]
    result = evaluate()
    if list(result) != expected or list(result.categories) != expected:
        raise AssertionError("interaction labels differ from the observed combinations")
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        evaluate()
        samples.append(1000 * (perf_counter() - start))
    tracemalloc.start()
    try:
        evaluate()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "levels": level_count,
        "factors": factor_count,
        "lex_order": lex_order,
        "median_ms": median(samples),
        "peak_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[60, 1000, 10000])
    parser.add_argument("--factors", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(*args.levels, *args.factors, args.repeats) < 1:
        parser.error("levels, factors and repeats must be positive")
    results = [
        measure(levels, factors, lex_order, args.repeats)
        for levels in args.levels
        for factors in args.factors
        for lex_order in (False, True)
    ]
    print(json.dumps({"python": platform.python_version(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
