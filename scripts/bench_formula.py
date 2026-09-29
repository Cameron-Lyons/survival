"""Measure cached and uncached formula expansion, without fitting a model.

Run with PYTHONPATH=python .venv/bin/python scripts/bench_formula.py.
"""

from functools import partial
from statistics import median
from timeit import repeat

from survival.r._formula import _split_terms_cached


def main():
    for rhs in (
        "age + sex + ph.ecog",
        "age * sex + strata(ph.ecog) + offset(log(wt.loss))",
        "(a+b+c+d+e)^2",
        "(a+b+c)*d + e",
    ):
        print(rhs)
        for mode, parse in (
            ("uncached", _split_terms_cached.__wrapped__),
            ("cached", _split_terms_cached),
        ):
            parse(rhs, None)
            samples = repeat(partial(parse, rhs, None), number=1000, repeat=5)
            print(f"  {mode:>8}: {median(samples) * 1000:.2f} us/parse")


if __name__ == "__main__":
    main()
