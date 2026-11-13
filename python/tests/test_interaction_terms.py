"""R interaction levels, collisions and bounded storage for sparse combinations."""

import importlib
import json
import tracemalloc
from pathlib import Path

import pytest
from survival import r

formula = importlib.import_module("survival.r._formula")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/interaction_reference.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_interaction_values_and_level_order_match_r(case):
    data = {
        name: r.Factor(column["values"], column["levels"]) for name, column in case["input"].items()
    }
    expression = (
        f"interaction({', '.join(data)}, drop = {'TRUE' if case['drop'] else 'FALSE'}, "
        f"lex.order = {'TRUE' if case['lex_order'] else 'FALSE'}, sep = {case['sep']!r})"
    )
    result = formula._expression_values(data, expression, len(next(iter(data.values()))))
    assert list(result) == case["values"]
    assert list(result.categories) == case["levels"]


@pytest.mark.parametrize("lex_order", ["TRUE", "FALSE"])
def test_dropped_interactions_allocate_only_observed_combinations(lex_order):
    levels = [f"level{i:02d}" for i in range(60)]
    data = {name: r.Factor(levels, levels) for name in ("a", "b", "c")}
    expression = f"interaction(a, b, c, drop = TRUE, lex.order = {lex_order})"
    # The full Cartesian product is 216,000 labels although only 60 occur.
    # Bound allocations, not noisy elapsed time, to guard against its return.
    tracemalloc.start()
    try:
        result = formula._expression_values(data, expression, len(levels))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert list(result) == [f"{value}.{value}.{value}" for value in levels]
    assert list(result.categories) == list(result)
    assert peak < 4_000_000


def test_dropped_interaction_groups_curves_in_r_level_order():
    data = {
        "time": [1, 2, 3, 4],
        "status": [1, 1, 0, 1],
        "a": r.Factor(["b", "a", "b", "a"], ["b", "a", "unused"]),
        "b": r.Factor(["y", "x", "x", "y"], ["y", "x", "unused"]),
    }
    fit = r.survfit("Surv(time, status) ~ interaction(a, b, drop = TRUE)", data, subset=[2, 0])
    assert fit.strata == {
        "interaction(a, b, drop = TRUE)=b.y": 1,
        "interaction(a, b, drop = TRUE)=b.x": 1,
    }
    assert fit.time == [1.0, 3.0]
    assert fit.surv == [0.0, 1.0]
