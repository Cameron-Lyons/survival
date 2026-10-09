"""Factor dispatch in expressions follows base R's Ops.factor/Ops.ordered.

The expected logical values and validation messages are derived independently
from the R 4.5 base methods, including their integer-code fallback for mixed
ordered/unordered factors. Model checks compare with explicit logical designs.
"""

import importlib

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from survival import r

expression = importlib.import_module("survival.r._expression")
formula = importlib.import_module("survival.r._formula")
coerce = importlib.import_module("survival.r._coerce")


def factor(values, levels=("z", "m", "a"), *, ordered=False):
    return expression._ExpressionVector(values, "factor", levels, ordered=ordered)


def evaluate(data, source):
    actual = formula._expression_values(data, source, len(next(iter(data.values()))))
    assert actual.kind == "logical"
    assert actual.categories is None
    return list(actual)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    ("operator", "expected"), [("==", [True, False, None]), ("!=", [False, True, None])]
)
def test_factor_equality_uses_labels_with_reordered_compatible_levels(ordered, operator, expected):
    data = {
        "g": factor(["z", "m", None], ordered=ordered),
        "h": factor(["z", "a", "z"], ("a", "m", "z"), ordered=ordered),
    }
    assert evaluate(data, f"I(identity(g)) {operator} h") == expected
    assert evaluate(data, f"h {operator} I(g)") == expected


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("operator", ["==", "!="])
@pytest.mark.parametrize("values", [["z", None], [None, None], []])
def test_factor_equality_validates_all_levels_before_values(ordered, operator, values):
    data = {
        "g": factor(values, ordered=ordered),
        "h": factor(values, ("z", "m", "a", "unused"), ordered=ordered),
    }
    with pytest.raises(ValueError, match="^level sets of factors are different$"):
        evaluate(data, f"g {operator} h")


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("<", [True, False, False, None]),
        ("<=", [True, True, False, None]),
        (">", [False, False, True, None]),
        (">=", [False, True, True, None]),
    ],
)
def test_ordered_comparisons_use_declared_levels_instead_of_lexical_order(operator, expected):
    data = {"g": factor(["z", "m", "a", None], ordered=True)}
    assert evaluate(data, f"g {operator} 'm'") == expected
    assert evaluate(data, f"I(identity(g)) {operator} 'm'") == expected
    opposite = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[operator]
    assert evaluate(data, f"'m' {opposite} g") == expected
    data["h"] = factor(["m"] * 4, ordered=True)
    assert evaluate(data, f"g {operator} h") == expected


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("values", [["z", "m", None], [None, None], []])
def test_ordered_comparisons_reject_different_level_order_even_when_missing(operator, values):
    data = {
        "g": factor(values, ordered=True),
        "h": factor(values, ("a", "m", "z"), ordered=True),
    }
    with pytest.raises(ValueError, match="^level sets of factors are different$"):
        evaluate(data, f"g {operator} h")


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("values", [["z", "m", None], [None, None], []])
def test_unordered_factor_ordering_warns_and_returns_missing(operator, values):
    with pytest.warns(UserWarning, match=f"'{operator}' not meaningful for factors"):
        actual = evaluate({"g": factor(values)}, f"g {operator} 'm'")
    assert actual == [None] * len(values)


def test_ordered_factor_matches_plain_values_to_levels_and_retains_missingness():
    data = {
        "g": factor(["z", "m", "a", None, "z"], ordered=True),
        "h": ["a", "unknown", "z", "m", None],
    }
    assert evaluate(data, "g < h") == [True, None, False, None, None]
    assert evaluate(data, "h > g") == [True, None, False, None, None]
    numeric = {"g": factor([2, 1, 3, None], (2, 1, 3), ordered=True), "x": [1, 2, 9, 1]}
    assert evaluate(numeric, "g < x") == [True, False, None, None]


@pytest.mark.parametrize("ordered_left", [False, True])
@pytest.mark.parametrize(
    ("operator", "expected"),
    [("==", [False, False, None]), ("!=", [True, True, None]), ("<", [True, False, None])],
)
def test_mixed_ordered_unordered_factors_warn_then_compare_integer_codes(
    ordered_left, operator, expected
):
    data = {
        "g": factor(["z", "m", None], ordered=ordered_left),
        "h": factor(["z", "m", "a"], ("m", "z", "a"), ordered=not ordered_left),
    }
    with pytest.warns(UserWarning, match="Incompatible methods") as caught:
        assert evaluate(data, f"g {operator} h") == expected
    methods = ("Ops.ordered", "Ops.factor") if ordered_left else ("Ops.factor", "Ops.ordered")
    assert str(caught[0].message) == (
        f'Incompatible methods ("{methods[0]}", "{methods[1]}") for "{operator}"'
    )


def test_factor_constructor_keeps_ordering_while_dropping_unused_levels():
    data = {"g": factor(["z", "a", None], ordered=True)}
    actual = formula._expression_values(data, "factor(I(g))", 3)
    assert actual.ordered
    assert actual.categories == ("z", "a")
    assert evaluate(data, "factor(g) < 'a'") == [True, False, None]


@pytest.mark.parametrize("source_kind", ["categorical", "series", "r_bridge"])
def test_ordered_source_metadata_survives_expression_evaluation_and_subsetting(source_kind):
    values = ["z", "m", "a", None]
    if source_kind == "r_bridge":
        source = coerce._r_factor(values, ["z", "m", "a"], ordered=True)
    else:
        source = pd.Categorical(values, categories=["z", "m", "a"], ordered=True)
        if source_kind == "series":
            source = pd.Series(source)
    assert evaluate({"g": source}, "g > 'm'") == [False, False, True, None]
    subset = coerce._rows_of(source, ["a", "z", None])
    assert subset.ordered
    assert evaluate({"g": subset}, "factor(g) > 'z'") == [True, False, None]


@pytest.mark.parametrize("kind", ["coxph", "survreg"])
@pytest.mark.parametrize("container", ["mapping", "dataframe"])
def test_ordered_formula_fits_and_predictions_match_explicit_logical_design(kind, container):
    index = np.arange(1, 49)
    grades: list[str | None] = ["z", "m", "a"] * 16
    grades[7] = None
    data = {
        "time": (10 + index * 37 % 97 + index / 7).tolist(),
        "status": ((index * 17 + 3) % 11 > 2).astype(int).tolist(),
        "x": np.sin(index * 0.63).tolist(),
        "g": pd.Categorical(grades, categories=["z", "m", "a"], ordered=True),
        "higher": [None if grade is None else grade == "a" for grade in grades],
    }
    if container == "dataframe":
        data = pd.DataFrame(data)
    fit = getattr(r, kind)(
        "Surv(time, status) ~ x + I(g > 'm')", data, na_action="na.exclude", model=True
    )
    explicit = getattr(r, kind)(
        "Surv(time, status) ~ x + higher", data, na_action="na.exclude", model=True
    )
    np.testing.assert_allclose(r.coef(fit), r.coef(explicit), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(r.vcov(fit), r.vcov(explicit), rtol=2e-13, atol=2e-13)
    assert tuple(fit.na_action.rows) == (8,)
    np.testing.assert_array_equal(r.model_matrix(fit)["data"], r.model_matrix(explicit)["data"])
    new = {
        "x": [0.0, 0.5, -0.5, 1.0],
        "g": pd.Categorical(["z", "m", "a", None], categories=["z", "m", "a"], ordered=True),
        "higher": [False, False, True, None],
    }
    if container == "dataframe":
        new = pd.DataFrame(new)
    for prediction_type in ("lp", "terms"):
        actual = r.predict(fit, new, type=prediction_type, na_action="na.exclude")
        expected = r.predict(explicit, new, type=prediction_type, na_action="na.exclude")
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True)
    # Missing comparison values participate in the prediction row policy.
    with pytest.raises(ValueError, match="missing"):
        r.predict(fit, {"x": [0.0], "g": factor([None], ordered=True)}, na_action="na.fail")


def test_incompatible_factor_equality_rejects_formula_model_frame():
    data = {
        "time": [1, 2, 3],
        "status": [1, 0, 1],
        "g": factor(["z", "m", "a"]),
        "h": factor(["z", "m", "a"], ("z", "m", "a", "unused")),
    }
    with pytest.raises(ValueError, match="level sets of factors are different"):
        r.coxph("Surv(time, status) ~ I(g == h)", data)
