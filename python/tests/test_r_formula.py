"""The formula layer: the tokenizer, R's term expansion and the shared ``model_frame``."""

import importlib
import math

import pytest

from .helpers import setup_survival_import
from .r_api_support import _numeric_data, _toy_data

survival = setup_survival_import()
r_types = importlib.import_module("survival.r._types")
r_formula = importlib.import_module("survival.r._formula")
r_coerce = importlib.import_module("survival.r._coerce")


def _term_name(term):
    if isinstance(term, r_types._InteractionTerm):
        return ":".join(r_formula._covariate_term_name(f) for f in term.factors)
    return r_formula._covariate_term_name(term)


# --- tokenizer ------------------------------------------------------------------


def test_tokenizer_respects_parentheses_backticks_and_quotes():
    assert r_formula._split_top_level("a:log(b, c):`d:e`", ":") == ["a", "log(b, c)", "`d:e`"]
    assert r_formula._split_top_level_token("a %in% (b %in% c)", "%in%") == ["a", "(b %in% c)"]
    assert r_formula._formula_tokens("- a + b - (c + d) +") == [
        ("-", "a"),
        ("+", "b"),
        ("-", "(c + d)"),
    ]
    assert r_formula._formula_name_items("`x y`, factor(a, b), c") == [
        ("x y", True),
        ("factor(a, b)", False),
        ("c", False),
    ]
    assert r_formula._formula_response_parts("time, status == 'a, b'") == [
        "time",
        "status == 'a, b'",
    ]
    assert r_formula._formula_named_option("type = 'left'") == ("type", "'left'")
    assert r_formula._formula_named_option("status == 2") is None
    assert r_formula._top_level_comparison("status >= 2") == ("status", ">=", "2")
    assert r_formula._top_level_comparison("I(a == 'b')") is None
    assert r_formula._find_top_level_arithmetic_operator("a - (b - c)", {"+", "-"}) == (
        "a",
        "-",
        "(b - c)",
    )
    assert r_formula._find_top_level_arithmetic_operator("-a", {"+", "-"}) is None
    assert r_formula._find_top_level_power_operator("(a + b)^2") == ("(a + b)", "^", "2")
    assert r_formula._strip_outer_formula_parentheses("((a + b))") == "a + b"
    assert r_formula._strip_outer_formula_parentheses("(a)+(b)") == "(a)+(b)"
    with pytest.raises(ValueError, match="unterminated backtick"):
        r_formula._formula_tokens("a + `b")
    with pytest.raises(ValueError, match="unterminated quote"):
        r_formula._formula_response_parts("status == 'a")


def test_term_expansion_follows_r_model_formulae():
    dot = ["x1", "x2", "x3"]
    cases = {
        "x1 * x2": ["x1", "x2", "x1:x2"],
        "x1 / x2": ["x1", "x1:x2"],
        "x1 + x2 %in% x1": ["x1", "x1:x2"],
        "(x1 + x2)^2": ["x1", "x2", "x1:x2"],
        "(x1 + x2 + x3)^2 - x1:x2": ["x1", "x2", "x3", "x1:x3", "x2:x3"],
        "(x1 + x2) * x3": ["x1", "x2", "x3", "x1:x3", "x2:x3"],
        "x1 * .": ["x1", "x2", "x3", "x1:x2", "x1:x3"],
        "x1:.": ["x1", "x1:x2", "x1:x3"],
        "(. - x2)^2": ["x1", "x3", "x1:x3"],
        "factor(x1) + log(x2) + I(x1^2)": ["factor(x1)", "log(x2)", "I(x1^2)"],
        "tcut(x1, c(0, 5)) + cut(x2 + 1, c(0, 5))": ["tcut(x1, c(0, 5))", "cut(x2 + 1, c(0, 5))"],
    }
    for rhs, expected in cases.items():
        terms = r_formula._split_terms(rhs, dot)
        assert [_term_name(t) for t in terms.covariates] == expected, rhs
    terms = r_formula._split_terms(
        "x1 + strata(x2, `x 3`) + cluster(id) + offset(log(x3)) - 1", dot
    )
    assert terms.strata == ["x2", "x 3"]
    assert terms.clusters == ["id"]
    assert [t.column for t in terms.offsets] == ["x3"]
    assert terms.intercept is False
    call_term = r_formula._split_terms("cut(x2 + 1, c(0, 5))", dot).covariates[0]
    assert (call_term.column, call_term.arithmetic, call_term.categorical) == ("x2", "x2 + 1", True)
    with pytest.raises(ValueError, match="unsupported formula term"):
        r_formula._split_terms("x1(x2)", dot)
    with pytest.raises(ValueError, match=r"factor\(\) requires exactly one column"):
        r_formula._split_terms("factor(x1, x2)", dot)
    with pytest.raises(ValueError, match="requires named tabular data"):
        r_formula._split_terms(".", None)


def test_formula_term_cache_returns_independent_terms():
    first = r_formula._split_terms("group + strata(x1) + offset(x2)", None)
    second = r_formula._split_terms("group + strata(x1) + offset(x2)", None)
    assert first is not second
    first.covariates.append(r_types._CovariateTerm("mutated"))
    first.strata.append("mutated")
    third = r_formula._split_terms("group + strata(x1) + offset(x2)", None)
    assert [term.column for term in third.covariates] == ["group"]
    assert third.strata == ["x1"]


# --- responses -----------------------------------------------------------------


def test_response_spec_covers_surv_plain_and_empty_left_hand_sides():
    surv = r_formula._response_spec("Surv(time, status == 2, type = 'right') ~ x")
    assert surv.surv
    assert surv.arguments == ("time", "status == 2")
    assert surv.type == "right"
    assert surv.columns == ("time", "status")
    assert surv.name == "Surv(time, status == 2)"
    named = r_formula._response_spec("Surv(event = s, time = t) ~ 1")
    assert named.arguments == ("t", "s")
    arithmetic = r_formula._response_spec("Surv(stop / 365.25, event) ~ 1")
    assert arithmetic.columns == ("stop", "event")
    plain = r_formula._response_spec("time ~ 1")
    assert plain is not None
    assert not plain.surv
    assert plain.name == "time"
    assert r_formula._response_spec("~ sex") is None
    with pytest.raises(ValueError, match="formula must contain '~'"):
        r_formula._response_spec("time")


def test_model_frame_builds_the_response_and_row_aligned_arguments():
    data = _toy_data()
    mf = r_formula.model_frame(
        "Surv(time, status) ~ x1 + strata(group) + offset(x2)",
        data,
        weights="x1",
        id=list(range(8)),
        offset=[1.0] * 8,
    )
    assert isinstance(mf.response, survival.Surv)
    assert mf.n == 8
    assert mf.response_name == "Surv(time, status)"
    assert mf.weights == data["x1"]
    assert mf.id == list(range(8))
    assert mf.offset == pytest.approx([1.0 + value for value in data["x2"]])
    assert mf.terms.strata == ["group"]
    assert dict(r_formula._model_variables(mf)) == {
        "x1": data["x1"],
        # a character strata variable gets R's short labels
        "strata(group)": ["A"] * 4 + ["B"] * 4,
        "offset(x2)": data["x2"],
    }
    groups = r_formula._model_strata(mf)
    assert len(groups.levels) == 8
    assert groups.levels[0] == "x1=0.1, strata(group)=A"
    plain = r_formula.model_frame("time ~ 1", data)
    assert plain.response is None
    assert plain.y == data["time"]
    empty = r_formula.model_frame("~ group", data)
    assert empty.response is None
    assert empty.y is None
    assert empty.n == 8
    assert r_formula._model_strata(empty).levels == ["group=A", "group=B"]
    assert r_formula._model_strata(plain) is None


def test_model_frame_applies_subset_then_na_action():
    data = {**_numeric_data(), "z": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
    mask = [True, True, True, True, False, False, False, False]
    kept = r_formula.model_frame("Surv(time, status) ~ z", data, subset=mask, na_action="na.omit")
    assert kept.n == 3
    assert list(kept.response.time) == [1.0, 3.0, 4.0]
    assert r_formula._column(kept.data, "z") == [1.0, 3.0, 4.0]
    passed = r_formula.model_frame("Surv(time, status) ~ z", data, na_action="na.pass")
    assert passed.n == 8
    with pytest.raises(ValueError, match="missing values in formula data"):
        r_formula.model_frame("Surv(time, status) ~ z", data, na_action="na.fail")
    extra = r_formula.model_frame(
        "Surv(time, status) ~ 1",
        data,
        na_action="na.omit",
        extra={"z": "z", "race": ["white"] * 8},
    )
    assert extra.n == 7
    assert extra.extra["race"] == ["white"] * 7
    with pytest.raises(ValueError, match="a data argument is required"):
        r_formula.model_frame("Surv(time, status) ~ 1", None)
    with pytest.raises(ValueError, match="must have the same length as the response"):
        r_formula.model_frame("Surv(time, status) ~ 1", data, weights=[1.0, 2.0])


def test_arithmetic_responses_and_call_terms_evaluate_against_data():
    data = {"stop": [365.25, 730.5], "event": [1, 0], "age": [40, 60]}
    mf = r_formula.model_frame("Surv(stop / 365.25, event) ~ cut(age + 48, c(0, 50, 100))", data)
    assert list(mf.response.time) == [1.0, 2.0]
    assert mf.terms.covariates[0].call == "cut(age + 48, c(0, 50, 100))"
    with pytest.raises(ValueError, match="unsupported formula term"):
        r_formula._term_values(data, mf.terms.covariates[0], 2)
    assert r_formula._response_arg_values(data, "stop / 365.25") == [1.0, 2.0]
    assert r_formula._response_arg_values(data, "event == 1") == [True, False]
    assert r_formula._response_arg_values(data, "rep(1, n)", 2) == [1, 1]
    assert math.isnan(
        r_formula._numeric_response({"t": [1, None]}, r_formula._response_spec("t ~ 1"), 2)[1]
    )
