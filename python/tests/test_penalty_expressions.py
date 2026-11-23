"""Penalty terms with expression arguments, which R's model frame evaluates.

``pspline(log(age))``, ``ridge(age, I(age^2))`` and ``frailty(factor(inst))`` fit as the
same terms on precomputed columns do, bit for bit, with R's deparsed argument names in
the coefficients and labels.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


@pytest.fixture(scope="module")
def lung():
    data = {name: list(values) for name, values in survival.datasets.load_lung().items()}
    data["log_age"] = [math.log(age) for age in data["age"]]
    data["sqrt_age"] = [math.sqrt(age) for age in data["age"]]
    data["age2"] = [age * age for age in data["age"]]
    data["age10"] = [age / 10 for age in data["age"]]
    shifted = [age - 40.5 for age in data["age"]]
    data["shifted"] = shifted
    data["log_shifted"] = [math.log(value) if value > 0 else math.nan for value in shifted]
    return data


PAIRS = [
    ("pspline(log(age))", "pspline(log_age)"),
    ("pspline(sqrt(age), df=2)", "pspline(sqrt_age, df=2)"),
    ("ridge(age, I(age^2), theta=1)", "ridge(age, age2, theta=1)"),
    ("ridge(age / 10, sex, theta=1)", "ridge(age10, sex, theta=1)"),
    ("frailty(factor(inst))", "frailty(inst)"),
    (
        "age + frailty(factor(inst), distribution='gaussian')",
        "age + frailty(inst, distribution='gaussian')",
    ),
    ("pspline(log(shifted))", "pspline(log_shifted)"),
    ("ridge(age, log(shifted), theta=1)", "ridge(age, log_shifted, theta=1)"),
]


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _same(actual, expected):
    """Bitwise-equal coefficients, an aliased (NaN) coefficient matching another."""
    return np.array_equal(
        np.asarray(list(actual), dtype=float),
        np.asarray(list(expected), dtype=float),
        equal_nan=True,
    )


def _fit_pair(data, expression, column, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        actual = r.coxph(f"Surv(time, status) ~ {expression} + sex", data, **kwargs)
        expected = r.coxph(f"Surv(time, status) ~ {column} + sex", data, **kwargs)
    return actual, expected


@pytest.mark.parametrize(("expression", "column"), PAIRS)
def test_expression_arguments_fit_as_precomputed_columns(lung, expression, column):
    actual, expected = _fit_pair(lung, expression, column)
    assert actual.n == expected.n
    assert list(actual.coefficients) == list(expected.coefficients)
    assert list(actual.loglik) == list(expected.loglik)
    assert np.array_equal(np.asarray(actual.var), np.asarray(expected.var))


def test_expression_arguments_drop_the_rows_they_make_missing(lung):
    actual, _expected = _fit_pair(lung, "pspline(log(shifted))", "pspline(log_shifted)")
    assert actual.n == len(lung["time"]) - sum(value <= 0 for value in lung["shifted"])


def test_coefficients_and_labels_carry_r_deparsed_arguments(lung):
    spline, _ = _fit_pair(lung, "pspline(log( age ))", "pspline(log_age)")
    assert r.coef_names(spline)[0] == "ps(log(age))3"
    ridge, _ = _fit_pair(
        lung, "ridge(age, I(age ^ 2), age/10, theta=1)", "ridge(age, age2, age10, theta=1)"
    )
    assert r.coef_names(ridge)[:3] == ["ridge(age)", "ridge(I(age^2))", "ridge(age/10)"]
    printed = str(r.print_coxph(ridge))
    assert "ridge(I(age^2))" in printed
    # R's constructors name pspline columns with deparse(substitute(x)), which keeps an
    # integer suffix, and ridge's with as.character() on cbind(...), which drops it.
    # (R's coxph and survreg then refuse the fit, a documented R defect the port fixes.)
    integer, _ = _fit_pair(lung, "pspline(age^2L, df=2)", "pspline(age2, df=2)")
    assert r.coef_names(integer)[0] == "ps(age^2L)3"
    simple, _ = _fit_pair(lung, "ridge(age, I(age^2L), theta=1)", "ridge(age, age2, theta=1)")
    assert r.coef_names(simple)[:2] == ["ridge(age)", "ridge(I(age^2))"]


def test_predictions_subsets_and_matrices_match_precomputed_columns(lung):
    actual, expected = _fit_pair(lung, "pspline(log(age))", "pspline(log_age)")
    newdata = {"age": [50.0, 60.0, 70.0], "sex": [1, 2, 1]}
    newdata["log_age"] = [math.log(age) for age in newdata["age"]]
    for kind in ("lp", "risk", "terms"):
        assert np.array_equal(
            np.asarray(r.predict(actual, newdata=newdata, type=kind)),
            np.asarray(r.predict(expected, newdata=newdata, type=kind)),
        )
    assert np.array_equal(
        np.asarray(r.predict(actual, type="expected")),
        np.asarray(r.predict(expected, type="expected")),
    )
    assert np.array_equal(
        np.asarray(r.model_matrix(actual)["data"]), np.asarray(r.model_matrix(expected)["data"])
    )
    keep = [row % 3 != 0 for row in range(len(lung["time"]))]
    subset, subset_expected = _fit_pair(lung, "pspline(log(age))", "pspline(log_age)", subset=keep)
    assert list(subset.coefficients) == list(subset_expected.coefficients)


def test_survreg_penalty_terms_evaluate_expressions(lung):
    actual = r.survreg("Surv(time, status) ~ pspline(log(age)) + ridge(I(age/10), theta=2)", lung)
    expected = r.survreg("Surv(time, status) ~ pspline(log_age) + ridge(age10, theta=2)", lung)
    assert list(actual.coefficients) == list(expected.coefficients)
    assert r.coef_names(actual)[-1] == "ridge(I(age/10))"


def test_frailty_predictions_reevaluate_factor_arguments(lung):
    actual, expected = _fit_pair(
        lung, "frailty(factor(inst), sparse=FALSE)", "frailty(inst, sparse=FALSE)"
    )
    rows = {name: [lung[name][row] for row in (0, 5, 9)] for name in ("inst", "sex", "age")}
    assert np.array_equal(
        np.asarray(r.predict(actual, newdata=rows)), np.asarray(r.predict(expected, newdata=rows))
    )
    # model_matrix(data=) refits the frailty constructor on the new rows.
    assert np.array_equal(
        np.asarray(r.model_matrix(actual, data=rows)["data"], dtype=float),
        np.asarray(r.model_matrix(expected, data=rows)["data"], dtype=float),
    )
    with pytest.raises(ValueError, match="new levels 999"):
        r.model_matrix(actual, data={**rows, "inst": [999.0, 3.0, 3.0]})


def test_backquoted_columns_stay_columns(lung):
    data = {
        **lung,
        "a-b": [age / 7 for age in lung["age"]],
        "log(age)": [age / 9 for age in lung["age"]],
    }
    data["plain"] = data["a-b"]
    data["decoy"] = data["log(age)"]
    data["age"] = [math.nan if row % 10 == 0 else age for row, age in enumerate(lung["age"])]
    data["log_age"] = [math.log(age) for age in data["age"]]
    keep = [sex == 1 for sex in lung["sex"]]
    for quoted, plain, kwargs in [
        ("pspline(`a-b`)", "pspline(plain)", {}),
        ("frailty(`a-b`)", "frailty(plain)", {}),
        ("ridge(`a-b`, age, theta=1)", "ridge(plain, age, theta=1)", {}),
        ("pspline(`log(age)`)", "pspline(decoy)", {}),
        ("pspline(`log(age)`)", "pspline(decoy)", {"subset": keep}),
        # An unquoted expression still evaluates, next to a column of the same name.
        ("pspline(log(age))", "pspline(log_age)", {}),
    ]:
        actual, expected = _fit_pair(data, quoted, plain, **kwargs)
        assert actual.n == expected.n, quoted
        assert _same(actual.coefficients, expected.coefficients), quoted


def test_missing_rows_follow_the_evaluated_arguments(lung):
    meal = lung["meal.cal"]
    data = {
        **lung,
        "imputed": [0.0 if _missing(value) else value for value in meal],
        "missing": [_missing(value) for value in meal],
        "log_karno": [
            math.nan if _missing(value) else math.log(value) for value in lung["ph.karno"]
        ],
    }
    pairs = [
        ("ridge(ifelse(is.na(meal.cal), 0, meal.cal), theta=1)", "ridge(imputed, theta=1)"),
        ("ridge(sex, is.na(meal.cal), theta=1)", "ridge(sex, missing, theta=1)"),
        ("ridge(I(age * ph.ecog^0), age, theta=1)", "ridge(age, age, theta=1)"),
        (
            "pspline(log(ph.karno), df=2) + strata(ph.ecog, na.group=TRUE)",
            "pspline(log_karno, df=2) + strata(ph.ecog, na.group=TRUE)",
        ),
        (
            "log(ph.karno) + strata(ph.ecog, na.group=TRUE)",
            "log_karno + strata(ph.ecog, na.group=TRUE)",
        ),
        (
            "frailty(factor(inst)) + strata(ph.ecog, na.group=TRUE)",
            "frailty(inst) + strata(ph.ecog, na.group=TRUE)",
        ),
    ]
    for expression, column in pairs:
        actual, expected = _fit_pair(data, expression, column)
        assert actual.n == expected.n, expression
        assert _same(actual.coefficients, expected.coefficients), expression
    complete, _ = _fit_pair(
        data, "ridge(ifelse(is.na(meal.cal), 0, meal.cal), sex, theta=1)", "sex"
    )
    assert complete.n == len(lung["time"])


def test_expressions_evaluate_their_source_text(lung):
    scale = "0.12345678901234567"
    data = {**lung, "scaled": [age * float(scale) for age in lung["age"]]}
    for expression, column in [
        (f"ridge(age * {scale}, theta=1)", "ridge(scaled, theta=1)"),
        (f"pspline(age * {scale}, df=2)", "pspline(scaled, df=2)"),
    ]:
        actual, expected = _fit_pair(data, expression, column)
        assert _same(actual.coefficients, expected.coefficients), expression


def test_factor_arguments_follow_cbind_and_range(lung):
    levels = sorted({value for value in lung["inst"] if not _missing(value)})
    codes = [
        math.nan if _missing(value) else float(levels.index(value) + 1) for value in lung["inst"]
    ]
    data = {
        **lung,
        "inst_code": codes,
        "sexf": r.Factor(["m" if sex == 1 else "f" for sex in lung["sex"]]),
    }
    data["sex_code"] = [1.0 if sex == "f" else 2.0 for sex in data["sexf"]]
    for expression, column in [
        ("ridge(factor(inst), theta=1)", "ridge(inst_code, theta=1)"),
        ("ridge(sexf, theta=1)", "ridge(sex_code, theta=1)"),
    ]:
        actual, expected = _fit_pair(data, expression, column)
        assert _same(actual.coefficients, expected.coefficients), expression
    for term in ("pspline(factor(sex))", "pspline(sexf)"):
        with pytest.raises(ValueError, match="'range' not meaningful for factors"):
            r.coxph(f"Surv(time, status) ~ {term}", data)


def test_ridge_matrix_columns_follow_cbind_names(lung):
    def columns(rhs):
        return r.model_matrix(r.coxph(f"Surv(time, status) ~ {rhs}", lung))["columns"]

    assert columns("ridge(age, I(age^2), theta = 1) + sex") == [
        "ridge(age, I(age^2), theta = 1)age",
        "ridge(age, I(age^2), theta = 1)",
        "sex",
    ]
    assert columns("ridge(age/10, sex, theta = 2)") == [
        "ridge(age/10, sex, theta = 2)",
        "ridge(age/10, sex, theta = 2)sex",
    ]
    assert columns("ridge(log(age), I(age^2), theta = 2)") == [
        "ridge(log(age), I(age^2), theta = 2)1",
        "ridge(log(age), I(age^2), theta = 2)2",
    ]


def test_arguments_sharing_a_name_but_not_their_values_are_refused(lung):
    data = {**lung, "log(age)": [age / 9 for age in lung["age"]]}
    for term in (
        "ridge(`log(age)`, log(age), theta=1)",
        "ridge(age * 0.1234567890123456, age * 0.1234567890123457, theta=1)",
    ):
        with pytest.raises(ValueError, match="has the same name"):
            r.coxph(f"Surv(time, status) ~ {term}", data)
    # The same values twice, however spelled, fit as before.
    for term in ("ridge(log(age), log( age ), theta=1)", "ridge(age, age, theta=1)"):
        assert r.coxph(f"Surv(time, status) ~ {term}", data).n == len(lung["time"])
