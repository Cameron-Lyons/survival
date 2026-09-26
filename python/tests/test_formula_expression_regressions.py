"""Formula expressions against R 4.5.3 / survival 3.8-12: transforms of expressions and
comparisons, ``%in%``, and the literal vectors formula arguments spell out (``c()``, ``:``,
``seq()``).
"""

from __future__ import annotations

import importlib
import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
datasets = survival.datasets
r_formula = importlib.import_module("survival.r._formula")


def approx(values, rel=1e-12):
    return pytest.approx(values, rel=rel, abs=1e-15)


# --- transforms of expressions and comparisons -------------------------------------


@pytest.mark.parametrize(
    ("rhs", "names", "coefficients", "n"),
    [
        # coxph(Surv(time, status) ~ as.numeric(sex == 2), lung)
        ("as.numeric(sex == 2)", ["as.numeric(sex == 2)"], [-0.53102353761950805], 228),
        ("I(sex == 2)", ["I(sex == 2)TRUE"], [-0.53102353761950805], 228),
        ("I(sqrt(age))", ["I(sqrt(age))"], [0.28880912639889705], 228),
        ("log(age + 1)", ["log(age + 1)"], [1.1261713555124697], 228),
        ("exp(age/100)", ["exp(age/100)"], [1.0220253601339027], 228),
        ("log(sqrt(age))", ["log(sqrt(age))"], [2.2142699085892552], 228),
        (
            "I(ph.ecog >= 2) + age",
            ["I(ph.ecog >= 2)TRUE", "age"],
            [0.63182160923667285, 0.011871099308713128],
            227,
        ),
        (
            "I(age > 60):sex",
            ["I(age > 60)FALSE:sex", "I(age > 60)TRUE:sex"],
            [-0.58067096519430828, -0.48897061636002537],
            228,
        ),
        (
            "I(ph.ecog == 1) + offset(log(age + 1))",
            ["I(ph.ecog == 1)TRUE"],
            [-0.021862150819588585],
            227,
        ),
        # the exponent's sign is part of the number (R's label is I(age * 0.001))
        ("I(age*1e-3)", ["I(age*1e-3)"], [18.720179204551467], 228),
        # %in% is an interaction whose variables R orders by first appearance
        (
            "age + sex %in% inst",
            ["age", "sex:inst"],
            [0.018635593427548759, -0.015021308244069964],
            227,
        ),
        (
            "sex %in% age + age",
            ["age", "sex:age"],
            [0.028018215136491366, -0.0081969333421342515],
            228,
        ),
    ],
)
def test_coxph_transforms_of_expressions_match_r(rhs, names, coefficients, n):
    fit = r.coxph(f"Surv(time, status) ~ {rhs}", datasets.load_lung())
    assert list(fit.coef_names) == names
    assert fit.coefficients == approx(coefficients, rel=1e-9)
    assert fit.n == n


def test_a_nan_made_by_a_transform_of_an_expression_is_missing():
    # coxph(Surv(time, status) ~ sqrt(wt.loss + 10), lung): sqrt of a negative is NaN
    with pytest.warns(UserWarning, match="NaNs produced"):
        fit = r.coxph("Surv(time, status) ~ sqrt(wt.loss + 10)", datasets.load_lung())
    assert fit.coefficients == approx([0.044140903193740207], rel=1e-9)
    assert fit.n == 209
    assert len(fit.na_action) == 19


def test_logical_terms_in_the_other_formula_functions():
    lung = datasets.load_lung()
    # survreg(Surv(time, status) ~ I(sex == 2) + as.numeric(ph.ecog > 1), lung)
    fit = r.survreg("Surv(time, status) ~ I(sex == 2) + as.numeric(ph.ecog > 1)", lung)
    assert fit.coefficients == approx(
        [6.0018222626144837, 0.38534972330497136, -0.51421690397087472], rel=1e-9
    )
    assert fit.scale == approx([0.73860628615804724], rel=1e-9)
    # survdiff(Surv(time, status) ~ I(sex == 2), lung)
    test = r.survdiff("Surv(time, status) ~ I(sex == 2)", lung)
    assert test.groups == ["I(sex == 2)=FALSE", "I(sex == 2)=TRUE"]
    assert test.chisq == approx(10.326741954885632, rel=1e-9)
    # survfit(Surv(time, status) ~ I(age > 60), lung)
    curves = r.survfit("Surv(time, status) ~ I(age > 60)", lung)
    assert list(curves.strata) == ["I(age > 60)=FALSE", "I(age > 60)=TRUE"]
    assert curves.n == [94, 134]
    # pyears(Surv(time, status) ~ I(age > 60), lung, scale = 1)
    table = r.pyears("Surv(time, status) ~ I(age > 60)", lung, scale=1)
    assert table.dimnames == {"I(age > 60)": ["FALSE", "TRUE"]}
    assert table.pyears == [30530, 39063]
    # a comparison with a string: coxph(Surv(time, status) ~ I(fac == "m") + age, lung2)
    labelled = {**lung, "fac": ["m" if sex == 1 else "f" for sex in lung["sex"]]}
    fit = r.coxph('Surv(time, status) ~ I(fac == "m") + age', labelled)
    assert list(fit.coef_names) == ['I(fac == "m")TRUE', "age"]
    assert fit.coefficients == approx([0.51321851710838429, 0.017045331845411283], rel=1e-9)
    # predict(coxph(Surv(time, status) ~ I(sex == 2) + log(age + 1), lung),
    #         newdata = data.frame(sex = c(1, 2), age = c(50, 70)))
    fit = r.coxph("Surv(time, status) ~ I(sex == 2) + log(age + 1)", lung)
    assert r.predict(fit, newdata={"sex": [1, 2], "age": [50, 70]}) == approx(
        [-0.21232280708123705, -0.38718647692208075], rel=1e-9
    )


def test_a_penalty_term_reads_its_first_unnamed_argument():
    # coxph(Surv(time, status) ~ pspline(df = 4, age), lung)
    fit = r.coxph("Surv(time, status) ~ pspline(df = 4, age)", datasets.load_lung())
    assert fit.coefficients[:2] == approx([0.376920844214556, 0.754403211878295], rel=1e-9)


def test_a_comparison_in_surv_is_not_a_variable_name():
    # survSplit(Surv(time, status == 2) ~ ., data.frame(time = c(5, 15), status = c(1, 2),
    #           x = 1:2), cut = 10): the event column is "event", not "status == 2"
    data = {"time": [5, 15], "status": [1, 2], "x": [1, 2]}
    split = r.survSplit("Surv(time, status == 2) ~ .", data, cut=[10])
    assert list(split) == ["time", "status", "x", "tstart", "event"]
    assert split["event"] == [0, 0, 1]


def test_unsupported_expressions_raise_the_formula_error():
    lung = datasets.load_lung()
    for rhs in ("I(sex == 2 & age > 60)", "as.numeric(!sex)", "I(age | sex)", "log(age, 2)"):
        with pytest.raises(ValueError, match="unsupported formula|requires exactly one"):
            r.coxph(f"Surv(time, status) ~ {rhs}", lung)


def test_in_operator_builds_interactions_left_to_right():
    terms = r_formula._split_terms("age + sex %in% inst", None)
    labels = [":".join(factor.column for factor in term.factors) for term in terms.covariates[1:]]
    assert labels == ["sex:inst"]


# --- literal vectors ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "values"),
    [
        ("c(0, 65, 100) * 365.25", [0.0, 23741.25, 36525.0]),
        ("365.25 * c(0, 65)", [0.0, 23741.25]),
        ("0:2 * 50", [0.0, 50.0, 100.0]),
        ("-1:2", [-1.0, 0.0, 1.0, 2.0]),
        ("1.5:4", [1.5, 2.5, 3.5]),
        ("3:1", [3.0, 2.0, 1.0]),
        ("-2^2", [-4.0]),
        ("2^-1", [0.5]),
        ("seq(0, 100, 50)", [0.0, 50.0, 100.0]),
        ("seq(0, 1, length.out = 4)", [0.0, 1 / 3, 2 / 3, 1.0]),
        ("seq(0, 100, length = 3)", [0.0, 50.0, 100.0]),
        ("seq(10, 0, -2.5)", [10.0, 7.5, 5.0, 2.5, 0.0]),
        ("seq(5)", [1.0, 2.0, 3.0, 4.0, 5.0]),
        ("seq(1, by = 2, length.out = 3)", [1.0, 3.0, 5.0]),
        ("seq(to = 10, by = 2, length.out = 3)", [6.0, 8.0, 10.0]),
        ("seq(5, length.out = 3)", [5.0, 6.0, 7.0]),
        ("seq_len(3)", [1.0, 2.0, 3.0]),
        ("c(1, 2) + c(10, 20, 30, 40)", [11.0, 22.0, 31.0, 42.0]),
        ("c(-Inf, 0, Inf)", [-math.inf, 0.0, math.inf]),
        ("2 * 1e-3", [0.002]),
        ("c('a', \"b-c\")", ["a", "b-c"]),
        ("c(T, FALSE)", [True, False]),
        ("NULL", []),
        ("c()", []),
    ],
)
def test_literal_vectors_follow_r(expression, values):
    assert r_formula._literal_vector(expression) == values


def test_literal_vectors_evaluate_literals_only():
    # seq(0.1, 0.7, 0.1): from + (0:n) * by, the last clamped to `to`
    assert r_formula._literal_vector("seq(0.1, 0.7, 0.1)") == [
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
        0.6,
        0.7,
    ]
    with pytest.raises(ValueError, match="unsupported formula vector expression: brk"):
        r_formula._literal_vector("c(0, brk)")
    with pytest.raises(ValueError, match="wrong sign in 'by' argument"):
        r_formula._literal_vector("seq(1, 0, 1)")
    with pytest.raises(ValueError, match="too many arguments"):
        r_formula._literal_vector("seq(1, 2, 3, 4)")
    with pytest.raises(ValueError, match=r"unused argument \(bogus = 3\)"):
        r_formula._literal_vector("seq(1, 2, bogus = 3)")
    with pytest.raises(ValueError, match="matches multiple formal arguments"):
        r_formula._match_arguments("cut", ["age", "r = 1"], ("x", "right", "range"))
    with pytest.raises(ValueError, match="matched by multiple actual arguments"):
        r_formula._match_arguments("cut", ["x = age", "x = 1"], ("x", "breaks"))
