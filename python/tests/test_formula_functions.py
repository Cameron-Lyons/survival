"""R functions in formula terms: ``pmin``, ``ifelse``, ``factor(levels=)``, ``relevel``,
``interaction``, ``%in%`` and the Math group, against R survival 3.8.11.

Coefficient names are R's deparsed term labels, whatever the spacing of the formula.
"""

import math
import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets

from survival.r import _expression as expression  # noqa: E402
from survival.r import _formula as formula  # noqa: E402

NAN = math.nan

COXPH_CASES = [
    ("pmin(age,70)", ["pmin(age, 70)"], [0.0183135282098939], 228),
    ("pmax(age, 50)", ["pmax(age, 50)"], [0.0190116937751735], 228),
    (
        'ifelse(age>60,"old","young")',
        ['ifelse(age > 60, "old", "young")young'],
        [-0.216154492880907],
        228,
    ),
    ("factor(sex, levels=c(2,1))", ["factor(sex, levels = c(2, 1))1"], [0.531023537619506], 228),
    (
        "factor(sex,labels=c('m','f'))",
        ['factor(sex, labels = c("m", "f"))f'],
        [-0.531023537619508],
        228,
    ),
    (
        "interaction(sex,ph.ecog)",
        [f"interaction(sex, ph.ecog){level}" for level in ("2.0", "1.1", "2.1", "1.2", "2.2")]
        + ["interaction(sex, ph.ecog)1.3", "interaction(sex, ph.ecog)2.3"],
        [
            -0.677101401795788,
            0.404104001392517,
            -0.195812176654052,
            0.846905611817744,
            0.499163730649121,
            2.01808423414983,
            NAN,
        ],
        227,
    ),
    (
        'interaction(sex, ph.ecog, lex.order=TRUE, drop=TRUE, sep="_")',
        [
            f'interaction(sex, ph.ecog, lex.order = TRUE, drop = TRUE, sep = "_"){level}'
            for level in ("1_1", "1_2", "1_3", "2_0", "2_1", "2_2")
        ],
        [
            0.404104001392517,
            0.846905611817744,
            2.01808423414983,
            -0.677101401795789,
            -0.195812176654053,
            0.499163730649121,
        ],
        227,
    ),
    (
        "relevel(factor(ph.ecog),ref='1')",
        [f'relevel(factor(ph.ecog), ref = "1"){level}' for level in "023"],
        [-0.368840089769305, 0.547546948414438, 1.83914016613167],
        227,
    ),
    (
        "relevel(factor(ph.ecog), ref = 3)",
        [f"relevel(factor(ph.ecog), ref = 3){level}" for level in "013"],
        [-0.916387038183741, -0.547546948414437, 1.29159321771723],
        227,
    ),
    (
        "factor(ph.ecog, levels=c(3,2,1,0), labels='e')",
        [f'factor(ph.ecog, levels = c(3, 2, 1, 0), labels = "e")e{level}' for level in "234"],
        [-1.29159321771725, -1.83914016613169, -2.20798025590099],
        227,
    ),
    ("round(age/10)", ["round(age/10)"], [0.190353314611302], 228),
    ("signif(age, 1)", ["signif(age, 1)"], [0.0190353314611302], 228),
    ("I(ph.ecog %in% c(2,3))", ["I(ph.ecog %in% c(2, 3))TRUE"], [0.681120526412912], 228),
    ("as.character(sex)", ["as.character(sex)2"], [-0.531023537619508], 228),
    (
        "floor(age/20) + ceiling(meal.cal/500)",
        ["floor(age/20)", "ceiling(meal.cal/500)"],
        [0.275309422739513, -0.056807707023298],
        181,
    ),
    (
        "abs(age - 60) + sign(wt.loss)",
        ["abs(age - 60)", "sign(wt.loss)"],
        [0.0185477380358504, -0.00892347608702667],
        214,
    ),
    (
        "log1p(age) + expm1(age/100)",
        ["log1p(age)", "expm1(age/100)"],
        [-1.00889011018014, 1.9063792742348],
        228,
    ),
    ("log(age, 2)", ["log(age, 2)"], [0.767407472068685], 228),
    ("as.integer(age/10)", ["as.integer(age/10)"], [0.192209170999411], 228),
    ("is.na(meal.cal)", ["is.na(meal.cal)TRUE"], [-0.23385021726332], 228),
    (
        "pmin(wt.loss, 20, na.rm=TRUE)",
        ["pmin(wt.loss, 20, na.rm = TRUE)"],
        [0.0142110084024669],
        228,
    ),
    (
        "ifelse(is.na(ph.ecog), 0, ph.ecog)",
        ["ifelse(is.na(ph.ecog), 0, ph.ecog)"],
        [0.46194491332715],
        228,
    ),
]


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.mark.parametrize(("rhs", "names", "coefficients", "n"), COXPH_CASES)
def test_coxph_terms_match_r(lung, rhs, names, coefficients, n):
    fit = r.coxph(f"Surv(time, status) ~ {rhs}", lung)
    assert list(fit.coef_names) == names
    assert fit.n == n
    for actual, expected in zip(r.coef(fit), coefficients, strict=True):
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize(
    ("rhs", "strata", "chisq"),
    [
        (
            'ifelse(age>60,"old","young")',
            ['ifelse(age > 60, "old", "young")=old', 'ifelse(age > 60, "old", "young")=young'],
            1.83419593068427,
        ),
        (
            "interaction(sex,ph.ecog)",
            [
                f"interaction(sex, ph.ecog)={level}"
                for level in ("1.0", "2.0", "1.1", "2.1", "1.2", "2.2", "1.3")
            ],
            32.9169781906067,
        ),
        (
            "pmin(ph.ecog, 2)",
            ["pmin(ph.ecog, 2)=0", "pmin(ph.ecog, 2)=1", "pmin(ph.ecog, 2)=2"],
            18.9792776459786,
        ),
    ],
)
def test_curves_and_tests_group_by_function_terms(lung, rhs, strata, chisq):
    assert list(r.survfit(f"Surv(time, status) ~ {rhs}", lung).strata) == strata
    assert r.survdiff(f"Surv(time, status) ~ {rhs}", lung).chisq == pytest.approx(chisq, rel=1e-9)


def test_new_data_reuses_fitted_levels(lung):
    fit = r.coxph(
        "Surv(time, status) ~ ifelse(age>60,'old','young') + relevel(factor(ph.ecog), ref='1')"
        " + pmin(age, 70)",
        lung,
    )
    new = {"age": [50.0, 65.0, 75.0], "ph.ecog": [0, 2, 1], "sex": [1, 2, 1]}
    assert r.predict(fit, newdata=new, type="lp") == pytest.approx(
        [-0.402294252129779, 0.598367250474664, 0.189771269499231], rel=1e-9
    )
    assert fit.xlevels == {
        'ifelse(age > 60, "old", "young")': ["old", "young"],
        'relevel(factor(ph.ecog), ref = "1")': ["1", "0", "2", "3"],
    }


def _values(expression, **columns):
    n = len(next(iter(columns.values())))
    return formula._expression_values(columns, expression, n)


def _same(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        if isinstance(b, float) and math.isnan(b):
            assert isinstance(a, float)
            assert math.isnan(a)
        else:
            assert a == b


X = [1.0, None, 3.0, NAN]
T = [True, None, False, True]


def test_ifelse_takes_the_type_of_the_branches_it_uses():
    result = _values('ifelse(t, "yes", x)', t=T, x=X)
    assert result.kind == "character"
    assert list(result) == ["yes", None, "3", "yes"]
    result = _values("ifelse(t, x, 0L)", t=T, x=X)
    assert result.kind == "numeric"
    _same(result, [1.0, NAN, 0.0, NAN])
    result = _values('ifelse(t, "a", 2L)', t=[False, False])
    assert result.kind == "numeric"
    assert result.storage == "integer"
    assert list(result) == [2, 2]


def test_pmin_and_pmax_follow_r_missing_values():
    result = _values("pmin(x, 2)", x=X)
    _same(result, [1.0, NAN, 2.0, NAN])
    assert expression._is_na_real(result[1])
    assert not expression._is_na_real(result[3])
    _same(_values("pmax(x, 2, na.rm = TRUE)", x=X), [2.0, 2.0, 3.0, 2.0])
    result = _values("pmin(x, 3L)", x=[1, 5])
    assert result.storage == "integer"
    assert list(result) == [1, 3]


def test_in_matches_na_and_nan_separately():
    assert list(_values("I(x %in% c(NA, 3))", x=X)) == [False, True, True, False]
    assert list(_values("I(x %in% NaN)", x=X)) == [False, False, False, True]
    assert list(_values('I(y %in% c("a", NA))', y=["a", "b", "c", None])) == [
        True,
        False,
        False,
        True,
    ]


def test_rounding_coercion_and_labels():
    assert list(_values("round(x)", x=[0.5, 1.5, 2.5, -0.5, 2.675])) == [0, 2, 2, 0, 3]
    assert list(_values("round(x, 2)", x=[2.675])) == [2.67]
    assert list(_values("signif(x, 2)", x=[123456.0])) == [120000.0]
    with pytest.warns(UserWarning, match="coercion to integer range"):
        result = _values("as.integer(x)", x=[3e9, -2.7, None])
    assert result[1] == -2
    assert math.isnan(result[0])
    assert math.isnan(result[2])
    assert list(_values("as.logical(x)", x=["T", "yes", "false", None])) == [
        True,
        None,
        False,
        None,
    ]
    assert list(_values("as.character(x)", x=[1.0, 0.1, 1e-20, 1e5, None, NAN])) == [
        "1",
        "0.1",
        "1e-20",
        "1e+05",
        None,
        "NaN",
    ]
    assert list(_values("is.na(x)", x=X)) == [False, True, False, True]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _values("log1p(x)", x=[0.0, -1.0])
    with pytest.warns(UserWarning, match="NaNs produced"):
        _values("log2(x)", x=[-1.0])


def test_factor_levels_labels_and_interaction_order():
    result = _values('factor(y, levels = c("b", "a"))', y=["a", "b", "c", None])
    assert result.categories == ("b", "a")
    assert list(result) == ["a", "b", None, None]
    result = _values('factor(x, labels = c("x", "x", "y"))', x=[1.0, 2.0, 3.0])
    assert result.categories == ("x", "y")
    assert list(result) == ["x", "x", "y"]
    result = _values("interaction(a, b)", a=["a", "b"], b=["x", "y"])
    assert result.categories == ("a.x", "b.x", "a.y", "b.y")
    assert list(result) == ["a.x", "b.y"]
    result = _values("interaction(a, b, lex.order = TRUE)", a=["a", "b"], b=["x", "y"])
    assert result.categories == ("a.x", "a.y", "b.x", "b.y")
    with pytest.raises(ValueError, match="'ref' must be an existing level"):
        _values('relevel(factor(y), ref = "z")', y=["a", "b"])
    with pytest.raises(ValueError, match="factor level \\[2\\] is duplicated"):
        _values("factor(y, levels = c('a', 'a'))", y=["a", "b"])
