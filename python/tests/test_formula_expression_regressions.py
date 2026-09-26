"""Formula expressions against R 4.5.3 / survival 3.8-12: ``cut()`` and ``tcut()`` terms of
``pyears`` (R's argument matching, ``base::cut.default``, values outside the breaks dropped
by ``na.action``), precomputed ``tcut`` columns, literal vectors (``c()``, ``:``,
``seq()``), ``rmap`` expressions, transforms of expressions and comparisons, and ``%in%``.
"""

from __future__ import annotations

import datetime
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


# R's d: data.frame(time = c(10, 20, 30, 40), status = c(1, 0, 1, 1), age = c(50, 65, 70, 80))
SMALL = {"time": [10, 20, 30, 40], "status": [1, 0, 1, 1], "age": [50, 65, 70, 80]}


def _pyears(term, data=SMALL, **kwargs):
    return r.pyears(f"Surv(time, status) ~ {term}", data, scale=1, **kwargs)


# --- cut() -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "levels", "pyears", "n", "event"),
    [
        # pyears(Surv(time, status) ~ cut(age, c(0, 65, 100), right = FALSE), d, scale = 1)
        (
            "cut(age, c(0, 65, 100), right = FALSE)",
            ["[0,65)", "[65,100)"],
            [10, 90],
            [1, 3],
            [1, 2],
        ),
        (
            "cut(age, c(50, 65, 100), include.lowest = TRUE)",
            ["[50,65]", "(65,100]"],
            [30, 70],
            [2, 2],
            [1, 2],
        ),
        (
            "cut(age, c(0, 65, 80), right = FALSE, include.lowest = TRUE)",
            ["[0,65)", "[65,80]"],
            [10, 90],
            [1, 3],
            [1, 2],
        ),
        # arguments by position, partial name and in any order, as R matches them
        (
            "cut(age, c(0, 65, 100), NULL, TRUE, FALSE)",
            ["[0,65)", "[65,100]"],
            [10, 90],
            [1, 3],
            [1, 2],
        ),
        ("cut(breaks = c(0, 65, 100), age)", ["(0,65]", "(65,100]"], [30, 70], [2, 2], [1, 2]),
        ("cut(age, c(0, 65, 100), right = F)", ["[0,65)", "[65,100)"], [10, 90], [1, 3], [1, 2]),
        ("cut(age, c(0, 65, 100), lab = c('y', 'o'))", ["y", "o"], [30, 70], [2, 2], [1, 2]),
        ("cut(age, c(0, 65, 100), c('young', 'old'))", ["young", "old"], [30, 70], [2, 2], [1, 2]),
        # formatC(breaks, digits = 3): 1e+03, and wider only when labels would collide
        (
            "cut(time, c(0, 1000, 2000, 5000))",
            ["(0,1e+03]", "(1e+03,2e+03]", "(2e+03,5e+03]"],
            [100, 0, 0],
            [4, 0, 0],
            [3, 0, 0],
        ),
        ("cut(age, c(-Inf, 65, Inf))", ["(-Inf,65]", "(65, Inf]"], [30, 70], [2, 2], [1, 2]),
        ("cut(age, c(100, 0, 65))", ["(0,65]", "(65,100]"], [30, 70], [2, 2], [1, 2]),
        (
            "cut(age, c(0, 65.123456789012, 65.123456789013, 100))",
            ["Range_1", "Range_2", "Range_3"],
            [30, 0, 70],
            [2, 0, 2],
            [1, 0, 2],
        ),
        # a number of intervals over the range of x, widened by a thousandth
        ("cut(age, 3)", ["(50,60]", "(60,70]", "(70,80]"], [10, 50, 40], [1, 2, 1], [1, 1, 1]),
        # labels = FALSE: the codes, which as.factor levels by those in use
        (
            "cut(age, c(0, 55, 60, 70, 100), labels = FALSE)",
            ["1", "3", "4"],
            [10, 50, 40],
            [1, 2, 1],
            [1, 1, 1],
        ),
        # factor() merges intervals that share a label
        (
            "cut(age, c(0, 60, 70, 100), labels = c('a', 'a', 'b'))",
            ["a", "b"],
            [60, 40],
            [3, 1],
            [2, 1],
        ),
        (
            "cut(age, c(0, 65, 100), ordered_result = TRUE)",
            ["(0,65]", "(65,100]"],
            [30, 70],
            [2, 2],
            [1, 2],
        ),
    ],
)
def test_cut_terms_follow_cut_default(term, levels, pyears, n, event):
    result = _pyears(term)
    assert result.dimnames == {term: levels}
    assert result.pyears == pyears
    assert result.n == n
    assert result.event == event
    assert result.observations == 4
    assert result.na_action is None


def test_cut_values_outside_the_breaks_are_missing():
    # pyears(Surv(time, status) ~ cut(age, c(55, 65, 100)), d, scale = 1)
    result = _pyears("cut(age, c(55, 65, 100))")
    assert result.pyears == [20, 70]
    assert result.n == [1, 2]
    assert result.observations == 3
    assert result.na_action.rows == (1,)
    assert len(result.na_action) == 1
    # ... cut(age, c(0, 65, 80), right = FALSE): 80 is outside [65, 80)
    result = _pyears("cut(age, c(0, 65, 80), right = FALSE)")
    assert result.pyears == [10, 50]
    assert result.na_action.rows == (4,)
    # ... cut(age * 20, c(1000, 2000, 3000, 12345)) and with dig.lab = 5
    result = _pyears("cut(age * 20, c(1000, 2000, 3000, 12345))")
    assert list(result.dimnames.values()) == [
        ["(1e+03,2e+03]", "(2e+03,3e+03]", "(3e+03,1.23e+04]"]
    ]
    assert result.pyears == [90, 0, 0]
    assert result.observations == 3
    result = _pyears("cut(age * 20, c(1000, 2000, 3000, 12345), dig.lab = 5)")
    assert list(result.dimnames.values()) == [["(1000,2000]", "(2000,3000]", "(3000,12345]"]]
    # na.exclude drops the row as well; na.fail stops
    assert _pyears("cut(age, c(55, 65, 100))", na_action="na.exclude").observations == 3
    with pytest.raises(ValueError, match="missing values"):
        _pyears("cut(age, c(55, 65, 100))", na_action="na.fail")


def test_cut_labels_widen_until_the_breaks_differ():
    # pyears(Surv(time, status) ~ cut(x, c(1.001, 1.002, 1.003, 1.004)), d, scale = 1)
    # with x = c(1.0015, 1.0025, 1.0035, 1.0012)
    data = {**SMALL, "x": [1.0015, 1.0025, 1.0035, 1.0012]}
    result = _pyears("cut(x, c(1.001, 1.002, 1.003, 1.004))", data)
    assert list(result.dimnames.values()) == [["(1.001,1.002]", "(1.002,1.003]", "(1.003,1.004]"]]
    assert result.pyears == [50, 20, 30]
    # d5 <- data.frame(time = 1:30, status = rep(c(1, 0, 1), 10),
    #                  x = seq(1.0, 1.006, length.out = 30)): 15 rows outside the breaks
    data = {
        "time": list(range(1, 31)),
        "status": [1, 0, 1] * 10,
        "x": r_formula._literal_vector("seq(1.0, 1.006, length.out = 30)"),
    }
    result = _pyears("cut(x, c(1.001, 1.002, 1.003, 1.004), dig.lab = 2)", data)
    assert list(result.dimnames.values()) == [["(1.001,1.002]", "(1.002,1.003]", "(1.003,1.004]"]]
    assert result.pyears == [40, 65, 90]
    assert result.n == [5, 5, 5]
    assert result.observations == 15
    assert result.na_action.rows == (1, 2, 3, 4, 5, *range(21, 31))
    # constant x: the breaks straddle it by a thousandth of |x| (or of 1 at zero)
    result = _pyears("cut(age, 2)", {**SMALL, "age": [65] * 4})
    assert list(result.dimnames.values()) == [["(64.9,65]", "(65,65.1]"]]
    result = _pyears("cut(age, 2)", {**SMALL, "age": [0] * 4})
    assert list(result.dimnames.values()) == [["(-0.001,0]", "(0,0.001]"]]


def test_cut_argument_errors_follow_r():
    with pytest.raises(ValueError, match="'breaks' are not unique"):
        _pyears("cut(age, c(0, 65, 65, 100))")
    with pytest.raises(ValueError, match="invalid number of intervals"):
        _pyears("cut(age, 1)")
    with pytest.raises(ValueError, match="number of intervals and length of 'labels' differ"):
        _pyears("cut(age, c(0, 65, 100), labels = c('a', 'b', 'c'))")
    with pytest.raises(ValueError, match="number of intervals and length of 'labels' differ"):
        _pyears("cut(age, c(0, 65, 100), TRUE)")
    # R's cut.default swallows an unknown argument in ...; the port refuses it
    with pytest.raises(ValueError, match=r"unused argument \(bogus = 1\)"):
        _pyears("cut(age, c(0, 65, 100), bogus = 1)")
    with pytest.raises(ValueError, match='argument "breaks" is missing'):
        _pyears("cut(age)")


def _lung_male():
    lung = datasets.load_lung()
    return lung, [sex == 1 for sex in lung["sex"]]


def test_cut_and_tcut_of_a_number_of_intervals_use_the_whole_data():
    # pyears(Surv(time, status) ~ cut(age, 3), lung, scale = 1, subset = sex == 1)
    lung, male = _lung_male()
    result = r.pyears("Surv(time, status) ~ cut(age, 3)", lung, scale=1, subset=male)
    assert list(result.dimnames.values()) == [["(39,53.3]", "(53.3,67.7]", "(67.7,82]"]]
    assert result.pyears == [5758, 20169, 13159]
    assert result.n == [20, 67, 51]
    assert result.observations == 138
    # ... tcut(age, 3): R's cutpoints 38.57, 53.19, 67.81, 82.43 from all 228 ages
    result = r.pyears("Surv(time, status) ~ tcut(age, 3)", lung, scale=1, subset=male)
    assert result.pyears == approx([110.80000000000004, 735.66999999999962, 1779.0099999999993])
    assert result.n == [20, 87, 138]
    result = r.pyears("Surv(time, status) ~ tcut(age, 3)", lung, scale=1)
    assert result.pyears == approx([199.41000000000003, 1351.8999999999962, 2983.4399999999923])


# --- tcut() ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("term", "levels", "pyears"),
    [
        # pyears(Surv(time, status) ~ tcut(age, c(0, 65, 100), scale = 2), lung, scale = 1)
        ("tcut(age, c(0, 65, 100), scale = 2)", [" 0+ thru  65", "65+ thru 100"], [2300, 14239]),
        ("tcut(age, c(0, 65, 100), c('young', 'old'))", ["young", "old"], [1150, 7273]),
        ("tcut(breaks = c(0, 65, 100), x = age)", [" 0+ thru  65", "65+ thru 100"], [1150, 7273]),
        ("tcut(age * 1, c(0, 65, 100))", [" 0+ thru  65", "65+ thru 100"], [1150, 7273]),
        ("tcut(age, seq(0, 100, 50))", [" 0+ thru  50", "50+ thru 100"], [108, 8315]),
        ("tcut(age, seq(0, 100, length.out = 3))", [" 0+ thru  50", "50+ thru 100"], [108, 8315]),
        ("tcut(age, 0:2 * 50)", [" 0+ thru  50", "50+ thru 100"], [108, 8315]),
        ("cut(age, breaks = seq(0, 100, 50))", ["(0,50]", "(50,100]"], [8494, 61099]),
        ("cut(age, c(0, 65, 100), right = FALSE)", ["[0,65)", "[65,100)"], [40989, 28604]),
    ],
)
def test_tcut_and_cut_arguments_on_lung(term, levels, pyears):
    result = r.pyears(f"Surv(time, status) ~ {term}", datasets.load_lung(), scale=1)
    assert result.dimnames == {term: levels}
    assert result.pyears == pyears


def _hearta():
    """R's ``hearta``: the last row of each subject of ``heart`` (pyears.Rd)."""

    heart = datasets.load_heart()
    last: dict[int, int] = {}
    for row, (subject, stop) in enumerate(zip(heart["id"], heart["stop"], strict=True)):
        if subject not in last or stop >= heart["stop"][last[subject]]:
            last[subject] = row
    rows = sorted(last.values())
    return {name: [values[row] for row in rows] for name, values in heart.items()}


def test_pyears_rd_example_with_tcut_of_an_expression():
    # hearta <- do.call("rbind", by(heart, heart$id, function(x) x[x$stop == max(x$stop), ]))
    # pyears(Surv(stop/365.25, event) ~ tcut(age + 48, c(0,50,60,70,100)) + surgery,
    #        hearta, scale = 1)
    result = r.pyears(
        "Surv(stop/365.25, event) ~ tcut(age + 48, c(0,50,60,70,100)) + surgery",
        _hearta(),
        scale=1,
    )
    assert [row[0] for row in result.pyears] == approx(
        [44.925393566050637, 16.750171115674206, 0.75564681724845861, 0.0]
    )
    assert [row[1] for row in result.pyears] == approx(
        [18.960985626283367, 6.093086926762493, 0, 0]
    )
    assert result.n == [[56, 13], [33, 6], [3, 0], [0, 0]]
    assert result.event == [[36, 5], [27, 4], [3, 0], [0, 0]]
    assert result.observations == 103
    assert result.tcut is True


# --- precomputed tcut columns --------------------------------------------------------


def _cohort_with_tcut():
    # d$age[2] <- NA; d$agecut <- tcut(d$age, c(0, 65, 100) * 365.25)
    age = [60 * 365.25, math.nan, 65 * 365.25, 80 * 365.25]
    return {
        "time": [100, 400, 900, 300],
        "status": [1, 0, 1, 1],
        "age": age,
        "sex": [1, 2, 1, 2],
        "year": [
            datetime.date(1995, 3, 1),
            datetime.date(1996, 6, 15),
            datetime.date(1997, 1, 1),
            datetime.date(1998, 9, 9),
        ],
        "grp": ["a", "b", "a", "b"],
        "agecut": r.tcut(age, [0, 65 * 365.25, 100 * 365.25]),
    }


def test_a_tcut_column_keeps_its_cutpoints_through_na_action_and_subset():
    data = _cohort_with_tcut()
    # pyears(Surv(time, status) ~ grp + agecut, d, scale = 365.25)
    result = r.pyears("Surv(time, status) ~ grp + agecut", data, scale=365.25)
    assert result.pyears[0] == approx([0.27378507871321012, 2.4640657084188913])
    assert result.pyears[1] == approx([0.0, 0.82135523613963035])
    assert result.n == [[1, 1], [0, 1]]
    assert result.dimnames["agecut"] == ["    0.00+ thru 23741.25", "23741.25+ thru 36525.00"]
    assert result.tcut is True
    assert result.observations == 3
    assert result.na_action.rows == (2,)
    # ... subset = c(1, 2, 4)
    result = r.pyears("Surv(time, status) ~ grp + agecut", data, scale=365.25, subset=[0, 1, 3])
    assert result.pyears[0] == approx([0.27378507871321012, 0.0])
    assert result.pyears[1] == approx([0.0, 0.82135523613963035])
    assert result.observations == 2
    # ... ~ agecut, na.action = na.pass, subset = c(1, 3, 4)
    result = r.pyears(
        "Surv(time, status) ~ agecut", data, scale=365.25, na_action="na.pass", subset=[0, 2, 3]
    )
    assert result.pyears == approx([0.27378507871321012, 3.2854209445585214])


def test_survexp_refuses_a_tcut_column():
    rmap = {"age": "age", "sex": "sex", "year": "year"}
    with pytest.raises(ValueError, match="Can't use tcut variables in expected survival"):
        r.survexp("~ agecut", _cohort_with_tcut(), rmap=rmap, times=[1])
    with pytest.raises(ValueError, match="Can't use tcut variables in expected survival"):
        r.survexp("~ agecut", _cohort_with_tcut(), rmap=rmap, times=[1], na_action="na.pass")


# --- rmap expressions ----------------------------------------------------------------


def _cohort():
    return {
        "time": [100, 400, 900, 300],
        "status": [1, 0, 1, 1],
        "ageyr": [60, 70, 65, 80],
        "sex": [1, 2, 1, 2],
        "year": [
            datetime.date(1995, 3, 1),
            datetime.date(1996, 6, 15),
            datetime.date(1997, 1, 1),
            datetime.date(1998, 9, 9),
        ],
        "birth": [
            datetime.date(1935, 3, 1),
            datetime.date(1926, 6, 15),
            datetime.date(1932, 1, 2),
            datetime.date(1918, 9, 9),
        ],
        "grp": ["a", "b", "a", "b"],
    }


def test_rmap_expressions_are_evaluated_in_the_data():
    us = r.survexp_us()
    # pyears(Surv(time, status) ~ grp, d, ratetable = survexp.us,
    #        rmap = list(age = ageyr * 365.25, sex = sex, year = year), scale = 1)
    rmap = {"age": "ageyr * 365.25", "sex": "sex", "year": "year"}
    result = r.pyears("Surv(time, status) ~ grp", _cohort(), ratetable=us, rmap=rmap, scale=1)
    assert result.expected == approx([0.058742502586283094, 0.065771124924939944])
    # the same rmap in survexp(~ 1, d, ratetable = survexp.us, times = c(0, 182.5, 365))
    expected = r.survexp("~ 1", _cohort(), ratetable=us, rmap=rmap, times=[0, 182.5, 365])
    assert expected.surv == approx([1.0, 0.98650866680065386, 0.97325663181952982])
    # a missing ageyr drops the row, as the variable is in R's model frame
    data = {**_cohort(), "ageyr": [60, 70, None, 80]}
    expected = r.survexp("~ 1", data, ratetable=us, rmap=rmap, times=[0, 182.5, 365])
    assert expected.surv == approx([1.0, 0.98552633365506292, 0.97133437877078055])
    assert expected.n == 3
    # rmap = list(age = year - birth, ...): date differences count days
    rmap = {"age": "year - birth", "sex": "sex", "year": "year"}
    expected = r.survexp("~ 1", _cohort(), ratetable=us, rmap=rmap, times=[0, 182.5, 365])
    assert expected.surv == approx([1.0, 0.98650888042295504, 0.97325652219744718])
    individual = r.survexp(
        "time ~ 1",
        _cohort(),
        ratetable=us,
        rmap={**rmap, "age": "(year - birth)"},
        method="individual.s",
    )
    assert individual == approx(
        [0.99604605679416347, 0.97843785221959145, 0.94669528682588988, 0.95697739097128665]
    )
    # a string reading no column stays a constant
    usr = r.pyears(
        "Surv(time, status) ~ 1",
        {**_cohort(), "age": [60 * 365.25] * 4},
        ratetable=r.survexp_usr(),
        rmap={"race": "white", "sex": "sex", "year": "year"},
        scale=1,
    )
    assert usr.expected > 0


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


def test_as_numeric_of_a_factor_inside_an_expression_is_its_codes():
    lung = datasets.load_lung()
    # d$f <- factor(ifelse(sex == 1, 10, 20)); d$e <- factor(ph.ecog)
    data = {
        **lung,
        "f": r._r_factor([10 if sex == 1 else 20 for sex in lung["sex"]], [10, 20]),
        "e": r._r_factor(lung["ph.ecog"], [0, 1, 2, 3]),
    }
    # coxph(Surv(time, status) ~ I(as.numeric(f) + 0), d), and so on
    for rhs, names, coefficient, n in [
        ("I(as.numeric(f) + 0)", ["I(as.numeric(f) + 0)"], -0.53102353761950816, 228),
        ("log(as.numeric(e))", ["log(as.numeric(e))"], 0.8180812666404994, 227),
        ("I(as.numeric(e) == 2)", ["I(as.numeric(e) == 2)TRUE"], -0.037615768663394278, 227),
    ]:
        fit = r.coxph(f"Surv(time, status) ~ {rhs}", data)
        assert list(fit.coef_names) == names
        assert fit.coefficients == approx([coefficient], rel=1e-9)
        assert fit.n == n


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
