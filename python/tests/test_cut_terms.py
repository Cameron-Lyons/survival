"""Bare ``cut()`` terms against R survival 3.8-11.

The reference values come from::

    library(survival)
    survfit(Surv(time, status) ~ cut(age, c(30, 60, 70, 90)), lung)
    survdiff(Surv(time, status) ~ cut(age, c(30, 60, 70, 90)), lung)$chisq
    survdiff(Surv(time, status) ~ cut(age, 3), lung)$chisq
    survdiff(Surv(time, status) ~ cut(age, c(30, 60, 70, 90), right = FALSE), lung)$chisq
    survdiff(Surv(time, status) ~ cut(age, 3), lung, subset = sex == 1)
    fit <- coxph(Surv(time, status) ~ cut(age, c(0, 55, 65, 100)) + sex, lung)
    coef(fit); predict(fit, data.frame(age = c(50, 60, 80), sex = c(1, 2, 1)))
    survexp(~ cut(age, c(30, 60, 70, 90)), data = d, ratetable = survexp.us,
            rmap = list(age = age * 365.25, sex = c("male", "female")[sex],
                        year = as.Date("2000-01-01")), times = c(100, 500))
"""

import datetime

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r

BREAKS = "cut(age, c(30, 60, 70, 90))"


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


def assert_close(actual, expected, rtol=1e-8):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float).ravel(), np.asarray(expected, dtype=float), rtol=rtol
    )


def test_survfit_groups_by_the_cut_intervals(lung):
    fit = r.survfit(f"Surv(time, status) ~ {BREAKS}", lung)
    assert list(fit.strata) == [f"{BREAKS}=(30,60]", f"{BREAKS}=(60,70]", f"{BREAKS}=(70,90]"]
    assert list(fit.n) == [94, 88, 46]


@pytest.mark.parametrize(
    ("term", "chisq"),
    [
        (BREAKS, 4.92894521338123),
        ("cut(age, 3)", 1.9023268771676),
        ("cut(age, c(30, 60, 70, 90), right = FALSE)", 4.37743327449445),
    ],
)
def test_survdiff_tests_the_cut_intervals(lung, term, chisq):
    result = r.survdiff(f"Surv(time, status) ~ {term}", lung)
    assert len(result.n) == 3
    assert_close(result.chisq, [chisq])


def test_cut_breaks_are_computed_before_subsetting(lung):
    result = r.survdiff(
        "Surv(time, status) ~ cut(age, 3)", lung, subset=[sex == 1 for sex in lung["sex"]]
    )
    assert list(result.n) == [20, 67, 51]
    assert_close(result.chisq, [1.85339294131282])


def test_coxph_fits_and_predicts_a_cut_covariate(lung):
    fit = r.coxph("Surv(time, status) ~ cut(age, c(0, 55, 65, 100)) + sex", lung)
    assert r.coef_names(fit) == [
        "cut(age, c(0, 55, 65, 100))(55,65]",
        "cut(age, c(0, 55, 65, 100))(65,100]",
        "sex",
    ]
    assert_close(r.coef(fit), [-0.0696091638071333, 0.227122459289241, -0.541176185281713], 1e-6)
    predicted = r.predict(fit, newdata={"age": [50.0, 60.0, 80.0], "sex": [1.0, 2.0, 1.0]})
    assert_close(predicted, [0.213622178400676, -0.39716317068817, 0.440744637689917], 1e-6)


def test_survexp_groups_by_the_cut_intervals(lung):
    data = dict(lung)
    data["agedays"] = [age * 365.25 for age in lung["age"]]
    data["sexc"] = ["male" if sex == 1 else "female" for sex in lung["sex"]]
    data["entry"] = [datetime.date(2000, 1, 1)] * len(lung["age"])
    expected = r.survexp(
        f"~ {BREAKS}",
        data=data,
        ratetable=r.survexp_us(),
        rmap={"age": "agedays", "sex": "sexc", "year": "entry"},
        times=[100, 500],
    )
    assert_close(
        expected.surv,
        [0.998158022710804, 0.994679862137917, 0.988553361476179]
        + [0.990655435211166, 0.973209018247732, 0.942916646103537],
    )


@pytest.mark.parametrize("function", ["survfit", "survdiff"])
def test_other_call_terms_are_refused_rather_than_read_as_columns(lung, function):
    with pytest.raises(ValueError, match="unsupported formula term"):
        getattr(r, function)("Surv(time, status) ~ tcut(age, c(0, 60, 100))", lung)
