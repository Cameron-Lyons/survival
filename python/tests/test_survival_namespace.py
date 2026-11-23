"""``survival::`` in formulas, as R's removeDoubleColonSurv rewrites it.

R's coxph, survreg, survfit, survdiff, concordance, aareg and survcheck drop ``survival::``
from strata, cluster, pspline, tt, ridge and the frailty family before reading the terms,
so a prefixed special is the special, and the fit is the unprefixed one
(survival's tests/doublecolon.R).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
formula_module = __import__("survival.r._formula", fromlist=["_without_survival_namespace"])


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


def _same(actual, expected):
    return np.array_equal(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), equal_nan=True
    )


def _tt(x, t, *_rest):
    return np.asarray(x, dtype=float) * np.log(np.asarray(t, dtype=float))


CASES = [
    ("coxph", "Surv(time, status) ~ age + {p}strata(sex) + {p}cluster(inst)", {}),
    ("coxph", "Surv(time, status) ~ {p}pspline(age, df = 2) + {p}frailty(inst)", {}),
    ("coxph", "Surv(time, status) ~ {p}ridge(age, sex, theta = 1)", {}),
    ("coxph", "Surv(time, status) ~ age + {p}frailty.gaussian(inst)", {}),
    ("coxph", "Surv(time, status) ~ age + {p}tt(ph.karno)", {"tt": _tt}),
    ("survreg", "Surv(time, status) ~ age + {p}strata(sex)", {}),
    ("survreg", "Surv(time, status) ~ {p}pspline(age, df = 2)", {}),
]


@pytest.mark.parametrize(("function", "template", "kwargs"), CASES)
def test_prefixed_specials_fit_as_the_specials(lung, function, template, kwargs):
    fit = getattr(r, function)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = fit(template.format(p="survival::"), lung, **kwargs)
        expected = fit(template.format(p=""), lung, **kwargs)
    assert actual.coefficient_names == expected.coefficient_names
    assert _same(list(actual.coefficients), list(expected.coefficients))
    assert _same(np.ravel(actual.var), np.ravel(expected.var))


def test_curves_tests_and_concordance_read_prefixed_strata(lung):
    curves = r.survfit("Surv(time, status) ~ survival::strata(sex)", lung)
    assert list(curves.strata) == ["strata(sex)=sex=1", "strata(sex)=sex=2"]
    assert _same(curves.surv, r.survfit("Surv(time, status) ~ strata(sex)", lung).surv)
    test = r.survdiff("Surv(time, status) ~ sex + survival::strata(ph.ecog)", lung)
    assert test.chisq == r.survdiff("Surv(time, status) ~ sex + strata(ph.ecog)", lung).chisq
    actual = r.concordance("Surv(time, status) ~ age + survival::strata(sex)", lung)
    expected = r.concordance("Surv(time, status) ~ age + strata(sex)", lung)
    assert actual.concordance == expected.concordance
    additive = r.aareg("Surv(time, status) ~ age + survival::cluster(inst)", lung)
    plain = r.aareg("Surv(time, status) ~ age + cluster(inst)", lung)
    assert _same(np.ravel(additive.coefficient), np.ravel(plain.coefficient))


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("y ~ survival::strata(g) + survival :: cluster(id)", "y ~ strata(g) + cluster(id)"),
        (
            "survival::Surv(t, s) ~ survival::frailty.gamma(g)",
            "survival::Surv(t, s) ~ frailty.gamma(g)",
        ),
        ("y ~ I(f('survival::strata(')) + `survival::strata(x)`", None),
        ("y ~ mysurvival::strata(g) + survival::ns(x)", None),
        ("y ~ survival::strata", None),
    ],
)
def test_only_survival_specials_lose_the_prefix(formula, expected):
    rewritten = formula_module._without_survival_namespace(formula)
    assert rewritten == (formula if expected is None else expected)


def test_lists_of_formulas_are_rewritten_and_other_values_pass_through():
    rewrite = formula_module._without_survival_namespace
    assert rewrite(["a ~ survival::strata(g)", "1:2 ~ survival::tt(x)"]) == [
        "a ~ strata(g)",
        "1:2 ~ tt(x)",
    ]
    assert rewrite(None) is None
    fit = object()
    assert rewrite(fit) is fit
