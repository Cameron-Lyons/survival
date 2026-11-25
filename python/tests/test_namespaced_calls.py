"""Namespaced calls in formulas: ``splines::ns(age, df = 2)``, ``stats::poly(age, 2)``,
``survival::nsk(age, df = 3)``, ``base::log(age)``.

R's model frame evaluates them as the unqualified function and keeps the call as written
in the term and coefficient names (``splines::ns(age, df = 2)1``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
expression = __import__("survival.r._expression", fromlist=["_expression_label"])


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


PAIRS = [
    ("splines::ns(age, df = 2)", "ns(age, df = 2)"),
    ("splines::bs(age, df = 3)", "bs(age, df = 3)"),
    ("stats::poly(age, 2)", "poly(age, 2)"),
    ("survival::nsk(age, df = 3)", "nsk(age, df = 3)"),
    ("base::log(age)", "log(age)"),
    ("splines:::ns(age, df = 2)", "ns(age, df = 2)"),
]


def _prefix(namespaced, plain):
    return namespaced[: len(namespaced) - len(plain)]


@pytest.mark.parametrize(("namespaced", "plain"), PAIRS)
def test_namespaced_calls_fit_as_the_function_and_keep_their_names(lung, namespaced, plain):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = r.coxph(f"Surv(time, status) ~ {namespaced} + sex", lung)
        expected = r.coxph(f"Surv(time, status) ~ {plain} + sex", lung)
    assert list(actual.coefficients) == list(expected.coefficients)
    prefix = _prefix(namespaced, plain)
    assert list(r.coef_names(actual)) == [
        name if name == "sex" else prefix + name for name in r.coef_names(expected)
    ]
    newdata = {"age": [45.0, 60.0, 75.0], "sex": [1, 2, 1]}
    np.testing.assert_array_equal(
        np.asarray(r.predict(actual, newdata=newdata)),
        np.asarray(r.predict(expected, newdata=newdata)),
    )


def test_namespaced_basis_calls_in_interactions_survreg_and_curves(lung):
    actual = r.coxph("Surv(time, status) ~ splines::ns(age, df = 2):sex", lung)
    expected = r.coxph("Surv(time, status) ~ ns(age, df = 2):sex", lung)
    assert list(actual.coefficients) == list(expected.coefficients)
    assert list(r.coef_names(actual)) == [
        "splines::ns(age, df = 2)1:sex",
        "splines::ns(age, df = 2)2:sex",
    ]
    model = r.survreg("Surv(time, status) ~ splines::ns(age, df = 2) + sex", lung)
    plain = r.survreg("Surv(time, status) ~ ns(age, df = 2) + sex", lung)
    assert list(model.coefficients) == list(plain.coefficients)


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("splines::ns(age,df=2)", "splines::ns(age, df = 2)"),
        ("stats :: poly(age, 2)", "stats::poly(age, 2)"),
        ("base:::log( age )", "base:::log(age)"),
    ],
)
def test_namespaced_calls_deparse_as_r_does(text, label):
    assert expression._expression_label(text) == label


def test_unknown_packages_and_quoted_text_are_left_alone(lung):
    with pytest.raises(ValueError, match="unsupported formula term"):
        r.coxph("Surv(time, status) ~ notapkg::ns(age, df = 2)", lung)
    # Namespace colons are part of a name; a single colon is still an interaction.
    split = __import__("survival.r._formula", fromlist=["_split_top_level"])._split_top_level
    assert split("splines::ns(age, df = 2):sex", ":") == ["splines::ns(age, df = 2)", "sex"]
    assert split("base:::log(age):x", ":") == ["base:::log(age)", "x"]
    assert split("f('a::b'):x", ":") == ["f('a::b')", "x"]
    fit = r.coxph("Surv(time, status) ~ age:sex", lung)
    assert r.coef_names(fit)[0] == "age:sex"
