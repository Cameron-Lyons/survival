"""Matrix basis terms ``ns()``, ``bs()``, ``poly()``, ``nsk()`` and ``scale()`` against
R survival 3.8.11 with splines.

New data reuse the fitted knots, boundary and polynomial coefficients, as R's
``makepredictcall`` methods arrange.
"""

import math
from contextlib import nullcontext

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets
core = survival.core

NAN = math.nan

COXPH_CASES = [
    (
        "ns(age, df=3)",
        ["ns(age, df = 3)1", "ns(age, df = 3)2", "ns(age, df = 3)3"],
        [0.158861626982333, 2.00117918619346, 1.25079392229487],
        228,
    ),
    (
        "ns(age,knots=c(55,65))",
        [f"ns(age, knots = c(55, 65)){j}" for j in (1, 2, 3)],
        [0.144299426824115, 2.33878640953123, 1.24360786971431],
        228,
    ),
    (
        "ns(age, df=4, intercept=TRUE)",
        [f"ns(age, df = 4, intercept = TRUE){j}" for j in (1, 2, 3, 4)],
        [-4.53686623567848, -3.32891426800605, -14.1628278873915, NAN],
        228,
    ),
    (
        "ns(wt.loss, 2)",
        ["ns(wt.loss, 2)1", "ns(wt.loss, 2)2"],
        [0.377566766030465, -0.0127361806429885],
        214,
    ),
    (
        "bs(age, df=4)",
        [f"bs(age, df = 4){j}" for j in (1, 2, 3, 4)],
        [1.64717794779812, 0.918785730321236, 1.40158784819732, 2.51700504939713],
        228,
    ),
    (
        "bs(age, degree=2, knots=60)",
        [f"bs(age, degree = 2, knots = 60){j}" for j in (1, 2, 3)],
        [1.22993971097563, 0.64948826909808, 2.07454585048694],
        228,
    ),
    (
        "bs(age, df=3, Boundary.knots=c(45, 75))",
        [f"bs(age, df = 3, Boundary.knots = c(45, 75)){j}" for j in (1, 2, 3)],
        [0.879324137489014, -0.169315064110481, 0.71818495078976],
        228,
    ),
    (
        "poly(age, 2)",
        ["poly(age, 2)1", "poly(age, 2)2"],
        [2.50652296458045, 0.617426045814411],
        228,
    ),
    (
        "poly(age,3,raw=TRUE)",
        [f"poly(age, 3, raw = TRUE){j}" for j in (1, 2, 3)],
        [1.53574449048284, -0.02574705724157, 0.000143114909466234],
        228,
    ),
    (
        "nsk(age, df=3)",
        ["nsk(age, df = 3)1", "nsk(age, df = 3)2", "nsk(age, df = 3)3"],
        [0.357266700230909, 0.297699635030749, 0.73361634491979],
        228,
    ),
    ("scale(age)", ["scale(age)"], [0.169856733059056], 228),
    (
        "ns(age, df = 3) + sex",
        ["ns(age, df = 3)1", "ns(age, df = 3)2", "ns(age, df = 3)3", "sex"],
        [0.128502203899397, 1.71818845516319, 1.12368192646775, -0.503733839103796],
        228,
    ),
]


@pytest.fixture(scope="module")
def lung():
    return datasets.load_lung()


@pytest.mark.parametrize(("rhs", "names", "coefficients", "n"), COXPH_CASES)
def test_coxph_basis_terms_match_r(lung, rhs, names, coefficients, n):
    beyond = pytest.warns(UserWarning, match="beyond boundary knots")
    with beyond if "Boundary" in rhs else nullcontext():
        fit = r.coxph(f"Surv(time, status) ~ {rhs}", lung)
    assert list(fit.coefficient_names) == names
    assert fit.n == n
    for actual, expected in zip(r.coef(fit), coefficients, strict=True):
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-8)


def test_new_data_reuse_the_fitted_basis(lung):
    new = {
        "age": [40.0, 60.0, 85.0],
        "wt.loss": [0.0, 10.0, 30.0],
        "meal.cal": [500.0, 1000.0, 2800.0],
    }
    fit = r.coxph(
        "Surv(time, status) ~ ns(age, df=3) + poly(wt.loss, 2, raw=TRUE) + bs(meal.cal, df=3)",
        lung,
    )
    with pytest.warns(UserWarning, match="beyond boundary knots"):
        lp = r.predict(fit, newdata=new, type="lp")
    assert lp == pytest.approx(
        [-0.745574558163275, -0.043606896653657, 0.942604574123655], rel=1e-9
    )
    fit = r.coxph("Surv(time, status) ~ poly(age, 2) + nsk(wt.loss, df=2)", lung)
    assert r.predict(fit, newdata=new, type="lp") == pytest.approx(
        [-0.243781756579702, -0.107055420399052, 0.845219428082601], rel=1e-9
    )
    fit = r.coxph("Surv(time, status) ~ ns(age, df=3) + sex", lung)
    terms = r.predict(fit, newdata={"age": [40.0, 60.0], "sex": [1, 2]}, type="terms")
    np.testing.assert_allclose(
        np.asarray(terms),
        [[-0.538469466441551, 0.198842304909393], [-0.017880681412357, -0.304891534194403]],
        rtol=1e-9,
    )


def test_survreg_takes_basis_terms(lung):
    fit = r.survreg("Surv(time, status) ~ ns(age, df=3) + sex", lung)
    assert list(fit.coefficients) == pytest.approx(
        [
            5.94512210502264,
            -0.0898143620090142,
            -1.25679681225063,
            -0.815529433527373,
            0.374557334999283,
        ],
        rel=1e-8,
    )
    assert fit.loglik == pytest.approx([-1153.85118808941, -1146.14807916446], rel=1e-10)


def test_native_bases_match_r():
    basis = core.ns_basis([1.0, 2.0, NAN, 4.0, 7.0, 8.0, 10.0], df=3)
    assert basis.knots == pytest.approx([3.333333333333333, 7.333333333333333])
    assert basis.boundary_knots == (1.0, 10.0)
    matrix = np.asarray(basis.basis).reshape(basis.n_rows, basis.n_cols)
    assert np.isnan(matrix[2]).all()
    assert matrix[1] == pytest.approx(
        [-0.065208376439665067, 0.25454510701251193, -0.18181793358036566]
    )
    basis = core.bs_basis([0.0, 2.5, 6.0], knots=[2.0], boundary_knots=(1.0, 4.0))
    matrix = np.asarray(basis.basis).reshape(3, 4)
    np.testing.assert_allclose(
        matrix,
        [
            [-8.4444444444444429, 1.5555555555555554, -0.1111111111111111, 0.0],
            [0.1875, 0.46875, 0.328125, 0.015625],
            [-0.44444444444444442, 3.5555555555555554, -10.111111111111111, 8.0],
        ],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("rhs", "message"),
    [
        ("ns(log(age), 3)", "requires a data column"),
        ("I(ns(age, 3))", "matrix term"),
        ("poly(meal.cal, 2)", "missing values are not allowed in 'poly'"),
        ("poly(age, 2, coefs = 1)", "coefs"),
    ],
)
def test_unsupported_basis_uses_raise(lung, rhs, message):
    with pytest.raises(ValueError, match=message):
        r.coxph(f"Surv(time, status) ~ {rhs}", lung)
