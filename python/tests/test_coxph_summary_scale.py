"""``summary(coxph, scale =)`` with one factor per coefficient, and the Wald test of
``coef - init``; the expected values are stock R's (survival 3.8-12)."""

from __future__ import annotations

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


def test_a_scale_vector_scales_each_coefficient(lung):
    fit = r.coxph("Surv(time, status) ~ age + sex + ph.ecog", lung)
    summary = r.summary_coxph(fit, scale=[10, 1, 1])
    assert [row["coef"] for row in summary.coefficients] == pytest.approx(
        [0.1106676456, -0.5526123957, 0.4637284754], rel=1e-9
    )
    assert [row["se(coef)"] for row in summary.coefficients] == pytest.approx(
        [0.09267411014, 0.16773905379, 0.11357726616], rel=1e-9
    )
    assert summary.conf_int[1]["exp(coef)"] == pytest.approx(0.5754445562, rel=1e-9)
    # Shorter vectors recycle, as R's beta * scale does.
    recycled = r.summary_coxph(fit, scale=[10, 2])
    assert [row["coef"] for row in recycled.coefficients] == pytest.approx(
        [0.1106676456, -1.1052247914, 4.6372847537], rel=1e-9
    )


def test_robust_and_penalized_summaries_take_scale_vectors(lung):
    robust = r.coxph("Surv(time, status) ~ age + sex", lung, robust=True)
    rows = r.summary_coxph(robust, scale=[10, 1]).coefficients
    # The robust se scales; the naive se(coef) stays as fitted.
    assert rows[1]["coef"] == pytest.approx(robust.coefficients[1])
    assert rows[0]["robust se"] == pytest.approx(10 * robust.var[0][0] ** 0.5)
    assert rows[0]["se(coef)"] == pytest.approx(robust.naive_var[0][0] ** 0.5)
    penalized = r.coxph("Surv(time, status) ~ pspline(age, df = 2) + sex", lung)
    factors = [2.0] + [1.0] * (len(penalized.coefficients) - 1)
    table = r.summary_coxph(penalized, scale=factors).conf_int
    assert table[0]["exp(coef)"] == pytest.approx(
        2.718281828459045 ** (2 * penalized.coefficients[0])
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, 13.47), ({"init": [0.02, -0.3]}, 1.68), ({"init": [0.02, -0.3], "iter_max": 0}, 0.0)],
)
def test_the_wald_test_compares_coef_with_init(lung, kwargs, expected):
    fit = r.coxph("Surv(time, status) ~ age + sex", lung, **kwargs)
    assert r.summary_coxph(fit).waldtest["test"] == pytest.approx(expected, abs=1e-12)


def test_scale_validation_errors(lung):
    fit = r.coxph("Surv(time, status) ~ age + sex", lung)
    # String scale is rejected.
    with pytest.raises(TypeError, match="scale must be numeric"):
        r.summary_coxph(fit, scale="invalid")
    # Empty scale is rejected.
    with pytest.raises(ValueError, match="scale must have at least one value"):
        r.summary_coxph(fit, scale=[])
