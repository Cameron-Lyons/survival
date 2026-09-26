"""Regression tests of ``brier``, ``survcheck``, ``survobrien`` and ``cch`` against R survival
3.8-12: the curves brier reads, survcheck's row numbers after ``na.omit``, survobrien's
``I()`` terms and the id order of cch's Borgan score residuals."""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
datasets = survival.datasets


def approx(values, rel=1e-8):
    return pytest.approx(values, rel=rel, abs=1e-12)


def assert_rows_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == approx(expected_row)


# --- brier -------------------------------------------------------------------


def test_brier_reads_the_model_curves_at_the_evaluation_times():
    # R: brier() of coxph(Surv(time, status) ~ age + sex, lung) at times 3, 180 and 365 with
    # detail = TRUE, and at its default times
    fit = r.coxph("Surv(time, status) ~ age + sex", data=datasets.load_lung(), model=True)
    result = r.brier(fit, times=[3, 180, 365], detail=True)
    assert result.brier == approx([0.0, 0.191605092010014, 0.236142139448665])
    assert math.isnan(result.rsquared[0])
    assert result.rsquared[1:] == approx([0.0460864863000336, 0.0232491313563413])
    # every curve is still 1 before the first event time
    assert result.phat[0] == [0.0] * 228
    assert result.phat[1][:4] == approx(
        [0.37766021846661, 0.348294500488361, 0.294579029600006, 0.29879827416173]
    )
    assert [result.phat[2][i] for i in (0, 1, 227)] == approx(
        [0.731243146155405, 0.694623967597056, 0.450503170391115]
    )

    default = r.brier(fit)
    assert len(default.times) == 139
    assert sum(default.brier) == approx(23.8320240782286)
    assert default.brier[-1] == approx(0.0513206230245228)
