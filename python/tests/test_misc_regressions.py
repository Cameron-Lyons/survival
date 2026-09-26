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


# --- survcheck ---------------------------------------------------------------


def test_survcheck_numbers_problem_rows_of_the_data_before_na_omit():
    # R: survcheck(Surv(t1, t2, st) ~ x, data = d, id = id), na.omit dropping the rows where x
    # is missing
    gap = r.survcheck(
        "Surv(t1, t2, st) ~ x",
        data={
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "t1": [0, 1, 0, 2, 0, 1, 0, 3],
            "t2": [1, 3, 2, 4, 2, 3, 2, 5],
            "st": [0, 1, 0, 1, 0, 1, 0, 1],
            "x": [1, None, 2, 3, None, 4, 5, 6],
        },
        id="id",
    )
    assert gap.na_action == [2, 5]
    assert (gap.flag.overlap, gap.flag.gap) == (0, 1)
    assert (gap.gap.row, gap.gap.id) == ([8], [4])
    assert gap.overlap is None

    overlap = r.survcheck(
        "Surv(t1, t2, st) ~ x",
        data={
            "id": [1, 1, 2, 2, 3, 3],
            "t1": [0, 1, 0, 1, 0, 3],
            "t2": [1, 3, 2, 4, 2, 5],
            "st": [0, 1, 0, 1, 0, 1],
            "x": [None, 1, 2, 3, None, 4],
        },
        id="id",
    )
    assert overlap.na_action == [1, 5]
    assert (overlap.flag.overlap, overlap.flag.gap) == (1, 0)
    assert (overlap.overlap.row, overlap.overlap.id) == ([4], [2])
    assert overlap.gap is None
