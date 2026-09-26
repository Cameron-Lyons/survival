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


# --- survobrien --------------------------------------------------------------


def test_survobrien_leaves_asis_terms_alone():
    # R's ?survobrien example: survobrien(Surv(futime, fustat) ~ age + factor(rx) + I(ecog.ps),
    # data = ovarian)
    frame = r.survobrien(
        "Surv(futime, fustat) ~ age + factor(rx) + I(ecog.ps)", data=datasets.load_ovarian()
    )
    assert list(frame) == ["time", "status", "rx", "ecog.ps", ".id.", "age", ".strata."]
    assert len(frame["time"]) == 230
    assert frame["rx"][:8] == [1, 1, 1, 2, 1, 1, 2, 2]
    assert frame["ecog.ps"][:8] == [1, 1, 2, 1, 1, 2, 2, 2]
    assert frame["ecog.ps"][-8:] == [1, 1, 2, 2, 1, 1, 1, 2]
    assert frame[".id."][:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert frame["age"][:4] == approx(
        [2.2407096892759584, 2.7932080094425165, 1.8607523407150068, -0.72213471743319757]
    )
    assert frame[".strata."][-3:] == [12, 12, 12]

    data = {
        "time": [1, 2, 3, 4, 5],
        "status": [1, 0, 1, 1, 1],
        "x": [0.1, 0.4, 0.2, 0.8, 0.5],
        "z": [2, 7, 1, 3, 5],
        "w": [1, 2, 1, 2, 2],
    }
    x_logits = [
        -2.19722457733621912,
        0.0,
        -0.84729786038720356,
        2.19722457733621956,
        0.84729786038720345,
        -1.6094379124341005,
        1.60943791243410073,
        0.0,
        1.09861228866810978,
        -1.09861228866810978,
        0.0,
    ]
    asis = r.survobrien("Surv(time, status) ~ x + I(z)", data=data)
    assert list(asis) == ["time", "status", "z", ".id.", "x", ".strata."]
    assert asis["z"] == [2, 7, 1, 3, 5, 1, 3, 5, 3, 5, 5]
    assert asis["x"] == approx(x_logits)

    # an I() expression keeps the variables it references (R: all.vars)
    expression = r.survobrien("Surv(time, status) ~ x + I(z^2) + I(z * w)", data=data)
    assert list(expression) == ["time", "status", "z", "w", ".id.", "x", ".strata."]
    assert expression["z"] == asis["z"]
    assert expression["w"] == [1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2]
    assert expression["x"] == approx(x_logits)

    # identity() does not protect a term
    identity = r.survobrien("Surv(time, status) ~ x + identity(z)", data=data)
    assert identity["identity(z)"] == approx(
        [
            -0.84729786038720356,
            2.19722457733621956,
            -2.19722457733621912,
            0.0,
            0.84729786038720345,
            -1.6094379124341005,
            0.0,
            1.60943791243410073,
            -1.09861228866810978,
            1.09861228866810978,
            0.0,
        ]
    )
    with pytest.raises(ValueError, match="No continuous variables to modify"):
        r.survobrien("Surv(time, status) ~ I(z) + factor(w)", data=data)
