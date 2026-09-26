"""Regression tests for the population routines (``pyears``, ``summary.pyears``, ``survexp``,
``cipoisson``) against R 4.5.3 / survival 3.8-12.
"""

from __future__ import annotations

import datetime
import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
population = survival.population
datasets = survival.datasets

NAN = math.nan


def approx(values, rel=1e-12):
    return pytest.approx(values, rel=rel, abs=1e-15, nan_ok=True)


def pairs(values, rel=1e-12):
    return [approx(pair, rel) for pair in values]


def _lung_entries():
    lung = datasets.load_lung()
    start = datetime.date(1985, 3, 1)
    return {
        "time": lung["time"],
        "status": lung["status"],
        "sex": lung["sex"],
        "age": [value * 365.25 for value in lung["age"]],
        "entry": [start + datetime.timedelta(days=29 * (i + 1)) for i in range(len(lung["time"]))],
    }


@pytest.mark.parametrize(
    ("method", "male", "female"),
    [
        (
            "hakulinen",
            [
                1.0,
                0.98811931793274022,
                0.97607310564936267,
                0.95300140444874726,
                0.93099104195715754,
            ],
            [1.0, 0.99398818878228878, 0.98767498249199237, 0.97415612281428432, 0.0],
        ),
        (
            "conditional",
            [
                1.0,
                0.98808479216144518,
                0.97592930641106446,
                0.95254022326552523,
                0.92981233627325666,
            ],
            [
                1.0,
                0.99397833237309152,
                0.98763459340096982,
                0.97400798864542881,
                0.96396602085868466,
            ],
        ),
    ],
)
def test_survexp_cohort_methods_follow_each_subject_across_rate_cells(method, male, female):
    # lung2$entry <- as.Date("1985-03-01") + (1:228) * 29
    # survexp(Surv(time, status) ~ sex, lung2, rmap = list(age = age * 365.25, sex = sex,
    #         year = entry), method = method, times = c(0, 182, 365, 730, 1000))
    result = r.survexp(
        "Surv(time, status) ~ sex",
        _lung_entries(),
        rmap={"age": "age", "sex": "sex", "year": "entry"},
        method=method,
        times=[0, 182, 365, 730, 1000],
    )
    assert [row[0] for row in result.surv] == approx(male, rel=1e-13)
    assert [row[1] for row in result.surv] == approx(female, rel=1e-13)
    assert result.n_risk == [[138, 90], [86, 71], [35, 30], [7, 6], [2, 0]]
