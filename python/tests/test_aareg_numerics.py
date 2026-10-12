"""Aalen regression with offset and rescaled covariates, against R survival 3.8-11::

d <- data.frame(time = 1:12, status = 1, x = c(-2, 1, 0, 2, -1, -2, 2, 0, 1, -1, 2, -2))
f <- aareg(Surv(time, status) ~ I(x + off), d)            # off = 0, 1e8
length(f$times); f$test.statistic; sum(f$coefficient[, 2])
l <- lung; l$agem <- l$age * 1e-6
f <- aareg(Surv(time, status) ~ agem + sex, l)
length(f$times); colSums(f$coefficient); f$test.statistic
f <- aareg(Surv(start, stop, event) ~ age + transplant, heart)   # and ~ transplant
f$n; f$test.statistic; summary(f)$table; summary(f)$chisq
"""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r

X = [-2, 1, 0, 2, -1, -2, 2, 0, 1, -1, 2, -2]


@pytest.mark.parametrize(
    ("offset", "statistic", "slope_sum"),
    [
        (0.0, [9.38025316915, -0.267929292929], 0.0319524688762),
        (1e8, [2.67929536695e-09, -0.267929285765], 0.031952468996),
    ],
)
def test_a_large_covariate_offset_keeps_r_precision(offset, statistic, slope_sum):
    data = {"time": list(range(1, 13)), "status": [1] * 12, "x": [v + offset for v in X]}
    fit = r.aareg("Surv(time, status) ~ x", data)
    assert len(fit.times) == 10
    assert list(fit.test_statistic) == pytest.approx(statistic, rel=1e-8)
    assert np.asarray(fit.coefficient)[:, 1].sum() == pytest.approx(slope_sum, rel=1e-8)


def test_a_rescaled_covariate_is_not_rank_deficient():
    lung = survival.datasets.load_lung()
    data = dict(lung)
    data["agem"] = [age * 1e-6 for age in lung["age"]]
    fit = r.aareg("Surv(time, status) ~ agem + sex", data)
    assert len(fit.times) == 164
    assert np.asarray(fit.coefficient).sum(axis=0) == pytest.approx(
        [0.331079734681, 47665.8013844, -0.480715823961], rel=1e-8
    )
    assert list(fit.test_statistic) == pytest.approx(
        [1.79065699184, 0.000201389125405, -19.3748958566], rel=1e-8
    )


@pytest.mark.parametrize(
    ("rhs", "n", "statistic", "coef", "chisq"),
    [
        (
            "age + transplant",
            [172, 62, 62],
            [30.123555, 179.853602, -0.288259],
            [0.0133854116423, 0.0004198857653, -0.0004186516881],
            5.264424681,
        ),
        (
            "transplant",
            [172, 61, 62],
            [30.0, 1.237516],
            [0.012014417301, 0.001725178811],
            0.1320429981,
        ),
    ],
)
def test_singular_early_risk_sets_of_counting_data_match_r(rhs, n, statistic, coef, chisq):
    from survival.r._aareg import summary_aareg

    heart = survival.datasets.load_heart()
    fit = r.aareg(f"Surv(start, stop, event) ~ {rhs}", heart)
    assert list(fit.n) == n
    assert list(fit.test_statistic) == pytest.approx(statistic, rel=1e-6)
    assert np.asarray(fit.coefficient)[0, -1] == 0.0
    summary = summary_aareg(fit)
    assert [row["coef"] for row in summary["table"]] == pytest.approx(coef, rel=1e-9)
    assert summary["chisq"] == pytest.approx(chisq, rel=1e-9)


def test_aliased_covariates_take_no_step_at_singular_interior_times():
    data = {
        "start": [0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 5],
        "stop": [2, 3, 8, 9, 11, 13, 17, 19, 20, 21, 15, 16],
        "event": [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        "x1": [1.5, 2, 0.3, 1.1, 2.2, 0.7, 1.9, 0.4, 1.3, 2.5, 0.9, 1.6],
        "x2": [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    }
    data = {key: [float(value) for value in values] for key, values in data.items()}
    fit = r.aareg("Surv(start, stop, event) ~ x1 + x2", data, nmin=4)
    assert list(fit.times) == [2.0, 3.0, 8.0, 11.0, 13.0, 16.0, 17.0]
    coefficient = np.asarray(fit.coefficient)
    assert list(coefficient[:2, 2]) == [0.0, 0.0]
    assert coefficient[:2, 1] == pytest.approx([0.0893743793445879, 0.285714285714286], rel=1e-12)
    assert list(fit.test_statistic) == pytest.approx(
        [1.037117404311659, 1.232142857142858, -0.735729613128925], rel=1e-12
    )
