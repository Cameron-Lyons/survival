"""Aalen regression with offset and rescaled covariates, against R survival 3.8-11::

d <- data.frame(time = 1:12, status = 1, x = c(-2, 1, 0, 2, -1, -2, 2, 0, 1, -1, 2, -2))
f <- aareg(Surv(time, status) ~ I(x + off), d)            # off = 0, 1e8
length(f$times); f$test.statistic; sum(f$coefficient[, 2])
l <- lung; l$agem <- l$age * 1e-6
f <- aareg(Surv(time, status) ~ agem + sex, l)
length(f$times); colSums(f$coefficient); f$test.statistic
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
