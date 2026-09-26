"""The survreg kernel and its name matching against R 4.5.3 / survival 3.8-12."""

from __future__ import annotations

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
core = survival._survival


@pytest.fixture(scope="module")
def lung():
    return survival.datasets.load_lung()


@pytest.fixture(scope="module")
def t_fit(lung):
    return r.survreg("Surv(time, status) ~ age + sex", data=lung, na_action="omit", dist="t")


def test_predict_and_residual_types_follow_match_arg(t_fit):
    fit = t_fit.fit
    # predict(fit, type = "line") is "linear"; "l" and "lin" also prefix "link" and "lp",
    # and match.arg is case sensitive.
    linear = fit.predict(predict_type="line")
    assert [row[0] for row in linear.fit[:2]] == pytest.approx(
        [253.08953763054083, 267.91429164491637], rel=1e-12
    )
    for bad in ("l", "lin", "Response"):
        with pytest.raises(ValueError, match='\'arg\' should be one of "response", "link"'):
            fit.predict(predict_type=bad)
    # residuals(fit, type = "dfbeta")[1, ]: an exact name wins over the longer "dfbetas"
    dfbeta = fit.residuals(residual_type="dfbeta")
    assert dfbeta.values[0] == pytest.approx(
        [
            -3.1340716920233254,
            0.066781296701935403,
            -0.65312877180374385,
            -0.0041293510240299489,
        ],
        rel=1e-12,
    )
    with pytest.raises(ValueError, match='\'arg\' should be one of "response", "deviance"'):
        fit.residuals(residual_type="Matrix")
