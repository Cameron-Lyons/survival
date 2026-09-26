"""The survreg kernel and its name matching against R 4.5.3 / survival 3.8-12."""

from __future__ import annotations

import math

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


def test_rsurvreg_seed_reproduces_r_set_seed():
    # R's set.seed(1) followed by rsurvreg(3, 0, 1)
    assert r.rsurvreg(3, 0, 1, seed=1) == pytest.approx(
        [0.30857707804919837, 0.46541242439391811, 0.85062791335182275], rel=1e-14
    )
    # set.seed(42) followed by rsurvreg(4, 1:4, 0.5, "lognormal")
    assert r.rsurvreg(4, [1, 2, 3, 4], 0.5, "lognormal", seed=42) == pytest.approx(
        [5.3950357131052664, 15.884418122831969, 15.144704122077824, 88.055546326660618],
        rel=1e-14,
    )
    # set.seed(-7) followed by rsurvreg(3, 1, 2, "t", parms = 5): R's seeds are signed
    assert r.rsurvreg(3, 1, 2, "t", parms=5, seed=-7) == pytest.approx(
        [0.044519774037484416, 1.2104777027768425, -0.419870560940246], rel=1e-14
    )


def test_rsurvreg_rejects_the_seed_r_reads_as_na():
    # R's set.seed(-2147483648) stops: -2^31 is NA_integer_
    with pytest.raises(ValueError, match="supplied seed is not a valid integer"):
        r.rsurvreg(3, 0, 1, seed=-(2**31))


def test_dpqr_distribution_names_are_case_folded_but_not_partially_matched():
    # R's dsurvreg(c(0.5, 2), 0.2, 1.5, "Weibull"), psurvreg(c(0.5, 2), 0.2, 1.5, "LogNormal")
    # and qsurvreg(c(0.25, 0.9), 1, 0.5, "LOGLOGISTIC")
    assert r.dsurvreg([0.5, 2], 0.2, 1.5, "Weibull") == pytest.approx(
        [0.42355410140929078, 0.11542912786314491], rel=1e-14
    )
    assert r.psurvreg([0.5, 2], 0.2, 1.5, "LogNormal") == pytest.approx(
        [0.27577755285638794, 0.62883325964355896], rel=1e-14
    )
    assert r.qsurvreg([0.25, 0.9], 1, 0.5, "LOGLOGISTIC") == pytest.approx(
        [1.5694007453940979, 8.1548454853771375], rel=1e-14
    )
    # dsurvreg(1, 0, 1, "weib"): survreg.distributions[["weib"]] is NULL
    for helper in (r.dsurvreg, r.psurvreg, r.qsurvreg):
        with pytest.raises(ValueError, match="Distribution not found"):
            helper([0.5], 0, 1, "weib")
    with pytest.raises(ValueError, match="Distribution not found"):
        r.rsurvreg(2, 0, 1, "exp", seed=1)


def test_t_fits_are_unchanged(lung, t_fit):
    # R's survreg(Surv(time, status) ~ age + sex, lung, dist = "t") to all the digits this
    # port computes: taking both t tails from one pt() call must not move the fit.
    assert t_fit.fit.coefficients == pytest.approx(
        [307.34358133147515, -2.4707923357292465, 128.58458914302994, 5.279364620574666],
        rel=1e-12,
    )
    assert t_fit.fit.log_likelihood == pytest.approx(-1179.86538713346, rel=1e-12)

    # survreg(Surv(log(time), status) ~ age + ph.ecog + strata(sex), lung, dist = "t",
    #         parms = 8)
    logged = dict(lung, ltime=[math.log(time) for time in lung["time"]])
    strata = r.survreg(
        "Surv(ltime, status) ~ age + ph.ecog + strata(sex)",
        data=logged,
        na_action="omit",
        dist="t",
        parms=8,
    )
    assert strata.fit.coefficients == pytest.approx(
        [
            6.743464277529713,
            -0.010257032556250977,
            -0.38265929760663714,
            -0.10147952338058526,
            -0.2598049628238328,
        ],
        rel=1e-12,
    )
    assert strata.fit.log_likelihood == pytest.approx(-273.81113429967894, rel=1e-12)


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


def test_survreg_distribution_names_follow_match_arg():
    assert core.SurvregDistribution("exp").name == "Exponential"
    assert core.SurvregDistribution("logn").name == "Log Normal"
    # survreg(..., dist = "Weibull") and dist = "log" (ambiguous) stop() in R
    for bad in ("Weibull", "log", "extreme_value"):
        with pytest.raises(ValueError, match="'arg' should be one of \"extreme\""):
            core.SurvregDistribution(bad)
