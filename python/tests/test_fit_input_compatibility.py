"""R control objects and scalar starts at the public fitting boundary."""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _data():
    return {
        "start": [0, 0, 1, 0, 2, 1],
        "time": [1, 2, 3, 4, 5, 6],
        "status": [1, 1, 0, 1, 0, 1],
        "x": [0.2, 0.4, 0.1, 0.8, 1.0, 1.2],
    }


@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("allowed", ["gap", ["gap", "overlap"], [], None, "unknown"])
def test_coxph_accepts_survival_control_for_ordinary_responses(counting, allowed):
    # R survival 3.8.11 coxph.control returns survcheckallow unchanged and
    # ordinary coxph fits do not consult it, including counting responses.
    formula = "Surv(start, time, status) ~ x" if counting else "Surv(time, status) ~ x"
    control = {
        "eps": 1e-9,
        "toler.chol": np.finfo(float).eps ** 0.75,
        "iter.max": 0,
        "toler.inf": math.sqrt(1e-9),
        "outer.max": 10,
        "timefix": True,
        "survcheckallow": allowed,
    }
    fit = survival.coxph(formula, _data(), init=[0.25], control=control)
    # R: coxph(formula, d, init=.25, control=coxph.control(iter.max=0)).
    expected_loglik, expected_variance = (
        (-3.7646175836431368, 3.7998535445072692)
        if counting
        else (-4.7400191511717029, 2.8617182212121146)
    )
    assert survival.coef(fit) == pytest.approx([0.25], abs=1e-14)
    assert fit.log_likelihood == pytest.approx([expected_loglik] * 2, rel=2e-14)
    assert survival.vcov(fit)[0] == pytest.approx([expected_variance], rel=2e-14)
    assert control["survcheckallow"] is allowed


@pytest.mark.parametrize("alias", ["init", "initial_beta"])
@pytest.mark.parametrize("value", [0.25, np.float64(0.25), np.array(0.25), [0.25]])
def test_coxph_scalar_initial_value_matches_r(alias, value):
    fit = survival.coxph("Surv(time, status) ~ x", _data(), max_iter=0, **{alias: value})
    assert survival.coef(fit) == pytest.approx([0.25], abs=1e-14)
    assert fit.log_likelihood == pytest.approx([-4.7400191511717029] * 2, rel=2e-14)
    assert survival.vcov(fit)[0] == pytest.approx([2.8617182212121146], rel=2e-14)


@pytest.mark.parametrize("alias", ["init", "initial", "initial_beta"])
@pytest.mark.parametrize("matrix", [False, True])
def test_survreg_scalar_initial_value_matches_r(alias, matrix):
    data = _data()
    inputs = (
        {
            "time": data["time"],
            "status": data["status"],
            "covariates": [[1.0] for _ in data["time"]],
        }
        if matrix
        else {"response": "Surv(time, status) ~ 1", "data": data}
    )
    fit = survival.survreg(**inputs, dist="logistic", scale=1, max_iter=0, **{alias: 2.0})
    # R: survreg(Surv(time,status)~1,d,dist='logistic',scale=1,init=2,
    #            control=survreg.control(maxiter=0)).
    assert survival.coef(fit) == pytest.approx([2.0], abs=1e-14)
    assert fit.log_likelihood == pytest.approx(-13.664822653169864, rel=2e-14)
    assert survival.vcov(fit)[0] == pytest.approx([0.72446704151070251], rel=2e-14)


@pytest.mark.parametrize("function", ["coxph", "survreg"])
def test_scalar_initial_value_does_not_broadcast_to_multiple_coefficients(function):
    if function == "coxph":
        with pytest.raises(ValueError, match="initial_beta has 1 values"):
            survival.coxph("Surv(time, status) ~ x + I(x*x)", _data(), init=0.0)
    else:
        with pytest.raises(ValueError, match="initial_beta has 1 values"):
            survival.survreg("Surv(time, status) ~ x", _data(), dist="logistic", init=2.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("function", ["coxph", "survreg"])
def test_initial_value_must_be_finite(function, value):
    fit = getattr(survival, function)
    with pytest.raises(ValueError, match="init must be finite"):
        fit("Surv(time, status) ~ x", _data(), init=value)


def test_coxph_still_rejects_unknown_control_names():
    with pytest.raises(ValueError, match="unsupported option.*survcheckalow"):
        survival.coxph("Surv(time, status) ~ x", _data(), control={"survcheckalow": "gap"})
