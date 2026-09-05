"""Real-fit warning references checked against R survival 3.8.11.

Cox references use coxph(Surv(time, status) ~ x), or the counting
response Surv(start, time, status), with x = rep(0:1, each = 5),
time = 1:10, status = 1 and start = 0. The two groups are separated.
"""

import warnings

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _separated_data():
    return {
        "start": [0.0] * 10,
        "time": list(range(1, 11)),
        "status": [1] * 10,
        "x": [0.0] * 5 + [1.0] * 5,
    }


def _fit_with_messages(function, *args, **kwargs):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        result = function(*args, **kwargs)
    return result, [str(item.message) for item in captured]


def _cox_fit(counting, method, **kwargs):
    response = "Surv(start, time, status)" if counting else "Surv(time, status)"
    return _fit_with_messages(
        survival.coxph,
        f"{response} ~ x",
        data=_separated_data(),
        method=method,
        **kwargs,
    )


@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("method", ["efron", "breslow", "exact"])
def test_cox_toler_inf_controls_separation_warning(counting, method):
    fit, messages = _cox_fit(counting, method)
    noun = "beta" if counting or method == "exact" else "coefficient"
    assert messages == [f"Loglik converged before variable 1; {noun} may be infinite."]
    assert fit.coefficients[0][0] == pytest.approx(-22.229054253571782, abs=2e-8, rel=0.0)
    assert fit.iterations == 20
    assert fit.convergence_flag == 1

    suppressed, messages = _cox_fit(counting, method, control={"toler.inf": 1e9})
    assert messages == []
    assert suppressed.coefficients == fit.coefficients
    assert suppressed.log_likelihood == fit.log_likelihood

    _, alias_messages = _cox_fit(counting, method, control={"toler_inf": 1e9})
    assert alias_messages == []


@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("method", ["efron", "breslow", "exact"])
def test_counting_cox_uses_r_infinity_threshold(counting, method):
    # R agreg.fit uses toler.inf * (1 + abs(beta)); right/exact fits
    # instead use toler.inf * abs(beta) and also require step > eps.
    _, messages = _cox_fit(counting, method, control={"toler.inf": 0.044})
    assert bool(messages) == (not counting or method == "exact")

    fit, messages = _cox_fit(counting, method, control={"eps": 2.0, "toler.inf": 1e-5})
    assert fit.iterations == 1
    assert bool(messages) == (counting and method != "exact")


@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("method", ["efron", "breslow", "exact"])
@pytest.mark.parametrize("max_iter", [0, 1, 2])
def test_cox_iteration_limit_diagnostics(counting, method, max_iter):
    fit, messages = _cox_fit(counting, method, control={"iter.max": max_iter})
    assert fit.iterations == max_iter
    assert messages == (["Ran out of iterations and did not converge"] if max_iter > 1 else [])
    if max_iter:
        assert fit.convergence_flag == 1000
    if max_iter == 2:
        assert fit.coefficients[0][0] == pytest.approx(-4.1523876700113895)


@pytest.mark.parametrize("counting", [False, True])
@pytest.mark.parametrize("method", ["efron", "breslow", "exact"])
def test_null_and_no_event_cox_models_do_not_warn(counting, method):
    response = "Surv(start, time, status)" if counting else "Surv(time, status)"
    _, messages = _fit_with_messages(
        survival.coxph, f"{response} ~ 1", data=_separated_data(), method=method
    )
    assert messages == []
    data = _separated_data()
    data["status"] = [0] * 10
    _, messages = _fit_with_messages(survival.coxph, f"{response} ~ x", data=data, method=method)
    assert messages == []


@pytest.mark.parametrize(
    ("offset", "expected_beta", "extra_warning"),
    [
        ([600.0] * 10, -4.1523876700113895, False),
        ([600.0] + [0.0] * 9, -4.4287696911351633, True),
    ],
)
def test_cox_overflow_diagnostic_centers_offsets(offset, expected_beta, extra_warning):
    # R formula references add offset(off); constant offsets are centered
    # before fitting, so they must not trigger the extra overflow warning.
    fit, messages = _cox_fit(False, "efron", offset=offset, max_iter=2)
    assert fit.coefficients[0][0] == pytest.approx(expected_beta)
    expected = ["Ran out of iterations and did not converge"]
    if extra_warning:
        expected.append("one or more coefficients may be infinite")
    assert messages == expected


@pytest.mark.parametrize("max_iter", [0, 1, 2, 30])
def test_survreg_iteration_limit_diagnostics(max_iter):
    # R survreg(Surv(time, status) ~ x, ..., control =
    # survreg.control(maxiter = max_iter)) warns only at max_iter = 2.
    data = {
        "time": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "status": [1, 1, 0, 1, 0, 1],
        "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    }
    fit, messages = _fit_with_messages(
        survival.survreg,
        "Surv(time, status) ~ x",
        data=data,
        control={"maxiter": max_iter, "rel.tolerance": 1e-9},
    )
    assert messages == (["Ran out of iterations and did not converge"] if max_iter == 2 else [])
    assert fit.convergence_flag == (0 if max_iter == 30 else -1)
    if max_iter == 30:
        assert fit.coefficients[:2] == pytest.approx([2.2010042310878, -0.41286628197668])


def test_survreg_default_tolerance_matches_r_control_and_native_api():
    data = {
        "time": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        "status": [1, 1, 0, 1, 0, 1],
        "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    }
    assert survival.SurvregConfig().eps == 1e-9
    # Keep a common prescribed start so this comparison isolates the tolerance.
    initial = [0.0, 0.0, 0.0]
    default = survival.survreg("Surv(time, status) ~ x", data=data, init=initial)
    controlled = survival.survreg(
        "Surv(time, status) ~ x", data=data, init=initial, control={"rel.tolerance": 1e-9}
    )
    native_kwargs = {
        "initial_beta": initial,
        "time": data["time"],
        "status": data["status"],
        "covariates": [[1.0, value] for value in data["x"]],
    }
    native_default = survival.regression.survreg(**native_kwargs)
    native_explicit = survival.regression.survreg(**native_kwargs, eps=1e-9)
    for fit in (controlled, native_default, native_explicit):
        assert fit.coefficients == default.coefficients
        assert fit.iterations == default.iterations
        assert fit.log_likelihood == default.log_likelihood
    # R survival 3.8.11: c(coef(survreg(...)), log(fit$scale)).
    assert default.coefficients == pytest.approx(
        [2.2010042306656832, -0.41286628148940124, -0.28061083829247357],
        abs=2e-9,
        rel=0.0,
    )
    assert max(abs(value) for value in default.score_vector) < 1e-10
    loose = survival.survreg("Surv(time, status) ~ x", data=data, init=initial, eps=1e-6)
    assert loose.iterations < default.iterations
    assert max(abs(value) for value in loose.score_vector) > 1e-8
