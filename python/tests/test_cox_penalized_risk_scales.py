"""Penalized fits must use the same risk scale as residuals and predictions.

Moderate per-risk-set calculations provide an independent likelihood/score
oracle. Disjoint counting intervals also have a separate-stratum control.
"""

import math

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
native = survival.kernels
r = survival.r


def _data(disjoint=False):
    time = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    x = np.array([-0.8, 0.3, 1.2, -0.2, 0.7, -1.1, 0.5, 1.5])
    status = np.array([1, 1, 1, 0, 1, 0, 1, 0])
    groups = np.repeat(np.arange(1, 5), 2)
    if disjoint:
        return {
            "time": np.concatenate((time, time + 10.0)),
            "start": np.repeat([0.0, 10.0], len(time)),
            "event": np.tile(status, 2),
            "x": np.tile(x, 2),
            "g": np.tile(groups, 2),
            "block": np.repeat([0, 1], len(time)),
            "one": np.ones(2 * len(time)),
        }
    return {
        "time": time,
        "start": np.zeros(len(time)),
        "event": status,
        "x": x,
        "g": groups,
        "block": np.zeros(len(time), dtype=int),
        "one": np.ones(len(time)),
    }


def _oracle(data, coefficient, ties, frailty=None):
    """Direct positive risk-set sums, without the implementation's kernels."""
    x = data["x"]
    eta = coefficient * x
    if frailty is not None:
        eta = eta + np.asarray(frailty)[data["g"] - 1]
    expected = np.zeros(len(x))
    baseline_expected = np.zeros(len(x))
    score = np.zeros(len(x))
    loglik = 0.0
    for group in np.unique(data["block"]):
        group_rows = data["block"] == group
        for time in np.unique(data["time"][group_rows & (data["event"] == 1)]):
            active = np.flatnonzero(group_rows & (data["start"] < time) & (time <= data["time"]))
            deaths = np.flatnonzero(group_rows & (data["time"] == time) & (data["event"] == 1))
            maximum = eta[active].max()
            risk = np.exp(eta[active] - maximum)
            dead = np.isin(active, deaths)
            steps = len(deaths) if ties == "efron" else 1
            mean = 0.0
            loglik += np.sum(eta[deaths] - maximum)
            for step in range(steps):
                fraction = step / len(deaths)
                adjusted = risk * np.where(dead, 1.0 - fraction, 1.0)
                denominator = adjusted.sum()
                xbar = np.dot(adjusted, x[active]) / denominator
                mean += xbar / steps
                increment = len(deaths) / steps / denominator
                counts = increment * adjusted
                expected[active] += counts
                baseline_expected[active] += increment * risk
                score[active] -= counts * (x[active] - xbar)
                loglik -= len(deaths) / steps * np.log(denominator)
            score[deaths] += x[deaths] - mean
    return loglik, score, data["event"] - expected, expected, baseline_expected


def _close(actual, expected, *, tolerance=2e-8):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _expected_se_oracle(data, coefficient, variance, ties, shift, frailty=None):
    """R's endpoint-variance difference, expanded before applying block offsets.

    Unlike the expected count, this includes a covariance term between the
    baseline before entry and the baseline within the observed interval.
    """
    x = data["x"]
    eta = coefficient * x
    if frailty is not None:
        eta = eta + np.asarray(frailty)[data["g"] - 1]
    risk = np.exp(eta)
    events = []
    for group in np.unique(data["block"]):
        rows = data["block"] == group
        for time in np.unique(data["time"][rows & (data["event"] == 1)]):
            active = rows & (data["start"] < time) & (time <= data["time"])
            deaths = rows & (data["time"] == time) & (data["event"] == 1)
            count = int(deaths.sum())
            steps = count if ties == "efron" else 1
            for step in range(steps):
                adjusted = risk[active] * np.where(deaths[active], 1.0 - step / count, 1.0)
                denominator = adjusted.sum()
                mean = np.dot(adjusted, x[active]) / denominator
                events.append((group, time, count / steps / denominator, mean, denominator))
    result = []
    for row, group in enumerate(data["block"]):
        local_delta = 0.0
        local_variance = 0.0
        prior = {}
        for event_group, time, hazard, mean, denominator in events:
            if event_group == group and data["start"][row] < time <= data["time"][row]:
                local_delta += hazard * (x[row] - mean)
                local_variance += hazard / denominator
            elif time <= data["start"][row]:
                prior[event_group] = prior.get(event_group, 0.0) + hazard * (x[row] - mean)
        total = local_variance + variance * local_delta**2
        for event_group, prior_delta in prior.items():
            cross = 2.0 * variance * prior_delta * local_delta
            if cross != 0.0:
                log_ratio = shift * (group - event_group)
                ratio = (
                    math.inf if log_ratio > math.log(np.finfo(float).max) else math.exp(log_ratio)
                )
                total += cross * ratio
        result.append(math.sqrt(total) * risk[row] if total >= 0 else math.nan)
    return np.asarray(result)


def _check_expected_se(fit, data, ties, shift):
    expected = _expected_se_oracle(
        data, fit.coefficients[-1], np.asarray(fit.var)[-1, -1], ties, shift, fit.frail
    )
    actual = fit.coxph.predict("expected", se_fit=True).se_fit
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8, equal_nan=True)


def _native_fit(
    data,
    ties,
    iterations,
    *,
    theta=None,
    sparse=False,
    constant=None,
    offset=None,
    split=False,
    counting=False,
):
    columns = [data["x"]]
    init = [0.2]
    assign = [[0]]
    if constant is not None:
        columns.insert(0, data["one"])
        init.insert(0, constant)
        assign = [[0], [1]]
    if sparse:
        penalty = native.CoxPenalty.frailty(
            distribution="gaussian", sparse=True, theta=0.4, n=len(data["x"])
        )
        pcols = [[len(columns)]]
        assign.append(pcols[0])
        columns.append(data["g"])
    else:
        penalty = native.CoxPenalty.ridge(theta=theta, scale=False)
        pcols = [[len(columns) - 1]]
    return native.coxpenal_fit(
        data["time"],
        data["event"],
        np.column_stack(columns),
        penalties=[penalty],
        pcols=pcols,
        assign=assign,
        entry=data["start"] if counting else None,
        strata=data["block"] if split else None,
        offset=offset,
        method=ties,
        init=init,
        iter_max=iterations,
        outer_max=1,
    )


def _check_oracle(fit, data, ties, theta):
    coefficient = fit.coefficients[-1]
    loglik, score, residuals, expected, explicit_expected = _oracle(
        data, coefficient, ties, fit.frail
    )
    _close(fit.loglik[-1], loglik)
    initial = _oracle(data, 0.2, ties)[0] - (0.0 if theta is None else 0.5 * theta * 0.2**2)
    _close(fit.loglik[0], initial)
    _close(fit.residuals, residuals)
    _close(np.asarray(fit.coxph.score_residuals())[:, -1], score)
    _close(fit.coxph.first[-1], score.sum() - (0.0 if theta is None else theta * coefficient))
    _close(fit.coxph.predict("expected").fit, expected)
    return explicit_expected


def _compare_native(actual, expected, *, compare_expected_se=True):
    # The added constant is aliased. Only its identifiable neighbour is compared.
    _close(actual.coefficients[-1], expected.coefficients[-1])
    _close(actual.loglik, expected.loglik)
    _close(actual.coxph.first[-1], expected.coxph.first[-1])
    _close(np.asarray(actual.var)[-1, -1], np.asarray(expected.var)[-1, -1])
    _close(actual.residuals, expected.residuals)
    _close(
        np.asarray(actual.coxph.score_residuals())[:, -1],
        np.asarray(expected.coxph.score_residuals())[:, -1],
    )
    a = actual.coxph.predict("expected", se_fit=compare_expected_se)
    b = expected.coxph.predict("expected", se_fit=compare_expected_se)
    _close(a.fit, b.fit)
    if compare_expected_se:
        _close(a.se_fit, b.se_fit)
    if actual.frail is not None:
        _close(actual.frail, expected.frail)
        _close(actual.fvar, expected.fvar)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("counting", [False, True])
def test_zero_ridge_does_not_clip_a_finite_partial_likelihood(ties, counting):
    fit = native.coxpenal_fit(
        [1.0, 1.0],
        [0, 1],
        [[1.0], [0.0]],
        penalties=[native.CoxPenalty.ridge(theta=0.0, scale=False)],
        pcols=[[0]],
        assign=[[0]],
        entry=[0.0, 0.0] if counting else None,
        method=ties,
        init=[1e16],
        iter_max=0,
        outer_max=1,
    )
    # The sole event has risk1 against a censored risk exp(1e16): its
    # log-probability is -1e16, not the result of clipping either predictor.
    _close(fit.loglik, [-1e16, -1e16], tolerance=1e-14)
    _close(fit.residuals, [-1.0, 1.0])
    _close(fit.coxph.first, [-1.0])
    _close(fit.coxph.predict("expected").fit, [1.0, 0.0])


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("theta", [0.0, 1.0, None], ids=["ridge_zero", "ridge_finite", "frailty"])
@pytest.mark.parametrize("iterations", [1, 8])
@pytest.mark.parametrize("counting", [False, True])
def test_moderate_penalized_fits_match_independent_risk_set_definitions(
    ties, theta, iterations, counting
):
    data = _data()
    fit = _native_fit(data, ties, iterations, theta=theta, sparse=theta is None, counting=counting)
    _check_oracle(fit, data, ties, theta)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("theta", [0.0, 1.0, None], ids=["ridge_zero", "ridge_finite", "frailty"])
@pytest.mark.parametrize("iterations", [0, 1, 8])
@pytest.mark.parametrize("constant", [-1e16, 1e16])
@pytest.mark.parametrize("counting", [False, True])
def test_penalized_constant_coefficients_preserve_small_effects(
    ties, theta, iterations, constant, counting
):
    data = _data()
    options = {"theta": theta, "sparse": theta is None, "counting": counting}
    reference = _native_fit(data, ties, iterations, **options)
    fit = _native_fit(data, ties, iterations, constant=constant, **options)
    _compare_native(fit, reference)
    _check_oracle(fit, data, ties, theta)
    if theta is None:
        # Stock R's public frailty survfit rejects newdata. The underlying Cox
        # fit still exposes curves for a specified dense covariate vector.
        for model, row in [(fit, [1.0, 0.1]), (reference, [0.1])]:
            with pytest.raises(ValueError, match="Newdata cannot be used.*frailty"):
                model.survfit(newdata=[row], se_fit=True)
        actual = fit.coxph.survfit(newdata=[[1.0, 0.1]], se_fit=True)
        expected = reference.coxph.survfit(newdata=[[0.1]], se_fit=True)
    else:
        actual = fit.survfit(newdata=[[1.0, 0.1]], se_fit=True)
        expected = reference.survfit(newdata=[[0.1]], se_fit=True)
    assert len(actual) == len(expected) == 1
    for name in ["surv", "cumhaz", "std_err"]:
        _close(getattr(actual[0], name), getattr(expected[0], name))


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("theta", [0.0, 1.0, None], ids=["ridge_zero", "ridge_finite", "frailty"])
@pytest.mark.parametrize("iterations", [1, 8])
@pytest.mark.parametrize("shift", [-1e16, 1e16])
def test_penalized_disjoint_offsets_match_separate_strata(ties, theta, iterations, shift):
    data = _data(disjoint=True)
    offset = shift * data["block"]
    options = {"theta": theta, "sparse": theta is None, "counting": True}
    reference = _native_fit(data, ties, iterations, split=True, **options)
    fit = _native_fit(data, ties, iterations, offset=offset, **options)
    _compare_native(fit, reference, compare_expected_se=False)
    _check_expected_se(fit, data, ties, shift)
    explicit_expected = _check_oracle(fit, data, ties, theta)
    # Newdata has no event indicator and integrates the ordinary baseline,
    # without the Efron correction used for training residuals.
    if theta is not None:
        newdata = {
            "newdata": data["x"][:, None],
            "new_entry": data["start"],
            "new_time": data["time"],
        }
        actual = fit.coxph.predict("expected", new_offset=offset, **newdata)
        expected = reference.coxph.predict("expected", new_strata=data["block"], **newdata)
        _close(actual.fit, expected.fit)
        _close(actual.fit, explicit_expected)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("theta", [0.0, 1.0])
def test_formula_ridge_constant_and_dynamic_offset_paths(ties, theta):
    term = f"ridge(x, theta={theta}, scale=FALSE)"
    data = _data()
    options = {"ties": ties, "iter_max": 1, "outer_max": 1}
    reference = r.coxph(f"Surv(time,event) ~ {term}", data, init=[0.2], **options)
    fit = r.coxph(f"Surv(time,event) ~ one + {term}", data, init=[1e16, 0.2], **options)
    _compare_native(fit.penalized, reference.penalized)
    _close(np.asarray(r.residuals(fit, type="score"))[:, -1], r.residuals(reference, type="score"))
    _close(r.predict(fit, type="expected"), r.predict(reference, type="expected"))

    data = _data(disjoint=True)
    data["off"] = -1e16 * data["block"]
    reference = r.coxph(f"Surv(start,time,event) ~ {term} + strata(block)", data, **options)
    fit = r.coxph(f"Surv(start,time,event) ~ {term} + offset(off)", data, **options)
    _compare_native(fit.penalized, reference.penalized, compare_expected_se=False)
    _check_expected_se(fit.penalized, data, ties, -1e16)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_formula_sparse_frailty_dynamic_offsets_match_stratified_control(ties):
    data = _data(disjoint=True)
    data["off"] = -1e16 * data["block"]
    term = "x + frailty(g, distribution='gaussian', theta=.4, sparse=TRUE)"
    options = {"ties": ties, "iter_max": 8, "outer_max": 1}
    reference = r.coxph(f"Surv(start,time,event) ~ {term} + strata(block)", data, **options)
    fit = r.coxph(f"Surv(start,time,event) ~ {term} + offset(off)", data, **options)
    _compare_native(fit.penalized, reference.penalized, compare_expected_se=False)
    _check_expected_se(fit.penalized, data, ties, -1e16)
    _close(r.residuals(fit), r.residuals(reference))
    _close(r.predict(fit, type="expected"), r.predict(reference, type="expected"))


@pytest.mark.parametrize(
    ("ties", "coefficient", "variance", "se"),
    [
        (
            "breslow",
            -0.157151137363819,
            0.130385251301319,
            [
                0.154684713814027,
                0.238363173113718,
                0.239800705107323,
                0.268169167922342,
                0.339927550569023,
                0.591647067095946,
                0.635929927393032,
                0.638224825414232,
                0.746449754106794,
                0.214553911380077,
                0.695077036647892,
                0.69082385454632,
                0.434953783355455,
                2.42089203946963,
                0.648892997694778,
                1.44032765833442,
            ],
        ),
        (
            "efron",
            -0.140685145195415,
            0.12925588884679,
            [
                0.152414498026023,
                0.252859368683061,
                0.258242732267147,
                0.281442665476821,
                0.352328476966487,
                0.59529672993519,
                0.638265787712145,
                0.654216974075429,
                0.738724495139655,
                0.223432644272883,
                0.735989968974867,
                0.69965726793309,
                0.4537696539053,
                2.40924327802454,
                0.649479267270939,
                1.48999081619199,
            ],
        ),
    ],
)
def test_endpoint_variance_oracle_matches_stock_r(ties, coefficient, variance, se):
    # R survival 3.8-12, ridge(x, theta=1, scale=FALSE) + offset(2*block),
    # coxph.control(iter.max=20, outer.max=1), predict(type="expected", se.fit=TRUE).
    data = _data(disjoint=True)
    _close(_expected_se_oracle(data, coefficient, variance, ties, 2.0), se)
    fit = _native_fit(data, ties, 20, theta=1.0, offset=2.0 * data["block"], counting=True)
    _close(fit.coefficients, [coefficient])
    _close(fit.var, [[variance]])
    _close(fit.coxph.predict("expected", se_fit=True).se_fit, se)


@pytest.mark.parametrize("counting", [False, True])
def test_unrepresentable_expected_count_raises_a_numerical_error(counting):
    # Weighted likelihood and information are finite, but the censored row's
    # unweighted expected count exceeds f64. This must not panic in cleanup.
    with pytest.raises(RuntimeError, match="martingale expected count is not finite"):
        native.coxpenal_fit(
            [1.0, 1.0],
            [0, 1],
            [[1.0], [1.0]],
            weights=[1e-310, 1.0],
            offset=[500.0, -500.0],
            penalties=[native.CoxPenalty.ridge(theta=0.0, scale=False)],
            pcols=[[0]],
            assign=[[0]],
            entry=[0.0, 0.0] if counting else None,
            init=[0.0],
            iter_max=0,
            outer_max=1,
        )
