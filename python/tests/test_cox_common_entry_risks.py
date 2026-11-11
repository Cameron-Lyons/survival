"""Common-entry Cox risk sweeps retain normalized moments and exposures."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
native = survival._survival


def _decimal(value):
    return Decimal.from_float(float(value))


def _decimal_residuals(
    time, status, x, weights, ties, *, entry=None, strata=None, offset=None, beta=None, risks=None
):
    """Direct positive risk-set sums; no sweep, risk subtraction, or native calls."""
    n, p = x.shape
    entry = np.full(n, -np.inf) if entry is None else entry
    strata = np.zeros(n, dtype=int) if strata is None else strata
    with localcontext() as context:
        context.prec = 100
        zero = Decimal(0)
        one = Decimal(1)
        covariates = [[_decimal(value) for value in row] for row in x]
        w = [_decimal(value) for value in weights]
        if risks is None:
            risks = [
                (
                    _decimal(offset[i])
                    + sum((covariates[i][j] * _decimal(beta[j]) for j in range(p)), zero)
                ).exp()
                for i in range(n)
            ]
        else:
            risks = [_decimal(value) for value in risks]
        expected = [zero] * n
        score = [[zero] * p for _ in range(n)]
        schoenfeld = {}
        for group in np.unique(strata):
            for event_time in sorted(
                {time[i] for i in range(n) if strata[i] == group and status[i]}
            ):
                active = [
                    i for i in range(n) if strata[i] == group and entry[i] < event_time <= time[i]
                ]
                deaths = {i for i in active if time[i] == event_time and status[i]}
                death_weight = sum((w[i] for i in deaths), zero)
                steps = len(deaths) if ties == "efron" else 1
                average_mean = [zero] * p
                for step in range(steps):
                    fraction = Decimal(step) / len(deaths)
                    adjusted = {
                        i: risks[i] * (one - fraction if i in deaths else one) for i in active
                    }
                    denominator = sum((w[i] * adjusted[i] for i in active), zero)
                    mean = [
                        sum((w[i] * adjusted[i] * covariates[i][j] for i in active), zero)
                        / denominator
                        for j in range(p)
                    ]
                    for j in range(p):
                        average_mean[j] += mean[j] / steps
                    for i in active:
                        count = death_weight / steps * adjusted[i] / denominator
                        expected[i] += count
                        for j in range(p):
                            score[i][j] -= count * (covariates[i][j] - mean[j])
                for i in deaths:
                    schoenfeld[i] = [covariates[i][j] - average_mean[j] for j in range(p)]
                    for j in range(p):
                        score[i][j] += schoenfeld[i][j]
        martingale = np.asarray([float(Decimal(int(status[i])) - expected[i]) for i in range(n)])
        score = np.asarray([[float(value) for value in row] for row in score])
        schoenfeld = {i: [float(value) for value in row] for i, row in schoenfeld.items()}
        return martingale, score, schoenfeld


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("counting", [False, True])
def test_constant_large_covariate_has_exactly_zero_score(ties, counting):
    time = np.arange(1.0, 9.0)
    status = [1, 1, 1, 1, 1, 0, 0, 0]
    offset = [-100.0] * 5 + [0.0] * 3
    x = np.full((8, 1), 1e16)
    fit = native.coxph_fit(
        time,
        status,
        x,
        offset=offset,
        entry=[0.0] * 8 if counting else None,
        init=[0.0],
        iter_max=0,
        method=ties,
    )
    # Every risk-set mean is exactly the sole covariate value. Routing this
    # counting input through ordinary agscore previously produced scores of -2.
    np.testing.assert_array_equal(fit.score_residuals(), np.zeros((8, 1)))
    np.testing.assert_array_equal(fit.schoenfeld_residuals().residuals, np.zeros((5, 1)))
    expected, _, _ = _decimal_residuals(
        time, status, x, np.ones(8), ties, offset=offset, beta=[0.0]
    )
    np.testing.assert_allclose(fit.residuals, expected, rtol=2e-13, atol=0)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("counting", [False, True])
def test_late_small_risk_set_does_not_corrupt_earlier_martingales(ties, counting):
    time = [1.0, 1.0, 1.0, 2.0, 3.0]
    status = [1, 0, 0, 1, 0]
    offset = [0.0, 0.0, 0.0, -100.0, np.log(1e-10)]
    x = np.empty((5, 0))
    fit = native.coxph_fit(
        time,
        status,
        x,
        offset=offset,
        entry=[0.0] * 5 if counting else None,
        method=ties,
    )
    expected, _, _ = _decimal_residuals(time, status, x, np.ones(5), ties, offset=offset, beta=[])
    # The first denominator is 3 + 1e-10 + exp(-100). Subtracting later
    # cumulative exposure in ordinary reverse arithmetic loses about 6e-7.
    np.testing.assert_allclose(
        expected[:3], [2 / 3 + 1 / 9e10, -1 / 3 + 1 / 9e10, -1 / 3 + 1 / 9e10], rtol=0, atol=2e-16
    )
    np.testing.assert_allclose(fit.residuals, expected, rtol=2e-13, atol=2e-15)


def _risk_data(entry_mode, weighted, stratified):
    time = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 7.0])
    status = np.asarray([1, 1, 1, 0, 1, 1, 0, 0])
    x = np.asarray(
        [
            [-1.2, 0.5],
            [0.1, 1.0],
            [0.4, -0.3],
            [1.6, 0.2],
            [-0.7, -1.1],
            [0.9, 0.6],
            [0.2, 1.3],
            [-0.3, -0.4],
        ]
    )
    offset = np.asarray([-80.0, -45.0, -80.0, 10.0, -60.0, -70.0, -10.0, -40.0])
    weights = np.asarray([0.5, 2.0, 1.25, 3.0, 0.75, 1.5, 4.0, 0.8]) if weighted else np.ones(8)
    starts = {
        "right": np.full(8, -np.inf),
        "common": np.zeros(8),
        "before-first-stop": np.asarray([-3.0, -1.0, 0.0, -0.5, -2.0, 0.0, 0.25, 0.5]),
        "delayed": np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 4.0, 5.0]),
    }
    entry = starts[entry_mode]
    strata = np.zeros(8, dtype=int)
    if stratified:
        time, status, weights, entry = [
            np.tile(values, 2) for values in [time, status, weights, entry]
        ]
        x = np.vstack([x, x * [-0.7, 1.3]])
        offset = np.concatenate([offset, offset + 1000.0])
        strata = np.repeat([3, 7], 8)
    # Neither the input order nor interleaving of strata can determine the
    # event order used by the independent oracle.
    order = np.random.default_rng(419).permutation(len(time))
    return tuple(values[order] for values in [time, status, x, offset, weights, entry, strata])


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("entry_mode", ["right", "common", "before-first-stop", "delayed"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("stratified", [False, True])
def test_event_moments_and_exposures_match_decimal(ties, entry_mode, weighted, stratified):
    time, status, x, offset, weights, entry, strata = _risk_data(entry_mode, weighted, stratified)
    beta = [0.35, -0.2]
    fit = native.coxph_fit(
        time,
        status,
        x,
        offset=offset,
        weights=weights,
        entry=None if entry_mode == "right" else entry,
        strata=strata if stratified else None,
        init=beta,
        iter_max=0,
        method=ties,
    )
    expected, score, schoenfeld = _decimal_residuals(
        time, status, x, weights, ties, entry=entry, strata=strata, offset=offset, beta=beta
    )
    np.testing.assert_allclose(fit.residuals, expected, rtol=3e-12, atol=2e-13)
    np.testing.assert_allclose(
        fit.martingale_residuals(weighted=True), expected * weights, rtol=3e-12, atol=2e-13
    )
    np.testing.assert_allclose(fit.score_residuals(), score, rtol=3e-12, atol=2e-13)
    np.testing.assert_allclose(
        fit.score_residuals(weighted=True), score * weights[:, None], rtol=3e-12, atol=2e-13
    )
    observed = fit.schoenfeld_residuals()
    np.testing.assert_allclose(
        observed.residuals, [schoenfeld[i] for i in observed.rows], rtol=3e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        fit.first, (score * weights[:, None]).sum(axis=0), rtol=3e-12, atol=2e-13
    )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("counting", [False, True])
def test_raw_zero_risks_have_zero_exposure(ties, counting):
    time = [1.0, 1.0, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1, 0]
    risks = [0.0, 1e-40, 1e-10, 1e-30, 0.0]
    weights = [0.5, 2.0, 1.5, 0.75, 2.0]
    if counting:
        data = native.AndersenGillInput(
            native.CountingProcessData([0.0] * 5, time, status), risks, native.Weights(weights)
        )
        actual = native.agmart(data, ties=ties)
    else:
        data = native.CoxMartInput(
            native.SurvivalData(time, status), risks, native.Weights(weights)
        )
        actual = native.coxmart(data, ties=ties)
    expected, _, _ = _decimal_residuals(time, status, np.empty((5, 0)), weights, ties, risks=risks)
    assert actual[0] == 1.0
    assert actual[4] == 0.0
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-13)
