"""Predictions at requested times agree with the full R-compatible Cox curves."""

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()


def _fit(method="efron", stratified=False, counting=False):
    rng = np.random.default_rng(27)
    n = 60
    time = rng.integers(2, 20, size=n).astype(float)
    status = (np.arange(n) % 3 != 0).astype(np.int32)
    x = rng.normal(size=(n, 2))
    strata = np.where(np.arange(n) % 2 == 0, 17, -3) if stratified else None
    return survival.regression.coxph_fit(
        time,
        status,
        x,
        method=method,
        strata=strata,
        offset=rng.normal(scale=0.1, size=n),
        weights=None if method == "exact" else rng.uniform(0.5, 2.0, size=n),
        entry=None if not counting else np.zeros(n),
    )


def _full_curve_predictions(fit, times, **newdata):
    columns = []
    for curve in fit.survfit(**newdata, se_fit=False):
        values = np.asarray(curve.surv)
        positions = np.searchsorted(curve.time, times, side="right") - 1
        selected = np.ones((len(times), values.shape[1]))
        valid = positions >= 0
        selected[valid] = values[positions[valid]]
        columns.append(selected)
    return np.concatenate(columns, axis=1)


@pytest.mark.parametrize("method", ["breslow", "efron", "exact"])
@pytest.mark.parametrize("stratified", [False, True])
@pytest.mark.parametrize("counting", [False, True])
def test_requested_times_match_full_curves(method, stratified, counting):
    fit = _fit(method, stratified, counting)
    # Out of order, repeated, before the first event, at a tie and past the end.
    times = np.array([30.0, 5.0, -1.0, 5.0, 7.5, 2.0])
    training = {"newdata": fit.x, "new_strata": fit.strata, "new_offset": fit.offset}
    expected = _full_curve_predictions(fit, times, **training)
    result = fit.predict_survival_at(times)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(times), fit.n)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-15)
    # The explicit training rows take the same path, without changing their order.
    np.testing.assert_array_equal(result, fit.predict_survival_at(times, **training))

    order = [15, 0, 43, 4, 15]
    new = {
        "newdata": np.asfortranarray(np.asarray(fit.x)[order] + 0.2),
        "new_offset": np.asarray(fit.offset)[order] - 0.15,
        "new_strata": None if fit.strata is None else np.asarray(fit.strata)[order],
    }
    np.testing.assert_allclose(
        fit.predict_survival_at(times, **new),
        _full_curve_predictions(fit, times, **new),
        rtol=1e-14,
        atol=1e-15,
    )
    assert fit.predict_survival_at([], **new).shape == (0, len(order))
    assert fit.predict_survival_at([]).shape == (0, fit.n)


def test_requested_times_validate_inputs():
    fit = _fit(stratified=True)
    for times in ([float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="times"):
            fit.predict_survival_at(times)
    with pytest.raises(ValueError, match="strata"):
        fit.predict_survival_at([3.0], newdata=[[0.0, 0.0]])
    with pytest.raises(ValueError, match="strata"):
        fit.predict_survival_at([3.0], newdata=[[0.0, 0.0]], new_strata=[99])
    with pytest.raises(ValueError, match="columns"):
        fit.predict_survival_at([3.0], newdata=[[0.0]], new_strata=[17])
    with pytest.raises(ValueError, match="require newdata"):
        fit.predict_survival_at([3.0], new_offset=[0.1])


@pytest.mark.parametrize("events", [False, True])
def test_requested_times_handle_null_models_and_no_events(events):
    time = np.arange(1.0, 9.0)
    status = (np.arange(8) % 3 != 0).astype(np.int32) if events else np.zeros(8, dtype=np.int32)
    x = np.empty((8, 0))
    fit = survival.regression.coxph_fit(time, status, x)
    times = [0.0, 3.0, 10.0]
    np.testing.assert_allclose(
        fit.predict_survival_at(times), _full_curve_predictions(fit, times, newdata=x)
    )


def test_brier_kernel_accepts_strided_numpy_inputs():
    fit = _fit()
    times = np.array([3.0, 6.0, 10.0])
    phat = 1.0 - fit.predict_survival_at(times)
    time = np.asarray(fit.time)[::-1]
    status = np.asarray(fit.status, dtype=np.float64)[::-1]
    weights = np.asarray(fit.weights)[::-1]
    predictions = phat[:, ::-1]
    kernel = survival._survival.brier
    result = kernel(time, status, times, predictions, weights=weights)
    reference = kernel(
        time.tolist(),
        status.astype(int).tolist(),
        times.tolist(),
        predictions.tolist(),
        weights=weights.tolist(),
    )
    np.testing.assert_array_equal(result.brier, reference.brier)
    np.testing.assert_array_equal(result.eff_n, reference.eff_n)
    with pytest.raises(TypeError, match="not an int32"):
        kernel(time, status + 0.5, times, predictions)


def test_estimator_keeps_requested_time_order_and_empty_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 2))
    y = np.column_stack([rng.exponential(size=40), np.arange(40) % 3 != 0])
    model = survival.CoxPHEstimator().fit(x, y)
    times = np.array([2.0, 0.0, 0.5, 0.5])
    returned_times, values = model.predict_survival_function(x[::3], times)
    np.testing.assert_array_equal(returned_times, times)
    np.testing.assert_allclose(
        values, _full_curve_predictions(model.model_, times, newdata=x[::3]).T
    )
    assert model.predict_survival_function(x[::3], [])[1].shape == (14, 0)


@pytest.mark.parametrize("stratified", [False, True])
def test_brier_preserves_time_order_and_training_row_alignment(stratified):
    fit = _fit(stratified=stratified)
    times = [12.0, 1.0, 5.0, 5.0, 8.5]
    expected = 1.0 - _full_curve_predictions(
        fit, times, newdata=fit.x, new_strata=fit.strata, new_offset=fit.offset
    )
    result = survival.r.brier(fit, times=times, detail=True)
    assert isinstance(result.phat, list)
    np.testing.assert_allclose(result.phat, expected, rtol=1e-14, atol=1e-15)
    assert result.times == times
    assert result.brier[2] == result.brier[3]
    empty = survival.r.brier(fit, times=[], detail=True)
    assert empty.times == empty.brier == empty.phat == []
