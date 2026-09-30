"""External prediction methods and caller-owned Yates normal draws."""

import gc
import weakref

import numpy as np
import pytest
from numpy.testing import assert_allclose
from survival import validation


@pytest.fixture
def simulation():
    return {
        "xmatlist": [[[1.0, 0.0], [1.0, 2.0]], [[1.0, 3.0]]],
        "beta": [0.2, 0.1],
        "vmat": [[0.01, 0.002], [0.002, 0.02]],
        "means": [1.0, 0.5],
        "nsim": 4,
        "normal_draws": [[-1.0, 0.5], [0.0, -1.0], [1.0, 0.0], [0.5, 1.0]],
    }


def test_matrix_predictions_match_independent_population_moments(simulation):
    def predict(eta):
        return np.column_stack((np.exp(eta), eta, eta**2))

    result = validation.yates_predict(**simulation, predict=predict)
    beta = np.asarray(simulation["beta"])
    values, vectors = np.linalg.eigh(simulation["vmat"])
    root = (vectors * np.sqrt(values)) @ vectors.T
    coefficients = beta + np.asarray(simulation["normal_draws"]) @ root

    def population(coef):
        return [
            predict((np.asarray(x) - simulation["means"]) @ coef).mean(axis=0)
            for x in simulation["xmatlist"]
        ]

    point = np.asarray(population(beta))
    draws = np.asarray([population(coef) for coef in coefficients])
    assert_allclose([row.pmm for row in result.estimate], point[:, 0])
    assert_allclose(result.mvar, np.cov(draws[:, :, 0], rowvar=False))
    assert_allclose(result.prediction_mean, draws.mean(axis=0))
    assert_allclose(result.prediction_variance, draws.var(axis=0, ddof=1))
    assert_allclose([row.std for row in result.estimate], draws[:, :, 0].std(axis=0, ddof=1))
    difference = point[0, 0] - point[1, 0]
    assert_allclose(
        result.test[0].chisq, difference**2 / (draws[:, 0, 0] - draws[:, 1, 0]).var(ddof=1)
    )


@pytest.mark.parametrize("kind", ["risk", "response", "predict", "survival"])
def test_draw_provider_called_once_and_seed_is_unused(simulation, kind):
    function = getattr(validation, f"yates_{kind}")
    arguments = dict(simulation)
    events = []
    if kind == "response":

        def inverse(eta):
            events.append("prediction")
            return np.exp(eta)

        arguments["inverse_link"] = inverse
    elif kind == "predict":

        def predict(eta):
            events.append("prediction")
            return np.exp(eta)[:, None]

        arguments["predict"] = predict
    elif kind == "survival":
        arguments.update(time=[1.0, 3.0], cumhaz=[0.1, 0.3], rmean=2.0)
    expected = function(**arguments)
    events.clear()

    def draws(n, p):
        events.append("draws")
        assert (n, p) == (4, 2)
        return np.asfortranarray(simulation["normal_draws"])

    result = function(**{**arguments, "normal_draws": draws}, seed=98327)
    assert_allclose(result.mvar, expected.mvar)
    assert [row.pmm for row in result.estimate] == [row.pmm for row in expected.estimate]
    assert events == (
        ["prediction", "draws"] + ["prediction"] * 4
        if kind in ("response", "predict")
        else ["draws"]
    )


@pytest.mark.parametrize(
    ("draws", "match"),
    [
        ([[0.0, 0.0]], "normal draws rows"),
        ([[0.0]] * 4, "normal draws columns"),
        ([[0.0, np.nan]] * 4, "normal draws"),
        ([[0.0, np.inf]] * 4, "normal draws"),
        (lambda n, p: "bad", "normal draws provider failed"),
    ],
)
def test_invalid_draws_fail_with_context(simulation, draws, match):
    with pytest.raises((ValueError, RuntimeError), match=match):
        validation.yates_risk(**{**simulation, "normal_draws": draws})


@pytest.mark.parametrize(
    ("predict", "match"),
    [
        (0, "predict must be callable"),
        (lambda eta: np.zeros((2, 1)), "prediction rows"),
        (lambda eta: np.zeros((len(eta), 0)), "at least one column"),
        (lambda eta: np.ones((len(eta), 1, 1)), "prediction callback failed"),
        (lambda eta: np.full((len(eta), 1), np.nan), "point prediction"),
    ],
)
def test_invalid_predictions_fail_with_context(simulation, predict, match):
    with pytest.raises((ValueError, TypeError, RuntimeError), match=match):
        validation.yates_predict(**simulation, predict=predict)


def test_prediction_width_cannot_change_between_calls(simulation):
    calls = 0

    def predict(eta):
        nonlocal calls
        calls += 1
        return np.ones((len(eta), calls))

    with pytest.raises(ValueError, match="prediction columns"):
        validation.yates_predict(**simulation, predict=predict)


def test_custom_summary_columns_allow_missing_values(simulation):
    result = validation.yates_predict(
        **simulation,
        predict=lambda eta: np.column_stack((np.exp(eta), np.full(len(eta), np.nan))),
        estimable=[True, False],
    )
    assert np.isnan(result.estimate[1].pmm)
    assert np.isfinite(result.estimate[1].std)
    assert np.isnan(np.asarray(result.prediction_mean)[:, 1]).all()
    assert np.isnan(np.asarray(result.prediction_variance)[:, 1]).all()
    assert np.isnan(result.test[0].chisq)


def test_zero_coefficient_simulation_preserves_row_counts():
    result = validation.yates_predict(
        [np.empty((2, 0)), np.empty((1, 0))],
        [],
        [],
        lambda eta: np.column_stack((np.ones(len(eta)), eta)),
        nsim=3,
        normal_draws=lambda n, p: np.empty((n, p)),
    )
    assert_allclose(result.prediction_mean, [[1.0, 0.0], [1.0, 0.0]])
    assert_allclose(result.prediction_variance, 0.0)


def test_callbacks_are_not_retained_on_success_or_failure(simulation):
    class Callback:
        def __init__(self, fail):
            self.fail = fail

        def __call__(self, eta):
            if self.fail:
                raise RuntimeError("custom failure")
            return np.exp(eta)[:, None]

    for fail in [False, True]:
        callback = Callback(fail)
        reference = weakref.ref(callback)
        if fail:
            with pytest.raises(RuntimeError, match="prediction callback failed.*custom failure"):
                validation.yates_predict(**simulation, predict=callback)
        else:
            validation.yates_predict(**simulation, predict=callback)
        del callback
        gc.collect()
        assert reference() is None
