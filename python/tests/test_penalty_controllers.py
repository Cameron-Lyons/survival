"""Fit-local custom penalty searches through both shared native solvers."""

import gc
import math
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from .helpers import setup_survival_import
from .test_aft_lowlevel import custom_distribution, gaussian_density, gaussian_init

survival = setup_survival_import()
r = survival.r
regression = survival.regression
X = np.column_stack((np.ones(12), [2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2]))
Y = np.column_stack(
    (
        [1.2, 2.5, 0.9, 3, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    )
)


def penalty(coef, theta, neff):
    assert math.isfinite(neff)
    assert neff > 0
    return {
        "penalty": theta * np.dot(coef, coef) / 2,
        "first": theta * coef,
        "second": np.full(len(coef), theta),
        "flag": False,
    }


def controller(old, info):
    if info["iter"] == 0:
        assert old is None
        assert info["eps2"] > 0
        return {"theta": 0.25, "history": [], "columns": ["theta", "df", "neff"]}
    assert info["iter"] == len(old["history"]) + 1
    assert np.all(np.isfinite(info["coef"]))
    assert all(math.isfinite(info[key]) for key in ("plik", "loglik", "df", "trH"))
    return {
        "theta": 1.0,
        "done": info["iter"] == 2,
        "columns": old["columns"],
        "history": [*old["history"], [old["theta"], info["df"], info["neff"]]],
    }


def fit_aft(value, **kwargs):
    return r.survpenal_fit(X, Y, dist="gaussian", pcols=[[1]], pattr=[value], **kwargs)


def test_controller_search_agrees_with_fixed_final_penalty():
    value = regression.CoxPenalty.controlled(penalty, controller, needs_df=True)
    expected = fit_aft(regression.CoxPenalty.ridge(theta=1, scale=False))
    for _ in range(2):
        fit = fit_aft(value)
        np.testing.assert_allclose(fit.coefficients, expected.coefficients, atol=2e-8)
        np.testing.assert_allclose(fit.var, expected.var, atol=2e-8)
        assert fit.iter[0] == 2
        history = next(iter(fit.history.values()))
        assert history.done
        assert history.theta == 1
        assert history.columns == ["theta", "df", "neff"]
        np.testing.assert_allclose(np.asarray(history.history)[:, 0], [0.25, 1])
    with ThreadPoolExecutor(2) as pool:
        fits = list(pool.map(lambda _: fit_aft(value), range(2)))
    assert fits[0].coefficients == fits[1].coefficients
    restored = pickle.loads(pickle.dumps(value))  # noqa: S301 - own round-trip state
    np.testing.assert_allclose(fit_aft(restored).coefficients, fit.coefficients)


def test_cox_controller_uses_event_count_and_shared_outer_search():
    value = regression.CoxPenalty.controlled(penalty, controller, needs_df=True)
    kwargs = {"time": Y[:, 0], "status": Y[:, 1].astype(int), "x": X[:, 1:], "pcols": [[0]]}
    actual = regression.coxpenal_fit(**kwargs, penalties=[value])
    expected = regression.coxpenal_fit(
        **kwargs, penalties=[regression.CoxPenalty.ridge(theta=1, scale=False)]
    )
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=2e-8)
    np.testing.assert_allclose(actual.var, expected.var, atol=2e-8)
    assert actual.iter[0] == 2
    np.testing.assert_allclose(np.asarray(actual.history[0].history)[:, 2], sum(Y[:, 1]))
    assert pickle.loads(pickle.dumps(actual)).coefficients == actual.coefficients  # noqa: S301


def fixed_controller(old, info):
    if info["iter"]:
        assert math.isnan(info["df"])
        assert math.isnan(info["trH"])
    return {"theta": 1, "done": True}


@pytest.mark.parametrize("solver", ["cox", "aft"])
def test_full_matrix_penalty_derivatives(solver):
    def full_penalty(coef, theta, neff):
        return {**penalty(coef, theta, neff), "second": np.eye(len(coef)).ravel(order="F")}

    custom = regression.CoxPenalty.controlled(full_penalty, fixed_controller, diag=False)
    ridge = regression.CoxPenalty.ridge(theta=1, scale=False)
    design = np.column_stack((X, np.arange(12) % 2))
    if solver == "aft":

        def fit(value):
            return r.survpenal_fit(design, Y, dist="gaussian", pcols=[[1, 2]], pattr=[value])
    else:

        def fit(value):
            return regression.coxpenal_fit(
                Y[:, 0], Y[:, 1].astype(int), design[:, 1:], [value], [[0, 1]]
            )

    actual, expected = fit(custom), fit(ridge)
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-9)
    np.testing.assert_allclose(actual.var, expected.var, atol=1e-9)


@pytest.mark.parametrize("solver", ["cox", "aft"])
@pytest.mark.parametrize("flagged", [False, True])
def test_sparse_penalty_recentering_and_flags(solver, flagged):
    def frailty_penalty(coef, theta, neff):
        if flagged:
            return {"penalty": 0, "flag": True}
        centered = coef - np.mean(coef)
        return {**penalty(centered, theta, neff), "recenter": float(np.mean(coef))}

    custom = regression.CoxPenalty.controlled(frailty_penalty, fixed_controller, sparse=True)
    frailty = regression.CoxPenalty.frailty(
        distribution="gaussian", sparse=True, theta=0 if flagged else 1
    )
    design = np.column_stack((X, np.arange(12) % 3))
    if solver == "aft":

        def fit(value):
            return r.survpenal_fit(design, Y, dist="gaussian", pcols=[[2]], pattr=[value])
    else:

        def fit(value):
            return regression.coxpenal_fit(
                Y[:, 0], Y[:, 1].astype(int), design[:, 1:], [value], [[1]]
            )

    actual, expected = fit(custom), fit(frailty)
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-9)
    np.testing.assert_allclose(actual.frail, expected.frail, atol=1e-9)
    np.testing.assert_allclose(actual.fvar, expected.fvar, atol=1e-9)


def test_controller_state_is_released_with_compact_result():
    class Token:
        pass

    weak = []

    def search(old, info):
        if old is None:
            token = Token()
            weak.append(weakref.ref(token))
            return {"theta": 1, "token": token}
        return {**old, "done": True}

    fit = fit_aft(regression.CoxPenalty.controlled(penalty, search))
    gc.collect()
    assert weak
    assert all(value() is None for value in weak)
    assert pickle.loads(pickle.dumps(fit)).coefficients == fit.coefficients  # noqa: S301


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"theta": math.nan}, "theta must be finite"),
        ({"theta": [1, 2]}, "one number"),
        ({"theta": 1, "history": [[1, 2]], "columns": ["one"]}, "history rows"),
    ],
)
def test_invalid_controller_initial_state_is_rejected(bad, message):
    with pytest.raises(RuntimeError, match=message):
        fit_aft(regression.CoxPenalty.controlled(penalty, lambda old, info: bad))


def test_controller_errors_and_missing_done_are_reported():
    with pytest.raises(TypeError, match="must be callable"):
        regression.CoxPenalty.controlled(None, controller)
    with pytest.raises(RuntimeError, match="must return done"):
        fit_aft(regression.CoxPenalty.controlled(penalty, lambda old, info: {"theta": 1}))

    def failing(old, info):
        raise ValueError("custom search failure")

    with pytest.raises(RuntimeError, match="custom search failure"):
        fit_aft(regression.CoxPenalty.controlled(penalty, failing))


def test_fitting_variance_receives_null_fit_scale_and_survives_pickle():
    values = []
    distribution = custom_distribution()
    distribution["fitting_variance"] = lambda scale_squared: values.append(scale_squared) or 1.0
    actual = r.survpenal_fit(
        X,
        Y,
        dist=distribution,
        pcols=[[1]],
        pattr=[regression.CoxPenalty.ridge(theta=1, scale=False)],
    )
    expected = fit_aft(regression.CoxPenalty.ridge(theta=1, scale=False))
    assert len(values) == 1
    assert values[0] > 0
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-9)
    assert pickle.loads(pickle.dumps(actual)).coefficients == actual.coefficients  # noqa: S301


def fitting_variance(scale_squared, parms):
    assert scale_squared > 0
    assert parms == {"variance": 1.0}
    return parms["variance"]


def parameterized_init(y, weights, parms):
    return gaussian_init(y, weights)


def parameterized_density(z, parms):
    return gaussian_density(z)


def test_fitting_variance_distribution_pickle_and_validation():
    distribution = regression.SurvregDistribution.from_callbacks(
        "Gaussian",
        parameterized_init,
        parameterized_density,
        gaussian_density,
        gaussian_density,
        fitting_variance=fitting_variance,
        parms=[1.0],
        parm_names=["variance"],
    )
    # The hook survives independently of any fitted model.
    restored = pickle.loads(pickle.dumps(distribution))  # noqa: S301
    actual = r.survpenal_fit(
        X, Y, dist=restored, pcols=[[1]], pattr=[regression.CoxPenalty.ridge(theta=1, scale=False)]
    )
    expected = fit_aft(regression.CoxPenalty.ridge(theta=1, scale=False))
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-9)
    with pytest.raises(ValueError, match="Missing or invalid fitting_variance function"):
        regression.SurvregDistribution.from_callbacks(
            "Gaussian",
            gaussian_init,
            gaussian_density,
            gaussian_density,
            gaussian_density,
            fitting_variance=1,
        )
