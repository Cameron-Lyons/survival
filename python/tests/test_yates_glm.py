"""External GLM marginal means and simulated tests against stock R survival."""

import gc
import json
import math
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import setup_survival_import
from .test_yates_setup import inverse_link

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads((Path(__file__).parent / "fixtures/yates_glm_reference.json").read_text())


def logistic(eta):
    return 1 / (1 + np.exp(-eta))


def test_native_response_is_public_and_matches_risk_with_centering():
    native = survival.validation
    matrices = [np.asfortranarray([[1, 0], [1, 2]]), np.array([[1, 3]])]
    beta, variance, means = [0.2, 0.1], [[0.01, 0.002], [0.002, 0.02]], [1, 0.5]
    actual = native.yates_response(matrices, beta, variance, np.exp, means=means, seed=42)
    expected = native.yates_risk(matrices, beta, variance, means, seed=42)
    close([row.pmm for row in actual.estimate], [row.pmm for row in expected.estimate])
    close(actual.mvar, expected.mvar)
    with pytest.raises(TypeError, match="inverse_link must be callable"):
        native.yates_response(matrices, beta, variance, None)


def model(case, protocol="mapping"):
    inverse = np.square if case["link"] == "sqrt" else inverse_link(case["link"])
    if protocol == "mapping":
        family = {"linkinv": inverse}
    elif protocol == "r":
        family = SimpleNamespace(linkinv=inverse)
    else:
        family = SimpleNamespace(link=SimpleNamespace(inverse=inverse))
    return r.YatesModel(
        case["formula"],
        case["data"],
        [math.nan if v is None else v for v in case["beta"]],
        case["variance"],
        family=family,
        weights=case["weights"],
    )


def fit(case, value=None):
    return r.yates(
        model(case) if value is None else value,
        case["term"],
        levels=case["levels"],
        population=case["population"],
        test=case["test"],
        method=case["method"],
        predict=case["predict"],
        nsim=case["nsim"],
        options={"seed": case["seed"]},
    )


def close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=3e-8, atol=3e-10
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("protocol", ["mapping", "r", "statsmodels"])
def test_external_glm_matches_r(case, protocol):
    result = fit(case, model(case, protocol))
    assert list(result.estimate) == list(case["estimate"])
    for name, expected in case["estimate"].items():
        if name in ("pmm", "std"):
            close(result.estimate[name], expected)
        else:
            assert result.estimate[name] == expected
    for name in ("cmat", "mvar", "sas"):
        if case[name] is not None:
            close(getattr(result, name), case[name])
    close([[row.chisq, row.df] for row in result.test], case["tests"])
    assert [row.name for row in result.test] == case["test_names"]
    assert all(row.ss is None for row in result.test)
    assert result.summary is None


def test_callback_batches_levels_and_runs_once_per_draw():
    case = REFERENCE["cases"][0]
    batches = []

    def callback(eta):
        assert isinstance(eta, np.ndarray)
        batches.append(eta.copy())
        return logistic(eta)

    actual = fit(case, replace(model(case), family={"linkinv": callback}))
    assert len(batches) == case["nsim"] + 1
    assert all(batch.shape == (len(case["data"]["y"]) * 3,) for batch in batches)
    point = logistic(batches[0]).reshape(3, -1).mean(axis=1)
    close(actual.estimate["pmm"], point)
    simulated = np.array([logistic(batch).reshape(3, -1).mean(axis=1) for batch in batches[1:]])
    close(actual.mvar, np.cov(simulated, rowvar=False))
    # The point estimate averages the response at fitted coefficients.
    assert not np.allclose(actual.estimate["pmm"], simulated.mean(axis=0))


def test_link_predictions_skip_inverse_and_simulation_options():
    case = REFERENCE["cases"][0]

    def forbidden(eta):
        pytest.fail("linear prediction called the inverse link")

    value = replace(model(case), family={"linkinv": forbidden})
    expected = r.yates(replace(value, family=None), "a")
    for predict in (None, "link", "linear"):
        actual = r.yates(value, "a", predict=predict, nsim=0, options=object())
        assert actual.estimate == expected.estimate
        assert actual.mvar == expected.mvar


def test_results_release_model_and_callback_and_roundtrip():
    case = REFERENCE["cases"][0]

    class Link:
        def __call__(self, eta):
            return logistic(eta)

    link = Link()
    value = replace(model(case), family={"linkinv": link})
    refs = weakref.ref(link), weakref.ref(value)
    result = fit(case, value)
    del value, link
    gc.collect()
    assert all(ref() is None for ref in refs)
    restored = pickle.loads(pickle.dumps(result))  # noqa: S301 - own round-trip data
    assert restored.estimate == result.estimate
    assert restored.mvar == result.mvar
    value = replace(model(case), family={"linkinv": logistic})
    restored_model = pickle.loads(pickle.dumps(value))  # noqa: S301
    assert fit(case, restored_model).estimate == result.estimate


def test_repeated_and_concurrent_simulations_have_independent_rng():
    case = REFERENCE["cases"][0]
    value = replace(model(case), family={"linkinv": logistic})
    expected = fit(case, value)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: fit(case, value), range(2)))
    for result in results:
        assert result.estimate == expected.estimate
        assert result.mvar == expected.mvar
    assert r.yates(value, "a", predict="response", options={"seed": 12}).mvar != expected.mvar


@pytest.mark.parametrize(
    ("callback", "message"),
    [
        (lambda x: np.ones(len(x) - 1), "inverse link response"),
        (lambda x: np.full(len(x), math.nan), "inverse link response"),
        (lambda x: np.full(len(x), math.inf), "inverse link response"),
        (lambda x: np.ones((len(x), 2)), "1-dimensional array"),
        (lambda x: {"wrong": x}, "inverse link callback"),
    ],
)
def test_invalid_inverse_link_results(callback, message):
    value = replace(model(REFERENCE["cases"][0]), family={"linkinv": callback})
    with pytest.raises((RuntimeError, ValueError), match=message):
        r.yates(value, "a", predict="response", nsim=2)


def test_callback_error_during_simulation_keeps_its_message():
    calls = 0

    def callback(eta):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("my inverse link failed")
        return logistic(eta)

    value = replace(model(REFERENCE["cases"][0]), family={"linkinv": callback})
    with pytest.raises(RuntimeError, match="my inverse link failed"):
        r.yates(value, "a", predict="response", nsim=3)
    assert calls == 3


@pytest.mark.parametrize("family", [{}, {"linkinv": 1}, SimpleNamespace(link=None)])
def test_invalid_family_is_rejected(family):
    with pytest.raises(ValueError, match="family must supply a callable"):
        replace(model(REFERENCE["cases"][0]), family=family)


@pytest.mark.parametrize("weights", [[1], [-1] * 72, [math.nan] * 72, [0] * 72])
def test_invalid_model_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        replace(model(REFERENCE["cases"][0]), weights=weights)


def test_response_validation_and_sgtt_restrictions():
    value = model(REFERENCE["cases"][0])
    for kind in ("terms", "unknown"):
        with pytest.raises(ValueError, match="terms|invalid GLM"):
            r.yates(value, "a", predict=kind)
    with pytest.raises(ValueError, match="sgtt method only applies"):
        r.yates(value, "a", predict="response", method="sgtt")
    with pytest.raises(ValueError, match="nsim must be at least two"):
        r.yates(value, "a", predict="response", nsim=1)
    with pytest.raises(TypeError, match="mapping"):
        r.yates(value, "a", predict="response", options=object())
    with pytest.raises(TypeError, match="unrecognized response options"):
        r.yates(value, "a", predict="response", options={"rmean": 1})


def test_data_frame_population_matches_mapping():
    pd = pytest.importorskip("pandas")
    case = next(case for case in REFERENCE["cases"] if case["name"] == "explicit_population")
    expected = fit(case)
    actual = fit({**case, "population": pd.DataFrame(case["population"])})
    assert actual.estimate == expected.estimate
    assert actual.mvar == expected.mvar
    with pytest.raises(ValueError, match="sgtt method only applies"):
        r.yates(model(case), "a", population=pd.DataFrame(case["population"]), method="sgtt")


@pytest.mark.parametrize("population", [{}, {"b": []}, {"b": ["x"], "z": [0, 1]}])
def test_invalid_population_shape_is_rejected(population):
    with pytest.raises(ValueError, match="population"):
        r.yates(model(REFERENCE["cases"][0]), "a", predict="response", population=population)
