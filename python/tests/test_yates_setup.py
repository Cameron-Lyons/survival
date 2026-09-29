"""Direct Yates prediction setup against stock R and corrected curve metadata."""

import gc
import json
import math
import pickle
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads((Path(__file__).parent / "fixtures/yates_setup_reference.json").read_text())


@pytest.fixture(scope="module")
def fits():
    result = {}
    for case in REFERENCE["cases"]:
        key = "/".join(case["name"].split("/")[:2])
        if key not in result:
            result[key] = r.coxph(case["formula"], case["data"], weights="wt", ties=case["ties"])
    return result


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_prepared_survival_predictions_and_summaries(case, fits):
    fit = fits["/".join(case["name"].split("/")[:2])]
    horizon = math.inf if case["rmean"] == "Inf" else case["rmean"]
    setup = r.yates_setup(fit, "survival", options={"rmean": horizon, "unused": True})
    assert isinstance(setup, r.YatesSurvivalSetup)
    assert list(setup) == ["predict", "summary"]
    prediction = setup["predict"](case["eta"])
    np.testing.assert_allclose(prediction, case["prediction"], rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(setup.predict(np.asarray(case["eta"])[:, None]), prediction)
    np.testing.assert_allclose(setup.predict(np.asarray(case["eta"])[None, :]), prediction)
    mean = np.asfortranarray(case["prediction"])
    variance = np.asfortranarray(case["variance"])
    summary = setup["summary"](mean, variance)
    np.testing.assert_allclose(summary.time, case["time"], atol=1e-14)
    for name, expected in case["corrected_summary"].items():
        np.testing.assert_allclose(getattr(summary, name), expected, rtol=2e-11, atol=2e-12)
    assert summary.std_chaz == summary.std_err
    # Keep the original R result as evidence of both documented metadata defects.
    assert np.asarray(case["raw_summary"]["surv"]).size == len(case["eta"]) * (
        len(case["time"]) + 1
    )
    np.testing.assert_allclose(case["raw_summary"]["cumhaz"], case["baseline_cumhaz"])
    if case["time"]:
        assert np.asarray(summary.surv).shape == (len(case["time"]), len(case["eta"]))
    else:
        assert summary.surv == []
    assert summary.ncurve == len(case["eta"])


@pytest.mark.parametrize("shape", [(), (3,), (3, 1), (1, 3), (2, 3)])
def test_risk_preserves_numeric_shape_and_nonfinite_values(shape, fits):
    setup = r.yates_setup(fits["right/efron"], "risk", options=object())
    values = np.linspace(-2, 2, num=int(np.prod(shape))).reshape(shape)
    assert isinstance(setup, r.YatesLinkPrediction)
    np.testing.assert_allclose(setup(values, object()), np.exp(values), rtol=2e-15)
    np.testing.assert_allclose(
        setup([-math.inf, 0, math.inf, math.nan]), [0, 1, math.inf, math.nan], equal_nan=True
    )


def inverse_link(name):
    return {
        "identity": lambda x: x,
        "log": np.exp,
        "logit": lambda x: 1 / (1 + np.exp(-x)),
        "probit": lambda x: np.vectorize(lambda t: 0.5 * math.erfc(-t / math.sqrt(2)))(x),
        "cloglog": lambda x: -np.expm1(-np.exp(x)),
        "cauchit": lambda x: 0.5 + np.arctan(x) / np.pi,
        "inverse": lambda x: 1 / x,
    }[name]


@pytest.mark.parametrize("case", REFERENCE["glm"], ids=lambda case: case["link"])
@pytest.mark.parametrize("protocol", ["r", "mapping", "statsmodels"])
def test_glm_family_inverse_links_match_r(case, protocol):
    inverse = inverse_link(case["link"])
    family = SimpleNamespace(linkinv=inverse) if protocol == "r" else {"linkinv": inverse}
    fit = SimpleNamespace(family=family)
    if protocol == "statsmodels":
        fit = SimpleNamespace(
            model=SimpleNamespace(family=SimpleNamespace(link=SimpleNamespace(inverse=inverse)))
        )
    assert r.yates_setup(fit) is None
    assert r.yates_setup(fit, "linear") is None
    callback = r.yates_setup(fit, "res")
    np.testing.assert_allclose(callback(case["eta"]), case["expected"], rtol=2e-14, atol=2e-16)
    matrix = np.asarray(case["eta"]).reshape(2, 3)
    np.testing.assert_allclose(
        callback(matrix), np.asarray(case["expected"]).reshape(2, 3), rtol=2e-14
    )
    with pytest.raises(ValueError, match="terms not yet supported"):
        r.yates_setup(fit, "terms")


def test_dispatch_options_and_default_warnings(fits):
    for choice in (None, "lp", "linear"):
        assert r.yates_setup(fits["right/efron"], choice, options=object()) is None
    for choice in ("expected", "terms"):
        with pytest.raises(ValueError, match="type expected is not supported"):
            r.yates_setup(fits["right/efron"], choice)
    with pytest.raises(ValueError, match="ambiguous"):
        r.yates_setup(fits["right/efron"], "l")
    with pytest.raises(TypeError, match="mapping"):
        r.yates_setup(fits["right/efron"], "survival", options=object())
    unknown = SimpleNamespace()
    assert r.yates_setup(unknown, "risk") is None
    assert r.yates_setup(unknown, type="link") is None
    with pytest.warns(UserWarning, match="no yates_setup method exists"):
        assert r.yates_setup(unknown, type="risk") is None
    assert survival.r_api.yates_setup is r.yates_setup


def test_setup_does_not_keep_fit_and_supports_pickle():
    fit = r.coxph("Surv(time, status) ~ age", survival.datasets.load_lung())
    weak = weakref.ref(fit)
    setup = r.yates_setup(fit, "survival")
    risk = r.yates_setup(fit, "risk")
    del fit
    gc.collect()
    assert weak() is None
    restored = pickle.loads(pickle.dumps(setup))  # noqa: S301 - own round-trip data
    np.testing.assert_array_equal(restored.predict([-0.3, 0.5]), setup.predict([-0.3, 0.5]))
    restored_risk = pickle.loads(pickle.dumps(risk))  # noqa: S301 - own round-trip data
    np.testing.assert_array_equal(restored_risk([-0.3, 0.5]), risk([-0.3, 0.5]))
    with pytest.raises(KeyError):
        setup["other"]


def test_shapes_strides_and_invalid_inputs(fits):
    setup = r.yates_setup(fits["right/efron"], "survival")
    eta = np.arange(8.0)[::2] / 10
    np.testing.assert_array_equal(setup.predict(eta), setup.predict(eta.tolist()))
    assert setup.predict([]).shape == (0, len(setup._baseline.time) + 2)
    with pytest.raises(ValueError, match="single-row/column"):
        setup.predict(np.ones((2, 2)))
    with pytest.raises(TypeError, match="numeric"):
        setup.predict(["bad"])
    with pytest.raises(ValueError, match="columns"):
        setup.summary([[1, 2]], [[0, 0]])
    with pytest.raises(ValueError, match="matching matrices"):
        setup.summary(setup.predict([0, 1]), np.zeros((1, len(setup._baseline.time) + 2)))
    with pytest.raises(ValueError, match="NaN"):
        r.yates_setup(fits["right/efron"], "survival", options={"rmean": math.nan})
    stratified = r.coxph("Surv(time, status) ~ age + strata(sex)", survival.datasets.load_lung())
    with pytest.raises(ValueError, match="stratified models"):
        r.yates_setup(stratified, "survival")


def test_empty_event_baseline_also_works_in_yates_simulation():
    fit = r.coxph(
        "Surv(time, status) ~ x",
        {"time": [1, 2, 3, 4], "status": [0, 0, 0, 0], "x": [0, 1, 0, 1]},
    )
    result = r.yates(fit, "x", levels=[0, 1], predict="survival", nsim=3)
    assert result.estimate["pmm"] == [0, 0]
    assert result.estimate["std"] == [0, 0]
    assert result.summary.time == []
    assert result.summary.surv == []
    assert result.summary.std_chaz == []
    assert result.summary.ncurve == 2
