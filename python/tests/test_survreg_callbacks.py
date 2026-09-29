"""Runtime AFT families through the public native and R-style Python APIs.

The asymmetric mixture, all censoring types and fixed/estimated/stratified
scales are independently referenced by generate_survreg_density_reference.R.
No SciPy dependency is needed for these user callbacks.
"""

# Pickles below are produced locally by these tests.
# ruff: noqa: S301

import copy
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r_api
core = survival._survival
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/survreg_density_reference.json").read_text()
)
PARMS = {"mix": 0.35, "shift": 1.4, "sd": 0.8}


def density(z, parms=PARMS):
    z = np.asarray(z)
    w, shift, sd = (parms[key] for key in ("mix", "shift", "sd"))
    v = (z - shift) / sd
    a = (1 - w) * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    b = w * np.exp(-v * v / 2) / (math.sqrt(2 * math.pi) * sd)
    pdf = a + b

    def tail(z):
        return np.array([math.erfc(x / math.sqrt(2)) / 2 for x in z])

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.column_stack(
            (
                (1 - w) * tail(-z) + w * tail(-v),
                (1 - w) * tail(z) + w * tail(v),
                pdf,
                (-z * a - v * b / sd) / pdf,
                ((z * z - 1) * a + (v * v - 1) * b / sd**2) / pdf,
            )
        )


def init(y, weights, parms=PARMS):
    mu = np.average(y, weights=weights)
    return np.array([mu, np.average((y - mu) ** 2, weights=weights)])


def variance(parms=PARMS):
    w, shift, sd = (parms[key] for key in ("mix", "shift", "sd"))
    return 1 - w + w * sd**2 + w * (1 - w) * shift**2


def quantile(p, parms=PARMS):
    p = np.asarray(p)
    lo, hi = np.full(len(p), -20.0), np.full(len(p), 20.0)
    for _ in range(65):
        mid = (lo + hi) / 2
        below = density(mid, parms)[:, 0] < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(
        p == 0,
        -np.inf,
        np.where(p == 1, np.inf, np.where((p > 0) & (p < 1), (lo + hi) / 2, np.nan)),
    )


def _root(lo, hi, fun):
    negative = fun(lo) < 0
    for _ in range(65):
        mid = (lo + hi) / 2
        if (fun(mid) < 0) == negative:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def deviance(y, scale, parms=PARMS):
    status = y[:, -1]
    center, loglik = y[:, 0].copy(), np.zeros(len(y))
    mode = _root(-2, 3, lambda z: density([z], parms)[0, 3])
    exact = status == 1
    center[exact] -= scale[exact] * mode
    loglik[exact] = math.log(density([mode], parms)[0, 2]) - np.log(scale[exact])
    for i in np.flatnonzero(status == 3):
        center[i] = _root(
            y[i, 0] - 8 * scale[i],
            y[i, 1] + 8 * scale[i],
            lambda eta, i=i: -np.diff(density((y[i, :2] - eta) / scale[i], parms)[:, 2])[0],
        )
        d = density((y[i, :2] - center[i]) / scale[i], parms)
        loglik[i] = math.log(d[1, 0] - d[0, 0])
    return {"center": center, "loglik": loglik}


def dtrans(y):
    return 1 / np.sqrt(1 + y * y)


def definition(**overrides):
    return dict(
        name="Two normal mixture",
        density=density,
        init=init,
        variance=variance,
        deviance=deviance,
        quantile=quantile,
        **overrides,
    )


def frame(*, transformed=False, intervals=True):
    data = {key: np.array(value) for key, value in REFERENCE["data"].items()}
    if transformed:
        for key in ("y1", "y2"):
            data[key] = np.sinh(data[key])
    if not intervals:
        data["status"][data["status"] == 3] = 1
    return data


def fit_case(
    case,
    *,
    transformed=False,
    intervals=True,
    penalized=False,
    robust=False,
    initialized=True,
    dist=None,
    **extra,
):
    data = frame(transformed=transformed, intervals=intervals)
    covariate = "ridge(x, theta=0.7, scale=False)" if penalized else "x"
    formula = f"Surv(y1, y2, status, type='interval') ~ {covariate} + offset(offset)"
    if case["nstrat"] == 2:
        formula += " + strata(g)"
    if dist is None:
        dist = definition()
        if transformed:
            dist.update(trans=np.arcsinh, dtrans=dtrans, itrans=np.sinh)
    return r.survreg(
        formula,
        data,
        dist=dist,
        weights=data["weights"],
        init=case["init"][: 2 + case["nstrat"]] if initialized else None,
        scale=0.9 if case["nstrat"] == 0 else 0,
        robust=robust,
        cluster=np.arange(40) % 10 if robust else None,
        control=r.survreg_control(maxiter=50, rel_tolerance=1e-11),
        **extra,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_fit_and_postfit_match_r(case, transformed, initialized):
    fit = fit_case(case, transformed=transformed, initialized=initialized)
    np.testing.assert_allclose(
        fit.coefficients + list(np.log(fit.scale)), case["expected"]["parameters"], atol=2e-7
    )
    np.testing.assert_allclose(fit.var, case["expected"]["variance"], atol=2e-7)
    assert fit.distribution.family == core.SurvregFamily.Custom
    assert fit.distribution.transform == (
        core.SurvregTransform.Custom if transformed else core.SurvregTransform.Identity
    )
    expected = case["transformed"] if transformed else case
    assert fit.loglik[1] == pytest.approx(
        expected["loglik"][1] if transformed else case["expected"]["loglik"], abs=1e-8
    )
    for kind in ("response", "quantile"):
        prediction = r.predict(fit, type=kind, p=[0.1, 0.5, 0.9], se_fit=True)
        np.testing.assert_allclose(prediction.fit, expected["postfit"][kind]["fit"], atol=2e-7)
        np.testing.assert_allclose(
            prediction.se_fit, expected["postfit"][kind]["se.fit"], atol=2e-7
        )
    for kind in ("response", "deviance", "working"):
        np.testing.assert_allclose(
            r.residuals(fit, type=kind), expected["postfit"]["residuals"][kind], atol=2e-7
        )
    actual = np.asarray(r.residuals(fit, type="matrix"))
    desired = np.asarray(expected["postfit"]["residuals"]["matrix"])
    np.testing.assert_allclose(actual[:, :3], desired[:, :3], atol=2e-7)
    noninterval = np.asarray(REFERENCE["data"]["status"]) != 3
    np.testing.assert_allclose(actual[noninterval], desired[noninterval], atol=2e-7)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_clustered_robust_callback_fit(case):
    fit = fit_case(case, intervals=False, robust=True)
    expected = case["robust_no_interval"]
    np.testing.assert_allclose(
        fit.coefficients + list(np.log(fit.scale)), expected["parameters"], atol=1e-8
    )
    np.testing.assert_allclose(fit.var, expected["variance"], atol=1e-8)
    np.testing.assert_allclose(r.residuals(fit, type="dfbeta"), expected["residuals"], atol=1e-8)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("intervals", [False, True])
def test_penalized_custom_family(case, intervals):
    fit = fit_case(case, intervals=intervals, penalized=True)
    expected = case["penalized_interval" if intervals else "penalized_no_interval"]
    np.testing.assert_allclose(
        fit.coefficients + list(np.log(fit.scale)),
        expected["parameters"],
        atol=2e-6 if intervals else 1e-8,
    )
    assert fit.loglik[1] - fit.penalty[1] == pytest.approx(expected["loglik"], abs=1e-8)


@pytest.mark.parametrize("penalized", [False, True])
@pytest.mark.parametrize(
    "roundtrip", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_models_keep_callbacks_after_copy_and_pickle(penalized, roundtrip):
    fit = fit_case(REFERENCE["cases"][1], transformed=True, penalized=penalized)
    restored = roundtrip(fit)
    for kind in ("response", "quantile"):
        np.testing.assert_allclose(r.predict(restored, type=kind), r.predict(fit, type=kind))
    for kind in ("response", "deviance", "matrix"):
        np.testing.assert_allclose(r.residuals(restored, type=kind), r.residuals(fit, type=kind))
    distribution = roundtrip(fit.distribution)
    np.testing.assert_allclose(
        r.qsurvreg([0.1, 0.5, 0.9], 0.3, distribution=distribution),
        r.qsurvreg([0.1, 0.5, 0.9], 0.3, distribution=fit.distribution),
    )


def test_registry_parameters_and_derived_distributions(monkeypatch):
    monkeypatch.setitem(r.survreg_distributions, "mixture", definition(parms=PARMS))
    base = fit_case(REFERENCE["cases"][0], dist="mixt", parms={"shift": 1.4})
    assert base.distribution.parm_names == list(PARMS)
    assert base.distribution.parms == list(PARMS.values())
    derived = {
        "name": "asinh mixture",
        "dist": "mixture",
        "trans": np.arcsinh,
        "dtrans": dtrans,
        "itrans": np.sinh,
    }
    transformed = fit_case(REFERENCE["cases"][0], transformed=True, dist=derived)
    np.testing.assert_allclose(transformed.coefficients, base.coefficients, atol=1e-8)
    changed = r.qsurvreg([0.2, 0.8], 0, distribution="MIXTURE", parms={"shift": 2.0})
    np.testing.assert_allclose(changed, quantile([0.2, 0.8], PARMS | {"shift": 2.0}))
    with pytest.raises(ValueError, match="Invalid parameter names"):
        fit_case(REFERENCE["cases"][0], dist="mixture", parms={"invalid": 3})
    with pytest.raises(ValueError, match="wrong number"):
        base.distribution.with_parms([0.1])
    with pytest.raises(ValueError, match="Distribution not found"):
        r.qsurvreg([0.5], 0, distribution="mixt")
    assert r.qsurvreg([0.1, 0.9], 0, distribution="T", parms=6) == r.qsurvreg(
        [0.1, 0.9], 0, distribution="t", parms=6
    )


def test_builtin_family_with_arbitrary_transform():
    dlist = {
        "name": "asinh normal",
        "dist": "gaussian",
        "trans": np.arcsinh,
        "dtrans": dtrans,
        "itrans": np.sinh,
    }
    transformed = fit_case(REFERENCE["cases"][0], transformed=True, dist=dlist)
    baseline = fit_case(REFERENCE["cases"][0], dist="gaussian")
    np.testing.assert_allclose(transformed.coefficients, baseline.coefficients, atol=1e-9)
    np.testing.assert_allclose(r.predict(transformed), np.sinh(r.predict(baseline)), atol=1e-9)
    restored = pickle.loads(pickle.dumps(transformed))
    np.testing.assert_allclose(r.residuals(restored), r.residuals(transformed))


class RecordedDensity:
    def __init__(self):
        self.shapes = []

    def __call__(self, z):
        assert isinstance(z, np.ndarray)
        self.shapes.append(z.shape)
        return density(z)


def test_density_calls_are_batches_through_fitting_postfit_and_dpqr():
    callback = RecordedDensity()
    dlist = definition()
    dlist["density"] = callback
    fit = fit_case(REFERENCE["cases"][0], dist=dlist)
    # Ten probe points, 40 response rows, or 48 endpoints including intervals.
    assert set(callback.shapes) <= {(10,), (40,), (48,)}
    callback.shapes.clear()
    r.residuals(fit, type="matrix")
    assert callback.shapes == [(48,)]
    callback.shapes.clear()
    z = np.linspace(-3, 3, 100)
    np.testing.assert_allclose(r.dsurvreg(z, 0, distribution=fit.distribution), density(z)[:, 2])
    assert callback.shapes == [(100,)]
    callback.shapes.clear()
    np.testing.assert_allclose(r.psurvreg(z, 0, distribution=fit.distribution), density(z)[:, 0])
    assert callback.shapes == [(100,)]
    p = np.linspace(0.01, 0.99, 20)
    np.testing.assert_allclose(
        r.psurvreg(r.qsurvreg(p, 0.3, 1.2, fit.distribution), 0.3, 1.2, fit.distribution),
        p,
        atol=1e-14,
    )
    gaussian_draws = r.rsurvreg(10, 0, distribution="gaussian", seed=81)
    uniforms = r.psurvreg(gaussian_draws, 0, distribution="gaussian")
    np.testing.assert_allclose(
        r.rsurvreg(10, 0, distribution=fit.distribution, seed=81), quantile(uniforms), atol=1e-14
    )
    assert math.isnan(r.dsurvreg([math.nan], 0, distribution=fit.distribution)[0])
    assert r.qsurvreg([0, 1], 0, distribution=fit.distribution) == [-math.inf, math.inf]
    assert r.rsurvreg(0, 0, distribution=fit.distribution, seed=81) == []


@pytest.mark.parametrize(
    ("name", "bad", "match"),
    [
        ("density", lambda z: np.zeros((len(z), 4)), "five-column"),
        ("density", lambda z: density(z[:-1]), "expected 10"),
        ("density", lambda z: np.full((len(z), 5), math.nan), "invalid probabilities"),
        ("init", None, "Missing or invalid init"),
    ],
)
def test_dtest_reports_bad_callbacks(name, bad, match):
    dlist = definition()
    dlist[name] = bad
    assert r.survregDtest(dlist) is False
    assert any(match in issue for issue in r.survregDtest(dlist, verbose=True))


@pytest.mark.parametrize(
    ("key", "bad", "match"),
    [
        ("itrans", np.arcsinh, "not inverses"),
        ("dtrans", lambda y: [-1.0] * len(y), "positive"),
        ("dtrans", lambda y: [], "expected 10"),
    ],
)
def test_dtest_probes_arbitrary_transforms(key, bad, match):
    dlist = definition(trans=np.arcsinh, dtrans=dtrans, itrans=np.sinh)
    dlist[key] = bad
    assert any(match in issue for issue in r.survregDtest(dlist, verbose=True))


def test_errors_propagate_from_initialization_and_postfit():
    dlist = definition()
    dlist["init"] = lambda y, weights: [0, math.nan]
    with pytest.raises(ValueError, match="finite location"):
        fit_case(REFERENCE["cases"][0], dist=dlist)
    dlist = definition()
    dlist["quantile"] = lambda p: []
    dlist["deviance"] = lambda y, s: ([], [])
    fit = fit_case(REFERENCE["cases"][0], dist=dlist)
    with pytest.raises(ValueError, match="quantile callback returned"):
        r.predict(fit, type="quantile")
    with pytest.raises(ValueError, match="deviance center callback returned"):
        r.residuals(fit, type="response")
    dlist = definition()
    del dlist["variance"]
    with pytest.raises(ValueError, match="no variance callback"):
        fit_case(REFERENCE["cases"][0], dist=dlist, penalized=True)


def test_python_callback_exception_keeps_method_and_original_message():
    def stopped(p):
        raise LookupError("custom quantile stopped")

    dlist = definition()
    dlist["quantile"] = stopped
    fit = fit_case(REFERENCE["cases"][0], dist=dlist)
    with pytest.raises(
        RuntimeError, match="quantile callback failed: LookupError: custom quantile stopped"
    ):
        r.predict(fit, type="quantile")


def test_custom_interval_residual_derivatives_match_finite_differences():
    case = REFERENCE["cases"][1]
    fit = fit_case(case)
    data = frame()
    deriv = np.asarray(r.residuals(fit, type="matrix"))
    h = 1e-4
    for i in np.flatnonzero(data["status"] == 3):
        eta = fit.linear_predictors[i]
        logs = math.log(fit.scale[data["g"][i]])

        def loglik(eta, logs, i=i):
            z = (np.array([data["y1"][i], data["y2"][i]]) - eta) / math.exp(logs)
            d = density(z)
            return math.log(d[0, 1] - d[1, 1] if z[0] > 0 else d[1, 0] - d[0, 0])

        g = loglik(eta, logs)
        de = (loglik(eta + h, logs) - loglik(eta - h, logs)) / (2 * h)
        dee = (loglik(eta + h, logs) - 2 * g + loglik(eta - h, logs)) / h**2
        ds = (loglik(eta, logs + h) - loglik(eta, logs - h)) / (2 * h)
        dss = (loglik(eta, logs + h) - 2 * g + loglik(eta, logs - h)) / h**2
        des = (
            loglik(eta + h, logs + h)
            - loglik(eta + h, logs - h)
            - loglik(eta - h, logs + h)
            + loglik(eta - h, logs - h)
        ) / (4 * h**2)
        np.testing.assert_allclose(deriv[i], [g, de, dee, ds, dss, des], atol=2e-6)


def test_unnamed_parameters_and_direct_native_fit_own_callbacks():
    class Shifted:
        def density(self, z, parms):
            assert isinstance(parms, np.ndarray)
            return density(z - parms[0])

        def quantile(self, p, parms):
            return quantile(p) + parms[0]

        def init(self, y, weights, parms):
            return init(y, weights)

        def deviance(self, y, scale, parms):
            result = deviance(y, scale)
            result["center"] -= scale * parms[0]
            return result

    callback = Shifted()
    dist = core.SurvregDistribution.from_callbacks(
        "shifted mixture",
        callback.init,
        callback.density,
        callback.deviance,
        callback.quantile,
        parms=[0.5],
    )
    data = frame()
    native = core.SurvregData(
        data["y1"],
        data["status"],
        np.column_stack([np.ones(40), data["x"]]),
        time2=data["y2"],
        weights=data["weights"],
        offset=data["offset"],
    )
    fit = core.survreg_fit(native, dist)
    del dist, callback
    baseline = fit_case(REFERENCE["cases"][0])
    assert fit.coefficients[0] == pytest.approx(
        baseline.coefficients[0] - 0.5 * baseline.scale[0], abs=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(fit.predict(predict_type="quantile").fit),
        r.predict(baseline, type="quantile"),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(fit.residuals().values)[:, 0], r.residuals(baseline), atol=1e-7
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 0.0])
def test_invalid_callback_variance_is_rejected(bad):
    dlist = definition()
    dlist["variance"] = lambda: bad
    with pytest.raises(ValueError, match="variance callback must return a finite positive value"):
        fit_case(REFERENCE["cases"][0], dist=dlist, penalized=True)


def test_callbacks_that_cannot_be_pickled_fail_explicitly():
    dlist = definition()
    dlist["density"] = lambda z: density(z)
    fit = fit_case(REFERENCE["cases"][0], dist=dlist)
    with pytest.raises((AttributeError, pickle.PicklingError), match="(local|lambda)"):
        pickle.dumps(fit)


def test_density_exception_inside_native_fitting_propagates():
    def bad_density(z):
        if len(z) > 10:
            raise ArithmeticError("density stopped on observation batch")
        return density(z)

    dlist = definition()
    dlist["density"] = bad_density
    assert r.survregDtest(dlist)
    with pytest.raises(
        RuntimeError, match="density callback failed: ArithmeticError: density stopped"
    ):
        fit_case(REFERENCE["cases"][0], dist=dlist)
