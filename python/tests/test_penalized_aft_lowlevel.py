"""Prepared penalized AFT fitting and reports against stock R survival."""

import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .test_aft_lowlevel import custom_distribution

survival = setup_survival_import()
r = survival.r
regression = survival.regression
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "penalized_aft_lowlevel_reference.json").read_text()
)


def arguments(case):
    kwargs = {
        **case["arguments"],
        "column_names": case["column_names"],
        "pcols": case["pcols"],
        "assign": case["assign"],
    }
    kwargs["controlvals"] = kwargs["controlvals"] or None
    kwargs["pattr"] = []
    for spec in case["specs"]:
        spec = dict(spec)
        kind = spec.pop("kind")
        value = (
            r.pspline(case["spline_x"], **spec)
            if kind == "pspline"
            else getattr(regression.CoxPenalty, kind)(**spec)
        )
        kwargs["pattr"].append(value)
    return kwargs


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_survpenal_fit_against_r(case):
    expected = case["expected"]
    if "error" in expected:
        assert case["name"] == "unused_stratum"
        assert "exactly singular" in expected["error"]
        # R tries to invert the unused scale's zero covariance row. The port
        # retains it and agrees with R's otherwise identical two-stratum fit.
        compact = next(c for c in REFERENCE["cases"] if c["name"] == "gaussian_strata")
        assert case["x"] == compact["x"]
        assert case["specs"] == compact["specs"]
        fit = r.survpenal_fit(case["x"], case["y"], **arguments(case))
        for name in ("coefficients", "icoef", "score"):
            values = getattr(fit, name)
            np.testing.assert_allclose(values[:-1], compact["expected"][name], atol=3e-8)
        for name in ("var", "var2"):
            values = np.asarray(getattr(fit, name))
            np.testing.assert_allclose(values[:-1, :-1], compact["expected"][name], atol=3e-8)
            np.testing.assert_array_equal(values[-1], 0)
        for name in ("df", "loglik", "linear_predictors"):
            np.testing.assert_allclose(getattr(fit, name), compact["expected"][name], atol=3e-8)
        assert fit.score[-1] == 0
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = r.survpenal_fit(case["x"], case["y"], **arguments(case))
    assert [str(w.message) for w in caught] == case["warnings"]
    for name in (
        "coefficients",
        "icoef",
        "var",
        "var2",
        "loglik",
        "df",
        "penalty",
        "score",
        "frail",
        "fvar",
    ):
        if expected[name] is None:
            assert getattr(fit, name) is None
        else:
            np.testing.assert_allclose(
                getattr(fit, name),
                np.asarray(expected[name], dtype=float),
                rtol=3e-7,
                atol=3e-8,
                err_msg=name,
            )
    # The established offset and aliased-coefficient corrections also apply to
    # bare fits. R omits dense offsets and propagates NA through every row.
    lp = np.asarray(expected["linear_predictors"], dtype=float)
    if case["name"] == "alias":
        lp = np.asarray(case["x"]) @ np.nan_to_num(fit.coefficients[:-1])
    elif fit.frail is None and case["arguments"]["offset"] is not None:
        lp += case["arguments"]["offset"]
    np.testing.assert_allclose(fit.linear_predictors, lp, rtol=3e-7, atol=3e-8)
    assert fit.iter == expected["iter"]
    assert fit.coefficient_names == tuple(expected["coefficient_names"])
    assert fit.pterms == expected["pterms"]
    assert fit.assign2 == expected["assign2"]
    assert fit.df2 is None
    assert set(fit.history) == set(expected["history"])
    for label, target in expected["history"].items():
        value = fit.history[label]
        assert value.done == target["done"]
        # Native history schemas remain available even when R omits names
        # for a fixed gamma penalty's empty search history.
        columns = target["columns"] or (
            ["theta", "loglik", "c.loglik"] if "gamma" in case["name"] else []
        )
        assert value.columns == columns
        for name in ("theta", "history", "c_loglik", "half"):
            if target[name] is None:
                assert getattr(value, name) is None
            else:
                np.testing.assert_allclose(getattr(value, name), target[name], rtol=3e-7, atol=3e-8)
    for terms, lines in zip((False, True), expected["print"], strict=True):
        report = r.print_survreg_penal(fit, terms=terms)
        if case["name"] == "alias":
            # Aliased ridge cancellation leaves coefficients near machine
            # zero. R and Rust print different tiny values at three digits.
            assert report.lines[6:] == lines[6:]
        else:
            assert report.lines == lines
            assert str(report) == "\n".join(lines) + "\n"


def test_callbacks_and_inputs_are_not_retained():
    case = REFERENCE["cases"][0]
    x = np.array(case["x"])
    kwargs = arguments(case)
    distribution = custom_distribution()
    distribution["variance"] = lambda *args: 1.0
    fit = r.survpenal_fit(x, case["y"], **{**kwargs, "dist": distribution})
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own test data
    expected = r.survpenal_fit(x, case["y"], **{**kwargs, "dist": "gaussian"})
    np.testing.assert_allclose(fit.coefficients, expected.coefficients, atol=1e-9)
    x[:] = 100
    fit.coefficients[0] = 100
    fit.var[0][0] = 100
    fit.assign2["ridge"][0] = 100
    assert fit.coefficients == restored.coefficients
    assert fit.var == restored.var
    assert fit.assign2 == restored.assign2
    assert not hasattr(fit._fit, "covariates")
    assert not hasattr(fit._fit, "distribution")
    assert not hasattr(fit._fit, "frail_index")


@pytest.mark.parametrize("layout", ["C", "F", "strided", "mapping", "frame"])
def test_default_terms_and_matrix_layout(layout):
    case = REFERENCE["cases"][0]
    x = np.array(case["x"], order="F" if layout == "F" else "C")
    if layout == "strided":
        backing = np.zeros((len(x), x.shape[1] * 2))
        backing[:, ::2] = x
        x = backing[:, ::2]
    if layout in {"mapping", "frame"}:
        x = dict(zip(case["column_names"], x.T, strict=True))
        if layout == "frame":
            x = pytest.importorskip("pandas").DataFrame(x)
    kwargs = arguments(case)
    kwargs.pop("assign")
    fit = r.survpenal_fit(x, case["y"], **kwargs)
    assert fit.assign2 == {"Intercept": [0], "term 2": [1, 2], "sigma": [3]}
    np.testing.assert_allclose(fit.coefficients, case["expected"]["coefficients"])


def test_zero_frailty_cox_report():
    fit = r.coxph(
        "Surv(time, event) ~ age + frailty(group, distribution='gaussian', theta=0, sparse=True)",
        {
            "time": [1, 3, 2, 6, 5, 4, 8, 7, 9],
            "event": [1] * 9,
            "age": [2, 1, 4, 3, 6, 4, 7, 3, 2],
            "group": [1, 2, 3] * 3,
        },
    )
    report = fit.summary()
    assert math.isnan(report["coefficients"][-1]["chisq"])


@pytest.mark.parametrize("sparse", [False, True])
def test_penalty_callbacks_release_their_state(sparse):
    case = REFERENCE["cases"][0]
    x = np.array(case["x"])
    if sparse:
        x[:, 2] = np.tile([10, 20, 30], 4)
    calls = []

    def penalty(coef, *, which):
        calls.append(which)
        return {
            "coef": coef,
            "first": -coef,
            "second": np.ones(len(coef)),
            "penalty": -0.5 * np.dot(coef, coef),
            "flag": False,
        }

    kwargs = {"pcols": [[2]] if sparse else [[1, 2]], "dist": "gaussian"}
    fit = r.survpenal_fit(
        x, case["y"], pattr=[regression.CoxPenalty.callback(penalty, sparse=sparse)], **kwargs
    )
    assert set(calls) == {1 if sparse else 2}
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own test data
    assert restored.coefficients == fit.coefficients
    assert restored.frail == fit.frail
    r.print_survreg_penal(restored)
    if not sparse:
        expected = r.survpenal_fit(
            x, case["y"], pattr=[regression.CoxPenalty.ridge(theta=1, scale=False)], **kwargs
        )
        np.testing.assert_allclose(fit.coefficients, expected.coefficients, atol=1e-10)
        np.testing.assert_allclose(fit.var, expected.var, atol=1e-10)


def test_spline_result_retains_only_print_metadata():
    case = next(case for case in REFERENCE["cases"] if case["name"] == "spline_fixed")
    kwargs = arguments(case)
    fit = r.survpenal_fit(case["x"], case["y"], **kwargs)
    before = r.print_survreg_penal(fit)
    kwargs["pattr"][0].basis[0][:] = [100] * len(kwargs["pattr"][0].basis[0])
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - own test data
    assert r.print_survreg_penal(restored).lines == before.lines
    assert not hasattr(restored._print_info[1], "basis")
    assert list(r.as_data_frame(before)) == ["term", *before.columns]


def test_native_cluster_guard_and_base_density_settings():
    case = REFERENCE["cases"][0]
    y = np.array(case["y"])
    distribution = regression.SurvregDistribution("gaussian")
    penalties = [regression.CoxPenalty.ridge(theta=1)]
    data = regression.SurvregData(
        y[:, 0], y[:, 1].astype(np.int32), case["x"], cluster=[0] * len(y)
    )
    with pytest.raises(ValueError, match="bare AFT fits do not compute robust variance"):
        regression.survpenal_fit_raw(data, distribution, penalties, [[1, 2]])

    def unused(values):
        pytest.fail("bare fitting invoked a transform")

    transformed = distribution.derived("Transformed", regression.SurvregTransform.Identity, 100)
    transformed = transformed.with_transform(unused, unused, unused)
    kwargs = arguments(case)
    expected = r.survpenal_fit(case["x"], case["y"], **{**kwargs, "dist": distribution})
    fit = r.survpenal_fit(case["x"], case["y"], **{**kwargs, "dist": transformed})
    assert fit.coefficients == expected.coefficients


def test_dense_penalty_after_sparse_term_matches_full_native_fit():
    case = REFERENCE["cases"][0]
    x = np.array(case["x"])
    x[:, 1] = np.tile([1, 2, 3], 4)
    y = np.array(case["y"])
    penalties = [
        regression.CoxPenalty.frailty(distribution="gaussian", theta=0.4, sparse=True),
        regression.CoxPenalty.ridge(theta=1),
    ]
    kwargs = {"pcols": [[1], [2]], "pattr": penalties, "dist": "gaussian", "scale": 1.5}
    raw = r.survpenal_fit(x, y, **kwargs)
    data = regression.SurvregData(y[:, 0], y[:, 1].astype(np.int32), x)
    full = regression.survpenal_fit(
        data, regression.SurvregDistribution("gaussian"), penalties, [[1], [2]], scale=1.5
    )
    np.testing.assert_allclose(raw.coefficients, full.coefficients)
    np.testing.assert_allclose(raw.var, full.var)
    np.testing.assert_allclose(raw.frail, full.frail)
    assert raw.nvar == 2
    assert raw.coefficient_names == ("x 1", "x 3")
    assert r.print_survreg_penal(raw).fixed_scale


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pattr": []}, "Invalid pcols or pattr"),
        ({"pcols": [[-1]]}, "zero-based"),
        ({"pcols": [[1.5]]}, "integer"),
        ({"pcols": [[4]]}, "outside x"),
        ({"assign": [[0], [1], [2]]}, "disagree"),
        ({"assign": [[0], [1, 2], [0]]}, "unique"),
        ({"assign": {"sigma": [0], "ridge": [1, 2]}}, "reserve sigma"),
        ({"assign": {1: [0], 2: [1, 2]}}, "strings"),
        ({"pattr": ["ridge"]}, "CoxPenalty or PsplineResult"),
        ({"scale": -1}, "Invalid scale"),
        ({"scale": 1, "nstrat": 2}, "fixed scale and strata"),
        ({"nstrat": 2}, "Invalid strata"),
        ({"weights": [0] * 12}, "Invalid weights"),
        ({"dist": "weibull"}, "Missing density"),
        ({"dist": "t"}, "explicit parms"),
        ({"init": [1]}, "initial|init"),
    ],
)
def test_invalid_prepared_inputs(kwargs, message):
    case = REFERENCE["cases"][0]
    default = {"pcols": [[1, 2]], "pattr": [regression.CoxPenalty.ridge(theta=1)]}
    with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
        r.survpenal_fit(case["x"], case["y"], **{**default, **kwargs})
