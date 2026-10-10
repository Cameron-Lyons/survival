"""DPQR staged recycling, exceptional arithmetic and callbacks against stock R."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
r = survival.r_api
NA_REAL = np.array([0x7FF80000000007A2], dtype=np.uint64).view(np.float64)[0]
FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "survreg_dpqr_reference.json").read_text())
CALLBACKS = json.loads((FIXTURES / "survreg_dpqr_callback_reference.json").read_text())["cases"]
EDGES = json.loads((FIXTURES / "survreg_dpqr_callback_edges.json").read_text())["cases"]
EMPTY = json.loads((FIXTURES / "survreg_dpqr_empty_reference.json").read_text())["cases"]
METHODS = {
    "dsurvreg": "pdf_values",
    "psurvreg": "cdf_values",
    "qsurvreg": "quantile_values",
    "rsurvreg": "sample",
}


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, str):
        return {"NA": NA_REAL, "NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}[value]
    return value


def assert_values(actual, expected):
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, decode(expected), rtol=1e-12, atol=1e-12, equal_nan=True)


def invoke(interface, method, query, mean, scale, distribution, parms=None):
    kwargs = {"seed": 123} if method == "rsurvreg" else {}
    if interface == "facade":
        return getattr(r, method)(query, mean, scale, distribution, parms, **kwargs)
    if interface == "named":
        return getattr(core, method)(query, mean, scale, distribution, parms, **kwargs)
    dist = core.SurvregDistribution(distribution, parms)
    return getattr(dist, METHODS[method])(query, mean, scale, **kwargs)


@pytest.mark.parametrize("interface", ["named", "method", "facade"])
@pytest.mark.parametrize(
    "case", REFERENCE["builtins"] + REFERENCE["exceptions"], ids=lambda c: c["name"]
)
def test_builtin_against_stock_stages(case, interface):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = invoke(
            interface,
            case["method"],
            decode(case["query"]),
            decode(case["mean"]),
            decode(case["scale"]),
            case["distribution"],
            [5] if case["distribution"] == "t" else None,
        )
    assert_values(result, case["expected"]["values"])
    assert [str(w.message) for w in caught] == [
        e["message"] for e in case["expected"]["events"] if e["kind"] == "warning"
    ]


def callback_definition(events, mode="ordinary", derived=False):
    base = core.SurvregDistribution("gaussian")

    def event(kind, values):
        events.append({"kind": kind, "values": list(values)})

    def density(z):
        event("density", z)
        if mode == "density_error":
            raise ValueError("density stopped")
        if mode == "callback_warnings":
            warnings.warn("density warning", RuntimeWarning, stacklevel=2)
        with np.errstate(all="ignore"):
            result = np.column_stack(
                (
                    base.cdf_values(z, [0], [1]),
                    base.cdf_values(-z, [0], [1]),
                    base.pdf_values(z, [0], [1]),
                    -z,
                    z * z - 1,
                )
            )
        if mode == "density_unused_invalid":
            result[:, [1, 3, 4]] = np.nan
        elif mode == "density_negative":
            result[:, 2] *= -1
        elif mode == "density_nan":
            result[:, :3] = np.nan
        elif mode == "density_short":
            result = result[:1]
        return result

    def quantile(p):
        if mode in ("quantile_error_unforced", "quantile_constant_unforced"):
            events.append({"kind": "quantile_unforced"})
            if mode == "quantile_error_unforced":
                raise ValueError("quantile stopped")
            return [0.5]
        event("quantile", p)
        if mode == "quantile_error":
            raise ValueError("quantile stopped")
        if mode == "callback_warnings":
            warnings.warn("quantile warning", RuntimeWarning, stacklevel=2)
        result = base.quantile_values(p, [0], [1])
        return result[:1] if mode == "quantile_short" else result

    def transform(x):
        event("transform", x)
        if mode == "transform_error":
            raise ValueError("transform stopped")
        if mode == "callback_warnings":
            warnings.warn("transform warning", RuntimeWarning, stacklevel=2)
        if any(v < 0 for v in x):
            warnings.warn("NaNs produced", RuntimeWarning, stacklevel=2)
        with np.errstate(all="ignore"):
            result = np.log(x)
        return np.append(result, 0.7) if mode == "transform_long" else result

    def derivative(x):
        event("derivative", x)
        if mode == "derivative_error":
            raise ValueError("derivative stopped")
        if mode == "callback_warnings":
            warnings.warn("derivative warning", RuntimeWarning, stacklevel=2)
        with np.errstate(all="ignore"):
            return 1 / x

    def inverse(x):
        event("inverse", x)
        if mode == "inverse_error":
            raise ValueError("inverse stopped")
        if mode == "callback_warnings":
            warnings.warn("inverse warning", RuntimeWarning, stacklevel=2)
        with np.errstate(all="ignore"):
            result = np.exp(x)
        return result[:1] if mode == "inverse_short" else result

    definition = {
        "name": "DPQR callback oracle",
        "init": lambda y, weights: [0, 1],
        "density": density,
        "quantile": quantile,
        "deviance": lambda y, scale: (np.zeros(len(y)), np.zeros(len(y))),
    }
    if derived:
        definition.update(trans=transform, dtrans=derivative, itrans=inverse)
    return definition


def run_callback_case(case):
    if "method" in case:
        family, method = case["distribution"], case["method"]
        query, mean, scale = decode(case["query"]), decode(case["mean"]), decode(case["scale"])
        mode = "ordinary"
    else:
        parts = case["name"].split("/")
        if parts[0] == "unknown_distribution":
            family, method, mode = "missing", parts[1], "ordinary"
            query, mean, scale = (3 if method == "rsurvreg" else [0.5]), [0], [1]
        elif parts[0] in (
            "audit_base",
            "audit_derived",
            "gaussian",
            "weibull",
            "logistic",
            "t",
            "extreme",
            "lognormal",
        ):
            family, method, empty = parts
            query = [-0.1, 0.5, 1.1, NA_REAL, math.nan, math.inf, -math.inf]
            mean, scale = ([] if empty == "mean" else [0]), ([] if empty == "scale" else [1])
            mode = "ordinary"
        else:
            mode, family, method = parts
            query, mean, scale = (
                (2 if method == "rsurvreg" else [0.2, 0.5]),
                [1, 2, 3],
                [1, 2, 3, 4, 5],
            )
    events = []
    distribution = (
        callback_definition(events, mode, family == "audit_derived")
        if family.startswith("audit_")
        else family
    )
    result = None
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, *_args, **_kwargs: events.append(
            {"kind": "warning", "message": str(message)}
        )
        if case["expected"]["error"] is not None:
            with pytest.raises(ValueError, match=case["expected"]["error"]):
                invoke(
                    "facade",
                    method,
                    query,
                    mean,
                    scale,
                    distribution,
                    [5] if family == "t" else None,
                )
        else:
            result = invoke(
                "facade", method, query, mean, scale, distribution, [5] if family == "t" else None
            )
    if result is not None:
        assert_values(result, case["expected"]["values"])
    expected_events = [e for e in case["expected"]["events"] if e["kind"] != "error"]
    assert len(events) == len(expected_events)
    for actual, expected in zip(events, expected_events, strict=True):
        assert actual["kind"] == expected["kind"]
        if "values" in expected:
            assert_values(actual["values"], expected["values"])
        if "message" in expected:
            assert actual["message"] == expected["message"]


@pytest.mark.parametrize("case", CALLBACKS + EDGES + EMPTY, ids=lambda c: c["name"])
def test_callback_order_and_raw_results_against_stock(case):
    run_callback_case(case)


@pytest.mark.parametrize("method", ["dsurvreg", "psurvreg", "qsurvreg"])
@pytest.mark.parametrize("argument", ["mean", "scale"])
def test_nullable_arguments_reach_numeric_kernel(method, argument):
    kwargs = {"mean": [0], "scale": [1], "distribution": "gaussian"}
    kwargs[argument] = [None, 1, math.nan]
    values = getattr(r, method)([0.2, 0.5, 0.8], **kwargs)
    assert math.isnan(values[0])
    assert math.isnan(values[2])


def test_warning_as_error_stops_before_density_callback():
    events = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="longer object length"):
            r.dsurvreg([0.2, 0.5], [1, 2, 3], [1], callback_definition(events, derived=True))
    assert [e["kind"] for e in events] == ["derivative", "transform"]
