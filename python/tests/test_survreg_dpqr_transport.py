"""Immediate DPQR warnings and original callback exceptions cross PyO3 intact."""

import copy
import pickle
import warnings

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
core = survival._survival
r = survival.r_api


def definition(**overrides):
    def density(z):
        return np.column_stack((z * 0 + 0.5, z * 0 + 0.5, z * 0 + 0.2, -z, z * z - 1))

    return {
        "name": "DPQR exception transport",
        "init": lambda y, weights: [0, 1],
        "deviance": lambda y, scale: (np.zeros(len(y)), np.zeros(len(y))),
        "density": density,
        "quantile": lambda p: p,
        **overrides,
    }


def distribution(callbacks):
    return core.SurvregDistribution.from_callbacks(**callbacks)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("method", ["pdf_values", "cdf_values", "quantile_values", "sample"])
def test_query_callback_preserves_exception_instance(method, error_type):
    error = error_type("DPQR callback sentinel")

    def fail(_):
        raise error

    callback = "density" if method in ("pdf_values", "cdf_values") else "quantile"
    dist = distribution(definition(**{callback: fail}))
    query = 3 if method == "sample" else [0.2, 0.4, 0.6]
    with pytest.raises(error_type) as caught:
        getattr(dist, method)(query, [0], [1])
    assert caught.value is error


@pytest.mark.parametrize("callback", ["trans", "dtrans", "itrans"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_transform_callback_preserves_exception_instance(callback, error_type):
    error = error_type("DPQR transform sentinel")

    def fail(_):
        raise error

    transforms = {"trans": lambda x: x, "dtrans": lambda x: x * 0 + 1, "itrans": lambda x: x}
    transforms[callback] = fail
    dist = distribution(definition()).with_transform(**transforms)
    method = "quantile_values" if callback == "itrans" else "pdf_values"
    with pytest.raises(error_type) as caught:
        getattr(dist, method)([0.2, 0.4, 0.6], [0], [1])
    assert caught.value is error


@pytest.mark.parametrize("nested", ["dtest", "query"])
def test_caught_nested_callback_error_does_not_replace_later_warning(nested):
    calls = []
    error = LookupError("caught inner callback")

    def fail(_):
        raise error

    bad = definition(density=fail)

    def quantile(p):
        calls.append("quantile")
        if nested == "dtest":
            assert r.survregDtest(bad, verbose=False) is False
        else:
            with pytest.raises(LookupError) as caught:
                distribution(bad).pdf_values([0], [0], [1])
            assert caught.value is error
        return p

    dist = distribution(definition(quantile=quantile))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="longer object length"):
            dist.quantile_values([0.2, 0.4, 0.6], [0], [1, 2])
    assert calls == ["quantile"]


@pytest.mark.parametrize("method", ["pdf_values", "quantile_values"])
def test_recycling_warning_stops_at_its_arithmetic_stage(method):
    calls = []

    def density(z):
        calls.append("density")
        return definition()["density"](z)

    def quantile(p):
        calls.append("quantile")
        return p

    dist = distribution(definition(density=density, quantile=quantile))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="longer object length"):
            getattr(dist, method)([0.2, 0.4, 0.6], [0, 1], [1])
    assert calls == ([] if method == "pdf_values" else ["quantile"])


@pytest.mark.parametrize("error_type", [RuntimeWarning, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("category", [RuntimeWarning, UserWarning])
def test_r_warning_hook_propagates_sink_error_and_restores_python_hook(error_type, category):
    from survival.pybridge import _call_dpqr_with_warnings

    events = []
    error = error_type("R warning sink sentinel")
    previous = warnings.showwarning

    def operation():
        events.append("before")
        warnings.warn("arithmetic warning", category, stacklevel=2)
        events.append("after")

    def sink(message):
        events.append(message)
        raise error

    with pytest.raises(error_type) as caught:
        _call_dpqr_with_warnings(operation, {}, warning=sink)
    assert caught.value is error
    assert events == ["before", "arithmetic warning"]
    assert warnings.showwarning is previous


def test_r_warning_hook_retains_previous_hook_for_other_categories():
    from survival.pybridge import _call_dpqr_with_warnings

    forwarded = []

    def operation():
        warnings.warn("deprecation sentinel", DeprecationWarning, stacklevel=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _call_dpqr_with_warnings(operation, {}, warning=forwarded.append)
    assert forwarded == []
    assert [str(issue.message) for issue in caught] == ["deprecation sentinel"]
    assert caught[0].category is DeprecationWarning


@pytest.mark.parametrize("marker", ["missing", "null"])
@pytest.mark.parametrize(
    "roundtrip",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],  # noqa: S301
)
def test_query_parameter_marker_survives_roundtrip(marker, roundtrip):
    original = core.SurvregDistribution.for_query("t", _parms_null=marker == "null")
    restored = roundtrip(original)
    expected = 'argument "parms" is missing' if marker == "missing" else "Non-numeric argument"
    for value in (original, restored):
        with pytest.raises(ValueError, match=expected):
            value.quantile_values([0.2], [0], [1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            with pytest.raises(ValueError, match=expected):
                value.pdf_values([0.2, 0.4], [0, 1, 2], [1, 2, 3, 4, 5])
        assert [str(issue.message) for issue in caught] == [
            "longer object length is not a multiple of shorter object length"
        ] * 2


def test_default_t_distribution_queries_keep_fitting_df_default():
    dist = core.SurvregDistribution("t")
    assert dist.parms == [4]
    np.testing.assert_allclose(
        dist.quantile_values([0.2, 0.4], [0], [1]),
        core.qsurvreg([0.2, 0.4], [0], [1], "t", [4]),
        rtol=2e-12,
    )


def query_identity(values):
    return values


def query_derivative(values):
    return np.ones(len(values))


@pytest.mark.parametrize("parms", [[2], [], [3, 5]])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize(
    "roundtrip",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],  # noqa: S301
)
def test_permissive_query_parameter_values_survive_roundtrip(parms, transformed, roundtrip):
    original = core.SurvregDistribution.for_query("t", parms)
    if transformed:
        original = original.with_transform(
            trans=query_identity, dtrans=query_derivative, itrans=query_identity
        )
    restored = roundtrip(original)
    assert restored.parms == parms
    for method in ("pdf_values", "quantile_values"):
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            expected = getattr(original, method)([0.2, 0.4], [0], [1])
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            actual = getattr(restored, method)([0.2, 0.4], [0], [1])
        np.testing.assert_allclose(actual, expected, rtol=2e-12, equal_nan=True)
        assert [str(issue.message) for issue in actual_warnings] == [
            str(issue.message) for issue in expected_warnings
        ]
