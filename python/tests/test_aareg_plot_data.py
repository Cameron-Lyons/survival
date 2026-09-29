"""Aalen plot arrays against the actual R graphics methods and independent identities."""

import builtins
import copy
import json
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
plotting = survival.plotting
r = survival.r
REFERENCE = json.loads((Path(__file__).parent / "fixtures/aareg_plot_reference.json").read_text())
for case in REFERENCE["cases"]:
    case["options"] = dict(case["options"])


def reference_fit(name):
    value = REFERENCE["fits"][name]
    influence = value["dfbeta"]
    if influence is not None:
        influence = np.asarray(influence["values"]).reshape(influence["dim"], order="F").tolist()
    return r.AaregModelResult(
        n=[],
        times=value["time"],
        coefficient=value["coefficient"],
        coefficient_names=value["names"],
        n_risk=[],
        test_statistic=[],
        test_statistic_names=value["names"],
        test_variance=[],
        test="aalen",
        time_weights=[],
        dfbeta=influence,
    )


def options_for(case):
    options = {key: value for key, value in case["options"].items() if key != "type"}
    if case["method"] == "lines":
        options.setdefault("se", False)
    if case["var"] is not None:
        options["var"] = case["var"]
    return options


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_cumulative_coordinates_match_r(case):
    data = plotting.aareg_plot_data(reference_fit(case["fit"]), **options_for(case))
    for panel, expected in enumerate(case["expected"]):
        np.testing.assert_array_equal(data.time, expected["x"])
        if data.std_err is None:
            values = data.coefficient
        else:
            values = np.column_stack(
                (data.coefficient[:, panel], data.upper[:, panel], data.lower[:, panel])
            )
        np.testing.assert_allclose(values, expected["y"], rtol=1e-12, atol=2e-14)


@pytest.mark.parametrize("robust", [False, True])
def test_real_fitted_model_has_the_reference_curves(robust):
    fit = r.aareg(
        "Surv(futime,fustat) ~ age + ecog.ps", survival.datasets.load_ovarian(), dfbeta=robust
    )
    actual = plotting.aareg_plot_data(fit)
    expected = plotting.aareg_plot_data(reference_fit("robust" if robust else "ovarian"))
    np.testing.assert_allclose(actual.coefficient, expected.coefficient, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(actual.std_err, expected.std_err, rtol=1e-9, atol=1e-12)


def test_ties_sum_variances_before_collapsing_and_use_grouped_influences():
    fit = reference_fit("ties_robust")
    unadjusted = plotting.aareg_plot_data(replace(fit, dfbeta=None))
    robust = plotting.aareg_plot_data(fit)
    first = np.asarray(fit.times) == fit.times[0]
    increment = np.asarray(fit.coefficient)[first]
    np.testing.assert_allclose(unadjusted.coefficient[1], increment.sum(axis=0))
    np.testing.assert_allclose(unadjusted.std_err[1] ** 2, np.sum(increment**2, axis=0))
    expected = np.sum(np.asarray(fit.dfbeta)[:, :, 0] ** 2, axis=0)
    np.testing.assert_allclose(robust.std_err[1] ** 2, expected)


def test_single_event_and_zero_origin_work_when_r_drops_dimensions():
    assert all(item["error"] for item in REFERENCE["upstream_failures"])
    fit = reference_fit("robust")
    first = plotting.aareg_plot_data(fit, maxtime=fit.times[0])
    assert first.coefficient.shape == (2, 3)
    np.testing.assert_allclose(first.coefficient[1], fit.coefficient[0])
    full = plotting.aareg_plot_data(fit)
    np.testing.assert_allclose(first.std_err, full.std_err[:2])
    empty = plotting.aareg_plot_data(fit, maxtime=fit.times[0] - 1)
    np.testing.assert_array_equal(empty.coefficient, [[0, 0, 0]])
    np.testing.assert_array_equal(empty.std_err, [[0, 0, 0]])
    zero = plotting.aareg_plot_data(reference_fit("zero_robust"))
    assert zero.time[0] == 0
    np.testing.assert_allclose(zero.coefficient, full.coefficient[1:])
    np.testing.assert_allclose(zero.std_err, full.std_err[1:])


def test_inputs_remain_unchanged_and_numerical_data_needs_no_renderer(monkeypatch):
    fit = reference_fit("robust")
    original = copy.deepcopy(fit)
    original_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    data = plotting.aareg_plot_data(fit)
    data.coefficient[:] = 0
    data.time[:] = -1
    assert fit == original
    with pytest.raises(ImportError, match=r"pip install survival\[plot\]"):
        plotting.plot_aareg(fit)


def test_disabled_standard_errors_do_not_read_influences():
    class Unreadable:
        def __len__(self):
            raise AssertionError("influences should not be read")

    fit = replace(reference_fit("robust"), dfbeta=Unreadable())
    assert plotting.aareg_plot_data(fit, se=False).std_err is None


@pytest.mark.parametrize("as_array", [False, True])
def test_influence_reduction_has_bounded_temporary_memory(as_array):
    groups, width, times = 500, 3, 600
    cube = np.arange(groups * width * times, dtype=float).reshape(groups, width, times) / 1e6
    expected = np.sqrt(np.cumsum(np.einsum("gpt,gpt->tp", cube, cube), axis=0))
    source = cube if as_array else cube.tolist()
    fit = replace(
        reference_fit("robust"),
        times=np.arange(1, times + 1).tolist(),
        coefficient=np.zeros((times, width)).tolist(),
        dfbeta=source,
    )
    tracemalloc.start()
    try:
        data = plotting.aareg_plot_data(fit)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_allclose(data.std_err[1:], expected, rtol=1e-12)
    assert peak < 4_000_000, f"temporary allocation {peak} grows with the influence cube"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"se": 1}, "boolean"),
        ({"maxtime": np.inf}, "finite"),
        ({"var": []}, "invalid variable"),
        ({"var": 0}, "invalid variable"),
        ({"var": "unknown"}, "term names"),
    ],
)
def test_invalid_options(options, message):
    with pytest.raises((TypeError, ValueError), match=message):
        plotting.aareg_plot_data(reference_fit("ovarian"), **options)


@pytest.mark.parametrize("as_array", [False, True])
def test_malformed_influences_are_rejected(as_array):
    fit = reference_fit("robust")
    for source, message in (
        ([[[0]]], "dfbeta must be group"),
        (np.full(np.shape(fit.dfbeta), np.nan).tolist(), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            plotting.aareg_plot_data(
                replace(fit, dfbeta=np.asarray(source) if as_array else source)
            )


def test_invalid_fit_arrays_are_rejected():
    fit = reference_fit("ovarian")
    with pytest.raises(TypeError, match="aareg"):
        plotting.aareg_plot_data(None)
    with pytest.raises(ValueError, match="finite and ordered"):
        plotting.aareg_plot_data(replace(fit, times=list(reversed(fit.times))))
    with pytest.raises(ValueError, match="finite value"):
        plotting.aareg_plot_data(replace(fit, coefficient=[]))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.float64])
def test_array_influences_use_float64_arithmetic_and_accept_strides(dtype):
    fit = reference_fit("robust")
    values = np.full(np.shape(fit.dfbeta), 10**10, dtype=dtype)
    strided = np.repeat(values, 2, axis=2)[:, :, ::2]
    actual = plotting.aareg_plot_data(replace(fit, dfbeta=strided))
    expected = plotting.aareg_plot_data(replace(fit, dfbeta=values.tolist()))
    np.testing.assert_allclose(actual.std_err, expected.std_err, rtol=1e-12)
