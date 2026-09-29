"""AFT and Aalen reports: native fits, R summary snapshots and covariance checks."""

import dataclasses
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aft_aalen_report_reference.json").read_text()
)


def matrix(source):
    return r.NamedMatrix(
        source["rows"], source["columns"], np.asarray(source["values"], dtype=float).tolist()
    )


def options(arguments):
    return {name.replace(".", "_"): value for name, value in dict(arguments).items()}


def summary_input(case):
    source = case["input"]
    result = {key: value for key, value in source.items() if value != {}}
    table = matrix(source["table"])
    if source["model_type"] == "aareg":
        keys = (
            ["slope", "coef", "se"]
            + (["robust_se"] if "robust se" in table.colnames else [])
            + ["z", "p"]
        )
        result["columns"] = table.colnames
        result["table"] = [
            dict(name=name, **dict(zip(keys, row, strict=True)))
            for name, row in zip(table.rownames, table.values, strict=True)
        ]
        result["test_var2"] = source["test_var2"] or None
    else:
        keys = (
            ["coef", "se", "naive_se", "z", "p"] if source["robust"] else ["coef", "se", "z", "p"]
        )
        result["coefficients"] = [
            dict(name=name, **dict(zip(keys, row, strict=True)))
            for name, row in zip(table.rownames, table.values, strict=True)
        ]
        result["coefficient_names"] = table.rownames
        result["location_coefficients"] = np.asarray(
            source["location_coefficients"], dtype=float
        ).tolist()
        result["na_action"] = r.NaAction(tuple(source["omit"]), "omit") if source["omit"] else None
        result["correlation"] = (
            matrix(source["correlation"]).values if source["correlation"] else None
        )
    return result


@pytest.fixture(scope="module")
def fits():
    @lru_cache(None)
    def fit(name):
        spec = REFERENCE["models"][name]
        data = pd.DataFrame(REFERENCE["datasets"][spec["data"]])
        return getattr(r, spec["kind"])(spec["formula"], data, **options(spec["arguments"]))

    return fit


def render(case, fits):
    fit = fits(case["fit"])
    spec = REFERENCE["models"][case["fit"]]
    function = getattr(
        r, "print_" + ("summary_" if case["method"] == "summary" else "") + spec["kind"]
    )
    value = (
        r.model_summary(fit, **options(case["summary_arguments"]))
        if case["method"] == "summary"
        else fit
    )
    return function(value, width=case["width"], **options(case["arguments"]))


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_native_model_report_matches_r(case, fits, capsys):
    report = render(case, fits)
    assert report.lines == case["lines"]
    assert capsys.readouterr().out == ""
    if isinstance(report, r.ModelPrint):
        assert str(report) == "\n".join(report.lines) + "\n"


@pytest.mark.parametrize(
    "case", [case for case in REFERENCE["cases"] if case["input"]], ids=lambda case: case["name"]
)
def test_summary_snapshot_matches_r(case):
    value = summary_input(case)
    function = getattr(r, "print_summary_" + value["model_type"])
    report = function(value, width=case["width"], **options(case["arguments"]))
    assert report.lines == case["lines"]
    np.testing.assert_array_equal(
        report.tables["coefficients"].values, matrix(case["input"]["table"]).values
    )


@pytest.mark.parametrize(
    "name", ["aalen_robust", "aalen_cluster", "aalen_ties", "aalen_delayed", "aalen_ties_cluster"]
)
def test_reweighted_robust_covariance_matches_r(name, fits):
    fit = fits(name)
    case = next(case for case in REFERENCE["cases"] if case["name"] == name + "_reweighted")
    expected = case["input"]
    kwargs = options(case["summary_arguments"])
    actual = r.model_summary(fit, **kwargs)
    assert "robust se" in actual["columns"]
    np.testing.assert_allclose(actual["test_var2"], expected["test_var2"], rtol=3e-10, atol=1e-11)
    assert actual["chisq"] == pytest.approx(expected["chisq"], rel=3e-10)
    # Both stored representation types use the same bounded reducer.
    array_fit = dataclasses.replace(fit, dfbeta=np.asarray(fit.dfbeta))
    array_result = r.model_summary(array_fit, **kwargs)
    np.testing.assert_allclose(
        array_result["test_var2"], actual["test_var2"], rtol=1e-14, atol=1e-14
    )


def test_late_cutoff_preserves_original_robust_covariance(fits):
    for name in ["aalen_robust", "aalen_cluster", "aalen_ties_cluster"]:
        fit = fits(name)
        actual = r.model_summary(fit, maxtime=max(fit.times) + 1)
        np.testing.assert_allclose(
            actual["test_var2"], fit.robust_test_variance, rtol=3e-10, atol=1e-11
        )


def test_missingness_metadata_and_conversion_are_independent(fits):
    fit = fits("aalen_missing")
    assert fit.na_action is not None
    report = r.print_aareg(fit)
    assert "deleted due to missingness" in report.lines[0]
    summary = r.model_summary(fits("stratified"), correlation=True)
    before = pickle.dumps(summary)
    report = r.print_summary_survreg(summary)
    assert report.statistics["scale_names"] == ["sex=1", "sex=2"]
    frame = r.as_data_frame(report)
    original = report.tables["coefficients"].values[0][0]
    frame["Value"][0] = 999
    assert report.tables["coefficients"].values[0][0] == original
    report.tables["correlation"].values[0][0] = 999
    report.statistics["scales"][0] = 999
    assert pickle.dumps(summary) == before
    assert pickle.loads(pickle.dumps(report)) == report  # noqa: S301 - this test's own bytes
    penal = r.print_survreg(fits("spline"))
    assert isinstance(penal, r.SurvregPenalPrint)
    np.testing.assert_array_equal(r.as_data_frame(penal)["coef"], [row[0] for row in penal.rows])


def test_aliased_correlation_labels_and_legacy_summary_metadata(fits):
    summary = r.model_summary(fits("aliased"), correlation=True)
    report = r.print_summary_survreg(summary)
    assert report.tables["correlation"].rownames == ["(Intercept)", "age", "sex", "Log(scale)"]
    summary = r.model_summary(fits("stratified"))
    expected = r.print_summary_survreg(summary)
    del summary["fixed_scale"], summary["scale_names"]
    assert r.print_summary_survreg(summary).lines == expected.lines


@pytest.mark.parametrize(
    "kwargs", [{"maxtime": 0}, {"scale": 0}, {"maxtime": float("inf")}, {"test": "wrong"}]
)
def test_aalen_summary_options_validate(fits, kwargs):
    with pytest.raises(ValueError, match="maxtime|scale|test"):
        r.print_aareg(fits("aalen"), **kwargs)


@pytest.mark.parametrize(
    "function", ["print_survreg", "print_summary_survreg", "print_aareg", "print_summary_aareg"]
)
def test_invalid_report_input(function):
    with pytest.raises(TypeError):
        getattr(r, function)(None)


def test_large_influence_sets_cross_block_boundaries(fits):
    fit = fits("aalen_cluster")
    baseline = r.model_summary(fit, maxtime=2000)["test_var2"]
    repeated = dataclasses.replace(fit, dfbeta=fit.dfbeta * 1000)
    actual = r.model_summary(repeated, maxtime=2000)["test_var2"]
    np.testing.assert_allclose(actual, np.asarray(baseline) * 1000, rtol=2e-13)


def test_clustered_aalen_covariance_is_invariant_to_input_row_order(fits):
    for name in ["aalen_cluster", "aalen_ties_cluster"]:
        spec = REFERENCE["models"][name]
        data = pd.DataFrame(REFERENCE["datasets"][spec["data"]])
        time, status = ("futime", "fustat") if name == "aalen_cluster" else ("time", "status")
        sorted_data = data.sort_values([time, status], ascending=[True, False], kind="stable")
        sorted_fit = r.aareg(spec["formula"], sorted_data, **options(spec["arguments"]))
        np.testing.assert_allclose(
            sorted_fit.robust_test_variance, fits(name).robust_test_variance, rtol=2e-12, atol=1e-12
        )
        case = next(case for case in REFERENCE["cases"] if case["name"] == name)
        assert case["original_lines"] != case["lines"]
        assert r.print_aareg(sorted_fit).lines == case["lines"]


def test_variance_weighted_fit_requires_supported_summary_weights():
    fit = r.aareg(
        "Surv(futime,fustat)~age+ecog.ps", survival.datasets.load_ovarian(), test="variance"
    )
    with pytest.raises(ValueError, match="test must be aalen or nrisk"):
        r.print_aareg(fit)
    assert r.print_aareg(fit, test="aalen").tables["coefficients"].values
