"""Standalone penalty bases/configuration and complete fits from stock R."""

import json
import math
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
core = survival.regression
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/penalty_constructor_reference.json").read_text()
)


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, str) and value in {"NaN", "Inf", "-Inf"}:
        return float(value)
    return value


def numeric(value):
    if isinstance(value, list):
        return [numeric(item) for item in value]
    return math.nan if value is None else decode(value)


def close(actual, expected, *, rtol=4e-7, atol=2e-8):
    np.testing.assert_allclose(actual, numeric(expected), rtol=rtol, atol=atol, equal_nan=True)


def constructor_inputs(case, layout):
    inputs = []
    for spec in case["inputs"]:
        values = decode(spec["values"])
        if spec["levels"] is not None:
            values = (
                pytest.importorskip("pandas").Categorical(values, categories=spec["levels"])
                if layout == "pandas"
                else RFactor(values, spec["levels"])
            )
        elif spec["matrix"]:
            array = np.asarray(numeric(values), dtype=float).reshape(spec["shape"])
            values = (
                pytest.importorskip("pandas").DataFrame(array, columns=spec["columns"])
                if layout == "pandas"
                else array
                if layout == "numpy" or not array.size
                else array.tolist()
            )
        elif layout == "numpy":
            values = np.asarray(values, dtype=object)
        elif layout == "pandas":
            values = pytest.importorskip("pandas").Series(values)
        inputs.append(values)
    return inputs


def invoke_constructor(case, layout):
    function = getattr(r, case["constructor"].replace(".", "_"))
    return function(*constructor_inputs(case, layout), **(decode(case["options"]) or {}))


@pytest.mark.parametrize("layout", ["list", "numpy", "pandas"])
@pytest.mark.parametrize("case", REFERENCE["constructors"], ids=lambda case: case["name"])
def test_constructor_basis_labels_configuration_and_errors_match_stock(case, layout):
    expected = case["expected"]
    if expected["error"] is not None:
        message = (
            "requires at least one numeric input"
            if case["name"] == "ridge/empty-arguments"
            else re.escape(expected["error"].replace("“", '"').replace("”", '"'))
        )
        with pytest.raises((TypeError, ValueError), match=message):
            invoke_constructor(case, layout)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = invoke_constructor(case, layout)
    assert len(caught) == len(expected["warnings"])
    if caught:
        assert "not a multiple" in str(caught[0].message)
    actual_basis = np.asarray(result.basis, dtype=float).reshape(expected["shape"])
    expected_basis = np.asarray(numeric(expected["basis"]), dtype=float).reshape(expected["shape"])
    close(actual_basis, expected_basis, rtol=2e-14, atol=0)
    assert result.penalty.kind == case["kind"]
    assert result.penalty.diag == expected["diag"]
    if case["kind"] == "ridge":
        assert isinstance(result, r.RidgeResult)
        close(result.scale_values, expected["pparm"], rtol=2e-14, atol=0)
        assert len(result.column_names) == expected["shape"][1]
        assert not result.penalty.sparse
        config = decode(expected["cparm"])
        options = decode(case["options"]) or {}
        reference_penalty = core.CoxPenalty.ridge(
            theta=config.get("theta"),
            df=config.get("df"),
            eps=config.get("eps", options.get("eps", 0.1)),
            scale=options.get("scale", True),
            scale_values=result.scale_values,
        )
        # Compare complete native configurations through their supported pickle form.
        assert pickle.dumps(result.penalty) == pickle.dumps(reference_penalty)
    else:
        assert isinstance(result, r.FrailtyResult)
        assert result.levels == tuple(expected["levels"])
        assert result.codes == tuple(expected["codes"])
        assert result.penalty.sparse == expected["sparse"]
        family = (
            case["constructor"].split(".")[1]
            if "." in case["constructor"]
            else (decode(case["options"]) or {}).get("distribution", "gamma")
        )
        family = "gaussian" if family == "gaus" else family
        assert result.penalty.distribution == family
        assert result.column_names == (None if expected["sparse"] else tuple(expected["varname"]))
        config = decode(expected["cparm"]) or {}
        options = decode(case["options"]) or {}
        method = expected["method"]
        reference_penalty = core.CoxPenalty.frailty(
            distribution=family,
            sparse=expected["sparse"],
            theta=config.get("theta"),
            df=config.get("df", options.get("df")),
            eps=config.get("eps"),
            method=method,
            tdf=options.get("tdf", 5),
            caic=config.get("caic", False),
            init=config.get("init") if method in {"aic", "em", "reml"} else None,
            n=expected["shape"][0],
        )
        assert pickle.dumps(result.penalty) == pickle.dumps(reference_penalty)


def data_for_fit():
    data = decode(REFERENCE["data"])
    data["g"] = RFactor(data["g"], REFERENCE["group_levels"])
    return data


def fit_case(case):
    data = data_for_fit()

    def transform(x, time, riskset, weights):
        if case["kind"] == "frailty":
            return r.frailty(x, **case["options"])
        values = np.asarray(x)
        columns = [
            values * np.log(time) if spec["transform"] == "log_time" else values * np.sqrt(time)
            for spec in case["columns"]
        ]
        return r.ridge(
            *columns,
            column_names=case["expected"]["coefficient_names"],
            **case["options"],
        )

    return r.coxph(
        case["formula"].replace("theta=.4", "theta = 0.4"),
        data,
        tt=None if case.get("transform") == "ordinary" else transform,
        subset=case["subset_zero_based"],
        robust=False,
        eps=1e-10,
        iter_max=100,
        outer_max=50,
    )


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda case: case["name"])
def test_complete_constructor_tt_fits_and_shared_formula_path_match_stock(case):
    fit = fit_case(case)
    expected = case["expected"]
    assert expected["error"] is None
    assert list(fit.coef_names) == expected["coefficient_names"]
    close(fit.coefficients, expected["coef"])
    close(fit.var, expected["variance"])
    close(fit.var2, expected["var2"])
    close(fit.loglik, expected["loglik"])
    close(r.fitted(fit), expected["lp"])
    close(fit.df, expected["df"])
    if expected["frail"] is not None:
        close(fit.frail, expected["frail"])
        close(fit.fvar, expected["fvar"])
    matrix = r.model_matrix(fit)
    close(matrix["data"], expected["x"])
    assert matrix["columns"] == expected["matrix_names"]
    assert matrix["assign"] == expected["matrix_assign"]
    close(np.column_stack((fit.y.time, fit.y.event)), expected["y"])
    summary = r.model_summary(fit)
    close(
        [
            [row[key] for key in ("coef", "se", "se2", "chisq", "df", "p")]
            for row in summary["coefficients"]
        ],
        expected["summary"],
    )
    assert [row["name"] for row in summary["coefficients"]] == expected["summary_names"]
    text = expected["print2"]
    assert summary["print2"] == ([] if text is None else [text] if isinstance(text, str) else text)
    if case["kind"] == "frailty":
        term = next(term for term in fit._frame.design.covariates if hasattr(term, "penalty"))
        used = {str(value) for value in REFERENCE["data"]["g"]}
        assert term.levels == tuple(value for value in REFERENCE["group_levels"] if value in used)
        restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - trusted test fixture
        assert r.model_summary(restored)["print2"] == summary["print2"]
        close(restored.coefficients, fit.coefficients)
        close(restored.var, fit.var)
        restored_term = next(
            term for term in restored._frame.design.covariates if hasattr(term, "penalty")
        )
        assert restored_term.levels == term.levels


def test_original_ridge_scaling_survives_subsetting_and_matches_stock_fit():
    case = next(case for case in REFERENCE["fits"] if case["name"] == "ordinary/ridge-subset")
    data = data_for_fit()
    basis = r.ridge(data["x"], **case["options"])
    rows = case["subset_zero_based"]
    selected = [basis.basis[row] for row in rows]
    result = core.coxpenal_fit(
        [data["futime"][row] for row in rows],
        [data["fustat"][row] for row in rows],
        selected,
        penalties=[basis.penalty],
        pcols=[[0]],
        assign=[[0]],
        eps=1e-10,
        iter_max=100,
        outer_max=50,
    )
    close(result.coefficients, case["expected"]["coef"])
    close(result.var, case["expected"]["variance"])
    close(result.df, case["expected"]["df"])
    assert basis.scale_values != r.ridge([data["x"][row] for row in rows]).scale_values


def test_many_frailty_groups_retain_linear_encoding_and_labels():
    values = list(range(10000, 0, -1)) * 3
    basis = r.frailty(values, theta=0.4)
    assert basis.penalty.sparse
    assert len(basis.levels) == 10000
    assert len(basis.basis) == len(values)
    assert basis.codes[:3] == (10000, 9999, 9998)
    assert [basis.levels[code - 1] for code in basis.codes] == [str(value) for value in values]


def test_named_ridge_inputs_explicit_labels_and_result_roundtrip():
    named = {"age": [1.0, 2.0, 3.0], "sex": [0.0, 1.0, 0.0]}
    result = r.ridge(named, theta=0.4)
    assert result.column_names == ("ridge(age)", "ridge(sex)")
    explicit = r.ridge(named, theta=0.4, column_names=["age penalty", "sex penalty"])
    assert explicit.column_names == ("age penalty", "sex penalty")
    with pytest.raises(ValueError, match="column_names must match"):
        r.ridge(named, column_names=["one"])
    with pytest.raises(TypeError, match="inputs must be numeric"):
        r.ridge(["a", "b"])
    restored = pickle.loads(pickle.dumps(explicit))  # noqa: S301 - trusted test fixture
    assert restored.basis == explicit.basis
    assert restored.column_names == explicit.column_names
    assert restored.scale_values == explicit.scale_values
    assert pickle.dumps(restored.penalty) == pickle.dumps(explicit.penalty)
