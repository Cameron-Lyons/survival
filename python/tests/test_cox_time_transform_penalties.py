"""Penalized time-transform bases against complete independent R fits."""

import json
import math
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
core = survival.regression
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/cox_time_transform_penalty_reference.json").read_text()
)


def close(actual, expected, *, atol=2e-8):
    def numeric(value):
        return (
            [numeric(v) for v in value]
            if isinstance(value, list)
            else math.nan
            if value is None
            else value
        )

    np.testing.assert_allclose(actual, numeric(expected), rtol=4e-7, atol=atol, equal_nan=True)


def data_for(case):
    data = dict(REFERENCE["data"])
    data["g"] = RFactor(data["g"], REFERENCE["levels"])
    if case["kind"] == "missing":
        data["x"] = list(data["x"])
        for row in (1, 8):
            data["x"][row] = None
    if case["kind"] == "counting":
        data["futime"] = [v + (i + 1) / 1000 for i, v in enumerate(data["futime"])]
    return data


def callback_for(case):
    def transform(x, time, riskset, weights):
        kind = case["kind"]
        if kind.startswith("spline"):
            options = (
                {"theta": 0.4}
                if kind == "spline_fixed"
                else {"df": 2, "nterm": 3, "degree": 1}
                if kind == "spline_unpenalized"
                else {"df": 3, "degree": 2}
                if kind == "spline_degree"
                else {"df": 2, "nterm": 6, "combine": [1, 1, 2, 2, 3, 3, 4, 4]}
                if kind == "spline_combined"
                else {"df": 3}
            )
            return r.pspline(
                np.asarray(x) * np.log(time), penalty=kind != "spline_unpenalized", **options
            )
        present = set(x)
        levels = (
            [level for level in x.categories if level in present]
            if hasattr(x, "categories")
            else sorted(present)
        )
        lookup = {level: i + 1 for i, level in enumerate(levels)}
        codes = [lookup[value] for value in x]
        options = {"df": 0.7} if kind == "df" else {} if kind == "search" else {"theta": 0.4}
        penalty = core.CoxPenalty.frailty(
            distribution=case["distribution"], sparse=case["sparse"], n=len(x), **options
        )
        if case["sparse"]:
            basis = [[41 if v == 1 else 73] for v in codes] if kind == "raw_codes" else codes
            names = None
        else:
            basis = [
                [float(code == group) for group in range(1, len(levels) + 1)] for code in codes
            ]
            prefix = {"gamma": "gamma", "gaussian": "gauss", "t": "t"}[case["distribution"]]
            names = [f"{prefix}:{level}" for level in levels]
        return r.CoxPenaltyBasis(basis, penalty, column_names=names)

    return transform


def fit_case(case):
    precise = case["kind"] in {"weighted", "counting"}
    return r.coxph(
        case["formula"],
        data_for(case),
        tt=callback_for(case),
        ties=case["method"],
        robust=False,
        weights="w" if case["kind"] in {"weighted", "counting"} else None,
        subset=[i for i in range(26) if (i + 1) % 4] if case["kind"] == "subset" else None,
        na_action="na.exclude" if case["kind"] == "missing" else "na.omit",
        eps=1e-14 if precise else 1e-10,
        toler_chol=1e-15 if precise else None,
        iter_max=100,
        outer_max=50,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_native_penalty_results_match_r_fit_summary_and_residuals(case):
    fit = fit_case(case)
    compare = partial(close, atol=5e-7) if case["information_reference"] else close
    expected_names = [name.replace("x * log(t)", "x") for name in case["coefficient_names"]]
    assert list(fit.coef_names) == expected_names
    compare(fit.coefficients, case["coefficients"])
    compare(fit.var, case["variance"])
    compare(fit.loglik, case["loglik"])
    compare(r.fitted(fit), case["lp"])
    matrix = r.model_matrix(fit)
    compare(matrix["data"], case["x"])
    assert matrix["assign"] == case["matrix_assign"]
    assert matrix["columns"] == case["matrix_names"]
    compare(np.column_stack((fit.y.time, fit.y.event)), case["y"])
    residuals = r.residuals(fit)
    if case["kind"] == "missing":
        assert math.isnan(residuals[1])
        assert math.isnan(residuals[8])
        residuals = [v for i, v in enumerate(residuals) if i not in (1, 8)]
    compare(residuals, case["martingale"])
    if fit.penalized is not None:
        compare(fit.df, case["df"])
        assert list(fit.pterms) == case["pterms"]
        if case["sparse"]:
            compare(fit.frail, case["frail"])
            compare(fit.fvar, case["fvar"])
        summary = r.model_summary(fit)
        compare(
            [
                [row[k] for k in ("coef", "se", "se2", "chisq", "df", "p")]
                for row in summary["coefficients"]
            ],
            case["summary"],
        )
        assert [row["name"] for row in summary["coefficients"]] == case["summary_names"]
        expected_print2 = case["print2"]
        if isinstance(expected_print2, str):
            expected_print2 = [expected_print2]
        assert summary["print2"] == expected_print2


@pytest.mark.parametrize(
    "name",
    [
        "fixed/gamma/sparse/efron",
        "factor/gamma/sparse/efron",
        "raw_codes/gamma/sparse/efron",
        "sparse_only/gamma/sparse/efron",
        "spline//dense/efron",
    ],
)
def test_penalty_models_preserve_group_values_and_summary_after_pickle(name):
    case = next(c for c in REFERENCE["cases"] if c["name"] == name)
    fit = fit_case(case)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - our own native model
    assert restored.coef_names == fit.coef_names
    close(r.model_matrix(restored)["data"], r.model_matrix(fit)["data"])
    close(r.residuals(restored), r.residuals(fit))
    assert r.model_summary(restored)["print2"] == r.model_summary(fit)["print2"]


def test_penalty_interactions_and_multiple_sparse_terms_fail_explicitly():
    case = REFERENCE["cases"][0]
    with pytest.raises(ValueError, match="Penalty terms cannot be in an interaction"):
        r.coxph("Surv(futime,fustat) ~ tt(rx):x", data_for(case), tt=callback_for(case))
    with pytest.raises(ValueError, match="Only one sparse penalty"):
        r.coxph(
            "Surv(futime,fustat) ~ tt(rx) + tt(resid.ds)", data_for(case), tt=callback_for(case)
        )


def test_basis_validation_happens_before_native_callbacks():
    called = []

    def penalty(coef, *, which):
        called.append(True)
        return {}

    value = core.CoxPenalty.callback(penalty)
    for basis in ([], [[0, 1]]):
        with pytest.raises(ValueError, match="one value per expanded row"):
            r.coxph(
                "Surv(futime,fustat) ~ tt(rx)",
                REFERENCE["data"],
                tt=lambda *args, basis=basis: r.CoxPenaltyBasis(basis, value),
            )
    assert called == []


def test_public_penalty_column_names_are_full_labels_and_validate_width():
    data = REFERENCE["data"]
    penalty = core.CoxPenalty.ridge(theta=2, scale=False)

    def callback(x, time, riskset, weights):
        return r.CoxPenaltyBasis(
            np.column_stack((np.asarray(x) * np.log(time), np.asarray(x) * np.sqrt(time))),
            penalty,
            column_names=np.array(["log term", "root term"]),
        )

    fit = r.coxph("Surv(futime,fustat) ~ x + tt(rx)", data, tt=callback)
    assert fit.coef_names == ("x", "log term", "root term")
    assert r.model_matrix(fit)["columns"] == ["x", "tt(rx)1", "tt(rx)2"]
    with pytest.raises(ValueError, match="an evaluated matrix is required"):
        r.model_matrix(fit, {"x": [1.0], "rx": [2.0]})
    with pytest.raises(ValueError, match="tt penalty column names must match its width"):
        r.coxph(
            "Surv(futime,fustat) ~ tt(rx)",
            data,
            tt=lambda x, *args: r.CoxPenaltyBasis([[v, v * v] for v in x], penalty, ["one"]),
        )
    with pytest.raises(TypeError, match="requires a native CoxPenalty"):
        r.coxph(
            "Surv(futime,fustat) ~ tt(rx)", data, tt=lambda x, *args: r.CoxPenaltyBasis(x, object())
        )
    with pytest.raises(ValueError, match="requires a PsplineResult basis"):
        r.coxph(
            "Surv(futime,fustat) ~ tt(rx)",
            data,
            tt=lambda x, *args: r.CoxPenaltyBasis(x, core.CoxPenalty.pspline()),
        )


def test_sparse_callback_without_formatter_has_a_group_wald_summary():
    def penalty(coef, *, which):
        assert which == 1
        return {
            "coef": coef,
            "first": -np.asarray(coef),
            "second": np.ones(len(coef)),
            "penalty": -0.5 * np.dot(coef, coef),
            "flag": False,
        }

    value = core.CoxPenalty.callback(penalty, sparse=True)
    fit = r.coxph(
        "Surv(futime,fustat) ~ x + tt(rx)",
        REFERENCE["data"],
        tt=lambda x, *args: r.CoxPenaltyBasis(x, value),
        iter_max=100,
    )
    row = r.model_summary(fit)["coefficients"][-1]
    assert row["name"] == "tt(rx)"
    assert math.isnan(row["coef"])
    close(row["chisq"], sum(b * b / v for b, v in zip(fit.frail, fit.fvar, strict=True)))
    close(row["df"], fit.df[-1])


def test_boolean_penalty_basis_keeps_its_numeric_penalty():
    value = core.CoxPenalty.ridge(theta=2, scale=False)
    formula = "Surv(futime,fustat) ~ x + tt(rx)"
    fit = r.coxph(
        formula, REFERENCE["data"], tt=lambda x, *args: r.CoxPenaltyBasis(np.asarray(x) > 1, value)
    )
    expected = r.coxph(
        formula,
        REFERENCE["data"],
        tt=lambda x, *args: r.CoxPenaltyBasis((np.asarray(x) > 1).astype(float), value),
    )
    assert fit.penalized is not None
    assert fit.coef_names == ("x", "tt(rx)")
    close(fit.coefficients, expected.coefficients)
    close(fit.var, expected.var)


def test_categorical_penalty_basis_is_rejected_before_the_penalty_callback():
    def penalty(*args, **kwargs):
        pytest.fail("a nonnumeric basis reached the penalty callback")

    value = core.CoxPenalty.callback(penalty)
    with pytest.raises(ValueError, match="could not convert string to float"):
        r.coxph(
            "Surv(futime,fustat) ~ tt(rx)",
            REFERENCE["data"],
            tt=lambda x, *args: r.CoxPenaltyBasis(["group"] * len(x), value),
        )


def test_stored_matrix_keeps_ordinary_sparse_frailty_column():
    data = survival.datasets.load_kidney()
    fit = r.coxph("Surv(time,status) ~ age + frailty(id,theta=.4)", data)
    result = r.model_matrix(fit)
    assert result["columns"] == ["age", "frailty(id,theta=.4)"]
    assert result["assign"] == [1, 2]
    assert [row[1] for row in result["data"]] == data["id"]
    assert len(fit.x[0]) == len(fit.coefficients) == 1
