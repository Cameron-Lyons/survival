"""Full rank-deficient AFT designs retain R's fitted values and generalized covariance."""

import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()
r = survival.r
coerce = importlib.import_module("survival.r._coerce")
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aft_rank_deficient_reference.json").read_text()
)
SAMPLE_ROWS = REFERENCE["sample_rows"]


def frame():
    data = dict(REFERENCE["data"])
    data["g"] = coerce._r_factor(
        data["g"],
        REFERENCE["factor_levels"],
        ordered=True,
        contrast={
            "data": [[-1, -1, -1], [1, -1, -1], [0, 2, -1], [0, 0, 3]],
            "columns": ["1", "2", "3"],
            "label": "contr.helmert",
        },
    )
    return data


def close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=3e-7,
        atol=3e-8,
        equal_nan=True,
    )


def fit_case(case, *, initialized=False):
    return r.survreg(
        case["formula"],
        frame(),
        dist=case["dist"],
        weights="w" if case["mode"] == "cluster" else None,
        cluster="id" if case["mode"] == "cluster" else None,
        init=case["init"] if initialized else None,
        model=True,
        x=True,
        score=True,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_rank_deficient_aft_keeps_complete_design_and_r_fitted_values(case):
    fit = fit_case(case)
    expected = case["expected"]
    assert fit.fit.converged
    assert r.coef_names(fit) == expected["coefficient_names"]
    assert len(fit.coefficients) == 4
    assert np.isnan(fit.coefficients[-1])
    assert r.model_matrix(fit)["columns"] == expected["column_names"]
    close(np.asarray(r.model_matrix(fit)["data"])[SAMPLE_ROWS], expected["x"])
    close(fit.coefficients, expected["coefficients"])
    close(fit.var, expected["var"])
    close(fit.naive_var, expected["naive"])
    close(fit.scale, expected["scale"])
    close(fit.loglik, expected["loglik"])
    close(fit.score, expected["score"])
    assert fit.df == expected["df"]
    assert fit.df_residual == expected["df_residual"]
    assert np.isfinite(fit.linear_predictors).all()
    assert (np.asarray(fit.var)[3, :] == 0).all()
    assert (np.asarray(fit.var)[:, 3] == 0).all()
    lp = r.predict(fit, type="lp", se_fit=True)
    close(np.asarray(lp.fit)[SAMPLE_ROWS], expected["lp"])
    close(np.asarray(lp.se_fit)[SAMPLE_ROWS], expected["lp_se"])
    quantile = r.predict(fit, type="quantile", p=[0.25, 0.75], se_fit=True)
    close(np.asarray(quantile.fit)[SAMPLE_ROWS], expected["quantile"])
    close(np.asarray(quantile.se_fit)[SAMPLE_ROWS], expected["quantile_se"])
    terms = r.predict(fit, type="terms", se_fit=True)
    close(np.asarray(terms.fit)[SAMPLE_ROWS], expected["terms"])
    close(np.asarray(terms.se_fit)[SAMPLE_ROWS], expected["terms_se"])
    close(
        np.asarray(r.residuals(fit, type="dfbeta", weighted=True))[SAMPLE_ROWS],
        expected["dfbeta"],
    )
    # Stock predict.survreg retains NA coefficients when evaluating new rows.
    new = {key: [value[row] for row in [2, 0, 1]] for key, value in frame().items()}
    close(r.predict(fit, new, type="lp"), expected["new_lp"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_full_rank_deficient_solver_accepts_stock_identified_initial_values(case):
    fit = fit_case(case, initialized=True)
    expected = case["initialized"]
    assert fit.fit.converged
    close(fit.coefficients, expected["coefficients"])
    close(fit.var, expected["var"])
    close(fit.scale, expected["scale"])
    close(fit.loglik, expected["loglik"])
    close(np.asarray(fit.linear_predictors)[SAMPLE_ROWS], expected["lp"])


@pytest.mark.parametrize("layout", ["numpy", "dataframe"])
@pytest.mark.parametrize("constant", [-1.0, 2.0, -0.1])
def test_public_containers_keep_nonbinary_constant_aliases_and_pickle(layout, constant):
    data = {key: value for key, value in frame().items() if key != "g"}
    data["constant"] = [constant] * len(data["time"])
    if layout == "numpy":
        data = {key: np.asarray(value) for key, value in data.items()}
    else:
        data = pytest.importorskip("pandas").DataFrame(data)
    fit = r.survreg("Surv(time, status) ~ h1 + h2 + constant", data)
    reduced = r.survreg("Surv(time, status) ~ h1 + h2", data)
    close(fit.linear_predictors, reduced.linear_predictors)
    close(np.asarray(fit.var)[np.ix_([0, 1, 2, 4], [0, 1, 2, 4])], reduced.var)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - our own model
    assert np.isnan(restored.coefficients[-1])
    close(r.predict(restored, type="lp", se_fit=True).fit, fit.linear_predictors)
    close(r.vcov(restored), fit.var)


@pytest.mark.parametrize(
    "case", [c for c in REFERENCE["cases"] if c["mode"] == "plain"], ids=lambda case: case["name"]
)
def test_bare_aft_fit_retains_zero_alias_before_full_model_marking(case):
    fitted = fit_case(case)
    matrix = r.model_matrix(fitted)["data"]
    data = frame()
    transformed = case["dist"] in {"weibull", "lognormal", "exponential"}
    y = np.column_stack([np.log(data["time"]) if transformed else data["time"], data["status"]])
    # The bare R fitter receives the already transformed response and its
    # underlying density, with exponential scale supplied explicitly.
    raw = r.survreg_fit(
        matrix,
        y,
        dist="gaussian" if case["dist"] in {"gaussian", "lognormal"} else "extreme",
        scale=1 if case["dist"] == "exponential" else 0,
    )
    assert raw.coefficients[3] == 0.0
    assert (np.asarray(raw.var)[3, :] == 0).all()
    close(np.asarray(raw.linear_predictors)[SAMPLE_ROWS], case["expected"]["lp"])
    close(raw.var, case["expected"]["var"])
    correction = -np.log(data["time"])[np.asarray(data["status"]) == 1].sum() if transformed else 0
    close(np.asarray(raw.loglik) + correction, case["expected"]["loglik"])
