"""Unused AFT scale strata against stock R and explicitly corrected R failures."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
core = survival._survival
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/aft_unused_strata_reference.json").read_text()
)


def frame(case, key="data"):
    data = dict(REFERENCE["frames"][case["frame"]] if key == "data" else REFERENCE[key])
    data["g"] = RFactor(data["g"], REFERENCE["levels"])
    return data


def close(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=3e-7, atol=2e-8
    )


def fit_case(case):
    return r.survreg(
        case["formula"],
        frame(case),
        subset=case["subset"],
        na_action=case["action"],
        weights="w",
        cluster="id" if case["robust"] else None,
        dist=case["dist"],
        model=True,
        x=True,
        score=True,
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_unused_scale_strata_match_r_fit_and_methods(case):
    fit = fit_case(case)
    assert list(fit.strata_levels) == case["scale_names"]
    assert r.coef_names(fit, complete=True) == case["variance_names"]
    assert list(fit.na_action.rows if fit.na_action else ()) == case["omitted"]
    close(r.coef(fit), case["coefficients"])
    close(fit.scale, case["scale"])
    close(r.vcov(fit), case["variance"])
    close(fit.loglik, case["loglik"])
    close(fit.df_residual, case["df_residual"])
    close(fit.score, case["score"])
    close(r.predict(fit, type="lp"), case["lp"])
    predicted = r.predict(fit, frame(case, "newdata"), type="quantile", p=[0.25, 0.75], se_fit=True)
    close(predicted.fit, case["quantile"])
    close(predicted.se_fit, case["quantile_se"])
    close(r.residuals(fit, type="dfbeta", weighted=True), case["dfbeta"])
    if case["robust"]:
        close(fit.fit.naive_variance_matrix, case["naive"])
    if fit.penalized is not None:
        close(fit.penalized.var2, case["var2"])
        close(fit.penalized.df, case["df"])
    else:
        close([fit.df], case["df"])
    summary = r.model_summary(fit)
    assert summary["scale_names"] == case["scale_names"]
    close(summary["scales"], case["scale"])


@pytest.mark.parametrize("kind", ["plain", "robust", "ridge"])
def test_retained_scales_survive_pickle_and_initial_values(kind):
    case = next(c for c in REFERENCE["cases"] if c["name"] == f"trailing_subset/{kind}/weibull")
    fit = fit_case(case)
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - round-trip our own model
    assert restored.strata_levels == fit.strata_levels
    close(r.vcov(restored), r.vcov(fit))
    close(
        r.predict(restored, frame(case, "newdata"), type="quantile", p=0.75),
        r.predict(fit, frame(case, "newdata"), type="quantile", p=0.75),
    )
    initialized = r.survreg(
        case["formula"],
        frame(case),
        subset=case["subset"],
        weights="w",
        dist=case["dist"],
        init=list(fit.fit.coefficients),
    )
    close(initialized.scale, fit.scale)
    close(initialized.coefficients, fit.coefficients)


@pytest.mark.parametrize("name", ["survreg_fit", "survpenal_fit"])
@pytest.mark.parametrize("count", [0, 1, 2])
def test_full_fit_rejects_scale_count_below_observed_codes(name, count):
    data = core.SurvregData(
        [1.0, 2.0, 3.0, 4.0], [1, 1, 0, 1], [[1.0, x] for x in range(4)], strata=[0, 2, 0, 2]
    )
    extra = (
        {"penalties": [core.CoxPenalty.ridge(theta=1)], "pcols": [[1]]}
        if name == "survpenal_fit"
        else {}
    )
    with pytest.raises(ValueError, match="Invalid strata"):
        getattr(core, name)(data, core.SurvregDistribution("gaussian"), nstrat=count, **extra)


@pytest.mark.parametrize("penalized", [False, True])
def test_anova_refits_retain_trailing_scale_columns(penalized):
    from survival.r._survreg import _refit_terms

    case = next(c for c in REFERENCE["cases"] if c["name"] == "trailing_subset/plain/weibull")
    data = frame(case)
    term = "ridge(age, theta=2)" if penalized else "age"
    fit = r.survreg(f"Surv(time, status) ~ strata(g) + {term} + sex", data, subset=case["subset"])
    reduced = _refit_terms(fit, 2)
    expected = r.survreg(f"Surv(time, status) ~ strata(g) + {term}", data, subset=case["subset"])
    close(reduced.coefficients, expected.fit.coefficients)
    close(reduced.variance_matrix, expected.var)
    assert len(reduced.scale) == 3


@pytest.mark.parametrize("kind", ["ridge(age, theta=2)", "ridge(age, df=.7)", "pspline(age, df=3)"])
def test_empty_strata_do_not_change_penalty_effective_sample_size(kind):
    case = next(c for c in REFERENCE["cases"] if c["name"] == "trailing_subset/plain/weibull")
    data = frame(case)
    formula = f"Surv(time, status) ~ {kind} + sex + strata(g)"
    full = r.survreg(formula, data, subset=case["subset"])
    # Keep the same pre-subset penalty basis and scaling. Only recode the group
    # on rows that will be removed, so no empty level reaches the compact fit.
    compact = {**data, "g": ["a" if g == "c" else g for g in data["g"]]}
    expected = r.survreg(formula, compact, subset=case["subset"])
    close(full.penalized.n_eff, expected.penalized.n_eff)
    close(full.coefficients, expected.coefficients)
    close(full.scale[:2], expected.scale)
    close(np.asarray(full.var)[:-1, :-1], expected.var)
    close(full.penalized.df, expected.penalized.df)


@pytest.mark.parametrize("case", REFERENCE["interactions"], ids=lambda case: case["name"])
def test_empty_strata_keep_interaction_design_columns(case):
    data = dict(REFERENCE["base"])
    data["g"] = RFactor(data["g"], REFERENCE["levels"])
    fit = r.survreg(
        "Surv(time, status) ~ age * strata(g)",
        data,
        subset=case["subset"],
        dist=case["dist"],
        weights="w",
        x=True,
    )
    assert r.coef_names(fit) == case["names"]
    close(fit.coefficients, case["coefficients"])
    close(fit.var, case["variance"])
    close(fit.scale, case["scale"])
    close(r.model_matrix(fit)["data"], case["x"])
    close(r.predict(fit, type="lp"), case["lp"])
