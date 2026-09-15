"""Differential tests against R's survival package.

One test per (topic, case, aspect) triple in ``test/r/fixtures/*.json``.  The
fixtures are produced by ``test/r/generate_fixtures.R``; the schema and the
regeneration recipe live in ``test/r/README.md``.

Tolerances (relative): 1e-8 for coefficients, curves, residuals and linear
predictors; 1e-6 for variances, standard errors, test statistics and
p-values; exact for counts.

Cases the Python API cannot reproduce yet are listed in ``KNOWN_FAILURES``
and marked ``xfail(strict=True)``: a fixed case turns into an XPASS failure
that forces its removal from the list.  Keys are either ``topic/case/aspect``
or ``topic/case`` (every aspect of the case).

Set ``R_FIXTURES_COLLECT=/path/file.jsonl`` to append one record per failing
test (kind + message) for building the burndown list.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import (
    RTOL_COEF,
    RTOL_VAR,
    FixtureMismatchError,
    RFactor,
    UnsupportedCaseError,
    array_scale,
    as_float_list,
    assert_close,
    assert_exact,
    assert_matrix_close,
    assert_named_values,
    case_data,
    case_id,
    cases,
    decode_frame,
    decode_vector,
    is_r_error,
    load_dataset,
    load_topic,
    newdata_frame,
    nrow,
    record_outcome,
    topic_names,
)

survival = setup_survival_import()
r = survival.r_api

# ---------------------------------------------------------------------------
# Known failures (burndown list).  Reasons start with one of
#   "missing feature:" the Python API cannot express the case
#   "mismatch:"        Python runs but the numbers differ from R
#   "error:"           the Python call raises
# ---------------------------------------------------------------------------

KNOWN_FAILURES: dict[str, str] = {
    "aareg/ovarian_age_ecog_dfbeta/dfbeta": "mismatch: dfbeta[0][3][0]: -0.010642 != 0.051349",
    "aareg/ovarian_age_rx_dfbeta_nrisk/dfbeta": (
        "mismatch: dfbeta[0][3][0]: -0.00015217 != 0.049983"
    ),
    "aareg/veteran_karno_celltype": "mismatch: times: length 104 differs from expected 117",
    "concordance/aml_x_numeric": (
        "error: ValueError: as.numeric() formula term 'x' requires numeric values"
    ),
    "concordance/coxph_survreg_fits/both.concordance": (
        "error: TypeError: argument is not an appropriate fit object"
    ),
    "coxph/heart_counting_age_surgery_transplant/coef_names": (
        "mismatch: coef: names ['age', 'surgery', 'transplant'] != ['age', 'surgery', 't..."
    ),
    "coxph/heart_counting_age_surgery_transplant/summary.coefficients": (
        "mismatch: coefficients.rows[2]: 'transplant' != 'transplant1'"
    ),
    "coxph/heart_counting_age_surgery_transplant/summary.conf_int": (
        "mismatch: conf_int has no row 'transplant1'"
    ),
    "coxph/heart_counting_breslow/coef_names": (
        "mismatch: coef: names ['age', 'surgery', 'transplant'] != ['age', 'surgery', 't..."
    ),
    "coxph/heart_counting_breslow/summary.coefficients": (
        "mismatch: coefficients.rows[2]: 'transplant' != 'transplant1'"
    ),
    "coxph/heart_counting_breslow/summary.conf_int": "mismatch: conf_int has no row 'transplant1'",
    # R's concordance decides ties by exact equality of the linear predictors,
    # and for these two cases the tie pattern depends on the platform's
    # floating-point rounding: the fixtures (R on CI's reference BLAS) count
    # 343 tied.x for lung_age_sex_init_iter0 where R with OpenBLAS and the
    # port count 334; the port reproduces the OpenBLAS results.
    "coxph/lung_age_sex_cluster_inst/concordance.cvar": (
        "mismatch: cvar[0]: 0.00069155 != 0.00069155 (platform-dependent lp ties)"
    ),
    "coxph/lung_age_sex_init_iter0/concordance.concordance": (
        "mismatch: concordance[0]: 0.60258 != 0.60255 (platform-dependent lp ties)"
    ),
    "coxph/lung_age_sex_init_iter0/concordance.count": (
        "mismatch: count[0]: 11893.0 != 11888.0 (platform-dependent lp ties)"
    ),
    "coxph/lung_age_sex_init_iter0/concordance.cvar": (
        "mismatch: cvar[0]: 0.00067813 != 0.00067811 (platform-dependent lp ties)"
    ),
    "coxph/lung_age_sex_init_iter0/concordance.var": (
        "mismatch: var[0]: 0.00065003 != 0.00064955 (platform-dependent lp ties)"
    ),
    "coxph/lung_age_sex_init_iter0/summary.concordance": (
        "mismatch: concordance.C: 0.60258 != 0.60255 (platform-dependent lp ties)"
    ),
    "coxph/synthetic_delayed_x_exact/residuals.deviance": (
        "mismatch: residuals.deviance[0]: 1.4061 != 0.83923"
    ),
    "coxph_penalized/cgd_frailty_gamma_id": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/kidney_frailty_gamma": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/kidney_frailty_gamma_theta_fixed": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/kidney_frailty_gaussian": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/kidney_frailty_gaussian_df": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/kidney_frailty_t": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/lung_pspline_age_df0_aic": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/lung_pspline_age_df4": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/lung_pspline_karno_df3_nterm6": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/lung_ridge_age_sex_theta1": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/lung_ridge_age_sex_theta5_scaled": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/lung_ridge_df2": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/rats_frailty_gamma_litter": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "coxph_penalized/rats_frailty_gaussian_litter": (
        "missing feature: penalised Cox terms (ridge/pspline/frailty) are not implemented"
    ),
    "finegray/mgus2_400_death/coxph.loglik": (
        "missing feature: result has none of the attributes ('log_likelihood',)"
    ),
    "finegray/mgus2_400_pcm/coxph.loglik": (
        "missing feature: result has none of the attributes ('log_likelihood',)"
    ),
    "finegray/mgus2_400_pcm_strata_sex/coxph.loglik": (
        "missing feature: result has none of the attributes ('log_likelihood',)"
    ),
    "finegray/synthetic_ties_a/coxph.loglik": (
        "missing feature: result has none of the attributes ('log_likelihood',)"
    ),
    "royston_brier/lung_age_sex/royston": (
        "error: AttributeError: 'dict' object has no attribute 'd'"
    ),
    "royston_brier/lung_age_sex/royston_adjust": (
        "error: AttributeError: 'dict' object has no attribute 'd'"
    ),
    "royston_brier/pbc_trial_bili_edema/royston": (
        "error: AttributeError: 'dict' object has no attribute 'd'"
    ),
    "royston_brier/veteran_karno_celltype/royston": (
        "error: AttributeError: 'dict' object has no attribute 'd'"
    ),
    "survcondense/lung_split_age_sex_epi": "mismatch: rows 106 != 0",
    "survexp/lung_coxph_ratetable": (
        "missing feature: survexp with a coxph fit as ratetable is not available"
    ),
    "survfit_km/lung_sex/summary_table": "mismatch: table[0][4]: 326.08 != 278.76",
    "survfit_km/lung_weighted/summary_table": "mismatch: table[0][4]: 306.45 != 243.32",
    "survfit_km/synthetic_timefix_false/curves.time_counts": (
        "mismatch: curve[1].time: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.surv": (
        "mismatch: curve[1].surv: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.std_err": (
        "mismatch: curve[1].std_err: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.cumhaz": (
        "mismatch: curve[1].cumhaz: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.std_chaz": (
        "mismatch: curve[1].std_chaz: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/curves.conf": (
        "mismatch: curve[1].lower: length 6 differs from expected 8"
    ),
    "survfit_km/synthetic_timefix_false/summary_std_err": (
        "mismatch: summary_std_err: length 5 differs from expected 6"
    ),
    "survfit_multistate/mgus2_400_1/summary_times": (
        "error: NotImplementedError: summary.survfitms is not available (no Rust kern..."
    ),
    "survfit_multistate/mgus2_sex/summary_times": (
        "error: NotImplementedError: summary.survfitms is not available (no Rust kern..."
    ),
    "survfit_multistate/myeloid_ms_trt/summary_times": (
        "error: NotImplementedError: summary.survfitms is not available (no Rust kern..."
    ),
    "survfit_multistate/transplant_abo/summary_times": (
        "error: NotImplementedError: summary.survfitms is not available (no Rust kern..."
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.dfbeta": (
        "mismatch: residuals.dfbeta[0][0]: -0.089422 != -0.097002"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.dfbetas": (
        "mismatch: residuals.dfbetas[0][0]: -0.32531 != -0.35288"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.ldcase": (
        "mismatch: residuals.ldcase[0]: 0.11253 != 0.13123"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.ldresp": (
        "mismatch: residuals.ldresp[0]: 0.27571 != 0.31686"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.ldshape": (
        "mismatch: residuals.ldshape[0]: 0.33231 != 0.55659"
    ),
    "survreg/interval2_synthetic_lognormal_g/residuals.matrix": (
        "mismatch: residuals.matrix[0][3]: -0.26241 != 0.26241"
    ),
    "survreg/interval2_synthetic_lognormal_g/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg/interval2_synthetic_weibull/residuals.dfbeta": (
        "mismatch: residuals.dfbeta[0][0]: -0.059073 != -0.051779"
    ),
    "survreg/interval2_synthetic_weibull/residuals.dfbetas": (
        "mismatch: residuals.dfbetas[0][0]: -0.30257 != -0.26521"
    ),
    "survreg/interval2_synthetic_weibull/residuals.ldcase": (
        "mismatch: residuals.ldcase[0]: 0.1033 != 0.082091"
    ),
    "survreg/interval2_synthetic_weibull/residuals.ldresp": (
        "mismatch: residuals.ldresp[0]: 0.13382 != 0.04104"
    ),
    "survreg/interval2_synthetic_weibull/residuals.ldshape": (
        "mismatch: residuals.ldshape[0]: 0.43207 != 0.12055"
    ),
    "survreg/interval2_synthetic_weibull/residuals.matrix": (
        "mismatch: residuals.matrix[0][3]: 0.32734 != -0.32734"
    ),
    "survreg/interval2_synthetic_weibull/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg/interval_status_synthetic_weibull/coef": "mismatch: coef[0]: 1.4798 != 1.5711",
    "survreg/interval_status_synthetic_weibull/coef_names": "mismatch: coef[0]: 1.4798 != 1.5711",
    "survreg/interval_status_synthetic_weibull/icoef": "mismatch: icoef[0]: 1.4798 != 1.5711",
    "survreg/interval_status_synthetic_weibull/scale": "mismatch: scale[0]: 0.42808 != 0.53481",
    "survreg/interval_status_synthetic_weibull/var": "mismatch: var[0][0]: 0.044858 != 0.038118",
    "survreg/interval_status_synthetic_weibull/loglik": "mismatch: loglik[0]: -7.1063 != -13.97",
    "survreg/interval_status_synthetic_weibull/iter": "mismatch: iter: 6 != 5",
    "survreg/interval_status_synthetic_weibull/df_residual": "mismatch: df_residual: 3 != 8",
    "survreg/interval_status_synthetic_weibull/linear_predictors": (
        "mismatch: linear_predictors: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.response": (
        "mismatch: residuals.response: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.deviance": (
        "mismatch: residuals.deviance: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.dfbeta": (
        "mismatch: residuals.dfbeta: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.dfbetas": (
        "mismatch: residuals.dfbetas: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.working": (
        "mismatch: residuals.working: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.ldcase": (
        "mismatch: residuals.ldcase: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.ldresp": (
        "mismatch: residuals.ldresp: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.ldshape": (
        "mismatch: residuals.ldshape: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/residuals.matrix": (
        "mismatch: residuals.matrix: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.response": (
        "mismatch: predict.response.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.lp": (
        "mismatch: predict.lp.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.quantile": (
        "mismatch: predict.quantile.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/predict.uquantile": (
        "mismatch: predict.uquantile.fit: length 5 differs from expected 10"
    ),
    "survreg/interval_status_synthetic_weibull/summary": (
        "mismatch: summary[(Intercept)].value: 1.4798 != 1.5711"
    ),
    "survreg/interval_status_synthetic_weibull/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg/lung_weibull_factor_ph_ecog/coef": "mismatch: coef[0]: 5.9672 != 6.2576",
    "survreg/lung_weibull_factor_ph_ecog/coef_names": (
        "mismatch: coef: names ['(Intercept)', 'age', 'sex', 'factor(ph.ecog)0', 'factor..."
    ),
    "survreg/lung_weibull_factor_ph_ecog/var": "mismatch: var[0][0]: 0.21528 != 0.21433",
    "survreg/lung_weibull_factor_ph_ecog/predict.terms": (
        "mismatch: predict.terms.se_fit[0][2]: 0.014034 != 0.020588"
    ),
    "survreg/lung_weibull_factor_ph_ecog/predict_newdata.terms": (
        "mismatch: predict_newdata.terms.se_fit[0][2]: 0.0359 != 0.041339"
    ),
    "survreg/lung_weibull_factor_ph_ecog/summary": (
        "mismatch: summary[(Intercept)].value: 5.9672 != 6.2576"
    ),
    "survreg/lung_weibull_strata_sex/concordance.concordance": "error: KeyError: 0",
    "survreg/tobin_gaussian_left/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg-extra/interval2_synthetic_gaussian_g/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg-extra/lung_lognormal_strata_sex_full/concordance.concordance": "error: KeyError: 0",
    "survreg-extra/lung_weibull_offset_sex/predict_newdata.response": (
        "mismatch: predict_newdata.response.fit[0]: 356.23 != 131.05"
    ),
    "survreg-extra/lung_weibull_offset_sex/predict_newdata.lp": (
        "mismatch: predict_newdata.lp.fit[0]: 5.8756 != 4.8756"
    ),
    "survreg-extra/lung_weibull_offset_sex/predict_newdata.quantile": (
        "mismatch: predict_newdata.quantile.fit[0][0]: 54.682 != 20.116"
    ),
    "survreg-extra/lung_weibull_offset_sex/predict_newdata.uquantile": (
        "mismatch: predict_newdata.uquantile.fit[0][0]: 4.0015 != 3.0015"
    ),
    "survreg-extra/lung_weibull_weighted_strata_sex/concordance.concordance": "error: KeyError: 0",
    "survreg-extra/tobin_extreme_left/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg-extra/tobin_logistic_left_full/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg-extra/tobin_t_df6_left/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "survreg-extra/tobin_t_left/concordance.concordance": (
        "missing feature: ValueError: left or interval censored data is not supported"
    ),
    "utilities/bounded_links/blogit_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/bounded_links/bprobit_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/bounded_links/bcloglog_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/bounded_links/blog_linkinv": (
        "missing feature: bounded link inverse functions are not exposed"
    ),
    "utilities/cipoisson": "error: AttributeError: 'list' object has no attribute 'lower'",
    "utilities/statefig": "error: AttributeError: 'StateFigResult' object has no attribute 'x'",
    "validation-extra/anova_lung_model_list": (
        "missing feature: no validation-extra handler for anova_lung_model_list"
    ),
    "validation-extra/survfit_lung_sex_rmean_individual": (
        "missing feature: no validation-extra handler for survfit_lung_sex_rmean_individual"
    ),
    "validation-extra/turnbull_dead_jump": (
        "missing feature: no validation-extra handler for turnbull_dead_jump"
    ),
    "validation-extra/turnbull_dead_jump_groups": (
        "missing feature: no validation-extra handler for turnbull_dead_jump_groups"
    ),
    "yates/lung_ph_ecog_factor/cmat_names": (
        "mismatch: cmat.colnames[0]: 'factor(ph.ecog)1' != 'factor(ph.ecog)1'"
    ),
    "yates/lung_ph_ecog_factor_pop_data/cmat_names": (
        "mismatch: cmat.colnames[0]: 'factor(ph.ecog)1' != 'factor(ph.ecog)1'"
    ),
    "yates/veteran_celltype_lm": "missing feature: yates on a lm fit",
    "yates/veteran_celltype_pop_factorial/cmat_names": (
        "mismatch: cmat.colnames[3]: 'factor(trt)2' != 'factor(trt)2'"
    ),
    "yates/veteran_celltype_pop_sas/cmat_names": (
        "mismatch: cmat.colnames[3]: 'factor(trt)2' != 'factor(trt)2'"
    ),
    "yates/veteran_celltype_predict_risk": (
        "missing feature: yates predict = 'risk' is not implemented (R simulates the coefficients)"
    ),
    "yates/veteran_trt_factor/cmat_names": (
        "mismatch: cmat.colnames[3]: 'factor(trt)2' != 'factor(trt)2'"
    ),
}


def _known_failure_reason(test_id: str) -> str | None:
    if test_id in KNOWN_FAILURES:
        return KNOWN_FAILURES[test_id]
    topic, name, _aspect = test_id.split("/", 2)
    return KNOWN_FAILURES.get(f"{topic}/{name}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FIT_CACHE: dict[str, Any] = {}

_ARG_NAMES = {
    "conf.type": "conf_type",
    "conf.int": "conf_int",
    "conf.lower": "conf_lower",
    "se.fit": "se_fit",
    "start.time": "start_time",
    "cohort.size": "cohort_size",
    "data.frame": "data_frame",
    "na.action": "na_action",
}


def _kwargs(
    args: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    drop: Sequence[str] = (),
    na_omit: bool = True,
) -> dict:
    """Translate R call arguments into Python keyword arguments.

    R's model functions drop rows with missing values (``na.action = na.omit``)
    by default, so ``na_action="omit"`` is passed unless the case says otherwise.
    """

    out: dict[str, Any] = {"na_action": "omit"} if na_omit else {}
    for name, value in args.items():
        if name in drop:
            continue
        key = _ARG_NAMES.get(name, name)
        if isinstance(value, list):
            value = decode_vector(value)
        out[key] = value
    return out


def _cached(key: str, build: Callable[[], Any]) -> Any:
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = build()
    value = _FIT_CACHE[key]
    if isinstance(value, BaseException):
        raise value
    return value


def _fit_key(family: str, case: Mapping[str, Any]) -> str:
    """Cache key for a fit: the same inputs fitted by the same function."""

    return "|".join(
        [
            family,
            str(case.get("dataset")),
            str(case.get("data_ref")),
            str(case.get("rows")),
            str(case.get("formula")),
            str(case.get("args")),
        ]
    )


def _expect(case: Mapping[str, Any], aspect: str) -> Any:
    """Look up ``expected[aspect]`` where aspect may be dotted."""

    value: Any = case["expected"]
    for part in aspect.split("."):
        value = value[part]
    if is_r_error(value):
        pytest.skip(f"R cannot compute this aspect: {value['r_error']}")
    return value


def _aspects_from_expected(expected: Mapping[str, Any], expand: Sequence[str]) -> list[str]:
    out: list[str] = []
    for key, value in expected.items():
        if key in ("newdata", "cox_formula"):
            continue
        if key in expand and isinstance(value, Mapping) and not is_r_error(value):
            out.extend(f"{key}.{sub}" for sub in value)
        else:
            out.append(key)
    return out


def _formula_is_null_model(formula: str) -> bool:
    return formula.split("~", 1)[1].strip() == "1"


def _attr(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise UnsupportedCaseError(f"result has none of the attributes {names}")


def _concordance_of_fit(fit: Any, **kwargs: Any) -> Any:
    try:
        return r.concordance(fit, **kwargs)
    except TypeError as exc:
        raise UnsupportedCaseError(f"concordance(fit) for model objects: {exc}") from exc


def _tt_function(source: str) -> Callable[..., Any]:
    if source.replace(" ", "") == "function(x,t,...)x*log(t+20)":
        return lambda x, t, *args: [xi * math.log(ti + 20) for xi, ti in zip(x, t, strict=True)]
    raise UnsupportedCaseError(f"no Python translation for tt = {source!r}")


# ---------------------------------------------------------------------------
# survfit curve comparison (shared by KM, Cox and multistate curves)
# ---------------------------------------------------------------------------

_CURVE_FIELDS = {
    # r field -> (python attribute candidates, rtol or None for exact)
    "time": (("time",), 0.0),
    "n_risk": (("n_risk",), 0.0),
    "n_event": (("n_event",), 0.0),
    "n_censor": (("n_censor",), 0.0),
    "n_enter": (("n_enter",), 0.0),
    "surv": (("surv", "estimate"), RTOL_COEF),
    "std_err": (("std_err",), RTOL_VAR),
    "cumhaz": (("cumhaz",), RTOL_COEF),
    "std_chaz": (("std_chaz",), RTOL_VAR),
    "lower": (("conf_lower", "lower"), RTOL_VAR),
    "upper": (("conf_upper", "upper"), RTOL_VAR),
    "pstate": (("pstate",), RTOL_COEF),
    "std_err0": (("std_err0",), RTOL_VAR),
    "n_transition": (("n_transition",), 0.0),
}

_CURVE_ASPECTS = {
    "curves.time_counts": ("time", "n_risk", "n_event", "n_censor"),
    "curves.n_enter": ("n_enter",),
    "curves.surv": ("surv",),
    "curves.pstate": ("pstate",),
    "curves.std_err": ("std_err", "std_err0"),
    "curves.cumhaz": ("cumhaz",),
    "curves.std_chaz": ("std_chaz",),
    "curves.conf": ("lower", "upper"),
    "curves.n_transition": ("n_transition",),
}


def _curve_aspects(fit_expected: Mapping[str, Any]) -> list[str]:
    present: set[str] = set()
    for curve in fit_expected["curves"]:
        present.update(curve)
    out = []
    for aspect, fields in _CURVE_ASPECTS.items():
        if any(field in present for field in fields):
            out.append(aspect)
    return out


class _Curve:
    """One stratum of a stacked survfit object: every per-row field sliced to the curve."""

    _ROW_FIELDS = (
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "n_enter",
        "surv",
        "std_err",
        "cumhaz",
        "std_chaz",
        "lower",
        "upper",
        "pstate",
        "n_transition",
    )

    def __init__(self, fit: Any, index: int, start: int, stop: int) -> None:
        for field in self._ROW_FIELDS:
            values = getattr(fit, field, None)
            setattr(self, field, None if values is None else values[start:stop])
        self.n = fit.n[index]
        self.n_id = None if fit.n_id is None else fit.n_id[index]
        for field in ("states", "transitions", "influence_pstate"):
            setattr(self, field, getattr(fit, field, None))
        p0 = getattr(fit, "p0", None)
        self.p0 = None if p0 is None else p0[index]


def _split_curves(fit: Any) -> dict[str, _Curve]:
    """The curves of a survfit object keyed by R's strata names (``"1"`` for a single curve)."""

    strata = fit.strata or {"1": len(fit.time)}
    curves: dict[str, _Curve] = {}
    start = 0
    for index, (name, size) in enumerate(strata.items()):
        curves[name] = _Curve(fit, index, start, start + size)
        start += size
    return curves


def _python_curves(result: Any, expected_curves: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Align the curves of a Python survfit object with the R strata order."""

    curves = _split_curves(result)
    if len(curves) != len(expected_curves):
        raise FixtureMismatchError(f"{len(curves)} python curves, R has {len(expected_curves)}")
    aligned = []
    for expected in expected_curves:
        name = expected["name"]
        if name not in curves:
            raise FixtureMismatchError(f"curve {name!r} not among {list(curves)}")
        aligned.append(curves[name])
    return aligned


def _compare_curve_fields(
    python_curve: Any, expected_curve: Mapping[str, Any], fields: Sequence[str], path: str
) -> None:
    for field in fields:
        if field not in expected_curve:
            continue
        candidates, rtol = _CURVE_FIELDS[field]
        actual = _attr(python_curve, *candidates)
        expected = expected_curve[field]
        if actual is None:
            raise FixtureMismatchError(f"{path}.{field}: python value is None")
        if expected and isinstance(expected[0], list):
            assert_matrix_close(actual, expected, rtol=rtol, path=f"{path}.{field}")
        else:
            assert_close(as_float_list(actual), expected, rtol=rtol, path=f"{path}.{field}")


def _check_curves(result: Any, fit_expected: Mapping[str, Any], aspect: str) -> None:
    fields = _CURVE_ASPECTS[aspect]
    for python_curve, expected_curve in zip(
        _python_curves(result, fit_expected["curves"]), fit_expected["curves"], strict=True
    ):
        _compare_curve_fields(
            python_curve, expected_curve, fields, f"curve[{expected_curve['name']}]"
        )


# ---------------------------------------------------------------------------
# Topic handlers
# ---------------------------------------------------------------------------

HANDLERS: dict[str, TopicHandler] = {}


class TopicHandler:
    topic: str = ""
    expand: tuple[str, ...] = ()

    def __init_subclass__(cls) -> None:
        if cls.topic:
            HANDLERS[cls.topic] = cls()

    def aspects(self, case: Mapping[str, Any]) -> list[str]:
        return _aspects_from_expected(case["expected"], self.expand)

    def check(self, case: Mapping[str, Any], aspect: str) -> None:
        raise NotImplementedError


# --- datasets ---------------------------------------------------------------


class DatasetsHandler(TopicHandler):
    topic = "datasets"

    def aspects(self, case):
        return ["shape", "values"]

    def check(self, case, aspect):
        expected = case["expected"]
        data = load_dataset(case["dataset"])
        if aspect == "shape":
            assert_exact(nrow(data), expected["nrow"], path="nrow")
            r_names = [col["name"] for col in expected["columns"]]
            if list(data) != r_names:
                raise FixtureMismatchError(f"columns {list(data)} != {r_names}")
            return
        for col in expected["columns"]:
            values = data.get(col["name"])
            if values is None:
                raise FixtureMismatchError(f"column {col['name']} missing")
            missing = sum(
                1 for v in values if v is None or (isinstance(v, float) and math.isnan(v))
            )
            assert_exact(missing, col["n_missing"], path=f"{col['name']}.n_missing")
            if col["type"] in ("factor", "character"):
                present = sorted({str(v) for v in values if v is not None})
                if col["type"] == "factor":
                    if present and not set(present) <= set(col["levels"]):
                        raise FixtureMismatchError(
                            f"{col['name']}: values {present[:5]} are not factor levels "
                            f"{col['levels']}"
                        )
                elif present != list(col["levels"]):
                    raise FixtureMismatchError(f"{col['name']}: distinct values differ from R")
            elif col["type"] == "logical":
                total = sum(1 for v in values if v)
                assert_exact(total, col["sum"], path=f"{col['name']}.sum")
            elif col["type"] == "date":
                continue
            else:
                total = sum(
                    float(v)
                    for v in values
                    if not (v is None or (isinstance(v, float) and math.isnan(v)))
                )
                assert_close(total, col["sum"], rtol=1e-10, atol=1e-8, path=f"{col['name']}.sum")


# --- survfit (KM) -----------------------------------------------------------


def _mstate_formula(formula: str, data: Mapping[str, Any]) -> str:
    """Add ``type = "mstate"`` when the status column is an R factor.

    R infers a multi-state response from a factor status column; the formula parser used by
    the finegray / rttright / survcheck handlers still needs the explicit type hint.
    """

    lhs, _, rhs = formula.partition("~")
    inner = lhs.strip()
    if not inner.startswith("Surv(") or "type" in inner:
        return formula
    args = [arg.strip() for arg in inner[5:-1].split(",")]
    status = args[-1]
    if isinstance(data.get(status), RFactor):
        return f'Surv({", ".join(args)}, type = "mstate") ~{rhs}'
    return formula


def _survfit_call(topic: str, case: Mapping[str, Any], *, drop: Sequence[str] = ()) -> Any:
    def build():
        data = case_data(topic, case)
        kwargs = _kwargs(case.get("args", {}), data, drop=drop)
        return r.survfit(case["formula"], data, **kwargs)

    return _cached(_fit_key("survfit", case), build)


def _check_summary_table(fit: Any, expected: Mapping[str, Any]) -> None:
    """``summary(fit)$table``: a matrix with strata row names, or a named vector."""

    table = r.summary_survfit(fit).table
    if "values" in expected:
        assert_exact(table.rownames, expected["rownames"], path="table.rownames")
        assert_exact(table.colnames, expected["colnames"], path="table.colnames")
        assert_matrix_close(table.values, expected["values"], rtol=RTOL_VAR, path="table")
        return
    if table.rownames is not None:
        raise FixtureMismatchError(f"table has strata rows {table.rownames}, R has none")
    assert_exact(table.colnames, list(expected), path="table.colnames")
    assert_close(table.values[0], list(expected.values()), rtol=RTOL_VAR, path="table")


def _check_summary_times(fit: Any, expected: Mapping[str, Any]) -> None:
    summary = r.summary_survfit(fit, times=expected["times"], extend=True)
    for field in (
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "surv",
        "std_err",
        "cumhaz",
        "std_chaz",
        "lower",
        "upper",
        "pstate",
    ):
        if field not in expected:
            continue
        actual = getattr(summary, field)
        if actual is None:
            raise FixtureMismatchError(f"summary.{field} is None")
        rtol = _CURVE_FIELDS[field][1]
        if expected[field] and isinstance(expected[field][0], list):
            assert_matrix_close(actual, expected[field], rtol=rtol, path=f"summary.{field}")
        else:
            assert_close(as_float_list(actual), expected[field], rtol=rtol, path=f"summary.{field}")
    if "strata" in expected:
        assert_exact(summary.strata, expected["strata"], path="summary.strata")


def _check_quantile(fit: Any, expected: Mapping[str, Any]) -> None:
    result = r.quantile_survfit(fit, probs=expected["probs"])
    single = fit.strata is None
    for field in ("quantile", "lower", "upper"):
        if field not in expected:
            continue
        actual = getattr(result, field)
        if actual is None:
            raise FixtureMismatchError(f"quantile.{field} is None")
        rows = expected[field]
        if single and rows and not isinstance(rows[0], list):
            rows = [rows]
        assert_matrix_close(actual, rows, rtol=RTOL_COEF, path=f"quantile.{field}")


class SurvfitKMHandler(TopicHandler):
    topic = "survfit_km"

    def aspects(self, case):
        expected = case["expected"]
        out = _curve_aspects(expected["fit"])
        out.append("fit.n")
        out.extend(key for key in expected if key != "fit")
        return out

    def check(self, case, aspect):
        fit = _survfit_call(self.topic, case)
        expected = case["expected"]
        if aspect.startswith("curves."):
            _check_curves(fit, expected["fit"], aspect)
        elif aspect == "fit.n":
            curves = _python_curves(fit, expected["fit"]["curves"])
            assert_exact([curve.n for curve in curves], expected["fit"]["n"], path="n")
            for field in ("strata", "conf_type", "conf_int", "type", "t0", "logse"):
                if field in expected["fit"]:
                    assert_exact(getattr(fit, field), expected["fit"][field], path=field)
        elif aspect == "summary_std_err":
            # summary(fit)$std.err = surv-scale standard error at event times only
            summary = r.summary_survfit(fit)
            assert_close(
                as_float_list(summary.std_err),
                expected["summary_std_err"],
                rtol=RTOL_VAR,
                path="summary_std_err",
            )
        elif aspect == "summary_table":
            _check_summary_table(fit, expected["summary_table"])
        elif aspect == "summary_times":
            _check_summary_times(fit, expected["summary_times"])
        elif aspect == "quantile":
            _check_quantile(fit, expected["quantile"])
        else:
            raise UnsupportedCaseError(f"unhandled survfit aspect {aspect}")


# --- survfit multistate -----------------------------------------------------


class SurvfitMultistateHandler(TopicHandler):
    topic = "survfit_multistate"

    def aspects(self, case):
        expected = case["expected"]
        out = _curve_aspects(expected["fit"])
        out.extend(["fit.states", "fit.p0", "fit.transitions", "mstate_from_factor"])
        out.extend(key for key in expected if key != "fit")
        return out

    def check(self, case, aspect):
        expected = case["expected"]
        fit = _survfit_call(self.topic, case)
        first = _python_curves(fit, expected["fit"]["curves"])[0]
        if aspect == "mstate_from_factor":
            if not hasattr(fit, "pstate"):
                raise FixtureMismatchError(
                    "factor status column was not treated as a multi-state response"
                )
        elif aspect.startswith("curves."):
            _check_curves(fit, expected["fit"], aspect)
        elif aspect == "fit.states":
            assert_exact(list(first.states), expected["fit"]["states"], path="states")
            for field in ("n", "strata", "conf_type", "conf_int", "type", "t0", "logse"):
                if field in expected["fit"]:
                    assert_exact(getattr(fit, field), expected["fit"][field], path=field)
        elif aspect == "fit.p0":
            p0 = expected["fit"]["p0"]
            if p0 and isinstance(p0[0], list):
                assert_matrix_close(fit.p0, p0, path="p0")
            else:
                assert_close(fit.p0[0], p0, path="p0")
        elif aspect == "fit.transitions":
            table = expected["fit"]["transitions"]
            assert_exact(fit.transitions.rownames, table["rownames"], path="transitions.rownames")
            assert_exact(fit.transitions.colnames, table["colnames"], path="transitions.colnames")
            assert_exact(fit.transitions.values, table["values"], path="transitions")
        elif aspect == "summary_times":
            _check_summary_times(fit, expected["summary_times"])
        elif aspect in ("influence_pstate", "influence_chaz"):
            if aspect == "influence_chaz":
                raise UnsupportedCaseError("survfitAJ reports the influence on pstate only")
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data)
            influence = r.survfit(case["formula"], data, influence=True, **kwargs).influence_pstate
            if influence is None:
                raise UnsupportedCaseError("survfit result has no influence_pstate")
            # R: subjects x times x states; the engine: [cluster][time][state] per curve
            layers = expected[aspect]
            actual = influence[0].values
            for state, layer in enumerate(layers):
                for subject, row in enumerate(layer):
                    assert_close(
                        [actual[subject][t][state] for t in range(len(row))],
                        row,
                        rtol=RTOL_VAR,
                        path=f"{aspect}[{subject}][state {state}]",
                    )
        else:
            raise UnsupportedCaseError(f"unhandled multistate aspect {aspect}")


# --- survfit interval -------------------------------------------------------


class SurvfitIntervalHandler(TopicHandler):
    topic = "survfit_interval"

    def aspects(self, case):
        expected = case["expected"]
        if "fit" in expected:
            return _curve_aspects(expected["fit"])
        return ["time_surv", "counts", "std_err", "conf", "n"]

    def check(self, case, aspect):
        fit = _survfit_call(self.topic, case)
        expected = case["expected"]
        # Turnbull's counts are EM weights, not integers: compare them like estimates
        if aspect == "curves.time_counts":
            for curve, exp in zip(
                _python_curves(fit, expected["fit"]["curves"]),
                expected["fit"]["curves"],
                strict=True,
            ):
                assert_close(curve.time, exp["time"], path=f"curve[{exp['name']}].time")
                for field in ("n_risk", "n_event", "n_censor"):
                    assert_close(
                        getattr(curve, field),
                        exp[field],
                        rtol=RTOL_COEF,
                        path=f"curve[{exp['name']}].{field}",
                    )
        elif aspect.startswith("curves."):
            _check_curves(fit, expected["fit"], aspect)
        elif aspect == "time_surv":
            assert_close(fit.time, expected["time"], path="time")
            assert_close(fit.surv, expected["surv"], path="surv")
        elif aspect == "counts":
            assert_close(fit.n_risk, expected["n_risk"], rtol=RTOL_COEF, path="n_risk")
            assert_close(fit.n_event, expected["n_event"], rtol=RTOL_COEF, path="n_event")
        elif aspect == "std_err":
            assert_close(fit.std_err, expected["std_err"], rtol=RTOL_VAR, path="std_err")
        elif aspect == "conf":
            assert_close(fit.lower, expected["lower"], rtol=RTOL_VAR, path="lower")
            assert_close(fit.upper, expected["upper"], rtol=RTOL_VAR, path="upper")
        elif aspect == "n":
            assert_exact(fit.n, expected["n"], path="n")


# --- survdiff ---------------------------------------------------------------


class SurvdiffHandler(TopicHandler):
    topic = "survdiff"

    def aspects(self, case):
        return ["counts", "var", "chisq", "pvalue", "df", "strata_names"]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data, drop=("expect",))
            if "expect" in case.get("args", {}):
                data = dict(data)
                data["expect"] = decode_vector(case["args"]["expect"])
            return r.survdiff(case["formula"], data, **kwargs)

        result = _cached(_fit_key("survdiff", case), build)
        if aspect == "counts":
            assert_exact(result.n, expected["n"], path="n")
            for field in ("obs", "exp"):
                actual = getattr(result, field)
                if isinstance(expected[field][0], list):
                    assert_matrix_close(actual, expected[field], rtol=RTOL_COEF, path=field)
                else:
                    assert_close(actual, expected[field], rtol=RTOL_COEF, path=field)
        elif aspect == "var":
            if isinstance(expected["var"][0], list):
                assert_matrix_close(result.var, expected["var"], rtol=RTOL_VAR, path="var")
            else:
                assert_close(
                    [row[0] for row in result.var], expected["var"], rtol=RTOL_VAR, path="var"
                )
        elif aspect == "chisq":
            assert_close(result.chisq, expected["chisq"], rtol=RTOL_VAR, path="chisq")
        elif aspect == "pvalue":
            assert_close(result.pvalue, expected["pvalue"], rtol=RTOL_VAR, path="pvalue")
        elif aspect == "df":
            assert_exact(result.df, expected["df"], path="df")
        elif aspect == "strata_names":
            # names(fit$n): the group labels
            assert_exact(result.groups, expected["strata_names"], path="strata_names")


# --- coxph ------------------------------------------------------------------


def _coxph_fit(topic: str, case: Mapping[str, Any]) -> Any:
    def build():
        data = case_data(topic, case)
        args = dict(case.get("args", {}))
        kwargs = _kwargs(args, data, drop=("tt", "init", "nocenter"))
        if "init" in args:
            kwargs["init"] = decode_vector(args["init"])
        if "tt" in args:
            kwargs["tt"] = _tt_function(args["tt"])
        if "nocenter" in args:
            kwargs["nocenter"] = args["nocenter"]  # None is R's NULL: centre everything
        return r.coxph(case["formula"], data, **kwargs)

    return _cached(_fit_key("coxph", case), build)


def _coef_names(fit: Any) -> list[str] | None:
    try:
        return list(r.coef_names(fit))
    except Exception:  # noqa: BLE001 - names are optional for the comparison
        return None


def _check_summary(fit: Any, aspect: str, expected: Mapping[str, Any]) -> None:
    summary = r.model_summary(fit)
    if aspect in ("summary.logtest", "summary.sctest", "summary.waldtest"):
        key = aspect.split(".", 1)[1]
        test = summary[key]
        if key == "waldtest":
            # summary.coxph reports round(fit$wald.test, 2) (R/summary.coxph.R):
            # compare within half a unit of the last reported digit (the extra
            # 1e-9 absorbs R's round-half-even at an exact .xx5); the unrounded
            # statistic is checked by the wald_test aspect.
            assert_close(
                test["test"], expected["test"], rtol=0.0, atol=0.005 + 1e-9, path="waldtest.test"
            )
        else:
            assert_close(test["test"], expected["test"], rtol=RTOL_VAR, path=f"{key}.test")
        assert_exact(test["df"], expected["df"], path=f"{key}.df")
        assert_close(test["pvalue"], expected["pvalue"], rtol=RTOL_VAR, path=f"{key}.pvalue")
    elif aspect == "summary.coefficients":
        rows = summary["coefficients"]
        r_cols = expected["colnames"]
        keys = {
            "coef": ("coef", RTOL_COEF),
            "exp(coef)": ("exp_coef", RTOL_COEF),
            "se(coef)": ("naive_se" if "robust se" in r_cols else "se", RTOL_VAR),
            "robust se": ("robust_se", RTOL_VAR),
            "z": ("z", RTOL_VAR),
            "Pr(>|z|)": ("p", RTOL_VAR),
            "Value": ("value", RTOL_COEF),
            "SE": ("se", RTOL_VAR),
            "Z": ("z", RTOL_VAR),
            "p": ("p", RTOL_VAR),
        }
        if expected["rownames"]:  # R drops the names of a one-column cch model matrix
            assert_exact(
                [row["name"] for row in rows], expected["rownames"], path="coefficients.rows"
            )
        for row, r_row in zip(rows, expected["values"], strict=True):
            for col_name, value in zip(r_cols, r_row, strict=True):
                key, rtol = keys[col_name]
                assert_close(
                    row[key], value, rtol=rtol, path=f"coefficients[{row['name']}].{col_name}"
                )
    elif aspect == "summary.conf_int":
        rows = summary["conf_int"]
        r_cols = expected["colnames"]
        for name, exp_row in zip(expected["rownames"], expected["values"], strict=True):
            row = next((item for item in rows if item["name"] == name), None)
            if row is None:
                raise FixtureMismatchError(f"conf_int has no row {name!r}")
            for col_name, key in (
                ("exp(coef)", "exp(coef)"),
                ("exp(-coef)", "exp(-coef)"),
                ("lower .95", "lower"),
                ("upper .95", "upper"),
            ):
                assert_close(
                    row[key],
                    exp_row[r_cols.index(col_name)],
                    rtol=RTOL_VAR,
                    path=f"conf_int[{name}].{col_name}",
                )
    elif aspect == "summary.concordance":
        concordance = summary["concordance"]
        assert_close(concordance["C"], expected["C"], rtol=RTOL_VAR, path="concordance.C")
        assert_close(concordance["se(C)"], expected["se(C)"], rtol=RTOL_VAR, path="concordance.se")
    elif aspect == "summary.rsq":
        assert_close(
            [summary["rsq"]["rsq"], summary["rsq"]["maxrsq"]],
            [expected["rsq"], expected["maxrsq"]],
            rtol=RTOL_VAR,
            path="rsq",
        )
    elif aspect in ("summary.n", "summary.nevent"):
        assert_exact(summary[aspect.split(".", 1)[1]], expected, path=aspect)
    elif aspect == "summary.used_robust":
        assert_exact(bool(summary["used_robust"]), expected, path=aspect)
    elif aspect == "summary.robscore":
        test = summary["robscore"]
        assert_close(test["test"], expected["test"], rtol=RTOL_VAR, path="robscore.test")
        assert_exact(test["df"], expected["df"], path="robscore.df")
    else:
        raise UnsupportedCaseError(f"unhandled summary aspect {aspect}")


def _check_concordance_result(cc: Any, expected: Mapping[str, Any], aspect: str) -> None:
    sub = aspect.rsplit(".", 1)[1] if "." in aspect else aspect
    if sub == "concordance":
        assert_close(cc.concordance, expected["concordance"], rtol=RTOL_COEF, path="concordance")
    elif sub == "n":
        assert_exact(cc.n, expected["n"], path="n")
    elif sub == "count":
        count = expected["count"]
        rows = cc.count if isinstance(cc.count, list) else [cc.count]
        if "values" in count:
            assert_exact(cc.names, count["rownames"], path="count.rownames")
            actual = [[row[name] for name in count["colnames"]] for row in rows]
            assert_matrix_close(actual, count["values"], rtol=1e-12, path="count")
        else:
            actual = [rows[0][name] for name in count]
            assert_close(actual, list(count.values()), rtol=1e-12, path="count")
    elif sub == "var":
        if cc.var is None:
            raise FixtureMismatchError("var is None")
        if isinstance(cc.var, list):
            assert_matrix_close(cc.var, expected["var"], rtol=RTOL_VAR, path="var")
        else:
            assert_close(cc.var, expected["var"], rtol=RTOL_VAR, path="var")
    elif sub == "cvar":
        assert_close(cc.cvar, expected["cvar"], rtol=RTOL_VAR, path="cvar")
    elif sub == "dfbeta":
        if cc.dfbeta is None:
            raise FixtureMismatchError("dfbeta is None")
        if isinstance(cc.dfbeta[0], list):
            assert_matrix_close(cc.dfbeta, expected["dfbeta"], rtol=RTOL_VAR, path="dfbeta")
        else:
            assert_close(cc.dfbeta, expected["dfbeta"], rtol=RTOL_VAR, path="dfbeta")
    elif sub == "influence":
        if cc.influence is None:
            raise FixtureMismatchError("influence is None")
        assert_matrix_close(cc.influence, expected["influence"], rtol=RTOL_VAR, path="influence")
    elif sub == "timewt":
        return
    else:
        raise UnsupportedCaseError(f"unhandled concordance aspect {sub}")


class CoxphHandler(TopicHandler):
    topic = "coxph"
    expand = ("residuals", "summary", "concordance", "anova", "wtest", "survfit")

    def aspects(self, case):
        expected = case["expected"]
        if "formulas" in case:
            return ["anova_nested"]
        out = []
        for key, value in expected.items():
            if key in ("method", "coef_names"):
                continue
            if key == "wald_test" and value is None:
                continue  # R has no wald.test for a null model
            if key == "coef":
                out.extend(["coef", "coef_names"])
            elif key in ("residuals", "summary", "concordance") and isinstance(value, Mapping):
                if not is_r_error(value):
                    out.extend(f"{key}.{sub}" for sub in value if not is_r_error(value[sub]))
            elif key == "survfit":
                out.extend(_curve_aspects(value))
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        if aspect == "anova_nested":
            self._check_anova_nested(case)
            return
        fit = _coxph_fit(self.topic, case)
        expected = case["expected"]
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "coef_names":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coef"], check_names=True, path="coef"
            )
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "naive_var":
            if fit.naive_var is None:
                raise FixtureMismatchError("fit has no naive variance")
            assert_matrix_close(
                fit.naive_var, expected["naive_var"], rtol=RTOL_VAR, path="naive_var"
            )
        elif aspect == "loglik":
            assert_close(fit.loglik, expected["loglik"], rtol=RTOL_COEF, path="loglik")
        elif aspect == "score":
            assert_close(fit.score, expected["score"], rtol=RTOL_VAR, path="score")
        elif aspect == "iter":
            actual = [] if fit.iter is None else [fit.iter]
            assert_exact(actual, expected["iter"][:1], path="iter")
        elif aspect == "wald_test":
            assert_close(fit.wald_test, expected["wald_test"], rtol=RTOL_VAR, path="wald_test")
        elif aspect == "n":
            assert_exact(fit.n, expected["n"], path="n")
        elif aspect == "nevent":
            assert_exact(fit.nevent, expected["nevent"], path="nevent")
        elif aspect == "means":
            assert_close(fit.means, expected["means"], rtol=RTOL_COEF, path="means")
        elif aspect == "linear_predictors":
            assert_close(
                fit.linear_predictors,
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect == "x":
            assert_matrix_close(fit.x, expected["x"]["values"], rtol=RTOL_COEF, path="x")
        elif aspect.startswith("residuals."):
            _check_cox_residual(fit, aspect.split(".", 1)[1], _expect(case, aspect))
        elif aspect.startswith("summary."):
            _check_summary(fit, aspect, _expect(case, aspect))
        elif aspect.startswith("concordance."):
            _check_concordance_result(r.concordance(fit), _expect(case, "concordance"), aspect)
        elif aspect == "wtest":
            wtest = _expect(case, "wtest")
            actual = r.coxph_wtest(r.vcov(fit), r.coef(fit))
            assert_close(actual.test, wtest["test"], rtol=RTOL_VAR, path="wtest.test")
            assert_exact(actual.df, wtest["df"], path="wtest.df")
            assert_close(
                as_float_list(actual.solve), wtest["solve"], rtol=RTOL_VAR, path="wtest.solve"
            )
        elif aspect == "anova":
            exp = _expect(case, "anova")
            _check_anova(r.anova(fit), exp)
        elif aspect.startswith("curves."):
            _check_cox_curves(r.survfit(fit), expected["survfit"], aspect)
        else:
            raise UnsupportedCaseError(f"unhandled coxph aspect {aspect}")

    def _check_anova_nested(self, case):
        data = case_data(self.topic, case)
        fits = [r.coxph(formula, data, na_action="omit") for formula in case["formulas"]]
        _check_anova(r.anova(*fits), case["expected"]["anova"], nested=True)


def _check_anova(result: Any, expected: Mapping[str, Any], nested: bool = False) -> None:
    rows = list(result.rows)
    assert_close(
        [row.loglik for row in rows], expected["loglik"], rtol=RTOL_COEF, path="anova.loglik"
    )
    assert_close(
        [row.chisq for row in rows][1:], expected["chisq"][1:], rtol=RTOL_VAR, path="anova.chisq"
    )
    assert_close(
        [row.p_value for row in rows][1:], expected["p"][1:], rtol=RTOL_VAR, path="anova.p"
    )
    if not nested:
        assert_exact(
            [row.df for row in rows][1:], [int(v) for v in expected["df"][1:]], path="anova.df"
        )


def _check_cox_residual(fit: Any, kind: str, expected: Any) -> None:
    actual = r.residuals(fit, type=kind)
    if kind in ("schoenfeld", "scaledsch"):
        expected = expected["values"]
    if expected and isinstance(expected[0], list):
        if actual and not isinstance(actual[0], list):
            actual = [[value] for value in actual]
        assert_matrix_close(actual, expected, rtol=RTOL_COEF, path=f"residuals.{kind}")
    else:
        assert_close(as_float_list(actual), expected, rtol=RTOL_COEF, path=f"residuals.{kind}")


class CoxphPredictHandler(TopicHandler):
    topic = "coxph_predict"

    def aspects(self, case):
        expected = case["expected"]
        out = []
        for key, value in expected.items():
            if key in ("coef", "newdata"):
                continue
            if key.startswith("survfit"):
                if is_r_error(value):
                    out.append(key)
                else:
                    out.extend(f"{key}:{aspect}" for aspect in _curve_aspects(value))
            elif key.startswith("predict"):
                out.extend(f"{key}.{sub}" for sub in value)
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        fit = _coxph_fit("coxph", case)
        expected = case["expected"]
        newdata = newdata_frame(expected)
        if aspect.startswith("basehaz_"):
            bh = r.basehaz(fit, centered=aspect == "basehaz_centered")
            exp = _expect(case, aspect)
            assert_close(bh.time, exp["time"], path="basehaz.time")
            assert_close(bh.hazard, exp["hazard"], rtol=RTOL_COEF, path="basehaz.hazard")
            if "strata" in exp:
                labels = bh.strata
                if labels is None or list(labels) != list(exp["strata"]):
                    raise FixtureMismatchError(
                        f"basehaz strata labels differ: {labels and labels[:3]} vs "
                        f"{exp['strata'][:3]}"
                    )
        elif aspect.startswith("survfit"):
            key, curve_aspect = aspect.split(":", 1)
            exp = _expect(case, key)
            kwargs: dict[str, Any] = {}
            if key.endswith("censor_false"):
                kwargs["censor"] = False
            elif key.endswith("stype2"):
                kwargs.update(stype=2, ctype=1)
            elif key.endswith("ctype2"):
                kwargs.update(stype=2, ctype=2)
            if key.startswith("survfit_newdata"):
                kwargs["newdata"] = newdata
                if key.endswith("loglog"):
                    kwargs["conf_type"] = "log-log"
            _check_cox_curves(r.survfit(fit, **kwargs), exp, curve_aspect)
        elif aspect.startswith("predict"):
            key, kind = aspect.split(".", 1)
            exp = _expect(case, aspect)
            nd = newdata if key == "predict_newdata" else None
            if kind == "lp_uncentered":
                actual = r.predict(fit, nd, type="lp", reference="zero")
                assert_close(as_float_list(actual), exp, rtol=RTOL_COEF, path=aspect)
                return
            result = r.predict(fit, nd, type=kind, se_fit=True)
            if kind == "terms":
                assert_matrix_close(result.fit, exp["fit"], rtol=RTOL_COEF, path=f"{aspect}.fit")
                assert_matrix_close(
                    result.se_fit, exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
                assert_close(
                    r.predict_terms_constant(fit),
                    exp["constant"],
                    rtol=RTOL_COEF,
                    path=f"{aspect}.constant",
                )
            else:
                assert_close(result.fit, exp["fit"], rtol=RTOL_COEF, path=f"{aspect}.fit")
                assert_close(result.se_fit, exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit")
        else:
            raise UnsupportedCaseError(f"unhandled coxph_predict aspect {aspect}")


def _check_cox_curves(result: Any, exp: Mapping[str, Any], curve_aspect: str) -> None:
    """Compare a Cox survfit result with R curves.

    Both store one block per stratum laid end to end; a field is a vector (one
    curve) or an ``ntime x ncurve`` matrix (one column per newdata row).
    """

    blocks = list(result.strata.values()) if result.strata is not None else [len(result.time)]
    names = list(result.strata) if result.strata is not None else None
    if len(blocks) != len(exp["curves"]):
        raise FixtureMismatchError(f"{len(blocks)} python strata, R has {len(exp['curves'])}")
    if names is not None:
        r_names = [curve["name"] for curve in exp["curves"]]
        if names != r_names:
            raise FixtureMismatchError(f"strata names differ: {names} vs {r_names}")
    offset = 0
    for block, curve in zip(blocks, exp["curves"], strict=True):
        rows = slice(offset, offset + block)
        offset += block
        for field in _CURVE_ASPECTS[curve_aspect]:
            if field not in curve:
                continue
            candidates, rtol = _CURVE_FIELDS[field]
            value = _attr(result, *candidates)
            if value is None:
                raise FixtureMismatchError(f"survfit.{field} is None")
            actual = list(value)[rows]
            expected = curve[field]
            path = f"curve[{curve['name']}].{field}"
            if expected and isinstance(expected[0], list):
                if actual and not isinstance(actual[0], list):
                    actual = [[item] for item in actual]
                assert_matrix_close(actual, expected, rtol=rtol, path=path)
            else:
                if actual and isinstance(actual[0], list):
                    if len(actual[0]) != 1:
                        raise FixtureMismatchError(
                            f"{path}: {len(actual[0])} python curves, R has 1"
                        )
                    actual = [item[0] for item in actual]
                assert_close(as_float_list(actual), expected, rtol=rtol, path=path)


class CoxphDiagnosticsHandler(TopicHandler):
    topic = "coxph_diagnostics"

    def aspects(self, case):
        expected = case["expected"]
        out = [
            f"residuals.{sub}"
            for sub, value in expected["residuals"].items()
            if not is_r_error(value)
        ]
        for key, value in expected["zph"].items():
            if is_r_error(value):
                continue
            out.extend(f"zph.{key}.{sub}" for sub in ("table", "x", "y", "var"))
        if not is_r_error(expected["detail"]):
            out.extend(f"detail.{sub}" for sub in expected["detail"] if sub != "strata")
        return out

    def check(self, case, aspect):
        fit = _coxph_fit("coxph", case)
        parts = aspect.split(".")
        if parts[0] == "residuals":
            _check_cox_residual(fit, parts[1], _expect(case, aspect))
        elif parts[0] == "zph":
            transform, terms = parts[1].rsplit("_", 1)
            exp = _expect(case, f"zph.{parts[1]}")
            z = r.cox_zph(fit, transform=transform, terms=(terms == "terms"))
            sub = parts[2]
            if sub == "table":
                rows = {row["name"]: row for row in z.table}
                for name, values in zip(
                    exp["table"]["rownames"], exp["table"]["values"], strict=True
                ):
                    if name not in rows:
                        raise FixtureMismatchError(f"zph table has no row {name!r} ({list(rows)})")
                    row = rows[name]
                    assert_close(row["chisq"], values[0], rtol=RTOL_VAR, path=f"zph[{name}].chisq")
                    assert_exact(row["df"], values[1], path=f"zph[{name}].df")
                    assert_close(row["p"], values[2], rtol=RTOL_VAR, path=f"zph[{name}].p")
            elif sub == "x":
                assert_close(z.x, exp["x"], rtol=RTOL_COEF, path="zph.x")
            elif sub == "y":
                assert_matrix_close(z.y, exp["y"], rtol=RTOL_COEF, path="zph.y")
            elif sub == "var":
                assert_matrix_close(z.var, exp["var"], rtol=RTOL_VAR, path="zph.var")
        elif parts[0] == "detail":
            exp = _expect(case, "detail")
            detail = r.coxph_detail(fit)
            sub = parts[1]
            rtol = {
                "time": 0.0,
                "nevent": 0.0,
                "nrisk": 0.0,
                "hazard": RTOL_COEF,
                "varhaz": RTOL_VAR,
                "wtrisk": RTOL_COEF,
                "score": RTOL_COEF,
                "means": RTOL_COEF,
                "imat": RTOL_VAR,
            }[sub]
            actual = getattr(detail, sub)
            expected = exp[sub]
            if sub == "imat":
                # R: nvar x nvar x ntime; Python: per time list of matrices
                if isinstance(expected[0], list) and isinstance(expected[0][0], list):
                    atol = rtol * array_scale(expected)
                    for t, layer in enumerate(expected):
                        assert_matrix_close(
                            actual[t], layer, rtol=rtol, atol=atol, path=f"detail.imat[{t}]"
                        )
                else:
                    assert_close([m[0][0] for m in actual], expected, rtol=rtol, path="detail.imat")
            elif expected and isinstance(expected[0], list):
                assert_matrix_close(actual, expected, rtol=rtol, path=f"detail.{sub}")
            else:
                if actual and isinstance(actual[0], list):
                    actual = [row[0] for row in actual]
                assert_close(as_float_list(actual), expected, rtol=rtol, path=f"detail.{sub}")


class CoxphPenalizedHandler(TopicHandler):
    topic = "coxph_penalized"

    def aspects(self, case):
        expected = case["expected"]
        out = []
        for key, value in expected.items():
            if key in ("method", "penalty", "pterms", "nocenter", "coef_names"):
                continue
            if key == "residuals":
                out.extend(f"residuals.{sub}" for sub in value)
            elif key == "survfit":
                out.extend(_curve_aspects(value)) if not is_r_error(value) else out.append(key)
            elif key == "concordance":
                out.append("concordance.concordance")
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        try:
            fit = _coxph_fit(self.topic, case)
        except ValueError as exc:
            if "unsupported formula term" in str(exc):
                raise UnsupportedCaseError(
                    "penalised Cox terms (ridge/pspline/frailty) are not implemented"
                ) from exc
            raise
        expected = case["expected"]
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "loglik":
            assert_close(fit.loglik, expected["loglik"], rtol=RTOL_COEF, path="loglik")
        elif aspect == "iter":
            assert_exact([fit.iter], expected["iter"][:1], path="iter")
        elif aspect == "wald_test":
            assert_close(fit.wald_test, expected["wald_test"], rtol=RTOL_VAR, path="wald_test")
        elif aspect == "means":
            assert_close(fit.means, expected["means"], rtol=RTOL_COEF, path="means")
        elif aspect == "linear_predictors":
            assert_close(
                fit.linear_predictors,
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect in ("n", "nevent", "score"):
            assert_close(getattr(fit, aspect), expected[aspect], rtol=RTOL_VAR, path=aspect)
        elif aspect.startswith("residuals."):
            _check_cox_residual(fit, aspect.split(".", 1)[1], _expect(case, aspect))
        elif aspect == "concordance.concordance":
            _check_concordance_result(r.concordance(fit), _expect(case, "concordance"), aspect)
        elif aspect in ("predict_lp", "predict_risk"):
            kind = aspect.split("_")[1]
            assert_close(
                r.predict(fit, type=kind), _expect(case, aspect), rtol=RTOL_COEF, path=aspect
            )
        elif aspect.startswith("curves."):
            _check_cox_curves(r.survfit(fit), _expect(case, "survfit"), aspect)
        elif aspect == "basehaz_centered":
            exp = _expect(case, aspect)
            assert_close(
                r.basehaz(fit, centered=True).hazard, exp["hazard"], rtol=RTOL_COEF, path="basehaz"
            )
        else:
            raise UnsupportedCaseError(
                f"penalised Cox terms (ridge/pspline/frailty) are not implemented: {aspect}"
            )


# --- survreg (handler owned by the survreg module) ----------------------------


def _survreg_fit(topic: str, case: Mapping[str, Any]) -> Any:
    def build():
        data = case_data(topic, case)
        kwargs = _kwargs(case.get("args", {}), data)
        return r.survreg(case["formula"], data, **kwargs)

    return _cached(_fit_key("survreg", case), build)


class SurvregHandler(TopicHandler):
    topic = "survreg"

    def aspects(self, case):
        expected = case["expected"]
        if case["name"] == "distribution_functions":
            return [f"dist.{key}" for key in expected]
        out = []
        for key, value in expected.items():
            if key in ("dist", "newdata", "means", "n", "coef_names"):
                continue
            if key == "coef":
                out.extend(["coef", "coef_names"])
            elif key in ("residuals", "predict", "predict_newdata"):
                out.extend(f"{key}.{sub}" for sub, item in value.items() if not is_r_error(item))
            elif key == "concordance":
                out.append("concordance.concordance")
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        expected = case["expected"]
        if aspect.startswith("dist."):
            key = aspect.split(".", 1)[1]
            _check_survreg_distribution(key, expected[key])
            return
        fit = _survreg_fit(self.topic, case)
        if aspect == "coef":
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coef"], path="coef")
        elif aspect == "coef_names":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coef"], check_names=True, path="coef"
            )
        elif aspect == "icoef":
            assert_close(
                as_float_list(_attr(fit, "icoef")), expected["icoef"], rtol=RTOL_COEF, path="icoef"
            )
        elif aspect == "scale":
            assert_close(
                as_float_list(_attr(fit, "scale")), expected["scale"], rtol=RTOL_COEF, path="scale"
            )
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "naive_var":
            assert_matrix_close(
                _attr(fit, "naive_var", "naive_variance"),
                expected["naive_var"],
                rtol=RTOL_VAR,
                path="naive_var",
            )
        elif aspect == "loglik":
            assert_close(
                as_float_list(_attr(fit, "loglik")),
                expected["loglik"],
                rtol=RTOL_COEF,
                path="loglik",
            )
        elif aspect == "iter":
            assert_exact(_attr(fit, "iterations"), expected["iter"], path="iter")
        elif aspect == "df":
            assert_exact(r.degrees_freedom(fit), expected["df"], path="df")
        elif aspect == "df_residual":
            assert_exact(r.df_residual(fit), expected["df_residual"], path="df_residual")
        elif aspect == "parms":
            assert_close(
                as_float_list(_attr(fit, "parms")), expected["parms"], rtol=RTOL_COEF, path="parms"
            )
        elif aspect == "linear_predictors":
            assert_close(
                as_float_list(_attr(fit, "linear_predictors")),
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect == "x":
            assert_matrix_close(_attr(fit, "x"), expected["x"]["values"], rtol=RTOL_COEF, path="x")
        elif aspect.startswith("residuals."):
            kind = aspect.split(".", 1)[1]
            exp = _expect(case, aspect)
            actual = r.residuals(fit, type=kind)
            if isinstance(exp, Mapping):
                assert_matrix_close(actual, exp["values"], rtol=RTOL_COEF, path=aspect)
            else:
                assert_close(as_float_list(actual), exp, rtol=RTOL_COEF, path=aspect)
        elif aspect.startswith("predict"):
            key, kind = aspect.split(".", 1)
            exp = _expect(case, aspect)
            nd = newdata_frame(expected) if key == "predict_newdata" else None
            kwargs: dict[str, Any] = {"type": kind, "se_fit": True}
            if kind in ("quantile", "uquantile"):
                kwargs["p"] = exp["p"]
            result = r.predict(fit, nd, **kwargs)
            fit_values = _attr(result, "fit")
            se_values = _attr(result, "se_fit")
            if isinstance(exp["fit"][0], list):
                assert_matrix_close(fit_values, exp["fit"], rtol=RTOL_COEF, path=f"{aspect}.fit")
                assert_matrix_close(
                    se_values, exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
            else:
                assert_close(
                    as_float_list(fit_values), exp["fit"], rtol=RTOL_COEF, path=f"{aspect}.fit"
                )
                assert_close(
                    as_float_list(se_values), exp["se_fit"], rtol=RTOL_VAR, path=f"{aspect}.se_fit"
                )
        elif aspect == "summary":
            exp = _expect(case, "summary")
            summary = r.model_summary(fit)
            rows = _attr(summary, "coefficients")
            cols = exp["table"]["colnames"]
            for row, values in zip(rows, exp["table"]["values"], strict=True):
                assert_close(
                    row["coef"],
                    values[cols.index("Value")],
                    rtol=RTOL_COEF,
                    path=f"summary[{row['name']}].value",
                )
                assert_close(
                    row["se"],
                    values[cols.index("Std. Error")],
                    rtol=RTOL_VAR,
                    path=f"summary[{row['name']}].se",
                )
                assert_close(
                    row["p"],
                    values[cols.index("p")],
                    rtol=RTOL_VAR,
                    path=f"summary[{row['name']}].p",
                )
        elif aspect == "anova":
            exp = _expect(case, "anova")
            result = r.anova(fit)
            assert_exact(list(_attr(result, "terms")), exp["terms"], path="anova.terms")
            for key in ("loglik", "resid_df", "df", "deviance", "p"):
                assert_close(
                    as_float_list(_attr(result, key)),
                    exp[key],
                    rtol=RTOL_VAR if key == "p" else RTOL_COEF,
                    path=f"anova.{key}",
                )
        elif aspect == "concordance.concordance":
            _check_concordance_result(
                _concordance_of_fit(fit), _expect(case, "concordance"), aspect
            )
        else:
            raise UnsupportedCaseError(f"unhandled survreg aspect {aspect}")


def _check_survreg_distribution(key: str, exp: Mapping[str, Any]) -> None:
    """dsurvreg/psurvreg/qsurvreg samples; ``key`` is ``<dist>_scale<s>`` or ``t_df4_scale1``."""

    dist = "t" if key.startswith("t_df4") else key.rpartition("_scale")[0]
    kwargs: dict[str, Any] = {"mean": exp["mean"], "scale": exp["scale"], "distribution": dist}
    if dist == "t":
        kwargs["parms"] = 4
    assert_close(
        as_float_list(r.dsurvreg(exp["x"], **kwargs)), exp["d"], rtol=RTOL_COEF, path=f"{key}.d"
    )
    assert_close(
        as_float_list(r.psurvreg(exp["x"], **kwargs)), exp["p_"], rtol=RTOL_COEF, path=f"{key}.p"
    )
    assert_close(
        as_float_list(r.qsurvreg(exp["p"], **kwargs)), exp["q"], rtol=RTOL_COEF, path=f"{key}.q"
    )


# --- survreg-extra (key: survreg) ------------------------------------------


class SurvregExtraHandler(SurvregHandler):
    """``test/r/fixtures/survreg-extra.json``: the families, censoring types and
    control options the ``survreg`` topic leaves light; same aspects."""

    topic = "survreg-extra"


# --- end survreg handlers ------------------------------------------------------


# --- concordance ------------------------------------------------------------


class ConcordanceHandler(TopicHandler):
    topic = "concordance"

    def aspects(self, case):
        expected = case["expected"]
        if case["name"] == "coxph_survreg_fits":
            return [f"{fit}.concordance" for fit in expected]
        return [key for key in expected if key != "timewt"]

    def check(self, case, aspect):
        expected = case["expected"]
        if case["name"] == "coxph_survreg_fits":
            data = case_data(self.topic, case)
            key = aspect.split(".")[0]
            if key == "coxph":
                cc = r.concordance(r.coxph(case["formula"], data, na_action="omit"))
            elif key == "coxph_timewt_S":
                cc = r.concordance(r.coxph(case["formula"], data, na_action="omit"), timewt="S")
            elif key == "survreg":
                cc = r.concordance(r.survreg(case["formula"], data, na_action="omit"))
            else:
                cc = r.concordance(
                    r.coxph(case["formula"], data, na_action="omit"),
                    r.survreg(case["formula"], data, na_action="omit"),
                )
            _check_concordance_result(cc, expected[key], aspect)
            return

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data)
            return r.concordance(case["formula"], data, **kwargs)

        cc = _cached(_fit_key("concordance", case), build)
        _check_concordance_result(cc, expected, f"x.{aspect}")


# --- aareg --------------------------------------------------------------------


class AaregHandler(TopicHandler):
    topic = "aareg"

    def aspects(self, case):
        return [key for key in case["expected"] if key not in ("test", "n")]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data)
            if "dfbeta" in expected:
                kwargs["dfbeta"] = True
            return r.aareg(case["formula"], data, **kwargs)

        fit = _cached(_fit_key("aareg", case), build)
        if aspect == "times":
            assert_close(fit.times, expected["times"], path="times")
        elif aspect == "nrisk":
            assert_exact(fit.nrisk, expected["nrisk"], path="nrisk")
        elif aspect == "coefficient":
            assert_exact(
                fit.coefficient_names,
                expected["coefficient"]["colnames"],
                path="coefficient.colnames",
            )
            assert_matrix_close(
                fit.coefficient,
                expected["coefficient"]["values"],
                rtol=RTOL_COEF,
                path="coefficient",
            )
        elif aspect == "test_statistic":
            assert_close(
                fit.test_statistic,
                list(expected["test_statistic"].values()),
                rtol=RTOL_VAR,
                path="test_statistic",
            )
        elif aspect == "test_var":
            assert_matrix_close(fit.test_var, expected["test_var"], rtol=RTOL_VAR, path="test_var")
        elif aspect == "test_var2":
            assert_matrix_close(
                fit.test_var2, expected["test_var2"], rtol=RTOL_VAR, path="test_var2"
            )
        elif aspect == "tweight":
            if isinstance(expected["tweight"][0], list):
                assert_matrix_close(
                    fit.tweight, expected["tweight"], rtol=RTOL_COEF, path="tweight"
                )
            else:
                assert_close(
                    [row[0] for row in fit.tweight],
                    expected["tweight"],
                    rtol=RTOL_COEF,
                    path="tweight",
                )
        elif aspect == "chisq":
            exp = _expect(case, "chisq")
            summary = r.model_summary(fit)
            assert_close(summary["chisq"], exp["chisq"][0][0], rtol=RTOL_VAR, path="chisq")
            table = exp["table"]
            columns = {
                "slope": "slope",
                "coef": "coef",
                "se(coef)": "se",
                "robust se": "robust_se",
                "z": "z",
                "p": "p",
            }
            assert_exact(
                [row["name"] for row in summary["table"]],
                table["rownames"],
                path="summary.rownames",
            )
            for row, values in zip(summary["table"], table["values"], strict=True):
                for name, value in zip(table["colnames"], values, strict=True):
                    assert_close(
                        row[columns[name]],
                        value,
                        rtol=RTOL_VAR,
                        path=f"summary[{row['name']}].{name}",
                    )
        elif aspect == "dfbeta":
            # Python: subjects x nvar x times (R's array layout); R fixture: one
            # subjects x nvar matrix per time.
            for t, layer in enumerate(expected["dfbeta"]):
                actual_layer = [[row[k][t] for k in range(len(row))] for row in fit.dfbeta]
                assert_matrix_close(actual_layer, layer, rtol=RTOL_COEF, path=f"dfbeta[{t}]")
        else:
            raise UnsupportedCaseError(f"unhandled aareg aspect {aspect}")


# --- cch ----------------------------------------------------------------------


class CchHandler(TopicHandler):
    topic = "cch"

    def aspects(self, case):
        out = [
            key
            for key in ("coef", "var", "naive_var", "subcohort_size", "cohort_size", "method")
            if case["expected"].get(key) is not None
        ]
        if not is_r_error(case["expected"].get("summary")):
            out.append("summary.coefficients")
        return out

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            args = case["args"]
            kwargs = {
                "subcoh": args["subcoh"],
                "id": args["id"],
                "method": args["method"],
                "cohort_size": args["cohort.size"],
                "na_action": "omit",
            }
            if "stratum" in args:
                kwargs["stratum"] = args["stratum"]
            return r.cch(case["formula"], data, **kwargs)

        fit = _cached(_fit_key("cch", case), build)
        if aspect == "coef":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coef"], check_names=True, path="coef"
            )
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "naive_var":
            assert_matrix_close(
                fit.naive_var, expected["naive_var"], rtol=RTOL_VAR, path="naive_var"
            )
        elif aspect == "subcohort_size":
            assert_exact(
                list(fit.subcohort_size), expected["subcohort_size"], path="subcohort_size"
            )
        elif aspect == "cohort_size":
            assert_exact(list(fit.cohort_size), expected["cohort_size"], path="cohort_size")
        elif aspect == "method":
            assert_exact(fit.method, expected["method"], path="method")
        elif aspect == "summary.coefficients":
            _check_summary(fit, aspect, _expect(case, aspect))


# --- clogit -------------------------------------------------------------------


class ClogitHandler(TopicHandler):
    topic = "clogit"

    def aspects(self, case):
        return [
            "coef",
            "var",
            "loglik",
            "iter",
            "score",
            "n",
            "nevent",
            "linear_predictors",
            "summary.coefficients",
        ]

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            return r.clogit(case["formula"], data, method=case["args"]["method"], na_action="omit")

        fit = _cached(_fit_key("clogit", case), build)
        if aspect == "coef":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coef"], check_names=True, path="coef"
            )
        elif aspect == "var":
            assert_matrix_close(r.vcov(fit), expected["var"], rtol=RTOL_VAR, path="var")
        elif aspect == "loglik":
            assert_close(fit.loglik, expected["loglik"], rtol=RTOL_COEF, path="loglik")
        elif aspect == "iter":
            assert_exact(fit.iter, expected["iter"], path="iter")
        elif aspect == "score":
            assert_close(fit.score, expected["score"], rtol=RTOL_VAR, path="score")
        elif aspect in ("n", "nevent"):
            assert_exact(getattr(fit, aspect), expected[aspect], path=aspect)
        elif aspect == "linear_predictors":
            assert_close(
                fit.linear_predictors,
                expected["linear_predictors"],
                rtol=RTOL_COEF,
                path="linear_predictors",
            )
        elif aspect == "summary.coefficients":
            _check_summary(fit, aspect, _expect(case, aspect))


# --- finegray (r-data section) --------------------------------------------------


class FinegrayHandler(TopicHandler):
    topic = "finegray"

    def aspects(self, case):
        out = ["frame"]
        if "coxph" in case["expected"]:
            out.extend(["coxph.coef", "coxph.loglik"])
        return out

    def check(self, case, aspect):
        expected = case["expected"]

        def build():
            data = case_data(self.topic, case)
            kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
            return r.finegray(_mstate_formula(case["formula"], data), data, **kwargs)

        frame = _cached(_fit_key("finegray", case), build)
        if aspect == "frame":
            _compare_frames(frame, expected["frame"])
            return
        fit = r.coxph(expected["cox_formula"], frame, weights="fgwt")
        if aspect == "coxph.coef":
            assert_named_values(
                _coef_names(fit), r.coef(fit), expected["coxph"]["coef"], path="coef"
            )
        else:
            assert_close(
                as_float_list(_attr(fit, "log_likelihood")),
                expected["coxph"]["loglik"],
                rtol=RTOL_COEF,
                path="loglik",
            )


# --- survobrien ---------------------------------------------------------------


class SurvobrienHandler(TopicHandler):
    topic = "survobrien"

    def aspects(self, case):
        return ["frame", "coxph_coef"]

    def check(self, case, aspect):
        self.check_in_topic(self.topic, case, aspect)

    @staticmethod
    def check_in_topic(topic: str, case: Mapping[str, Any], aspect: str) -> None:
        expected = case["expected"]
        data = case_data(topic, case)
        frame = r.survobrien(case["formula"], data=data)
        r_frame = decode_frame(expected["frame"])
        if aspect == "frame":
            missing = [name for name in r_frame if name not in frame]
            if missing:
                raise FixtureMismatchError(f"survobrien frame lacks {missing}: {list(frame)}")
            for name, values in r_frame.items():
                if values and isinstance(values[0], str):
                    assert_exact([str(v) for v in frame[name]], values, path=f"frame.{name}")
                else:
                    assert_close(
                        as_float_list(frame[name]), values, rtol=RTOL_COEF, path=f"frame.{name}"
                    )
        else:
            fit = r.coxph(expected["cox_formula"], frame)
            assert_named_values(_coef_names(fit), r.coef(fit), expected["coxph_coef"], path="coef")


# --- yates --------------------------------------------------------------------


def _r_level_label(value: Any) -> str:
    """R's ``as.character`` of a factor level built from a number (``factor(trt)`` -> "1")."""

    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, int | float) and float(value).is_integer():
        return str(int(value))
    return str(value)


class YatesHandler(TopicHandler):
    topic = "yates"

    def aspects(self, case):
        return ["estimate", "test", "mvar", "cmat", "cmat_names"]

    def check(self, case, aspect):
        self.check_in_topic(self.topic, case, aspect)

    @staticmethod
    def check_in_topic(topic: str, case: Mapping[str, Any], aspect: str) -> None:
        if case.get("fit", "coxph") != "coxph":
            raise UnsupportedCaseError(f"yates on a {case['fit']} fit")
        args = dict(case.get("args", {}))
        term = args.pop("term")
        expected = case["expected"]
        # yates reads the model frame of the fit (R re-evaluates the call; Python keeps it)
        fit = _coxph_fit(topic, {**case, "args": {"model": True}})

        def build():
            try:
                return r.yates(fit, term, **args)
            except NotImplementedError as exc:
                raise UnsupportedCaseError(str(exc)) from exc

        result = _cached(_fit_key("yates", {**case, "args": {"term": term, **args}}), build)
        if aspect == "estimate":
            columns = expected["estimate"]["columns"]
            for name, values in columns.items():
                if name not in result.estimate:
                    raise FixtureMismatchError(f"estimate lacks column {name!r}")
                if name in ("pmm", "std"):
                    rtol = RTOL_COEF if name == "pmm" else RTOL_VAR
                    assert_close(
                        as_float_list(result.estimate[name]),
                        values,
                        rtol=rtol,
                        path=f"estimate.{name}",
                    )
                else:
                    actual = [_r_level_label(v) for v in result.estimate[name]]
                    assert_exact(
                        actual, [_r_level_label(v) for v in values], path=f"estimate.{name}"
                    )
        elif aspect == "test":
            table = expected["test"]
            assert_exact([row.name for row in result.test], table["rownames"], path="test.names")
            actual = [[row.chisq, float(row.df)] for row in result.test]
            values = [row[:2] for row in table["values"]]
            assert_matrix_close(actual, values, rtol=RTOL_VAR, path="test")
        elif aspect == "mvar":
            assert_matrix_close(result.mvar, expected["mvar"], rtol=RTOL_VAR, path="mvar")
        elif aspect == "cmat":
            assert_matrix_close(
                result.cmat, expected["cmat"]["values"], rtol=RTOL_COEF, path="cmat"
            )
        else:
            assert_exact(result.cmat_names, expected["cmat"]["colnames"], path="cmat.colnames")


# --- royston / brier ----------------------------------------------------------

_ROYSTON_FIELDS = {
    "D": "d",
    "se(D)": "se_d",
    "R.D": "r_d",
    "R.KO": "r_ko",
    "R.N": "r_n",
    "C.GH": "c_gh",
}


class RoystonBrierHandler(TopicHandler):
    topic = "royston_brier"

    def aspects(self, case):
        return list(case["expected"])

    def check(self, case, aspect):
        expected = case["expected"]
        fit = _coxph_fit(self.topic, {**case, "args": {}})
        if aspect.startswith("royston"):
            result = r.royston(fit, adjust=aspect.endswith("adjust"))
            for name, value in expected[aspect].items():
                actual = getattr(result, _ROYSTON_FIELDS[name])
                if actual is None:
                    raise FixtureMismatchError(f"royston result lacks {name!r}")
                assert_close(actual, value, rtol=RTOL_VAR, path=f"{aspect}.{name}")
        else:
            exp = expected[aspect]
            kwargs: dict[str, Any] = {}
            if aspect != "brier_default_times":
                kwargs["times"] = exp["times"]
            if aspect == "brier_ties_false":
                kwargs["ties"] = False
            result = r.brier(fit, **kwargs)
            assert_close(as_float_list(result.times), exp["times"], path=f"{aspect}.times")
            assert_close(
                as_float_list(result.brier), exp["brier"], rtol=RTOL_VAR, path=f"{aspect}.brier"
            )
            assert_close(
                as_float_list(result.rsquared),
                exp["rsquared"],
                rtol=RTOL_VAR,
                path=f"{aspect}.rsquared",
            )


# --- pseudo / residuals.survfit / survfit0 -------------------------------------


class PseudoHandler(TopicHandler):
    topic = "pseudo"

    def aspects(self, case):
        out = []
        for key, value in case["expected"].items():
            if key == "times" or is_r_error(value):
                continue
            if key == "survfit0":
                out.extend(f"survfit0:{aspect}" for aspect in _curve_aspects(value))
            else:
                out.append(key)
        return out

    def check(self, case, aspect):
        expected = case["expected"]
        fit = _survfit_call(self.topic, case, drop=("times",))
        times = expected["times"]
        if aspect.startswith("survfit0"):
            _check_curves(r.survfit0(fit), expected["survfit0"], aspect.split(":", 1)[1])
            return
        kind, type_name = aspect.split("_", 1)
        exp = expected[aspect]
        if kind == "pseudo":
            result = r.pseudo(fit, times=times, type=type_name)
        else:
            result = r.survfit_residuals(fit, times=times, type=type_name).resid
        if exp and isinstance(exp[0], list) and isinstance(exp[0][0], list):
            # R stores the subjects x states x times array as one matrix per time; the
            # Python layout is subject, then state, then time
            for t, layer in enumerate(exp):
                for subject, row in enumerate(layer):
                    actual = [
                        result[subject][state][t] if len(times) > 1 else result[subject][state]
                        for state in range(len(row))
                    ]
                    assert_close(actual, row, rtol=RTOL_VAR, path=f"{aspect}[{subject}][t {t}]")
        elif exp and isinstance(exp[0], list):
            assert_matrix_close(result, exp, rtol=RTOL_VAR, path=aspect)
        else:
            assert_close(as_float_list(result), exp, rtol=RTOL_VAR, path=aspect)


# --- aggregate.survfit ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _CurveMatrices:
    """The ``data`` margin of ``survfit(coxfit, newdata)`` as ``aggregate.survfit`` reads it."""

    surv: list[list[float]] | None = None
    pstate: list[list[list[float]]] | None = None
    newdata: Any | None = None


def _layers_to_array(layers: Sequence[Sequence[Sequence[float]]]) -> list[list[list[float]]]:
    """R's ``time x data x state`` array (one ``time x data`` matrix per state) as
    ``[time][data][state]``."""

    n_time = len(layers[0])
    n_data = len(layers[0][0])
    return [[[layer[t][j] for layer in layers] for j in range(n_data)] for t in range(n_time)]


class AggregateSurvfitHandler(TopicHandler):
    topic = "km-aggregate_survfit"

    def aspects(self, case):
        return list(case["expected"])

    def check(self, case, aspect):
        args = case["args"]
        expected = case["expected"]
        curves = _CurveMatrices(
            surv=decode_vector(args["surv"]) if "surv" in args else None,
            pstate=_layers_to_array(decode_vector(args["pstate"])) if "pstate" in args else None,
        )
        by = args.get("by")
        if isinstance(by, dict):
            by = {name: decode_vector(by[name]) for name in args.get("by_names", list(by))}
        elif by is not None:
            by = decode_vector(by)
        result = r.aggregate_survfit(curves, by=by, FUN=args.get("fun", "mean"))
        if aspect == "surv":
            exp = expected["surv"]
            if isinstance(exp[0], list):
                assert_matrix_close(result.surv, exp, rtol=RTOL_COEF, path="surv")
            else:
                # no by: R drops the data margin
                assert_close([row[0] for row in result.surv], exp, rtol=RTOL_COEF, path="surv")
        elif aspect == "pstate":
            exp = expected["pstate"]
            if isinstance(exp[0][0], list):
                assert_matrix_close(
                    [[v for group in row for v in group] for row in result.pstate],
                    [[v for group in row for v in group] for row in _layers_to_array(exp)],
                    rtol=RTOL_COEF,
                    path="pstate",
                )
            else:
                # no by: a time x state matrix
                assert_matrix_close(
                    [row[0] for row in result.pstate], exp, rtol=RTOL_COEF, path="pstate"
                )
        elif aspect == "newdata":
            frame = newdata_frame(expected)
            if frame is None:
                assert_exact(result.newdata, None, path="newdata")
            else:
                assert_exact(
                    {name: [str(v) for v in values] for name, values in result.newdata.items()},
                    {name: [str(v) for v in values] for name, values in frame.items()},
                    path="newdata",
                )


# --- survcheck ----------------------------------------------------------------


def _check_named_table(actual_rows, actual_cols, actual_values, table, path: str) -> None:
    assert_exact(list(actual_rows), table["rownames"], path=f"{path}.rownames")
    assert_exact([str(c) for c in actual_cols], table["colnames"], path=f"{path}.colnames")
    assert_matrix_close(actual_values, table["values"], rtol=0.0, path=path)


class SurvcheckHandler(TopicHandler):
    topic = "survcheck"

    def aspects(self, case):
        return ["states", "transitions", "events", "flag", "istate", "n", "problems"]

    def check(self, case, aspect):
        self.check_in_topic(self.topic, case, aspect)

    @staticmethod
    def check_in_topic(topic: str, case: Mapping[str, Any], aspect: str) -> None:
        expected = case["expected"]

        def build():
            data = case_data(topic, case)
            kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
            return r.survcheck(_mstate_formula(case["formula"], data), data, **kwargs)

        result = _cached(_fit_key("survcheck", case), build)
        if aspect == "states":
            assert_exact(list(result.states), expected["states"], path="states")
        elif aspect == "istate":
            assert_exact(list(result.istate), expected["istate"], path="istate")
        elif aspect == "transitions":
            table = result.transitions
            _check_named_table(
                table.from_states,
                table.to_states,
                table.counts,
                expected["transitions"],
                "transitions",
            )
        elif aspect == "events":
            if result.events is None:
                if expected["events"] is not None:
                    raise FixtureMismatchError("no events table")
                return
            events = result.events
            _check_named_table(
                events.states, events.count, events.subjects, expected["events"], "events"
            )
        elif aspect == "flag":
            for name, value in expected["flag"].items():
                assert_exact(getattr(result.flag, name), value, path=f"flag.{name}")
        elif aspect == "n":
            assert_exact(result.n, expected["n"], path="n")
        else:
            for name in ("overlap", "gap", "teleport", "jump"):
                problem = getattr(result, name)
                if expected[name] is None:
                    if problem is not None:
                        raise FixtureMismatchError(f"{name}: unexpected problems {problem}")
                    continue
                if problem is None:
                    raise FixtureMismatchError(f"{name}: expected problems {expected[name]}")
                assert_exact(problem.row, expected[name]["row"], path=f"{name}.row")
                assert_exact(
                    as_float_list(problem.id),
                    as_float_list(expected[name]["id"]),
                    path=f"{name}.id",
                )


# --- validation-extra (survobrien, yates and survcheck cases; the anova, survfit and
#     Turnbull cases belong to the coxph and survfit handlers) -------------------


class ValidationExtraHandler(TopicHandler):
    topic = "validation-extra"
    delegates = {
        "survobrien_": SurvobrienHandler,
        "yates_": YatesHandler,
        "survcheck_": SurvcheckHandler,
    }

    def _delegate(self, case):
        for prefix, handler in self.delegates.items():
            if case["name"].startswith(prefix):
                return handler
        return None

    def aspects(self, case):
        handler = self._delegate(case)
        return HANDLERS[handler.topic].aspects(case) if handler else ["(no handler)"]

    def check(self, case, aspect):
        handler = self._delegate(case)
        if handler is None:
            raise UnsupportedCaseError(f"no validation-extra handler for {case['name']}")
        handler.check_in_topic(self.topic, case, aspect)


# --- survSplit / survcondense / tmerge / neardate (r-data section) --------------
# Owner: the r-data module (python/survival/r/_data_prep.py).


def _compare_frames(
    actual: Mapping[str, Any],
    expected_frame: Mapping[str, Any],
    *,
    rtol: float = RTOL_COEF,
    columns: Sequence[str] | None = None,
    renames: Mapping[str, str] | None = None,
) -> None:
    """Compare a column dict against a fixture frame (``renames`` maps R names to ours)."""

    r_frame = decode_frame(expected_frame)
    if nrow(actual) != nrow(r_frame):
        raise FixtureMismatchError(f"rows {nrow(actual)} != {nrow(r_frame)}")
    for name, values in r_frame.items():
        if columns is not None and name not in columns:
            continue
        our_name = (renames or {}).get(name, name)
        if our_name not in actual:
            raise FixtureMismatchError(f"frame lacks column {name!r} (has {list(actual)})")
        actual_values = list(actual[our_name])
        if isinstance(values, RFactor) or (values and isinstance(values[0], str)):
            if [str(v) for v in actual_values] != [str(v) for v in values]:
                raise FixtureMismatchError(f"column {name!r} differs")
        else:
            assert_close(as_float_list(actual_values), values, rtol=rtol, path=name)


class SurvSplitHandler(TopicHandler):
    topic = "survSplit"

    def aspects(self, case):
        return ["frame"]

    def check(self, case, aspect):
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        frame = r.survSplit(case["formula"], data, **kwargs)
        _compare_frames(frame, case["expected"]["frame"])


class SurvcondenseHandler(TopicHandler):
    topic = "survcondense"

    def aspects(self, case):
        return ["frame"]

    def check(self, case, aspect):
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        frame = r.survcondense(case["formula"], data, **kwargs)
        # R names the id column after the deparsed ``id`` argument, which the
        # generator's do.call turns into the first id value ("1").
        _compare_frames(frame, case["expected"]["frame"], renames={"1": kwargs["id"]})


def _check_tcount(frame: Any, expected: Mapping[str, Any], path: str) -> None:
    """The ``tcount`` attribute: one row per tmerge argument in call order."""

    names = list(expected["rownames"])
    actual = [
        [frame.tcount[name][column] for column in expected["colnames"]]
        for name in dict.fromkeys(names)
    ]
    rows = [list(row) for row in expected["values"]]
    # R stacks one row per argument, repeats included; ours keys by name.
    seen: dict[str, int] = {}
    unique_rows = []
    for name, row in zip(names, rows, strict=True):
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == names.count(name):
            unique_rows.append(row)
    assert_matrix_close(actual, unique_rows, rtol=0.0, path=path)


class TmergeHandler(TopicHandler):
    topic = "tmerge"

    def aspects(self, case):
        return [key for key in case["expected"] if key != "matches_cgd"]

    def check(self, case, aspect):
        expected = case["expected"]
        if case["name"] == "cgd0_vignette":
            cgd0 = load_dataset("cgd0")
            base = {name: cgd0[name] for name in list(cgd0)[:13]}
            frame = r.tmerge(base, cgd0, id="id", tstop="futime")
            if aspect == "after_base":
                return _compare_frames(frame, expected["after_base"])
            for k in range(1, 8):
                frame = r.tmerge(frame, cgd0, id="id", infect=r.event(f"etime{k}"))
            if aspect == "after_events":
                return _compare_frames(frame, expected["after_events"])
            frame = r.tmerge(frame, frame, id="id", enum=r.cumtdc("tstart"))
            if aspect == "tcount":
                return _check_tcount(frame, expected["tcount"], "tcount")
            return _compare_frames(frame, expected["final"])
        doc_data = case_data(self.topic, {"data_ref": case["data_ref"]})
        long = decode_frame(load_topic(self.topic)["data"][case["data_ref2"]])
        if case["name"] == "pbcseq_20_vignette":
            death = [value == 2 for value in doc_data["status"]]
            frame = r.tmerge(doc_data, doc_data, id="id", death=r.event("time", death))
            frame = r.tmerge(
                frame,
                long,
                id="id",
                bili=r.tdc("day", "bili"),
                albumin=r.tdc("day", "albumin"),
                protime=r.tdc("day", "protime"),
                edema=r.tdc("day", "edema"),
            )
            if aspect == "tcount":
                return _check_tcount(frame, expected["tcount"], "tcount")
            return _compare_frames(frame, expected["frame"])
        d1 = r.tmerge(doc_data, doc_data, id="id", death=r.event("futime", "death"))
        if aspect == "step1_death_event":
            return _compare_frames(d1, expected[aspect])
        if aspect == "tdc_init":
            return _compare_frames(
                r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab", init=0.5)), expected[aspect]
            )
        if aspect == "tdc_tdcstart":
            return _compare_frames(
                r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab"), options={"tdcstart": -1}),
                expected[aspect],
            )
        d2 = r.tmerge(d1, long, id="id", lab=r.tdc("time", "lab"))
        if aspect == "step2_lab_tdc":
            return _compare_frames(d2, expected[aspect])
        d3 = r.tmerge(d2, long, id="id", nlab=r.cumtdc("time"))
        if aspect == "step3_nlab_cumtdc":
            return _compare_frames(d3, expected[aspect])
        d4 = r.tmerge(d3, long, id="id", infect=r.event("time", "infection"))
        if aspect == "step4_infect_event":
            return _compare_frames(d4, expected[aspect])
        d5 = r.tmerge(d4, long, id="id", ninfect=r.cumevent("time", "infection"))
        if aspect == "tcount_final":
            return _check_tcount(d5, expected["tcount_final"], "tcount_final")
        return _compare_frames(d5, expected["step5_ninfect_cumevent"])


class DataprepTmergeHandler(TopicHandler):
    topic = "dataprep-tmerge"

    def aspects(self, case):
        return list(case["expected"])

    def check(self, case, aspect):
        expected = case["expected"]
        base = case_data(self.topic, {"data_ref": case["data_ref"]})
        long = decode_frame(load_topic(self.topic)["data"][case["data_ref2"]])
        if case["name"] == "lvcf_after_event_split":
            d1 = r.tmerge(base, base, id="id", death=r.event("futime", "death"))
            d2 = r.tmerge(d1, long, id="id", visit=r.event("time", "visit"))
            if aspect == "step2_visit_event":
                return _compare_frames(d2, expected[aspect])
            if aspect == "x_tdc_na_kept":
                frame = r.tmerge(d2, long, id="id", x=r.tdc("time", "x"), options={"na.rm": False})
                return _compare_frames(frame, expected[aspect])
            if aspect == "x_tdc_init_0":
                return _compare_frames(
                    r.tmerge(d2, long, id="id", x=r.tdc("time", "x", 0)), expected[aspect]
                )
            d3 = r.tmerge(d2, long, id="id", x=r.tdc("time", "x"))
            if aspect == "step3_x_tdc":
                return _compare_frames(d3, expected[aspect])
            d4 = r.tmerge(d3, long, id="id", nx=r.cumtdc("time", "x"))
            if aspect == "step4_nx_cumtdc":
                return _compare_frames(d4, expected[aspect])
            d5 = r.tmerge(d4, long, id="id", seen=r.tdc("time"))
            if aspect == "tcount_step5":
                return _check_tcount(d5, expected[aspect], aspect)
            return _compare_frames(d5, expected["step5_seen_tdc"])
        death = [value == 2 for value in base["status"]]
        pbc2 = r.tmerge(base, base, id="id", death=r.event("time", death))
        if aspect == "chol_na_kept":
            frame = r.tmerge(
                pbc2, long, id="id", chol=r.tdc("day", "chol"), options={"na.rm": False}
            )
            return _compare_frames(frame, expected[aspect])
        pbc3 = r.tmerge(
            pbc2,
            long,
            id="id",
            bili=r.tdc("day", "bili"),
            chol=r.tdc("day", "chol"),
            ascites=r.tdc("day", "ascites"),
            hepato=r.tdc("day", "hepato"),
        )
        if aspect == "tcount":
            return _check_tcount(pbc3, expected["tcount"], "tcount")
        return _compare_frames(pbc3, expected["frame"])


class NeardateHandler(TopicHandler):
    topic = "neardate"

    def aspects(self, case):
        return [key for key in case["expected"] if key != "after_dates"]

    def check(self, case, aspect):
        args = case["args"]
        best, _, nomatch = aspect.partition("_")
        kwargs: dict[str, Any] = {"best": best}
        if nomatch == "nomatch0":
            kwargs["nomatch"] = -1  # R's nomatch = 0 is one-based; ours is zero-based
        result = r.neardate(args["id1"], args["id2"], args["y1"], args["y2"], **kwargs)
        # R reports one-based row numbers, Python zero-based ones.
        actual = [None if value is None else value + 1 for value in result]
        assert_exact(actual, case["expected"][aspect], path=aspect)


# --- rttright (r-data section) ----------------------------------------------------


class RttrightHandler(TopicHandler):
    topic = "rttright"

    def aspects(self, case):
        return ["weights"]

    def check(self, case, aspect):
        expected = case["expected"]
        data = case_data(self.topic, case)
        kwargs = _kwargs(case.get("args", {}), data, na_omit=False)
        result = r.rttright(_mstate_formula(case["formula"], data), data=data, **kwargs)
        if "times" in expected and len(expected["times"]) > 1:
            assert_matrix_close(result, expected["weights"], rtol=RTOL_COEF, path="weights")
        else:
            assert_close(as_float_list(result), expected["weights"], rtol=RTOL_COEF, path="weights")


# --- pyears / survexp (r-data section) --------------------------------------------


def _pyears_call(case: Mapping[str, Any]) -> Any:
    data = case_data("pyears", case)
    args = dict(case.get("args", {}))
    kwargs = _kwargs(args, data, na_omit=False, drop=("ratetable", "rmap"))
    if "ratetable" in args:
        kwargs["ratetable"] = _ratetable_by_name(args["ratetable"])
        kwargs["rmap"] = _rmap_arguments(args["rmap"])
    return r.pyears(case["formula"], data, **kwargs)


def _ratetable_by_name(name: str) -> Any:
    tables = {"survexp.us": r.survexp_us, "survexp.usr": r.survexp_usr, "survexp.mn": r.survexp_mn}
    if name not in tables:
        raise UnsupportedCaseError(f"ratetable {name} is not a bundled rate table")
    return tables[name]()


def _rmap_arguments(rmap: str) -> dict[str, Any]:
    """``list(age = agedays, race = "white")`` as a Python mapping."""

    inner = rmap.strip()[len("list(") : -1]
    out: dict[str, Any] = {}
    for part in inner.split(","):
        name, _sep, value = part.partition("=")
        value = value.strip()
        out[name.strip()] = value.strip('"') if value.startswith('"') else value
    return out


class PyearsHandler(TopicHandler):
    topic = "pyears"

    def aspects(self, case):
        expected = case["expected"]
        if case["name"] == "tcut_basis":
            return ["tcut"]
        if "data" in expected:
            return ["data", "offtable", "observations"]
        return [
            key
            for key in ("pyears", "n", "event", "expected", "offtable", "observations", "dimnames")
            if key in expected
        ]

    def check(self, case, aspect):
        expected = case["expected"]
        if case["name"] == "tcut_basis":
            args = case["args"]
            tc = r.tcut(args["x"], args["breaks"])
            assert_exact(as_float_list(tc.values), expected["values"], path="values")
            assert_exact(as_float_list(tc.cutpoints), expected["cutpoints"], path="cutpoints")
            assert_exact(list(tc.labels), expected["labels"], path="labels")
            labelled = r.tcut(args["x"], args["breaks"], labels=expected["labelled"])
            assert_exact(list(labelled.labels), expected["labelled"], path="labelled")
            three = r.tcut(args["x"], 3)
            assert_close(
                as_float_list(three.cutpoints),
                expected["scalar_breaks_cutpoints"],
                rtol=1e-12,
                path="scalar_breaks_cutpoints",
            )
            assert_exact(
                as_float_list(three.values), expected["scalar_breaks_values"], path="values"
            )
            return
        result = _cached(_fit_key("pyears", case), lambda: _pyears_call(case))
        if aspect == "data":
            _compare_frames(result.data, expected["data"], rtol=RTOL_COEF)
            return
        if aspect == "dimnames":
            assert_exact(list(result.dim), expected["dim"], path="dim")
            for label, levels in expected["dimnames"].items():
                if label not in result.dimnames:
                    raise FixtureMismatchError(
                        f"dimnames lacks {label!r} ({list(result.dimnames)})"
                    )
                assert_exact(list(result.dimnames[label]), levels, path=f"dimnames[{label}]")
            return
        actual = getattr(result, aspect)
        exp = expected[aspect]
        rtol = RTOL_COEF if aspect in ("pyears", "expected", "offtable") else 0.0
        if isinstance(exp, list) and exp and isinstance(exp[0], list):
            assert_matrix_close(actual, exp, rtol=rtol, path=aspect)
        elif isinstance(exp, list):
            assert_close(as_float_list(actual), exp, rtol=rtol, path=aspect)
        else:
            assert_close(float(actual), exp, rtol=rtol, path=aspect)


class SurvexpHandler(TopicHandler):
    topic = "survexp"

    def aspects(self, case):
        name = case["name"]
        if name == "ratetableDate":
            return ["from_date", "from_numeric"]
        if name == "survexp_us_table":
            return ["dim", "dimnames", "type", "cutpoints", "sample", "summary"]
        if name == "lung_coxph_ratetable":
            return ["by_sex", "overall", "individual"]
        if "time" not in case["expected"]:
            return ["surv"]
        return ["time", "n", "surv", "n_risk", "method"]

    def check(self, case, aspect):
        expected = case["expected"]
        name = case["name"]
        if name == "ratetableDate":
            args = case["args"]
            if aspect == "from_numeric":
                result = r.ratetableDate(args["numeric"])
            else:
                import datetime

                result = r.ratetableDate([datetime.date.fromisoformat(v) for v in args["dates"]])
            assert_close(as_float_list(result), expected[aspect], rtol=1e-12, path=aspect)
            return
        if name == "survexp_us_table":
            self._check_table(aspect, expected)
            return
        if name == "lung_coxph_ratetable":
            raise UnsupportedCaseError("survexp with a coxph fit as ratetable is not available")
        args = dict(case.get("args", {}))
        data = case_data(self.topic, case)
        kwargs = _kwargs(args, data, na_omit=False, drop=("ratetable", "rmap"))
        kwargs["ratetable"] = _ratetable_by_name(args["ratetable"])
        kwargs["rmap"] = _rmap_arguments(args["rmap"])
        result = _cached(
            _fit_key("survexp", case), lambda: r.survexp(case["formula"], data, **kwargs)
        )
        if "time" not in expected:
            assert_close(as_float_list(result), expected["surv"], rtol=RTOL_VAR, path="surv")
            return
        if aspect == "method":
            assert_exact(result.method, expected["method"], path="method")
        elif aspect == "time":
            assert_close(as_float_list(result.time), expected["time"], rtol=1e-12, path="time")
        elif aspect == "n":
            n_risk = result.n_risk
            if n_risk and isinstance(n_risk[0], list):
                flat = [row[g] for g in range(len(n_risk[0])) for row in n_risk]
            else:
                flat = list(n_risk)
            assert_exact(as_float_list(flat), expected["n"], path="n")
        elif "strata_names" in expected:
            assert_exact(list(result.strata), expected["strata_names"], path="strata")
            rtol = RTOL_VAR if aspect == "surv" else 0.0
            assert_matrix_close(getattr(result, aspect), expected[aspect], rtol=rtol, path=aspect)
        else:
            rtol = RTOL_VAR if aspect == "surv" else 0.0
            assert_close(
                as_float_list(getattr(result, aspect)), expected[aspect], rtol=rtol, path=aspect
            )

    @staticmethod
    def _check_table(aspect: str, expected: Mapping[str, Any]) -> None:
        table = r.survexp_us()
        if aspect == "dim":
            assert_exact(list(table.dims), expected["dim"], path="dim")
        elif aspect == "type":
            assert_exact(list(table.type_codes()), expected["type"], path="type")
        elif aspect == "dimnames":
            for d, dimid in enumerate(table.dimid):
                assert_exact(list(table.dimnames[d]), expected["dimnames"][dimid], path=dimid)
        elif aspect == "cutpoints":
            for d, cuts in enumerate(expected["cutpoints"]):
                actual = table.cutpoints[d]
                if cuts is None:
                    assert_exact(actual, None, path=f"cutpoints[{d}]")
                else:
                    assert_close(as_float_list(actual), cuts, rtol=0.0, path=f"cutpoints[{d}]")
        elif aspect == "sample":
            sample = expected["sample"]

            def at(labels: Sequence[str]) -> float:
                index = [list(table.dimnames[d]).index(label) for d, label in enumerate(labels)]
                return float(table.rate(index))

            assert_close(at(["0", "male", "1990"]), sample["age0_male_1990"], rtol=1e-14, path="s1")
            assert_close(
                at(["50", "female", "2000"]), sample["age50_female_2000"], rtol=1e-14, path="s2"
            )
            assert_close(
                at(["100", "male", "1940"]), sample["age100_male_1940"], rtol=1e-14, path="s3"
            )
            assert_close(sum(table.rates), sample["sum"], rtol=1e-10, path="sum")
        else:
            summary = expected["summary"]
            usr, mn = r.survexp_usr(), r.survexp_mn()
            assert_exact(list(usr.dims), summary["usr_dim"], path="usr_dim")
            assert_exact(list(mn.dims), summary["mn_dim"], path="mn_dim")
            assert_close(sum(usr.rates), summary["usr_sum"], rtol=1e-10, path="usr_sum")
            assert_close(sum(mn.rates), summary["mn_sum"], rtol=1e-10, path="mn_sum")
            for d, dimid in enumerate(usr.dimid):
                assert_exact(list(usr.dimnames[d]), summary["usr_dimnames"][dimid], path=dimid)
            for d, dimid in enumerate(mn.dimid):
                assert_exact(list(mn.dimnames[d]), summary["mn_dimnames"][dimid], path=dimid)


# --- utilities ----------------------------------------------------------------


class UtilitiesHandler(TopicHandler):
    topic = "utilities"

    def aspects(self, case):
        return list(case["expected"])

    def check(self, case, aspect):
        name = case["name"]
        expected = case["expected"][aspect]
        args = case["args"]
        if name == "cipoisson":
            if aspect.startswith("scalar"):
                result = r.cipoisson(5) if aspect == "scalar_k5" else r.cipoisson(0, time=2)
                assert_close([*result.lower, *result.upper], expected, rtol=RTOL_VAR, path=aspect)
                return
            method, _, p = aspect.partition("_")
            result = r.cipoisson(
                args["k"], time=args["time"], p=0.90 if p == "p90" else 0.95, method=method
            )
            rows = [list(pair) for pair in zip(result.lower, result.upper, strict=True)]
            assert_matrix_close(rows, expected, rtol=RTOL_VAR, path=aspect)
        elif name == "bounded_links":
            x = args["x"]
            edge = 0.05
            if aspect.endswith("_linkinv"):
                raise UnsupportedCaseError("bounded link inverse functions are not exposed")
            link, _, suffix = aspect.partition("_")
            if suffix == "edge01":
                edge = 0.1
            elif suffix == "edge001":
                edge = 0.01
            fn = getattr(r, link)
            assert_close(as_float_list(fn(x, edge)), expected, rtol=RTOL_VAR, path=aspect)
        elif name == "nsk_basis":
            if aspect == "lung_age_df3":
                x = load_dataset("lung")["age"]
                basis = r.nsk(x, df=3)
            elif aspect == "df4":
                basis = r.nsk(args["x"], df=4)
            elif aspect == "knots_35_50_65":
                basis = r.nsk(args["x"], knots=[35, 50, 65], Boundary_knots=[20, 80])
            elif aspect == "df3_intercept":
                basis = r.nsk(args["x"], df=3, intercept=True)
            else:
                basis = r.nsk(args["x"], df=4, b=0.1)
            values = _basis_rows(basis)
            assert_matrix_close(values, expected["values"], rtol=RTOL_COEF, path=aspect)
            assert_close(
                as_float_list(_attr(basis, "knots")),
                expected["knots"],
                rtol=RTOL_COEF,
                path=f"{aspect}.knots",
            )
        elif name == "pspline_basis":
            x = load_dataset("lung")["age"]
            kwargs = {
                "df4": {"df": 4},
                "df4_nterm8": {"df": 4, "nterm": 8},
                "degree2_nterm6_df3": {"degree": 2, "nterm": 6, "df": 3},
                "df0": {"df": 0},
                "theta05": {"theta": 0.5},
            }[aspect]
            basis = r.pspline(x, **kwargs)
            values = _basis_rows(basis)
            assert_matrix_close(values, expected["values"], rtol=RTOL_COEF, path=aspect)
            assert_exact(_attr(basis, "nterm"), expected["nterm"], path=f"{aspect}.nterm")
        elif name == "aeqSurv":
            if aspect == "synthetic_timefix":
                data = case_data(self.topic, case)
                surv = r.Surv(data["time"], data["status"])
            elif aspect.startswith("right"):
                surv = r.Surv(args["time2"], args["status2"])
            else:
                surv = r.Surv(args["start3"], args["stop3"], args["status3"])
            tol = 1e-8 if aspect.endswith("tol_1e8") else None
            result = r.aeqSurv(surv, tolerance=tol) if tol else r.aeqSurv(surv)
            assert_matrix_close(_surv_matrix(result), expected["values"], rtol=1e-15, path=aspect)
        elif name == "surv_types":
            self._check_surv(aspect, expected)
        elif name == "statefig":
            connect = args["connect3"] if "1_2_1" not in aspect else args["connect4"]
            layout = [1, 2] if "1_2_1" not in aspect else [1, 2, 1]
            states = ["A", "B", "C"] if "1_2_1" not in aspect else ["A", "B", "C", "D"]
            if aspect.endswith("_column"):  # R: matrix(layout, ncol = 1)
                layout = [[count] for count in layout]
            result = r.statefig(layout, connect, states=states)
            coords = [[x, y] for x, y in zip(result.x, result.y, strict=True)]
            assert_matrix_close(coords, expected, rtol=RTOL_VAR, path=aspect)
        else:
            raise UnsupportedCaseError(f"unhandled utilities case {name}")

    @staticmethod
    def _check_surv(aspect: str, expected: Any) -> None:
        if aspect.startswith("format_"):
            kind = aspect.split("_", 1)[1]
            surv = {
                "right": lambda: r.Surv([1, 2, 3], [1, 0, 1]),
                "counting": lambda: r.Surv([0, 1], [3, 4], [1, 0]),
                "interval2": lambda: r.Surv([1, None, 3], [2, 3, None], type="interval2"),
                "mstate": lambda: r.Surv([1, 2], RFactor(["a", "censor"], ["censor", "a"])),
            }[kind]()
            assert_exact(list(r.format_surv(surv)), expected, path=aspect)
            return
        if aspect == "is_na":
            surv = r.Surv([1, None, 3], [1, 0, None])
            assert_exact([bool(v) for v in r.is_na_surv(surv)], expected, path=aspect)
            return
        builders = {
            "right": lambda: r.Surv([1, 2, 3, 4], [1, 0, 1, 1]),
            "right_12": lambda: r.Surv([1, 2, 3, 4], [2, 1, 2, 2]),
            "right_logical": lambda: r.Surv([1, 2, 3, 4], [True, False, True, True]),
            "left": lambda: r.Surv([1, 2, 3, 4], [1, 0, 1, 1], type="left"),
            "interval": lambda: r.Surv(
                [1, 2, 3, 4, 5], [2, None, 6, None, 7], [3, 0, 3, 1, 2], type="interval"
            ),
            "interval2": lambda: r.Surv([1, None, 3, 4, 5], [2, 3, None, 4, 8], type="interval2"),
            "counting": lambda: r.Surv([0, 1, 2, 0], [3, 4, 5, 2], [1, 0, 1, 1]),
            "counting_type": lambda: r.Surv(
                [0, 1, 2, 0], [3, 4, 5, 2], [1, 0, 1, 1], type="counting"
            ),
            "mstate": lambda: r.Surv(
                [1, 2, 3, 4],
                RFactor(["a", "censor", "b", "a"], ["censor", "a", "b"]),
                type="mstate",
            ),
            "mstate_factor": lambda: r.Surv(
                [1, 2, 3, 4], RFactor(["a", "censor", "b", "a"], ["censor", "a", "b"])
            ),
            "mcounting": lambda: r.Surv(
                [0, 1, 2, 0], [3, 4, 5, 2], RFactor(["a", "censor", "b", "a"], ["censor", "a", "b"])
            ),
            "origin": lambda: r.Surv([3, 4, 5], [1, 0, 1], origin=2),
        }
        surv = builders[aspect]()
        assert_exact(surv.type, expected["type"], path=f"{aspect}.type")
        assert_matrix_close(
            _surv_matrix(surv), expected["values"], rtol=1e-15, path=f"{aspect}.values"
        )
        if "states" in expected:
            assert_exact(list(_attr(surv, "states")), expected["states"], path=f"{aspect}.states")


def _basis_rows(basis: Any) -> list[list[float]]:
    """Rows of a spline basis returned either as nested lists or flat row-major."""

    values = _attr(basis, "basis", "values", "matrix")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        return [list(row) for row in values]
    n_cols = _attr(basis, "n_cols")
    return [list(values[i : i + n_cols]) for i in range(0, len(values), n_cols)]


def _surv_matrix(surv: Any) -> list[list[Any]]:
    """R's Surv matrix columns for a Python Surv object."""

    time = list(surv.time)
    event = list(surv.event)
    if surv.type in ("right", "left"):
        return [[t, e] for t, e in zip(time, event, strict=True)]
    if surv.type in ("counting", "mcounting"):
        return [[s, t, e] for s, t, e in zip(surv.start, time, event, strict=True)]
    if surv.type in ("mright",):
        return [[t, e] for t, e in zip(time, event, strict=True)]
    if surv.type in ("interval", "interval2"):
        time2 = list(surv.time2)
        return [[t, t2, e] for t, t2, e in zip(time, time2, event, strict=True)]
    raise UnsupportedCaseError(f"unhandled Surv type {surv.type}")


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------


# Cases are looked up by name inside the test: parametrising over the case
# dictionaries themselves makes pytest's failure reports (which repr every
# argument) prohibitively slow for the large fixture cases.
_CASES: dict[tuple[str, str], Mapping[str, Any]] = {}


def _collect_params() -> list[Any]:
    params = []
    for topic in topic_names():
        handler = HANDLERS.get(topic)
        for case in cases(topic):
            _CASES[(topic, case["name"])] = case
            aspects = handler.aspects(case) if handler is not None else ["(no handler)"]
            for aspect in aspects:
                test_id = case_id(topic, case, aspect)
                marks = []
                reason = _known_failure_reason(test_id)
                if reason is not None:
                    marks.append(pytest.mark.xfail(reason=reason, strict=True))
                params.append(pytest.param(topic, case["name"], aspect, id=test_id, marks=marks))
    return params


@pytest.mark.parametrize(("topic", "name", "aspect"), _collect_params())
def test_r_fixture(topic: str, name: str, aspect: str) -> None:
    __tracebackhide__ = True
    handler = HANDLERS.get(topic)
    case = _CASES[(topic, name)]
    test_id = case_id(topic, case, aspect)
    if handler is None:
        record_outcome(test_id, "missing feature", f"no handler for topic {topic}")
        raise UnsupportedCaseError(f"no handler for topic {topic}")
    # Every failure is re-raised from this frame without the original
    # traceback: pytest re-parses the whole source file of every frame it
    # prints (the 20k-line r_api.py included), which multiplies the runtime of
    # the ~2700 expected failures by ~20.  The message keeps the innermost
    # location; set R_FIXTURES_FULL_TRACEBACK=1 to debug with full tracebacks.
    full_traceback = bool(os.environ.get("R_FIXTURES_FULL_TRACEBACK"))
    try:
        handler.check(case, aspect)
        record_outcome(test_id, "pass", "")
    except UnsupportedCaseError as exc:
        record_outcome(test_id, "missing feature", str(exc))
        if full_traceback:
            raise
        raise UnsupportedCaseError(str(exc)) from None
    except FixtureMismatchError as exc:
        record_outcome(test_id, "mismatch", str(exc))
        if full_traceback:
            raise
        raise FixtureMismatchError(str(exc)) from None
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure kind feeds the burndown list
        message = f"{type(exc).__name__}: {exc} [{_innermost_location(exc)}]"
        record_outcome(test_id, "error", message)
        if full_traceback:
            raise
        raise PythonApiError(message) from None


class PythonApiError(Exception):
    """An exception raised by the Python API while running a fixture case."""


def _innermost_location(exc: BaseException) -> str:
    tb = exc.__traceback__
    location = "?"
    while tb is not None:
        code = tb.tb_frame.f_code
        location = f"{code.co_filename.rsplit('/', 1)[-1]}:{tb.tb_lineno}"
        tb = tb.tb_next
    return location


# ---------------------------------------------------------------------------
# Harness self-tests
# ---------------------------------------------------------------------------


def test_assert_close_floor_is_scaled_to_the_vector() -> None:
    # Noise around zero inside a vector whose other entries are O(1) is
    # within rtol * max|expected| and is not a mismatch ...
    assert_close([1.0, 0.0, -2.0], [1.0, -1.2e-17, -2.0], rtol=1e-8)
    assert_close([[0.0, 1.0], [1e-15, 0.0]], [[0.0, 1.0], [0.0, 0.0]], rtol=1e-8)
    # ... while a real difference of the same absolute size in a vector of
    # tiny values, or against a scalar zero, still is.
    with pytest.raises(FixtureMismatchError):
        assert_close([1e-17, 0.0], [1e-17, 1e-17], rtol=1e-8)
    with pytest.raises(FixtureMismatchError):
        assert_close(1e-17, 0.0, rtol=1e-8)
    with pytest.raises(FixtureMismatchError):
        assert_close([1.0, 1e-7], [1.0, 0.0], rtol=1e-8)
    # Exact comparisons keep a zero floor.
    with pytest.raises(FixtureMismatchError):
        assert_exact([3, 1e-17], [3, 0])


def test_array_scale_skips_non_numbers_and_mappings() -> None:
    assert array_scale([None, "NaN", "Inf", -3.0, [2.0, True]]) == 3.0
    assert array_scale([{"a": 100.0}, 0.5]) == 0.5
    assert array_scale([]) == 0.0
