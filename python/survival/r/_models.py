"""Model generics: predict, residuals, coef, vcov, confint, summaries, as_data_frame."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from operator import index
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _clamp_probability,
    _collapse_is_false,
    _collapse_prediction_result,
    _collapse_prediction_se,
    _collapse_residual_result,
    _float_vector,
    _integer_scalar,
    _materialize_1d,
    _materialize_labels,
    _model_residual_weights,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
    _normalize_optional_bool_option,
    _normalize_predict_type,
    _normalize_residual_type,
    _normalize_survreg_residual_type,
    _pop_dotted_keyword,
    _safe_exp,
    _strata_value_label,
    _weight_residual_result,
)
from ._coxph import (
    _cox_deviance_from_martingale,
    _cox_event_indices,
    _cox_expected_events_for_newdata,
    _cox_expected_events_with_se,
    _cox_linear_prediction_se,
    _cox_partial_residuals,
    _cox_predict_term_groups,
    _cox_predict_terms,
    _cox_survival_curve,
    _cox_survival_curve_with_se,
    _cox_term_prediction_se,
    _predict_terms_selection,
    _step_curve_at,
)
from ._fit import (
    _cox_alias_mask,
    _cox_degrees_of_freedom,
    _cox_event_count,
    _cox_full_loglik,
    _cox_loglik_values,
    _cox_reference_centers,
    _cox_reference_means,
    _cox_variance_matrix,
    _fallback_coef_names,
    _fit_location_coef_names,
    _formula_design_for_fit,
    _formula_design_output_names,
    _is_clogit_fit,
    _is_coxph_fit,
    _is_model_fit,
    _is_survreg_fit,
    _linear_predictors_for_fit,
    _location_beta,
    _newdata_has_formula_response,
    _normalize_predict_reference,
    _prediction_inputs,
    _require_model_fit,
    _survreg_scale_coef_names,
    _survreg_scales,
    _survreg_variance_matrix,
    _unwrap_formula_fit,
)
from ._formula import _design_term_output_names
from ._pyears import _finegray_frame, _pyears_result_frame
from ._surv import Surv
from ._survfit import _optional_float_list
from ._survreg import (
    _drop_single_quantile,
    _survreg_dfbeta_residuals,
    _survreg_distribution_family,
    _survreg_influence_residuals,
    _survreg_original_scale_loglik,
    _survreg_predict_terms,
    _survreg_prediction_se,
    _survreg_quantile_prediction_matrix,
    _survreg_quantile_prediction_se_matrix,
    _survreg_quantile_probabilities,
    _survreg_quantile_scores,
    _survreg_t_fit_degrees_of_freedom,
    _survreg_term_prediction_se,
)
from ._types import (
    ConcordanceResult,
    CoxBaseHazardResult,
    CoxPHDetailResult,
    CoxSurvfitResult,
    CoxZPHResult,
    FineGrayFrame,
    FineGrayOutput,
    PredictResult,
    PyearsResult,
    SurvfitMultiStateResult,
    SurvfitResult,
    TurnbullSurvfitResult,
    _cox_beta,
    _cox_scaled_schoenfeld_from_raw,
    _FormulaFit,
)


def coef(fit: Any) -> list[float]:
    """Return fitted model coefficients, like R's coef generic."""

    _require_model_fit(fit, "coef")
    if _is_survreg_fit(fit):
        return _location_beta(fit)
    beta = _cox_beta(fit)
    return [
        math.nan if aliased else value
        for value, aliased in zip(beta, _cox_alias_mask(fit), strict=True)
    ]


def coef_names(fit: Any, *, complete: Any | None = None) -> list[str]:
    """Return fitted coefficient names for R-style model helpers."""

    _require_model_fit(fit, "coef_names")
    include_complete = (
        _is_coxph_fit(fit) if complete is None else _normalize_bool_option(complete, "complete")
    )
    if _is_survreg_fit(fit):
        location_width = len(_location_beta(fit))
        names = _fit_location_coef_names(fit, location_width)
        if include_complete:
            total_width = len(list(fit.coefficients))
            names.extend(_survreg_scale_coef_names(fit, total_width - location_width))
        return names

    beta = _cox_beta(fit)
    names = _fit_location_coef_names(fit, len(beta))
    if include_complete:
        return names
    return [name for name, aliased in zip(names, _cox_alias_mask(fit), strict=True) if not aliased]


def vcov(fit: Any, *, complete: Any = True) -> list[list[float]]:
    """Return a fitted model variance-covariance matrix, like R's vcov generic."""

    _require_model_fit(fit, "vcov")
    include_complete = _normalize_bool_option_with_default(complete, "complete", True)
    if _is_survreg_fit(fit):
        width = len(list(fit.coefficients)) if include_complete else len(_location_beta(fit))
        return _survreg_variance_matrix(fit, width)
    variance = _cox_variance_matrix(fit, len(_cox_beta(fit)))
    if include_complete:
        return variance
    active = [idx for idx, aliased in enumerate(_cox_alias_mask(fit)) if not aliased]
    return [[variance[row_idx][col_idx] for col_idx in active] for row_idx in active]


def loglik(fit: Any) -> float:
    """Return a fitted model log likelihood."""

    _require_model_fit(fit, "loglik")
    if _is_survreg_fit(fit):
        return _survreg_original_scale_loglik(fit)
    return _cox_full_loglik(_unwrap_formula_fit(fit))


def _model_row_count(fit: Any) -> int:
    if isinstance(fit, _FormulaFit) and fit.n_observations is not None:
        return fit.n_observations
    values = getattr(fit, "status", None)
    if values is None:
        values = getattr(fit, "event_times", None)
    if values is None:
        raise TypeError("model does not expose stored observations")
    return len(list(values))


def nobs(fit: Any) -> int:
    """Return the model-specific observation count used by likelihood metadata."""

    _require_model_fit(fit, "nobs")
    if _is_coxph_fit(fit):
        return _cox_event_count(fit)
    return _model_row_count(fit)


def degrees_freedom(fit: Any) -> int:
    """Return the number of fitted parameters counted by model log likelihoods."""

    _require_model_fit(fit, "degrees_freedom")
    if _is_survreg_fit(fit):
        return len(list(fit.coefficients))
    return _cox_degrees_of_freedom(_unwrap_formula_fit(fit))


def df_residual(fit: Any) -> int:
    """Return residual degrees of freedom for fitted ``survreg`` models."""

    _require_model_fit(fit, "df_residual")
    if not _is_survreg_fit(fit):
        raise TypeError("df_residual is only defined for fitted survreg models")
    return nobs(fit) - degrees_freedom(fit)


def _finite_numeric_option(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def aic(fit: Any, *, k: Any = 2.0) -> float:
    """Return Akaike-style information criterion for a fitted model."""

    penalty = _finite_numeric_option(k, "k")
    return -2.0 * loglik(fit) + penalty * degrees_freedom(fit)


def bic(fit: Any) -> float:
    """Return Bayesian information criterion for a fitted model."""

    observation_count = nobs(fit)
    if observation_count == 0:
        return math.nan
    return aic(fit, k=math.log(observation_count))


def extract_aic(fit: Any, *, scale: Any = 0.0, k: Any = 2.0) -> list[float]:
    """Return ``[df, AIC]`` like R's ``extractAIC`` generic."""

    _finite_numeric_option(scale, "scale")
    return [float(degrees_freedom(fit)), aic(fit, k=k)]


def model_formula(fit: Any) -> str:
    """Return the formula string used to create a formula-based model fit."""

    _require_model_fit(fit, "model_formula")
    if isinstance(fit, _FormulaFit) and fit.formula is not None:
        return fit.formula
    raise TypeError("model_formula requires a formula-based fitted model")


def model_term_names(fit: Any, terms: Any | None = None) -> list[str]:
    """Return one label for each fitted model term, excluding the intercept."""

    _require_model_fit(fit, "model_term_names")
    groups = _cox_predict_term_groups(fit, len(_location_beta(fit)))
    names = [name for name, _columns in groups]
    return [names[idx] for idx in _predict_terms_selection(terms, names)]


def predict_terms_constant(fit: Any) -> float:
    """Return the sample-reference constant used by Cox term predictions."""

    if not _is_coxph_fit(fit):
        raise TypeError("predict_terms_constant requires a fitted coxph model")
    beta = _cox_beta(fit)
    means = _cox_reference_means(fit, "sample")
    return sum(value * coefficient for value, coefficient in zip(means, beta, strict=True))


def model_weights(fit: Any) -> list[float] | None:
    """Return explicit case weights for a fitted model, or ``None`` when absent."""

    _require_model_fit(fit, "model_weights")
    if isinstance(fit, _FormulaFit) and fit.case_weights is not None:
        return list(fit.case_weights)
    values = getattr(_unwrap_formula_fit(fit), "weights", None)
    if values is None:
        return None
    weights = [float(value) for value in _materialize_1d(values, "weights")]
    if all(abs(value - 1.0) <= 1e-12 for value in weights):
        return None
    return weights


def _model_matrix_column_names(fit: Any, width: int) -> list[str]:
    design = _formula_design_for_fit(fit)
    if design is not None:
        names = _formula_design_output_names(design)
        if len(names) == width:
            return names
    if _is_model_fit(fit):
        names = coef_names(fit)
        if len(names) == width:
            return names
    return _fallback_coef_names(width)


def _model_matrix_assignments(fit: Any, width: int) -> list[int]:
    design = _formula_design_for_fit(fit)
    if design is not None and len(design.term_assignments) == len(design.covariates):
        assignments = [0] if design.intercept else []
        for term, assignment in zip(
            design.covariates,
            design.term_assignments,
            strict=True,
        ):
            assignments.extend([assignment] * len(_design_term_output_names(term)))
        if len(assignments) == width:
            return assignments
    return list(range(1, width + 1))


def model_matrix(fit: Any) -> dict[str, Any]:
    """Return the training design matrix and column names for a fitted model."""

    _require_model_fit(fit, "model_matrix")
    rows = getattr(fit, "covariates", None)
    if rows is None:
        rows = getattr(fit, "x", None)
    if rows is None:
        raise TypeError("model_matrix requires a fitted model with stored covariates")
    matrix = [[float(value) for value in row] for row in rows]
    width = len(matrix[0]) if matrix else 0
    if any(len(row) != width for row in matrix):
        raise ValueError("stored model matrix must be rectangular")
    return {
        "data": matrix,
        "columns": _model_matrix_column_names(fit, width),
        "assign": _model_matrix_assignments(fit, width),
    }


def _model_frame_surv_columns(response: Surv, existing: set[str]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    if response.start is not None:
        if "start" not in existing:
            columns["start"] = list(response.start)
        if "stop" not in existing:
            columns["stop"] = list(response.time)
    elif "time" not in existing:
        columns["time"] = list(response.time)
    if response.time2 is not None and "time2" not in existing:
        columns["time2"] = list(response.time2)
    if "status" not in existing:
        columns["status"] = list(response.event)
    return columns


def model_frame(fit: Any) -> dict[str, list[Any]]:
    """Return a plain stored model frame for compatible fitted objects."""

    if isinstance(fit, Mapping):
        if not fit:
            raise TypeError("model_frame requires a non-empty grouped survfit result")
        return model_frame(next(iter(fit.values())))

    frame = getattr(fit, "model", None)
    if frame is None:
        raise TypeError("model_frame requires a stored model frame")
    if not isinstance(frame, Mapping):
        raise TypeError("stored model frame must be mapping-like")

    columns: dict[str, list[Any]] = {}
    for name, values in frame.items():
        if isinstance(values, Surv):
            columns.update(_model_frame_surv_columns(values, set(columns)))
            continue
        if isinstance(values, Mapping):
            continue
        text_name = str(name)
        if text_name in {"group", "(id)", "(cluster)", "(strata)"}:
            columns[text_name] = _materialize_labels(values, text_name)
            continue
        materialized = _materialize_1d(values, text_name)
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns[text_name] = list(materialized)
    return columns


def fitted(
    fit: Any,
    *,
    type: str | None = None,
    centered: bool | None = None,
    terms: Any | None = None,
    collapse: Any = False,
    reference: str | None = None,
    se_fit: bool = False,
    times: Any | None = None,
    p: Any | None = None,
    quantiles: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return fitted values for the training observations of a model."""

    _require_model_fit(fit, "fitted")
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"fitted got unexpected keyword argument(s): {unexpected}")

    return predict(
        fit,
        type=type,
        centered=centered,
        terms=terms,
        collapse=collapse,
        reference=reference,
        se_fit=se_fit,
        times=times,
        p=p,
        quantiles=quantiles,
    )


def _normal_two_sided_p_value(statistic: float) -> float:
    if math.isnan(statistic):
        return math.nan
    if math.isinf(statistic):
        return 0.0
    return 2.0 * NormalDist().cdf(-abs(statistic))


def _coefficient_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _coefficient_summary_rows(
    names: list[str],
    coefficients: list[float],
    variance: list[list[float]],
    *,
    naive_variance: list[list[float]] | None = None,
    robust: bool = False,
    coxph: bool = False,
    survreg: bool = False,
) -> list[dict[str, float | str]]:
    if len(names) != len(coefficients):
        raise ValueError("coefficient names do not match coefficient width")
    if len(variance) != len(coefficients) or any(len(row) != len(coefficients) for row in variance):
        raise ValueError("variance matrix does not match coefficient width")
    if naive_variance is None:
        naive_variance = variance
    if len(naive_variance) != len(coefficients) or any(
        len(row) != len(coefficients) for row in naive_variance
    ):
        raise ValueError("naive variance matrix does not match coefficient width")

    rows = []
    for idx, value in enumerate(coefficients):
        standard_error = math.sqrt(max(float(variance[idx][idx]), 0.0))
        naive_standard_error = math.sqrt(max(float(naive_variance[idx][idx]), 0.0))
        if math.isnan(value):
            statistic = math.nan
        elif standard_error > 0.0:
            statistic = value / standard_error
        elif value == 0.0:
            statistic = math.nan
        else:
            statistic = math.copysign(math.inf, value)
        row: dict[str, float | str] = {
            "name": names[idx],
            "coef": value,
            "se": standard_error,
            "naive_se": naive_standard_error,
            "statistic": statistic,
            "z": statistic,
            "p": _normal_two_sided_p_value(statistic),
        }
        if coxph:
            row["exp_coef"] = _coefficient_exp(value)
        if survreg:
            row["value"] = value
        if robust:
            row["robust_se"] = standard_error
        rows.append(row)
    return rows


def _summary_naive_variance(
    fit: Any,
    width: int,
    active_variance: list[list[float]],
) -> list[list[float]]:
    raw_variance = fit.naive_variance if isinstance(fit, _FormulaFit) else None
    if raw_variance is None:
        return active_variance
    variance = [[float(value) for value in row[:width]] for row in raw_variance[:width]]
    if len(variance) != width or any(len(row) != width for row in variance):
        raise ValueError("stored naive variance matrix does not match coefficient width")
    return variance


def _survreg_summary_coef_names(fit: Any, location_width: int, total_width: int) -> list[str]:
    names = _fit_location_coef_names(fit, location_width)
    scale_width = total_width - location_width
    if scale_width <= 0:
        return names
    if scale_width == 1:
        names.append("Log(scale)")
        return names

    design = _formula_design_for_fit(fit)
    if design is not None and len(design.strata_levels) == scale_width:
        level_parts = [
            (level,) if len(design.strata) == 1 else tuple(level) for level in design.strata_levels
        ]
        short_labels = all(
            len(parts) == len(design.strata) and all(isinstance(value, str) for value in parts)
            for parts in level_parts
        )
        for parts in level_parts:
            if len(parts) != len(design.strata):
                names.append(str(parts))
                continue
            labels = [_strata_value_label(value) for value in parts]
            if not short_labels:
                labels = [
                    f"{term}={label}" for term, label in zip(design.strata, labels, strict=True)
                ]
            names.append(", ".join(labels))
    else:
        names.extend(_survreg_scale_coef_names(fit, scale_width))
    return names


def _coefficient_selection_indices(parm: Any, names: list[str]) -> list[int]:
    if parm is None:
        return list(range(len(names)))

    if isinstance(parm, str):
        values: list[Any] = [parm]
    elif isinstance(parm, bool):
        raise TypeError("parm must be coefficient names or 1-based indices")
    else:
        if isinstance(parm, Sequence) and not isinstance(parm, bytes):
            values = list(_materialize_1d(parm, "parm"))
        else:
            values = [parm]

    indices: list[int] = []
    for value in values:
        if isinstance(value, str):
            try:
                idx = names.index(value)
            except ValueError as exc:
                raise ValueError(f"unknown coefficient name {value!r}") from exc
        else:
            if isinstance(value, bool):
                raise TypeError("parm must be coefficient names or 1-based indices")
            try:
                raw_idx = index(value)
            except TypeError:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError("parm must be coefficient names or 1-based indices") from exc
                if not numeric.is_integer():
                    raise TypeError("parm must be coefficient names or 1-based indices") from None
                raw_idx = int(numeric)
            except ValueError as exc:
                raise TypeError("parm must be coefficient names or 1-based indices") from exc
            idx = raw_idx - 1
            if idx < 0 or idx >= len(names):
                raise IndexError("parm index out of range")
        indices.append(idx)
    return indices


def confint(
    fit: Any,
    parm: Any | None = None,
    *,
    level: Any = 0.95,
) -> list[dict[str, float | str]]:
    """Return normal-approximation confidence intervals for model coefficients."""

    _require_model_fit(fit, "confint")
    confidence_level = _normalize_conf_level(level, "level")
    alpha = 1.0 - confidence_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    names = coef_names(fit)
    coefficients = coef(fit)
    variance = vcov(fit, complete=not _is_survreg_fit(fit))
    indices = _coefficient_selection_indices(parm, names)

    intervals = []
    for idx in indices:
        standard_error = math.sqrt(max(float(variance[idx][idx]), 0.0))
        margin = z * standard_error
        intervals.append(
            {
                "name": names[idx],
                "lower": coefficients[idx] - margin,
                "upper": coefficients[idx] + margin,
            }
        )
    return intervals


def model_summary(fit: Any) -> dict[str, Any]:
    """Return a compact R-style model summary as plain Python data."""

    _require_model_fit(fit, "model_summary")
    is_survreg = _is_survreg_fit(fit)
    robust = bool(getattr(fit, "robust", False))
    if is_survreg:
        location_width = len(_location_beta(fit))
        coefficients = [float(value) for value in fit.coefficients]
        names = _survreg_summary_coef_names(fit, location_width, len(coefficients))
        variance = vcov(fit, complete=True)
    else:
        names = coef_names(fit)
        coefficients = coef(fit)
        variance = vcov(fit, complete=True)
    naive_variance = _summary_naive_variance(fit, len(coefficients), variance)
    result: dict[str, Any] = {
        "model_type": "survreg" if is_survreg else "coxph",
        "coefficients": _coefficient_summary_rows(
            names,
            coefficients,
            variance,
            naive_variance=naive_variance,
            robust=robust,
            coxph=not is_survreg,
            survreg=is_survreg,
        ),
        "coefficient_names": names,
        "loglik": loglik(fit),
        "df": degrees_freedom(fit),
        "n": _model_row_count(fit),
        "robust": robust,
    }
    if is_survreg:
        result["location_coefficients"] = coef(fit)
        result["location_coefficient_names"] = coef_names(fit)
        result["scale"] = float(fit.scale)
        result["scales"] = _survreg_scales(fit)
        distribution = getattr(fit, "distribution", None)
        if distribution is not None:
            result["distribution"] = str(distribution)
        distribution_parameters = getattr(fit, "distribution_parameters", None)
        if distribution_parameters is not None:
            parameter_values = [float(value) for value in distribution_parameters]
            if parameter_values:
                result["distribution_parameters"] = parameter_values
    else:
        model = _unwrap_formula_fit(fit)
        logliks = _cox_loglik_values(model)
        result["null_loglik"] = logliks[0]
        result["score_test"] = float(model.score_test)
        result["n_event"] = sum(1 for event in model.status if int(event) == 1)
        result["method"] = str(getattr(model, "method", "breslow"))
    return result


def _empty_columns(names: tuple[str, ...]) -> dict[str, list[Any]]:
    return {name: [] for name in names}


def _add_optional_survfit_column(
    frame: dict[str, list[Any]],
    name: str,
    values: Sequence[Any],
    row_count: int,
) -> None:
    column = list(values)
    if not column:
        return
    if len(column) != row_count:
        raise ValueError(f"survfit column {name!r} must match time length")
    frame[name] = column


def _survfit_frame(result: SurvfitResult) -> dict[str, list[Any]]:
    std_err = list(result.std_err)
    conf_lower = list(result.conf_lower)
    conf_upper = list(result.conf_upper)
    for idx, survival in enumerate(result.estimate):
        if survival <= 0.0:
            if idx < len(std_err):
                std_err[idx] = math.nan
            if idx < len(conf_lower):
                conf_lower[idx] = math.nan
            if idx < len(conf_upper):
                conf_upper[idx] = math.nan
    row_count = len(result.time)
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "surv": result.estimate,
        "cumhaz": result.cumhaz,
    }
    _add_optional_survfit_column(frame, "std.err", std_err, row_count)
    _add_optional_survfit_column(frame, "lower", conf_lower, row_count)
    _add_optional_survfit_column(frame, "upper", conf_upper, row_count)
    _add_optional_survfit_column(frame, "std.chaz", result.std_chaz, row_count)
    if result.n_enter is not None:
        frame["n.enter"] = result.n_enter
    return frame


def _survfit_multistate_column(
    values: Sequence[Sequence[Any]],
    state_index: int,
    row_count: int,
    state_count: int,
    name: str,
) -> list[Any]:
    if len(values) != row_count:
        raise ValueError(f"multi-state survfit column {name!r} must match time length")
    column: list[Any] = []
    for row in values:
        if len(row) != state_count:
            raise ValueError(f"multi-state survfit column {name!r} must have one value per state")
        column.append(row[state_index])
    return column


def _survfit_multistate_frame(result: SurvfitMultiStateResult) -> dict[str, list[Any]]:
    row_count = len(result.time)
    state_count = len(result.states)
    required = {
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "pstate": result.pstate,
    }
    optional = {
        "std.err": result.std_err,
        "lower": result.conf_lower,
        "upper": result.conf_upper,
    }
    frame: dict[str, list[Any]] = {
        "time": [],
        **{name: [] for name in required},
        **{name: [] for name, values in optional.items() if values is not None},
        "state": [],
    }
    for state_index, state in enumerate(result.states):
        frame["time"].extend(float(value) for value in result.time)
        for name, values in required.items():
            frame[name].extend(
                _survfit_multistate_column(
                    values,
                    state_index,
                    row_count,
                    state_count,
                    name,
                )
            )
        for name, values in optional.items():
            if values is not None:
                frame[name].extend(
                    _survfit_multistate_column(
                        values,
                        state_index,
                        row_count,
                        state_count,
                        name,
                    )
                )
        frame["state"].extend([state] * row_count)
    return frame


def _subset_survfit_multistate(
    result: SurvfitMultiStateResult,
    state_indices: Any,
) -> SurvfitMultiStateResult:
    if not isinstance(result, SurvfitMultiStateResult):
        raise TypeError("multi-state survfit subsetting requires a multi-state result")
    indices = [
        _integer_scalar(value, "state_indices")
        for value in _materialize_1d(state_indices, "state_indices")
    ]
    if not indices:
        raise ValueError("multi-state survfit subsetting must select at least one state")
    if any(index < 0 or index >= len(result.states) for index in indices):
        raise IndexError("multi-state survfit state index is out of bounds")

    def select_columns(values: list[list[float]]) -> list[list[float]]:
        return [[float(row[index]) for index in indices] for row in values]

    def select_optional(
        values: list[list[float]] | None,
    ) -> list[list[float]] | None:
        return None if values is None else select_columns(values)

    empty_transitions = [[] for _ in result.time]
    return SurvfitMultiStateResult(
        time=[float(value) for value in result.time],
        n_risk=select_columns(result.n_risk),
        n_event=select_columns(result.n_event),
        n_censor=select_columns(result.n_censor),
        pstate=select_columns(result.pstate),
        cumhaz=empty_transitions,
        states=tuple(result.states[index] for index in indices),
        transitions=(),
        p0=[float(result.p0[index]) for index in indices],
        t0=result.t0,
        n=result.n,
        n_id=result.n_id,
        std_err=select_optional(result.std_err),
        std_err0=(
            None
            if result.std_err0 is None
            else [float(result.std_err0[index]) for index in indices]
        ),
        std_chaz=None if result.std_chaz is None else empty_transitions,
        std_auc=select_optional(result.std_auc),
        conf_lower=select_optional(result.conf_lower),
        conf_upper=select_optional(result.conf_upper),
        n_risk_count=select_optional(result.n_risk_count),
        n_event_count=select_optional(result.n_event_count),
        n_censor_count=select_optional(result.n_censor_count),
        n_enter=select_optional(result.n_enter),
        n_enter_count=select_optional(result.n_enter_count),
        n_transition=empty_transitions,
        n_transition_count=(None if result.n_transition_count is None else empty_transitions),
        model=result.model,
        surv_type=result.surv_type,
        conf_type=result.conf_type,
        conf_level=result.conf_level,
        oldstate=result.states if result.oldstate is None else result.oldstate,
        p0_fixed=result.p0_fixed,
        timefix=result.timefix,
    )


def _survfit_multistate_structure(
    result: SurvfitMultiStateResult | Mapping[Any, Any],
) -> dict[str, Any]:
    if isinstance(result, SurvfitMultiStateResult):
        curves = [(None, result)]
        grouped = False
    elif (
        isinstance(result, Mapping)
        and result
        and all(isinstance(curve, SurvfitMultiStateResult) for curve in result.values())
    ):
        curves = list(result.items())
        grouped = True
    else:
        raise TypeError("survfit structure requires a multi-state result")

    first = curves[0][1]
    for _label, curve in curves:
        if curve.states != first.states:
            raise ValueError("grouped multi-state results must share state columns")
        if curve.transitions != first.transitions:
            raise ValueError("grouped multi-state results must share transition columns")
        if curve.surv_type != first.surv_type:
            raise ValueError("grouped multi-state results must share a response type")

    def combined_matrix(name: str) -> list[list[float]] | None:
        matrices = [getattr(curve, name) for _label, curve in curves]
        if all(matrix is None for matrix in matrices):
            return None
        if any(matrix is None for matrix in matrices):
            raise ValueError(f"grouped multi-state results must share {name} output")
        return [[float(value) for value in row] for matrix in matrices for row in matrix]

    transition_names = [f"{source + 1}:{target + 1}" for source, target in first.transitions]
    structure: dict[str, Any] = {
        "n": [curve.n for _label, curve in curves] if grouped else first.n,
        "time": [float(value) for _label, curve in curves for value in curve.time],
        "n.risk": combined_matrix("n_risk"),
        "n.event": combined_matrix("n_event"),
        "n.censor": combined_matrix("n_censor"),
        "pstate": combined_matrix("pstate"),
    }
    if first.transitions:
        structure["n.transition"] = combined_matrix("n_transition")
    if grouped or first.oldstate is None:
        structure["n.id"] = [curve.n_id for _label, curve in curves] if grouped else first.n_id
    if first.transitions:
        structure["cumhaz"] = combined_matrix("cumhaz")
    n_enter = combined_matrix("n_enter")
    if n_enter is not None:
        structure["n.enter"] = n_enter
    structure["p0"] = (
        [[float(value) for value in curve.p0] for _label, curve in curves]
        if grouped
        else [float(value) for value in first.p0]
    )
    if grouped:
        structure["strata"] = {str(label): len(curve.time) for label, curve in curves}
    for field_name, attribute in (
        ("std.err", "std_err"),
        ("std.chaz", "std_chaz"),
        ("std.auc", "std_auc"),
    ):
        values = combined_matrix(attribute)
        if values is not None and (field_name != "std.chaz" or first.transitions):
            structure[field_name] = values
    structure["logse"] = False

    if first.transitions:
        target_states = list(dict.fromkeys(target for _source, target in first.transitions))
        target_columns = {state: index for index, state in enumerate(target_states)}
        transition_table = [[0.0] * (len(target_states) + 1) for _state in first.states]
        for _label, curve in curves:
            transition_values = (
                curve.n_transition_count
                if curve.n_transition_count is not None
                else curve.n_transition
            )
            for row in transition_values:
                for transition_index, (source, target) in enumerate(curve.transitions):
                    transition_table[source][target_columns[target]] += float(row[transition_index])
            censor_values = (
                curve.n_censor_count if curve.n_censor_count is not None else curve.n_censor
            )
            for row in censor_values:
                for state, value in enumerate(row):
                    transition_table[state][-1] += float(value)
        structure["transitions"] = {
            "values": transition_table,
            "rows": list(first.states),
            "columns": [first.states[state] for state in target_states] + ["(censored)"],
        }

    for field_name, attribute in (("lower", "conf_lower"), ("upper", "conf_upper")):
        values = combined_matrix(attribute)
        if values is not None:
            structure[field_name] = values
    structure.update(
        {
            "conf.type": first.conf_type,
            "conf.int": first.conf_level,
            "states": list(first.states),
            "type": first.surv_type,
            "t0": first.t0,
            "_transition_names": transition_names,
        }
    )
    if first.oldstate is not None:
        structure["oldstate"] = list(first.oldstate)
    return structure


def _turnbull_survfit_frame(result: TurnbullSurvfitResult) -> dict[str, list[Any]]:
    return {
        "time": result.time_points,
        "surv": result.survival,
        "lower": result.survival_lower,
        "upper": result.survival_upper,
    }


def _grouped_survfit_frame(result: Mapping[Any, Any]) -> dict[str, list[Any]]:
    if result and all(isinstance(curve, SurvfitMultiStateResult) for curve in result.values()):
        curve_frames = {label: _survfit_multistate_frame(curve) for label, curve in result.items()}
        columns = list(next(iter(curve_frames.values())))
        if "state" not in columns:
            raise ValueError("multi-state survfit frame is missing its state column")
        frame = {
            name: [] for name in [*[name for name in columns if name != "state"], "strata", "state"]
        }
        states = next(iter(result.values())).states
        if any(curve.states != states for curve in result.values()):
            raise ValueError("grouped multi-state results must share state columns")
        for state in states:
            for label, curve_frame in curve_frames.items():
                if list(curve_frame) != columns:
                    raise ValueError("grouped multi-state results must share tabular columns")
                indices = [
                    index for index, value in enumerate(curve_frame["state"]) if value == state
                ]
                frame["strata"].extend([str(label)] * len(indices))
                frame["state"].extend([state] * len(indices))
                for name in columns:
                    if name != "state":
                        frame[name].extend(curve_frame[name][index] for index in indices)
        return frame

    frame: dict[str, list[Any]] = {}
    for label, curve in result.items():
        curve_frame = as_data_frame(curve)
        if not curve_frame:
            continue
        n_rows = len(next(iter(curve_frame.values())))
        if not frame:
            frame = {"strata": []}
            for name in curve_frame:
                frame[name] = []
        elif set(curve_frame) != set(frame) - {"strata"}:
            raise ValueError("grouped survfit results must share tabular columns")
        frame["strata"].extend([str(label)] * n_rows)
        for name, values in curve_frame.items():
            frame[name].extend(values)
    return frame


def _raw_survfit_frame(result: Any) -> dict[str, list[Any]]:
    return _survfit_frame(
        SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[float(value) for value in result.std_err],
            conf_lower=[float(value) for value in result.conf_lower],
            conf_upper=[float(value) for value in result.conf_upper],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[float(value) for value in result.std_chaz],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
        )
    )


def _raw_turnbull_survfit_frame(result: Any) -> dict[str, list[Any]]:
    return _turnbull_survfit_frame(
        TurnbullSurvfitResult(
            time_points=[float(value) for value in result.time_points],
            survival=[float(value) for value in result.survival],
            survival_lower=[float(value) for value in result.survival_lower],
            survival_upper=[float(value) for value in result.survival_upper],
            n_iter=int(result.n_iter),
            converged=bool(result.converged),
        )
    )


def _cox_basehaz_frame(result: CoxBaseHazardResult) -> dict[str, list[Any]]:
    if result.cumhaz and isinstance(result.cumhaz[0], Sequence):
        frame: dict[str, list[Any]] = {
            "curve": [],
            "time": [],
            "cumhaz": [],
        }
        curve_strata = result.curve_strata_labels or result.curve_strata
        if curve_strata is not None:
            frame["strata"] = []
        for curve_idx, curve in enumerate(result.cumhaz):
            if isinstance(curve, (str, bytes)):
                raise TypeError("basehaz cumulative hazards must be numeric")
            if len(curve) != len(result.time):
                raise ValueError("basehaz curve length must match time length")
            frame["curve"].extend([curve_idx + 1] * len(result.time))
            frame["time"].extend(result.time)
            frame["cumhaz"].extend([float(value) for value in curve])
            if curve_strata is not None:
                frame["strata"].extend([curve_strata[curve_idx]] * len(result.time))
        return frame

    frame = {
        "time": result.time,
        "cumhaz": [float(value) for value in result.cumhaz],
    }
    strata = result.strata_labels or result.strata
    if strata is not None:
        frame["strata"] = strata
    return frame


def _cox_survfit_optional_curve_column(
    values: list[list[float]],
    curve_idx: int,
    time_count: int,
) -> list[float] | None:
    if len(values) <= curve_idx:
        return None
    column = values[curve_idx]
    if len(column) != time_count:
        raise ValueError("Cox survfit curve columns must match time length")
    return column


def _cox_survfit_frame(result: CoxSurvfitResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "curve": [],
        "time": [],
        "surv": [],
        "cumhaz": [],
        "linear.predictor": [],
    }
    strata = result.strata_labels or result.strata
    if strata is not None:
        frame["strata"] = []
    if result.start_time is not None:
        frame["start.time"] = []

    optional_columns = {
        "std.err": result.std_err,
        "std.chaz": result.std_chaz,
        "lower": result.conf_lower,
        "upper": result.conf_upper,
    }
    active_optional = {name: values for name, values in optional_columns.items() if values}
    for name in active_optional:
        frame[name] = []

    for curve_idx, (surv_curve, cumhaz_curve, linear_predictor) in enumerate(
        zip(result.surv, result.cumhaz, result.linear_predictors, strict=True)
    ):
        if len(surv_curve) != len(result.time) or len(cumhaz_curve) != len(result.time):
            raise ValueError("Cox survfit curves must match time length")
        n_times = len(result.time)
        frame["curve"].extend([curve_idx + 1] * n_times)
        frame["time"].extend(result.time)
        frame["surv"].extend(surv_curve)
        frame["cumhaz"].extend(cumhaz_curve)
        frame["linear.predictor"].extend([linear_predictor] * n_times)
        if strata is not None:
            frame["strata"].extend([strata[curve_idx]] * n_times)
        if result.start_time is not None:
            frame["start.time"].extend([result.start_time] * n_times)
        for name, values in active_optional.items():
            optional_curve = _cox_survfit_optional_curve_column(values, curve_idx, n_times)
            if optional_curve is not None:
                frame[name].extend(optional_curve)
    return frame


def _survdiff_frame(result: Any) -> dict[str, list[Any]]:
    observed = [float(value) for value in result.observed]
    expected = [float(value) for value in result.expected]
    if len(observed) != len(expected):
        raise ValueError("survdiff observed and expected lengths differ")
    variance = getattr(result, "variance", None)
    if isinstance(variance, int | float):
        variance_diag = [float(variance)] * len(observed)
    elif variance is not None:
        variance_diag = [float(row[idx]) for idx, row in enumerate(variance)]
    else:
        variance_diag = [math.nan] * len(observed)
    return {
        "group": [idx + 1 for idx in range(len(observed))],
        "observed": observed,
        "expected": expected,
        "variance": variance_diag,
    }


def _cox_zph_frame(result: CoxZPHResult) -> dict[str, list[Any]]:
    rows = result.table
    return {
        "name": [str(row["name"]) for row in rows],
        "chisq": [float(row["chisq"]) for row in rows],
        "df": [int(row["df"]) for row in rows],
        "p": [float(row["p"]) for row in rows],
    }


def _coxph_detail_frame(result: CoxPHDetailResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "n.event": result.nevent,
        "n.risk": result.nrisk,
        "hazard": result.hazard,
        "varhaz": result.varhaz,
        "cumhaz": result.cumulative_hazard,
        "wtrisk": result.wtrisk,
    }
    if result.nevent_wt is not None:
        frame["n.event.weight"] = result.nevent_wt
    if result.nrisk_wt is not None:
        frame["n.risk.weight"] = result.nrisk_wt
    if result.strata is not None:
        frame["strata"] = []
        for stratum, count in result.strata.items():
            frame["strata"].extend([stratum] * int(count))
    return frame


def _anova_frame(result: Any) -> dict[str, list[Any]]:
    rows = list(result.rows)
    return {
        "model": [str(row.model_name) for row in rows],
        "loglik": [float(row.loglik) for row in rows],
        "df": [int(row.df) for row in rows],
        "chisq": [math.nan if row.chisq is None else float(row.chisq) for row in rows],
        "p": [math.nan if row.p_value is None else float(row.p_value) for row in rows],
    }


def _concordance_frame(result: ConcordanceResult) -> dict[str, list[Any]]:
    if isinstance(result.concordance, list):
        n_scores = len(result.concordance)
        score_names = result.score_names or [f"score{idx + 1}" for idx in range(n_scores)]
        variance = result.variance if isinstance(result.variance, list) else [math.nan] * n_scores
        tied_x = result.tied_x if isinstance(result.tied_x, list) else [result.tied_x] * n_scores
        tied_y = result.tied_y if isinstance(result.tied_y, list) else [result.tied_y] * n_scores
        tied_xy = (
            result.tied_xy if isinstance(result.tied_xy, list) else [result.tied_xy] * n_scores
        )
        return {
            "score": score_names,
            "concordance": [float(value) for value in result.concordance],
            "concordant": [float(value) for value in result.concordant],
            "comparable": [float(value) for value in result.comparable],
            "tied.x": [float(value) for value in tied_x],
            "tied.y": [float(value) for value in tied_y],
            "tied.xy": [float(value) for value in tied_xy],
            "n": [result.n] * n_scores,
            "n.event": [result.n_event] * n_scores,
            "variance": [math.nan if value is None else float(value) for value in variance],
        }

    variance_value = result.variance if isinstance(result.variance, int | float) else math.nan
    return {
        "score": [result.score_names[0] if result.score_names else "score"],
        "concordance": [float(result.concordance)],
        "concordant": [float(result.concordant)],
        "comparable": [float(result.comparable)],
        "tied.x": [float(result.tied_x)],
        "tied.y": [float(result.tied_y)],
        "tied.xy": [float(result.tied_xy)],
        "n": [result.n],
        "n.event": [result.n_event],
        "variance": [float(variance_value)],
    }


def _surv_response_frame(response: Surv) -> dict[str, list[Any]]:
    if response.start is not None:
        frame: dict[str, list[Any]] = {
            "start": list(response.start),
            "stop": list(response.time),
            "status": list(response.event),
        }
    else:
        frame = {
            "time": list(response.time),
            "status": list(response.event),
        }
        if response.time2 is not None:
            frame["time2"] = list(response.time2)
    frame["type"] = [response.type] * len(response)
    return frame


def as_data_frame(result: Any) -> dict[str, list[Any]]:
    """Return a plain column-oriented table for common R-style result objects."""

    if isinstance(result, Surv):
        return _surv_response_frame(result)
    if isinstance(result, CoxSurvfitResult):
        return _cox_survfit_frame(result)
    if isinstance(result, CoxBaseHazardResult):
        return _cox_basehaz_frame(result)
    if isinstance(result, SurvfitMultiStateResult):
        return _survfit_multistate_frame(result)
    if isinstance(result, SurvfitResult):
        return _survfit_frame(result)
    if isinstance(result, TurnbullSurvfitResult):
        return _turnbull_survfit_frame(result)
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return _raw_survfit_frame(result)
    if all(
        hasattr(result, name)
        for name in ("time_points", "survival", "survival_lower", "survival_upper")
    ):
        return _raw_turnbull_survfit_frame(result)
    if isinstance(result, CoxZPHResult):
        return _cox_zph_frame(result)
    if isinstance(result, CoxPHDetailResult):
        return _coxph_detail_frame(result)
    if isinstance(result, ConcordanceResult):
        return _concordance_frame(result)
    if isinstance(result, PyearsResult):
        return _pyears_result_frame(result)
    if isinstance(result, FineGrayFrame):
        return {name: list(values) for name, values in result.items()}
    if isinstance(result, FineGrayOutput):
        return _finegray_frame(result)
    if isinstance(result, Mapping):
        return _grouped_survfit_frame(result)
    if hasattr(result, "observed") and hasattr(result, "expected") and hasattr(result, "variance"):
        return _survdiff_frame(result)
    if hasattr(result, "rows") and hasattr(result, "test_type"):
        return _anova_frame(result)
    raise TypeError("as_data_frame requires a survival result object")


def predict(
    fit: Any,
    newdata: Any | None = None,
    *,
    type: str | None = None,
    centered: bool | None = None,
    terms: Any | None = None,
    collapse: Any = False,
    reference: str | None = None,
    se_fit: bool = False,
    times: Any | None = None,
    p: Any | None = None,
    quantiles: Any | None = None,
    **kwargs: Any,
) -> Any:
    """R-style prediction generic for fitted survival models."""

    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"predict got unexpected keyword argument(s): {unexpected}")

    is_survreg = _is_survreg_fit(fit)
    predict_type = _normalize_predict_type(
        type if type is not None else ("response" if is_survreg else "lp"),
        survreg=is_survreg,
    )
    centered_value = _normalize_optional_bool_option(centered, "centered")
    include_se = _normalize_bool_option(se_fit, "se_fit")
    rows, offsets = _prediction_inputs(fit, newdata)

    if is_survreg:
        if not _collapse_is_false(collapse):
            raise ValueError("collapse is only supported for Cox model predictions")
        if reference is not None:
            raise ValueError("reference is only supported for Cox model predictions")
        if centered_value:
            raise ValueError("centered predictions are only supported for Cox models")
        if predict_type in {"survival", "expected", "risk"}:
            raise ValueError(f"predict type={predict_type!r} is not supported for survreg fits")
        if predict_type == "terms":
            term_predictions = _survreg_predict_terms(fit, rows, terms)
            if include_se:
                return PredictResult(
                    term_predictions,
                    _survreg_term_prediction_se(fit, rows, terms),
                )
            return term_predictions
        if predict_type in {"quantile", "uquantile"}:
            if p is not None and quantiles is not None:
                raise ValueError("use only one of p or quantiles")
            q_values = quantiles if quantiles is not None else p
            q = _survreg_quantile_probabilities(q_values)
            scores = _survreg_quantile_scores(fit, q)
            predictions = _survreg_quantile_prediction_matrix(
                fit,
                rows,
                offsets,
                scores,
                predict_type,
                newdata,
            )
            if include_se:
                se = _survreg_quantile_prediction_se_matrix(
                    fit,
                    rows,
                    scores,
                    predictions,
                    predict_type,
                    newdata,
                )
                return PredictResult(
                    _drop_single_quantile(predictions, q),
                    _drop_single_quantile(se, q),
                )
            return _drop_single_quantile(predictions, q)
        result = fit.predict(rows, predict_type, offsets, False)
        if include_se:
            return PredictResult(
                result.predictions,
                _survreg_prediction_se(fit, rows, predict_type, result.predictions),
            )
        return result.predictions

    reference_name = _normalize_predict_reference(reference, centered_value, predict_type)

    if predict_type == "survival":
        if not hasattr(fit, "survival_curve"):
            raise TypeError("model does not support survival curve prediction")
        has_response = (
            rows is not None and times is None and _newdata_has_formula_response(fit, newdata)
        )
        if include_se:
            if times is not None or (rows is not None and not has_response):
                return _cox_survival_curve_with_se(
                    fit,
                    rows,
                    offsets,
                    True if centered_value is None else centered_value,
                    newdata,
                    times,
                    collapse,
                )
            expected = _cox_expected_events_with_se(fit, rows, offsets, newdata)
            probabilities = [_clamp_probability(_safe_exp(-value)) for value in expected.fit]
            probability_se = [
                float(se) * probability
                for se, probability in zip(expected.se_fit, probabilities, strict=True)
            ]
            return PredictResult(
                _collapse_prediction_result(probabilities, collapse),
                _collapse_prediction_se(probability_se, collapse),
            )
        if has_response:
            if rows is None:
                raise AssertionError("has_response implies prediction rows are available")
            probabilities = [
                _clamp_probability(_safe_exp(-expected))
                for expected in _cox_expected_events_for_newdata(fit, rows, offsets, newdata)
            ]
            return _collapse_prediction_result(probabilities, collapse)
        curve_times, curves = _cox_survival_curve(
            fit,
            rows,
            offsets,
            True if centered_value is None else centered_value,
            newdata,
        )
        if times is None:
            return curve_times, _collapse_prediction_result(curves, collapse)
        requested_times = _float_vector(times, "times")
        stepped_curves = [
            _step_curve_at(curve_times, [float(value) for value in curve], requested_times)
            for curve in curves
        ]
        return requested_times, _collapse_prediction_result(stepped_curves, collapse)

    if predict_type == "expected":
        if include_se:
            expected = _cox_expected_events_with_se(fit, rows, offsets, newdata)
            return PredictResult(
                _collapse_prediction_result(expected.fit, collapse),
                _collapse_prediction_se(expected.se_fit, collapse),
            )
        if rows is not None:
            expected_values = _cox_expected_events_for_newdata(fit, rows, offsets, newdata)
            return _collapse_prediction_result(expected_values, collapse)
        if not hasattr(fit, "expected_events"):
            raise TypeError("model does not support expected event prediction")
        return _collapse_prediction_result(fit.expected_events(), collapse)

    if predict_type == "terms":
        term_predictions = _cox_predict_terms(fit, rows, terms, reference_name, newdata)
        if include_se:
            term_se = _cox_term_prediction_se(fit, rows, terms, reference_name, newdata)
            return PredictResult(
                _collapse_prediction_result(term_predictions, collapse),
                _collapse_prediction_se(term_se, collapse),
            )
        return _collapse_prediction_result(term_predictions, collapse)

    linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    linear_se = (
        _cox_linear_prediction_se(fit, rows, reference_name, newdata)
        if include_se and predict_type in {"lp", "risk"}
        else None
    )
    if reference_name != "zero":
        centers = _cox_reference_centers(fit, reference_name, len(linear_predictors), newdata)
        linear_predictors = [
            value - center for value, center in zip(linear_predictors, centers, strict=True)
        ]
    if predict_type == "lp":
        if include_se:
            return PredictResult(
                _collapse_prediction_result(linear_predictors, collapse),
                _collapse_prediction_se(linear_se, collapse),
            )
        return _collapse_prediction_result(linear_predictors, collapse)
    if predict_type == "risk":
        risks = [_safe_exp(value) for value in linear_predictors]
        if include_se:
            if linear_se is None:
                raise AssertionError("se_fit risk predictions require linear SEs")
            risk_se = [float(se) * risk for se, risk in zip(linear_se, risks, strict=True)]
            return PredictResult(
                _collapse_prediction_result(risks, collapse),
                _collapse_prediction_se(risk_se, collapse),
            )
        return _collapse_prediction_result(risks, collapse)
    if predict_type == "response":
        raise ValueError("predict type='response' is only supported for survreg fits")
    if predict_type == "quantile":
        raise ValueError("predict type='quantile' is only supported for survreg fits")
    raise AssertionError(f"unhandled predict type {predict_type!r}")


def residuals(
    fit: Any,
    *,
    type: str = "martingale",
    terms: Any | None = None,
    collapse: Any = False,
    weighted: bool | None = None,
    rsigma: bool | None = None,
) -> Any:
    """R-style residual generic for fitted survival models."""

    weighted_value = _normalize_optional_bool_option(weighted, "weighted")
    rsigma_value = _normalize_optional_bool_option(rsigma, "rsigma")
    if _is_survreg_fit(fit):
        if terms is not None:
            raise ValueError("terms is only supported for Cox partial residuals")
        residual_type = _normalize_survreg_residual_type(type)
        if residual_type in {"dfbeta", "dfbetas"}:
            dfbeta_values = _survreg_dfbeta_residuals(fit, residual_type, rsigma=rsigma_value)
            if weighted_value:
                dfbeta_values = _weight_residual_result(
                    dfbeta_values,
                    _model_residual_weights(fit, len(dfbeta_values)),
                )
            return _collapse_residual_result(dfbeta_values, collapse, len(dfbeta_values))
        if residual_type == "matrix":
            matrix_values = _core.survreg_residual_matrix(
                fit.time,
                fit.status,
                fit.linear_predictors,
                fit.scale,
                fit.distribution,
                time2=getattr(fit, "time2", None),
                distribution_parameter=(
                    _survreg_t_fit_degrees_of_freedom(
                        getattr(fit, "distribution_parameters", None),
                    )
                    if _survreg_distribution_family(fit) == "t"
                    else None
                ),
            )
            if weighted_value:
                matrix_values = _weight_residual_result(
                    matrix_values,
                    _model_residual_weights(fit, len(matrix_values)),
                )
            return _collapse_residual_result(matrix_values, collapse, len(matrix_values))
        if residual_type in {"ldcase", "ldresp", "ldshape"}:
            influence_values = _survreg_influence_residuals(
                fit,
                residual_type,
                rsigma=rsigma_value,
            )
            if weighted_value:
                influence_values = _weight_residual_result(
                    influence_values,
                    _model_residual_weights(fit, len(influence_values)),
                )
            return _collapse_residual_result(influence_values, collapse, len(influence_values))
        if residual_type in {"response", "deviance", "working"}:
            scalar_values = fit.residuals(residual_type).residuals
            if weighted_value:
                scalar_values = _weight_residual_result(
                    scalar_values,
                    _model_residual_weights(fit, len(scalar_values)),
                )
            return _collapse_residual_result(scalar_values, collapse, len(scalar_values))
        raise AssertionError(f"unhandled survreg residual type {residual_type!r}")

    residual_type = _normalize_residual_type(type)
    if (
        _is_clogit_fit(fit)
        and getattr(_unwrap_formula_fit(fit), "method", None) == "exact"
        and residual_type in {"score", "schoenfeld", "dfbeta", "dfbetas", "scaledsch"}
    ):
        raise ValueError(f"{residual_type} residuals are not available for the exact method")
    if terms is not None and residual_type != "partial":
        raise ValueError("terms is only supported for Cox partial residuals")
    method_names = {
        "martingale": "martingale_residuals",
        "deviance": "deviance_residuals",
        "score": "score_residuals",
        "dfbeta": "dfbeta",
        "dfbetas": "dfbetas",
        "schoenfeld": "schoenfeld_residuals",
        "scaledsch": "scaled_schoenfeld_residuals",
        "partial": "partial_residuals",
    }
    use_weights = (
        residual_type in {"dfbeta", "dfbetas"} if weighted_value is None else weighted_value
    )

    if residual_type in {"schoenfeld", "scaledsch"}:
        method = getattr(fit, "schoenfeld_residuals", None)
        if method is None:
            raise TypeError(f"model does not support {residual_type} residuals")
        raw = method()
        if use_weights:
            weights = _model_residual_weights(fit, len(fit.status))
            event_weights = [weights[idx] for idx in _cox_event_indices(fit)]
            raw = _weight_residual_result(raw, event_weights)
        return raw if residual_type == "schoenfeld" else _cox_scaled_schoenfeld_from_raw(fit, raw)

    if residual_type == "deviance" and (use_weights or not _collapse_is_false(collapse)):
        martingale_method = getattr(fit, "martingale_residuals", None)
        if martingale_method is None:
            raise TypeError("model does not support deviance residuals")
        martingale = [float(value) for value in martingale_method()]
        status = [float(value) for value in fit.status]
        if use_weights:
            martingale = _weight_residual_result(
                martingale,
                _model_residual_weights(fit, len(martingale)),
            )
        if not _collapse_is_false(collapse):
            martingale = _collapse_residual_result(martingale, collapse, len(martingale))
            status = _collapse_residual_result(status, collapse, len(status))
        return _cox_deviance_from_martingale(martingale, status)

    if residual_type == "partial" and (
        _formula_design_for_fit(fit) is not None
        or terms is not None
        or use_weights
        or not _collapse_is_false(collapse)
    ):
        result = _cox_partial_residuals(
            fit,
            terms,
            _model_residual_weights(fit, len(fit.status)) if use_weights else None,
        )
        if not _collapse_is_false(collapse):
            result = _collapse_residual_result(result, collapse, len(result))
        return result

    method_name = method_names[residual_type]
    method = getattr(fit, method_name, None)
    if method is None:
        raise TypeError(f"model does not support {residual_type} residuals")
    result = method()
    if use_weights:
        result = _weight_residual_result(result, _model_residual_weights(fit, len(result)))
    if not _collapse_is_false(collapse):
        result = _collapse_residual_result(result, collapse, len(result))
    return result
