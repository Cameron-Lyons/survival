"""``survreg`` fitting plus its prediction/residual helpers and d/p/q/rsurvreg."""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Mapping
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _apply_survreg_control,
    _as_rows,
    _encode_groups,
    _encode_labels,
    _finite_float,
    _float_vector,
    _integer_code_vector,
    _integer_scalar,
    _is_bool_like,
    _keep_rows_after_na_action,
    _label_levels,
    _materialize_1d,
    _materialize_labels,
    _matrix_input_column_names,
    _missing_row_indices,
    _model_residual_weights,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_survreg_distribution,
    _optional_float_vector,
    _pop_dotted_keyword,
    _quantile_vector,
    _safe_exp,
    _subset_indices,
    _subset_optional_sequence,
    _validated_matrix_column_names,
)
from ._coxph import _cox_predict_term_groups, _predict_terms_selection
from ._fit import (
    _cox_training_rows,
    _formula_design_for_fit,
    _location_beta,
    _location_variance_matrix,
    _survreg_has_variance_width,
    _survreg_scales,
    _survreg_strata,
    _survreg_variance_matrix,
    _unwrap_formula_fit,
)
from ._formula import (
    _apply_formula_na_action,
    _combined_columns,
    _design_rows_from_spec,
    _fit_formula_design,
    _formula_model_frame,
    _formula_response_spec,
    _matrix_model_frame,
    _offset_vector,
    _parse_formula,
    _subset_formula_inputs,
    _survreg_matrix_model_frame,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv, _survreg_response_arrays
from ._types import _FormulaDesign, _FormulaFit


def _survreg_derivative_context(
    fit: Any,
    *,
    rsigma: bool | None,
) -> tuple[list[list[float]], list[list[float]], list[float], list[int], list[list[float]], bool]:
    nvar = len(_location_beta(fit))
    rows = _cox_training_rows(fit, nvar)
    if not rows and nvar:
        raise ValueError("stored training covariates are required for survreg residuals")
    matrix = _core.survreg_residual_matrix(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        time2=getattr(fit, "time2", None),
        distribution_parameter=(
            _survreg_t_fit_degrees_of_freedom(getattr(fit, "distribution_parameters", None))
            if _survreg_distribution_family(fit) == "t"
            else None
        ),
    )
    scales = _survreg_scales(fit)
    strata = _survreg_strata(fit, len(matrix), len(scales))
    rsigma_requested = True if rsigma is None else rsigma
    include_scale = rsigma_requested and _survreg_has_variance_width(
        fit,
        nvar + len(scales),
    )
    width = nvar + (len(scales) if include_scale else 0)
    variance = _survreg_variance_matrix(fit, width)
    return matrix, rows, scales, strata, variance, include_scale


def _survreg_dfbeta_residuals(
    fit: Any,
    residual_type: str,
    *,
    rsigma: bool | None,
) -> list[list[float]]:
    matrix, rows, scales, strata, variance, include_scale = _survreg_derivative_context(
        fit,
        rsigma=rsigma,
    )
    return _core.survreg_dfbeta_residuals(
        matrix,
        rows,
        scales,
        strata,
        variance,
        include_scale,
        residual_type == "dfbetas",
    )


def _survreg_influence_residuals(
    fit: Any,
    residual_type: str,
    *,
    rsigma: bool | None,
) -> list[float]:
    matrix, rows, scales, strata, variance, include_scale = _survreg_derivative_context(
        fit,
        rsigma=rsigma,
    )
    return _core.survreg_influence_residuals(
        matrix,
        rows,
        scales,
        strata,
        variance,
        residual_type,
        include_scale,
    )


def _survreg_prediction_rows(
    fit: Any,
    rows: list[list[float]] | None,
    purpose: str,
) -> list[list[float]]:
    nvar = len(_location_beta(fit))
    if rows is None:
        training_rows = _cox_training_rows(fit, nvar)
        if not training_rows and nvar:
            raise ValueError(f"stored training covariates are required for {purpose}")
        return training_rows

    prediction_rows = [[float(value) for value in row] for row in rows]
    if any(len(row) != nvar for row in prediction_rows):
        raise ValueError(f"newdata must have {nvar} columns")
    return prediction_rows


def _survreg_training_means(fit: Any, nvar: int) -> list[float]:
    rows = _cox_training_rows(fit, nvar)
    if not rows:
        return [0.0] * nvar
    return [sum(row[col_idx] for row in rows) / len(rows) for col_idx in range(nvar)]


def _survreg_term_design_rows(
    fit: Any,
    rows: list[list[float]] | None,
) -> list[list[float]]:
    prediction_rows = _survreg_prediction_rows(fit, rows, "predict type='terms'")
    design = _formula_design_for_fit(fit)
    if design is None or not design.intercept:
        return prediction_rows
    means = _survreg_training_means(fit, len(_location_beta(fit)))
    return [
        [float(value) - means[col_idx] for col_idx, value in enumerate(row)]
        for row in prediction_rows
    ]


def _survreg_robust_variance_matrix(
    fit: Any,
    cluster: Any,
) -> tuple[list[list[float]], list[list[float]], list[Any]]:
    cluster_values = _materialize_labels(cluster, "cluster")
    _label_levels(cluster_values, "cluster")
    dfbeta_rows = _survreg_dfbeta_residuals(fit, "dfbeta", rsigma=True)
    n = len(dfbeta_rows)
    if len(cluster_values) != n:
        raise ValueError("cluster must have the same length as the Surv response")

    width = len(dfbeta_rows[0]) if dfbeta_rows else len(list(getattr(fit, "variance_matrix", [])))
    naive = _survreg_variance_matrix(fit, width)
    weights = _model_residual_weights(fit, n)
    cluster_codes = _encode_labels(cluster_values, "cluster")
    robust = _core.clustered_crossprod(dfbeta_rows, weights, cluster_codes, width)
    return robust, naive, cluster_values


def _survreg_predict_terms(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_term_design_rows(fit, rows)
    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    return [
        [
            sum(float(row[col_idx]) * beta[col_idx] for col_idx in groups[group_idx][1])
            for group_idx in selected
        ]
        for row in prediction_rows
    ]


def _survreg_term_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_term_design_rows(fit, rows)
    variance = _location_variance_matrix(fit, len(beta))
    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    return _core.term_prediction_se_from_variance(
        prediction_rows,
        variance,
        [groups[group_idx][1] for group_idx in selected],
    )


def _survreg_linear_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
) -> list[float]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_prediction_rows(fit, rows, "prediction SEs")
    variance = _location_variance_matrix(fit, len(beta))
    return _core.prediction_se_from_variance(prediction_rows, variance)


def _survreg_response_uses_log_transform(fit: Any) -> bool:
    distribution = str(getattr(fit, "distribution", "")).lower().replace("-", "_")
    return distribution in {
        "weibull",
        "exponential",
        "rayleigh",
        "lognormal",
        "log_normal",
        "loggaussian",
        "loglogistic",
        "log_logistic",
    }


def _survreg_original_scale_loglik(fit: Any) -> float:
    model = _unwrap_formula_fit(fit)
    log_likelihood = float(model.log_likelihood)
    if not _survreg_response_uses_log_transform(model):
        return log_likelihood

    times = [float(value) for value in model.time]
    status = [int(value) for value in model.status]
    weights = [float(value) for value in getattr(model, "weights", [1.0] * len(times))]
    if len(status) != len(times) or len(weights) != len(times):
        raise ValueError("fitted survreg model has inconsistent likelihood arrays")
    jacobian = math.fsum(
        weight * math.log(time)
        for time, event, weight in zip(times, status, weights, strict=True)
        if event == 1
    )
    return log_likelihood - jacobian


def _survreg_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    predict_type: str,
    predictions: list[float],
) -> list[float]:
    se = _survreg_linear_prediction_se(fit, rows)
    if predict_type != "response" or not _survreg_response_uses_log_transform(fit):
        return se
    return [
        value * abs(float(prediction)) for value, prediction in zip(se, predictions, strict=True)
    ]


def _survreg_distribution_family(fit: Any) -> str:
    distribution = str(getattr(fit, "distribution", "")).lower().replace("-", "_")
    if distribution in {"logistic", "loglogistic", "log_logistic"}:
        return "logistic"
    if distribution in {"gaussian", "normal", "lognormal", "log_normal", "loggaussian"}:
        return "gaussian"
    if distribution in {"t", "student", "student_t", "studentt"}:
        return "t"
    return "extreme"


def _survreg_quantile_probabilities(values: Any | None) -> list[float]:
    probabilities = _quantile_vector(values, "p") if values is not None else [0.1, 0.9]
    if any(not math.isfinite(value) or value <= 0.0 or value >= 1.0 for value in probabilities):
        raise ValueError("p must be between 0 and 1")
    return probabilities


def _survreg_quantile_scores(fit: Any, probabilities: list[float]) -> list[float]:
    family = _survreg_distribution_family(fit)
    if family == "logistic":
        return [math.log(value / (1.0 - value)) for value in probabilities]
    if family == "gaussian":
        normal = NormalDist()
        return [normal.inv_cdf(value) for value in probabilities]
    if family == "t":
        df = _survreg_t_fit_degrees_of_freedom(
            getattr(fit, "distribution_parameters", None),
        )
        return _core.survreg_distribution(
            probabilities,
            [0.0] * len(probabilities),
            [1.0] * len(probabilities),
            "t",
            "quantile",
            df,
        )
    return [math.log(-math.log1p(-value)) for value in probabilities]


def _normalize_survreg_distribution_helper(distribution: Any | None) -> str:
    if distribution is None:
        return "weibull"
    if not isinstance(distribution, str):
        raise TypeError("distribution must be a string")
    value = distribution.strip().lower().replace("_", "-")
    if value in {"t", "student", "student-t"}:
        return "t"
    normalized = _normalize_survreg_distribution(distribution)
    return normalized or "weibull"


def _survreg_numeric_vector(values: Any, name: str) -> list[float]:
    return _quantile_vector(values, name)


def _survreg_t_degrees_of_freedom(parms: Any | None) -> float:
    if parms is None:
        raise TypeError("parms is required for distribution='t'")
    values = _survreg_numeric_vector(parms, "parms")
    if len(values) != 1:
        raise ValueError("parms for distribution='t' must be a single degrees-of-freedom value")
    df = values[0]
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("parms for distribution='t' must be a positive finite value")
    return df


def _survreg_t_fit_degrees_of_freedom(parms: Any | None) -> float:
    df = 4.0 if parms is None else _survreg_t_degrees_of_freedom(parms)
    if df <= 2.0:
        raise ValueError("Degrees of freedom must be >=3")
    return df


def _expand_survreg_distribution_inputs(
    values: Any,
    value_name: str,
    mean: Any,
    scale: Any,
    *,
    target_length: int | None = None,
) -> tuple[list[float], list[float], list[float]]:
    vectors = {
        value_name: _survreg_numeric_vector(values, value_name),
        "mean": _survreg_numeric_vector(mean, "mean"),
        "scale": _survreg_numeric_vector(scale, "scale"),
    }
    main_length = len(vectors[value_name])
    n = (
        target_length
        if target_length is not None
        else main_length
        if main_length > 1
        else max(len(vector) for vector in vectors.values())
    )
    if n < 0:
        raise ValueError("n must be non-negative")

    expanded: dict[str, list[float]] = {}
    for name, vector in vectors.items():
        if len(vector) == n:
            expanded[name] = vector
        elif len(vector) == 1:
            expanded[name] = vector * n
        else:
            raise ValueError(f"{name} must have length 1 or {n}")
    return expanded[value_name], expanded["mean"], expanded["scale"]


def _survreg_distribution_values(
    values: Any,
    value_name: str,
    mean: Any,
    scale: Any,
    distribution: Any | None,
    parms: Any | None,
    kind: str,
) -> list[float]:
    distribution_name = _normalize_survreg_distribution_helper(distribution)
    value_values, mean_values, scale_values = _expand_survreg_distribution_inputs(
        values,
        value_name,
        mean,
        scale,
    )
    distribution_parms = _survreg_t_degrees_of_freedom(parms) if distribution_name == "t" else None
    return _core.survreg_distribution(
        value_values,
        mean_values,
        scale_values,
        distribution_name,
        kind,
        distribution_parms,
    )


def dsurvreg(
    x: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Density for R ``survreg`` location-scale distributions."""

    return _survreg_distribution_values(x, "x", mean, scale, distribution, parms, "density")


def psurvreg(
    q: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Distribution function for R ``survreg`` location-scale distributions."""

    return _survreg_distribution_values(q, "q", mean, scale, distribution, parms, "distribution")


def qsurvreg(
    p: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Quantiles for R ``survreg`` location-scale distributions."""

    return _survreg_distribution_values(p, "p", mean, scale, distribution, parms, "quantile")


def rsurvreg(
    n: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Random draws from R ``survreg`` location-scale distributions."""

    count = _integer_scalar(n, "n")
    if count < 0:
        raise ValueError("n must be non-negative")
    distribution_name = _normalize_survreg_distribution_helper(distribution)
    if count == 0:
        return []
    probabilities = [random.random() for _ in range(count)]
    probability_values, mean_values, scale_values = _expand_survreg_distribution_inputs(
        probabilities,
        "p",
        mean,
        scale,
        target_length=count,
    )
    distribution_parms = _survreg_t_degrees_of_freedom(parms) if distribution_name == "t" else None
    return _core.survreg_distribution(
        probability_values,
        mean_values,
        scale_values,
        distribution_name,
        "quantile",
        distribution_parms,
    )


def _survreg_training_strata(fit: Any, n: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata = [int(value) for value in values]
    if len(strata) != n:
        raise ValueError("fitted survreg strata do not match prediction rows")
    scales = _survreg_scales(fit)
    if any(value < 0 or value >= len(scales) for value in strata):
        raise ValueError("fitted survreg strata do not match scale estimates")
    return strata


def _survreg_prediction_strata(
    fit: Any,
    newdata: Any | None,
    n: int,
) -> list[int]:
    scales = _survreg_scales(fit)
    if len(scales) <= 1:
        return [0] * n
    if newdata is None:
        return _survreg_training_strata(fit, n)

    design = _formula_design_for_fit(fit)
    if (
        design is not None
        and design.strata
        and (isinstance(newdata, Mapping) or hasattr(newdata, "columns"))
    ):
        labels = _combined_columns(newdata, list(design.strata), n)
        level_map = {value: idx for idx, value in enumerate(design.strata_levels)}
        strata: list[int] = []
        for value in labels:
            try:
                strata.append(level_map[value])
            except KeyError as exc:
                raise ValueError(f"newdata contains unknown strata level {value!r}") from exc
        if any(value >= len(scales) for value in strata):
            raise ValueError("newdata strata do not match scale estimates")
        return strata

    raise ValueError("newdata strata are required for survreg quantile predictions")


def _survreg_quantile_variance_matrix(
    fit: Any,
    nvar: int,
    nscale: int,
) -> list[list[float]]:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    matrix = [[float(value) for value in row] for row in raw_variance]
    full_width = nvar + nscale
    if len(matrix) >= full_width and all(len(row) >= full_width for row in matrix[:full_width]):
        return [row[:full_width] for row in matrix[:full_width]]
    if len(matrix) >= nvar and all(len(row) >= nvar for row in matrix[:nvar]):
        return [row[:nvar] for row in matrix[:nvar]]
    raise ValueError("fitted survreg variance matrix does not match quantile width")


def _survreg_quantile_linear_values(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    quantile_scores: list[float],
    newdata: Any | None,
) -> list[list[float]]:
    result = fit.predict(rows, "lp", offsets, False)
    linear_predictors = [float(value) for value in result.predictions]
    strata = _survreg_prediction_strata(fit, newdata, len(linear_predictors))
    scales = _survreg_scales(fit)
    return [
        [linear_predictor + score * scales[strata[row_idx]] for score in quantile_scores]
        for row_idx, linear_predictor in enumerate(linear_predictors)
    ]


def _survreg_quantile_prediction_matrix(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    quantile_scores: list[float],
    predict_type: str,
    newdata: Any | None,
) -> list[list[float]]:
    linear_values = _survreg_quantile_linear_values(
        fit,
        rows,
        offsets,
        quantile_scores,
        newdata,
    )
    if predict_type != "quantile" or not _survreg_response_uses_log_transform(fit):
        return linear_values
    return [[_safe_exp(value) for value in row] for row in linear_values]


def _survreg_quantile_prediction_se_matrix(
    fit: Any,
    rows: list[list[float]] | None,
    quantile_scores: list[float],
    predictions: list[list[float]],
    predict_type: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_prediction_rows(fit, rows, "survreg quantile prediction SEs")
    scales = _survreg_scales(fit)
    strata = _survreg_prediction_strata(fit, newdata, len(prediction_rows))
    variance = _survreg_quantile_variance_matrix(fit, len(beta), len(scales))
    transform_se = predict_type == "quantile" and _survreg_response_uses_log_transform(fit)
    return _core.survreg_quantile_prediction_se_matrix(
        prediction_rows,
        scales,
        strata,
        variance,
        quantile_scores,
        predictions,
        transform_se,
    )


def _drop_single_quantile(values: list[list[float]], probabilities: list[float]) -> Any:
    if len(probabilities) == 1:
        return [row[0] for row in values]
    return values


def survreg(
    response: Surv | str | None = None,
    data: Any | None = None,
    *,
    x: Any | None = None,
    time: Any | None = None,
    time2: Any | None = None,
    status: Any | None = None,
    covariates: Any | None = None,
    weights: Any | None = None,
    offset: Any | None = None,
    offsets: Any | None = None,
    init: Any | None = None,
    initial: Any | None = None,
    initial_beta: Any | None = None,
    strata: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    dist: str | None = None,
    distribution: str | None = None,
    scale: Any = 0.0,
    parms: Any | None = None,
    model: Any = False,
    y: Any = True,
    robust: Any | None = None,
    cluster: Any | None = None,
    score: Any = False,
    max_iter: int | None = None,
    eps: float | None = None,
    tol_chol: float | None = None,
    control: Any | None = None,
    **kwargs: Any,
):
    """Fit an accelerated failure-time model using R-style or matrix inputs."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survreg got unexpected keyword argument(s): {unexpected}")

    if offset is not None and offsets is not None:
        raise ValueError("use only one of offset or offsets")
    initial_options = {
        "init": init,
        "initial": initial,
        "initial_beta": initial_beta,
    }
    if sum(value is not None for value in initial_options.values()) > 1:
        raise ValueError("use only one of init, initial, or initial_beta")
    if x is not None and covariates is not None:
        raise ValueError("use only one of x or covariates")
    scale_value = _finite_float(scale, "scale")
    if scale_value < 0.0:
        raise ValueError("scale must be non-negative")
    keep_model = _normalize_bool_option_with_default(model, "model", False)
    keep_y = _normalize_bool_option_with_default(y, "y", True)
    keep_score = _normalize_bool_option_with_default(score, "score", False)
    explicit_weights = weights is not None
    robust_requested = None if robust is None else _normalize_bool_option(robust, "robust")
    if max_iter is not None:
        max_iter = _integer_scalar(max_iter, "max_iter")
    max_iter, eps, tol_chol = _apply_survreg_control(control, max_iter, eps, tol_chol)

    formula_rows: list[list[float]] | None = None
    formula_design: _FormulaDesign | None = None
    formula_string: str | None = None
    formula_x_matrix: list[list[float]] | None = None
    formula_model_data: Any | None = None
    formula_cluster_columns: tuple[str, ...] = ()
    direct_coefficient_names: tuple[str, ...] | None = None
    matrix_input = response is None and time is not None and status is not None
    if matrix_input:
        response_time = _float_vector(time, "time")
        response_time2 = _float_vector(time2, "time2") if time2 is not None else None
        response_status = _materialize_1d(status, "status")
        matrix_values = covariates if covariates is not None else x
        direct_coefficient_names = _matrix_input_column_names(matrix_values)
        rows = _as_rows(matrix_values, "covariates")
        direct_coefficient_names = _validated_matrix_column_names(
            direct_coefficient_names,
            rows,
        )
        if subset is not None:
            indices = _subset_indices(subset, len(response_time))
            response_time = [response_time[idx] for idx in indices]
            response_time2 = (
                [response_time2[idx] for idx in indices] if response_time2 is not None else None
            )
            response_status = [response_status[idx] for idx in indices]
            rows = [rows[idx] for idx in indices]
            weights = _subset_optional_sequence(weights, indices, "weights")
            offset = _subset_optional_sequence(offset, indices, "offset")
            offsets = _subset_optional_sequence(offsets, indices, "offsets")
            strata = _subset_optional_sequence(strata, indices, "strata")
            cluster = _subset_optional_sequence(cluster, indices, "cluster")
            subset = None
        keep = _keep_rows_after_na_action(
            _missing_row_indices(
                [
                    ("time", response_time),
                    *([("time2", response_time2)] if response_time2 is not None else []),
                    ("status", response_status),
                    ("covariates", rows),
                    *(
                        (name, values)
                        for name, values in (
                            ("weights", weights),
                            ("offset", offset),
                            ("offsets", offsets),
                            ("strata", strata),
                            ("cluster", cluster),
                        )
                        if values is not None
                    ),
                ],
                len(response_time),
            ),
            len(response_time),
            na_action,
            "survreg inputs",
        )
        if keep is not None:
            response_time = [response_time[idx] for idx in keep]
            response_time2 = (
                [response_time2[idx] for idx in keep] if response_time2 is not None else None
            )
            response_status = [response_status[idx] for idx in keep]
            rows = [rows[idx] for idx in keep]
            weights = _subset_optional_sequence(weights, keep, "weights")
            offset = _subset_optional_sequence(offset, keep, "offset")
            offsets = _subset_optional_sequence(offsets, keep, "offsets")
            strata = _subset_optional_sequence(strata, keep, "strata")
            cluster = _subset_optional_sequence(cluster, keep, "cluster")
        response_status = [
            float(value)
            for value in _integer_code_vector(
                response_status,
                "status",
                "0/1/2/3 censoring codes",
            )
        ]
        distribution_name = distribution or dist or "weibull"
    else:
        if time2 is not None:
            raise ValueError("time2 is only supported with matrix time/status input")
        if isinstance(response, str):
            formula_string = response
            response_spec = _formula_response_spec(response)
            if subset is not None:
                data, aligned = _subset_formula_inputs(
                    response,
                    data,
                    subset,
                    weights=weights,
                    offset=offset,
                    offsets=offsets,
                    strata=strata,
                    cluster=cluster,
                )
                weights = aligned["weights"]
                offset = aligned["offset"]
                offsets = aligned["offsets"]
                strata = aligned["strata"]
                cluster = aligned["cluster"]
                subset = None
            data, aligned = _apply_formula_na_action(
                response,
                data,
                na_action,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
            weights = aligned["weights"]
            offset = aligned["offset"]
            offsets = aligned["offsets"]
            strata = aligned["strata"]
            cluster = aligned["cluster"]
            na_action = "pass"
            formula_x = False
            if x is not None:
                if not _is_bool_like(x):
                    raise TypeError("x must be True or False for survreg formula input")
                formula_x = _normalize_bool_option(x, "x")
            if covariates is not None:
                raise ValueError("survreg formula input cannot be combined with x or covariates")
            response, terms = _parse_formula(response, data)
            formula_design = _fit_formula_design(
                data,
                response_spec,
                terms,
                len(response),
                include_intercept=True,
            )
            formula_rows = (
                _design_rows_from_spec(data, formula_design, len(response))
                if terms.covariates or formula_design.intercept
                else [[] for _ in range(len(response))]
            )
            formula_x_matrix = [list(row) for row in formula_rows] if formula_x else None
            if terms.strata:
                if strata is not None:
                    raise ValueError("use only one of formula strata(...) or strata")
                strata = _combined_columns(data, terms.strata, len(response))
            if terms.offsets:
                if offset is not None or offsets is not None:
                    raise ValueError("use only one of formula offset(...) or offset")
                offsets = _offset_vector(data, terms.offsets, len(response))
            if terms.clusters:
                if cluster is not None:
                    raise ValueError("use only one of formula cluster(...) or cluster")
                cluster = _combined_columns(data, terms.clusters, len(response))
                formula_cluster_columns = tuple(terms.clusters)
            formula_model_data = data

        if not isinstance(response, Surv):
            raise TypeError("survreg response must be a Surv object, formula, or time/status input")
        if formula_design is None:
            direct_coefficient_names = _matrix_input_column_names(
                covariates if covariates is not None else x,
            )
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            x = _subset_optional_sequence(x, indices, "x")
            covariates = _subset_optional_sequence(covariates, indices, "covariates")
            weights = _subset_optional_sequence(weights, indices, "weights")
            offset = _subset_optional_sequence(offset, indices, "offset")
            offsets = _subset_optional_sequence(offsets, indices, "offsets")
            strata = _subset_optional_sequence(strata, indices, "strata")
            cluster = _subset_optional_sequence(cluster, indices, "cluster")
        response, aligned = _apply_surv_na_action(
            response,
            na_action,
            "survreg inputs",
            x=x,
            covariates=covariates,
            weights=weights,
            offset=offset,
            offsets=offsets,
            strata=strata,
            cluster=cluster,
        )
        x = aligned["x"]
        covariates = aligned["covariates"]
        weights = aligned["weights"]
        offset = aligned["offset"]
        offsets = aligned["offsets"]
        strata = aligned["strata"]
        cluster = aligned["cluster"]

        response_time, response_status, response_time2 = _survreg_response_arrays(response)
        rows = (
            formula_rows
            if formula_rows is not None
            else _as_rows(covariates if covariates is not None else x, "x")
        )
        direct_coefficient_names = _validated_matrix_column_names(
            direct_coefficient_names,
            rows,
        )
        distribution_name = distribution or dist or "weibull"

    n = len(response_time)
    if len(response_status) != n:
        raise ValueError("status must have the same length as time")
    if rows and len(rows) != n:
        raise ValueError("covariates must have the same number of rows as the Surv response")

    normalized_distribution = _normalize_survreg_distribution(distribution_name)
    if normalized_distribution is None:
        normalized_distribution = "weibull"
    distribution_name = normalized_distribution
    distribution_parameter = None
    if distribution_name == "t":
        distribution_parameter = _survreg_t_fit_degrees_of_freedom(parms)
    elif parms is not None:
        raise ValueError("parms is only supported for distribution='t'")
    exponential_fixed_scale = distribution_name == "exponential"
    rayleigh_fixed_scale = distribution_name == "rayleigh"
    if exponential_fixed_scale and scale_value > 0.0:
        warnings.warn(
            "Exponential has a fixed scale; user specified value ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    if rayleigh_fixed_scale and scale_value > 0.0:
        warnings.warn(
            "Rayleigh has a fixed scale; user specified value ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    offset_values = _optional_float_vector(offsets if offsets is not None else offset, "offsets", n)
    weight_values = _optional_float_vector(weights, "weights", n)
    case_weights = weight_values if explicit_weights else None
    strata_values = _encode_groups(strata, n) if strata is not None else None
    if (
        (scale_value > 0.0 or exponential_fixed_scale or rayleigh_fixed_scale)
        and strata_values is not None
        and len(set(strata_values)) > 1
    ):
        raise ValueError("cannot have both a fixed scale and strata")
    cluster_values_for_validation = (
        _materialize_labels(cluster, "cluster") if cluster is not None else None
    )
    if cluster_values_for_validation is not None:
        if len(cluster_values_for_validation) != n:
            raise ValueError("cluster must have the same length as the Surv response")
        _label_levels(cluster_values_for_validation, "cluster")
    robust_value = (
        cluster_values_for_validation is not None if robust_requested is None else robust_requested
    )
    model_frame = None
    if keep_model:
        if formula_design is not None:
            model_frame = _formula_model_frame(
                formula_model_data,
                response,
                formula_design,
                extra_columns=formula_cluster_columns,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
        elif isinstance(response, Surv):
            model_frame = _matrix_model_frame(
                response,
                rows,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
        else:
            model_frame = _survreg_matrix_model_frame(
                response_time,
                response_status,
                response_time2,
                rows,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
    initial_name, initial_source = next(
        ((name, value) for name, value in initial_options.items() if value is not None),
        ("initial", None),
    )
    initial_values = (
        _float_vector(initial_source, initial_name) if initial_source is not None else None
    )

    fixed_scale = (
        0.5
        if rayleigh_fixed_scale
        else 1.0
        if exponential_fixed_scale
        else (scale_value if scale_value > 0.0 else None)
    )
    fit = _core.survreg(
        response_time,
        response_status,
        rows,
        weights=weight_values,
        offsets=offset_values,
        initial_beta=initial_values,
        strata=strata_values,
        distribution=distribution_name,
        max_iter=max_iter,
        eps=eps,
        tol_chol=tol_chol,
        time2=response_time2,
        fixed_scale=fixed_scale,
        distribution_parameter=distribution_parameter,
    )
    robust_cluster = cluster_values_for_validation
    if robust_value and robust_cluster is None:
        robust_cluster = list(range(n))
    robust_variance = None
    naive_variance = None
    cluster_values = None
    if robust_cluster is not None and robust_value:
        robust_variance, naive_variance, cluster_values = _survreg_robust_variance_matrix(
            fit,
            robust_cluster,
        )
    score_values = list(fit.score_vector) if keep_score else None
    return (
        _FormulaFit(
            fit,
            formula_design,
            formula=formula_string,
            coefficient_names=direct_coefficient_names,
            case_weights=case_weights,
            robust_variance=robust_variance,
            naive_variance=naive_variance,
            cluster=cluster_values,
            x_matrix=formula_x_matrix,
            y_response=response if keep_y else None,
            model_frame=model_frame,
            score_values=score_values,
        )
        if (
            formula_design is not None
            or direct_coefficient_names is not None
            or case_weights is not None
            or robust_variance is not None
            or model_frame is not None
            or score_values is not None
        )
        else fit
    )
