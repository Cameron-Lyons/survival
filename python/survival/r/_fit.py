"""Accessors on fitted models and prediction-input helpers shared by coxph/survreg/models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._coerce import _as_rows, _match_string_arg, _model_residual_weights
from ._formula import (
    _column,
    _combined_columns,
    _data_column_names,
    _design_rows_from_spec,
    _design_term_output_names,
    _formula_design_row_count,
    _formula_response_values,
    _offset_vector,
)
from ._surv import Surv
from ._types import CchModelResult, _cox_beta, _FormulaDesign, _FormulaFit

_COX_NONCONVERGENCE_FLAG = 1000


def _unwrap_formula_fit(fit: Any) -> Any:
    return fit.fit if isinstance(fit, _FormulaFit | CchModelResult) else fit


def _is_clogit_fit(fit: Any) -> bool:
    return isinstance(fit, _FormulaFit) and fit.conditional_logistic


def _cox_event_count(fit: Any) -> int:
    model = _unwrap_formula_fit(fit)
    return sum(int(value) == 1 for value in model.status)


def _cox_alias_mask(fit: Any) -> list[bool]:
    """Identify coefficients removed by the fitted Cox information rank."""

    model = _unwrap_formula_fit(fit)
    width = len(_cox_beta(model))
    aliases = [False] * width
    if width == 0 or int(getattr(model, "iterations", 0)) <= 0:
        return aliases

    try:
        rank = int(model.convergence_flag)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return aliases
    if rank == _COX_NONCONVERGENCE_FLAG or rank >= width:
        return aliases

    raw_variance = getattr(model, "information_matrix", None)
    if raw_variance is None:
        return aliases
    variance = list(raw_variance)
    if len(variance) != width or any(len(row) != width for row in variance):
        return aliases

    aliases = [float(variance[idx][idx]) == 0.0 for idx in range(width)]
    if rank >= 0 and sum(aliases) != width - rank:
        return [False] * width
    return aliases


def _is_coxph_fit(fit: Any) -> bool:
    model = _unwrap_formula_fit(fit)
    return all(
        hasattr(model, name)
        for name in (
            "coefficients",
            "covariates",
            "event_times",
            "status",
            "log_likelihood",
        )
    ) and not _is_survreg_fit(model)


def _require_coxph_fit(fit: Any) -> Any:
    if not _is_coxph_fit(fit):
        raise TypeError("anova requires fitted Cox model objects")
    return _unwrap_formula_fit(fit)


def _cox_loglik_values(fit: Any) -> list[float]:
    values = [float(value) for value in getattr(fit, "log_likelihood", [])]
    if len(values) < 2:
        raise ValueError("fitted Cox model must expose null and fitted log likelihoods")
    return values


def _cox_full_loglik(fit: Any) -> float:
    return _cox_loglik_values(fit)[-1]


def _cox_degrees_of_freedom(fit: Any) -> int:
    if _cox_event_count(fit) == 0:
        return 0
    return sum(not aliased for aliased in _cox_alias_mask(fit))


def _formula_design_output_names(design: _FormulaDesign) -> list[str]:
    names = [name for term in design.covariates for name in _design_term_output_names(term)]
    if design.intercept:
        names.insert(0, "(Intercept)")
    return names


def _fallback_coef_names(width: int) -> list[str]:
    return [f"x{idx + 1}" for idx in range(width)]


def _fit_location_coef_names(fit: Any, width: int) -> list[str]:
    design = _formula_design_for_fit(fit)
    if design is not None:
        names = _formula_design_output_names(design)
        if len(names) == width:
            return names
    coefficient_names = fit.coefficient_names if isinstance(fit, _FormulaFit) else None
    if coefficient_names is not None and len(coefficient_names) == width:
        return list(coefficient_names)
    return _fallback_coef_names(width)


def _survreg_scale_coef_names(fit: Any, width: int) -> list[str]:
    if width <= 0:
        return []
    if width == 1:
        return ["Log(scale)"]
    design = _formula_design_for_fit(fit)
    if design is not None and len(design.strata_levels) == width:
        return [f"Log(scale:{level})" for level in design.strata_levels]
    return [f"Log(scale{idx + 1})" for idx in range(width)]


def _is_model_fit(fit: Any) -> bool:
    return _is_coxph_fit(fit) or _is_survreg_fit(fit)


def _require_model_fit(fit: Any, generic: str) -> Any:
    if not _is_model_fit(fit):
        raise TypeError(f"{generic} requires a fitted coxph or survreg model")
    return fit


def _cox_fit_offset(fit: Any, beta: list[float]) -> list[float] | None:
    rows = getattr(fit, "covariates", None)
    linear_predictors = getattr(fit, "linear_predictors", None)
    if rows is None or linear_predictors is None:
        return None

    offsets = []
    for row, linear_predictor in zip(rows, linear_predictors, strict=True):
        row_values = [float(value) for value in row]
        fitted = sum(
            value * coefficient for value, coefficient in zip(row_values, beta, strict=True)
        )
        offsets.append(float(linear_predictor) - fitted)
    if all(abs(value) <= 1e-12 for value in offsets):
        return None
    return offsets


def _cox_detail_method(fit: Any) -> str:
    method = str(getattr(fit, "method", "breslow")).lower().replace("_", "-")
    if method in {"breslow", "efron"}:
        return method
    raise ValueError(f"detailed output is not available for the {method} method")


def _cox_detail_rorder(rorder: Any) -> str:
    return _match_string_arg(
        rorder,
        "rorder",
        ("data", "time"),
        "rorder must be 'data' or 'time'",
    )


def _cox_detail_y(
    time: list[float],
    status: list[int],
    entry: list[float] | None,
) -> list[list[float]]:
    if entry is None:
        return [[stop, float(event)] for stop, event in zip(time, status, strict=True)]
    return [
        [start, stop, float(event)] for start, stop, event in zip(entry, time, status, strict=True)
    ]


def _cox_detail_row_order(
    time: list[float],
    status: list[int],
    strata: list[int],
    rorder: str,
) -> list[int]:
    if rorder == "data":
        return list(range(len(time)))
    if rorder == "time":
        return sorted(range(len(time)), key=lambda idx: (strata[idx], time[idx], -status[idx], idx))
    raise ValueError("rorder must be 'data' or 'time'")


def _cox_detail_strata_table(
    strata: list[int],
    detail_rows: Sequence[Any],
) -> dict[int, int] | None:
    if len(set(strata)) <= 1:
        return None
    table: dict[int, int] = {}
    for row in detail_rows:
        stratum = int(row.stratum)
        table[stratum] = table.get(stratum, 0) + 1
    return table


def _is_survreg_fit(fit: Any) -> bool:
    return hasattr(fit, "n_covariates") and hasattr(fit, "location_coefficients")


def _location_beta(fit: Any) -> list[float]:
    values = getattr(fit, "location_coefficients", None)
    if values is not None:
        return [float(value) for value in values]
    return _cox_beta(fit)


def _training_linear_predictor_center(fit: Any) -> float:
    values = getattr(fit, "linear_predictors", None)
    if values is None:
        return 0.0
    linear_predictors = [float(value) for value in values]
    if not linear_predictors:
        return 0.0
    return sum(linear_predictors) / len(linear_predictors)


def _normalize_predict_reference(
    reference: str | None,
    centered: bool | None,
    predict_type: str,
) -> str:
    if reference is None:
        if centered is False:
            return "zero"
        if centered is True or predict_type == "terms":
            return "sample"
        return "strata"
    return _match_string_arg(
        reference,
        "reference",
        ("sample", "zero", "strata"),
        "reference must be 'sample', 'zero', or 'strata'",
    )


def _cox_reference_means(fit: Any, reference: str) -> list[float]:
    beta = _cox_beta(fit)
    if reference == "zero" or not beta:
        return [0.0] * len(beta)

    means = getattr(fit, "means", None)
    if means is None:
        covariates = getattr(fit, "covariates", None)
        if covariates is None:
            return [0.0] * len(beta)
        rows = [[float(value) for value in row] for row in covariates]
        means = (
            [sum(row[col_idx] for row in rows) / len(rows) for col_idx in range(len(beta))]
            if rows
            else [0.0] * len(beta)
        )
    else:
        means = [float(value) for value in means]

    if len(means) != len(beta):
        return [0.0] * len(beta)

    nocenter = _cox_nocenter_columns(fit, len(beta))
    for col_idx in nocenter:
        means[col_idx] = 0.0
    return means


def _cox_nocenter_columns(fit: Any, nvar: int) -> set[int]:
    covariates = getattr(fit, "covariates", None)
    if covariates is None:
        return set()
    rows = [[float(value) for value in row] for row in covariates]
    if not rows or any(len(row) != nvar for row in rows):
        return set()
    values = getattr(fit, "nocenter", (-1.0, 0.0, 1.0))
    if values is None:
        return set()
    nocenter_values = [float(value) for value in values]
    if not nocenter_values:
        return set()
    nocenter: set[int] = set()
    for col_idx in range(nvar):
        if all(any(row[col_idx] == value for value in nocenter_values) for row in rows):
            nocenter.add(col_idx)
    return nocenter


def _cox_training_rows(fit: Any, nvar: int) -> list[list[float]]:
    covariates = getattr(fit, "covariates", None)
    if covariates is None:
        return []
    rows = [[float(value) for value in row] for row in covariates]
    if any(len(row) != nvar for row in rows):
        return []
    return rows


def _cox_training_strata(fit: Any, n: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata = [int(value) for value in values]
    if len(strata) != n:
        raise ValueError("fitted Cox model strata do not match training rows")
    return strata


def _cox_strata_reference_means(fit: Any, nvar: int) -> dict[int, list[float]]:
    rows = _cox_training_rows(fit, nvar)
    if not rows:
        return {0: [0.0] * nvar}
    strata = _cox_training_strata(fit, len(rows))
    weights = _model_residual_weights(fit, len(rows))
    totals: dict[int, list[float]] = {}
    weight_totals: dict[int, float] = {}
    for row, stratum, weight in zip(rows, strata, weights, strict=True):
        totals.setdefault(stratum, [0.0] * nvar)
        weight_totals[stratum] = weight_totals.get(stratum, 0.0) + weight
        for col_idx, value in enumerate(row):
            totals[stratum][col_idx] += value * weight

    nocenter = _cox_nocenter_columns(fit, nvar)
    means: dict[int, list[float]] = {}
    for stratum, values in totals.items():
        denom = weight_totals[stratum]
        row_means = [value / denom if denom > 0.0 else 0.0 for value in values]
        for col_idx in nocenter:
            row_means[col_idx] = 0.0
        means[stratum] = row_means
    return means


def _cox_prediction_strata(fit: Any, newdata: Any | None, n: int) -> list[int]:
    training_rows = _cox_training_rows(fit, len(_cox_beta(fit)))
    training_strata = _cox_training_strata(fit, len(training_rows)) if training_rows else [0]
    if newdata is None:
        if n == len(training_strata):
            return training_strata
        if len(set(training_strata)) == 1:
            return [training_strata[0]] * n
        raise ValueError("newdata strata are required for reference='strata'")

    if len(set(training_strata)) <= 1:
        return [training_strata[0]] * n

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
        return strata

    raise ValueError("newdata strata are required for reference='strata'")


def _cox_reference_means_for_rows(
    fit: Any,
    reference: str,
    rows: list[list[float]],
    newdata: Any | None,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    if reference != "strata":
        means = _cox_reference_means(fit, reference)
        return [means for _row in rows]

    strata = _cox_prediction_strata(fit, newdata, len(rows))
    means_by_stratum = _cox_strata_reference_means(fit, len(beta))
    sample_means = _cox_reference_means(fit, "sample")
    return [means_by_stratum.get(stratum, sample_means) for stratum in strata]


def _cox_prediction_design_rows(
    fit: Any,
    rows: list[list[float]] | None,
    reference: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    if rows is None:
        rows = _cox_training_rows(fit, nvar)
        if not rows and nvar:
            raise ValueError("stored training covariates are required for prediction SEs")
    if any(len(row) != nvar for row in rows):
        raise ValueError(f"newdata must have {nvar} columns")

    means_by_row = _cox_reference_means_for_rows(fit, reference, rows, newdata)
    return [
        [float(value) - float(means_by_row[row_idx][col_idx]) for col_idx, value in enumerate(row)]
        for row_idx, row in enumerate(rows)
    ]


def _cox_variance_matrix(fit: Any, nvar: int) -> list[list[float]]:
    raw_variance = getattr(fit, "information_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row] for row in raw_variance]
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        raise ValueError("fitted Cox model information matrix does not match coefficient width")
    return variance


def _location_variance_matrix(fit: Any, nvar: int) -> list[list[float]]:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row[:nvar]] for row in list(raw_variance)[:nvar]]
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        raise ValueError("fitted survreg variance matrix does not match coefficient width")
    return variance


def _survreg_has_variance_width(fit: Any, width: int) -> bool:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        return False
    matrix = list(raw_variance)
    return len(matrix) >= width and all(len(row) >= width for row in matrix[:width])


def _survreg_variance_matrix(fit: Any, width: int) -> list[list[float]]:
    if width == 0:
        return []
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row[:width]] for row in list(raw_variance)[:width]]
    if len(variance) != width or any(len(row) != width for row in variance):
        raise ValueError("fitted survreg variance matrix does not match residual width")
    return variance


def _survreg_scales(fit: Any) -> list[float]:
    values = getattr(fit, "scales", None)
    if values is None:
        values = [getattr(fit, "scale", 1.0)]
    scales = [float(value) for value in values]
    if not scales:
        raise ValueError("fitted survreg model does not expose scale values")
    return scales


def _survreg_strata(fit: Any, n: int, nstrata: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata = [int(value) for value in values]
    if len(strata) != n:
        raise ValueError("fitted survreg strata do not match training rows")
    if any(value < 0 or value >= nstrata for value in strata):
        raise ValueError("fitted survreg strata reference missing scale values")
    return strata


def _cox_reference_centers(
    fit: Any,
    reference: str,
    n: int,
    newdata: Any | None,
) -> list[float]:
    if reference == "zero":
        return [0.0] * n
    beta = _cox_beta(fit)
    offset_center = _cox_training_offset_center(fit, beta)
    if reference != "strata":
        return [_cox_reference_center(fit, reference)] * n

    strata = _cox_prediction_strata(fit, newdata, n)
    means_by_stratum = _cox_strata_reference_means(fit, len(beta))
    sample_means = _cox_reference_means(fit, "sample")
    centers: list[float] = []
    for stratum in strata:
        means = means_by_stratum.get(stratum, sample_means)
        centers.append(
            sum(value * coefficient for value, coefficient in zip(means, beta, strict=True))
            + offset_center
        )
    return centers


def _cox_training_offset_center(fit: Any, beta: list[float]) -> float:
    offsets = _cox_fit_offset(fit, beta)
    if offsets is None:
        return 0.0
    return sum(offsets) / len(offsets) if offsets else 0.0


def _cox_reference_center(fit: Any, reference: str) -> float:
    if reference == "zero":
        return 0.0
    beta = _cox_beta(fit)
    means = _cox_reference_means(fit, reference)
    return sum(
        value * coefficient for value, coefficient in zip(means, beta, strict=True)
    ) + _cox_training_offset_center(fit, beta)


def _formula_design_for_fit(fit: Any) -> _FormulaDesign | None:
    return fit.design if isinstance(fit, _FormulaFit | CchModelResult) else None


def _cox_strata_labels_for_fit(
    fit: Any,
    strata: Sequence[int] | None,
) -> list[Any] | None:
    if strata is None:
        return None
    design = _formula_design_for_fit(fit)
    if design is None or not design.strata_levels:
        return None
    labels = list(design.strata_levels)
    result: list[Any] = []
    for value in strata:
        idx = int(value)
        if idx < 0 or idx >= len(labels):
            return None
        result.append(labels[idx])
    return result


def _direct_named_prediction_rows(
    newdata: Any,
    coefficient_names: tuple[str, ...],
) -> list[list[float]]:
    columns = [_column(newdata, name) for name in coefficient_names]
    row_count = len(columns[0]) if columns else 0
    if any(len(column) != row_count for column in columns):
        raise ValueError("newdata columns must have the same length")
    return [[float(column[row_idx]) for column in columns] for row_idx in range(row_count)]


def _prediction_inputs(
    fit: Any,
    newdata: Any | None,
) -> tuple[list[list[float]] | None, list[float] | None]:
    if newdata is None:
        return None, None
    design = _formula_design_for_fit(fit)
    if design is not None and (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
        n = _formula_design_row_count(newdata, design)
        offsets = _offset_vector(newdata, list(design.offsets), n) if design.offsets else None
        return _design_rows_from_spec(newdata, design, n), offsets
    coefficient_names = getattr(fit, "coefficient_names", None)
    if coefficient_names is not None and (
        isinstance(newdata, Mapping) or hasattr(newdata, "columns")
    ):
        return _direct_named_prediction_rows(newdata, coefficient_names), None
    if isinstance(newdata, Mapping):
        raise TypeError("newdata must be a design matrix unless the fit was created from a formula")
    rows = _as_rows(newdata, "newdata")
    if (
        design is not None
        and design.intercept
        and _is_survreg_fit(fit)
        and rows
        and len(rows[0]) + 1 == len(_location_beta(fit))
    ):
        return [[1.0, *row] for row in rows], None
    return rows, None


def _linear_predictors_for_fit(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None = None,
) -> list[float]:
    if rows is None:
        values = getattr(fit, "linear_predictors", None)
        if values is not None:
            return [float(value) for value in values]
        raise ValueError(
            "newdata is required when the fitted model does not store training predictors"
        )
    if not hasattr(fit, "predict"):
        raise TypeError("model does not support prediction")
    linear_predictors = [float(value) for value in fit.predict(rows)]
    if offsets is not None:
        if len(offsets) != len(linear_predictors):
            raise ValueError("newdata offset columns must match prediction rows")
        linear_predictors = [
            value + offset for value, offset in zip(linear_predictors, offsets, strict=True)
        ]
    return linear_predictors


def _cox_prediction_offset_vector(fit: Any, n: int) -> list[float]:
    beta = _cox_beta(fit)
    offsets = _cox_fit_offset(_unwrap_formula_fit(fit), beta)
    if offsets is None:
        return [0.0] * n
    if len(offsets) != n:
        raise ValueError("fitted Cox model offsets do not match training rows")
    return [float(value) for value in offsets]


def _surv_from_formula_design(data: Any, design: _FormulaDesign) -> Surv:
    try:
        args = _formula_response_values(data, design.response)
        if len(args) == 1:
            return Surv(
                args[0],
                type=design.response.type,
                origin=design.response.origin,
            )
        if len(args) == 2:
            return Surv(
                args[0],
                args[1],
                type=design.response.type,
                origin=design.response.origin,
            )
        if len(args) == 3:
            return Surv(
                args[0],
                args[1],
                args[2],
                type=design.response.type,
                origin=design.response.origin,
            )
    except KeyError as exc:
        raise ValueError(
            "predict type='expected' with newdata requires formula response columns"
        ) from exc
    raise ValueError("formula response must have 1, 2, or 3 columns")


def _newdata_has_formula_response(fit: Any, newdata: Any | None) -> bool:
    design = _formula_design_for_fit(fit)
    if design is None or newdata is None:
        return False
    names = _data_column_names(newdata)
    if names is None:
        return False
    available = set(names)
    response_columns = set(design.response.columns)
    if response_columns <= available:
        return True
    if response_columns & available:
        raise ValueError(
            "predict type='survival' with partial formula response columns is ambiguous"
        )
    return False
