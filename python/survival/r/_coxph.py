"""``coxph``/``clogit`` fitting plus Cox tests, baseline hazards, curves, and expected events."""

from __future__ import annotations

import math
import warnings
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from operator import index
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _SURVFIT_TIME_EPSILON,
    _apply_coxph_control,
    _as_matrix_rows,
    _clamp_probability,
    _coerce_array_like,
    _collapse_prediction_result,
    _collapse_prediction_se,
    _cox_tie_method,
    _encode_groups,
    _encode_labels,
    _event_vector,
    _finite_float,
    _float_vector,
    _hashable_group_value,
    _integer_scalar,
    _is_bool_like,
    _is_missing_value,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _matrix_input_column_names,
    _model_residual_weights,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_numeric_sequence_or_none,
    _normalize_optional_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
    _quadratic_form,
    _safe_exp,
    _subset_data,
    _subset_indices,
    _subset_optional_sequence,
    _survdiff_timefix_values,
    _survfit_confidence_interval,
    _timefix_vectors,
    _validated_matrix_column_names,
)
from ._fit import (
    _cox_alias_mask,
    _cox_degrees_of_freedom,
    _cox_detail_method,
    _cox_detail_rorder,
    _cox_detail_row_order,
    _cox_detail_strata_table,
    _cox_detail_y,
    _cox_fit_offset,
    _cox_full_loglik,
    _cox_loglik_values,
    _cox_prediction_design_rows,
    _cox_prediction_offset_vector,
    _cox_prediction_strata,
    _cox_reference_center,
    _cox_reference_means,
    _cox_reference_means_for_rows,
    _cox_strata_labels_for_fit,
    _cox_training_rows,
    _cox_training_strata,
    _cox_variance_matrix,
    _fallback_coef_names,
    _formula_design_for_fit,
    _is_clogit_fit,
    _is_coxph_fit,
    _is_survreg_fit,
    _linear_predictors_for_fit,
    _location_beta,
    _prediction_inputs,
    _require_coxph_fit,
    _surv_from_formula_design,
    _training_linear_predictor_center,
    _unwrap_formula_fit,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _combined_columns,
    _cox_time_transform_expansion,
    _cox_time_transform_functions,
    _cox_time_transform_terms,
    _cox_time_transform_values,
    _design_rows_from_spec,
    _design_term_name,
    _design_term_output_names,
    _dot_terms,
    _fit_formula_design,
    _formula_model_frame,
    _formula_response_spec,
    _matrix_model_frame,
    _offset_vector,
    _parse_formula,
    _response_arg_columns,
    _split_terms,
    _subset_formula_inputs,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._types import (
    CoxBaseHazardResult,
    CoxPHDetailResult,
    CoxPHWTestResult,
    CoxSurvfitResult,
    CoxZPHResult,
    PredictResult,
    _CovariateTerm,
    _cox_beta,
    _cox_scaled_schoenfeld_from_raw,
    _FormulaDesign,
    _FormulaFit,
)


def _cox_flat_basehaz_with_training_times(
    fit: Any,
    centered: bool,
) -> CoxBaseHazardResult:
    with_strata = getattr(fit, "basehaz_with_strata", None)
    if with_strata is None:
        base_times, base_hazards = fit.basehaz(centered)
        event_times = getattr(fit, "event_times", None)
        training_times = (
            sorted({float(value) for value in event_times})
            if event_times is not None
            else [float(value) for value in base_times]
        )
        hazards = [float(value) for value in base_hazards]
        times = [float(value) for value in base_times]
        return CoxBaseHazardResult(
            time=training_times,
            cumhaz=_core.step_values_at(times, hazards, training_times, 0.0),
            centered=centered,
        )

    base_times, base_hazards, base_strata = with_strata(centered)
    event_times = getattr(fit, "event_times", None)
    if event_times is None:
        strata_values = [int(value) for value in base_strata]
        strata = strata_values if len(set(strata_values)) > 1 else None
        return CoxBaseHazardResult(
            time=[float(value) for value in base_times],
            cumhaz=[float(value) for value in base_hazards],
            strata=strata,
            centered=centered,
            strata_labels=_cox_strata_labels_for_fit(fit, strata),
        )

    stop_times = [float(value) for value in event_times]
    row_strata = _cox_training_strata(fit, len(stop_times))
    expanded_times: list[float] = []
    expanded_hazards: list[float] = []
    expanded_strata: list[int] = []
    baselines = _cox_baselines_by_stratum(base_times, base_hazards, base_strata)
    stop_times_by_stratum: dict[int, set[float]] = {}
    for stop, row_stratum in zip(stop_times, row_strata, strict=True):
        stop_times_by_stratum.setdefault(row_stratum, set()).add(stop)

    for stratum, stratum_stop_times in sorted(stop_times_by_stratum.items()):
        stratum_times, stratum_hazards = baselines.get(stratum, ([], []))
        requested_times = sorted(stratum_stop_times)
        requested_hazards = _core.step_values_at(
            stratum_times,
            stratum_hazards,
            requested_times,
            0.0,
        )
        for time, hazard in zip(requested_times, requested_hazards, strict=True):
            expanded_times.append(time)
            expanded_hazards.append(hazard)
            expanded_strata.append(stratum)

    strata = expanded_strata if len(set(row_strata)) > 1 else None
    return CoxBaseHazardResult(
        time=expanded_times,
        cumhaz=expanded_hazards,
        strata=strata,
        centered=centered,
        strata_labels=_cox_strata_labels_for_fit(fit, strata),
    )


def _coxph_wtest_b_matrix(b: Any) -> tuple[list[list[float | None]], bool]:
    raw = _coerce_array_like(b, "b")
    if raw and isinstance(raw[0], list | tuple):
        rows: list[list[float | None]] = []
        width = len(raw[0])
        for row in raw:
            if not isinstance(row, list | tuple) or len(row) != width:
                raise ValueError("b matrix rows must be rectangular")
            rows.append([None if _is_missing_value(value) else float(value) for value in row])
        return rows, True
    return [[None if _is_missing_value(value) else float(value)] for value in raw], False


def _coxph_wtest_var_matrix(var: Any) -> tuple[list[list[float]], int]:
    raw = _coerce_array_like(var, "var")
    if raw and isinstance(raw[0], list | tuple):
        rows = _as_matrix_rows(raw, "var", allow_empty_columns=False)
        return rows, len(raw) * (len(raw[0]) if raw else 0)
    values = [float(value) for value in raw]
    if len(values) == 1:
        return [[values[0]]], 1
    return [], len(values)


def coxph_wtest(var: Any, b: Any, toler_chol: Any = 1e-9) -> CoxPHWTestResult:
    """Compute the Wald test helper exported as R's ``coxph.wtest``."""

    toler_value = _finite_float(toler_chol, "toler_chol")
    if toler_value < 0.0:
        raise ValueError("toler_chol must be non-negative")
    b_rows, b_is_matrix = _coxph_wtest_b_matrix(b)
    if any(value is None for row in b_rows for value in row):
        return CoxPHWTestResult(test=[], df=0, solve=0.0)
    b_numeric = [[float(value) for value in row] for row in b_rows]
    if any(not math.isfinite(value) for row in b_numeric for value in row):
        raise ValueError("infinite argument in coxph.wtest")

    nvar = len(b_rows)
    ntest = len(b_rows[0]) if b_rows and b_is_matrix else 1
    matrix, raw_var_length = _coxph_wtest_var_matrix(var)
    if raw_var_length == 0:
        if nvar == 0:
            return CoxPHWTestResult(test=[], df=0, solve=0.0)
        raise ValueError("Argument lengths do not match")
    if raw_var_length == 1:
        if nvar != 1:
            raise ValueError("Argument lengths do not match")
        if b_is_matrix and ntest != 1:
            raise ValueError("non-conformable arrays")
        variance = matrix[0][0]
        if not math.isfinite(variance):
            raise ValueError("infinite argument in coxph.wtest")
        if variance == 0.0:
            raise ZeroDivisionError("division by zero")
        values = [row[0] for row in b_numeric]
        return CoxPHWTestResult(
            test=[value * value / variance for value in values],
            df=1,
            solve=[value / variance for value in values],
        )

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("First argument must be a square matrix")
    if len(matrix) != nvar:
        raise ValueError("Argument lengths do not match")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("infinite argument in coxph.wtest")

    b_columns = [
        [b_numeric[row_idx][col_idx] for row_idx in range(nvar)] for col_idx in range(ntest)
    ]
    tests, df, solve_rows = _core.coxph_wtest(matrix, b_columns, toler_value)
    solve: list[float] | list[list[float]] = (
        solve_rows if b_is_matrix and ntest > 1 else [row[0] for row in solve_rows]
    )
    return CoxPHWTestResult(test=tests, df=df, solve=solve)


def basehaz(
    fit: Any | None = None,
    status: Any | None = None,
    linear_predictors: Any | None = None,
    centered: bool = True,
    *,
    newdata: Any | None = None,
    time: Any | None = None,
    entry_times: Any | None = None,
    weights: Any | None = None,
):
    """Return Cox baseline cumulative hazard, like R's basehaz."""

    if _is_clogit_fit(fit):
        raise ValueError("predicted survival curves are not defined for a clogit model")

    centered_value = _normalize_bool_option(centered, "centered")
    if time is not None:
        if newdata is not None:
            raise ValueError("newdata is only supported with fitted Cox models")
        if fit is not None:
            raise ValueError("use either a fitted Cox model or time=, not both")
        if status is None or linear_predictors is None:
            raise ValueError("status and linear_predictors are required with time=")
        time_values = _float_vector(time, "time")
        return _core.basehaz(
            time_values,
            _event_vector(status, "status"),
            _float_vector(linear_predictors, "linear_predictors"),
            centered_value,
            _optional_float_vector(entry_times, "entry_times", len(time_values)),
            _optional_float_vector(weights, "weights", len(time_values)),
        )

    if fit is None:
        raise TypeError("basehaz requires a fitted Cox model or time/status inputs")
    if hasattr(fit, "basehaz") and status is not None and linear_predictors is None:
        if newdata is not None:
            raise ValueError("use either positional newdata or newdata=, not both")
        newdata = status
        status = None
    if hasattr(fit, "basehaz") and status is None and linear_predictors is None:
        if entry_times is not None:
            raise ValueError("entry_times is already stored on fitted Cox models")
        if weights is not None:
            raise ValueError("weights are already stored on fitted Cox models")
        if newdata is not None:
            rows, offsets = _prediction_inputs(fit, newdata)
            result = _cox_survfit_result(
                fit,
                rows,
                offsets,
                True,
                newdata,
                compute_confidence=False,
            )
            if len(result.cumhaz) == 1:
                curve_strata = result.strata
                strata = [curve_strata[0]] * len(result.time) if curve_strata is not None else None
                curve_strata_labels = _cox_strata_labels_for_fit(fit, curve_strata)
                return CoxBaseHazardResult(
                    time=result.time,
                    cumhaz=result.cumhaz[0],
                    strata=strata,
                    centered=True,
                    curve_strata=curve_strata,
                    strata_labels=_cox_strata_labels_for_fit(fit, strata),
                    curve_strata_labels=curve_strata_labels,
                )
            curve_strata_labels = _cox_strata_labels_for_fit(fit, result.strata)
            return CoxBaseHazardResult(
                time=result.time,
                cumhaz=result.cumhaz,
                centered=True,
                curve_strata=result.strata,
                curve_strata_labels=curve_strata_labels,
            )
        return _cox_flat_basehaz_with_training_times(fit, centered_value)
    if newdata is not None:
        raise ValueError("newdata is only supported with fitted Cox models")
    if status is None or linear_predictors is None:
        raise ValueError("status and linear_predictors are required with raw time input")
    time_values = _float_vector(fit, "time")
    return _core.basehaz(
        time_values,
        _event_vector(status, "status"),
        _float_vector(linear_predictors, "linear_predictors"),
        centered_value,
        _optional_float_vector(entry_times, "entry_times", len(time_values)),
        _optional_float_vector(weights, "weights", len(time_values)),
    )


def cox_zph(
    fit: Any,
    transform: Any = "km",
    *,
    terms: bool = True,
    singledf: bool = False,
    global_test: bool = True,
    **kwargs: Any,
) -> CoxZPHResult:
    """R-style proportional hazards diagnostic for fitted Cox models."""

    global_test = _pop_dotted_keyword(kwargs, "global", "global_test", global_test, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected cox_zph argument(s): {unexpected}")
    if _is_clogit_fit(fit) and getattr(_unwrap_formula_fit(fit), "method", None) == "exact":
        raise ValueError("schoenfeld residuals are not available for the exact method")
    if _is_survreg_fit(fit) or not hasattr(fit, "schoenfeld_residuals"):
        raise TypeError("cox_zph requires a fitted Cox model")
    group_terms = _normalize_bool_option(terms, "terms")
    single_df = _normalize_bool_option(singledf, "singledf")
    include_global = _normalize_bool_option(global_test, "global")

    raw = [[float(value) for value in row] for row in fit.schoenfeld_residuals()]
    scaled = _cox_scaled_schoenfeld_from_raw(fit, raw)
    if len(raw) != len(scaled):
        raise ValueError("Schoenfeld residual arrays have inconsistent lengths")
    beta = _cox_beta(fit)
    aliases = _cox_alias_mask(fit)
    if len(aliases) != len(beta):
        raise ValueError("fitted Cox model alias metadata does not match coefficient width")
    active_columns = [idx for idx, aliased in enumerate(aliases) if not aliased]
    if not active_columns:
        raise ValueError("cox_zph requires at least one estimable coefficient")
    if not raw:
        raise ValueError("cox_zph requires at least one event")

    full_nvar = len(raw[0])
    if any(len(row) != full_nvar for row in raw) or any(len(row) != full_nvar for row in scaled):
        raise ValueError("Schoenfeld residual arrays must be rectangular")
    if len(beta) != full_nvar:
        raise ValueError("fitted Cox model coefficients do not match residual width")
    groups = _cox_zph_active_groups(
        _cox_zph_column_groups(fit, full_nvar, group_terms),
        active_columns,
    )
    scaled = _matrix_columns(scaled, active_columns)
    beta = [beta[idx] for idx in active_columns]

    event_indices = _cox_event_indices(fit)
    if len(event_indices) != len(raw):
        raise ValueError("fitted Cox model event times do not match Schoenfeld residuals")
    event_times = [float(fit.event_times[idx]) for idx in event_indices]
    row_strata = _cox_training_strata(fit, len(fit.status))
    event_strata_codes = [row_strata[idx] for idx in event_indices]
    design = _formula_design_for_fit(fit)
    event_strata = None
    if (design is not None and design.strata) or len(set(row_strata)) > 1:
        event_strata = _cox_strata_labels_for_fit(fit, event_strata_codes)
        if event_strata is None:
            event_strata = event_strata_codes
    transform_name, transformed_time = _cox_zph_transform(fit, event_times, transform)
    test_residuals = scaled

    test = _core.cox_zph_tests(
        test_residuals,
        transformed_time,
        [columns for _name, columns in groups],
        beta,
        single_df,
    )
    grouped_y = (
        _cox_zph_term_matrix(scaled, groups, beta)
        if group_terms and groups
        else _matrix_columns(scaled, [idx for _name, columns in groups for idx in columns])
    )
    return CoxZPHResult(
        variable_names=[name for name, _columns in groups],
        chi2_values=[float(value) for value in test.chi2_values],
        df=[1 if single_df else len(columns) for _name, columns in groups],
        p_values=[float(value) for value in test.p_values],
        x=transformed_time,
        time=event_times,
        y=grouped_y,
        var=_cox_zph_group_variance(fit, groups, beta, active_columns, len(raw)),
        transform=transform_name,
        global_chi2=float(test.global_chi2) if include_global else None,
        global_df=int(test.global_df) if include_global else None,
        global_p_value=float(test.global_p_value) if include_global else None,
        strata=event_strata,
    )


def _cox_deviance_from_martingale(martingale: list[float], status: list[float]) -> list[float]:
    if len(martingale) != len(status):
        raise ValueError("status must have the same length as martingale residuals")
    residuals: list[float] = []
    for residual, event_count in zip(martingale, status, strict=True):
        log_term = 0.0
        if event_count > 0.0:
            expected = max(event_count - residual, 1e-12)
            log_term = event_count * math.log(expected)
        magnitude = math.sqrt(max(-2.0 * (residual + log_term), 0.0))
        residuals.append(magnitude if residual >= 0.0 else -magnitude)
    return residuals


def _cox_predict_term_groups(fit: Any, nvar: int) -> list[tuple[str, list[int]]]:
    design = _formula_design_for_fit(fit)
    if design is None:
        coefficient_names = fit.coefficient_names if isinstance(fit, _FormulaFit) else None
        names = (
            list(coefficient_names)
            if coefficient_names is not None and len(coefficient_names) == nvar
            else _fallback_coef_names(nvar)
        )
        return [(name, [idx]) for idx, name in enumerate(names)]

    groups: list[tuple[str, list[int]]] = []
    cursor = 1 if design.intercept else 0
    for term in design.covariates:
        output_names = _design_term_output_names(term)
        indices = list(range(cursor, cursor + len(output_names)))
        groups.append((_design_term_name(term), indices))
        cursor += len(output_names)

    if cursor != nvar:
        return [(f"x{idx + 1}", [idx]) for idx in range(nvar)]
    return groups


def _predict_terms_selection(terms: Any | None, names: list[str]) -> list[int]:
    if terms is None:
        return list(range(len(names)))
    requested = [terms] if isinstance(terms, str) else _coerce_array_like(terms, "terms")

    selected: list[int] = []
    for value in requested:
        if isinstance(value, str):
            try:
                term_idx = names.index(value)
            except ValueError as exc:
                raise ValueError(f"terms contains unknown model term {value!r}") from exc
        else:
            try:
                term_idx = index(value) - 1
            except TypeError as exc:
                raise TypeError("terms must contain term names or 1-based term indices") from exc
            if term_idx < 0 or term_idx >= len(names):
                raise ValueError("terms indices must be between 1 and the number of model terms")
        if term_idx not in selected:
            selected.append(term_idx)
    return selected


def _cox_predict_terms(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
    reference: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    if rows is None:
        covariates = getattr(fit, "covariates", None)
        if covariates is None:
            raise ValueError("newdata is required for predict type='terms'")
        rows = [[float(value) for value in row] for row in covariates]
    if any(len(row) != len(beta) for row in rows):
        raise ValueError(f"newdata must have {len(beta)} columns")

    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    means_by_row = _cox_reference_means_for_rows(fit, reference, rows, newdata)
    return [
        [
            sum(
                (float(row[col_idx]) - means_by_row[row_idx][col_idx]) * beta[col_idx]
                for col_idx in groups[group_idx][1]
            )
            for group_idx in selected
        ]
        for row_idx, row in enumerate(rows)
    ]


def _cox_partial_residuals(
    fit: Any,
    terms: Any | None,
    martingale_weights: list[float] | None = None,
) -> list[list[float]]:
    martingale_method = getattr(fit, "martingale_residuals", None)
    if martingale_method is None:
        raise TypeError("model does not support partial residuals")
    martingale = [float(value) for value in martingale_method()]
    if martingale_weights is not None:
        if len(martingale_weights) != len(martingale):
            raise ValueError("weights must have the same length as martingale residuals")
        martingale = [
            residual * float(weight)
            for residual, weight in zip(martingale, martingale_weights, strict=True)
        ]
    contributions = _cox_predict_terms(fit, None, terms, "sample", None)
    if len(contributions) != len(martingale):
        raise ValueError("Cox term predictions do not match martingale residual length")
    return [
        [martingale[row_idx] + contribution for contribution in row]
        for row_idx, row in enumerate(contributions)
    ]


def _cox_event_indices(fit: Any) -> list[int]:
    status = [int(value) for value in fit.status]
    times = [float(value) for value in fit.event_times]
    strata_values = fit.strata if hasattr(fit, "strata") else [0] * len(status)
    strata = [int(value) for value in strata_values]
    return [int(idx) for idx in _core.cox_event_indices(times, status, strata)]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = average
        start = end + 1
    return ranks


def _cox_zph_km_transform(fit: Any, event_times: list[float]) -> list[float]:
    all_times = [float(value) for value in fit.event_times]
    status = [int(value) for value in fit.status]
    entry_times = getattr(fit, "entry_times", None)
    km = _core.survfitkm(
        all_times,
        status,
        entry_times=[float(value) for value in entry_times] if entry_times is not None else None,
        conf_type="none",
    )
    curve_times = [float(value) for value in km.time]
    estimates = [float(value) for value in km.estimate]
    transformed: list[float] = []
    cursor = 0
    for event_time in event_times:
        while (
            cursor < len(curve_times) and curve_times[cursor] < event_time - _SURVFIT_TIME_EPSILON
        ):
            cursor += 1
        previous_survival = estimates[cursor - 1] if cursor > 0 else 1.0
        transformed.append(1.0 - previous_survival)
    return transformed


def _cox_zph_transform(
    fit: Any,
    event_times: list[float],
    transform: Any,
) -> tuple[str, list[float]]:
    if callable(transform):
        transformed = _float_vector(transform(event_times), "transform result")
        if len(transformed) != len(event_times):
            raise ValueError("transform result must have the same length as event times")
        return getattr(transform, "__name__", "user"), transformed

    message = "transform must be 'km', 'rank', 'identity', 'log', or a callable"
    transform_name = "km" if transform is None else str(transform).strip().lower()
    transform_name = transform_name.replace("_", "-")
    normalized = (
        "km"
        if transform_name in {"kaplan", "kaplan-meier"}
        else _match_string_arg(
            transform_name,
            "transform",
            ("km", "rank", "identity", "log"),
            message,
        )
    )

    if normalized == "km":
        return "km", _cox_zph_km_transform(fit, event_times)
    if normalized == "rank":
        return "rank", _average_ranks(event_times)
    if normalized == "log":
        if any(value <= 0.0 for value in event_times):
            raise ValueError("log transform requires positive event times")
        return "log", [math.log(value) for value in event_times]
    return "identity", event_times


def _matrix_columns(rows: list[list[float]], columns: list[int]) -> list[list[float]]:
    return [[float(row[col_idx]) for col_idx in columns] for row in rows]


def _cox_zph_column_groups(
    fit: Any,
    nvar: int,
    terms: bool,
) -> list[tuple[str, list[int]]]:
    design = _formula_design_for_fit(fit)
    if design is None:
        return [(f"var{idx}", [idx]) for idx in range(nvar)]

    groups: list[tuple[str, list[int]]] = []
    cursor = 0
    for term in design.covariates:
        output_names = _design_term_output_names(term)
        indices = list(range(cursor, cursor + len(output_names)))
        if terms:
            groups.append((_design_term_name(term), indices))
        else:
            groups.extend((name, [idx]) for name, idx in zip(output_names, indices, strict=True))
        cursor += len(output_names)

    if cursor != nvar:
        return [(f"var{idx}", [idx]) for idx in range(nvar)]
    return groups


def _cox_zph_active_groups(
    groups: list[tuple[str, list[int]]],
    active_columns: list[int],
) -> list[tuple[str, list[int]]]:
    active_index = {
        original_index: dense_index for dense_index, original_index in enumerate(active_columns)
    }
    active_groups: list[tuple[str, list[int]]] = []
    for name, columns in groups:
        remapped = [active_index[column] for column in columns if column in active_index]
        if remapped:
            active_groups.append((name, remapped))
    return active_groups


def _cox_zph_term_matrix(
    scaled: list[list[float]],
    groups: list[tuple[str, list[int]]],
    beta: list[float],
) -> list[list[float]]:
    return _core.cox_zph_term_matrix(scaled, [columns for _name, columns in groups], beta)


def _cox_zph_group_variance(
    fit: Any,
    groups: list[tuple[str, list[int]]],
    beta: list[float],
    active_columns: list[int],
    event_count: int,
) -> list[list[float]]:
    raw_variance = getattr(fit, "naive_information_matrix", None)
    if raw_variance is None:
        raw_variance = getattr(fit, "information_matrix", None)
    if raw_variance is None:
        return []
    full_variance = [[float(value) for value in row] for row in raw_variance]
    full_nvar = len(full_variance)
    if any(len(row) != full_nvar for row in full_variance):
        return []
    if any(column < 0 or column >= full_nvar for column in active_columns):
        return []
    variance = [[full_variance[row][column] for column in active_columns] for row in active_columns]
    nvar = len(beta)
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        return []
    grouped = _core.cox_zph_group_variance(
        variance,
        [columns for _name, columns in groups],
        beta,
    )
    return [[event_count * value for value in row] for row in grouped]


def _cox_anova_test(test: str | None) -> tuple[str, bool]:
    if test is None:
        return "none", False
    if not isinstance(test, str):
        raise TypeError("anova test must be a string or None")
    value = test.strip().lower().replace("_", "-")
    if not value:
        return "none", False
    aliases = {
        "chisquare": "chisq",
        "chi-square": "chisq",
        "chi-squared": "chisq",
        "likelihood": "lrt",
        "likelihood-ratio": "lrt",
        "likelihood-ratio-test": "lrt",
    }
    normalized = aliases.get(value) or _match_string_arg(
        value,
        "anova test",
        ("chisq", "lrt", "none"),
        "anova test must be 'Chisq', 'LRT', or 'none'",
    )
    if normalized == "none":
        return "none", False
    return ("Chisq" if normalized == "chisq" else "LRT"), True


def _anova_result(
    logliks: list[float],
    dfs: list[int],
    names: list[str],
    test_name: str,
    with_tests: bool,
) -> Any:
    if with_tests and len(logliks) >= 2:
        return _core.anova_coxph(logliks, dfs, names, test_name)

    rows = []
    for name, loglik, df in zip(names, logliks, dfs, strict=True):
        rows.append(_core.AnovaRow(name, loglik, df, None, None))
    return _core.AnovaCoxphResult(rows, test_name)


def _cox_design_groups(fit: Any, n_columns: int) -> list[tuple[str, int]]:
    design = _formula_design_for_fit(fit)
    if design is None:
        return [(f"x{idx + 1}", 1) for idx in range(n_columns)]

    groups = [
        (_design_term_name(term), len(_design_term_output_names(term)))
        for term in design.covariates
    ]
    if sum(width for _, width in groups) != n_columns:
        return [(f"x{idx + 1}", 1) for idx in range(n_columns)]
    return groups


def _cox_refit_loglik_and_df(
    fit: Any,
    width: int,
    offset: list[float] | None,
) -> tuple[float, int]:
    rows = [[float(value) for value in row[:width]] for row in fit.covariates]
    nocenter = getattr(fit, "nocenter", None)
    refit = _core.coxph_fit(
        [float(value) for value in fit.event_times],
        [int(value) for value in fit.status],
        rows,
        strata=[int(value) for value in fit.strata] if hasattr(fit, "strata") else None,
        weights=[float(value) for value in fit.weights] if hasattr(fit, "weights") else None,
        offset=offset,
        initial_beta=None,
        max_iter=None,
        eps=None,
        toler=None,
        method=getattr(fit, "method", None),
        entry_times=(
            [float(value) for value in fit.entry_times]
            if getattr(fit, "entry_times", None) is not None
            else None
        ),
        nocenter=[float(value) for value in nocenter] if nocenter is not None else None,
    )
    return _cox_full_loglik(refit), _cox_degrees_of_freedom(refit)


def _anova_single_coxph(fit: Any, test_name: str, with_tests: bool) -> Any:
    model = _require_coxph_fit(fit)
    beta = _cox_beta(model)
    n_columns = len(beta)
    names = ["NULL"]
    dfs = [0]
    logliks = [_cox_loglik_values(model)[0]]
    if n_columns == 0:
        return _anova_result(logliks, dfs, names, test_name, with_tests)

    groups = _cox_design_groups(fit, n_columns)
    offset = _cox_fit_offset(model, beta)
    width = 0
    for idx, (name, group_width) in enumerate(groups):
        width += group_width
        names.append(name)
        if idx == len(groups) - 1:
            logliks.append(_cox_full_loglik(model))
            dfs.append(_cox_degrees_of_freedom(model))
        else:
            refit_loglik, refit_df = _cox_refit_loglik_and_df(model, width, offset)
            logliks.append(refit_loglik)
            dfs.append(refit_df)
    return _anova_result(logliks, dfs, names, test_name, with_tests)


def _anova_multiple_coxph(fits: tuple[Any, ...], test_name: str, with_tests: bool) -> Any:
    models = [_require_coxph_fit(fit) for fit in fits]
    logliks = [_cox_full_loglik(model) for model in models]
    dfs = [_cox_degrees_of_freedom(model) for model in models]
    names = [f"Model {idx + 1}" for idx in range(len(models))]
    return _anova_result(logliks, dfs, names, test_name, with_tests)


def _cox_robust_variance_matrix(
    fit: Any,
    cluster: Any,
) -> tuple[list[list[float]], list[list[float]], list[Any]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    naive = _cox_variance_matrix(fit, nvar)
    cluster_values = _materialize_labels(cluster, "cluster")
    score = fit.score_residuals()
    n = len(score)
    if len(cluster_values) != n:
        raise ValueError("cluster must have the same length as the Surv response")
    if any(len(row) != nvar for row in score):
        raise ValueError("fitted Cox model score residuals do not match coefficient width")

    weights = [float(value) for value in getattr(fit, "weights", [1.0] * n)]
    if len(weights) != n:
        raise ValueError("fitted Cox model weights do not match residual length")

    cluster_codes = _encode_labels(cluster_values, "cluster")
    robust = _core.clustered_sandwich_variance(score, weights, cluster_codes, naive)
    return robust, naive, cluster_values


def _cox_has_repeated_event_id(response: Surv, id_values: Sequence[Any]) -> bool:
    seen: set[Any] = set()
    for event, id_value in zip(response.event, id_values, strict=True):
        if int(event) != 1:
            continue
        key = _hashable_group_value(id_value)
        if key in seen:
            return True
        seen.add(key)
    return False


def _cox_linear_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    reference: str,
    newdata: Any | None,
) -> list[float]:
    design_rows = _cox_prediction_design_rows(fit, rows, reference, newdata)
    variance = _cox_variance_matrix(fit, len(_cox_beta(fit)))
    return _core.prediction_se_from_variance(design_rows, variance)


def _cox_term_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
    reference: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    design_rows = _cox_prediction_design_rows(fit, rows, reference, newdata)
    variance = _cox_variance_matrix(fit, len(beta))
    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    return _core.term_prediction_se_from_variance(
        design_rows,
        variance,
        [groups[group_idx][1] for group_idx in selected],
    )


@dataclass(frozen=True)
class _CoxExpectedBaseline:
    times: list[float]
    cumhaz: list[float]
    varhaz: list[float]
    xbar: list[list[float]]


def _cox_expected_baseline_by_stratum(fit: Any) -> dict[int, _CoxExpectedBaseline]:
    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    nvar = len(beta)
    rows = _cox_training_rows(model, nvar)
    times = [float(value) for value in model.event_times]
    status = [int(value) for value in model.status]
    n = len(times)
    if len(rows) != n or len(status) != n:
        raise ValueError("fitted Cox model event arrays have inconsistent lengths")

    entry_values = getattr(model, "entry_times", None)
    entry = [float(value) for value in entry_values] if entry_values is not None else None
    if entry is not None and len(entry) != n:
        raise ValueError("fitted Cox model entry times do not match event rows")
    weights = _model_residual_weights(model, n)
    strata = _cox_training_strata(model, n)
    offsets = _cox_prediction_offset_vector(model, n)
    means = _cox_reference_means(model, "sample")
    method = _cox_detail_method(model)
    strata_values, baseline_times, cumhaz, varhaz, xbar = _core.cox_expected_baseline_by_stratum(
        times,
        status,
        rows,
        beta,
        weights,
        strata,
        offsets,
        means,
        entry,
        method,
    )
    return {
        int(stratum): _CoxExpectedBaseline(
            times=[float(value) for value in stratum_times],
            cumhaz=[float(value) for value in stratum_cumhaz],
            varhaz=[float(value) for value in stratum_varhaz],
            xbar=[[float(value) for value in row] for row in stratum_xbar],
        )
        for stratum, stratum_times, stratum_cumhaz, stratum_varhaz, stratum_xbar in zip(
            strata_values,
            baseline_times,
            cumhaz,
            varhaz,
            xbar,
            strict=True,
        )
    }


def _cox_expected_baseline_at(
    baseline: _CoxExpectedBaseline,
    time: float,
    nvar: int,
) -> tuple[float, float, list[float]]:
    pos = bisect_right(baseline.times, time)
    if pos == 0:
        return 0.0, 0.0, [0.0] * nvar
    idx = pos - 1
    return baseline.cumhaz[idx], baseline.varhaz[idx], list(baseline.xbar[idx])


def _cox_training_response(fit: Any) -> Surv:
    model = _unwrap_formula_fit(fit)
    entry_values = getattr(model, "entry_times", None)
    if entry_values is None:
        return Surv(model.event_times, model.status)
    return Surv(entry_values, model.event_times, model.status)


def _cox_expected_events_with_se(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    newdata: Any | None,
) -> PredictResult:
    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    nvar = len(beta)
    if rows is None:
        rows = _cox_training_rows(model, nvar)
        if len(rows) != len(model.event_times):
            raise ValueError("stored training covariates are required for expected prediction SEs")
        response = _cox_training_response(model)
        prediction_strata = _cox_training_strata(model, len(rows))
        linear_predictors = _linear_predictors_for_fit(model, None)
    else:
        design = _formula_design_for_fit(fit)
        if design is None or not (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
            raise ValueError(
                "predict type='expected' with newdata requires formula response columns"
            )
        response = _surv_from_formula_design(newdata, design)
        if len(response) != len(rows):
            raise ValueError("newdata response and covariates must have the same row count")
        model_is_counting = getattr(model, "entry_times", None) is not None
        if model_is_counting != (response.start is not None):
            raise ValueError("newdata survival type differs from the fitted Cox model")
        prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)

    if any(len(row) != nvar for row in rows):
        raise ValueError(f"newdata must have {nvar} columns")
    means = _cox_reference_means(model, "sample")
    variance = _cox_variance_matrix(model, nvar)
    baselines = _cox_expected_baseline_by_stratum(model)

    predictions: list[float] = []
    centered_rows: list[list[float]] = []
    start_hazards: list[float] = []
    start_varhazes: list[float] = []
    start_xbars: list[list[float]] = []
    stop_hazards: list[float] = []
    stop_varhazes: list[float] = []
    stop_xbars: list[list[float]] = []
    risks: list[float] = []
    for row_idx, (row, stop, stratum, linear_predictor) in enumerate(
        zip(rows, response.time, prediction_strata, linear_predictors, strict=True)
    ):
        baseline = baselines.get(stratum)
        if baseline is None:
            raise ValueError(f"newdata contains unknown strata level {stratum!r}")
        start = response.start[row_idx] if response.start is not None else None
        start_hazard, start_varhaz, start_xbar = (
            _cox_expected_baseline_at(baseline, float(start), nvar)
            if start is not None
            else (0.0, 0.0, [0.0] * nvar)
        )
        stop_hazard, stop_varhaz, stop_xbar = _cox_expected_baseline_at(
            baseline,
            float(stop),
            nvar,
        )
        centered_row = [float(value) - means[col_idx] for col_idx, value in enumerate(row)]
        risk = _safe_exp(float(linear_predictor))
        predictions.append(max(stop_hazard - start_hazard, 0.0) * risk)
        centered_rows.append(centered_row)
        start_hazards.append(start_hazard)
        start_varhazes.append(start_varhaz)
        start_xbars.append(start_xbar)
        stop_hazards.append(stop_hazard)
        stop_varhazes.append(stop_varhaz)
        stop_xbars.append(stop_xbar)
        risks.append(risk)
    se = _core.cox_interval_cumulative_hazard_se(
        centered_rows,
        start_hazards,
        start_varhazes,
        start_xbars,
        stop_hazards,
        stop_varhazes,
        stop_xbars,
        risks,
        variance,
    )
    return PredictResult(predictions, se)


def _cox_expected_events_for_newdata(
    fit: Any,
    rows: list[list[float]],
    offsets: list[float] | None,
    newdata: Any,
) -> list[float]:
    design = _formula_design_for_fit(fit)
    if design is None or not (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
        raise ValueError("predict type='expected' with newdata requires formula response columns")

    response = _surv_from_formula_design(newdata, design)
    if len(response) != len(rows):
        raise ValueError("newdata response and covariates must have the same row count")

    model = _unwrap_formula_fit(fit)
    model_is_counting = getattr(model, "entry_times", None) is not None
    if model_is_counting != (response.start is not None):
        raise ValueError("newdata survival type differs from the fitted Cox model")

    basehaz_with_strata = getattr(model, "basehaz_with_strata", None)
    if basehaz_with_strata is None:
        base_times, base_hazards = model.basehaz(False)
        base_strata = [0] * len(base_times)
    else:
        base_times, base_hazards, base_strata = basehaz_with_strata(False)

    prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
    linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    baselines = _cox_baselines_by_stratum(base_times, base_hazards, base_strata)
    expected: list[float] = []
    for idx, (stop, stratum, linear_predictor) in enumerate(
        zip(response.time, prediction_strata, linear_predictors, strict=True)
    ):
        stratum_times, stratum_hazards = baselines.get(stratum, ([], []))
        start_hazard = (
            _step_hazard_at(stratum_times, stratum_hazards, float(response.start[idx]))
            if response.start is not None
            else 0.0
        )
        stop_hazard = _step_hazard_at(stratum_times, stratum_hazards, float(stop))
        expected.append(max(stop_hazard - start_hazard, 0.0) * _safe_exp(linear_predictor))
    return expected


def _step_curve_at(
    times: list[float],
    curve: list[float],
    requested_times: list[float],
) -> list[float]:
    return _core.step_values_at(times, curve, requested_times, 1.0)


def _step_std_err_at(
    times: list[float],
    curve: list[float],
    requested_times: list[float],
) -> list[float]:
    return _core.step_values_at(times, curve, requested_times, 0.0)


def _step_hazard_at(times: list[float], hazards: list[float], time: float) -> float:
    pos = bisect_right(times, time)
    return 0.0 if pos == 0 else hazards[pos - 1]


def _cox_baselines_by_stratum(
    base_times: list[float],
    base_hazards: list[float],
    base_strata: list[int],
) -> dict[int, tuple[list[float], list[float]]]:
    baselines: dict[int, tuple[list[float], list[float]]] = {}
    for time, hazard, stratum_value in zip(base_times, base_hazards, base_strata, strict=True):
        times, hazards = baselines.setdefault(int(stratum_value), ([], []))
        times.append(float(time))
        hazards.append(float(hazard))
    return baselines


def _cox_baseline_survival_curves(
    base_times: list[float],
    base_hazards: list[float],
    linear_predictors: list[float],
    center: float,
    base_strata: list[int] | None = None,
    curve_strata: list[int] | None = None,
    requested_times: list[float] | None = None,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    times, curves, cumhaz = _core.cox_survfit_from_baseline(
        [float(value) for value in base_times],
        [float(value) for value in base_hazards],
        [float(value) for value in linear_predictors],
        float(center),
        None if base_strata is None else [int(value) for value in base_strata],
        None if curve_strata is None else [int(value) for value in curve_strata],
        None if requested_times is None else [float(value) for value in requested_times],
    )
    return (
        [float(value) for value in times],
        [[float(value) for value in curve] for curve in curves],
        [[float(value) for value in curve] for curve in cumhaz],
    )


def _cox_survival_curve(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    centered: bool,
    newdata: Any | None,
) -> tuple[list[float], list[list[float]]]:
    with_strata = getattr(fit, "survival_curve_with_strata", None)
    if rows is not None and with_strata is not None:
        prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
        if offsets is None:
            times, curves = with_strata(rows, prediction_strata, centered)
            return [float(value) for value in times], curves

        basehaz_with_strata = getattr(fit, "basehaz_with_strata", None)
        if basehaz_with_strata is None:
            raise TypeError("model does not support stratified baseline hazard prediction")
        base_times, base_hazards, base_strata = basehaz_with_strata(centered)
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
        center = _training_linear_predictor_center(fit) if centered else 0.0
        curve_times, curves, _ = _cox_baseline_survival_curves(
            [float(value) for value in base_times],
            [float(value) for value in base_hazards],
            linear_predictors,
            center,
            [int(value) for value in base_strata],
            prediction_strata,
        )
        return curve_times, curves

    if offsets is None:
        try:
            times, curves = fit.survival_curve(rows, centered)
        except TypeError:
            if rows is None:
                raise ValueError("newdata is required for survival prediction") from None
            times, curves = fit.survival_curve(rows, None)
        return [float(value) for value in times], curves

    if rows is None:
        raise ValueError("newdata is required for survival prediction")
    if not hasattr(fit, "basehaz"):
        raise TypeError("model does not support baseline hazard prediction")

    curve_times, hazards = fit.basehaz(centered)
    linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    center = _training_linear_predictor_center(fit) if centered else 0.0
    curve_times, curves, _ = _cox_baseline_survival_curves(
        [float(value) for value in curve_times],
        [float(value) for value in hazards],
        linear_predictors,
        center,
    )
    return curve_times, curves


def _cox_default_survfit_linear_predictor(fit: Any) -> float:
    means = getattr(fit, "means", None)
    if means is None:
        return 0.0
    beta = _cox_beta(fit)
    mean_values = [float(value) for value in means]
    if len(mean_values) != len(beta):
        return 0.0
    return sum(value * coefficient for value, coefficient in zip(mean_values, beta, strict=True))


def _cox_survfit_curve_strata(
    fit: Any,
    rows: list[list[float]] | None,
    newdata: Any | None,
    n_curves: int,
) -> list[int] | None:
    if getattr(fit, "basehaz_with_strata", None) is None:
        return None
    beta = _cox_beta(fit)
    training_rows = _cox_training_rows(fit, len(beta))
    training_strata = _cox_training_strata(fit, len(training_rows)) if training_rows else [0]
    unique_strata = sorted(set(training_strata))
    if len(unique_strata) <= 1:
        return None
    if rows is None:
        if n_curves == len(unique_strata):
            return unique_strata
        return None
    prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
    return prediction_strata if len(prediction_strata) == n_curves else None


def _cox_survfit_default_time0(fit: Any) -> float:
    values = [0.0]
    event_times = getattr(fit, "event_times", None)
    if event_times is not None:
        values.extend(float(value) for value in event_times)
    entry_times = getattr(fit, "entry_times", None)
    if entry_times is not None:
        values.extend(float(value) for value in entry_times)
    return min(values)


def _cox_survfit_training_times(
    fit: Any,
    curve_strata: list[int] | None,
) -> list[float]:
    event_times = getattr(fit, "event_times", None)
    if event_times is None:
        return []
    times = [float(value) for value in event_times]
    if not times:
        return []
    strata = _cox_training_strata(fit, len(times))
    selected_strata = set(curve_strata) if curve_strata is not None else set(strata)
    return sorted(
        {time for time, stratum in zip(times, strata, strict=True) if stratum in selected_strata}
    )


def _cox_survfit_with_censor_times(
    fit: Any,
    result: CoxSurvfitResult,
) -> CoxSurvfitResult:
    times = _cox_survfit_training_times(fit, result.strata)
    if not times or times == result.time:
        return result

    expanded_cumhaz: list[list[float]] = []
    expanded_surv: list[list[float]] = []
    for hazards in result.cumhaz:
        hazard_values = [float(value) for value in hazards]
        curve_hazards = _core.step_values_at(result.time, hazard_values, times, 0.0)
        expanded_cumhaz.append(curve_hazards)
        expanded_surv.append([_clamp_probability(_safe_exp(-hazard)) for hazard in curve_hazards])

    return CoxSurvfitResult(
        time=times,
        surv=expanded_surv,
        cumhaz=expanded_cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
        start_time=result.start_time,
        std_err=result.std_err,
        std_chaz=result.std_chaz,
        conf_lower=result.conf_lower,
        conf_upper=result.conf_upper,
    )


def _cox_survfit_conditioned(
    fit: Any,
    result: CoxSurvfitResult,
    start_time: float | None,
    include_time0: bool,
) -> CoxSurvfitResult:
    if start_time is None and not include_time0:
        return result

    t0 = start_time if start_time is not None else _cox_survfit_default_time0(fit)
    times = [float(value) for value in result.time]
    kept_times, conditioned_surv, conditioned_cumhaz = _core.condition_cox_survfit_curves(
        times,
        [[float(value) for value in curve] for curve in result.cumhaz],
        float(t0),
        include_time0,
        start_time is not None,
        _SURVFIT_TIME_EPSILON,
    )

    return CoxSurvfitResult(
        time=kept_times,
        surv=conditioned_surv,
        cumhaz=conditioned_cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
        start_time=t0 if start_time is not None else None,
        std_err=result.std_err,
        std_chaz=result.std_chaz,
        conf_lower=result.conf_lower,
        conf_upper=result.conf_upper,
    )


def _cox_survfit_curve_rows(
    fit: Any,
    rows: list[list[float]] | None,
    n_curves: int,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    if rows is not None:
        curve_rows = [[float(value) for value in row] for row in rows]
        if len(curve_rows) != n_curves:
            raise ValueError("newdata rows do not match fitted Cox survival curves")
        if any(len(row) != nvar for row in curve_rows):
            raise ValueError(f"newdata must have {nvar} columns")
        return curve_rows

    means = getattr(_unwrap_formula_fit(fit), "means", None)
    if means is None:
        row = [0.0] * nvar
    else:
        row = [float(value) for value in means]
        if len(row) != nvar:
            row = _cox_reference_means(fit, "sample")
    return [list(row) for _ in range(n_curves)]


def _cox_survfit_with_confidence(
    fit: Any,
    result: CoxSurvfitResult,
    rows: list[list[float]],
    conf_level: float,
    conf_type: str,
) -> CoxSurvfitResult:
    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    nvar = len(beta)
    variance = _cox_variance_matrix(model, nvar)
    baselines = _cox_expected_baseline_by_stratum(model)
    means = _cox_reference_means(model, "sample")
    z = NormalDist().inv_cdf(1.0 - (1.0 - conf_level) / 2.0)

    std_err: list[list[float]] = []
    std_chaz: list[list[float]] = []
    conf_lower: list[list[float]] = []
    conf_upper: list[list[float]] = []

    for curve_idx, (survival_curve, row, linear_predictor) in enumerate(
        zip(result.surv, rows, result.linear_predictors, strict=True)
    ):
        stratum = result.strata[curve_idx] if result.strata is not None else 0
        baseline = baselines.get(
            stratum,
            _CoxExpectedBaseline([], [], [], []),
        )
        centered_row = [float(value) - means[col_idx] for col_idx, value in enumerate(row)]
        start_hazard, start_varhaz, start_xbar = (
            _cox_expected_baseline_at(baseline, result.start_time, nvar)
            if result.start_time is not None
            else (0.0, 0.0, [0.0] * nvar)
        )
        curve_std_err: list[float] = []
        curve_std_chaz: list[float] = []
        curve_lower: list[float] = []
        curve_upper: list[float] = []
        risk = _safe_exp(float(linear_predictor))

        for time, survival in zip(result.time, survival_curve, strict=True):
            stop_hazard, stop_varhaz, stop_xbar = _cox_expected_baseline_at(
                baseline,
                float(time),
                nvar,
            )
            start_delta = [
                start_hazard * centered_row[col_idx] - start_xbar[col_idx]
                for col_idx in range(nvar)
            ]
            stop_delta = [
                stop_hazard * centered_row[col_idx] - stop_xbar[col_idx] for col_idx in range(nvar)
            ]
            interval_delta = [stop_delta[col_idx] - start_delta[col_idx] for col_idx in range(nvar)]
            variance_value = stop_varhaz - start_varhaz + _quadratic_form(interval_delta, variance)
            chaz_se = math.sqrt(max(variance_value, 0.0)) * risk
            surv_se = float(survival) * chaz_se
            curve_std_chaz.append(chaz_se)
            curve_std_err.append(surv_se)
            if conf_type != "none":
                lower, upper = _survfit_confidence_interval(
                    float(survival),
                    surv_se,
                    z,
                    conf_type,
                )
                curve_lower.append(lower)
                curve_upper.append(upper)

        std_chaz.append(curve_std_chaz)
        std_err.append(curve_std_err)
        if conf_type != "none":
            conf_lower.append(curve_lower)
            conf_upper.append(curve_upper)

    return CoxSurvfitResult(
        time=result.time,
        surv=result.surv,
        cumhaz=result.cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
        start_time=result.start_time,
        std_err=std_err,
        std_chaz=std_chaz,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
    )


def _cox_survfit_result(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    centered: bool,
    newdata: Any | None,
    start_time: float | None = None,
    include_time0: bool = False,
    include_censor: bool = True,
    conf_level: float = 0.95,
    conf_type: str = "log",
    compute_confidence: bool = True,
) -> CoxSurvfitResult:
    times, curves = _cox_survival_curve(fit, rows, offsets, centered, newdata)
    center = _training_linear_predictor_center(fit) if centered else 0.0
    if rows is None:
        linear_predictors = [_cox_default_survfit_linear_predictor(fit)] * len(curves)
    else:
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    curve_strata = _cox_survfit_curve_strata(fit, rows, newdata, len(curves))
    curve_strata_labels = None
    if curve_strata is not None:
        curve_strata_labels = (
            _cox_strata_labels_for_fit(fit, curve_strata)
            if rows is None
            else list(range(1, len(curve_strata) + 1))
        )
    basehaz_with_strata = getattr(fit, "basehaz_with_strata", None)
    if curve_strata is not None and basehaz_with_strata is not None:
        base_times, base_hazards, base_strata = basehaz_with_strata(centered)
        _, _, cumhaz = _cox_baseline_survival_curves(
            [float(value) for value in base_times],
            [float(value) for value in base_hazards],
            linear_predictors,
            center,
            [int(value) for value in base_strata],
            curve_strata,
            times,
        )
    else:
        baseline_times, baseline_hazards = fit.basehaz(centered)
        _, _, cumhaz = _cox_baseline_survival_curves(
            [float(value) for value in baseline_times],
            [float(value) for value in baseline_hazards],
            linear_predictors,
            center,
            requested_times=times,
        )
    result = CoxSurvfitResult(
        time=times,
        surv=[[float(value) for value in curve] for curve in curves],
        cumhaz=cumhaz,
        linear_predictors=linear_predictors,
        centered=centered,
        strata=curve_strata,
        strata_labels=curve_strata_labels,
    )
    if include_censor:
        result = _cox_survfit_with_censor_times(fit, result)
    result = _cox_survfit_conditioned(fit, result, start_time, include_time0)
    if not compute_confidence:
        return result
    curve_rows = _cox_survfit_curve_rows(fit, rows, len(result.surv))
    return _cox_survfit_with_confidence(fit, result, curve_rows, conf_level, conf_type)


def _cox_survival_curve_with_se(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    centered: bool,
    newdata: Any | None,
    times: Any | None,
    collapse: Any,
) -> PredictResult:
    result = _cox_survfit_result(
        fit,
        rows,
        offsets,
        centered,
        newdata,
        include_censor=False,
        conf_type="none",
    )
    curve_times = [float(value) for value in result.time]
    curves = [[float(value) for value in curve] for curve in result.surv]
    std_err = [[float(value) for value in curve] for curve in result.std_err]

    if times is not None:
        requested_times = _float_vector(times, "times")
        curves = [_step_curve_at(curve_times, curve, requested_times) for curve in curves]
        std_err = [_step_std_err_at(curve_times, curve, requested_times) for curve in std_err]
        curve_times = requested_times

    return PredictResult(
        (curve_times, _collapse_prediction_result(curves, collapse)),
        (curve_times, _collapse_prediction_se(std_err, collapse)),
    )


def anova(*fits: Any, test: str | None = "Chisq") -> Any:
    """Analysis of deviance for one or more fitted Cox models."""

    if not fits:
        raise TypeError("anova requires at least one fitted model")
    if len(fits) == 1 and isinstance(fits[0], list | tuple):
        fits = tuple(fits[0])
        if not fits:
            raise TypeError("anova requires at least one fitted model")

    test_name, with_tests = _cox_anova_test(test)
    if len(fits) == 1:
        return _anova_single_coxph(fits[0], test_name, with_tests)
    return _anova_multiple_coxph(fits, test_name, with_tests)


def coxph_detail(
    fit: Any | None = None,
    riskmat: bool = False,
    rorder: str = "data",
    *,
    time: Any | None = None,
    status: Any | None = None,
    covariates: Any | None = None,
    coefficients: Any | None = None,
    weights: Any | None = None,
) -> Any:
    """Return event-time Cox model details, like R's coxph.detail."""

    include_riskmat = _normalize_bool_option(riskmat, "riskmat")
    raw_args = (time, status, covariates, coefficients)
    if any(value is not None for value in raw_args):
        if fit is not None:
            raise ValueError("use either a fitted Cox model or raw Cox detail arrays")
        if not all(value is not None for value in raw_args):
            raise ValueError("time, status, covariates, and coefficients are required")
        rows = _as_matrix_rows(covariates, "covariates", allow_empty_columns=True)
        return _core.coxph_detail(
            _float_vector(time, "time"),
            _event_vector(status, "status"),
            rows,
            _float_vector(coefficients, "coefficients"),
            _optional_float_vector(weights, "weights", len(rows)) if weights is not None else None,
            riskmat=include_riskmat,
        )

    if fit is None:
        raise TypeError("coxph_detail requires a fitted Cox model")
    if not _is_coxph_fit(fit):
        raise TypeError("coxph_detail requires a fitted Cox model")
    rorder_name = _cox_detail_rorder(rorder)
    model = _unwrap_formula_fit(fit)
    method = _cox_detail_method(model)
    beta = _cox_beta(model)
    nvar = len(beta)
    rows = _cox_training_rows(model, nvar)
    time = [float(value) for value in model.event_times]
    status = [int(value) for value in model.status]
    n = len(time)
    if len(status) != n or len(rows) != n:
        raise ValueError("fitted Cox model detail arrays have inconsistent lengths")

    entry_values = getattr(model, "entry_times", None)
    entry = [float(value) for value in entry_values] if entry_values is not None else None
    if entry is not None and len(entry) != n:
        raise ValueError("fitted Cox model entry times do not match event rows")
    weights = _model_residual_weights(model, n)
    strata = _cox_training_strata(model, n)
    linear_predictors = [float(value) for value in model.linear_predictors]
    if len(linear_predictors) != n:
        raise ValueError("fitted Cox model linear predictors do not match event rows")

    center = _cox_reference_center(model, "sample")
    offset = _cox_fit_offset(model, beta)
    detail = _core.coxph_detail(
        time,
        status,
        rows,
        beta,
        weights,
        entry_times=entry,
        strata=strata,
        offset=offset,
        method=method,
        center=center,
        riskmat=include_riskmat,
    )
    detail_rows = list(detail.rows)

    row_order = _cox_detail_row_order(time, status, strata, rorder_name)
    x_rows = [rows[idx] for idx in row_order]
    y_rows = _cox_detail_y(time, status, entry)
    y_rows = [y_rows[idx] for idx in row_order]
    ordered_weights = [weights[idx] for idx in row_order]
    risk_matrix = None
    sortorder = row_order if rorder_name == "time" else None
    if include_riskmat:
        native_risk_matrix = detail.riskmat
        if native_risk_matrix is None:
            raise RuntimeError("native Cox detail omitted the requested risk matrix")
        risk_matrix = [list(native_risk_matrix[idx]) for idx in row_order]

    has_case_weights = any(abs(weight - 1.0) > 1e-12 for weight in weights)
    return CoxPHDetailResult(
        time=[float(row.time) for row in detail_rows],
        nevent=[int(row.n_event) for row in detail_rows],
        nrisk=[int(row.n_risk) for row in detail_rows],
        means=[[float(value) for value in row.means] for row in detail_rows],
        score=[[float(value) for value in row.score] for row in detail_rows],
        imat=[
            [[float(value) for value in matrix_row] for matrix_row in row.imat]
            for row in detail_rows
        ],
        hazard=[float(row.hazard) for row in detail_rows],
        varhaz=[float(row.varhaz) for row in detail_rows],
        wtrisk=[float(row.wtrisk) for row in detail_rows],
        x=x_rows,
        y=y_rows,
        strata=_cox_detail_strata_table(strata, detail_rows),
        riskmat=risk_matrix,
        weights=ordered_weights if has_case_weights else None,
        nevent_wt=[float(row.n_event_weight) for row in detail_rows] if has_case_weights else None,
        nrisk_wt=[float(row.wtrisk) for row in detail_rows] if has_case_weights else None,
        sortorder=sortorder,
    )


def coxph(
    response: Surv | str,
    data: Any | None = None,
    *,
    x: Any | None = None,
    weights: Any | None = None,
    offset: Any | None = None,
    strata: Any | None = None,
    cluster: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    init: Any | None = None,
    initial_beta: Any | None = None,
    max_iter: int = 20,
    eps: float | None = None,
    toler: float | None = None,
    method: str | None = None,
    ties: str | None = None,
    robust: Any | None = None,
    model: Any = False,
    y: Any = True,
    tt: Any | None = None,
    id: Any | None = None,
    istate: Any | None = None,
    statedata: Any | None = None,
    singular_ok: Any = True,
    nocenter: Any = (-1, 0, 1),
    control: Any | None = None,
    **kwargs: Any,
):
    """Fit a Cox proportional hazards model from Surv plus covariates."""

    case_weight_column = kwargs.pop("_weights_column", None)
    id_column = kwargs.pop("_id_column", None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    singular_ok = _pop_dotted_keyword(
        kwargs,
        "singular.ok",
        "singular_ok",
        singular_ok,
        True,
    )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"coxph got unexpected keyword argument(s): {unexpected}")

    method_name = _cox_tie_method(method, ties)
    robust_value = _normalize_optional_bool_option(robust, "robust")
    explicit_weights = weights is not None
    if case_weight_column is None and isinstance(weights, str):
        case_weight_column = weights
    if id_column is None and isinstance(id, str):
        id_column = id
    for column, name in (
        (case_weight_column, "_weights_column"),
        (id_column, "_id_column"),
    ):
        if column is not None and (not isinstance(column, str) or not column):
            raise TypeError(f"{name} must be a non-empty string")
    keep_model = _normalize_bool_option_with_default(model, "model", False)
    keep_y = _normalize_bool_option_with_default(y, "y", True)
    singular_ok_value = _normalize_bool_option_with_default(singular_ok, "singular_ok", True)
    nocenter_values = _normalize_numeric_sequence_or_none(nocenter, "nocenter")
    id_arg = id
    if init is not None and initial_beta is not None:
        raise ValueError("use only one of init or initial_beta")
    max_iter = _integer_scalar(max_iter, "max_iter")
    max_iter, eps, toler, fix_time = _apply_coxph_control(control, max_iter, eps, toler)

    formula_design: _FormulaDesign | None = None
    formula_string: str | None = None
    formula_x_matrix: list[list[float]] | None = None
    formula_model_data: Any | None = None
    formula_cluster_columns: tuple[str, ...] = ()
    direct_coefficient_names: tuple[str, ...] | None = None
    time_transform_terms: list[_CovariateTerm] = []
    time_transform_functions: list[Any | None] = []
    time_transform_expanded = False
    time_transform_observed_n: int | None = None
    formula_x = False
    istate_column: str | None = None
    if isinstance(response, str):
        formula_string = response
        response_spec = _formula_response_spec(response)
        weights = _column_or_values(data, weights, "weights") if weights is not None else None
        id_arg = _column_or_values(data, id_arg, "id") if id_arg is not None else None
        istate_column = istate if isinstance(istate, str) else None
        if istate_column is not None:
            istate = _column(data, istate_column)
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                response,
                data,
                subset,
                weights=weights,
                offset=offset,
                strata=strata,
                cluster=cluster,
                id=id_arg,
                istate=istate,
            )
            weights = aligned["weights"]
            offset = aligned["offset"]
            strata = aligned["strata"]
            cluster = aligned["cluster"]
            id_arg = aligned["id"]
            istate = aligned["istate"]
            subset = None
        data, aligned = _apply_formula_na_action(
            response,
            data,
            na_action,
            weights=weights,
            offset=offset,
            strata=strata,
            cluster=cluster,
            id=id_arg,
            istate=istate,
        )
        weights = aligned["weights"]
        offset = aligned["offset"]
        strata = aligned["strata"]
        cluster = aligned["cluster"]
        id_arg = aligned["id"]
        istate = aligned["istate"]
        na_action = "pass"
        if x is not None:
            if not _is_bool_like(x):
                raise TypeError("x must be True or False for coxph formula input")
            formula_x = _normalize_bool_option(x, "x")
        response, terms = _parse_formula(response, data)
        time_transform_terms = _cox_time_transform_terms(terms)
        time_transform_functions = _cox_time_transform_functions(tt, len(time_transform_terms))
        if time_transform_terms and keep_model:
            raise ValueError("model=True is not supported for coxph fits with tt terms")
        if terms.strata:
            if strata is not None:
                raise ValueError("use only one of formula strata(...) or strata")
            strata = _combined_columns(data, terms.strata, len(response))
        if terms.offsets:
            if offset is not None:
                raise ValueError("use only one of formula offset(...) or offset")
            offset = _offset_vector(data, terms.offsets, len(response))
        if terms.clusters:
            if cluster is not None:
                raise ValueError("use only one of formula cluster(...) or cluster")
            cluster = _combined_columns(data, terms.clusters, len(response))
            formula_cluster_columns = tuple(terms.clusters)
        formula_design = _fit_formula_design(data, response_spec, terms, len(response))
        x = _design_rows_from_spec(data, formula_design, len(response))
        formula_x_matrix = [list(row) for row in x] if formula_x else None
        formula_model_data = data

    if not isinstance(response, Surv):
        raise TypeError("coxph response must be a Surv object or formula")
    if formula_design is None:
        direct_coefficient_names = _matrix_input_column_names(x)
    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        x = _subset_optional_sequence(x, indices, "x")
        weights = _subset_optional_sequence(weights, indices, "weights")
        offset = _subset_optional_sequence(offset, indices, "offset")
        strata = _subset_optional_sequence(strata, indices, "strata")
        cluster = _subset_optional_sequence(cluster, indices, "cluster")
        id_arg = _subset_optional_sequence(id_arg, indices, "id")
        istate = _subset_optional_sequence(istate, indices, "istate")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "coxph inputs",
        x=x,
        weights=weights,
        offset=offset,
        strata=strata,
        cluster=cluster,
        id=id_arg,
        istate=istate,
    )
    x = aligned["x"]
    weights = aligned["weights"]
    offset = aligned["offset"]
    strata = aligned["strata"]
    cluster = aligned["cluster"]
    id_arg = aligned["id"]
    istate = aligned["istate"]
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "coxph currently supports right-censored and counting Surv responses"
        )

    if time_transform_terms:
        if formula_design is None or formula_model_data is None:
            raise AssertionError("tt terms require formula design metadata")
        time_transform_observed_n = len(response)
        expansion = _cox_time_transform_expansion(response, strata, fix_time)
        source_indices = expansion.source_indices
        expanded_n = len(source_indices)
        expanded_data = _subset_data(formula_model_data, source_indices)
        weights = _subset_optional_sequence(weights, source_indices, "weights")
        offset = _subset_optional_sequence(offset, source_indices, "offset")
        cluster = _subset_optional_sequence(cluster, source_indices, "cluster")
        id_arg = _subset_optional_sequence(id_arg, source_indices, "id")
        istate = _subset_optional_sequence(istate, source_indices, "istate")
        transform_weights = _optional_float_vector(weights, "weights", expanded_n)
        transformed = _cox_time_transform_values(
            formula_model_data,
            time_transform_terms,
            time_transform_functions,
            expansion,
            transform_weights,
        )
        x = _design_rows_from_spec(
            expanded_data,
            formula_design,
            expanded_n,
            time_transform_values=transformed,
        )
        formula_x_matrix = [list(row) for row in x] if formula_x else None
        response = expansion.response
        strata = expansion.strata
        time_transform_expanded = True

    rows = _as_matrix_rows(x, "x", allow_empty_columns=True)
    direct_coefficient_names = _validated_matrix_column_names(direct_coefficient_names, rows)
    if len(rows) != len(response):
        raise ValueError("x must have the same number of rows as the Surv response")

    n = len(response)
    id_values = _materialize_labels(id_arg, "id") if id_arg is not None else None
    if id_values is not None and len(id_values) != n:
        raise ValueError("id must have the same length as the Surv response")
    istate_values = _materialize_1d(istate, "istate") if istate is not None else None
    if istate_values is not None and len(istate_values) != n:
        raise ValueError("istate must have the same length as the Surv response")
    fit_strata = _encode_groups(strata, n) if strata is not None else None
    fit_weights = _optional_float_vector(weights, "weights", n)
    case_weights = fit_weights if explicit_weights else None
    fit_offset = _optional_float_vector(offset, "offset", n)
    model_frame = None
    if keep_model:
        model_frame = (
            _formula_model_frame(
                formula_model_data,
                response,
                formula_design,
                extra_columns=formula_cluster_columns,
                weights=weights,
                offset=offset,
                strata=strata,
                cluster=cluster,
                id=id_values,
            )
            if formula_design is not None
            else _matrix_model_frame(
                response,
                rows,
                weights=weights,
                offset=offset,
                strata=strata,
                cluster=cluster,
                id=id_values,
            )
        )
        if istate_values is not None:
            model_frame["(istate)"] = istate_values
            if (
                istate_column is not None
                and formula_model_data is not None
                and istate_column not in model_frame
            ):
                model_frame[istate_column] = _column(formula_model_data, istate_column)

    fit_times = list(response.time)
    entry_times = list(response.start) if response.start is not None else None
    if fix_time and not time_transform_expanded:
        if entry_times is None:
            fit_times = _survdiff_timefix_values(fit_times, True)
        else:
            entry_times, fit_times = _timefix_vectors(entry_times, fit_times)
    fit = _core.coxph_fit(
        fit_times,
        list(response.event),
        rows,
        strata=fit_strata,
        weights=fit_weights,
        offset=fit_offset,
        initial_beta=(
            _float_vector(initial_beta if initial_beta is not None else init, "init")
            if init is not None or initial_beta is not None
            else None
        ),
        max_iter=max_iter,
        eps=eps,
        toler=toler,
        method=method_name,
        entry_times=entry_times,
        nocenter=nocenter_values,
    )
    if not singular_ok_value and any(_cox_alias_mask(fit)):
        raise ValueError(
            "coxph design matrix is singular; use singular_ok=True to allow dependent covariates"
        )
    has_fractional_weights = fit_weights is not None and any(
        not float(weight).is_integer() for weight in fit_weights
    )
    has_repeated_event_id = id_values is not None and _cox_has_repeated_event_id(
        response,
        id_values,
    )
    automatically_robust = cluster is not None or has_fractional_weights or has_repeated_event_id
    use_robust_variance = automatically_robust if robust_value is None else robust_value
    if cluster is not None and not use_robust_variance:
        warnings.warn(
            "cluster specified with robust=FALSE, cluster ignored",
            RuntimeWarning,
            stacklevel=2,
        )
        cluster = None

    robust_cluster = None
    if use_robust_variance:
        robust_cluster = cluster if cluster is not None else id_values
        if robust_cluster is None:
            if response.start is not None and robust_value is True:
                raise ValueError("one of cluster or id is needed for robust variance")
            robust_cluster = list(range(n))
        if method_name == "exact":
            raise ValueError("dfbeta residuals are not available for the exact method")
    robust_variance = None
    naive_variance = None
    cluster_values = None
    if robust_cluster is not None:
        robust_variance, naive_variance, cluster_values = _cox_robust_variance_matrix(
            fit,
            robust_cluster,
        )
    if (
        formula_design is not None
        or direct_coefficient_names is not None
        or case_weights is not None
        or robust_variance is not None
        or model_frame is not None
    ):
        return _FormulaFit(
            fit,
            formula_design,
            formula=formula_string,
            coefficient_names=direct_coefficient_names,
            case_weights=case_weights,
            case_weight_column=case_weight_column,
            robust_variance=robust_variance,
            naive_variance=naive_variance,
            cluster=cluster_values,
            id_values=id_values,
            id_column=id_column,
            x_matrix=formula_x_matrix,
            y_response=response if formula_design is not None and keep_y else None,
            model_frame=model_frame,
            n_observations=time_transform_observed_n,
        )
    return fit


def clogit(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    method: str = "exact",
    **kwargs: Any,
) -> Any:
    """Fit a conditional logistic model through stratified Cox regression."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if not isinstance(formula, str):
        raise TypeError("clogit formula must be a string")
    response, separator, rhs = formula.partition("~")
    response = response.strip()
    rhs = rhs.strip()
    if not separator or not response or not rhs:
        raise ValueError("clogit formula must contain a response and '~'")

    method_name = _match_string_arg(
        method,
        "method",
        ("exact", "approximate", "efron", "breslow"),
        "clogit method must be 'exact', 'approximate', 'efron', or 'breslow'",
    )
    cox_method = "breslow" if method_name == "approximate" else method_name
    cox_formula = f"Surv(rep(1, n), {response}) ~ {rhs}"
    response_columns = _response_arg_columns(response)
    terms = _split_terms(rhs, _dot_terms(data, response_columns))

    if cox_method == "exact":
        if terms.clusters:
            raise ValueError("robust variance plus the exact method is not supported")
        if weights is not None:
            warnings.warn(
                "weights ignored: not possible for the exact method",
                RuntimeWarning,
                stacklevel=2,
            )
            weights = None

    if kwargs.get("eps") is None and kwargs.get("control") is None:
        kwargs["eps"] = 1e-9
    fit = coxph(
        cox_formula,
        data=data,
        weights=weights,
        subset=subset,
        na_action=na_action,
        method=cox_method,
        **kwargs,
    )
    if not isinstance(fit, _FormulaFit):
        raise AssertionError("clogit formula fit did not preserve formula metadata")
    return replace(fit, conditional_logistic=True)
