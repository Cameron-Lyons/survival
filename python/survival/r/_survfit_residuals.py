"""``survfit_residuals`` (residuals.survfit) and ``pseudo`` values."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _float_vector,
    _group_indices,
    _int_vector,
    _label_levels,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option,
    _pop_dotted_keyword,
)
from ._coxph import _step_curve_at
from ._formula import _combine_aligned_columns, _formula_response_args
from ._surv import Surv, _subset_surv
from ._survfit import (
    _survfit_counting_positions,
    _survfit_multistate_curve,
    _survfit_multistate_state_data,
    _survfit_start_time_indices,
    survfitkm_counting_influence,
    survfitkm_influence,
)
from ._types import (
    CoxSurvfitResult,
    SurvfitMultiStateResult,
    SurvfitResult,
    TurnbullSurvfitResult,
    _PseudoMatrixResult,
    _SurvfitComputation,
)


def _normalize_pseudo_type(value: Any | None) -> str:
    if value is None:
        return "survival"
    if not isinstance(value, str):
        raise TypeError("type must be a string or None")
    normalized = value.strip().lower()
    aliases = {
        "pstate": "survival",
        "survival": "survival",
        "cumhaz": "cumhaz",
        "chaz": "cumhaz",
        "rmst": "rmst",
        "rmts": "rmst",
        "auc": "rmst",
        "sojourn": "rmst",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "type must be 'pstate', 'survival', 'cumhaz', 'chaz', 'rmst', 'rmts', 'auc', "
            "or 'sojourn'"
        ) from exc


def _pseudo_eval_times(times: Any | None, eval_times: Any | None) -> list[float] | None:
    if times is not None and eval_times is not None:
        raise TypeError("use at most one of times or eval_times")
    values = times if times is not None else eval_times
    if values is None:
        return None
    if isinstance(values, str | bytes):
        raise TypeError("times must be numeric")
    try:
        result = [float(values)]
    except (TypeError, ValueError):
        result = _float_vector(values, "times")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("times must be finite")
    return result


def _pseudo_model_response(fit: Any) -> Surv:
    model = getattr(fit, "model", None)
    if model is None:
        raise TypeError("pseudo requires a survfit result with a stored model frame")
    if not isinstance(model, Mapping):
        raise TypeError("stored survfit model frame must be mapping-like")
    responses = [value for value in model.values() if isinstance(value, Surv)]
    if len(responses) != 1:
        raise TypeError("stored survfit model frame must contain exactly one Surv response")
    return responses[0]


def _pseudo_group_values_from_model(model: Mapping[Any, Any], response: Surv) -> list[Any]:
    n = len(response)
    if "group" in model:
        group_values = _materialize_1d(model["group"], "group")
        if len(group_values) == n:
            return group_values

    response_columns: set[str] = {"time", "time1", "time2", "start", "stop", "event", "status"}
    for name, value in model.items():
        if value is response and isinstance(name, str) and "~" in name:
            response_columns.update(_formula_response_args(name))

    candidate_columns: list[list[Any]] = []
    for name, values in model.items():
        if values is response or (isinstance(name, str) and name.startswith("(")):
            continue
        if isinstance(name, str) and name in response_columns:
            continue
        try:
            column = _materialize_1d(values, str(name))
        except TypeError:
            continue
        if len(column) == n:
            candidate_columns.append(column)

    if not candidate_columns:
        raise TypeError("stored grouped survfit model frame does not contain grouping columns")
    return _combine_aligned_columns(candidate_columns, n)


def _pseudo_model_id_values(model: Mapping[Any, Any], response: Surv) -> list[Any] | None:
    values = model.get("(id)")
    if values is None:
        return None
    id_values = _materialize_labels(values, "id")
    if len(id_values) != len(response):
        raise TypeError("stored survfit model frame id does not match response length")
    return id_values


def _pseudo_model_weights(model: Mapping[Any, Any], response: Surv) -> list[float] | None:
    values = model.get("(weights)")
    if values is None:
        return None
    weights = _float_vector(values, "weights")
    if len(weights) != len(response):
        raise TypeError("stored survfit model frame weights do not match response length")
    return weights


def _pseudo_subset_model_frame(
    model: Mapping[Any, Any],
    response: Surv,
    indices: Sequence[int],
) -> dict[str, Any]:
    subset: dict[str, Any] = {"response": response}
    for name in ("(weights)", "(id)", "(cluster)"):
        values = model.get(name)
        if values is not None:
            materialized = _materialize_1d(values, name)
            subset[name] = [materialized[idx] for idx in indices]
    return subset


def _pseudo_rmst_values(
    curve_time: Sequence[float],
    curve_survival: Sequence[float],
    eval_times: Sequence[float],
) -> list[float]:
    result: list[float] = []
    times = [float(value) for value in curve_time]
    survival = [float(value) for value in curve_survival]
    for eval_time in eval_times:
        target = float(eval_time)
        area = 0.0
        previous_time = 0.0
        previous_survival = 1.0
        for time, estimate in zip(times, survival, strict=True):
            if target <= previous_time:
                break
            upper = min(target, time)
            if upper > previous_time:
                area += previous_survival * (upper - previous_time)
                previous_time = upper
            if time > target:
                break
            previous_survival = estimate
        if target > previous_time:
            area += previous_survival * (target - previous_time)
        result.append(area)
    return result


def _pseudo_curve_values(
    curve: SurvfitResult,
    eval_times: Sequence[float],
    pseudo_type: str,
) -> list[float]:
    times = [float(value) for value in curve.time]
    if pseudo_type == "survival":
        return _step_curve_at(times, [float(value) for value in curve.estimate], list(eval_times))
    if pseudo_type == "cumhaz":
        return _core.step_values_at(
            times,
            [float(value) for value in curve.cumhaz],
            [float(value) for value in eval_times],
            0.0,
        )
    return _pseudo_rmst_values(times, [float(value) for value in curve.estimate], eval_times)


def _integrated_step_values(
    curve_time: Sequence[float],
    step_values: Sequence[float],
    eval_times: Sequence[float],
    *,
    start_time: float,
    initial_value: float,
) -> list[float]:
    times = [float(value) for value in curve_time]
    values = [float(value) for value in step_values]
    if len(times) != len(values):
        raise ValueError("curve_time and step_values must have the same length")

    origin = float(start_time)
    value_at_origin = float(initial_value)
    active_times: list[float] = []
    active_values: list[float] = []
    prefix_areas: list[float] = []
    previous_time = origin
    previous_value = value_at_origin
    area = 0.0
    for time, value in zip(times, values, strict=True):
        if time < origin:
            value_at_origin = value
            previous_value = value
            continue
        area += previous_value * (time - previous_time)
        active_times.append(time)
        active_values.append(value)
        prefix_areas.append(area)
        previous_time = time
        previous_value = value

    result: list[float] = []
    for eval_time in eval_times:
        target = float(eval_time)
        if target <= origin:
            result.append(0.0)
            continue
        position = bisect_right(active_times, target) - 1
        if position < 0:
            result.append(value_at_origin * (target - origin))
            continue
        result.append(
            prefix_areas[position] + active_values[position] * (target - active_times[position])
        )
    return result


def _pseudo_integrated_step_values(
    curve_time: Sequence[float],
    step_values: Sequence[float],
    eval_times: Sequence[float],
) -> list[float]:
    return _integrated_step_values(
        curve_time,
        step_values,
        eval_times,
        start_time=0.0,
        initial_value=0.0,
    )


def _pseudo_counting_candidate_cumhaz(fit: SurvfitResult, ctype: int) -> list[float]:
    hazard = 0.0
    cumhaz: list[float] = []
    event_counts = fit.n_event_count if fit.n_event_count is not None else fit.n_event
    for risk, events, event_count in zip(
        fit.n_risk,
        fit.n_event,
        event_counts,
        strict=True,
    ):
        risk_value = float(risk)
        event_value = float(events)
        event_count_value = float(event_count)
        if risk_value > 0.0 and event_value > 0.0 and event_count_value > 0.0:
            if ctype == 1:
                hazard += event_value / risk_value
            else:
                unweighted_events = int(round(event_count_value))
                if unweighted_events > 0:
                    event_step = event_value / unweighted_events
                    for step in range(unweighted_events):
                        denominator = risk_value - step * event_step
                        if denominator > 0.0:
                            hazard += event_step / denominator
        cumhaz.append(hazard)
    return cumhaz


def _pseudo_values_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-10,
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(left_value), float(right_value), rel_tol=rel_tol, abs_tol=abs_tol)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _pseudo_counting_computation(fit: SurvfitResult) -> _SurvfitComputation:
    ctype1_hazard = _pseudo_counting_candidate_cumhaz(fit, 1)
    ctype2_hazard = _pseudo_counting_candidate_cumhaz(fit, 2)
    ctype = 2 if _pseudo_values_close(fit.cumhaz, ctype2_hazard) else 1
    hazard = ctype2_hazard if ctype == 2 else ctype1_hazard
    fh_survival = [math.exp(-value) for value in hazard]
    stype = 2 if _pseudo_values_close(fit.estimate, fh_survival) else 1
    return _SurvfitComputation(stype, ctype)


def _pseudo_counting_residual_rows(
    influence: Any,
    fit: SurvfitResult,
    eval_times: Sequence[float],
    pseudo_type: str,
) -> list[list[float]]:
    times = [float(value) for value in fit.time]
    requested_times = [float(value) for value in eval_times]
    if pseudo_type == "survival":
        return _core.step_matrix_values_at(times, influence.influence_surv, requested_times, 0.0)
    if pseudo_type == "cumhaz":
        return _core.step_matrix_values_at(times, influence.influence_chaz, requested_times, 0.0)
    return [
        _pseudo_integrated_step_values(times, [float(value) for value in row], requested_times)
        for row in influence.influence_surv
    ]


def _pseudo_counting_survfit(
    fit: SurvfitResult,
    response: Surv,
    model: Mapping[Any, Any],
    eval_times: list[float] | None,
    pseudo_type: str,
    collapse: bool,
    data_frame: bool,
) -> Any:
    if eval_times is None:
        raise TypeError("times are required for counting-process pseudo-values")
    id_values = _pseudo_model_id_values(model, response)
    weights = _pseudo_model_weights(model, response)
    n_rows = len(response)
    if id_values is None:
        id_values = list(range(n_rows))
        subject_count = n_rows
    else:
        subject_count = len(_label_levels(id_values, "id"))
    if subject_count == 0:
        result = _PseudoMatrixResult([], [float(value) for value in eval_times])
        return _pseudo_matrix_or_frame(result, data_frame)

    full_values = _pseudo_curve_values(fit, eval_times, pseudo_type)
    computation = _pseudo_counting_computation(fit)
    start_values = [] if response.start is None else list(response.start)
    influence = survfitkm_counting_influence(
        start_values,
        list(response.time),
        [int(value) for value in response.event],
        [float(value) for value in fit.time],
        [float(value) for value in fit.estimate],
        cluster=id_values if collapse else list(range(n_rows)),
        weights=weights,
        stype=computation.stype,
        ctype=computation.ctype,
    )
    residual_rows = _pseudo_counting_residual_rows(influence, fit, eval_times, pseudo_type)
    scale = float(subject_count)
    pseudo_matrix: list[list[float]] = []
    for residuals in residual_rows:
        pseudo_matrix.append(
            [
                full_value + scale * residual
                for full_value, residual in zip(full_values, residuals, strict=True)
            ]
        )

    result = _PseudoMatrixResult(pseudo_matrix, [float(value) for value in eval_times])
    return _pseudo_matrix_or_frame(result, data_frame)


def _pseudo_for_grouped_survfit(
    fit: Mapping[Any, Any],
    eval_times: list[float] | None,
    pseudo_type: str,
    collapse: bool,
    data_frame: bool,
) -> Any:
    grouped_result: dict[Any, Any] = {}
    for label, curve in fit.items():
        response = _pseudo_model_response(curve)
        if response.type not in {"right", "counting"}:
            raise NotImplementedError(
                "pseudo currently supports right-censored or counting survfit results"
            )
        group_values = _pseudo_group_values_from_model(curve.model, response)
        indices = [idx for idx, value in enumerate(group_values) if value == label]
        if not indices:
            raise TypeError("stored grouped survfit model frame does not match curve labels")
        group_response = _subset_surv(response, indices)
        if response.type == "counting":
            group_model = _pseudo_subset_model_frame(curve.model, group_response, indices)
            grouped_result[label] = _pseudo_counting_survfit(
                curve,
                group_response,
                group_model,
                eval_times,
                pseudo_type,
                collapse,
                data_frame,
            )
            continue
        result = _core.pseudo(
            list(group_response.time),
            list(group_response.event),
            eval_times,
            pseudo_type,
        )
        grouped_result[label] = _pseudo_matrix_or_frame(result, data_frame)

    if not data_frame:
        return grouped_result

    frame: dict[str, list[Any]] = {"strata": [], "id": [], "time": [], "pseudo": []}
    for label, group_frame in grouped_result.items():
        row_count = len(group_frame["pseudo"])
        frame["strata"].extend([str(label)] * row_count)
        frame["id"].extend(group_frame["id"])
        frame["time"].extend(group_frame["time"])
        frame["pseudo"].extend(group_frame["pseudo"])
    return frame


def _pseudo_matrix_or_frame(result: Any, data_frame: bool) -> Any:
    matrix = [[float(value) for value in row] for row in result.pseudo]
    if not data_frame:
        return matrix
    frame: dict[str, list[float | int]] = {"id": [], "time": [], "pseudo": []}
    for row_idx, row in enumerate(matrix, start=1):
        for time, value in zip(result.time, row, strict=True):
            frame["id"].append(row_idx)
            frame["time"].append(float(time))
            frame["pseudo"].append(float(value))
    return frame


def _survfit_multistate_integrated_estimate(
    curve_time: Sequence[float],
    estimates: Sequence[float],
    initial: float,
    t0: float,
    eval_times: Sequence[float],
) -> list[float]:
    return _integrated_step_values(
        curve_time,
        estimates,
        eval_times,
        start_time=t0,
        initial_value=initial,
    )


def _survfit_multistate_estimates_at_times(
    curve: SurvfitMultiStateResult,
    eval_times: Sequence[float],
    residual_type: str,
) -> list[list[float]]:
    if residual_type == "cumhaz":
        return [
            _core.step_values_at(
                list(curve.time),
                [float(row[column]) for row in curve.cumhaz],
                [float(value) for value in eval_times],
                0.0,
            )
            for column in range(len(curve.transitions))
        ]
    if residual_type == "auc":
        return [
            _survfit_multistate_integrated_estimate(
                curve.time,
                [float(row[state]) for row in curve.pstate],
                curve.p0[state],
                curve.t0,
                eval_times,
            )
            for state in range(len(curve.states))
        ]
    return [
        _core.step_values_at(
            list(curve.time),
            [float(row[state]) for row in curve.pstate],
            [float(value) for value in eval_times],
            float(curve.p0[state]),
        )
        for state in range(len(curve.states))
    ]


def _pseudo_multistate_survfit(
    fit: SurvfitMultiStateResult | Mapping[Any, Any],
    eval_times: list[float] | None,
    pseudo_type: str,
    collapse: bool,
    data_frame: bool,
) -> dict[str, Any]:
    if eval_times is None:
        raise TypeError("the times argument is required")
    residual_type = {
        "survival": "pstate",
        "cumhaz": "cumhaz",
        "rmst": "auc",
    }[pseudo_type]
    residual_result = survfit_residuals(
        fit,
        times=eval_times,
        type=residual_type,
        collapse=collapse,
        weighted=collapse,
        data_frame=False,
        extra=True,
    )
    curves = list(fit.values()) if isinstance(fit, Mapping) else [fit]
    if not all(isinstance(curve, SurvfitMultiStateResult) for curve in curves):
        raise TypeError("multi-state pseudo-values require multi-state survfit results")
    estimate_cache = [
        _survfit_multistate_estimates_at_times(curve, eval_times, residual_type) for curve in curves
    ]
    curve_numbers = residual_result["curve"]
    pseudo_values: list[list[list[float]]] = []
    for row_idx, residual_columns in enumerate(residual_result["resid"]):
        curve_idx = 0 if curve_numbers is None else int(curve_numbers[row_idx]) - 1
        curve = curves[curve_idx]
        scale = float(curve.n_id)
        estimates = estimate_cache[curve_idx]
        pseudo_values.append(
            [
                [
                    float(estimate) + scale * float(residual)
                    for estimate, residual in zip(
                        estimates[column],
                        residual_columns[column],
                        strict=True,
                    )
                ]
                for column in range(len(estimates))
            ]
        )
    result = dict(residual_result)
    result["pseudo"] = pseudo_values
    result["data_frame"] = data_frame
    return result


def pseudo(
    fit: Any = None,
    status: Any | None = None,
    eval_times: Any | None = None,
    type_: Any | None = None,
    *,
    times: Any | None = None,
    type: Any | None = None,
    collapse: bool = True,
    data_frame: bool = False,
    time: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Compute pseudo-values, preserving direct vector and R-style ``survfit`` calls."""

    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"pseudo got unexpected keyword argument(s): {unexpected}")
    if type is not None and type_ is not None:
        raise TypeError("use at most one of type or type_")
    pseudo_type = _normalize_pseudo_type(type if type is not None else type_)
    eval_time_values = _pseudo_eval_times(times, eval_times)

    if time is not None:
        if fit is not None:
            raise TypeError("time= cannot be combined with fit")
        fit = time
    if status is not None:
        return _core.pseudo(
            _float_vector(fit, "time"),
            _int_vector(status, "status"),
            eval_time_values,
            pseudo_type,
        )
    if fit is None:
        raise TypeError("pseudo requires a survfit result or time/status vectors")
    collapse_value = _normalize_bool_option(collapse, "collapse")
    data_frame = _normalize_bool_option(data_frame, "data_frame")
    is_multistate = isinstance(fit, SurvfitMultiStateResult) or (
        isinstance(fit, Mapping)
        and bool(fit)
        and all(isinstance(curve, SurvfitMultiStateResult) for curve in fit.values())
    )
    if is_multistate:
        return _pseudo_multistate_survfit(
            fit,
            eval_time_values,
            pseudo_type,
            collapse_value,
            data_frame,
        )
    if isinstance(fit, Mapping):
        return _pseudo_for_grouped_survfit(
            fit,
            eval_time_values,
            pseudo_type,
            collapse_value,
            data_frame,
        )

    response = _pseudo_model_response(fit)
    model = getattr(fit, "model", None)
    if response.type == "counting":
        if not isinstance(fit, SurvfitResult) or not isinstance(model, Mapping):
            raise TypeError(
                "counting-process pseudo-values require a survfit result with a stored model frame"
            )
        return _pseudo_counting_survfit(
            fit,
            response,
            model,
            eval_time_values,
            pseudo_type,
            collapse_value,
            data_frame,
        )
    if response.type != "right":
        raise NotImplementedError("pseudo currently supports right-censored survfit results")
    result = _core.pseudo(
        list(response.time),
        list(response.event),
        eval_time_values,
        pseudo_type,
    )
    return _pseudo_matrix_or_frame(result, data_frame)


def _survfit_residual_type(value: str) -> str:
    choices = ("pstate", "cumhaz", "sojourn", "survival", "chaz", "rmst", "rmts", "auc")
    normalized = value.casefold().replace("-", "_")
    if normalized in choices:
        matched = normalized
    else:
        matches = [choice for choice in choices if choice.startswith(normalized)]
        if len(matches) != 1:
            raise ValueError(
                "type must be one of pstate, cumhaz, sojourn, survival, chaz, rmst, rmts, or auc"
            )
        matched = matches[0]
    if matched in {"pstate", "survival"}:
        return "pstate"
    if matched in {"cumhaz", "chaz"}:
        return "cumhaz"
    return "auc"


def _survfit_residual_times(times: Any | None) -> list[float]:
    if times is None:
        raise TypeError("the times argument is required")
    if isinstance(times, int | float) and not isinstance(times, bool):
        return [float(times)]
    return sorted(set(_float_vector(times, "times")))


def _survfit_residual_model_frame(fit: Any) -> Mapping[Any, Any]:
    if isinstance(fit, Mapping):
        if not fit:
            raise TypeError("residuals.survfit requires a non-empty survfit result")
        return _survfit_residual_model_frame(next(iter(fit.values())))
    frame = getattr(fit, "model", None)
    if frame is None:
        raise TypeError("residuals.survfit requires a survfit result with a stored model frame")
    if not isinstance(frame, Mapping):
        raise TypeError("stored survfit model frame must be mapping-like")
    return frame


def _survfit_residual_response(frame: Mapping[Any, Any]) -> Surv:
    for value in frame.values():
        if isinstance(value, Surv):
            if value.type not in {"right", "counting", "mright", "mcounting"}:
                raise NotImplementedError(
                    "residuals.survfit currently supports right-censored or counting fits"
                )
            return value
    raise TypeError("stored survfit model frame does not contain a Surv response")


def _survfit_residual_weights(
    frame: Mapping[Any, Any],
    n: int,
) -> tuple[list[float], bool]:
    if "(weights)" not in frame:
        return [1.0] * n, False
    weights = _float_vector(frame["(weights)"], "weights")
    if len(weights) != n:
        raise ValueError("weights must have the same length as the Surv response")
    return weights, True


def _survfit_residual_ids(frame: Mapping[Any, Any], n: int) -> tuple[list[Any], str | None, bool]:
    if "(id)" not in frame:
        return list(range(1, n + 1)), None, False
    values = _materialize_labels(frame["(id)"], "id")
    if len(values) != n:
        raise ValueError("id must have the same length as the Surv response")
    return values, "(id)", True


def _survfit_residual_group_values(frame: Mapping[Any, Any], n: int) -> list[Any] | None:
    if "group" in frame:
        values = _materialize_labels(frame["group"], "group")
        if len(values) != n:
            raise ValueError("group must have the same length as the Surv response")
        return values

    columns: list[list[Any]] = []
    for name, values in frame.items():
        text_name = str(name)
        if isinstance(values, (Surv, Mapping)) or text_name.startswith("("):
            continue
        if text_name in {"response", "time", "status", "start", "stop", "time2"}:
            continue
        materialized = _materialize_1d(values, text_name)
        if len(materialized) != n:
            continue
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns.append(materialized)

    if not columns:
        return None
    return _combine_aligned_columns(columns, n)


def _survfit_residual_grouped_indices(
    fit: Mapping[Any, Any],
    group_values: list[Any] | None,
    n: int,
) -> list[tuple[Any, Any, list[int], int]]:
    if group_values is None:
        raise TypeError("grouped survfit residuals require stored grouping columns")
    grouped = _group_indices(group_values, n)
    ordered_groups = list(grouped.items())
    curves: list[tuple[Any, Any, list[int], int]] = []
    for curve_idx, (label, curve) in enumerate(fit.items(), start=1):
        indices = grouped.get(label)
        if indices is None:
            matching = [values for key, values in grouped.items() if str(key) == str(label)]
            if len(matching) == 1:
                indices = matching[0]
        if indices is None and curve_idx <= len(ordered_groups):
            indices = ordered_groups[curve_idx - 1][1]
        if indices is None:
            raise ValueError(f"could not match grouped survfit curve {label!r} to model rows")
        curves.append((label, curve, indices, curve_idx))
    return curves


def _survfit_residual_curve_specs(
    fit: Any,
    frame: Mapping[Any, Any],
    n: int,
) -> list[tuple[Any, Any, list[int], int]]:
    if isinstance(fit, Mapping):
        return _survfit_residual_grouped_indices(
            fit,
            _survfit_residual_group_values(frame, n),
            n,
        )
    return [(None, fit, list(range(n)), 1)]


def _survfit_multistate_auc_influence_at_times(
    curve_times: Sequence[float],
    state_influence: Sequence[float],
    area_influence: Sequence[float],
    initial_influence: float,
    t0: float,
    eval_times: Sequence[float],
) -> list[float]:
    times = [float(value) for value in curve_times]
    state = [float(value) for value in state_influence]
    area = [float(value) for value in area_influence]
    values: list[float] = []
    for target_value in eval_times:
        target = float(target_value)
        if target <= t0:
            values.append(0.0)
            continue
        time_idx = bisect_right(times, target) - 1
        if time_idx < 0:
            values.append((target - t0) * initial_influence)
            continue
        values.append(area[time_idx] + (target - times[time_idx]) * state[time_idx])
    return values


def _survfit_multistate_residual_block(
    curve: SurvfitMultiStateResult,
    response: Surv,
    case_weights: list[float],
    id_values: list[Any] | None,
    istate: list[Any] | None,
    group_labels: list[Any],
    eval_times: Sequence[float],
    residual_type: str,
    weighted: bool,
) -> tuple[list[Any], list[list[list[float]]], list[str]]:
    if curve.oldstate is not None:
        raise ValueError("residuals not available for a subscripted survfit object")
    normalized, current_states, states, history_ids = _survfit_multistate_state_data(
        response,
        id_values,
        istate,
        curve.timefix,
    )
    if states != curve.states:
        raise ValueError("error in residuals.survfit, non-matching states")
    keep = _survfit_start_time_indices(normalized, curve.t0, curve.timefix)
    normalized = _subset_surv(normalized, keep)
    current_states = [current_states[idx] for idx in keep]
    history_ids = [history_ids[idx] for idx in keep]
    case_weights = [case_weights[idx] for idx in keep]
    kept_group_labels = [group_labels[idx] for idx in keep]
    influence_weights = list(case_weights) if weighted else [1.0] * len(keep)
    positions = (
        [3] * len(normalized)
        if normalized.start is None
        else _survfit_counting_positions(
            normalized.start,
            normalized.time,
            history_ids,
            curve.timefix,
        )
    )
    initial_rows = (
        [True] * len(normalized)
        if normalized.start is None
        else [
            start <= curve.t0 <= stop
            if curve.t0 == min(normalized.start)
            else start < curve.t0 <= stop
            for start, stop in zip(normalized.start, normalized.time, strict=True)
        ]
    )
    influence_curve = _survfit_multistate_curve(
        normalized,
        case_weights,
        history_ids,
        kept_group_labels,
        current_states,
        positions,
        curve.states,
        curve.transitions,
        t0=curve.t0,
        output_times=list(curve.time),
        initial_rows=initial_rows,
        p0_override=list(curve.p0) if curve.p0_fixed else None,
        report_initial_error=False,
        include_se=True,
        include_entry=False,
        conf_level=curve.conf_level,
        conf_type="none",
        model_frame=None,
        timefix=curve.timefix,
        save_influence=True,
        influence_weights=influence_weights,
    )
    groups = _label_levels(kept_group_labels, "influence groups")
    group_count = len(groups)
    if residual_type == "cumhaz":
        raw_values = influence_curve.influence_chaz
        columns = [f"{source + 1}:{target + 1}" for source, target in curve.transitions]
        column_count = len(columns)
    else:
        raw_values = (
            influence_curve.influence_auc
            if residual_type == "auc"
            else influence_curve.influence_state
        )
        columns = list(curve.states)
        column_count = len(columns)
    if raw_values is None:
        raise RuntimeError("multi-state influence values were not returned by the core fit")
    initial_values = influence_curve.influence_state0
    state_values = influence_curve.influence_state
    if initial_values is None or state_values is None:
        raise RuntimeError("multi-state state influence values were not returned by the core fit")

    block = [
        [[0.0 for _time in eval_times] for _column in range(column_count)]
        for _group in range(group_count)
    ]
    for column in range(column_count):
        for group in range(group_count):
            raw_row = raw_values[group + column * group_count]
            if residual_type == "auc":
                values = _survfit_multistate_auc_influence_at_times(
                    influence_curve.time,
                    state_values[group + column * group_count],
                    raw_row,
                    initial_values[group + column * group_count],
                    curve.t0,
                    eval_times,
                )
            else:
                initial = (
                    initial_values[group + column * group_count]
                    if residual_type == "pstate"
                    else 0.0
                )
                values = _core.step_values_at(
                    list(influence_curve.time),
                    [float(value) for value in raw_row],
                    [float(value) for value in eval_times],
                    initial,
                )
            block[group][column] = [float(value) for value in values]
    return groups, block, columns


def _survfit_multistate_residual_result(
    frame: Mapping[Any, Any],
    response: Surv,
    case_weights: list[float],
    id_values: list[Any],
    id_name: str | None,
    curve_specs: list[tuple[Any, Any, list[int], int]],
    eval_times: Sequence[float],
    residual_type: str,
    collapse: bool,
    weighted: bool,
    data_frame: bool,
    extra: bool,
) -> dict[str, Any]:
    first_curve = curve_specs[0][1]
    if not isinstance(first_curve, SurvfitMultiStateResult):
        raise TypeError("multi-state residuals require a multi-state survfit result")
    columns = (
        [f"{source + 1}:{target + 1}" for source, target in first_curve.transitions]
        if residual_type == "cumhaz"
        else list(first_curve.states)
    )
    output_ids = list(_label_levels(id_values, "id")) if collapse else list(id_values)
    output_index = {value: idx for idx, value in enumerate(output_ids)} if collapse else None
    cube = [[[0.0 for _time in eval_times] for _column in columns] for _row in output_ids]
    curve_numbers = [0 for _row in output_ids] if len(curve_specs) > 1 else None
    istate_values = None
    if "(istate)" in frame:
        istate_values = _materialize_1d(frame["(istate)"], "istate")
        if len(istate_values) != len(response):
            raise ValueError("istate must have the same length as the Surv response")

    for _label, curve, indices, curve_idx in curve_specs:
        if not isinstance(curve, SurvfitMultiStateResult):
            raise TypeError("grouped survfit results must not mix curve types")
        source_response = _subset_surv(response, indices)
        source_weights = [case_weights[idx] for idx in indices]
        source_ids = [id_values[idx] for idx in indices]
        source_istate = None if istate_values is None else [istate_values[idx] for idx in indices]
        group_labels = source_ids if collapse else list(indices)
        groups, block, block_columns = _survfit_multistate_residual_block(
            curve,
            source_response,
            source_weights,
            source_ids if "(id)" in frame else None,
            source_istate,
            group_labels,
            eval_times,
            residual_type,
            weighted,
        )
        if block_columns != columns:
            raise ValueError("grouped multi-state residual curves must share output columns")
        for group_idx, group in enumerate(groups):
            target = output_index[group] if output_index is not None else int(group)
            if curve_numbers is not None and curve_numbers[target] not in {0, curve_idx}:
                raise ValueError("same id appears in multiple curves, cannot collapse")
            cube[target] = block[group_idx]
            if curve_numbers is not None:
                curve_numbers[target] = curve_idx

    return {
        "resid": cube,
        "id": output_ids,
        "id_name": id_name,
        "time": [float(value) for value in eval_times],
        "columns": columns,
        "column_name": "transition" if residual_type == "cumhaz" else "state",
        "curve": curve_numbers,
        "data_frame": data_frame,
        "extra": extra,
    }


def _survfit_residual_rows_at_times(
    influence: Any,
    times: Sequence[float],
    residual_type: str,
) -> list[list[float]]:
    curve_times = [float(value) for value in influence.time]
    eval_times = [float(value) for value in times]
    if residual_type == "cumhaz":
        return _core.step_matrix_values_at(curve_times, influence.influence_chaz, eval_times, 0.0)
    if residual_type == "auc":
        return [
            _pseudo_integrated_step_values(curve_times, [float(value) for value in row], eval_times)
            for row in influence.influence_surv
        ]
    return _core.step_matrix_values_at(curve_times, influence.influence_surv, eval_times, 0.0)


def _survfit_residual_matrix(
    response: Surv,
    weights: list[float],
    curve_specs: list[tuple[Any, Any, list[int], int]],
    times: Sequence[float],
    residual_type: str,
) -> tuple[list[list[float]], list[int] | None]:
    matrix = [[0.0 for _ in times] for _ in range(len(response))]
    curve_numbers = [0 for _ in range(len(response))] if len(curve_specs) > 1 else None
    for _label, curve, indices, curve_idx in curve_specs:
        if not isinstance(curve, SurvfitResult):
            raise TypeError("residuals.survfit currently supports Kaplan-Meier survfit results")
        group_response = _subset_surv(response, indices)
        if group_response.type == "counting":
            if group_response.start is None:
                raise ValueError("counting-process Surv response is missing start times")
            group_weights = [weights[idx] for idx in indices]
            computation = _pseudo_counting_computation(curve)
            influence = survfitkm_counting_influence(
                list(group_response.start),
                list(group_response.time),
                [int(value) for value in group_response.event],
                [float(value) for value in curve.time],
                [float(value) for value in curve.estimate],
                list(range(len(indices))),
                weights=group_weights,
                stype=computation.stype,
                ctype=computation.ctype,
            )
        elif group_response.type == "right" and group_response.start is None:
            group_weights = [weights[idx] for idx in indices]
            computation = _pseudo_counting_computation(curve)
            influence = survfitkm_influence(
                list(group_response.time),
                [int(value) for value in group_response.event],
                list(range(len(indices))),
                weights=group_weights,
                stype=computation.stype,
                ctype=computation.ctype,
            )
        else:
            raise NotImplementedError(
                "residuals.survfit currently supports right-censored or counting Kaplan-Meier fits"
            )
        rows = _survfit_residual_rows_at_times(influence, times, residual_type)
        for local_idx, source_idx in enumerate(indices):
            matrix[source_idx] = rows[local_idx]
            if curve_numbers is not None:
                curve_numbers[source_idx] = curve_idx
    return matrix, curve_numbers


def _collapse_survfit_residual_matrix(
    matrix: list[list[float]],
    ids: list[Any],
    weights: list[float],
    curve_numbers: list[int] | None,
) -> tuple[list[list[float]], list[Any], list[int] | None]:
    levels = _label_levels(ids, "id")
    index_by_id = {value: idx for idx, value in enumerate(levels)}
    collapsed = [[0.0 for _ in matrix[0]] for _ in levels] if matrix else [[] for _ in levels]
    collapsed_curve = [0 for _ in levels] if curve_numbers is not None else None
    for row_idx, id_value in enumerate(ids):
        target = index_by_id[id_value]
        collapsed[target] = [
            current + float(weights[row_idx]) * float(value)
            for current, value in zip(collapsed[target], matrix[row_idx], strict=True)
        ]
        if collapsed_curve is not None and collapsed_curve[target] == 0:
            collapsed_curve[target] = curve_numbers[row_idx]
    return collapsed, list(levels), collapsed_curve


def _weight_survfit_residual_matrix(
    matrix: list[list[float]],
    weights: list[float],
) -> list[list[float]]:
    return [
        [float(weight) * float(value) for value in row]
        for row, weight in zip(matrix, weights, strict=True)
    ]


def survfit_residuals(
    fit: Any,
    times: Any | None = None,
    *,
    type: str = "pstate",
    collapse: Any = False,
    weighted: Any = False,
    data_frame: Any = False,
    extra: Any = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return R-style ``residuals.survfit`` influence residuals for KM curves."""

    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit_residuals got unexpected keyword argument(s): {unexpected}")
    if isinstance(fit, CoxSurvfitResult):
        raise TypeError("residuals method for coxph survival curve not found")
    if isinstance(fit, TurnbullSurvfitResult):
        raise NotImplementedError("residuals for interval-censored data are not available")

    eval_times = _survfit_residual_times(times)
    residual_type = _survfit_residual_type(type)
    collapse_value = _normalize_bool_option(collapse, "collapse")
    weighted_value = _normalize_bool_option(weighted, "weighted")
    data_frame_value = _normalize_bool_option(data_frame, "data_frame")
    extra_value = _normalize_bool_option(extra, "extra")
    if collapse_value and not weighted_value:
        raise ValueError("invalid combination of options: collapse=True and weighted=False")

    frame = _survfit_residual_model_frame(fit)
    response = _survfit_residual_response(frame)
    n = len(response)
    if n == 0:
        raise ValueError("data set has no non-missing observations")
    weights_values, has_case_weights = _survfit_residual_weights(frame, n)
    id_values, id_name, has_id = _survfit_residual_ids(frame, n)
    if not has_case_weights:
        weighted_value = False
    if not has_id or len(_label_levels(id_values, "id")) == n:
        collapse_value = False

    curve_specs = _survfit_residual_curve_specs(fit, frame, n)
    if response.type in {"mright", "mcounting"}:
        return _survfit_multistate_residual_result(
            frame,
            response,
            weights_values,
            id_values,
            id_name,
            curve_specs,
            eval_times,
            residual_type,
            collapse_value,
            weighted_value,
            data_frame_value,
            extra_value,
        )

    matrix, curve_numbers = _survfit_residual_matrix(
        response,
        weights_values,
        curve_specs,
        eval_times,
        residual_type,
    )

    if collapse_value:
        matrix, id_values, curve_numbers = _collapse_survfit_residual_matrix(
            matrix,
            id_values,
            weights_values,
            curve_numbers,
        )
    elif weighted_value and any(weight != 1.0 for weight in weights_values):
        matrix = _weight_survfit_residual_matrix(matrix, weights_values)

    return {
        "resid": matrix,
        "id": id_values,
        "id_name": id_name,
        "time": eval_times,
        "curve": curve_numbers,
        "data_frame": data_frame_value,
        "extra": extra_value,
    }
