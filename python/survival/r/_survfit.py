"""``survfit`` (KM/AJ/Turnbull/Cox curves), ``survfit0``, aggregation, confint, influence."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _SURVFIT_TIME_EPSILON,
    _encode_labels,
    _encode_labels_with_levels,
    _float_vector,
    _group_indices,
    _integer_code_vector,
    _is_bool_like,
    _is_missing_value,
    _label_levels,
    _materialize_1d,
    _materialize_labels,
    _mstate_event_label,
    _mstate_levels,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
    _normalize_optional_bool_option,
    _normalize_start_time,
    _normalize_survfit_conf_level,
    _normalize_survfit_conf_type,
    _normalize_survfit_style,
    _normalize_survfit_type,
    _pop_dotted_keyword,
    _r_formula_ordered_levels,
    _r_numeric_vector,
    _subset_indices,
    _subset_optional_sequence,
    _survdiff_timefix_values,
    _timefix_vectors,
)
from ._coxph import _cox_survfit_result
from ._fit import _is_clogit_fit, _prediction_inputs
from ._formula import (
    _apply_formula_na_action,
    _column,
    _combined_columns,
    _combined_formula_groups,
    _cox_survfit_model_frame,
    _parse_formula,
    _subset_formula_inputs,
    _survfit_formula_model_frame,
    _survfit_model_frame,
)
from ._surv import (
    Surv,
    _apply_surv_na_action,
    _subset_surv,
    _survfit_response_with_etype,
    _turnbull_intervals,
)
from ._types import (
    CoxSurvfitResult,
    SurvfitConfidenceIntervalResult,
    SurvfitMultiStateResult,
    SurvfitResult,
    TurnbullSurvfitResult,
    _SurvfitComputation,
)


def survfit_confint(
    p: Any,
    se: Any,
    logse: Any = True,
    conf_type: str | None = None,
    conf_int: Any = 0.95,
    selow: Any | None = None,
    ulimit: Any = True,
    **kwargs: Any,
) -> SurvfitConfidenceIntervalResult:
    """Return R ``survival::survfit_confint`` confidence bounds."""

    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, None)
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected survfit_confint argument(s): {unexpected}")
    if conf_type is None:
        raise TypeError("conf_type is required")
    if not isinstance(conf_type, str):
        raise TypeError("conf_type must be a string")
    if conf_type not in {"plain", "log", "log-log", "logit", "arcsin"}:
        raise ValueError("invalid conf.int type")
    if not _is_bool_like(logse):
        raise TypeError("logse must be True or False")
    if not _is_bool_like(ulimit):
        raise TypeError("ulimit must be True or False")

    p_values = _r_numeric_vector(p, "p")
    se_values = _r_numeric_vector(se, "se")
    confidence = _normalize_conf_level(conf_int, "conf_int")
    zval = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    selow_values = None if selow is None else _r_numeric_vector(selow, "selow")
    lower, upper = _core.survfit_confint_native(
        p_values,
        se_values,
        bool(logse),
        conf_type,
        zval,
        selow_values,
        bool(ulimit),
    )
    return SurvfitConfidenceIntervalResult(lower=lower, upper=upper)


def _survfitkm(
    time: list[float],
    status: list[int],
    *,
    weights: list[float] | None,
    entry_times: list[float] | None,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    timefix: bool,
) -> Any:
    return _core.survfitkm(
        time,
        status,
        weights=weights,
        entry_times=entry_times,
        reverse=reverse,
        computation_type=0,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
    )


def survfitkm_influence(
    time: Any,
    status: Any,
    cluster: Any | None = None,
    *,
    weights: Any | None = None,
    reverse: Any = False,
    stype: Any = 1,
    ctype: Any = 1,
    conf_level: Any = 0.95,
    conf_type: Any = "log",
    timefix: Any = True,
) -> Any:
    """Return right-censored ``survfitKM`` influence matrices."""

    time_values = _float_vector(time, "time")
    status_values = _float_vector(status, "status")
    if len(status_values) != len(time_values):
        raise ValueError("status must have the same length as time")
    if cluster is None:
        cluster_values = list(range(len(time_values)))
    else:
        labels = _materialize_labels(cluster, "cluster")
        if len(labels) != len(time_values):
            raise ValueError("cluster must have the same length as time")
        cluster_values = _encode_labels(labels, "cluster")
    weight_values = None if weights is None else _float_vector(weights, "weights")
    if weight_values is not None and len(weight_values) != len(time_values):
        raise ValueError("weights must have the same length as time")
    return _core.survfitkm_influence(
        time_values,
        status_values,
        cluster_values,
        weights=weight_values,
        reverse=_normalize_bool_option(reverse, "reverse"),
        stype=_normalize_survfit_style(stype, "stype"),
        ctype=_normalize_survfit_style(ctype, "ctype"),
        conf_level=_normalize_conf_level(conf_level),
        conf_type=_normalize_survfit_conf_type(conf_type),
        timefix=_normalize_bool_option(timefix, "timefix"),
    )


def survfitkm_counting_influence(
    start: Any,
    stop: Any,
    status: Any,
    curve_time: Any,
    curve_estimate: Any,
    cluster: Any | None = None,
    *,
    weights: Any | None = None,
    reverse: Any = False,
    stype: Any = 1,
    ctype: Any = 1,
    conf_level: Any = 0.95,
    conf_type: Any = "log",
    timefix: Any = True,
) -> Any:
    """Return counting-process ``survfitKM`` influence matrices."""

    start_values = _float_vector(start, "start")
    stop_values = _float_vector(stop, "stop")
    status_values = _integer_code_vector(status, "status", "status")
    if len(stop_values) != len(start_values):
        raise ValueError("stop must have the same length as start")
    if len(status_values) != len(start_values):
        raise ValueError("status must have the same length as start")
    curve_time_values = _float_vector(curve_time, "curve_time")
    curve_estimate_values = _float_vector(curve_estimate, "curve_estimate")
    if len(curve_estimate_values) != len(curve_time_values):
        raise ValueError("curve_estimate must have the same length as curve_time")
    if cluster is None:
        cluster_values = list(range(len(start_values)))
    else:
        labels = _materialize_labels(cluster, "cluster")
        if len(labels) != len(start_values):
            raise ValueError("cluster must have the same length as start")
        cluster_values = _encode_labels(labels, "cluster")
    weight_values = None if weights is None else _float_vector(weights, "weights")
    if weight_values is not None and len(weight_values) != len(start_values):
        raise ValueError("weights must have the same length as start")
    return _core.survfitkm_counting_influence(
        start_values,
        stop_values,
        status_values,
        curve_time_values,
        curve_estimate_values,
        cluster_values,
        weights=weight_values,
        reverse=_normalize_bool_option(reverse, "reverse"),
        stype=_normalize_survfit_style(stype, "stype"),
        ctype=_normalize_survfit_style(ctype, "ctype"),
        conf_level=_normalize_conf_level(conf_level),
        conf_type=_normalize_survfit_conf_type(conf_type),
        timefix=_normalize_bool_option(timefix, "timefix"),
    )


def _survfit_from_km_counts(
    km: Any,
    conf_level: float,
    computation: _SurvfitComputation,
    conf_type: str,
) -> SurvfitResult:
    curve = _core.survfit_curve_from_tables(
        [float(value) for value in km.time],
        [float(value) for value in km.n_risk],
        [float(value) for value in km.n_event],
        [float(value) for value in km.n_event_count],
        [float(value) for value in km.n_censor],
        [float(value) for value in km.n_censor_count],
        None if getattr(km, "n_enter", None) is None else [float(value) for value in km.n_enter],
        False,
        computation.stype,
        computation.ctype,
        conf_level,
        conf_type,
    )

    return SurvfitResult(
        time=[float(value) for value in curve.time],
        n_risk=[float(value) for value in curve.n_risk],
        n_event=[float(value) for value in curve.n_event],
        n_censor=[float(value) for value in curve.n_censor],
        estimate=[float(value) for value in curve.estimate],
        std_err=[float(value) for value in curve.std_err],
        conf_lower=[float(value) for value in curve.conf_lower],
        conf_upper=[float(value) for value in curve.conf_upper],
        cumhaz=[float(value) for value in curve.cumhaz],
        std_chaz=[float(value) for value in curve.std_chaz],
        n_enter=(
            [float(value) for value in curve.n_enter]
            if getattr(curve, "n_enter", None) is not None
            else None
        ),
        n_risk_count=(
            [float(value) for value in km.n_risk_count]
            if getattr(km, "n_risk_count", None) is not None
            else None
        ),
        n_event_count=(
            [float(value) for value in km.n_event_count]
            if getattr(km, "n_event_count", None) is not None
            else None
        ),
        n_censor_count=(
            [float(value) for value in km.n_censor_count]
            if getattr(km, "n_censor_count", None) is not None
            else None
        ),
        n_enter_count=(
            [float(value) for value in km.n_enter_count]
            if getattr(km, "n_enter_count", None) is not None
            else None
        ),
        model=getattr(km, "model", None),
    )


def _survfit_from_count_tables(
    times: list[float],
    n_risk: list[float],
    n_event: list[float],
    n_event_count: list[float],
    n_censor: list[float],
    n_censor_count: list[float],
    n_enter: list[float] | None,
    n_risk_count: list[float] | None = None,
    n_enter_count: list[float] | None = None,
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
) -> SurvfitResult:
    curve = _core.survfit_curve_from_tables(
        times,
        n_risk,
        n_event,
        n_event_count,
        n_censor,
        n_censor_count,
        n_enter,
        reverse,
        computation.stype,
        computation.ctype,
        conf_level,
        conf_type,
    )

    return SurvfitResult(
        time=[float(value) for value in curve.time],
        n_risk=[float(value) for value in curve.n_risk],
        n_event=[float(value) for value in curve.n_event],
        n_censor=[float(value) for value in curve.n_censor],
        estimate=[float(value) for value in curve.estimate],
        std_err=[float(value) for value in curve.std_err],
        conf_lower=[float(value) for value in curve.conf_lower],
        conf_upper=[float(value) for value in curve.conf_upper],
        cumhaz=[float(value) for value in curve.cumhaz],
        std_chaz=[float(value) for value in curve.std_chaz],
        n_enter=(
            [float(value) for value in curve.n_enter]
            if getattr(curve, "n_enter", None) is not None
            else None
        ),
        n_risk_count=(None if n_risk_count is None else [float(value) for value in n_risk_count]),
        n_event_count=[float(value) for value in n_event_count],
        n_censor_count=[float(value) for value in n_censor_count],
        n_enter_count=(
            None if n_enter_count is None else [float(value) for value in n_enter_count]
        ),
    )


def _survfit_cluster_values(cluster: Any, n: int) -> list[Any]:
    values = _materialize_labels(cluster, "cluster")
    if len(values) != n:
        raise ValueError("cluster must have the same length as the Surv response")
    _label_levels(values, "cluster")
    return values


def _survfit_robust_cluster_values(
    response: Surv,
    cluster: Any | None,
    id_values: list[Any] | None,
    weights: list[float] | None,
    robust: bool | None,
) -> list[Any] | None:
    if robust is False:
        if cluster is not None:
            warnings.warn(
                "cluster specified with robust=False; cluster will be ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        return None
    if cluster is not None:
        return _survfit_cluster_values(cluster, len(response))
    if robust is not True:
        if weights is not None and any(not float(weight).is_integer() for weight in weights):
            return list(range(len(response)))
        return None
    if id_values is not None:
        return _survfit_cluster_values(id_values, len(response))
    if response.start is not None:
        raise NotImplementedError(
            "survfit robust variance for counting-process data requires cluster or id"
        )
    return list(range(len(response)))


def _survfit_robust_km_result(
    result: Any,
    response: Surv,
    weights: list[float] | None,
    cluster_values: list[Any],
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    timefix: bool,
) -> SurvfitResult:
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survfit robust variance is currently supported only for right-censored or "
            "counting-process Kaplan-Meier curves"
        )
    if len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")

    if response.start is not None:
        std_err, std_chaz, conf_lower, conf_upper = _core.robust_counting_survfit_variance(
            list(response.start),
            list(response.time),
            [int(value) for value in response.event],
            [float(value) for value in result.time],
            [float(value) for value in result.estimate],
            _encode_labels(cluster_values, "cluster"),
            weights=weights,
            reverse=reverse,
            conf_level=conf_level,
            conf_type=conf_type,
            timefix=timefix,
        )
        return SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[float(value) for value in std_err],
            conf_lower=[float(value) for value in conf_lower],
            conf_upper=[float(value) for value in conf_upper],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[float(value) for value in std_chaz],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
            model=getattr(result, "model", None),
        )

    robust = _core.robust_survfitkm(
        list(response.time),
        list(response.event),
        _encode_labels(cluster_values, "cluster"),
        weights=weights,
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
    )
    return SurvfitResult(
        time=[float(value) for value in robust.time],
        n_risk=[float(value) for value in robust.n_risk],
        n_event=[float(value) for value in robust.n_event],
        n_censor=[float(value) for value in robust.n_censor],
        estimate=[float(value) for value in robust.estimate],
        std_err=[float(value) for value in robust.std_err],
        conf_lower=[float(value) for value in robust.conf_lower],
        conf_upper=[float(value) for value in robust.conf_upper],
        cumhaz=[float(value) for value in robust.cumhaz],
        std_chaz=[float(value) for value in robust.std_chaz],
        n_enter=None,
        n_risk_count=_optional_float_list(robust, "n_risk_count"),
        n_event_count=_optional_float_list(robust, "n_event_count"),
        n_censor_count=_optional_float_list(robust, "n_censor_count"),
        n_enter_count=_optional_float_list(robust, "n_enter_count"),
        model=getattr(result, "model", None),
    )


def _survfit_robust_right_result(
    result: SurvfitResult,
    response: Surv,
    weights: list[float] | None,
    cluster_values: list[Any],
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
    timefix: bool,
) -> SurvfitResult:
    if response.start is not None or response.type != "right":
        raise NotImplementedError(
            "survfit robust variance for non-Kaplan-Meier curves is currently supported only "
            "for right-censored data"
        )
    if len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")

    std_err, std_chaz, conf_lower, conf_upper = _core.robust_right_survfit_variance(
        list(response.time),
        list(response.event),
        [float(value) for value in result.time],
        [float(value) for value in result.estimate],
        _encode_labels(cluster_values, "cluster"),
        weights=weights,
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
        stype=computation.stype,
        ctype=computation.ctype,
    )

    return SurvfitResult(
        time=result.time,
        n_risk=result.n_risk,
        n_event=result.n_event,
        n_censor=result.n_censor,
        estimate=result.estimate,
        std_err=[float(value) for value in std_err],
        conf_lower=[float(value) for value in conf_lower],
        conf_upper=[float(value) for value in conf_upper],
        cumhaz=result.cumhaz,
        std_chaz=[float(value) for value in std_chaz],
        n_enter=result.n_enter,
        n_risk_count=result.n_risk_count,
        n_event_count=result.n_event_count,
        n_censor_count=result.n_censor_count,
        n_enter_count=result.n_enter_count,
        model=result.model,
    )


def _survfit_robust_counting_result(
    result: SurvfitResult,
    response: Surv,
    weights: list[float] | None,
    cluster_values: list[Any],
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
    timefix: bool,
) -> SurvfitResult:
    if response.start is None or response.type != "counting":
        raise NotImplementedError(
            "survfit robust variance for counting-process curves requires counting-process data"
        )
    if len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")

    std_err, std_chaz, conf_lower, conf_upper = _core.robust_counting_survfit_variance(
        list(response.start),
        list(response.time),
        [int(value) for value in response.event],
        [float(value) for value in result.time],
        [float(value) for value in result.estimate],
        _encode_labels(cluster_values, "cluster"),
        weights=weights,
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
        stype=computation.stype,
        ctype=computation.ctype,
    )

    return SurvfitResult(
        time=result.time,
        n_risk=result.n_risk,
        n_event=result.n_event,
        n_censor=result.n_censor,
        estimate=result.estimate,
        std_err=[float(value) for value in std_err],
        conf_lower=[float(value) for value in conf_lower],
        conf_upper=[float(value) for value in conf_upper],
        cumhaz=result.cumhaz,
        std_chaz=[float(value) for value in std_chaz],
        n_enter=result.n_enter,
        n_risk_count=result.n_risk_count,
        n_event_count=result.n_event_count,
        n_censor_count=result.n_censor_count,
        n_enter_count=result.n_enter_count,
        model=result.model,
    )


def _survfit_counting_with_id(
    response: Surv,
    weights: list[float] | None,
    id_values: list[Any],
    *,
    include_entry: bool,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
    timefix: bool,
) -> SurvfitResult:
    if response.start is None:
        raise ValueError("survfit id-aware entry counts require counting-process Surv input")

    n = len(response)
    starts = [float(value) for value in response.start]
    stops = [float(value) for value in response.time]
    status = [int(value) for value in response.event]
    case_weights = [1.0] * n if weights is None else [float(value) for value in weights]
    id_codes = _encode_labels(id_values, "id")
    tables = _core.counting_survfit_tables(
        starts,
        stops,
        status,
        id_codes,
        case_weights,
        include_entry,
        timefix,
    )

    return _survfit_from_count_tables(
        [float(value) for value in tables.time],
        [float(value) for value in tables.n_risk],
        [float(value) for value in tables.n_event],
        [float(value) for value in tables.n_event_count],
        [float(value) for value in tables.n_censor],
        [float(value) for value in tables.n_censor_count],
        None if tables.n_enter is None else [float(value) for value in tables.n_enter],
        n_risk_count=[float(value) for value in tables.n_risk_count],
        n_enter_count=(
            [float(value) for value in tables.n_enter_count]
            if getattr(tables, "n_enter_count", None) is not None
            else None
        ),
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        computation=computation,
    )


def _survfit_start_time_indices(
    response: Surv,
    start_time: float,
    timefix: bool,
) -> list[int]:
    if timefix:
        indices = [
            idx
            for idx, stop_time in enumerate(response.time)
            if stop_time >= start_time - _SURVFIT_TIME_EPSILON
        ]
    else:
        indices = [idx for idx, stop_time in enumerate(response.time) if stop_time >= start_time]
    if not indices:
        raise ValueError("all observations removed by start_time")
    return indices


def _survfit_default_time0(response: Surv) -> float:
    values = [0.0, *response.time]
    if response.start is not None:
        values.extend(response.start)
    return float(min(values))


def _initial_survfit_risk(
    response: Surv,
    weights: list[float] | None,
    t0: float,
    timefix: bool,
) -> float:
    case_weights = [1.0] * len(response) if weights is None else weights
    if response.start is None:
        return float(sum(case_weights))
    if not timefix:
        return float(
            sum(
                weight
                for start, stop, weight in zip(
                    response.start,
                    response.time,
                    case_weights,
                    strict=True,
                )
                if start <= t0 <= stop
            )
        )
    return float(
        sum(
            weight
            for start, stop, weight in zip(response.start, response.time, case_weights, strict=True)
            if start <= t0 + _SURVFIT_TIME_EPSILON and stop >= t0 - _SURVFIT_TIME_EPSILON
        )
    )


def _cumhaz_from_survfit_counts(n_risk: list[float], n_event: list[float]) -> list[float]:
    hazard = 0.0
    cumhaz = []
    for risk, events in zip(n_risk, n_event, strict=True):
        if risk > 0.0:
            hazard += events / risk
        cumhaz.append(hazard)
    return cumhaz


def _std_chaz_from_survfit_counts(n_risk: list[float], n_event: list[float]) -> list[float]:
    variance = 0.0
    std_chaz = []
    for risk, events in zip(n_risk, n_event, strict=True):
        if risk > 0.0:
            variance += events / (risk * risk)
        std_chaz.append(math.sqrt(max(variance, 0.0)))
    return std_chaz


def _survfit_with_time0(
    result: Any,
    t0: float,
    conf_type: str,
    initial_n_risk: float,
    timefix: bool,
) -> Any:
    times = [float(value) for value in result.time]
    if times and (abs(times[0] - t0) < _SURVFIT_TIME_EPSILON if timefix else times[0] == t0):
        return result

    n_risk = [float(value) for value in result.n_risk]
    n_event = [float(value) for value in result.n_event]
    n_censor = [float(value) for value in result.n_censor]
    estimate = [float(value) for value in result.estimate]
    std_err = [float(value) for value in result.std_err]
    conf_lower = [float(value) for value in result.conf_lower]
    conf_upper = [float(value) for value in result.conf_upper]
    cumhaz = (
        [float(value) for value in result.cumhaz]
        if hasattr(result, "cumhaz")
        else _cumhaz_from_survfit_counts(n_risk, n_event)
    )
    std_chaz = (
        [float(value) for value in result.std_chaz]
        if hasattr(result, "std_chaz")
        else _std_chaz_from_survfit_counts(n_risk, n_event)
    )
    n_risk0 = n_risk[0] if n_risk else initial_n_risk

    return SurvfitResult(
        time=[t0, *times],
        n_risk=[n_risk0, *n_risk],
        n_event=[0.0, *n_event],
        n_censor=[0.0, *n_censor],
        estimate=[1.0, *estimate],
        std_err=[0.0, *std_err],
        conf_lower=([1.0, *conf_lower] if conf_type != "none" else []),
        conf_upper=([1.0, *conf_upper] if conf_type != "none" else []),
        cumhaz=[0.0, *cumhaz],
        std_chaz=[0.0, *std_chaz],
        n_enter=([0.0, *result.n_enter] if getattr(result, "n_enter", None) is not None else None),
        n_risk_count=(
            [result.n_risk_count[0] if result.n_risk_count else 0.0, *result.n_risk_count]
            if getattr(result, "n_risk_count", None) is not None
            else None
        ),
        n_event_count=(
            [0.0, *result.n_event_count]
            if getattr(result, "n_event_count", None) is not None
            else None
        ),
        n_censor_count=(
            [0.0, *result.n_censor_count]
            if getattr(result, "n_censor_count", None) is not None
            else None
        ),
        n_enter_count=(
            [0.0, *result.n_enter_count]
            if getattr(result, "n_enter_count", None) is not None
            else None
        ),
    )


def _needs_time0_insert(times: Sequence[float], t0: float) -> bool:
    return not times or abs(float(times[0]) - t0) >= _SURVFIT_TIME_EPSILON


def _prepend_curve_time0(values: list[float], initial: float) -> list[float]:
    return [initial, *[float(value) for value in values]]


def _prepend_curve_time0_optional(values: list[float], initial: float) -> list[float]:
    return _prepend_curve_time0(values, initial) if values else []


def _optional_float_list(value: Any, name: str) -> list[float] | None:
    items = getattr(value, name, None)
    if items is None:
        return None
    return [float(item) for item in items]


def _prepend_matrix_time0(values: list[list[float]], initial: float) -> list[list[float]]:
    return [[initial, *[float(value) for value in row]] for row in values] if values else []


def _survfit0_default_time(result: Any) -> float:
    if isinstance(result, CoxSurvfitResult) and result.start_time is not None:
        return float(result.start_time)
    times = getattr(result, "time", None)
    if times is None:
        times = getattr(result, "time_points", None)
    values = [0.0]
    if times is not None:
        values.extend(float(value) for value in times)
    return min(values)


def _survfit0_result(result: SurvfitResult, t0: float | None = None) -> SurvfitResult:
    initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time, initial_time):
        return result
    n_risk0 = float(result.n_risk[0]) if result.n_risk else 0.0
    return SurvfitResult(
        time=_prepend_curve_time0(result.time, initial_time),
        n_risk=_prepend_curve_time0(result.n_risk, n_risk0),
        n_event=_prepend_curve_time0(result.n_event, 0.0),
        n_censor=_prepend_curve_time0(result.n_censor, 0.0),
        estimate=_prepend_curve_time0(result.estimate, 1.0),
        std_err=_prepend_curve_time0_optional(result.std_err, 0.0),
        conf_lower=_prepend_curve_time0_optional(result.conf_lower, 1.0),
        conf_upper=_prepend_curve_time0_optional(result.conf_upper, 1.0),
        cumhaz=_prepend_curve_time0(result.cumhaz, 0.0),
        std_chaz=_prepend_curve_time0_optional(result.std_chaz, 0.0),
        n_enter=(_prepend_curve_time0(result.n_enter, 0.0) if result.n_enter is not None else None),
        n_risk_count=(
            _prepend_curve_time0(
                result.n_risk_count,
                result.n_risk_count[0] if result.n_risk_count else 0.0,
            )
            if result.n_risk_count is not None
            else None
        ),
        n_event_count=(
            _prepend_curve_time0(result.n_event_count, 0.0)
            if result.n_event_count is not None
            else None
        ),
        n_censor_count=(
            _prepend_curve_time0(result.n_censor_count, 0.0)
            if result.n_censor_count is not None
            else None
        ),
        n_enter_count=(
            _prepend_curve_time0(result.n_enter_count, 0.0)
            if result.n_enter_count is not None
            else None
        ),
        model=result.model,
    )


def _prepend_multistate_time0(
    values: list[list[float]],
    initial: Sequence[float],
) -> list[list[float]]:
    return [
        [float(value) for value in initial],
        *[[float(value) for value in row] for row in values],
    ]


def _prepend_multistate_time0_optional(
    values: list[list[float]] | None,
    initial: Sequence[float],
) -> list[list[float]] | None:
    return None if values is None else _prepend_multistate_time0(values, initial)


def _survfit0_multistate_result(
    result: SurvfitMultiStateResult,
    t0: float | None = None,
) -> SurvfitMultiStateResult:
    initial_time = float(result.t0) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time, initial_time):
        return result

    state_count = len(result.states)
    transition_count = len(result.transitions)
    zero_states = [0.0] * state_count
    zero_transitions = [0.0] * transition_count
    initial_risk = result.n_risk[0] if result.n_risk else zero_states
    initial_risk_count = (
        result.n_risk_count[0] if result.n_risk_count else [0.0] * len(initial_risk)
    )
    initial_standard_error = result.std_err0 if result.std_err0 is not None else zero_states

    return SurvfitMultiStateResult(
        time=[initial_time, *[float(value) for value in result.time]],
        n_risk=_prepend_multistate_time0(result.n_risk, initial_risk),
        n_event=_prepend_multistate_time0(result.n_event, zero_states),
        n_censor=_prepend_multistate_time0(result.n_censor, zero_states),
        pstate=_prepend_multistate_time0(result.pstate, result.p0),
        cumhaz=_prepend_multistate_time0(result.cumhaz, zero_transitions),
        states=result.states,
        transitions=result.transitions,
        p0=[float(value) for value in result.p0],
        t0=result.t0,
        n=result.n,
        n_id=result.n_id,
        std_err=_prepend_multistate_time0_optional(result.std_err, initial_standard_error),
        std_err0=result.std_err0,
        std_chaz=_prepend_multistate_time0_optional(result.std_chaz, zero_transitions),
        std_auc=_prepend_multistate_time0_optional(result.std_auc, zero_states),
        conf_lower=_prepend_multistate_time0_optional(result.conf_lower, zero_states),
        conf_upper=_prepend_multistate_time0_optional(result.conf_upper, zero_states),
        n_risk_count=_prepend_multistate_time0_optional(
            result.n_risk_count,
            initial_risk_count,
        ),
        n_event_count=_prepend_multistate_time0_optional(
            result.n_event_count,
            zero_states,
        ),
        n_censor_count=_prepend_multistate_time0_optional(
            result.n_censor_count,
            zero_states,
        ),
        n_enter=_prepend_multistate_time0_optional(result.n_enter, zero_states),
        n_enter_count=_prepend_multistate_time0_optional(
            result.n_enter_count,
            zero_states,
        ),
        n_transition=_prepend_multistate_time0(result.n_transition, zero_transitions),
        n_transition_count=_prepend_multistate_time0_optional(
            result.n_transition_count,
            zero_transitions,
        ),
        model=result.model,
        surv_type=result.surv_type,
        conf_type=result.conf_type,
        conf_level=result.conf_level,
        oldstate=result.oldstate,
        p0_fixed=result.p0_fixed,
        timefix=result.timefix,
        influence_state=result.influence_state,
        influence_state0=result.influence_state0,
        influence_chaz=result.influence_chaz,
        influence_auc=result.influence_auc,
    )


def _survfit0_cox_result(
    result: CoxSurvfitResult,
    t0: float | None = None,
) -> CoxSurvfitResult:
    initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time, initial_time):
        return result
    return CoxSurvfitResult(
        time=_prepend_curve_time0(result.time, initial_time),
        surv=_prepend_matrix_time0(result.surv, 1.0),
        cumhaz=_prepend_matrix_time0(result.cumhaz, 0.0),
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
        start_time=result.start_time,
        std_err=_prepend_matrix_time0(result.std_err, 0.0),
        std_chaz=_prepend_matrix_time0(result.std_chaz, 0.0),
        conf_lower=_prepend_matrix_time0(result.conf_lower, 1.0),
        conf_upper=_prepend_matrix_time0(result.conf_upper, 1.0),
        model=result.model,
    )


def _survfit0_turnbull_result(
    result: TurnbullSurvfitResult,
    t0: float | None = None,
) -> TurnbullSurvfitResult:
    initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time_points, initial_time):
        return result
    return TurnbullSurvfitResult(
        time_points=_prepend_curve_time0(result.time_points, initial_time),
        survival=_prepend_curve_time0(result.survival, 1.0),
        survival_lower=_prepend_curve_time0_optional(result.survival_lower, 1.0),
        survival_upper=_prepend_curve_time0_optional(result.survival_upper, 1.0),
        n_iter=result.n_iter,
        converged=result.converged,
        model=result.model,
    )


def _is_survfit_result_like(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    )


def _coerce_survfit_result_like(value: Any) -> SurvfitResult:
    if isinstance(value, SurvfitResult):
        return value
    return SurvfitResult(
        time=[float(item) for item in value.time],
        n_risk=[float(item) for item in value.n_risk],
        n_event=[float(item) for item in value.n_event],
        n_censor=[float(item) for item in value.n_censor],
        estimate=[float(item) for item in value.estimate],
        std_err=[float(item) for item in getattr(value, "std_err", [])],
        conf_lower=[float(item) for item in getattr(value, "conf_lower", [])],
        conf_upper=[float(item) for item in getattr(value, "conf_upper", [])],
        cumhaz=[float(item) for item in value.cumhaz],
        std_chaz=[float(item) for item in getattr(value, "std_chaz", [])],
        n_enter=(
            [float(item) for item in value.n_enter]
            if getattr(value, "n_enter", None) is not None
            else None
        ),
        n_risk_count=_optional_float_list(value, "n_risk_count"),
        n_event_count=_optional_float_list(value, "n_event_count"),
        n_censor_count=_optional_float_list(value, "n_censor_count"),
        n_enter_count=_optional_float_list(value, "n_enter_count"),
        model=getattr(value, "model", None),
    )


def _is_turnbull_result_like(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("time_points", "survival", "survival_lower", "survival_upper")
    )


def _coerce_turnbull_result_like(value: Any) -> TurnbullSurvfitResult:
    if isinstance(value, TurnbullSurvfitResult):
        return value
    return TurnbullSurvfitResult(
        time_points=[float(item) for item in value.time_points],
        survival=[float(item) for item in value.survival],
        survival_lower=[float(item) for item in value.survival_lower],
        survival_upper=[float(item) for item in value.survival_upper],
        n_iter=int(getattr(value, "n_iter", 0)),
        converged=bool(getattr(value, "converged", True)),
        model=getattr(value, "model", None),
    )


def _survfit0_any_result(value: Any, t0: float | None = None) -> Any:
    if isinstance(value, SurvfitMultiStateResult):
        return _survfit0_multistate_result(value, t0)
    if isinstance(value, SurvfitResult) or _is_survfit_result_like(value):
        result = _coerce_survfit_result_like(value)
        initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
        return (
            value
            if not _needs_time0_insert(result.time, initial_time)
            else _survfit0_result(
                result,
                initial_time,
            )
        )
    if isinstance(value, CoxSurvfitResult):
        return _survfit0_cox_result(value, t0)
    if isinstance(value, TurnbullSurvfitResult) or _is_turnbull_result_like(value):
        result = _coerce_turnbull_result_like(value)
        initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
        return (
            value
            if not _needs_time0_insert(
                result.time_points,
                initial_time,
            )
            else _survfit0_turnbull_result(result, initial_time)
        )
    return value


def _mapping_survfit0_time(results: Mapping[Any, Any]) -> float:
    if results and all(isinstance(result, SurvfitMultiStateResult) for result in results.values()):
        return min(float(result.t0) for result in results.values())
    values = [0.0]
    for result in results.values():
        times = getattr(result, "time", None)
        if times is None:
            times = getattr(result, "time_points", None)
        if times is not None:
            values.extend(float(value) for value in times)
    return min(values)


def survfit0(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Insert an initial survival row into an existing survfit result."""

    if args or kwargs:
        raise TypeError("survfit0 got unexpected arguments")
    if isinstance(x, Mapping):
        t0 = _mapping_survfit0_time(x)
        return {label: _survfit0_any_result(result, t0) for label, result in x.items()}
    result = _survfit0_any_result(x)
    if result is not x or isinstance(
        x,
        SurvfitResult | SurvfitMultiStateResult | CoxSurvfitResult | TurnbullSurvfitResult,
    ):
        return result
    if _is_survfit_result_like(x) or _is_turnbull_result_like(x):
        return result
    raise TypeError("survfit0 requires a survfit result")


def _cox_survfit_result_cumhaz(surv: list[float]) -> list[float]:
    return [math.inf if value <= 0.0 else -math.log(value) for value in surv]


def _cox_survfit_result_std_chaz(
    surv: list[float],
    std_err: list[float],
) -> list[float]:
    result: list[float] = []
    for survival, se in zip(surv, std_err, strict=True):
        if survival > 0.0:
            result.append(float(se) / float(survival))
        else:
            result.append(math.inf if se > 0.0 else 0.0)
    return result


def _weighted_average(values: Sequence[float], weights: Sequence[float] | None) -> float:
    if not values:
        return math.nan
    if weights is None:
        return sum(float(value) for value in values) / len(values)
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        return math.nan
    return (
        sum(float(value) * float(weight) for value, weight in zip(values, weights, strict=True))
        / total
    )


def _cox_survfit_from_aggregates(
    source: CoxSurvfitResult,
    aggregates: Sequence[Any],
    linear_predictors: Sequence[float],
) -> CoxSurvfitResult:
    surv = [[float(value) for value in aggregate.surv] for aggregate in aggregates]
    std_err = (
        [[float(value) for value in aggregate.std_err] for aggregate in aggregates]
        if source.std_err
        else []
    )
    std_chaz = (
        [
            _cox_survfit_result_std_chaz(surv_curve, se_curve)
            for surv_curve, se_curve in zip(surv, std_err, strict=True)
        ]
        if source.std_chaz and std_err
        else []
    )
    conf_lower = (
        [[float(value) for value in aggregate.lower] for aggregate in aggregates]
        if source.conf_lower and source.conf_upper and std_err
        else []
    )
    conf_upper = (
        [[float(value) for value in aggregate.upper] for aggregate in aggregates]
        if source.conf_lower and source.conf_upper and std_err
        else []
    )

    return CoxSurvfitResult(
        time=[float(value) for value in aggregates[0].time] if aggregates else [],
        surv=surv,
        cumhaz=[_cox_survfit_result_cumhaz(curve) for curve in surv],
        linear_predictors=[float(value) for value in linear_predictors],
        centered=source.centered,
        start_time=source.start_time,
        std_err=std_err,
        std_chaz=std_chaz,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        model=source.model,
    )


def aggregate_survfit_result(
    result: CoxSurvfitResult,
    groups: Any | None = None,
    weights: Any | None = None,
) -> CoxSurvfitResult:
    """Average Cox survfit prediction curves, optionally by group code."""

    if not isinstance(result, CoxSurvfitResult):
        raise TypeError("survfit object does not have a 'data' margin")

    n_curves = len(result.surv)
    if len(result.cumhaz) != n_curves or len(result.linear_predictors) != n_curves:
        raise ValueError("Cox survfit result has inconsistent curve counts")
    if n_curves == 0:
        return CoxSurvfitResult(
            time=[float(value) for value in result.time],
            surv=[],
            cumhaz=[],
            linear_predictors=[],
            centered=result.centered,
            start_time=result.start_time,
            model=result.model,
        )

    curve_time = [float(value) for value in result.time]
    curve_survs = [[float(value) for value in curve] for curve in result.surv]
    curve_std_errs = (
        [[float(value) for value in curve] for curve in result.std_err] if result.std_err else None
    )
    curve_weights = _float_vector(weights, "weights") if weights is not None else None

    if groups is None:
        aggregates = _core.aggregate_shared_survfit(
            curve_time,
            curve_survs,
            curve_std_errs,
            curve_weights,
            None,
        )
        linear_predictor = _weighted_average(result.linear_predictors, curve_weights)
        return _cox_survfit_from_aggregates(result, aggregates, [linear_predictor])

    group_codes = _integer_code_vector(groups, "groups", "integer group codes")
    if len(group_codes) != n_curves:
        raise ValueError("groups must have same length as number of curves")
    if any(code < 1 for code in group_codes):
        raise ValueError("groups must use positive integer group codes")

    aggregates = _core.aggregate_shared_survfit(
        curve_time,
        curve_survs,
        curve_std_errs,
        curve_weights,
        group_codes,
    )
    predictor_sums: dict[int, float] = {}
    predictor_weights: dict[int, float] = {}
    for idx, code in enumerate(group_codes):
        weight = 1.0 if curve_weights is None else curve_weights[idx]
        predictor_sums[code] = predictor_sums.get(code, 0.0) + (
            float(result.linear_predictors[idx]) * weight
        )
        predictor_weights[code] = predictor_weights.get(code, 0.0) + weight
    linear_predictors = [
        predictor_sums[code] / predictor_weights[code] for code in sorted(predictor_sums)
    ]

    return _cox_survfit_from_aggregates(result, aggregates, linear_predictors)


def _survfit_without_standard_errors(result: Any) -> Any:
    if isinstance(result, SurvfitResult):
        return SurvfitResult(
            time=result.time,
            n_risk=result.n_risk,
            n_event=result.n_event,
            n_censor=result.n_censor,
            estimate=result.estimate,
            std_err=[],
            conf_lower=[],
            conf_upper=[],
            cumhaz=result.cumhaz,
            std_chaz=[],
            n_enter=result.n_enter,
            n_risk_count=result.n_risk_count,
            n_event_count=result.n_event_count,
            n_censor_count=result.n_censor_count,
            n_enter_count=result.n_enter_count,
            model=result.model,
        )
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[],
            conf_lower=[],
            conf_upper=[],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
            model=getattr(result, "model", None),
        )
    if isinstance(result, CoxSurvfitResult):
        return CoxSurvfitResult(
            time=result.time,
            surv=result.surv,
            cumhaz=result.cumhaz,
            linear_predictors=result.linear_predictors,
            centered=result.centered,
            strata=result.strata,
            start_time=result.start_time,
            std_err=[],
            std_chaz=[],
            conf_lower=[],
            conf_upper=[],
            model=result.model,
        )
    if isinstance(result, Mapping):
        return {label: _survfit_without_standard_errors(curve) for label, curve in result.items()}
    return result


def _survfit_with_model_frame(result: Any, model_frame: dict[str, Any]) -> Any:
    if isinstance(result, SurvfitResult):
        return SurvfitResult(
            time=result.time,
            n_risk=result.n_risk,
            n_event=result.n_event,
            n_censor=result.n_censor,
            estimate=result.estimate,
            std_err=result.std_err,
            conf_lower=result.conf_lower,
            conf_upper=result.conf_upper,
            cumhaz=result.cumhaz,
            std_chaz=result.std_chaz,
            n_enter=result.n_enter,
            n_risk_count=result.n_risk_count,
            n_event_count=result.n_event_count,
            n_censor_count=result.n_censor_count,
            n_enter_count=result.n_enter_count,
            model=model_frame,
        )
    if all(
        hasattr(result, name)
        for name in (
            "time_points",
            "survival",
            "survival_lower",
            "survival_upper",
            "n_iter",
            "converged",
        )
    ):
        return TurnbullSurvfitResult(
            time_points=[float(value) for value in result.time_points],
            survival=[float(value) for value in result.survival],
            survival_lower=[float(value) for value in result.survival_lower],
            survival_upper=[float(value) for value in result.survival_upper],
            n_iter=int(result.n_iter),
            converged=bool(result.converged),
            model=model_frame,
        )
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return SurvfitResult(
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
            model=model_frame,
        )
    if isinstance(result, CoxSurvfitResult):
        return CoxSurvfitResult(
            time=result.time,
            surv=result.surv,
            cumhaz=result.cumhaz,
            linear_predictors=result.linear_predictors,
            centered=result.centered,
            strata=result.strata,
            start_time=result.start_time,
            std_err=result.std_err,
            std_chaz=result.std_chaz,
            conf_lower=result.conf_lower,
            conf_upper=result.conf_upper,
            model=model_frame,
        )
    if isinstance(result, Mapping):
        return {
            label: _survfit_with_model_frame(curve, model_frame) for label, curve in result.items()
        }
    return result


def _survfit_multistate_matrix(values: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in values]


def _survfit_multistate_confidence(
    pstate: list[list[float]],
    std_err: list[list[float]] | None,
    conf_level: float,
    conf_type: str,
) -> tuple[list[list[float]] | None, list[list[float]] | None]:
    if std_err is None or conf_type == "none":
        return None, None
    if not pstate:
        return [], []
    width = len(pstate[0])
    intervals = survfit_confint(
        [value for row in pstate for value in row],
        [value for row in std_err for value in row],
        logse=False,
        conf_type=conf_type,
        conf_int=conf_level,
    )
    lower = [
        intervals.lower[offset : offset + width] for offset in range(0, len(intervals.lower), width)
    ]
    upper = [
        intervals.upper[offset : offset + width] for offset in range(0, len(intervals.upper), width)
    ]
    return lower, upper


def _survfit_multistate_cluster_codes(
    n: int,
    cluster: Any | None,
    id_values: list[Any] | None,
) -> tuple[list[int], int]:
    if cluster is not None:
        labels = _survfit_cluster_values(cluster, n)
    elif id_values is not None:
        labels = id_values
    else:
        labels = list(range(n))
    codes = _encode_labels(labels, "cluster")
    return codes, len(set(codes))


def _survfit_multistate_state_data(
    response: Surv,
    id_values: list[Any] | None,
    istate: Any | None,
    timefix: bool,
) -> tuple[Surv, list[int], tuple[str, ...], list[Any]]:
    n = len(response)
    ids = list(range(n)) if id_values is None else id_values
    id_codes = _encode_labels(ids, "id")
    if istate is None:
        initial_labels = ["(s0)"]
        provided_states: list[int] | None = None
    else:
        raw_initial, initial_levels = _mstate_levels(istate, "istate")
        if len(raw_initial) != n:
            raise ValueError("istate must have the same length as the Surv response")
        observed_initial = {
            _mstate_event_label(value) for value in raw_initial if not _is_missing_value(value)
        }
        initial_labels = [level for level in initial_levels if level in observed_initial]
        provided_states = []

    states = tuple(
        [state for state in initial_labels if state not in response.states] + list(response.states)
    )
    state_index = {state: idx for idx, state in enumerate(states)}
    if istate is not None:
        provided_states = [
            state_index[_mstate_event_label(value)] for value in _materialize_1d(istate, "istate")
        ]

    mapped_events = [
        None if event is None else 0 if event == 0 else state_index[response.states[event - 1]] + 1
        for event in response.event
    ]
    if response.start is None:
        stop = _survdiff_timefix_values(list(response.time), timefix)
        start = None
    else:
        raw_start = list(response.start)
        raw_stop = list(response.time)
        start, stop = _timefix_vectors(raw_start, raw_stop) if timefix else (raw_start, raw_stop)
        if any(left >= right for left, right in zip(start, stop, strict=True)):
            raise ValueError("timefix produced an empty multi-state interval")

    current_states = [0] * n
    if response.start is None:
        seen_ids: set[int] = set()
        for idx in sorted(range(n), key=lambda row: (id_codes[row], stop[row], row)):
            subject = id_codes[idx]
            if subject in seen_ids:
                raise ValueError("a subject has overlapping right-censored multi-state rows")
            seen_ids.add(subject)
            current_states[idx] = 0 if provided_states is None else provided_states[idx]
    else:
        previous_by_id: dict[int, int] = {}
        for idx in sorted(
            range(n),
            key=lambda row: (id_codes[row], stop[row], start[row], row),
        ):
            subject = id_codes[idx]
            previous = previous_by_id.get(subject)
            if previous is None:
                current = 0 if provided_states is None else provided_states[idx]
            else:
                tolerance = _SURVFIT_TIME_EPSILON if timefix else 0.0
                if start[idx] < stop[previous] - tolerance:
                    raise ValueError("a subject has overlapping time intervals")
                if start[idx] > stop[previous] + tolerance:
                    raise ValueError("a subject has a gap between time intervals")
                prior_event = mapped_events[previous]
                expected = (
                    current_states[previous] if prior_event in {None, 0} else int(prior_event) - 1
                )
                if provided_states is not None and provided_states[idx] != expected:
                    raise ValueError("istate is inconsistent with the subject transition history")
                current = expected
            current_states[idx] = current
            previous_by_id[subject] = idx

    normalized = Surv._from_normalized(
        time=stop,
        event=mapped_events,
        start=start,
        time2=None,
        surv_type=response.type,
        states=states,
    )
    return normalized, current_states, states, ids


def _survfit_counting_positions(
    start: Sequence[float],
    stop: Sequence[float],
    ids: Sequence[Any],
    timefix: bool,
) -> list[int]:
    id_codes = _encode_labels(list(ids), "id")
    order = sorted(range(len(stop)), key=lambda idx: (id_codes[idx], stop[idx], idx))
    positions = [0] * len(stop)
    tolerance = _SURVFIT_TIME_EPSILON if timefix else 0.0
    for order_idx, row_idx in enumerate(order):
        previous = order[order_idx - 1] if order_idx > 0 else None
        following = order[order_idx + 1] if order_idx + 1 < len(order) else None
        first = (
            previous is None
            or id_codes[previous] != id_codes[row_idx]
            or stop[previous] < start[row_idx] - tolerance
        )
        last = (
            following is None
            or id_codes[following] != id_codes[row_idx]
            or stop[row_idx] < start[following] - tolerance
        )
        positions[row_idx] = int(first) + 2 * int(last)
    return positions


def _survfit_multistate_initial_distribution(
    current_states: Sequence[int],
    initial_rows: Sequence[bool],
    weights: Sequence[float],
    cluster_codes: Sequence[int],
    cluster_count: int,
    state_count: int,
    p0_override: list[float] | None,
    include_se: bool,
    influence_weights: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    if p0_override is not None:
        p0 = list(p0_override)
        return p0, [0.0] * (cluster_count * state_count) if include_se else []

    total_weight = sum(
        weight for weight, at_risk in zip(weights, initial_rows, strict=True) if at_risk
    )
    if total_weight <= 0.0:
        raise ValueError("positive total weight is required at the multi-state start time")
    p0 = [0.0] * state_count
    for state, weight, at_risk in zip(current_states, weights, initial_rows, strict=True):
        if at_risk:
            p0[state] += weight / total_weight
    if not include_se or any(value == 1.0 for value in p0):
        return p0, [0.0] * (cluster_count * state_count) if include_se else []

    influence_case_weights = weights if influence_weights is None else influence_weights
    influence = [[0.0] * state_count for _ in range(cluster_count)]
    for state, influence_weight, at_risk, cluster_code in zip(
        current_states,
        influence_case_weights,
        initial_rows,
        cluster_codes,
        strict=True,
    ):
        if not at_risk:
            continue
        for target in range(state_count):
            influence[cluster_code][target] += (
                influence_weight * (float(state == target) - p0[target]) / total_weight
            )
    return p0, [
        influence[cluster_code][state]
        for state in range(state_count)
        for cluster_code in range(cluster_count)
    ]


def _survfit_multistate_p0(value: Any | None, state_count: int) -> list[float] | None:
    if value is None:
        return None
    try:
        raw_probabilities = _materialize_1d(value, "p0")
        if not raw_probabilities:
            return None
        if any(isinstance(probability, bool) for probability in raw_probabilities):
            raise TypeError
        probabilities = [float(probability) for probability in raw_probabilities]
    except (TypeError, ValueError) as exc:
        raise TypeError("p0 must be a numeric vector") from exc
    if len(probabilities) != state_count:
        raise ValueError("p0 must have one probability per multi-state outcome")
    if any(not math.isfinite(probability) for probability in probabilities):
        raise ValueError("p0 must contain only finite probabilities")
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("p0 probabilities must be non-negative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("p0 probabilities must sum to 1")
    return probabilities


def _survfit_multistate_curve(
    response: Surv,
    weights: list[float] | None,
    id_values: list[Any] | None,
    cluster: Any | None,
    current_states: list[int],
    positions: list[int],
    states: tuple[str, ...],
    transitions: tuple[tuple[int, int], ...],
    *,
    t0: float,
    output_times: list[float],
    initial_rows: list[bool],
    p0_override: list[float] | None,
    report_initial_error: bool,
    include_se: bool,
    include_entry: bool,
    conf_level: float,
    conf_type: str,
    model_frame: dict[str, Any] | None,
    timefix: bool = True,
    save_influence: bool = False,
    influence_weights: list[float] | None = None,
) -> SurvfitMultiStateResult:
    n = len(response)
    stop = list(response.time)
    if response.start is None:
        initial_time = t0 if t0 < min(stop) else math.nextafter(min(stop), -math.inf)
        start = [initial_time] * n
    else:
        start = list(response.start)
    if any(event is None for event in response.event):
        raise ValueError("missing values in multi-state survfit inputs")
    case_weights = [1.0] * n if weights is None else list(weights)
    if any(weight < 0.0 or not math.isfinite(weight) for weight in case_weights):
        raise ValueError("weights must contain only non-negative finite values")
    influence_case_weights = case_weights if influence_weights is None else list(influence_weights)
    if len(influence_case_weights) != n:
        raise ValueError("influence_weights must have the same length as the Surv response")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in influence_case_weights):
        raise ValueError("influence_weights must contain only non-negative finite values")

    state_count = len(states)
    transition_count = len(transitions)
    hindx = [[transition_count] * state_count for _ in range(state_count)]
    for transition_idx, (source, target) in enumerate(transitions):
        hindx[source][target] = transition_idx

    cluster_codes, cluster_count = _survfit_multistate_cluster_codes(n, cluster, id_values)
    p0, initial_influence = _survfit_multistate_initial_distribution(
        current_states,
        initial_rows,
        case_weights,
        cluster_codes,
        cluster_count,
        state_count,
        p0_override,
        include_se,
        influence_case_weights,
    )
    std_err0 = (
        [
            math.sqrt(
                sum(
                    initial_influence[cluster_code + state * cluster_count] ** 2
                    for cluster_code in range(cluster_count)
                )
            )
            for state in range(state_count)
        ]
        if include_se and report_initial_error and all(value < 1.0 for value in p0)
        else None
    )
    y = [
        value
        for start, end, event in zip(
            start,
            stop,
            response.event,
            strict=True,
        )
        for value in (start, end, float(event))
    ]
    raw = _core.survfitaj(
        y=y,
        sort1=sorted(range(n), key=lambda idx: (start[idx], idx)),
        sort2=sorted(range(n), key=lambda idx: (stop[idx], idx)),
        utime=output_times,
        cstate=current_states,
        wt=case_weights,
        grp=cluster_codes,
        ngrp=cluster_count,
        p0=p0,
        i0=initial_influence,
        sefit=2 if include_se and save_influence else 1 if include_se else 0,
        entry=include_entry,
        position=positions,
        hindx=hindx,
        trmat=[list(transition) for transition in transitions],
        t0=t0,
        influence_weights=influence_case_weights,
    )
    n_risk_raw = _survfit_multistate_matrix(raw.n_risk)
    n_censor_raw = _survfit_multistate_matrix(raw.n_censor)
    n_transition_raw = _survfit_multistate_matrix(raw.n_transition)
    pstate = _survfit_multistate_matrix(raw.pstate)
    std_err = None if raw.std_err is None else _survfit_multistate_matrix(raw.std_err)
    conf_lower, conf_upper = _survfit_multistate_confidence(pstate, std_err, conf_level, conf_type)
    transition_counts = [row[transition_count:] for row in n_transition_raw]
    n_event_count = [
        [
            sum(
                row[transition_idx]
                for transition_idx, (_source, target) in enumerate(transitions)
                if target == state
            )
            for state in range(state_count)
        ]
        for row in transition_counts
    ]
    n_enter_raw = None if raw.n_enter is None else _survfit_multistate_matrix(raw.n_enter)
    return SurvfitMultiStateResult(
        time=[float(value) for value in output_times],
        n_risk=[row[:state_count] for row in n_risk_raw],
        n_event=_survfit_multistate_matrix(raw.n_event),
        n_censor=[row[:state_count] for row in n_censor_raw],
        pstate=pstate,
        cumhaz=_survfit_multistate_matrix(raw.cumhaz),
        states=states,
        transitions=transitions,
        p0=p0,
        t0=t0,
        n=n,
        n_id=len(_label_levels(id_values, "id")) if id_values is not None else n,
        std_err=std_err,
        std_err0=std_err0,
        std_chaz=None if raw.std_chaz is None else _survfit_multistate_matrix(raw.std_chaz),
        std_auc=None if raw.std_auc is None else _survfit_multistate_matrix(raw.std_auc),
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        n_risk_count=[row[state_count:] for row in n_risk_raw],
        n_event_count=n_event_count,
        n_censor_count=[row[state_count:] for row in n_censor_raw],
        n_enter=None if n_enter_raw is None else [row[:state_count] for row in n_enter_raw],
        n_enter_count=(None if n_enter_raw is None else [row[state_count:] for row in n_enter_raw]),
        n_transition=[row[:transition_count] for row in n_transition_raw],
        n_transition_count=transition_counts,
        model=model_frame,
        surv_type=response.type,
        conf_type=conf_type,
        conf_level=conf_level,
        p0_fixed=p0_override is not None,
        timefix=timefix,
        influence_state=(
            None if raw.influence is None else _survfit_multistate_matrix(raw.influence)
        ),
        influence_state0=(list(initial_influence) if include_se and save_influence else None),
        influence_chaz=(
            None if raw.influence_chaz is None else _survfit_multistate_matrix(raw.influence_chaz)
        ),
        influence_auc=(
            None if raw.influence_auc is None else _survfit_multistate_matrix(raw.influence_auc)
        ),
    )


def _survfit_multistate_output_times(
    response: Surv,
    positions: list[int],
    *,
    t0: float,
    include_time0: bool,
    include_entry: bool,
) -> list[float]:
    if response.start is None:
        times = sorted(set(response.time))
    elif include_entry:
        times = sorted(
            {
                value
                for idx, (start, stop, event) in enumerate(
                    zip(response.start, response.time, response.event, strict=True)
                )
                if not (positions[idx] == 0 and event == 0)
                for value in (start, stop)
            }
        )
    else:
        times = sorted(
            {
                stop
                for stop, event, position in zip(
                    response.time, response.event, positions, strict=True
                )
                if position >= 2 or event != 0
            }
        )
    if include_time0:
        return [t0, *[time for time in times if time > t0]]
    return [time for time in times if time >= t0]


def _survfit_multistate(
    response: Surv,
    group: Any | None,
    weights: list[float] | None,
    id_values: list[Any] | None,
    cluster: Any | None,
    istate: Any | None,
    p0: Any | None,
    *,
    start_time: float | None,
    include_time0: bool,
    include_se: bool,
    include_entry: bool,
    conf_level: float,
    conf_type: str,
    timefix: bool,
    group_levels: Sequence[Any] | None,
    model_frame: dict[str, Any] | None,
) -> SurvfitMultiStateResult | dict[Any, SurvfitMultiStateResult]:
    response, current_states, states, ids = _survfit_multistate_state_data(
        response, id_values, istate, timefix
    )
    supplied_p0 = _survfit_multistate_p0(p0, len(states))

    def initial_curve_indices(
        curve_response: Surv,
        curve_groups: Sequence[Any],
        curve_ids: Sequence[Any],
    ) -> list[int]:
        group_codes = _encode_labels(list(curve_groups), "group")
        id_codes = _encode_labels(list(curve_ids), "id")
        order = sorted(
            range(len(curve_response)),
            key=lambda idx: (
                group_codes[idx],
                id_codes[idx],
                curve_response.start[idx] if curve_response.start is not None else 0.0,
                idx,
            ),
        )
        first_indices: list[int] = []
        seen_subjects: set[tuple[int, int]] = set()
        for idx in order:
            key = (group_codes[idx], id_codes[idx])
            if key not in seen_subjects:
                seen_subjects.add(key)
                first_indices.append(idx)
        return first_indices

    group_values = [0] * len(response) if group is None else _materialize_labels(group, "group")
    initial_indices = initial_curve_indices(response, group_values, ids)
    initial_states = [current_states[idx] for idx in initial_indices]
    same_initial_state = len(set(initial_states)) == 1
    if start_time is not None:
        t0 = start_time
    elif response.start is None:
        t0 = min(0.0, *response.time)
    elif same_initial_state:
        t0 = min(response.start)
    else:
        initial_starts = [response.start[idx] for idx in initial_indices]
        if max(initial_starts) == min(initial_starts):
            t0 = initial_starts[0]
        else:
            event_times = [
                stop for stop, event in zip(response.time, response.event, strict=True) if event
            ]
            if not event_times:
                raise ValueError("start_time is required when initial states have staggered entry")
            t0 = min(event_times)

    if start_time is not None:
        for label, indices in _group_indices(group_values, len(response)).items():
            if max(response.time[idx] for idx in indices) <= t0:
                raise ValueError(f"start_time has removed all observations from curve {label!r}")
    keep = _survfit_start_time_indices(response, t0, timefix)
    transitions = tuple(
        sorted(
            {
                (current_states[idx], int(event) - 1)
                for idx, event in enumerate(response.event)
                if event
            },
            key=lambda transition: (transition[1], transition[0]),
        )
    )
    response = _subset_surv(response, keep)
    current_states = [current_states[idx] for idx in keep]
    group = _subset_optional_sequence(group, keep, "group")
    weights = _subset_optional_sequence(weights, keep, "weights")
    ids = [ids[idx] for idx in keep]
    id_values = _subset_optional_sequence(id_values, keep, "id")
    cluster = _subset_optional_sequence(cluster, keep, "cluster")
    if start_time is not None:
        kept_groups = [0] * len(response) if group is None else _materialize_labels(group, "group")
        initial_indices = initial_curve_indices(response, kept_groups, ids)
        initial_states = [current_states[idx] for idx in initial_indices]
        same_initial_state = len(set(initial_states)) == 1
    p0_override = supplied_p0
    if p0_override is None and same_initial_state:
        p0_override = [float(state == initial_states[0]) for state in range(len(states))]

    def fit_curve(indices: list[int]) -> SurvfitMultiStateResult:
        curve_response = _subset_surv(response, indices)
        curve_states = [current_states[idx] for idx in indices]
        curve_ids = [ids[idx] for idx in indices]
        curve_positions = (
            [3] * len(indices)
            if curve_response.start is None
            else _survfit_counting_positions(
                curve_response.start,
                curve_response.time,
                curve_ids,
                timefix,
            )
        )
        output_times = _survfit_multistate_output_times(
            curve_response,
            curve_positions,
            t0=t0,
            include_time0=include_time0,
            include_entry=include_entry,
        )
        if curve_response.start is not None and p0_override is None:
            output_times = [time for time in output_times if time > t0]
        if not output_times:
            raise ValueError("multi-state survfit has no output times")
        initial_rows = (
            [True] * len(indices)
            if curve_response.start is None
            else [
                start <= t0 <= stop if t0 == min(response.start) else start < t0 <= stop
                for start, stop in zip(
                    curve_response.start,
                    curve_response.time,
                    strict=True,
                )
            ]
        )
        return _survfit_multistate_curve(
            curve_response,
            _subset_optional_sequence(weights, indices, "weights"),
            _subset_optional_sequence(id_values, indices, "id"),
            _subset_optional_sequence(cluster, indices, "cluster"),
            curve_states,
            curve_positions,
            states,
            transitions,
            t0=t0,
            output_times=output_times,
            initial_rows=initial_rows,
            p0_override=p0_override,
            report_initial_error=not include_time0 and p0_override is None,
            include_se=include_se,
            include_entry=include_entry,
            conf_level=conf_level,
            conf_type=conf_type,
            model_frame=model_frame,
            timefix=timefix,
        )

    if group is None:
        return fit_curve(list(range(len(response))))
    return {
        label: fit_curve(indices)
        for label, indices in _group_indices(group, len(response), levels=group_levels).items()
    }


def survfit(
    response: Any,
    data: Any | None = None,
    *,
    group: Any | None = None,
    newdata: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    conf_level: float = 0.95,
    conf_int: Any | None = None,
    conf_type: str | None = "log",
    se_fit: Any = True,
    start_time: Any | None = None,
    time0: bool = False,
    reverse: bool = False,
    censor: bool = True,
    type: str | None = None,
    stype: int | None = None,
    ctype: int | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    robust: Any | None = None,
    istate: Any | None = None,
    etype: Any | None = None,
    p0: Any | None = None,
    model: Any = False,
    error: Any | None = None,
    entry: Any = False,
    timefix: bool = True,
    **kwargs: Any,
):
    """Fit Kaplan--Meier, Aalen--Johansen, or Cox-model survival curves."""

    if _is_clogit_fit(response):
        raise ValueError("predicted survival curves are not defined for a clogit model")

    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, None)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, True)
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", timefix, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit got unexpected keyword argument(s): {unexpected}")

    keep_model = _normalize_bool_option_with_default(model, "model", False)
    include_entry = _normalize_bool_option_with_default(entry, "entry", False)
    include_se = _normalize_bool_option_with_default(se_fit, "se_fit", True)
    robust_value = _normalize_optional_bool_option(robust, "robust")
    id_arg = id
    if istate is not None and etype is not None:
        raise ValueError("survfit cannot use both istate and etype")

    computation = _normalize_survfit_type(type, stype, ctype)
    normalized_conf_level = _normalize_survfit_conf_level(conf_level, conf_int)
    normalized_conf_type = _normalize_survfit_conf_type(conf_type)
    normalized_start_time = _normalize_start_time(start_time)
    include_time0 = _normalize_bool_option(time0, "time0")
    reverse_curve = _normalize_bool_option(reverse, "reverse")
    include_censor = _normalize_bool_option(censor, "censor")
    fix_time = _normalize_bool_option(timefix, "timefix")
    model_frame = None
    formula_group_levels: tuple[Any, ...] | None = None
    if isinstance(response, str):
        formula = response
        id_column = id_arg if isinstance(id_arg, str) else None
        cluster_column = cluster if isinstance(cluster, str) else None
        etype_column = etype if isinstance(etype, str) else None
        istate_column = istate if isinstance(istate, str) else None
        if id_column is not None:
            id_arg = _column(data, id_column)
        if cluster_column is not None:
            cluster = _column(data, cluster_column)
        if etype_column is not None:
            etype = _column(data, etype_column)
        if istate_column is not None:
            istate = _column(data, istate_column)
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                formula,
                data,
                subset,
                weights=weights,
                id=id_arg,
                cluster=cluster,
                etype=etype,
                istate=istate,
            )
            weights = aligned["weights"]
            id_arg = aligned["id"]
            cluster = aligned["cluster"]
            etype = aligned["etype"]
            istate = aligned["istate"]
            subset = None
        data, aligned = _apply_formula_na_action(
            formula,
            data,
            na_action,
            weights=weights,
            id=id_arg,
            cluster=cluster,
            etype=etype,
            istate=istate,
        )
        weights = aligned["weights"]
        id_arg = aligned["id"]
        cluster = aligned["cluster"]
        etype = aligned["etype"]
        istate = aligned["istate"]
        na_action = "pass"
        response, terms = _parse_formula(formula, data)
        if terms.clusters:
            if cluster is not None:
                raise ValueError("survfit formula cluster() cannot be combined with cluster")
            cluster = _combined_columns(data, terms.clusters, len(response))
        model_frame = _survfit_formula_model_frame(
            formula,
            data,
            response,
            weights,
            id_arg,
            id_column,
            cluster,
            cluster_column,
        )
        if etype is not None:
            model_frame["(etype)"] = _materialize_1d(etype, "etype")
            if etype_column is not None and etype_column not in model_frame:
                model_frame[etype_column] = _column(data, etype_column)
        if istate is not None:
            model_frame["(istate)"] = _materialize_1d(istate, "istate")
            if istate_column is not None and istate_column not in model_frame:
                model_frame[istate_column] = _column(data, istate_column)
        if terms.strata or terms.covariates:
            group = _combined_formula_groups(data, terms.strata, terms.covariates, len(response))
            formula_group_levels = _r_formula_ordered_levels(group, "survfit formula groups")

    if p0 is not None and (
        not isinstance(response, Surv) or response.type not in {"mright", "mcounting"}
    ):
        raise ValueError("p0 is only supported for multi-state Surv responses")

    if not isinstance(response, Surv) and hasattr(response, "survival_curve"):
        if etype is not None or istate is not None:
            raise ValueError("etype and istate are only supported for Surv or formula inputs")
        if not computation.is_kaplan_meier:
            raise ValueError(
                "non-Kaplan-Meier survfit styles are only supported for Surv or formula inputs"
            )
        if reverse_curve:
            raise ValueError("reverse survfit is only supported for Surv or formula inputs")
        if subset is not None:
            raise ValueError("subset is only supported for Surv or formula inputs")
        rows, offsets = _prediction_inputs(response, newdata)
        if hasattr(response, "means"):
            result = _cox_survfit_result(
                response,
                rows,
                offsets,
                True,
                newdata,
                normalized_start_time,
                include_time0,
                include_censor,
                normalized_conf_level,
                normalized_conf_type,
                compute_confidence=include_se,
            )
            return (
                _survfit_with_model_frame(result, _cox_survfit_model_frame(response, newdata))
                if keep_model
                else result
            )
        if normalized_start_time is not None:
            raise ValueError("start_time is only supported for Surv, formula, or fitted Cox inputs")
        if include_time0:
            raise ValueError("time0 is only supported for Surv, formula, or fitted Cox inputs")
        if keep_model:
            raise NotImplementedError(
                "survfit model=TRUE is only supported for Surv, formula, or fitted Cox inputs"
            )
        if rows is None:
            coefficients = getattr(response, "coefficients", [])
            width = len(coefficients[0]) if coefficients else 0
            if width == 0:
                raise ValueError("newdata is required for an unfitted Cox model")
            rows = [[0.0] * width]
        return response.survival_curve(rows, None)

    if not isinstance(response, Surv):
        raise TypeError("survfit response must be a Surv object, formula, or fitted Cox model")
    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
        weights = _subset_optional_sequence(weights, indices, "weights")
        id_arg = _subset_optional_sequence(id_arg, indices, "id")
        cluster = _subset_optional_sequence(cluster, indices, "cluster")
        etype = _subset_optional_sequence(etype, indices, "etype")
        istate = _subset_optional_sequence(istate, indices, "istate")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "survfit inputs",
        group=group,
        weights=weights,
        id=id_arg,
        cluster=cluster,
        etype=etype,
        istate=istate,
    )
    group = aligned["group"]
    weights = aligned["weights"]
    id_arg = aligned["id"]
    cluster = aligned["cluster"]
    etype = aligned["etype"]
    istate = aligned["istate"]
    if etype is not None:
        response = _survfit_response_with_etype(response, etype)
        if model_frame is not None:
            for name, value in model_frame.items():
                if isinstance(value, Surv):
                    model_frame[name] = response
                    break
    id_values = _materialize_labels(id_arg, "id") if id_arg is not None else None
    if id_values is not None and len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if model_frame is None:
        model_frame = _survfit_model_frame(response, group, weights, id_values, cluster)
        if istate is not None:
            model_frame["(istate)"] = _materialize_1d(istate, "istate")
    if newdata is not None:
        raise ValueError("newdata is only supported for fitted Cox models")
    if not include_censor:
        raise ValueError("censor is only supported for fitted Cox models")
    if (
        include_entry
        and response.type != "mcounting"
        and (response.start is None or id_values is None)
    ):
        raise ValueError("survfit entry=TRUE requires counting-process Surv input and id")
    if response.type in {"mright", "mcounting"}:
        if robust_value is False:
            raise ValueError("multi-state survfit supports only a robust variance")
        if not computation.is_kaplan_meier:
            raise ValueError("multi-state survfit supports only the Aalen-Johansen estimator")
        if reverse_curve:
            raise ValueError("reverse survfit is not supported for multi-state responses")
        wt = _float_vector(weights, "weights") if weights is not None else None
        if wt is not None and len(wt) != len(response):
            raise ValueError("weights must have the same length as the Surv response")
        return _survfit_multistate(
            response,
            group,
            wt,
            id_values,
            cluster,
            istate,
            p0,
            start_time=normalized_start_time,
            include_time0=include_time0,
            include_se=include_se,
            include_entry=include_entry and response.type == "mcounting",
            conf_level=normalized_conf_level,
            conf_type=normalized_conf_type,
            timefix=fix_time,
            group_levels=formula_group_levels,
            model_frame=model_frame if keep_model else None,
        )
    if response.type in {"left", "interval", "interval2"}:
        if (
            include_se
            and _survfit_robust_cluster_values(response, cluster, id_values, None, robust_value)
            is not None
        ):
            raise NotImplementedError(
                "survfit robust variance is currently supported only for right-censored or "
                "counting-process Kaplan-Meier curves"
            )
        if not computation.is_kaplan_meier:
            raise ValueError(
                "non-Kaplan-Meier survfit styles are only supported for right-censored data"
            )
        if normalized_conf_type != "log":
            raise ValueError("conf_type is only supported for right-censored data")
        if normalized_start_time is not None:
            raise ValueError("start_time is only supported for right-censored data")
        if include_time0:
            raise ValueError("time0 is only supported for right-censored data")
        if reverse_curve:
            raise ValueError("reverse survfit is only supported for right-censored data")
        wt = _float_vector(weights, "weights") if weights is not None else None
        if wt is not None and len(wt) != len(response):
            raise ValueError("weights must have the same length as the Surv response")
        if response.start is not None:
            raise ValueError("interval-censored survfit does not support entry times")
        if group is None:
            left, right = _turnbull_intervals(response)
            result = _core.turnbull_estimator(left, right, weights=wt)
            return (
                _survfit_with_model_frame(result, model_frame)
                if keep_model and model_frame is not None
                else result
            )

        grouped_indices = _group_indices(group, len(response), levels=formula_group_levels)
        group_codes = [0] * len(response)
        for group_code, indices in enumerate(grouped_indices.values()):
            for idx in indices:
                group_codes[idx] = group_code
        left, right = _turnbull_intervals(response)
        raw_grouped = _core.turnbull_estimator_grouped(
            left,
            right,
            group_codes,
            weights=wt,
        )
        raw_groups = [int(value) for value in raw_grouped.groups]
        if raw_groups != list(range(len(grouped_indices))):
            raise RuntimeError("grouped Turnbull fit returned inconsistent group codes")
        raw_time_points = raw_grouped.time_points
        raw_survival = raw_grouped.survival
        raw_survival_lower = raw_grouped.survival_lower
        raw_survival_upper = raw_grouped.survival_upper
        raw_n_iter = raw_grouped.n_iter
        raw_converged = raw_grouped.converged
        return {
            label: TurnbullSurvfitResult(
                time_points=raw_time_points[curve_idx],
                survival=raw_survival[curve_idx],
                survival_lower=raw_survival_lower[curve_idx],
                survival_upper=raw_survival_upper[curve_idx],
                n_iter=raw_n_iter[curve_idx],
                converged=raw_converged[curve_idx],
                model=model_frame if keep_model else None,
            )
            for curve_idx, label in enumerate(grouped_indices)
        }

    wt = _float_vector(weights, "weights") if weights is not None else None
    if wt is not None and len(wt) != len(response):
        raise ValueError("weights must have the same length as the Surv response")
    t0 = (
        normalized_start_time
        if normalized_start_time is not None
        else _survfit_default_time0(response)
    )
    if normalized_start_time is not None:
        indices = _survfit_start_time_indices(response, normalized_start_time, fix_time)
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
        wt = _subset_optional_sequence(wt, indices, "weights")
        id_values = _subset_optional_sequence(id_values, indices, "id")
        cluster = _subset_optional_sequence(cluster, indices, "cluster")
    entry_times = list(response.start) if response.start is not None else None
    robust_clusters = (
        _survfit_robust_cluster_values(response, cluster, id_values, wt, robust_value)
        if include_se
        else None
    )
    if robust_clusters is not None and response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survfit robust variance is currently supported only for right-censored or "
            "counting-process curves"
        )
    if group is None:
        km = (
            _survfit_counting_with_id(
                response,
                wt,
                id_values,
                include_entry=include_entry,
                reverse=reverse_curve,
                conf_level=normalized_conf_level,
                conf_type=normalized_conf_type,
                computation=computation,
                timefix=fix_time,
            )
            if response.start is not None and id_values is not None
            else _survfitkm(
                list(response.time),
                list(response.event),
                weights=wt,
                entry_times=entry_times,
                reverse=reverse_curve,
                conf_level=normalized_conf_level,
                conf_type=normalized_conf_type,
                timefix=fix_time,
            )
        )
        if computation.is_kaplan_meier:
            if robust_clusters is not None:
                km = _survfit_robust_km_result(
                    km,
                    response,
                    wt,
                    robust_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    timefix=fix_time,
                )
            result = (
                _survfit_with_time0(
                    km,
                    t0,
                    normalized_conf_type,
                    _initial_survfit_risk(response, wt, t0, fix_time),
                    fix_time,
                )
                if include_time0
                else km
            )
            result = _survfit_without_standard_errors(result) if not include_se else result
            return (
                _survfit_with_model_frame(result, model_frame)
                if model_frame is not None
                else result
            )
        result = _survfit_from_km_counts(
            km,
            normalized_conf_level,
            computation,
            normalized_conf_type,
        )
        if robust_clusters is not None:
            if response.start is not None:
                result = _survfit_robust_counting_result(
                    result,
                    response,
                    wt,
                    robust_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    computation=computation,
                    timefix=fix_time,
                )
            else:
                result = _survfit_robust_right_result(
                    result,
                    response,
                    wt,
                    robust_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    computation=computation,
                    timefix=fix_time,
                )
        result = (
            _survfit_with_time0(
                result,
                t0,
                normalized_conf_type,
                _initial_survfit_risk(response, wt, t0, fix_time),
                fix_time,
            )
            if include_time0
            else result
        )
        result = _survfit_without_standard_errors(result) if not include_se else result
        return _survfit_with_model_frame(result, model_frame) if model_frame is not None else result

    grouped_indices = _group_indices(group, len(response), levels=formula_group_levels)
    batched_km: dict[int, Any] | None = None
    if (
        robust_clusters is None
        and not include_time0
        and not (response.start is not None and id_values is not None)
    ):
        labels = list(grouped_indices)
        group_codes = _encode_labels_with_levels(
            _materialize_labels(group, "group"),
            labels,
            "group",
        )
        raw_grouped = _core.survfitkm_grouped(
            list(response.time),
            list(response.event),
            group_codes,
            weights=wt,
            entry_times=list(response.start) if response.start is not None else None,
            reverse=reverse_curve,
            conf_level=normalized_conf_level,
            conf_type=normalized_conf_type,
            timefix=fix_time,
        )
        raw_groups = [int(value) for value in raw_grouped.groups]
        raw_time = raw_grouped.time
        raw_n_risk = raw_grouped.n_risk
        raw_n_risk_count = raw_grouped.n_risk_count
        raw_n_event = raw_grouped.n_event
        raw_n_event_count = raw_grouped.n_event_count
        raw_n_censor = raw_grouped.n_censor
        raw_n_censor_count = raw_grouped.n_censor_count
        raw_estimate = raw_grouped.estimate
        raw_std_err = raw_grouped.std_err
        raw_cumhaz = raw_grouped.cumhaz
        raw_std_chaz = raw_grouped.std_chaz
        raw_conf_lower = raw_grouped.conf_lower
        raw_conf_upper = raw_grouped.conf_upper
        batched_km = {
            group_code: SurvfitResult(
                time=raw_time[curve_idx],
                n_risk=raw_n_risk[curve_idx],
                n_event=raw_n_event[curve_idx],
                n_censor=raw_n_censor[curve_idx],
                estimate=raw_estimate[curve_idx],
                std_err=raw_std_err[curve_idx],
                conf_lower=raw_conf_lower[curve_idx],
                conf_upper=raw_conf_upper[curve_idx],
                cumhaz=raw_cumhaz[curve_idx],
                std_chaz=raw_std_chaz[curve_idx],
                n_risk_count=raw_n_risk_count[curve_idx],
                n_event_count=raw_n_event_count[curve_idx],
                n_censor_count=raw_n_censor_count[curve_idx],
                model=model_frame,
            )
            for curve_idx, group_code in enumerate(raw_groups)
        }
        if set(batched_km) != set(range(len(labels))):
            raise RuntimeError("grouped survfit returned inconsistent group codes")

    results: dict[Any, Any] = {}
    for group_code, (label, indices) in enumerate(grouped_indices.items()):
        if batched_km is not None:
            group_response = response
            group_weights = wt
            group_clusters = None
            km = batched_km[group_code]
        else:
            group_response = _subset_surv(response, indices)
            group_weights = [wt[idx] for idx in indices] if wt is not None else None
            group_ids = [id_values[idx] for idx in indices] if id_values is not None else None
            group_clusters = (
                [robust_clusters[idx] for idx in indices] if robust_clusters is not None else None
            )
            km = (
                _survfit_counting_with_id(
                    group_response,
                    group_weights,
                    group_ids,
                    include_entry=include_entry,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    computation=computation,
                    timefix=fix_time,
                )
                if group_response.start is not None and group_ids is not None
                else _survfitkm(
                    list(group_response.time),
                    list(group_response.event),
                    weights=group_weights,
                    entry_times=(
                        list(group_response.start) if group_response.start is not None else None
                    ),
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    timefix=fix_time,
                )
            )
        if computation.is_kaplan_meier:
            if group_clusters is not None:
                km = _survfit_robust_km_result(
                    km,
                    group_response,
                    group_weights,
                    group_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    timefix=fix_time,
                )
            results[label] = (
                _survfit_with_time0(
                    km,
                    t0,
                    normalized_conf_type,
                    _initial_survfit_risk(group_response, group_weights, t0, fix_time),
                    fix_time,
                )
                if include_time0
                else km
            )
        else:
            result = _survfit_from_km_counts(
                km,
                normalized_conf_level,
                computation,
                normalized_conf_type,
            )
            if group_clusters is not None:
                if group_response.start is not None:
                    result = _survfit_robust_counting_result(
                        result,
                        group_response,
                        group_weights,
                        group_clusters,
                        reverse=reverse_curve,
                        conf_level=normalized_conf_level,
                        conf_type=normalized_conf_type,
                        computation=computation,
                        timefix=fix_time,
                    )
                else:
                    result = _survfit_robust_right_result(
                        result,
                        group_response,
                        group_weights,
                        group_clusters,
                        reverse=reverse_curve,
                        conf_level=normalized_conf_level,
                        conf_type=normalized_conf_type,
                        computation=computation,
                        timefix=fix_time,
                    )
            results[label] = (
                _survfit_with_time0(
                    result,
                    t0,
                    normalized_conf_type,
                    _initial_survfit_risk(group_response, group_weights, t0, fix_time),
                    fix_time,
                )
                if include_time0
                else result
            )
    result = _survfit_without_standard_errors(results) if not include_se else results
    return (
        _survfit_with_model_frame(result, model_frame)
        if model_frame is not None and batched_km is None
        else result
    )
