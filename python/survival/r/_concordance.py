"""``concordance`` and the legacy ``survConcordance`` entry points."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, NoReturn

from .. import _survival as _core
from ._coerce import (
    _as_matrix_rows,
    _concordance_core_time_values,
    _encode_groups,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _is_bool_like,
    _label_levels,
    _materialize_labels,
    _normalize_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
    _subset_indices,
    _subset_optional_sequence,
    _subset_sequence,
    _survdiff_timefix_values,
    _timefix_vectors,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _combined_columns,
    _design_term_columns,
    _design_term_output_names,
    _fit_design_term,
    _parse_formula,
    _subset_formula_inputs,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._types import ConcordanceResult, _FormulaTerms


def _matrix_score_columns(rows: list[list[float]]) -> tuple[list[list[float]], list[str]]:
    width = len(rows[0]) if rows else 0
    columns = [[row[col_idx] for row in rows] for col_idx in range(width)]
    names = [f"score{col_idx + 1}" for col_idx in range(width)]
    return columns, names


def _external_concordance_score_columns(
    scores: Any,
    n: int,
) -> tuple[list[list[float]], list[str]]:
    rows = _as_matrix_rows(scores, "scores", allow_empty_columns=False)
    if len(rows) != n:
        raise ValueError("scores must have the same number of rows as the Surv response")
    return _matrix_score_columns(rows)


def _concordance_score_columns(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[list[float]], list[str]]:
    if terms.offsets:
        raise ValueError("Offset terms not allowed")
    if not terms.covariates:
        raise ValueError("concordance formula requires a risk score")

    columns: list[list[float]] = []
    names: list[str] = []
    for term in terms.covariates:
        design_term = _fit_design_term(data, term, n)
        term_columns = _design_term_columns(data, design_term, n)
        columns.extend(term_columns)
        names.extend(_design_term_output_names(design_term))

    if not columns:
        raise ValueError("concordance formula requires a risk score")
    return columns, names


_CONCORDANCE_TIMEWT_CHOICES = ("n", "S", "S/G", "n/G2", "I")


def _normalize_concordance_timewt(timewt: Any) -> str:
    if timewt is None:
        return "n"
    if isinstance(timewt, str):
        value = timewt
    else:
        try:
            values = list(timewt)
        except TypeError as exc:
            raise TypeError("timewt must be a string") from exc
        if tuple(values) == _CONCORDANCE_TIMEWT_CHOICES:
            return "n"
        if len(values) != 1:
            raise ValueError("timewt must be a single value")
        value = str(values[0])
    if value not in _CONCORDANCE_TIMEWT_CHOICES:
        choices = "', '".join(_CONCORDANCE_TIMEWT_CHOICES)
        raise ValueError(f"timewt must be one of '{choices}'")
    return value


def _normalize_concordance_influence(influence: Any) -> int:
    if influence is None:
        return 0
    if _is_bool_like(influence):
        value = int(bool(influence))
    else:
        value = _integer_scalar(influence, "influence")
    if value not in {0, 1, 2, 3}:
        raise ValueError("influence must be 0, 1, 2, or 3")
    return value


def _validate_concordance_keepstrata(keepstrata: Any) -> None:
    if keepstrata is None:
        return
    if _is_bool_like(keepstrata):
        return
    _finite_float(keepstrata, "keepstrata")


def _normalize_concordance_time_bound(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _formula_weight_values(data: Any, weights: Any | None) -> Any | None:
    if isinstance(weights, str):
        return _column(data, weights)
    return weights


def _formula_cluster_values(data: Any, cluster: Any | None) -> Any | None:
    if isinstance(cluster, str):
        return _column(data, cluster)
    return cluster


def _concordance_weight_values(weights: Any | None, n: int) -> list[float] | None:
    values = _optional_float_vector(weights, "weights", n)
    if values is None:
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("weights must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("weights must be non-negative")
    return values


def _concordance_bounded_times_and_status(
    times: list[float],
    status: list[int],
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[float], list[int]]:
    bounded_times = [max(value, ymin) for value in times] if ymin is not None else list(times)
    bounded_status = list(status)
    if ymax is not None:
        bounded_status = [
            0 if event == 1 and bounded_times[idx] > ymax else event
            for idx, event in enumerate(bounded_status)
        ]
    return bounded_times, bounded_status


def _concordance_rank_row_dicts(
    rows: list[tuple[float, float, float, float]],
    display_by_core_time: dict[float, float] | None = None,
) -> list[dict[str, float]]:
    return [
        {
            "time": float(display_by_core_time.get(time, time) if display_by_core_time else time),
            "rank": float(rank),
            "timewt": float(time_weight),
            "casewt": float(case_weight),
        }
        for time, rank, time_weight, case_weight in rows
    ]


@dataclass(frozen=True)
class _RightConcordanceData:
    times: list[float]
    status: list[int]
    display_by_core_time: dict[float, float] | None


@dataclass(frozen=True)
class _CountingConcordanceData:
    start: list[float]
    stop: list[float]
    status: list[int]


def _unsupported_concordance_response() -> NoReturn:
    raise NotImplementedError(
        "concordance currently supports right-censored and counting Surv responses"
    )


def _right_concordance_data(
    response: Surv,
    timefix: bool,
    ymin: float | None,
    ymax: float | None,
) -> _RightConcordanceData:
    times = _survdiff_timefix_values(list(response.time), timefix)
    status = list(response.event)
    times, status = _concordance_bounded_times_and_status(times, status, ymin, ymax)
    core_times, display_by_core_time = _concordance_core_time_values(times, timefix)
    return _RightConcordanceData(core_times, status, display_by_core_time)


def _counting_concordance_data(
    response: Surv,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
    *,
    preapply_timefix: bool,
) -> _CountingConcordanceData:
    if timewt in {"S/G", "n/G2"}:
        raise ValueError("S/G and n/G2 timewt options are not supported for counting-process data")
    if response.start is None:
        raise ValueError("counting-process concordance requires start times")

    start = list(response.start)
    stop = list(response.time)
    if preapply_timefix and timefix:
        start, stop = _timefix_vectors(start, stop)
    status = list(response.event)
    stop, status = _concordance_bounded_times_and_status(stop, status, ymin, ymax)
    return _CountingConcordanceData(start, stop, status)


def _single_concordance_ranks(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> list[dict[str, float]]:
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_rank_row_dicts(
            _core.concordance_rank_rows(
                data.times,
                data.status,
                risk_values,
                case_weights,
                timewt,
            ),
            data.display_by_core_time,
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_rank_row_dicts(
            _core.counting_concordance_rank_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


def _concordance_ranks(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> list[dict[str, float]]:
    if strata is None:
        return _single_concordance_ranks(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
    strata_codes = _encode_groups(strata, len(response))
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_rank_row_dicts(
            _core.stratified_concordance_rank_rows(
                data.times,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
            ),
            data.display_by_core_time,
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_rank_row_dicts(
            _core.stratified_counting_concordance_rank_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


def _concordance_influence_result(
    result: tuple[list[list[float]], list[float], float],
) -> tuple[list[list[float]], list[float], float]:
    influence_rows, dfbeta, variance = result
    return (
        [[float(value) for value in row] for row in influence_rows],
        [float(value) for value in dfbeta],
        float(variance),
    )


def _single_concordance_influence(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[list[float]], list[float], float | None]:
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_influence_result(
            _core.concordance_influence_rows(
                data.times,
                data.status,
                risk_values,
                case_weights,
                timewt,
            )
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_influence_result(
            _core.counting_concordance_influence_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


def _concordance_influence(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[list[float]], list[float], float | None]:
    if strata is None:
        return _single_concordance_influence(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
    strata_codes = _encode_groups(strata, len(response))
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_influence_result(
            _core.stratified_concordance_influence_rows(
                data.times,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
            )
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_influence_result(
            _core.stratified_counting_concordance_influence_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


def _concordance_cluster_values(cluster: Any, n: int) -> list[Any]:
    values = _materialize_labels(cluster, "cluster")
    if len(values) != n:
        raise ValueError("cluster must have the same length as the Surv response")
    _label_levels(values, "cluster")
    return values


def _clustered_concordance_dfbeta(
    dfbeta: list[float],
    cluster: list[Any],
) -> tuple[list[float], float]:
    collapsed: dict[Any, float] = {}
    order: list[Any] = []
    for label, value in zip(cluster, dfbeta, strict=True):
        if label not in collapsed:
            collapsed[label] = 0.0
            order.append(label)
        collapsed[label] += value
    cluster_dfbeta = [collapsed[label] for label in order]
    return cluster_dfbeta, math.fsum(value * value for value in cluster_dfbeta)


def _single_concordance_summary(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> dict[str, float]:
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        summary = _core.concordance_summary(
            data.times,
            data.status,
            risk_values,
            weights,
            timewt,
        )
        summary["n_event"] = float(sum(1 for event in data.status if event == 1))
        return summary
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=False,
        )
        summary = _core.counting_concordance_summary(
            data.start,
            data.stop,
            data.status,
            risk_values,
            weights,
            timewt,
            timefix,
        )
        summary["n_event"] = float(sum(1 for event in data.status if event == 1))
        return summary
    return _unsupported_concordance_response()


def _concordance_summary(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> dict[str, float]:
    if strata is None:
        return _single_concordance_summary(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )

    strata_codes = _encode_groups(strata, len(response))
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return {
            key: float(value)
            for key, value in _core.stratified_concordance_summary(
                data.times,
                data.status,
                risk_values,
                strata_codes,
                weights,
                timewt,
            ).items()
        }
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=False,
        )
        return {
            key: float(value)
            for key, value in _core.stratified_counting_concordance_summary(
                data.start,
                data.stop,
                data.status,
                risk_values,
                strata_codes,
                weights,
                timewt,
                timefix,
            ).items()
        }
    return _unsupported_concordance_response()


def _single_score_concordance_result(
    response: Surv,
    score_values: list[float],
    weight_values: list[float] | None,
    strata_values: Any | None,
    cluster_values: list[Any] | None,
    reverse_scores: bool,
    fix_time: bool,
    time_weight: str,
    lower_bound: float | None,
    upper_bound: float | None,
    influence_value: int,
    include_ranks: bool,
) -> ConcordanceResult:
    if len(score_values) != len(response):
        raise ValueError("scores must have the same length as the Surv response")

    risk_values = [-value for value in score_values] if reverse_scores else score_values
    summary = _concordance_summary(
        response,
        risk_values,
        weight_values,
        strata_values,
        fix_time,
        time_weight,
        lower_bound,
        upper_bound,
    )
    tied_x = float(summary.get("tied_x", 0.0))
    tied_y = float(summary.get("tied_y", 0.0))
    tied_xy = float(summary.get("tied_xy", 0.0))
    rank_rows = (
        _concordance_ranks(
            response,
            risk_values,
            weight_values,
            strata_values,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
        )
        if include_ranks
        else None
    )
    influence_rows = None
    dfbeta = None
    variance = None
    if influence_value or cluster_values is not None:
        influence_rows, dfbeta, variance = _concordance_influence(
            response,
            risk_values,
            weight_values,
            strata_values,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
        )
        if cluster_values is not None and dfbeta is not None:
            dfbeta, variance = _clustered_concordance_dfbeta(dfbeta, cluster_values)
    return ConcordanceResult(
        concordance=float(summary["concordance"]),
        n=len(response),
        n_event=int(summary["n_event"]),
        reverse=reverse_scores,
        concordant=float(summary["concordant"]),
        comparable=float(summary["comparable"]),
        tied_x=tied_x,
        tied_y=tied_y,
        tied_xy=tied_xy,
        ranks=rank_rows,
        dfbeta=dfbeta if influence_value in {1, 3} else None,
        influence=influence_rows if influence_value in {2, 3} else None,
        variance=variance if influence_value or cluster_values is not None else None,
        conditional_variance=float(summary["conditional_variance"]),
    )


def _multi_score_concordance_result(
    response: Surv,
    score_columns: list[list[float]],
    score_names: list[str],
    weight_values: list[float] | None,
    strata_values: Any | None,
    cluster_values: list[Any] | None,
    reverse_scores: bool,
    fix_time: bool,
    time_weight: str,
    lower_bound: float | None,
    upper_bound: float | None,
    influence_value: int,
    include_ranks: bool,
) -> ConcordanceResult:
    results = [
        _single_score_concordance_result(
            response,
            score_values,
            weight_values,
            strata_values,
            cluster_values,
            reverse_scores,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
            influence_value,
            include_ranks,
        )
        for score_values in score_columns
    ]
    return ConcordanceResult(
        concordance=[float(result.concordance) for result in results],
        n=len(response),
        n_event=results[0].n_event if results else 0,
        reverse=reverse_scores,
        concordant=[float(result.concordant) for result in results],
        comparable=[float(result.comparable) for result in results],
        tied_x=[float(result.tied_x) for result in results],
        tied_y=[float(result.tied_y) for result in results],
        tied_xy=[float(result.tied_xy) for result in results],
        ranks=[result.ranks for result in results] if include_ranks else None,
        dfbeta=[result.dfbeta for result in results] if influence_value in {1, 3} else None,
        influence=[result.influence for result in results] if influence_value in {2, 3} else None,
        variance=(
            [result.variance for result in results]
            if influence_value or cluster_values is not None
            else None
        ),
        conditional_variance=[
            float(result.conditional_variance)
            if isinstance(result.conditional_variance, int | float)
            else math.nan
            for result in results
        ],
        score_names=score_names,
    )


def concordance(
    response: Surv | str,
    data: Any | None = None,
    *,
    scores: Any | None = None,
    risk_scores: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    cluster: Any | None = None,
    ymin: Any | None = None,
    ymax: Any | None = None,
    timewt: Any = "n",
    influence: Any = 0,
    ranks: bool = False,
    reverse: bool = False,
    timefix: bool = True,
    keepstrata: Any = 10,
    **kwargs: Any,
) -> ConcordanceResult:
    """R-style concordance wrapper backed by Rust Harrell C-index."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", timefix, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"concordance got unexpected keyword argument(s): {unexpected}")

    reverse_scores = _normalize_bool_option(reverse, "reverse")
    fix_time = _normalize_bool_option(timefix, "timefix")
    time_weight = _normalize_concordance_timewt(timewt)
    lower_bound = _normalize_concordance_time_bound(ymin, "ymin")
    upper_bound = _normalize_concordance_time_bound(ymax, "ymax")
    influence_value = _normalize_concordance_influence(influence)
    include_ranks = _normalize_bool_option(ranks, "ranks")
    _validate_concordance_keepstrata(keepstrata)
    if scores is not None and risk_scores is not None:
        raise ValueError("use only one of scores or risk_scores")
    external_scores = risk_scores if risk_scores is not None else scores
    strata_values = None
    cluster_values = None
    score_names: list[str] | None = None
    formula_input = isinstance(response, str)
    effective_reverse_scores = reverse_scores

    if formula_input:
        effective_reverse_scores = not reverse_scores
        if external_scores is not None:
            raise ValueError("concordance formula input cannot be combined with scores")
        weights = _formula_weight_values(data, weights)
        cluster = _formula_cluster_values(data, cluster)
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                response,
                data,
                subset,
                weights=weights,
                cluster=cluster,
            )
            weights = aligned["weights"]
            cluster = aligned["cluster"]
            subset = None
        data, aligned = _apply_formula_na_action(
            response,
            data,
            na_action,
            weights=weights,
            cluster=cluster,
        )
        weights = aligned["weights"]
        cluster = aligned["cluster"]
        na_action = "pass"
        response, terms = _parse_formula(response, data)
        if terms.clusters:
            if cluster is not None:
                raise ValueError("use only one of formula cluster(...) or cluster")
            cluster = _combined_columns(data, terms.clusters, len(response))
        score_columns, score_names = _concordance_score_columns(data, terms, len(response))
        if terms.strata:
            strata_values = _combined_columns(data, terms.strata, len(response))
        weight_values = _concordance_weight_values(weights, len(response))
    else:
        if not isinstance(response, Surv):
            raise TypeError("concordance response must be a Surv object or formula")
        if external_scores is None:
            raise ValueError("scores are required when response is not a formula")
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            external_scores = _subset_sequence(external_scores, indices, "scores")
            weights = _subset_optional_sequence(weights, indices, "weights")
            cluster = _subset_optional_sequence(cluster, indices, "cluster")
        response, aligned = _apply_surv_na_action(
            response,
            na_action,
            "concordance inputs",
            scores=external_scores,
            weights=weights,
            cluster=cluster,
        )
        external_scores = aligned["scores"]
        weights = aligned["weights"]
        cluster = aligned["cluster"]
        score_columns, score_names = _external_concordance_score_columns(
            external_scores,
            len(response),
        )
        weight_values = _concordance_weight_values(weights, len(response))

    if cluster is not None:
        cluster_values = _concordance_cluster_values(cluster, len(response))

    if len(score_columns) == 1:
        result = _single_score_concordance_result(
            response,
            score_columns[0],
            weight_values,
            strata_values,
            cluster_values,
            effective_reverse_scores,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
            influence_value,
            include_ranks,
        )
        return (
            ConcordanceResult(
                concordance=result.concordance,
                n=result.n,
                n_event=result.n_event,
                reverse=result.reverse,
                concordant=result.concordant,
                comparable=result.comparable,
                tied_x=result.tied_x,
                tied_y=result.tied_y,
                tied_xy=result.tied_xy,
                ranks=result.ranks,
                dfbeta=result.dfbeta,
                influence=result.influence,
                variance=result.variance,
                conditional_variance=result.conditional_variance,
                score_names=score_names,
            )
            if score_names is not None
            else result
        )

    return _multi_score_concordance_result(
        response,
        score_columns,
        score_names or [f"score{idx + 1}" for idx in range(len(score_columns))],
        weight_values,
        strata_values,
        cluster_values,
        effective_reverse_scores,
        fix_time,
        time_weight,
        lower_bound,
        upper_bound,
        influence_value,
        include_ranks,
    )


def survConcordance(
    formula: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    **kwargs: Any,
) -> ConcordanceResult:
    """Deprecated R-compatible alias for ``concordance``."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    warnings.warn(
        "survConcordance is deprecated; use concordance instead",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.setdefault("reverse", True)
    return concordance(
        formula,
        data=data,
        weights=weights,
        subset=subset,
        na_action=na_action,
        **kwargs,
    )


def _survConcordance_legacy_stats(
    result: ConcordanceResult,
) -> dict[str, float]:
    if isinstance(result.concordance, list):
        raise ValueError("survConcordance.fit expects a single score vector")
    concordant = float(result.concordant)
    comparable = float(result.comparable)
    discordant = max(comparable - concordant, 0.0)
    variance = result.variance
    if isinstance(variance, list):
        variance = variance[0] if variance else None
    std_cd = math.nan
    if variance is not None and math.isfinite(float(variance)) and variance >= 0.0:
        std_cd = math.sqrt(float(variance)) * 2.0 * comparable
    return {
        "concordant": concordant,
        "discordant": discordant,
        "tied.risk": 0.0,
        "tied.time": 0.0,
        "std(c-d)": std_cd,
    }


def survConcordance_fit(
    y: Any,
    x: Any,
    strata: Any | None = None,
    weight: Any | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Deprecated R-compatible ``survConcordance.fit`` statistics helper."""

    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", True, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survConcordance.fit got unexpected keyword argument(s): {unexpected}")
    if not isinstance(y, Surv):
        raise TypeError("y must be a Surv object")
    result = _single_score_concordance_result(
        y,
        _float_vector(x, "x"),
        _concordance_weight_values(weight, len(y)),
        None if strata is None else _materialize_labels(strata, "strata"),
        None,
        False,
        _normalize_bool_option(timefix, "timefix"),
        "n",
        None,
        None,
        1,
        False,
    )
    warnings.warn(
        "survConcordance.fit is deprecated; use concordance instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _survConcordance_legacy_stats(result)
