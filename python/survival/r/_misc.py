"""statefig, brier, royston, yates, cipoisson, bounded links, survobrien, survcheck, splines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_rows,
    _coerce_array_like,
    _encode_groups,
    _finite_float,
    _float_vector,
    _hashable_group_value,
    _int_vector,
    _integer_code_vector,
    _integer_scalar,
    _is_bool_like,
    _is_missing_value,
    _materialize_1d,
    _materialize_labels,
    _model_residual_weights,
    _normalize_bool_option,
    _normalize_conf_level,
    _normalize_na_action,
    _normalize_numeric_sequence_or_none,
    _pop_dotted_keyword,
    _recycle_r_vector,
    _scalar_or_vector,
    _scalar_or_vector_with_flag,
    _subset_indices,
    _subset_optional_sequence,
    _survcheck_integer_labels,
    _survdiff_timefix_values,
    _timefix_vectors,
)
from ._coxph import _cox_survival_curve, _cox_training_response, _step_curve_at, coxph
from ._data_prep import _rttright_counting_common_start
from ._fit import (
    _cox_loglik_values,
    _cox_training_rows,
    _cox_variance_matrix,
    _formula_design_for_fit,
    _is_coxph_fit,
    _surv_from_formula_design,
    _unwrap_formula_fit,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _combined_columns,
    _covariate_term_name,
    _parse_formula,
    _subset_formula_inputs,
    _term_values,
)
from ._models import predict
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._survfit import survfit, survfit0
from ._types import (
    _MISSING,
    CoxSurvfitResult,
    SurvfitResult,
    SurvObrienResult,
    YatesPairwiseResult,
    YatesResult,
    _CovariateTerm,
    _cox_beta,
    _FormulaFit,
    _FormulaTerms,
    _InteractionTerm,
)


def _statefig_layout_matrix(layout: Any) -> tuple[list[list[float]], bool]:
    rows = _coerce_array_like(layout, "layout")
    if not rows:
        raise ValueError("layout must not be empty")
    if isinstance(rows[0], list | tuple):
        width = len(rows[0])
        matrix = [[_finite_float(value, "layout") for value in row] for row in rows]
        if any(len(row) != width for row in matrix):
            raise ValueError("layout must be rectangular")
        return matrix, True
    return [[_finite_float(value, "layout") for value in rows]], False


def _statefig_space(n: int) -> list[float]:
    return [(idx + 0.5) / n for idx in range(n)]


def _statefig_positions_from_layout(
    layout: Any,
    n_states: int,
) -> tuple[list[list[float]], list[int]]:
    matrix, is_matrix = _statefig_layout_matrix(layout)
    n_row = len(matrix)
    n_col = len(matrix[0]) if matrix else 0

    if is_matrix and n_col == 2 and n_row > 1:
        if n_row != n_states:
            raise ValueError("layout matrix should have one row per state")
        positions = []
        for row in matrix:
            x, y = row
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                raise ValueError("layout coordinates must be between 0 and 1")
            positions.append([x, y])
        return positions, [n_states]

    values = [value for row in matrix for value in row]
    layout_counts = []
    for value in values:
        if value <= 0.0 or not float(value).is_integer():
            raise ValueError("non-integer number of states in layout argument")
        layout_counts.append(int(value))
    if sum(layout_counts) != n_states:
        raise ValueError("number of boxes != number of states")

    positions = [[0.0, 0.0] for _ in range(n_states)]
    group_space = _statefig_space(len(layout_counts))
    state_idx = 0
    column_layout = (not is_matrix) or n_col > 1
    for group_idx, count in enumerate(layout_counts):
        within = _statefig_space(count)
        for offset in range(count):
            if column_layout:
                positions[state_idx] = [group_space[group_idx], 1.0 - within[offset]]
            else:
                positions[state_idx] = [within[offset], 1.0 - group_space[group_idx]]
            state_idx += 1
    return positions, layout_counts


def statefig(
    layout: Any,
    connect: Any,
    states: Any | None = None,
    *,
    margin: Any = 0.03,
    box: Any = True,
    cex: Any = 1,
    col: Any = 1,
    lwd: Any = 1,
    lty: Any = 1,
    bcol: Any | None = None,
    acol: Any | None = None,
    alwd: Any | None = None,
    alty: Any | None = None,
    offset: Any = 0,
) -> dict[str, Any]:
    """Return R ``survival::statefig`` state coordinates from layout/connect inputs."""

    del cex, col, lwd, lty, bcol, acol, alwd, alty
    _finite_float(margin, "margin")
    _finite_float(offset, "offset")
    _normalize_bool_option(box, "box")

    connect_rows = _as_rows(connect, "connect")
    n_states = len(connect_rows)
    if n_states == 0 or any(len(row) != n_states for row in connect_rows):
        raise ValueError("connect must be a square matrix")
    state_names = (
        [str(value) for value in _materialize_1d(states, "states")]
        if states is not None
        else [str(idx + 1) for idx in range(n_states)]
    )
    if len(state_names) != n_states:
        raise ValueError("states must have one entry per connect row")

    positions, layout_counts = _statefig_positions_from_layout(layout, n_states)
    edges = [
        [row_idx, col_idx, int(value)]
        for row_idx, row in enumerate(connect_rows)
        for col_idx, value in enumerate(row)
        if row_idx != col_idx and value != 0.0
    ]
    return {
        "states": state_names,
        "positions": positions,
        "layout": layout_counts,
        "edges": edges,
    }


def _brier_response_from_model_frame(frame: Mapping[str, Any]) -> Surv | None:
    for value in frame.values():
        if isinstance(value, Surv):
            return value
    return None


def _brier_fit_response_and_data(fit: Any, newdata: Any | None) -> tuple[Surv, Any | None]:
    if newdata is not None:
        if not isinstance(fit, _FormulaFit) or fit.formula is None:
            raise ValueError("newdata brier calculations require a formula Cox model")
        response, _terms = _parse_formula(fit.formula, newdata)
        return response, newdata

    if isinstance(fit, _FormulaFit):
        if fit.y_response is not None:
            return fit.y_response, fit.model_frame
        if fit.model_frame is not None:
            response = _brier_response_from_model_frame(fit.model_frame)
            if response is not None:
                return response, fit.model_frame
        raise ValueError("fitted Cox model does not retain its response; refit with y/model data")

    model = _unwrap_formula_fit(fit)
    if not hasattr(model, "event_times") or not hasattr(model, "status"):
        raise ValueError("fitted Cox model does not expose response data")
    return Surv(list(model.event_times), list(model.status)), None


def _brier_case_weights(
    fit: Any,
    model_data: Any | None,
    n: int,
    *,
    use_newdata: bool,
) -> list[float]:
    weight_column = getattr(fit, "case_weight_column", None)
    if use_newdata and weight_column is not None:
        if model_data is None:
            raise ValueError("newdata is required to evaluate case weights")
        try:
            source = _column(model_data, weight_column)
        except KeyError as exc:
            raise ValueError(f"newdata is missing weights column {weight_column!r}") from exc
        weights = _float_vector(source, "weights")
        if len(weights) != n:
            raise ValueError("weights must have the same length as the Surv response")
    else:
        weights = _model_residual_weights(fit, n)
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return weights


def _brier_id_column(data: Any | None, name: str) -> list[Any] | None:
    if data is None:
        return None
    if isinstance(data, Mapping) and name not in data:
        return None
    try:
        return _materialize_labels(_column(data, name), name)
    except KeyError:
        return None


def _brier_id_values(
    fit: Any,
    model_data: Any | None,
    n: int,
    *,
    use_newdata: bool,
) -> list[Any] | None:
    id_column = getattr(fit, "id_column", None)
    if use_newdata and id_column is not None:
        if model_data is None:
            raise ValueError("newdata is required to evaluate id")
        try:
            values = _materialize_labels(_column(model_data, id_column), "id")
        except KeyError as exc:
            raise ValueError(f"newdata is missing id column {id_column!r}") from exc
        if len(values) != n:
            raise ValueError("id must have the same length as the Surv response")
        return values

    for name in ("(id)", "id"):
        values = _brier_id_column(model_data, name)
        if values is not None:
            if len(values) != n:
                raise ValueError("id must have the same length as the Surv response")
            return values

    if isinstance(fit, _FormulaFit) and fit.id_values is not None:
        values = _materialize_labels(fit.id_values, "id")
        if len(values) == n:
            return values
    return None


def _brier_counting_has_gaps_or_overlaps(
    starts: Sequence[float],
    stops: Sequence[float],
    id_values: Sequence[Any],
) -> bool:
    intervals_by_id: dict[Any, list[tuple[float, float]]] = {}
    for start, stop, id_value in zip(starts, stops, id_values, strict=True):
        intervals_by_id.setdefault(_hashable_group_value(id_value), []).append(
            (float(stop), float(start))
        )

    for intervals in intervals_by_id.values():
        previous_stop: float | None = None
        for stop, start in sorted(intervals):
            if start > stop:
                return True
            if previous_stop is not None and start != previous_stop:
                return True
            previous_stop = stop
    return False


def _brier_validate_counting_response(
    starts: Sequence[float],
    stops: Sequence[float],
    status: Sequence[int],
    id_values: Sequence[Any] | None,
) -> None:
    if id_values is None:
        raise ValueError("id is required for start-stop data")
    if len(id_values) != len(stops):
        raise ValueError("id must have the same length as the Surv response")
    if any(value not in (0, 1) for value in status):
        raise ValueError("response must be right censored")
    if _brier_counting_has_gaps_or_overlaps(starts, stops, id_values):
        raise ValueError("one or more flags are >0 in survcheck")
    if not _rttright_counting_common_start(starts, id_values):
        raise NotImplementedError("delayed entry is not yet implemented")


def _brier_event_times(
    response: Surv,
    timefix: bool,
    id_values: Sequence[Any] | None = None,
) -> tuple[list[float], list[int]]:
    if response.type not in {"right", "counting"}:
        raise ValueError("response must be right censored")
    times = [float(value) for value in response.time]
    status = [int(value) for value in response.event]
    if response.start is not None:
        starts = [float(value) for value in response.start]
        if any(not math.isfinite(value) for value in starts):
            raise ValueError("start times must be finite")
        if timefix:
            starts, times = _timefix_vectors(starts, times)
        _brier_validate_counting_response(starts, times, status, id_values)
    elif timefix:
        times = [float(value) for value in _core.aeq_surv(times, None).time]
    return times, status


def _brier_prediction_curves(
    fit: Any,
    prediction_data: Any | None,
) -> tuple[list[float], list[list[float]]]:
    if prediction_data is not None:
        cox_survfit = survfit(fit, newdata=prediction_data, se_fit=False)
        if not isinstance(cox_survfit, CoxSurvfitResult):
            raise TypeError("brier requires Cox survival curves")
        return cox_survfit.time, cox_survfit.surv

    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    rows = _cox_training_rows(model, len(beta))
    return _cox_survival_curve(model, rows, None, True, None)


def _brier_default_times(response: Surv, weights: list[float], efron: bool) -> list[float]:
    baseline = survfit(
        response,
        weights=weights,
        se_fit=False,
        stype=2 if efron else 1,
        ctype=2 if efron else 1,
    )
    if not isinstance(baseline, SurvfitResult):
        raise TypeError("brier baseline curve must be a Kaplan-Meier survfit result")
    return [
        float(time)
        for time, event_count in zip(baseline.time, baseline.n_event, strict=True)
        if float(event_count) > 0.0
    ]


def _brier_censoring_survival(
    dtime: list[float],
    dstat: list[int],
    weights: list[float],
) -> SurvfitResult:
    censor_response = Surv(dtime, [1 - int(value) for value in dstat])
    censor_fit = survfit(censor_response, weights=weights, se_fit=False)
    censor_fit0 = survfit0(censor_fit)
    if not isinstance(censor_fit0, SurvfitResult):
        raise TypeError("brier censoring curve must be a Kaplan-Meier survfit result")
    return censor_fit0


def _brier_apply_ties(dtime: list[float], dstat: list[int], ties: bool) -> list[float]:
    if not ties:
        return list(dtime)
    unique_times = sorted(set(dtime))
    if len(unique_times) < 2:
        return list(dtime)
    mindiff = min(b - a for a, b in zip(unique_times[:-1], unique_times[1:], strict=True))
    return [
        time + mindiff / 2.0 if status == 0 else time
        for time, status in zip(dtime, dstat, strict=True)
    ]


def brier(
    fit: Any,
    times: Any | None = None,
    newdata: Any | None = None,
    ties: Any = True,
    detail: Any = False,
    timefix: Any = True,
    efron: Any = False,
) -> dict[str, Any]:
    """Compute R ``survival::brier`` IPCW Brier scores for Cox model fits."""

    if not _is_coxph_fit(fit):
        raise TypeError("fit must be a coxph object")
    ties_value = _normalize_bool_option(ties, "ties")
    detail_value = _normalize_bool_option(detail, "detail")
    timefix_value = _normalize_bool_option(timefix, "timefix")
    efron_value = _normalize_bool_option(efron, "efron")
    response, prediction_data = _brier_fit_response_and_data(fit, newdata)
    using_newdata = newdata is not None
    id_values = _brier_id_values(
        fit,
        prediction_data,
        len(response),
        use_newdata=using_newdata,
    )
    dtime, dstat = _brier_event_times(response, timefix_value, id_values)
    n = len(dtime)
    weights = _brier_case_weights(
        fit,
        prediction_data,
        n,
        use_newdata=using_newdata,
    )
    eval_times = (
        _float_vector(times, "times")
        if times is not None
        else _brier_default_times(
            response,
            weights,
            efron_value and getattr(_unwrap_formula_fit(fit), "method", None) == "efron",
        )
    )

    baseline = survfit(response, weights=weights, se_fit=False, stype=1)
    if not isinstance(baseline, SurvfitResult):
        raise TypeError("brier baseline curve must be a Kaplan-Meier survfit result")
    p0 = [1.0 - value for value in _step_curve_at(baseline.time, baseline.estimate, eval_times)]

    curve_times, curves = _brier_prediction_curves(fit, prediction_data)
    if len(curves) != n:
        raise ValueError("Cox survival predictions do not match response length")
    phat = [[0.0] * n for _ in eval_times]
    for row_idx, curve in enumerate(curves):
        survival = _step_curve_at(curve_times, [float(value) for value in curve], eval_times)
        for time_idx, value in enumerate(survival):
            phat[time_idx][row_idx] = 1.0 - value

    adjusted_time = _brier_apply_ties(dtime, dstat, ties_value)
    censor_fit = _brier_censoring_survival(adjusted_time, dstat, weights)
    components = _core.perform_brier_calculation(
        adjusted_time,
        dstat,
        weights,
        eval_times,
        p0,
        phat,
        censor_fit.time,
        censor_fit.estimate,
    )

    result: dict[str, Any] = {
        "rsquared": [float(value) for value in components["rsquared"]],
        "brier": [float(value) for value in components["brier"]],
        "times": eval_times,
    }
    if detail_value:
        result["p0"] = p0
        result["phat"] = phat
        result["eff.n"] = [float(value) for value in components["eff_n"]]
    return result


def _quantile_type7(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("x must contain at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    p = min(max(probability, 0.0), 1.0)
    position = p * (len(sorted_values) - 1)
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    weight = position - lower_idx
    return sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight


def _unique_sorted_floats(values: Sequence[float]) -> list[float]:
    return sorted(set(values))


def _validate_nsk_boundary_pair(boundary_knots: tuple[float, float]) -> tuple[float, float]:
    low, high = boundary_knots
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("Boundary.knots must be finite and strictly increasing")
    return boundary_knots


def _normalize_nsk_knots(knots: Any | None) -> list[float] | None:
    if knots is None:
        return None
    knot_values = _normalize_numeric_sequence_or_none(knots, "knots") or []
    return _unique_sorted_floats(knot_values)


def _default_nsk_boundary_knots(x: list[float], b: Any) -> tuple[float, float]:
    b_value = _finite_float(b, "b")
    if b_value < 0.0 or b_value > 1.0:
        raise ValueError("b must be between 0 and 1")
    sorted_x = sorted(x)
    return _validate_nsk_boundary_pair(
        tuple(
            sorted(
                (
                    _quantile_type7(sorted_x, b_value),
                    _quantile_type7(sorted_x, 1.0 - b_value),
                )
            )
        )
    )


def _nsk_boundary_from_knots(knots: list[float] | None) -> tuple[tuple[float, float], list[float]]:
    if knots is None or len(knots) < 2:
        raise ValueError("wrong length for Boundary.knots")
    return _validate_nsk_boundary_pair((knots[0], knots[-1])), knots[1:-1]


def _adjust_nsk_boundary_for_knots(
    boundary_knots: tuple[float, float],
    knots: list[float] | None,
) -> tuple[tuple[float, float], list[float] | None]:
    if not knots:
        return boundary_knots, None

    kept_boundary = [boundary_knots[0], boundary_knots[1]]
    if kept_boundary[1] <= max(knots):
        kept_boundary = kept_boundary[:1]
    if kept_boundary and kept_boundary[0] >= min(knots):
        kept_boundary = kept_boundary[1:]

    all_knots = _unique_sorted_floats([*knots, *kept_boundary])
    if len(all_knots) < 2:
        raise ValueError("at least two distinct finite knots are required")
    return _validate_nsk_boundary_pair((all_knots[0], all_knots[-1])), all_knots[1:-1]


def _pop_nsk_boundary_alias(kwargs: dict[str, Any], current: Any, alias: str) -> Any:
    if alias not in kwargs:
        return current
    value = kwargs.pop(alias)
    if current is not _MISSING:
        raise ValueError(f"use only one of Boundary_knots or {alias}")
    return value


def _normalize_nsk_boundary_knots(
    x: list[float],
    knots: list[float] | None,
    b: Any,
    boundary_arg: Any,
) -> tuple[tuple[float, float], list[float] | None]:
    if boundary_arg is _MISSING:
        boundary_knots = _default_nsk_boundary_knots(x, b)
        return _adjust_nsk_boundary_for_knots(boundary_knots, knots)

    if _is_bool_like(boundary_arg):
        if bool(boundary_arg):
            boundary_knots = _validate_nsk_boundary_pair((min(x), max(x)))
            return _adjust_nsk_boundary_for_knots(boundary_knots, knots)
        return _nsk_boundary_from_knots(knots)

    if boundary_arg is None:
        return _nsk_boundary_from_knots(knots)

    boundary_values = _normalize_numeric_sequence_or_none(boundary_arg, "Boundary.knots") or []
    if len(boundary_values) == 0:
        return _nsk_boundary_from_knots(knots)
    if len(boundary_values) != 2:
        raise ValueError("wrong length for Boundary.knots")

    boundary_knots = _validate_nsk_boundary_pair(tuple(sorted(boundary_values)))
    return _adjust_nsk_boundary_for_knots(boundary_knots, knots)


def _computed_nsk_knots(
    x: list[float],
    boundary_knots: tuple[float, float],
    df: int | None,
    intercept: bool,
) -> list[float] | None:
    minimum_df = 2 if intercept else 1
    effective_df = minimum_df if df is None else df
    if effective_df < minimum_df:
        return None

    n_interior = effective_df - minimum_df
    if n_interior == 0:
        return None

    low, high = boundary_knots
    inside = sorted(value for value in x if low <= value <= high)
    if not inside:
        raise ValueError(
            f"not enough x values inside Boundary.knots to compute {n_interior} interior knots"
        )
    return [_quantile_type7(inside, idx / (n_interior + 1)) for idx in range(1, n_interior + 1)]


def _pspline_difference_penalty(n_cols: int) -> list[list[float]]:
    if n_cols == 0:
        return []
    diff_rows = []
    for row_idx in range(n_cols - 2):
        row = [0.0] * n_cols
        row[row_idx] = 1.0
        row[row_idx + 1] = -2.0
        row[row_idx + 2] = 1.0
        diff_rows.append(row)

    penalty = [[0.0] * n_cols for _ in range(n_cols)]
    for row in diff_rows:
        for i, left in enumerate(row):
            if left == 0.0:
                continue
            for j, right in enumerate(row):
                if right != 0.0:
                    penalty[i][j] += left * right
    return penalty


def _frailty_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _frailty_encoding(
    x: Any,
    *,
    levels: Any | None = None,
    sparse: Any | None = None,
) -> dict[str, Any]:
    values = _materialize_labels(x, "x")
    if levels is None:
        level_values = sorted({str(value) for value in values if not _frailty_missing(value)})
    else:
        level_values = [str(value) for value in _materialize_1d(levels, "levels")]
    level_index = {level: idx + 1 for idx, level in enumerate(level_values)}

    codes: list[int | None] = []
    for value in values:
        if _frailty_missing(value):
            codes.append(None)
        else:
            key = str(value)
            if key not in level_index:
                raise ValueError(f"x contains value {key!r} outside supplied levels")
            codes.append(level_index[key])

    sparse_value = (
        len(level_values) > 5 if sparse is None else _normalize_bool_option(sparse, "sparse")
    )
    return {
        "codes": codes,
        "levels": level_values,
        "nclass": len(level_values),
        "sparse": sparse_value,
    }


def _normalize_pspline_method(
    df: Any,
    theta: Any | None,
    nterm: Any | None,
    method: Any | None,
    eps: Any,
) -> tuple[int | float, float | None, int, float, str]:
    df_value = float(df)
    if not math.isfinite(df_value):
        raise ValueError("df must be finite")
    if df_value.is_integer():
        df_value = int(df_value)

    eps_value = 0.1 if eps is None else _finite_float(eps, "eps")
    nterm_value = None if nterm is None else int(round(_finite_float(nterm, "nterm")))
    if theta is not None:
        theta_value = _finite_float(theta, "theta")
        if theta_value <= 0.0 or theta_value >= 1.0:
            raise ValueError("Invalid value for theta")
        if nterm_value is None:
            nterm_value = int(round(2.5 * float(df_value)))
        return df_value, theta_value, nterm_value, eps_value, "fixed"

    method_value = None if method is None else str(method).lower()
    if float(df_value) == 0.0 or method_value == "aic":
        return df_value, None, 15, 1e-5, "aic"

    if float(df_value) <= 1.0:
        raise ValueError("Too few degrees of freedom")
    if nterm_value is None:
        nterm_value = int(round(2.5 * float(df_value)))
    if float(df_value) > nterm_value:
        raise ValueError(f"`nterm' too small for df={df_value:g}")
    return df_value, None, nterm_value, eps_value, "df"


def _pspline_combine_matrix(
    matrix: list[list[float]],
    combine: Any | None,
    intercept: bool,
) -> tuple[list[list[float]], list[int] | None]:
    if combine is None:
        return matrix, None

    raw_values = [float(value) for value in _materialize_1d(combine, "combine")]
    combine_values = [int(value) for value in raw_values]
    if any(value != math.floor(value) or value < 0.0 for value in raw_values):
        raise ValueError("combine must be an increasing vector of positive integers")
    if any(
        later < earlier for earlier, later in zip(combine_values, combine_values[1:], strict=False)
    ):
        raise ValueError("combine must be an increasing vector of positive integers")

    n_cols = len(matrix[0]) if matrix else 0
    column_groups = combine_values if intercept else [0, *combine_values]
    if len(column_groups) != n_cols:
        raise ValueError("wrong length for combine")

    unique_groups = sorted(set(column_groups))
    group_index = {group: idx for idx, group in enumerate(unique_groups)}
    combined = [[0.0] * len(unique_groups) for _ in matrix]
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            combined[row_idx][group_index[column_groups[col_idx]]] += value
    return combined, combine_values


def pspline(
    x: Any,
    df: Any = 4,
    theta: Any | None = None,
    nterm: Any | None = None,
    degree: Any = 3,
    eps: Any = 0.1,
    method: Any | None = None,
    Boundary_knots: Any | None = None,
    *,
    boundary_knots: Any | None = None,
    intercept: Any = False,
    penalty: Any = True,
    combine: Any | None = None,
) -> dict[str, Any]:
    """Create R-compatible ``survival::pspline`` basis data."""

    if Boundary_knots is not None and boundary_knots is not None:
        raise ValueError("use only one of Boundary_knots or boundary_knots")
    boundary_arg = boundary_knots if boundary_knots is not None else Boundary_knots

    x_values = _float_vector(x, "x")
    finite_x = [value for value in x_values if not math.isnan(value)]
    if not finite_x:
        raise ValueError("x must contain at least one non-missing value")
    if any(math.isinf(value) for value in finite_x):
        raise ValueError("x must contain only finite values")

    degree_value = _integer_scalar(degree, "degree")
    if degree_value < 1:
        raise ValueError("degree must be positive")
    intercept_value = _normalize_bool_option(intercept, "intercept")
    penalty_value = _normalize_bool_option(penalty, "penalty")

    df_value, theta_value, nterm_value, eps_value, method_value = _normalize_pspline_method(
        df,
        theta,
        nterm,
        method,
        eps,
    )
    if nterm_value < 3:
        raise ValueError("Too few basis functions")

    if boundary_arg is None:
        boundary = (min(finite_x), max(finite_x))
    else:
        boundary_values = _float_vector(boundary_arg, "Boundary.knots")
        if len(boundary_values) != 2:
            raise ValueError("Invalid values for Boundary.knots")
        boundary = (boundary_values[0], boundary_values[1])
    if not math.isfinite(boundary[0]) or not math.isfinite(boundary[1]):
        raise ValueError("Invalid values for Boundary.knots")
    if boundary[0] > boundary[1] or (boundary_arg is not None and boundary[0] == boundary[1]):
        raise ValueError("Invalid values for Boundary.knots")

    full_matrix, knots = _core.pspline_basis(
        x_values,
        nterm_value,
        degree_value,
        boundary,
    )

    full_matrix, combine_values = _pspline_combine_matrix(
        full_matrix,
        combine,
        intercept_value,
    )
    dmat = _pspline_difference_penalty(len(full_matrix[0]) if full_matrix else 0)
    if not intercept_value:
        full_matrix = [row[1:] for row in full_matrix]
        dmat = [row[1:] for row in dmat[1:]]

    n_cols = len(full_matrix[0]) if full_matrix else 0
    cbase_length = max(0, n_cols - 1) if intercept_value else n_cols
    return {
        "basis": full_matrix,
        "n_cols": n_cols,
        "nterm": nterm_value,
        "degree": degree_value,
        "df": df_value,
        "theta": theta_value,
        "eps": eps_value,
        "method": method_value,
        "boundary_knots": [boundary[0], boundary[1]],
        "dmat": dmat,
        "combine": combine_values,
        "penalty": penalty_value,
        "intercept": intercept_value,
        "cbase": [knots[idx] + (boundary[0] - knots[0]) for idx in range(1, cbase_length + 1)],
    }


def nsk(
    x: Any,
    df: Any | None = None,
    knots: Any | None = None,
    intercept: Any = False,
    b: Any = 0.05,
    Boundary_knots: Any = _MISSING,
    **kwargs: Any,
) -> Any:
    """Create a Rust-backed natural spline basis with R ``survival::nsk`` arguments."""

    boundary_arg = _pop_nsk_boundary_alias(kwargs, Boundary_knots, "Boundary.knots")
    boundary_arg = _pop_nsk_boundary_alias(kwargs, boundary_arg, "boundary_knots")
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"nsk got an unexpected keyword argument {unexpected!r}")

    x_values = _float_vector(x, "x")
    if not x_values:
        raise ValueError("x must contain at least one value")
    observed_x = [value for value in x_values if not math.isnan(value)]
    if not observed_x:
        raise ValueError("x must contain at least one non-missing value")
    if any(not math.isfinite(value) for value in observed_x):
        raise ValueError("x must contain only finite values")

    intercept_value = _normalize_bool_option(intercept, "intercept")
    df_value: int | None = None
    if df is not None:
        df_value = _integer_scalar(df, "df")
        if df_value <= 0:
            raise ValueError("df must be positive")

    normalized_knots = _normalize_nsk_knots(knots)
    boundary_knots, core_knots = _normalize_nsk_boundary_knots(
        observed_x,
        normalized_knots,
        b,
        boundary_arg,
    )
    if not normalized_knots and core_knots is None:
        core_knots = _computed_nsk_knots(observed_x, boundary_knots, df_value, intercept_value)
    spline = _core.NaturalSplineKnot(core_knots, boundary_knots, df_value, intercept_value)
    return spline.basis(x_values)


def _survobrien_default_transform(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start + 1
        while end < n and values[order[end]] == values[order[start]]:
            end += 1
        rank_sum = sum(range(start + 1, end + 1))
        rank = rank_sum / (end - start)
        for pos in range(start, end):
            ranks[order[pos]] = float(rank)
        start = end
    transformed = []
    for rank in ranks:
        probability = (rank - 0.5) / n
        transformed.append(math.log(probability / (1.0 - probability)))
    return transformed


def _survobrien_transform_values(
    values: Sequence[float],
    transform: Any | None,
) -> list[float]:
    if transform is None:
        return _survobrien_default_transform(values)
    if not callable(transform):
        raise TypeError("transform must be callable")
    raw_result = transform(list(values))
    try:
        result = _materialize_1d(raw_result, "transform")
    except TypeError:
        if len(values) != 1:
            raise
        result = [raw_result]
    if len(result) != len(values):
        raise ValueError("Transform function must be 1 to 1")
    try:
        transformed = [float(value) for value in result]
    except (TypeError, ValueError) as exc:
        raise ValueError("transform must return numeric values") from exc
    if any(not math.isfinite(value) for value in transformed):
        raise ValueError("transform must return finite values")
    return transformed


def _survobrien_term_name(term: _CovariateTerm) -> str:
    return _covariate_term_name(term)


def _survobrien_formula_terms(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[tuple[str, list[Any]]], list[tuple[str, list[float]]]]:
    keepers: list[tuple[str, list[Any]]] = []
    continuous: list[tuple[str, list[float]]] = []
    for term in terms.covariates:
        if isinstance(term, _InteractionTerm):
            raise ValueError("This function cannot deal with interaction terms")
        values = _term_values(data, term, n)
        if term.categorical:
            keepers.append((term.column, values))
            continue
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            keepers.append((_survobrien_term_name(term), values))
            continue
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"formula term {term.column!r} must be finite")
        continuous.append((_survobrien_term_name(term), numeric))
    if not continuous:
        raise ValueError("No continuous variables to modify")
    return keepers, continuous


def _survobrien_event_sets(
    response: Surv,
    strata_values: list[Any] | None,
) -> list[tuple[float, list[int]]]:
    if response.type == "right":
        if strata_values is None:
            event_times = sorted(
                {
                    float(time)
                    for time, event in zip(response.time, response.event, strict=True)
                    if event == 1
                }
            )
            return [
                (
                    event_time,
                    [idx for idx, time in enumerate(response.time) if float(time) >= event_time],
                )
                for event_time in event_times
            ]

        seen: set[tuple[float, Any]] = set()
        result: list[tuple[float, list[int]]] = []
        for event_time, event, stratum in zip(
            response.time,
            response.event,
            strata_values,
            strict=True,
        ):
            if event != 1:
                continue
            key = (float(event_time), _hashable_group_value(stratum))
            if key in seen:
                continue
            seen.add(key)
            result.append(
                (
                    float(event_time),
                    [
                        idx
                        for idx, (row_event, row_stratum) in enumerate(
                            zip(response.event, strata_values, strict=True)
                        )
                        if float(row_event) >= float(event_time) and row_stratum == stratum
                    ],
                )
            )
        return result

    if response.type == "counting":
        if response.start is None:
            raise ValueError("counting Surv response is missing start times")
        if strata_values is None:
            event_times = sorted(
                {
                    float(stop)
                    for stop, event in zip(response.time, response.event, strict=True)
                    if event == 1
                }
            )
            return [
                (
                    event_time,
                    [
                        idx
                        for idx, (start, stop) in enumerate(
                            zip(response.start, response.time, strict=True)
                        )
                        if float(start) < event_time <= float(stop)
                    ],
                )
                for event_time in event_times
            ]

        seen: set[tuple[float, Any]] = set()
        result: list[tuple[float, list[int]]] = []
        for event_time, event, stratum in zip(
            response.time,
            response.event,
            strata_values,
            strict=True,
        ):
            if event != 1:
                continue
            event_time_float = float(event_time)
            stratum_key = _hashable_group_value(stratum)
            key = (event_time_float, stratum_key)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                (
                    event_time_float,
                    [
                        idx
                        for idx, (start, stop, row_stratum) in enumerate(
                            zip(
                                response.start,
                                response.time,
                                strata_values,
                                strict=True,
                            )
                        )
                        if float(start) < event_time_float <= float(stop)
                        and _hashable_group_value(row_stratum) != stratum_key
                    ],
                )
            )
        return result

    raise ValueError("Response must be right censored or (start, stop] data")


def _survobrien_formula_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None,
    na_action: Any | None,
    transform: Any | None,
) -> dict[str, list[Any]]:
    if data is None:
        raise ValueError("survobrien formula requires data")
    if subset is not None:
        data, _aligned = _subset_formula_inputs(formula, data, subset)
    data, _aligned = _apply_formula_na_action(formula, data, na_action)
    response, terms = _parse_formula(formula, data)
    if len(terms.clusters) > 1:
        raise ValueError("Can have only 1 cluster term")
    n = len(response)
    keepers, continuous = _survobrien_formula_terms(data, terms, n)
    strata_values = _combined_columns(data, terms.strata, n) if terms.strata else None
    event_sets = _survobrien_event_sets(response, strata_values)

    row_indices: list[int] = []
    set_numbers: list[int] = []
    event_times: list[float] = []
    for set_idx, (event_time, indices) in enumerate(event_sets, start=1):
        row_indices.extend(indices)
        set_numbers.extend([set_idx] * len(indices))
        event_times.extend([event_time] * len(indices))

    frame: dict[str, list[Any]] = {}
    if response.type == "counting":
        if response.start is None:
            raise ValueError("counting Surv response is missing start times")
        frame["start"] = [float(response.start[idx]) for idx in row_indices]
        frame["stop"] = [float(response.time[idx]) for idx in row_indices]
    else:
        frame["time"] = [float(response.time[idx]) for idx in row_indices]
    frame["status"] = [
        1 if response.event[idx] == 1 and float(response.time[idx]) == event_time else 0
        for idx, event_time in zip(row_indices, event_times, strict=True)
    ]
    for name, values in keepers:
        frame[name] = [values[idx] for idx in row_indices]
    if not terms.clusters:
        frame[".id."] = [idx + 1 for idx in row_indices]

    group_sizes = [len(indices) for _event_time, indices in event_sets]
    if transform is None:
        transformed_columns = _core.survobrien_transform_groups(
            [values for _name, values in continuous],
            row_indices,
            group_sizes,
        )
        for (name, _values), output in zip(
            continuous,
            transformed_columns,
            strict=True,
        ):
            frame[name] = output
    else:
        for name, values in continuous:
            output = [0.0] * len(row_indices)
            offset = 0
            for group_size in group_sizes:
                positions = range(offset, offset + group_size)
                transformed = _survobrien_transform_values(
                    [values[row_indices[pos]] for pos in positions],
                    transform,
                )
                output[offset : offset + group_size] = transformed
                offset += group_size
            frame[name] = output
    frame[".strata."] = set_numbers
    return frame


def survobrien(
    time: Any,
    status: Any | None = None,
    covariate: Any | None = None,
    strata: Any | None = None,
    *,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    transform: Any | None = None,
) -> SurvObrienResult | dict[str, list[Any]]:
    """Run O'Brien's direct statistic or build R-style formula transformed rows."""

    if isinstance(time, str) and "~" in time:
        formula_data = data if data is not None else status
        return _survobrien_formula_frame(
            time,
            formula_data,
            subset=subset,
            na_action=na_action,
            transform=transform,
        )
    if status is None or covariate is None:
        raise TypeError("direct survobrien calls require time, status, and covariate")

    time_values = _float_vector(time, "time")
    strata_groups = None
    if strata is not None:
        strata_values = _materialize_labels(strata, "strata")
        if len(strata_values) != len(time_values):
            raise ValueError("strata length mismatch")
        strata_groups = _encode_groups(strata_values, len(time_values))
    return _core.survobrien(
        time_values,
        _integer_code_vector(status, "status", "0/1 event coding"),
        _float_vector(covariate, "covariate"),
        strata_groups,
    )


def yates(
    predictions: Any,
    factor: Any,
    weights: Any | None = None,
    conf_level: Any | None = None,
) -> YatesResult:
    """Compute direct Yates-style adjusted means from predictions and a factor."""

    prediction_values = _float_vector(predictions, "predictions")
    factor_values = [str(value) for value in _materialize_labels(factor, "factor")]
    weight_values = None if weights is None else _float_vector(weights, "weights")
    confidence = None if conf_level is None else _normalize_conf_level(conf_level)
    return _core.yates(prediction_values, factor_values, weight_values, confidence)


def yates_contrast(
    x: Any,
    coef: Any,
    n_obs: Any,
    n_vars: Any,
    factor_col: Any,
    factor_levels: Any,
    predict_type: str | None = None,
) -> YatesResult:
    """Compute model-based direct Yates contrasts from a flattened design matrix."""

    return _core.yates_contrast(
        _float_vector(x, "x"),
        _float_vector(coef, "coef"),
        _integer_scalar(n_obs, "n_obs"),
        _integer_scalar(n_vars, "n_vars"),
        _integer_scalar(factor_col, "factor_col"),
        _float_vector(factor_levels, "factor_levels"),
        predict_type,
    )


def yates_pairwise(result: YatesResult) -> YatesPairwiseResult:
    """Compute pairwise differences from a direct Yates result."""

    return _core.yates_pairwise(result)


def _cipoisson_count(value: Any) -> float | None:
    if _is_missing_value(value):
        return None
    count = float(value)
    if count < 0:
        raise ValueError("k must be non-negative")
    return count


def _cipoisson_float(value: Any, name: str) -> float | None:
    if _is_missing_value(value):
        return None
    return float(value)


def cipoisson(
    k: Any,
    time: Any = 1.0,
    p: Any = 0.95,
    method: Any = "exact",
) -> tuple[float, float] | list[tuple[float, float]]:
    """Return Poisson rate confidence intervals, like R's ``cipoisson``."""

    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method_value = method.strip().lower()
    k_values = _scalar_or_vector(k, "k")
    time_values = _scalar_or_vector(time, "time")
    p_values = _scalar_or_vector(p, "p")
    n = max(len(k_values), len(time_values), len(p_values))
    if n == 0:
        return []

    k_values = _recycle_r_vector(k_values, n, "k")
    time_values = _recycle_r_vector(time_values, n, "time")
    p_values = _recycle_r_vector(p_values, n, "p")
    if not k_values or not time_values or not p_values:
        return []

    intervals: list[tuple[float, float]] = []
    for raw_k, raw_time, raw_p in zip(k_values, time_values, p_values, strict=True):
        count = _cipoisson_count(raw_k)
        exposure = _cipoisson_float(raw_time, "time")
        confidence = _cipoisson_float(raw_p, "p")
        if count is None or exposure is None or exposure <= 0.0:
            intervals.append((math.nan, math.nan))
            continue
        if confidence is None:
            intervals.append(
                (0.0, math.nan) if method_value == "exact" and count == 0 else (math.nan, math.nan)
            )
            continue
        lower, upper = _core.cipoisson(count, exposure, confidence, method_value)
        intervals.append((float(lower), float(upper)))

    return intervals[0] if n == 1 else intervals


def _bounded_link_transform(x: Any, edge: Any, method_name: str) -> float | list[float]:
    edge_value = _finite_float(edge, "edge")
    values, is_scalar = _scalar_or_vector_with_flag(x, "x")
    link = _core.LinkFunctionParams(edge_value)
    transform = getattr(link, f"{method_name}_many")
    prepared = [None if _is_missing_value(value) else float(value) for value in values]
    result = [float(value) for value in transform(prepared)]
    return result[0] if is_scalar else result


def blogit(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded logit link transform."""

    return _bounded_link_transform(x, edge, "blogit")


def bprobit(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded probit link transform."""

    return _bounded_link_transform(x, edge, "bprobit")


def bcloglog(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded complementary log-log link transform."""

    return _bounded_link_transform(x, edge, "bcloglog")


def blog(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded log link transform."""

    return _bounded_link_transform(x, edge, "blog")


def _survcheck_old_style_call(
    id_values: Any,
    time1: Any,
    time2: Any,
    status: Any,
    istate: Any | None,
):
    return _core.survcheck(
        _survcheck_integer_labels(id_values, "id"),
        _float_vector(time1, "time1"),
        _float_vector(time2, "time2"),
        _int_vector(status, "status"),
        None if istate is None else _survcheck_integer_labels(istate, "istate"),
    )


def _survcheck_response_from_formula(
    formula: str,
    data: Any,
    subset: Any | None,
    na_action: str | None,
    id_values: Any | None,
    istate: Any | None,
) -> tuple[Surv, Any | None, Any | None]:
    if data is None:
        raise ValueError("survcheck formula requires data")
    id_values = _column_or_values(data, id_values, "id") if id_values is not None else None
    istate = _column_or_values(data, istate, "istate") if istate is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            id=id_values,
            istate=istate,
        )
        id_values = aligned["id"]
        istate = aligned["istate"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        id=id_values,
        istate=istate,
    )
    response, _terms = _parse_formula(formula, data)
    return response, aligned["id"], aligned["istate"]


def survcheck(
    response: Any = _MISSING,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "pass",
    id: Any | None = None,
    istate: Any | None = None,
    istate0: str = "(s0)",
    timefix: bool = True,
    *,
    time1: Any = _MISSING,
    time2: Any = _MISSING,
    status: Any = _MISSING,
    **kwargs: Any,
):
    """Check survival response consistency, like R's ``survcheck`` for common inputs."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survcheck got unexpected keyword argument(s): {unexpected}")

    if response is _MISSING:
        if time1 is _MISSING or time2 is _MISSING or status is _MISSING or id is None:
            raise TypeError(
                "survcheck requires a Surv response/formula or low-level "
                "id=, time1=, time2=, and status= vectors"
            )
        return _survcheck_old_style_call(id, time1, time2, status, istate)
    if time1 is not _MISSING or time2 is not _MISSING or status is not _MISSING:
        raise TypeError("time1, time2, and status are only valid for low-level survcheck calls")

    if not isinstance(response, Surv | str):
        if data is None or subset is None or na_action is None:
            raise TypeError(
                "survcheck requires a Surv response/formula or low-level "
                "(id, time1, time2, status) vectors"
            )
        return _survcheck_old_style_call(response, data, subset, na_action, id)

    if not isinstance(timefix, bool):
        raise TypeError("timefix must be True or False")
    if not isinstance(istate0, str):
        raise TypeError("istate0 must be a string")

    id_values = id
    if isinstance(response, str):
        response, id_values, istate = _survcheck_response_from_formula(
            response,
            data,
            subset,
            _normalize_na_action(na_action),
            id_values,
            istate,
        )
        subset = None
        na_action = "pass"
    elif subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        id_values = _subset_optional_sequence(id_values, indices, "id")
        istate = _subset_optional_sequence(istate, indices, "istate")
        subset = None

    response, aligned = _apply_surv_na_action(
        response,
        _normalize_na_action(na_action),
        "survcheck inputs",
        id=id_values,
        istate=istate,
    )
    id_values = aligned["id"]
    istate = aligned["istate"]

    if response.type == "right":
        times = list(response.time)
        if timefix:
            times = _survdiff_timefix_values(times, True)
        return _core.survcheck_simple(times, list(response.event))
    if response.type not in {"counting", "mright", "mcounting"}:
        raise ValueError(f"survcheck is not valid for {response.type} censored survival data")
    if response.type in {"counting", "mcounting"} and response.start is None:
        raise ValueError("counting Surv response is missing start times")
    if id_values is None:
        raise ValueError("an id argument is required")
    if len(_materialize_labels(id_values, "id")) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if istate is not None and len(_materialize_labels(istate, "istate")) != len(response):
        raise ValueError("istate must have the same length as the Surv response")

    start = [0.0] * len(response) if response.start is None else list(response.start)
    stop = list(response.time)
    if timefix:
        start, stop = _timefix_vectors(start, stop)
    status_values = list(response.event)
    initial_codes: list[int] | None
    if response.type in {"mright", "mcounting"} and istate is not None:
        initial_labels = _materialize_labels(istate, "istate")
        state_names = list(dict.fromkeys([*map(str, initial_labels), *response.states]))
        state_index = {name: index + 1 for index, name in enumerate(state_names)}
        initial_codes = [state_index[str(value)] for value in initial_labels]
        status_values = [
            0 if value == 0 else state_index[response.states[int(value) - 1]]
            for value in response.event
        ]
    else:
        initial_codes = None if istate is None else _survcheck_integer_labels(istate, "istate")
    return _core.survcheck(
        _survcheck_integer_labels(id_values, "id"),
        start,
        stop,
        status_values,
        initial_codes,
    )


def _sample_variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    return math.fsum((value - mean) ** 2 for value in values) / (n - 1)


def _rank_average(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[indexed[pos][0]] = rank
        idx = end
    return ranks


def _rank_first(values: Sequence[float]) -> list[int]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0] * len(values)
    for rank, (original_idx, _value) in enumerate(indexed, start=1):
        ranks[original_idx] = rank
    return ranks


def _royston_normal_scores(eta: Sequence[float], ties: bool) -> list[float]:
    n = len(eta)
    normal = NormalDist()
    if ties and len(set(eta)) != n:
        z = [normal.inv_cdf((rank - 0.375) / (n + 0.25)) for rank in range(1, n + 1)]
        rank_first = _rank_first(eta)
        grouped: dict[float, list[float]] = {value: [] for value in sorted(set(eta))}
        for value, rank in zip(eta, rank_first, strict=True):
            grouped[value].append(z[rank - 1])
        means = {value: math.fsum(scores) / len(scores) for value, scores in grouped.items()}
        return [means[value] for value in eta]

    return [normal.inv_cdf((rank - 0.375) / (n + 0.25)) for rank in _rank_average(eta)]


def _royston_gonen_heller(eta: Sequence[float]) -> float:
    if len(eta) < 2:
        return math.nan
    ordered = sorted(eta)
    total = 0.0
    for idx, value in enumerate(ordered[:-1]):
        total += math.fsum(1.0 / (1.0 + math.exp(value - later)) for later in ordered[idx + 1 :])
    return total * 2.0 / (len(ordered) * (len(ordered) - 1))


def _royston_response_for_fit(fit: Any, newdata: Any | None) -> Surv:
    if newdata is None:
        response = getattr(fit, "y", None)
        return response if isinstance(response, Surv) else _cox_training_response(fit)
    design = _formula_design_for_fit(fit)
    if design is None:
        raise ValueError("newdata royston predictions require a formula Cox model")
    return _surv_from_formula_design(newdata, design)


def royston(
    fit: Any,
    newdata: Any | None = None,
    ties: Any = True,
    adjust: Any = False,
) -> dict[str, float]:
    """R-compatible ``survival::royston`` statistics for fitted Cox models."""

    if not _is_coxph_fit(fit):
        raise TypeError("function defined only for coxph models")
    ties_value = _normalize_bool_option(ties, "ties")
    adjust_value = _normalize_bool_option(adjust, "adjust")
    response = _royston_response_for_fit(fit, newdata)
    if response.type not in {"right", "counting"}:
        raise ValueError("royston requires a right-censored or counting-process response")

    eta = [float(value) for value in predict(fit, newdata, type="lp")]
    if newdata is not None:
        preliminary = coxph(response, x=[[value] for value in eta])
        eta = [float(value) for value in predict(preliminary, type="lp")]

    n = len(eta)
    if n != len(response):
        raise ValueError("linear predictor length must match response length")
    if n < 2:
        raise ValueError("at least two observations are required")

    qhat = _royston_normal_scores(eta, ties_value)
    rfit = coxph(response, x=[[value] for value in qhat])
    beta_values = _cox_beta(rfit)
    if len(beta_values) != 1:
        raise ValueError("internal royston Cox fit did not return one coefficient")
    beta = beta_values[0]
    variance = _cox_variance_matrix(_unwrap_formula_fit(rfit), 1)[0][0]

    pi = math.pi
    d_value = beta * math.sqrt(8.0 / pi)
    se_d = math.sqrt(max(variance, 0.0) * 8.0 / pi)
    r_d = beta * beta / (pi * pi / 6.0 + beta * beta)
    r_i = beta * beta / (1.0 + beta * beta)

    if adjust_value:
        n_events = sum(1 for value in response.event if value == 1)
        n_coef = len(_cox_beta(fit))
        if n_events <= n_coef:
            raise ValueError("adjusted royston statistic requires events > model coefficients")
        ratio = n_events / (n_events - n_coef)
        temp = (1.0 + beta * beta - ratio) / ratio
        d_value = math.copysign(math.sqrt(abs(temp) * 8.0 / pi), beta * temp)
        se_d = se_d * abs(beta) / (ratio * math.sqrt(abs(temp))) if temp != 0.0 else math.inf
        r_d = 1.0 - ratio * (1.0 - r_i)

    eta_variance = _sample_variance(eta)
    result = {
        "D": d_value,
        "se(D)": se_d,
        "R.D": r_d,
        "R.KO": eta_variance / (pi * pi / 6.0 + eta_variance),
        "C.GH": _royston_gonen_heller(eta),
    }
    if newdata is None:
        loglik_values = _cox_loglik_values(_unwrap_formula_fit(fit))
        logtest = -2.0 * (loglik_values[0] - loglik_values[1])
        denominator = 1.0 - math.exp(2.0 * loglik_values[0] / n)
        result["R.N"] = (
            (1.0 - math.exp(-logtest / n)) / denominator if denominator != 0.0 else math.nan
        )
        return {
            "D": result["D"],
            "se(D)": result["se(D)"],
            "R.D": result["R.D"],
            "R.KO": result["R.KO"],
            "R.N": result["R.N"],
            "C.GH": result["C.GH"],
        }
    return result
