"""Data utilities: tmerge, survSplit, survcondense, neardate, tcut, aeqSurv, lvcf, rttright."""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real
from operator import index
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _encode_labels,
    _finite_float,
    _float_vector,
    _hashable_group_value,
    _int_vector,
    _integer_code_vector,
    _integer_scalar,
    _is_missing_value,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _mstate_categories,
    _normalize_bool_option_with_default,
    _normalize_na_action,
    _normalize_positive_scale,
    _pop_dotted_keyword,
    _scalar_or_vector,
    _strata_value_label,
    _subset_indices,
    _subset_optional_sequence,
    _survcheck_integer_labels,
    _timefix_vectors,
)
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_or_values,
    _column_source,
    _combined_formula_groups,
    _covariate_term_name,
    _data_column_names,
    _numeric_term_values,
    _parse_formula,
    _reject_formula_clusters,
    _subset_formula_inputs,
    _term_raw_values,
    _term_values,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._types import (
    TcutResult,
    TMergeFrame,
    TMergeOperation,
    _CovariateSpec,
    _FormulaModelTerm,
    _FormulaTerms,
    _InteractionTerm,
    _ModelCovariateTerm,
    _ModelOffsetTerm,
    _ModelStrataTerm,
)


def _lvcf_order_key(value: Any) -> tuple[int, Any]:
    if _is_missing_value(value):
        raise ValueError("id must not contain missing values")
    if isinstance(value, Real):
        return (0, float(value))
    return (1, str(value))


def _lvcf_category_ranks(values: Any, name: str) -> dict[Any, int] | None:
    categories = _mstate_categories(values)
    if categories is None:
        return None
    levels = _materialize_1d(categories, f"{name} categories")
    return {_hashable_group_value(level): rank for rank, level in enumerate(levels)}


def _lvcf_id_ranks(values: Any, raw: list[Any] | None = None) -> tuple[list[Any], list[int]]:
    if raw is None:
        raw = _materialize_labels(values, "id")
    if any(_is_missing_value(value) for value in raw):
        raise ValueError("id must not contain missing values")

    category_ranks = _lvcf_category_ranks(values, "id")
    if category_ranks is not None:
        try:
            return raw, [category_ranks[_hashable_group_value(value)] for value in raw]
        except KeyError as exc:
            raise ValueError("id contains a value outside the declared categories") from exc

    unique: dict[Any, Any] = {}
    for value in raw:
        unique.setdefault(_hashable_group_value(value), value)
    ordered = sorted(unique, key=lambda key: _lvcf_order_key(unique[key]))
    ranks = {key: rank for rank, key in enumerate(ordered)}
    return raw, [ranks[_hashable_group_value(value)] for value in raw]


def _lvcf_time_order(values: Any) -> tuple[list[Any], list[Any]]:
    raw = _lvcf_vector(values, "time")
    category_ranks = _lvcf_category_ranks(values, "time")
    if category_ranks is not None:
        try:
            return raw, [
                len(category_ranks)
                if _is_missing_value(value)
                else category_ranks[_hashable_group_value(value)]
                for value in raw
            ]
        except KeyError as exc:
            raise ValueError("time contains a value outside the declared categories") from exc

    first_observed = next((value for value in raw if not _is_missing_value(value)), None)
    if isinstance(first_observed, Real):
        try:
            numeric = [math.nan if value is None else float(value) for value in raw]
        except (TypeError, ValueError):
            pass
        else:
            return raw, numeric

    return raw, [
        (2, "")
        if _is_missing_value(value)
        else (0, float(value))
        if isinstance(value, Real)
        else (1, str(value))
        for value in raw
    ]


def _lvcf_time_ranks(values: list[Any]) -> list[float]:
    if all(isinstance(value, Real) for value in values):
        return [float(value) for value in values]
    levels = {value: rank for rank, value in enumerate(sorted(set(values)))}
    return [float(levels[value]) for value in values]


def _lvcf_numeric_ids_are_exact(values: list[Any]) -> bool:
    for value in values:
        if not isinstance(value, Real):
            return False
        try:
            exact_integer = index(value)
        except TypeError:
            pass
        else:
            if abs(exact_integer) > 1 << 53:
                return False
    return True


def _lvcf_vector(values: Any, name: str, *, labels: bool = False) -> list[Any]:
    if values is None:
        return [None]
    try:
        return _materialize_labels(values, name) if labels else _materialize_1d(values, name)
    except TypeError:
        return [values]


def _integerish_vector_or_none(values: Any, name: str) -> list[int] | None:
    try:
        return _integer_code_vector(values, name, "integer id values")
    except (TypeError, ValueError):
        return None


def _neardate_float_vector(values: Any, name: str) -> list[float]:
    result: list[float] = []
    for value in _materialize_1d(values, name):
        if _is_missing_value(value):
            result.append(math.nan)
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be sortable numeric values") from exc
    return result


def neardate(
    id1: Any,
    id2: Any,
    y1: Any,
    y2: Any,
    best: Any = "after",
    nomatch: Any | None = None,
) -> list[int | None]:
    """Find nearest matching dates by id, returning R-style 1-based indices."""

    best_value = _match_string_arg(
        best,
        "best",
        ("after", "prior", "closest"),
        "best must be 'after', 'prior', or 'closest'",
    )
    y1_values = _neardate_float_vector(y1, "y1")
    y2_values = _neardate_float_vector(y2, "y2")
    id1_values = _materialize_1d(id1, "id1")
    id2_values = _materialize_1d(id2, "id2")
    id1_missing = [_is_missing_value(value) for value in id1_values]
    id1_integer = _integerish_vector_or_none(id1_values, "id1")
    id2_integer = _integerish_vector_or_none(id2_values, "id2")
    nomatch_value = None if nomatch is None else _integer_scalar(nomatch, "nomatch")

    if id1_integer is not None and id2_integer is not None:
        result = _core.neardate(
            id1_integer,
            y1_values,
            id2_integer,
            y2_values,
            best_value,
            None,
        )
    else:
        result = _core.neardate_str(
            [
                "\0missing" if missing else f"\0value:{value}"
                for value, missing in zip(id1_values, id1_missing, strict=True)
            ],
            y1_values,
            [
                "\0missing" if _is_missing_value(value) else f"\0value:{value}"
                for value in id2_values
            ],
            y2_values,
            best_value,
            None,
        )

    return [
        None if id1_missing[pos] else nomatch_value if idx is None else int(idx) + 1
        for pos, idx in enumerate(result.indices)
    ]


def _tcut_default_labels(breaks: list[float]) -> list[str]:
    return [f"{breaks[idx]:g}+ thru {breaks[idx + 1]:g}" for idx in range(len(breaks) - 1)]


def tcut(
    x: Any,
    breaks: Any,
    labels: Any | None = None,
    scale: Any = 1,
) -> TcutResult:
    """Create a Rust-backed R-style ``tcut`` interval result."""

    x_values = [
        math.nan if _is_missing_value(value) else float(value) for value in _materialize_1d(x, "x")
    ]
    break_values = [
        math.nan if _is_missing_value(value) else float(value)
        for value in _scalar_or_vector(breaks, "breaks")
    ]
    if not break_values:
        raise ValueError("breaks must have at least 1 element")
    if any(math.isnan(value) for value in break_values):
        raise ValueError("breaks must be given in ascending order and contain no NA's")
    if len(break_values) == 1:
        label_values = (
            None if labels is None else [str(value) for value in _materialize_1d(labels, "labels")]
        )
        return _core.tcut(
            [value * _normalize_positive_scale(scale) for value in x_values],
            break_values[0],
            label_values,
        )
    if any(
        later < earlier for earlier, later in zip(break_values[:-1], break_values[1:], strict=True)
    ):
        raise ValueError("breaks must be given in ascending order and contain no NA's")
    scale_value = _normalize_positive_scale(scale)
    label_values = (
        _tcut_default_labels(break_values)
        if labels is None
        else [str(value) for value in _materialize_1d(labels, "labels")]
    )
    if len(label_values) != len(break_values) - 1:
        raise ValueError("labels length must equal length(breaks) - 1")
    return _core.tcut(
        [value * scale_value for value in x_values],
        [value * scale_value for value in break_values],
        label_values,
    )


def lvcf(id: Any, x: Any, time: Any | None = None) -> list[Any]:
    """Carry the last non-missing value forward within each id, like R's ``lvcf``."""

    id_values = _lvcf_vector(id, "id", labels=True)
    result = _lvcf_vector(x, "x")
    if len(result) != len(id_values):
        raise ValueError("x must have the same length as id")
    if any(_is_missing_value(value) for value in id_values):
        raise ValueError("id must not contain missing values")
    missing = [_is_missing_value(value) for value in result]
    numeric_ids_are_exact = _lvcf_numeric_ids_are_exact(id_values)
    if time is None:
        source = (
            _core.lvcf_numeric_indices(id_values, missing)
            if numeric_ids_are_exact
            else _core.lvcf_indices(_lvcf_id_ranks(id, id_values)[1], missing)
        )
        return [result[row_idx] for row_idx in source]

    time_values, time_order = _lvcf_time_order(time)
    if len(time_values) != len(id_values):
        raise ValueError("time must have the same length as id")
    time_ranks = _lvcf_time_ranks(time_order)
    source = (
        _core.lvcf_numeric_indices(id_values, missing, time_ranks)
        if numeric_ids_are_exact
        else _core.lvcf_indices(_lvcf_id_ranks(id, id_values)[1], missing, time_ranks)
    )
    return [result[row_idx] for row_idx in source]


def nostutter(
    id: Any,
    x: Any,
    censor: Any = 0,
    single: bool = False,
) -> list[Any]:
    """Replace repeated adjacent states within each id by the censor value."""

    id_values = _materialize_labels(id, "id")
    result = _materialize_1d(x, "x")
    if len(result) != len(id_values):
        raise ValueError("x must have the same length as id")
    if any(_is_missing_value(value) for value in id_values):
        raise ValueError("id must not contain missing values")

    if not result:
        return []

    id_sample = id_values[0]
    state_sample = next((value for value in result if not _is_missing_value(value)), censor)
    replacements: list[bool] | None = None
    try:
        if isinstance(id_sample, Real):
            if isinstance(state_sample, Real) and isinstance(censor, Real):
                replacements = list(
                    _core.nostutter_numeric_numeric(id_values, result, censor, single)
                )
            if isinstance(state_sample, str) and isinstance(censor, str):
                replacements = list(_core.nostutter_numeric_str(id_values, result, censor, single))
        elif isinstance(id_sample, str):
            if isinstance(state_sample, Real) and isinstance(censor, Real):
                replacements = list(_core.nostutter_str_numeric(id_values, result, censor, single))
            if isinstance(state_sample, str) and isinstance(censor, str):
                replacements = list(_core.nostutter_str_str(id_values, result, censor, single))
    except TypeError:
        pass

    if replacements is not None:
        return [
            censor if replace else value
            for replace, value in zip(replacements, result, strict=True)
        ]

    censor_key = _hashable_group_value(censor)
    id_levels: dict[Any, int] = {}
    id_codes: list[int] = []
    for value in id_values:
        key = _hashable_group_value(value)
        id_codes.append(id_levels.setdefault(key, len(id_levels)))

    state_levels = {censor_key: 0}
    state_codes: list[int | None] = []
    for value in result:
        if _is_missing_value(value):
            state_codes.append(None)
            continue
        key = _hashable_group_value(value)
        state_codes.append(state_levels.setdefault(key, len(state_levels)))

    replacements = _core.nostutter_replacements(id_codes, state_codes, 0, single)
    return [
        censor if replace else value for replace, value in zip(replacements, result, strict=True)
    ]


def _survcondense_legacy_call(
    id_values: Any,
    time1: Any,
    time2: Any,
    status: Any,
) -> Any:
    return _core.survcondense(
        _int_vector(id_values, "id"),
        _float_vector(time1, "time1"),
        _float_vector(time2, "time2"),
        _int_vector(status, "status"),
    )


def _survcondense_term_name(term: _CovariateSpec) -> str:
    if isinstance(term, _InteractionTerm):
        return ":".join(_survcondense_term_name(factor) for factor in term.factors)
    return _covariate_term_name(term)


def _survcondense_strata_name(columns: Sequence[str]) -> str:
    return f"strata({', '.join(columns)})"


def _survcondense_strata_values(data: Any, columns: Sequence[str], n: int) -> list[Any]:
    values = [_column(data, column) for column in columns]
    if any(len(column) != n for column in values):
        raise ValueError("formula columns must have the same length as the Surv response")
    if len(values) == 1:
        return list(values[0])
    return [
        ", ".join(_strata_value_label(column[row_idx]) for column in values) for row_idx in range(n)
    ]


def _survcondense_model_columns(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> list[tuple[str, list[Any]]]:
    columns: list[tuple[str, list[Any]]] = []
    model_terms: Sequence[_FormulaModelTerm]
    model_terms = terms.model_terms or [_ModelCovariateTerm(term) for term in terms.covariates]
    for model_term in model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            columns.append(
                (
                    _survcondense_term_name(model_term.term),
                    _term_values(data, model_term.term, n),
                )
            )
        elif isinstance(model_term, _ModelStrataTerm):
            columns.append(
                (
                    _survcondense_strata_name(model_term.columns),
                    _survcondense_strata_values(data, model_term.columns, n),
                )
            )
        elif isinstance(model_term, _ModelOffsetTerm):
            name = f"offset({_survcondense_term_name(model_term.term)})"
            values = _numeric_term_values(
                _term_raw_values(data, model_term.term, n),
                model_term.term,
            )
            columns.append((name, values))
        else:
            continue
    return columns


def _survcondense_order_key(value: Any) -> tuple[int, int, Any]:
    if _is_missing_value(value):
        return (1, 0, "")
    if isinstance(value, bool):
        return (0, 0, int(value))
    if isinstance(value, int | float):
        numeric = float(value)
        if math.isfinite(numeric):
            return (0, 0, numeric)
    return (0, 1, str(value))


def _survcondense_id_order_codes(values: Sequence[Any]) -> list[int]:
    if all(
        isinstance(value, bool | int | float) and math.isfinite(float(value)) for value in values
    ):
        keys: list[Any] = [float(value) for value in values]
    elif all(isinstance(value, str) for value in values):
        keys = list(values)
    else:
        keys = [_survcondense_order_key(value) for value in values]
    levels = {value: idx for idx, value in enumerate(sorted(set(keys)))}
    return [levels[value] for value in keys]


def _survcondense_compress_rows(columns: Sequence[Sequence[Any]]) -> list[int]:
    normalized: list[Sequence[Any]] = []
    for column in columns:
        if any(_is_missing_value(value) for value in column):
            normalized.append(
                [
                    (0, None) if _is_missing_value(value) else (1, _hashable_group_value(value))
                    for value in column
                ]
            )
        else:
            normalized.append(column)

    levels: dict[Any, int] = {}
    codes: list[int] = []
    try:
        for signature in zip(*normalized, strict=True):
            code = levels.get(signature)
            if code is None:
                code = len(levels) + 1
                levels[signature] = code
            codes.append(code)
        return codes
    except TypeError:
        levels.clear()
        codes.clear()
        robust_columns = [
            [
                (0, None) if _is_missing_value(value) else (1, _hashable_group_value(value))
                for value in column
            ]
            for column in columns
        ]
        for signature in zip(*robust_columns, strict=True):
            code = levels.get(signature)
            if code is None:
                code = len(levels) + 1
                levels[signature] = code
            codes.append(code)
        return codes


def _survcondense_unique_columns(
    columns: Sequence[tuple[str, list[Any]]],
) -> list[tuple[str, list[Any]]]:
    seen: set[str] = set()
    unique: list[tuple[str, list[Any]]] = []
    for name, values in columns:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, values))
    return unique


def _survcondense_from_formula(
    formula: str,
    data: Any,
    subset: Any | None,
    na_action: str | None,
    id_values: Any,
    weights: Any | None,
    start: str,
    end: str,
    event: str,
    id_name: str | None = None,
    weights_name: str | None = None,
) -> dict[str, list[Any]]:
    if data is None:
        raise ValueError("survcondense formula requires data")
    if id_values is None:
        raise ValueError("survcondense requires an id argument")
    if not isinstance(start, str) or not start:
        raise ValueError("start must be a non-empty string")
    if not isinstance(end, str) or not end:
        raise ValueError("end must be a non-empty string")
    if not isinstance(event, str) or not event:
        raise ValueError("event must be a non-empty string")

    id_output_name = id_name or (id_values if isinstance(id_values, str) else "id")
    weights_output_name = weights_name or (weights if isinstance(weights, str) else "(weights)")
    id_values = _column_or_values(data, id_values, "id")
    weights = _column_or_values(data, weights, "weights") if weights is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            id=id_values,
            weights=weights,
        )
        id_values = aligned["id"]
        weights = aligned["weights"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        id=id_values,
        weights=weights,
    )
    id_values = _materialize_1d(aligned["id"], "id")
    weights = aligned["weights"]
    response, terms = _parse_formula(formula, data)
    _reject_formula_clusters("survcondense", terms)
    if response.type not in {"counting", "mcounting"}:
        raise ValueError("survcondense requires a counting-process Surv response")
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    if len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if weights is not None:
        weights = _materialize_1d(weights, "weights")
        if len(weights) != len(response):
            raise ValueError("weights must have the same length as the Surv response")

    model_columns = _survcondense_model_columns(data, terms, len(response))

    comparison_columns = [values for _name, values in model_columns]
    if weights is not None:
        comparison_columns.append(weights)
    comparison_columns.append(id_values)

    plan = _core.survcondense_plan(
        _survcondense_id_order_codes(id_values),
        list(response.start),
        list(response.time),
        _survcondense_compress_rows(comparison_columns),
    )
    starts = [float(value) for value in plan.start]
    keep_indices = [int(value) for value in plan.keep]

    if weights is not None:
        model_columns.append((str(weights_output_name), weights))
    model_columns.append((str(id_output_name), id_values))

    output: dict[str, list[Any]] = {
        name: [values[idx] for idx in keep_indices]
        for name, values in _survcondense_unique_columns(model_columns)
    }
    output[start] = [starts[idx] for idx in keep_indices]
    output[end] = [response.time[idx] for idx in keep_indices]
    if response.type == "mcounting":
        output[event] = [
            "censor" if response.event[idx] == 0 else response.states[int(response.event[idx]) - 1]
            for idx in keep_indices
        ]
    else:
        output[event] = [response.event[idx] for idx in keep_indices]
    return output


def survcondense(
    formula: Any,
    data: Any | None = None,
    subset: Any | None = None,
    weights: Any | None = None,
    na_action: Any | None = "pass",
    *,
    id: Any | None = None,
    start: str = "tstart",
    end: str = "tstop",
    event: str = "event",
    **kwargs: Any,
) -> Any:
    """Condense counting-process survival data, preserving the legacy vector API."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    id_name = kwargs.pop("_id_name", None)
    weights_name = kwargs.pop("_weights_name", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survcondense got unexpected keyword argument(s): {unexpected}")

    if not isinstance(formula, str):
        if data is None or subset is None or weights is None or id is not None:
            raise TypeError(
                "survcondense requires a formula/data/id call or legacy "
                "(id, time1, time2, status) vectors"
            )
        return _survcondense_legacy_call(formula, data, subset, weights)

    return _survcondense_from_formula(
        formula,
        data,
        subset,
        _normalize_na_action(na_action),
        id,
        weights,
        start,
        end,
        event,
        id_name,
        weights_name,
    )


def _rttright_original_order_weights(result: Any, n: int) -> list[float]:
    weights = [0.0] * n
    for sorted_pos, original_idx in enumerate(result.order):
        weights[int(original_idx)] = float(result.weights[sorted_pos])
    return weights


def _rttright_formula_groups(data: Any, terms: _FormulaTerms, n: int) -> list[int] | None:
    if not terms.strata and not terms.covariates:
        return None
    labels = _combined_formula_groups(data, terms.strata, terms.covariates, n)
    return _encode_labels(labels, "rttright strata")


def _rttright_response_from_formula(
    formula: str,
    data: Any,
    subset: Any | None,
    na_action: str | None,
    weights: Any | None,
    id: Any | None,
    *,
    warn_offset: bool = True,
) -> tuple[Surv, Any | None, Any | None, list[int] | None]:
    if data is None:
        raise ValueError("rttright formula requires data")
    weights = _column_or_values(data, weights, "weights") if weights is not None else None
    id_values = _column_or_values(data, id, "id") if id is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            weights=weights,
            id=id_values,
        )
        weights = aligned["weights"]
        id_values = aligned["id"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        weights=weights,
        id=id_values,
    )
    response, terms = _parse_formula(formula, data)
    if terms.offsets and warn_offset:
        warnings.warn("Offset term ignored", RuntimeWarning, stacklevel=2)
    return (
        response,
        aligned["weights"],
        aligned["id"],
        _rttright_formula_groups(data, terms, len(response)),
    )


def _rttright_initial_weights(weights: Any | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    result = _float_vector(weights, "weights")
    if len(result) != n:
        raise ValueError("weights must have same length as time")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("weights must be finite")
    if any(value < 0.0 for value in result):
        raise ValueError("weights must be non-negative")
    return result


def _rttright_times_vector(times: Any) -> list[float]:
    if isinstance(times, str | bytes):
        raise TypeError("times must be numeric")
    try:
        result = [float(times)]
    except (TypeError, ValueError):
        result = _float_vector(times, "times")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("times must be finite")
    return result


def _rttright_validate_id(response: Surv, id: Any | None) -> list[Any] | None:
    if id is None:
        return None
    id_values = _materialize_labels(id, "id")
    if len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if response.type not in {"right", "mright", "counting"}:
        raise NotImplementedError(
            "rttright id handling is currently supported only for right-censored data"
        )

    seen: set[Any] = set()
    for value in id_values:
        try:
            key = _hashable_group_value(value)
        except TypeError as exc:
            raise TypeError("id values must be hashable") from exc
        if response.type == "counting":
            continue
        if key in seen:
            raise ValueError("one or more flags are >0 in survcheck")
        seen.add(key)
    return id_values


def _rttright_binary_status(response: Surv) -> list[int]:
    return [1 if int(value) > 0 else 0 for value in response.event]


def _rttright_divide(numerator: float, denominator: float) -> float:
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.inf


def _rttright_counting_common_start(
    start: Sequence[float],
    id_values: Sequence[Any],
) -> bool:
    first_by_id: dict[Any, float] = {}
    for row_idx, id_value in enumerate(id_values):
        key = _hashable_group_value(id_value)
        first_by_id[key] = min(first_by_id.get(key, start[row_idx]), start[row_idx])
    if not first_by_id:
        return False
    first_start = next(iter(first_by_id.values()))
    return all(value == first_start for value in first_by_id.values())


def _rttright_counting_last_rows(
    stop: Sequence[float],
    id_values: Sequence[Any],
) -> list[bool]:
    last_by_id: dict[Any, tuple[float, int]] = {}
    for row_idx, id_value in enumerate(id_values):
        key = _hashable_group_value(id_value)
        candidate = (float(stop[row_idx]), row_idx)
        if key not in last_by_id or candidate > last_by_id[key]:
            last_by_id[key] = candidate
    return [
        row_idx == last_by_id[_hashable_group_value(id_value)][1]
        for row_idx, id_value in enumerate(id_values)
    ]


def _rttright_counting_validate_subject_weights(
    weights: Sequence[float],
    id_values: Sequence[Any],
) -> None:
    ranges: dict[Any, list[float]] = {}
    for weight, id_value in zip(weights, id_values, strict=True):
        key = _hashable_group_value(id_value)
        if key not in ranges:
            ranges[key] = [float(weight), float(weight)]
        else:
            ranges[key][0] = min(ranges[key][0], float(weight))
            ranges[key][1] = max(ranges[key][1], float(weight))
    if any(high > low for low, high in ranges.values()):
        raise ValueError("there are subjects with multiple weights")


def _rttright_counting_group_values(group: Sequence[int] | None, n: int) -> list[int]:
    if group is None:
        return [0] * n
    group_values = [int(value) for value in group]
    if len(group_values) != n:
        raise ValueError("rttright strata must have the same length as the Surv response")
    return group_values


def _rttright_counting_case_weights(
    weights: Any | None,
    id_values: Sequence[Any],
    group_values: Sequence[int],
    n: int,
    renorm: bool,
) -> list[float]:
    case_weights = _rttright_initial_weights(weights, n)
    _rttright_counting_validate_subject_weights(case_weights, id_values)
    if not renorm:
        return case_weights

    normalized = list(case_weights)
    group_indices: dict[int, list[int]] = {}
    for row_idx, group_value in enumerate(group_values):
        group_indices.setdefault(group_value, []).append(row_idx)

    for indices in group_indices.values():
        seen_ids: set[Any] = set()
        denominator = 0.0
        for row_idx in indices:
            key = _hashable_group_value(id_values[row_idx])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            denominator += case_weights[row_idx]
        if denominator <= 0.0:
            raise ValueError("weights must have positive sum when renorm is true")
        for row_idx in indices:
            normalized[row_idx] = case_weights[row_idx] / denominator
    return normalized


def _rttright_counting_delta(
    start: Sequence[float],
    stop: Sequence[float],
    query_times: Sequence[float] | None,
) -> float:
    values = [*map(float, start), *map(float, stop)]
    if query_times is not None:
        values.extend(float(value) for value in query_times)
    unique = sorted(set(values))
    diffs = [right - left for left, right in zip(unique, unique[1:], strict=False) if right > left]
    if not diffs:
        raise NotImplementedError("function not defined for delayed entry or multistate data")
    return min(diffs) / 2.0


def _rttright_counting_km(
    start: Sequence[float],
    stop: Sequence[float],
    censor: Sequence[int],
    weights: Sequence[float],
) -> tuple[list[float], list[float]]:
    event_times = sorted({float(stop[idx]) for idx, value in enumerate(censor) if value == 1})
    survival_times: list[float] = []
    survival_values: list[float] = []
    current = 1.0
    for event_time in event_times:
        risk = sum(
            float(weight)
            for left, right, weight in zip(start, stop, weights, strict=True)
            if float(left) < event_time <= float(right)
        )
        events = sum(
            float(weights[idx])
            for idx, value in enumerate(censor)
            if value == 1 and float(stop[idx]) == event_time
        )
        if risk > 0.0:
            current *= 1.0 - events / risk
        survival_times.append(event_time)
        survival_values.append(current)
    return survival_times, survival_values


def _rttright_km_survival_at(
    survival_times: Sequence[float],
    survival_values: Sequence[float],
    time: float,
) -> float:
    index = bisect_left(survival_times, float(time))
    return 1.0 if index == 0 else float(survival_values[index - 1])


def _rttright_time_matrix(
    time: Sequence[float],
    status: Sequence[int],
    weights: Any | None,
    times: Any,
    group: Sequence[int] | None,
    timefix: bool,
    renorm: bool,
) -> list[float] | list[list[float]]:
    time_values = [float(value) for value in time]
    status_values = [int(value) for value in status]
    n = len(time_values)
    if len(status_values) != n:
        raise ValueError("time and status must have same length")
    if any(value not in (0, 1) for value in status_values):
        raise ValueError("status must contain only 0/1 values")
    if any(not math.isfinite(value) for value in time_values):
        raise ValueError("time must be finite")

    query_times = _rttright_times_vector(times)
    case_weights = _rttright_initial_weights(weights, n)
    if group is None:
        group_values = None
    else:
        group_values = [int(value) for value in group]
        if len(group_values) != n:
            raise ValueError("rttright strata must have the same length as the Surv response")

    matrix = _core.rttright_time_matrix(
        time_values,
        status_values,
        query_times,
        case_weights,
        group_values,
        timefix,
        renorm,
    )

    if len(query_times) == 1:
        return [row[0] for row in matrix]
    return matrix


def _rttright_counting_group_result(
    start: Sequence[float],
    stop: Sequence[float],
    status: Sequence[int],
    weights: Sequence[float],
    last: Sequence[bool],
    query_times: Sequence[float] | None,
    delta: float,
) -> list[float] | list[list[float]]:
    km_stop = [float(value) for value in stop]
    censor = [
        1 if is_last and int(event) == 0 else 0 for is_last, event in zip(last, status, strict=True)
    ]
    for row_idx, value in enumerate(censor):
        if value == 1:
            km_stop[row_idx] += delta
    survival_times, survival_values = _rttright_counting_km(start, km_stop, censor, weights)

    if query_times is None:
        result: list[float] = []
        for row_stop, row_status, row_weight, is_last in zip(
            stop,
            status,
            weights,
            last,
            strict=True,
        ):
            if is_last and int(row_status) > 0:
                gwt = _rttright_km_survival_at(survival_times, survival_values, float(row_stop))
                result.append(_rttright_divide(float(row_weight), gwt))
            else:
                result.append(0.0)
        return result

    matrix = [[0.0] * len(query_times) for _ in range(len(start))]
    gwt = [_rttright_km_survival_at(survival_times, survival_values, time) for time in query_times]
    gwt2 = [
        _rttright_km_survival_at(survival_times, survival_values, row_stop) for row_stop in stop
    ]
    for row_idx, (row_start, row_stop, _row_status, row_weight, is_last) in enumerate(
        zip(start, stop, status, weights, last, strict=True)
    ):
        for col_idx, query_time in enumerate(query_times):
            if float(row_start) < float(query_time) <= float(row_stop):
                matrix[row_idx][col_idx] = _rttright_divide(float(row_weight), gwt[col_idx])
        if is_last and float(row_stop) > 0.0:
            for col_idx, query_gwt in enumerate(gwt):
                matrix[row_idx][col_idx] = _rttright_divide(
                    float(row_weight),
                    max(gwt2[row_idx], query_gwt),
                )
    return matrix


def _rttright_counting_result(
    response: Surv,
    weights: Any | None,
    times: Any | None,
    group: Sequence[int] | None,
    id_values: Sequence[Any] | None,
    timefix: bool,
    renorm: bool,
) -> list[float] | list[list[float]]:
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    if id_values is None:
        raise ValueError("id is required for start-stop data")

    id_labels = _materialize_labels(id_values, "id")
    n = len(response)
    if len(id_labels) != n:
        raise ValueError("id must have the same length as the Surv response")
    start = [float(value) for value in response.start]
    stop = [float(value) for value in response.time]
    status_values = [int(value) for value in response.event]
    if any(value not in (0, 1) for value in status_values):
        raise ValueError("rttright counting response must contain only 0/1 status values")
    if timefix:
        start, stop = _timefix_vectors(start, stop)

    check = _core.survcheck(
        _survcheck_integer_labels(id_labels, "id"),
        start,
        stop,
        status_values,
        None,
    )
    if any(flag > 0 for flag in check.flags):
        raise ValueError("one or more flags are >0 in survcheck")
    if not _rttright_counting_common_start(start, id_labels):
        raise NotImplementedError("function not defined for delayed entry or multistate data")

    last = _rttright_counting_last_rows(stop, id_labels)
    if (
        sum(1 for is_last, event in zip(last, status_values, strict=True) if is_last and event > 0)
        <= 1
    ):
        raise NotImplementedError("function not defined for delayed entry or multistate data")

    query_times = None if times is None else _rttright_times_vector(times)
    group_values = _rttright_counting_group_values(group, n)
    case_weights = _rttright_counting_case_weights(weights, id_labels, group_values, n, renorm)
    delta = _rttright_counting_delta(start, stop, query_times)

    matrix: list[list[float]] | None = None
    vector = [0.0] * n
    if query_times is not None:
        matrix = [[0.0] * len(query_times) for _ in range(n)]

    group_indices: dict[int, list[int]] = {}
    for row_idx, group_value in enumerate(group_values):
        group_indices.setdefault(group_value, []).append(row_idx)

    for indices in group_indices.values():
        group_result = _rttright_counting_group_result(
            [start[idx] for idx in indices],
            [stop[idx] for idx in indices],
            [status_values[idx] for idx in indices],
            [case_weights[idx] for idx in indices],
            [last[idx] for idx in indices],
            query_times,
            delta,
        )
        if query_times is None:
            for local_idx, row_idx in enumerate(indices):
                vector[row_idx] = float(group_result[local_idx])
        else:
            if matrix is None:
                raise RuntimeError("rttright counting matrix was not initialized")
            for local_idx, row_idx in enumerate(indices):
                matrix[row_idx] = [float(value) for value in group_result[local_idx]]

    if query_times is None:
        return vector
    if matrix is None:
        raise RuntimeError("rttright counting matrix was not initialized")
    if len(query_times) == 1:
        return [row[0] for row in matrix]
    return matrix


def _rttright_core_weights(
    response: Surv,
    weights: Any | None,
    group: Sequence[int] | None,
    timefix: bool,
    renorm: bool,
) -> list[float]:
    status = _rttright_binary_status(response)
    if group is not None:
        result = _core.rttright_stratified(
            list(response.time),
            status,
            list(group),
            None if weights is None else _float_vector(weights, "weights"),
            timefix,
            renorm,
        )
        return [float(weight) for weight in result.weights]
    result = _core.rttright(
        list(response.time),
        status,
        None if weights is None else _float_vector(weights, "weights"),
        timefix,
        renorm,
    )
    return _rttright_original_order_weights(result, len(response))


def rttright(
    response: Any,
    status: Any | None = None,
    weights: Any | None = None,
    *,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "pass",
    times: Any | None = None,
    id: Any | None = None,
    timefix: bool = True,
    renorm: bool = True,
    **kwargs: Any,
) -> Any:
    """Redistribute censored mass to the right, like R's ``rttright``."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    warn_offset = kwargs.pop("_warn_offset", True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"rttright got unexpected keyword argument(s): {unexpected}")
    if not isinstance(warn_offset, bool):
        raise TypeError("_warn_offset must be True or False")
    if not isinstance(timefix, bool):
        raise TypeError("timefix must be True or False")
    if not isinstance(renorm, bool):
        raise TypeError("renorm must be True or False")

    group: list[int] | None = None
    id_values = id
    if isinstance(response, str):
        surv_response, weights, id_values, group = _rttright_response_from_formula(
            response,
            data,
            subset,
            _normalize_na_action(na_action),
            weights,
            id,
            warn_offset=warn_offset,
        )
        response = surv_response
        subset = None
        na_action = "pass"
    elif isinstance(response, Surv):
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            weights = _subset_optional_sequence(weights, indices, "weights")
            id_values = _subset_optional_sequence(id_values, indices, "id")
            subset = None
        response, aligned = _apply_surv_na_action(
            response,
            _normalize_na_action(na_action),
            "rttright inputs",
            weights=weights,
            id=id_values,
        )
        weights = aligned["weights"]
        id_values = aligned["id"]
    else:
        if status is None:
            raise TypeError("rttright direct-vector calls require status")
        time_values = _float_vector(response, "time")
        status_values = _int_vector(status, "status")
        if len(time_values) != len(status_values):
            raise ValueError("time and status must have same length")
        if any(value not in (0, 1) for value in status_values):
            raise ValueError("status must contain only 0/1 values")
        response_values = Surv(time_values, status_values)
        _rttright_validate_id(response_values, id_values)
        if times is not None:
            return _rttright_time_matrix(
                list(response_values.time),
                list(response_values.event),
                weights,
                times,
                None,
                timefix,
                renorm,
            )
        return _core.rttright(
            list(response_values.time),
            list(response_values.event),
            None if weights is None else _float_vector(weights, "weights"),
            timefix,
            renorm,
        )

    if not isinstance(response, Surv):
        raise TypeError("rttright response must be a Surv object, formula, or time vector")
    if response.type == "mcounting":
        raise NotImplementedError("function not defined for delayed entry or multistate data")
    _rttright_validate_id(response, id_values)
    if response.type == "counting":
        return _rttright_counting_result(
            response,
            weights,
            times,
            group,
            id_values,
            timefix,
            renorm,
        )
    if response.type not in {"right", "mright"}:
        raise ValueError(f"rttright is not valid for {response.type} censored survival data")
    if times is not None:
        status_values = _rttright_binary_status(response)
        return _rttright_time_matrix(
            list(response.time),
            status_values,
            weights,
            times,
            group,
            timefix,
            renorm,
        )
    return _rttright_core_weights(
        response,
        weights,
        group,
        timefix,
        renorm,
    )


def _aeq_adjust_time_columns(
    columns: Sequence[Sequence[float]],
    tolerance: float | None,
) -> list[list[float]]:
    adjusted = [[float(value) for value in column] for column in columns]
    finite_values: list[float] = []
    finite_positions: list[tuple[int, int]] = []

    for col_idx, column in enumerate(adjusted):
        for row_idx, value in enumerate(column):
            if math.isfinite(value):
                finite_values.append(value)
                finite_positions.append((col_idx, row_idx))

    if not finite_values:
        return adjusted

    result = _core.aeq_surv(finite_values, tolerance)
    for (col_idx, row_idx), value in zip(finite_positions, result.time, strict=True):
        adjusted[col_idx][row_idx] = float(value)
    return adjusted


def _raise_if_aeq_zero_interval(
    original_left: Sequence[float],
    original_right: Sequence[float],
    adjusted_left: Sequence[float],
    adjusted_right: Sequence[float],
) -> None:
    for left, right, new_left, new_right in zip(
        original_left,
        original_right,
        adjusted_left,
        adjusted_right,
        strict=True,
    ):
        if left != right and new_left == new_right:
            raise ValueError("aeqSurv exception, an interval has effective length 0")


def aeqSurv(x: Any, tolerance: Any | None = None) -> Surv:
    """Adjudicate near-tied times in a ``Surv`` response, like R's ``aeqSurv``."""

    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    tolerance_value = None if tolerance is None else _finite_float(tolerance, "tolerance")
    if tolerance_value is not None and tolerance_value <= 0.0:
        return x

    if x.start is not None:
        start, stop = _aeq_adjust_time_columns((x.start, x.time), tolerance_value)
        _raise_if_aeq_zero_interval(x.start, x.time, start, stop)
        if x.type == "mcounting":
            return Surv._from_normalized(
                time=stop,
                event=x.event,
                start=start,
                time2=None,
                surv_type=x.type,
                states=x.states,
            )
        return Surv(start, stop, list(x.event), type="counting")

    if x.time2 is not None:
        left, right = _aeq_adjust_time_columns((x.time, x.time2), tolerance_value)
        _raise_if_aeq_zero_interval(x.time, x.time2, left, right)
        if x.type == "interval2":
            return Surv(left, right, type="interval2")
        return Surv(left, right, list(x.event), type=x.type)

    (time,) = _aeq_adjust_time_columns((x.time,), tolerance_value)
    if x.type == "mright":
        return Surv._from_normalized(
            time=time,
            event=x.event,
            start=None,
            time2=None,
            surv_type=x.type,
            states=x.states,
        )
    return Surv(time, list(x.event), type=x.type)


_TMERGE_COUNT_NAMES = (
    "early",
    "late",
    "gap",
    "within",
    "boundary",
    "leading",
    "trailing",
    "tied",
    "missid",
)


def tdc(time: Any, value: Any | None = None, init: Any | None = None) -> TMergeOperation:
    """Describe a time-dependent covariate update for :func:`tmerge`."""

    return TMergeOperation("tdc", time, value=value, default=init)


def cumtdc(time: Any, value: Any | None = None, init: Any | None = None) -> TMergeOperation:
    """Describe a cumulative time-dependent covariate for :func:`tmerge`."""

    return TMergeOperation("cumtdc", time, value=value, default=init)


def event(time: Any, value: Any | None = None, censor: Any | None = None) -> TMergeOperation:
    """Describe an event update for :func:`tmerge`."""

    return TMergeOperation("event", time, value=value, censor=censor)


def cumevent(
    time: Any,
    value: Any | None = None,
    censor: Any | None = None,
) -> TMergeOperation:
    """Describe a cumulative event update for :func:`tmerge`."""

    return TMergeOperation("cumevent", time, value=value, censor=censor)


def _tmerge_data_columns(data: Any, name: str) -> dict[str, list[Any]]:
    if isinstance(data, TMergeFrame):
        return {column: list(values) for column, values in data.columns.items()}
    names = _data_column_names(data)
    if names is None:
        raise TypeError(f"{name} must be a mapping or data frame")
    if any(not isinstance(column, str) or not column for column in names):
        raise ValueError(f"{name} column names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} column names must be unique")
    columns = {column: _column(data, column) for column in names}
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"{name} columns must have equal lengths")
    return columns


def _tmerge_row_count(columns: Mapping[str, Sequence[Any]]) -> int:
    return len(next(iter(columns.values()))) if columns else 0


def _tmerge_resolve_vector(
    value: Any,
    data: Any,
    n: int,
    name: str,
    *,
    scalar: bool = False,
) -> list[Any]:
    column_names = _data_column_names(data) or []
    if isinstance(value, str) and value in column_names:
        result = _column(data, value)
    else:
        scalar_value = isinstance(value, str | bytes)
        if scalar and not scalar_value:
            try:
                iter(value)
            except TypeError:
                scalar_value = True
        if scalar and scalar_value:
            result = [value] * n
        else:
            result = _materialize_1d(value, name)
            if scalar and len(result) == 1 and n != 1:
                result *= n
    if len(result) != n:
        raise ValueError(f"{name} must have length {n}")
    return result


def _tmerge_operation(value: Any, name: str) -> TMergeOperation:
    if isinstance(value, TMergeOperation):
        operation = value
    elif isinstance(value, Mapping):
        kind = value.get("kind", value.get("type", value.get("class")))
        operation = TMergeOperation(
            kind=str(kind),
            time=value.get("time"),
            value=value.get("value"),
            default=value.get("default", value.get("init")),
            censor=value.get("censor"),
        )
    else:
        raise TypeError(f"operation {name!r} must be created by tdc, cumtdc, event, or cumevent")
    kind = operation.kind.lower()
    if kind not in {"tdc", "cumtdc", "event", "cumevent"}:
        raise ValueError(f"operation {name!r} has unrecognized type {operation.kind!r}")
    if operation.time is None:
        raise ValueError(f"operation {name!r} requires time values")
    return replace(operation, kind=kind)


def _tmerge_options(options: Mapping[str, Any] | None, id: Any) -> dict[str, Any]:
    raw = {} if options is None else dict(options)
    aliases = {"na.rm": "na_rm"}
    normalized = {aliases.get(key, key): value for key, value in raw.items()}
    allowed = {"idname", "tstartname", "tstopname", "delay", "na_rm", "tdcstart"}
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(f"unrecognized tmerge option(s): {', '.join(unexpected)}")
    idname = normalized.get("idname", id if isinstance(id, str) else "id")
    tstartname = normalized.get("tstartname", "tstart")
    tstopname = normalized.get("tstopname", "tstop")
    for option, label in (
        (idname, "idname"),
        (tstartname, "tstartname"),
        (tstopname, "tstopname"),
    ):
        if not isinstance(option, str) or not option:
            raise ValueError(f"{label} option must be a non-empty variable name")
    delay = _finite_float(normalized.get("delay", 0.0), "delay")
    if delay < 0.0:
        raise ValueError("delay option must be a number >= 0")
    na_rm = _normalize_bool_option_with_default(normalized.get("na_rm", True), "na.rm", True)
    tdcstart = normalized.get("tdcstart", math.nan)
    if _is_missing_value(tdcstart):
        tdcstart = math.nan
    return {
        "idname": idname,
        "tstartname": tstartname,
        "tstopname": tstopname,
        "delay": delay,
        "na_rm": na_rm,
        "tdcstart": tdcstart,
    }


def _tmerge_retained_metadata(
    data1: Any,
    metadata: Mapping[str, Any] | None,
) -> tuple[dict[str, str] | None, dict[str, Any], tuple[str, ...], dict[str, dict[str, int]]]:
    if isinstance(data1, TMergeFrame):
        return (
            dict(data1.tname),
            dict(data1.tevent),
            tuple(data1.tdcvar),
            {name: dict(counts) for name, counts in data1.tcount.items()},
        )
    if metadata is None:
        return None, {}, (), {}
    tname_raw = metadata.get("tname")
    tname = (
        None
        if tname_raw is None
        else {str(key): str(value) for key, value in dict(tname_raw).items()}
    )
    tevent_raw = metadata.get("tevent", {})
    tevent = (
        {str(key): value for key, value in tevent_raw.items()}
        if isinstance(tevent_raw, Mapping)
        else {}
    )
    tdcvar = tuple(str(value) for value in metadata.get("tdcvar", ()))
    counts_raw = metadata.get("tcount", {})
    tcount = {
        str(operation): {str(kind): int(count) for kind, count in dict(values).items()}
        for operation, values in dict(counts_raw).items()
    }
    return tname, tevent, tdcvar, tcount


def _tmerge_unique_levels(values: Sequence[Any], name: str) -> tuple[list[Any], dict[Any, int]]:
    levels: list[Any] = []
    lookup: dict[Any, int] = {}
    for value in values:
        if _is_missing_value(value):
            raise ValueError(f"{name} cannot have missing values")
        key = _hashable_group_value(value)
        if key not in lookup:
            lookup[key] = len(levels)
            levels.append(value)
    return levels, lookup


def _tmerge_initial_frame(
    data1: Any,
    data2: Any,
    id: Any,
    tstart: Any | None,
    tstop: Any | None,
    operations: Mapping[str, TMergeOperation],
    options: Mapping[str, Any],
) -> dict[str, list[Any]]:
    columns1 = _tmerge_data_columns(data1, "data1")
    columns2 = _tmerge_data_columns(data2, "data2")
    n2 = _tmerge_row_count(columns2)
    if not isinstance(id, str):
        raise TypeError("on the first call id must be a single column name")
    if id not in columns1 or id not in columns2:
        raise KeyError(f"id column {id!r} must exist in data1 and data2")
    base_ids = list(columns1[id])
    base_levels, base_lookup = _tmerge_unique_levels(base_ids, "id")
    if len(base_levels) != len(base_ids):
        raise ValueError("data1 must have no duplicate identifiers on the first call")
    id2 = list(columns2[id])
    _tmerge_unique_levels(id2, "id")
    id2_keys = {_hashable_group_value(value) for value in id2}
    if any(_hashable_group_value(value) not in base_lookup for value in id2):
        raise ValueError("setting the range found data2 id values not in data1")
    if any(_hashable_group_value(value) not in id2_keys for value in base_ids):
        raise ValueError("setting the range found data1 id values not in data2")

    if tstop is None:
        if not operations or next(iter(operations.values())).kind != "event":
            raise ValueError("neither tstop nor an initial event operation was provided")
        event_times = _tmerge_resolve_vector(
            next(iter(operations.values())).time,
            data2,
            n2,
            "initial event time",
        )
        latest: dict[Any, float] = {}
        for row_id, value in zip(id2, event_times, strict=True):
            if _is_missing_value(value):
                continue
            key = _hashable_group_value(row_id)
            latest[key] = max(
                latest.get(key, -math.inf), _finite_float(value, "initial event time")
            )
        id_rows = []
        source_rows = []
        seen_ids: set[Any] = set()
        for row_id in id2:
            key = _hashable_group_value(row_id)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            id_rows.append(row_id)
            source_rows.append(base_lookup[key])
        stop_values = [latest[_hashable_group_value(row_id)] for row_id in id_rows]
    else:
        stop_values = [
            _finite_float(value, "tstop")
            for value in _tmerge_resolve_vector(tstop, data2, n2, "tstop")
        ]
        source_rows = [base_lookup[_hashable_group_value(value)] for value in id2]
        id_rows = list(id2)

    if tstart is None:
        start_values = [0.0] * len(stop_values)
    else:
        start_values = [
            _finite_float(value, "tstart")
            for value in _tmerge_resolve_vector(
                tstart, data2, len(stop_values), "tstart", scalar=True
            )
        ]
    if any(start >= stop for start, stop in zip(start_values, stop_values, strict=True)):
        raise ValueError("tstart must be less than tstop")

    id_keys = [_hashable_group_value(value) for value in id_rows]
    if len(set(id_keys)) != len(id_keys):
        first_seen = {key: idx for idx, key in enumerate(dict.fromkeys(id_keys))}
        order = sorted(
            range(len(stop_values)),
            key=lambda idx: (first_seen[id_keys[idx]], stop_values[idx], idx),
        )
    else:
        order = list(range(len(stop_values)))
    result = {
        name: [values[source_rows[idx]] for idx in order] for name, values in columns1.items()
    }
    result[str(options["idname"])] = [id_rows[idx] for idx in order]
    result[str(options["tstartname"])] = [start_values[idx] for idx in order]
    result[str(options["tstopname"])] = [stop_values[idx] for idx in order]
    ids = result[str(options["idname"])]
    starts = result[str(options["tstartname"])]
    stops = result[str(options["tstopname"])]
    for idx in range(1, len(stops)):
        if ids[idx] == ids[idx - 1] and starts[idx] < stops[idx - 1]:
            raise ValueError("first call created overlapping or duplicated time intervals")
    return result


def _tmerge_default_censor(
    values: Sequence[Any],
    categories: Sequence[Any] | None = None,
) -> Any:
    if categories:
        return categories[0]
    sample = next((value for value in values if not _is_missing_value(value)), 0)
    if isinstance(sample, bool):
        return False
    if isinstance(sample, str):
        return ""
    if isinstance(sample, float):
        return 0.0
    return 0


def _tmerge_numeric_value(value: Any, name: str) -> float:
    if _is_missing_value(value):
        return math.nan
    return _finite_float(value, name)


def _tmerge_operation_categories(operation: TMergeOperation, data2: Any) -> list[Any] | None:
    source = operation.value
    column_names = _data_column_names(data2) or []
    if isinstance(source, str) and source in column_names:
        source = _column_source(data2, source)
    categories = _mstate_categories(source)
    if categories is None:
        return None
    return _materialize_1d(categories, "event categories")


def _tmerge_apply_operation(
    columns: dict[str, list[Any]],
    data2: Any,
    data2_ids: list[Any],
    name: str,
    operation: TMergeOperation,
    options: Mapping[str, Any],
    event_censors: dict[str, Any],
    tdc_names: list[str],
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    n2 = len(data2_ids)
    times_raw = _tmerge_resolve_vector(operation.time, data2, n2, f"{name} time")
    values_raw = (
        None
        if operation.value is None
        else _tmerge_resolve_vector(operation.value, data2, n2, f"{name} value", scalar=True)
    )
    idname = str(options["idname"])
    startname = str(options["tstartname"])
    stopname = str(options["tstopname"])
    base_id_values = columns[idname]
    base_start_values = columns[startname]
    level_order: dict[Any, int] = {}
    base_codes: list[int] = []
    subject_min_starts: dict[int, float] = {}
    for base_id, start in zip(base_id_values, base_start_values, strict=True):
        if _is_missing_value(base_id):
            raise ValueError("id cannot have missing values")
        key = _hashable_group_value(base_id)
        code = level_order.setdefault(key, len(level_order))
        base_codes.append(code)
        subject_min_starts[code] = min(subject_min_starts.get(code, math.inf), float(start))
    updates: list[tuple[int, float, Any, int]] = []
    missid = 0
    for idx, (row_id, raw_time) in enumerate(zip(data2_ids, times_raw, strict=True)):
        code = level_order.get(_hashable_group_value(row_id))
        if code is None:
            missid += 1
            continue
        raw_value = None if values_raw is None else values_raw[idx]
        if _is_missing_value(raw_time) or (
            bool(options["na_rm"]) and values_raw is not None and _is_missing_value(raw_value)
        ):
            continue
        time = _finite_float(raw_time, f"{name} time")
        if (
            operation.kind in {"tdc", "cumtdc"}
            and float(options["delay"]) > 0.0
            and time > subject_min_starts[code]
        ):
            time += float(options["delay"])
        updates.append((code, time, raw_value, idx))

    updates.sort(key=lambda item: (item[0], item[1], item[3]))
    update_codes = [item[0] for item in updates]
    update_times = [item[1] for item in updates]
    update_values = [item[2] for item in updates]
    plan = _core.tmerge_plan(
        base_codes,
        [float(value) for value in base_start_values],
        [float(value) for value in columns[stopname]],
        update_codes,
        update_times,
    )
    classifications = list(plan.kind)
    counts = dict(zip(_TMERGE_COUNT_NAMES[:8], plan.count, strict=True))
    counts["missid"] = missid

    source_rows = list(plan.row)
    if len(source_rows) > len(base_codes):
        split_censor = list(plan.censor)
        columns = {
            column_name: [
                event_censors[column_name] if split_censor[idx] else values[source_row]
                for idx, source_row in enumerate(source_rows)
            ]
            if column_name in event_censors
            else [values[source_row] for source_row in source_rows]
            for column_name, values in columns.items()
        }
        columns[startname] = list(plan.start)
        columns[stopname] = list(plan.stop)
        base_codes = [base_codes[source_row] for source_row in source_rows]
    base_ids = columns[idname]
    starts = list(plan.start)

    if operation.kind == "tdc":
        if name in event_censors:
            raise ValueError(f"attempt to turn event variable {name!r} into a tdc")
        indices = _core.tmerge2(base_codes, starts, update_codes, update_times)
        existing = columns.get(name) if name in tdc_names else None
        if name in columns and name not in tdc_names:
            warnings.warn(f"replacement of variable {name!r}", stacklevel=3)
        if existing is None:
            if operation.value is None:
                new_values = [int(position != 0) for position in indices]
            else:
                default = options["tdcstart"] if operation.default is None else operation.default
                new_values = [
                    default if position == 0 else update_values[position - 1]
                    for position in indices
                ]
        else:
            new_values = list(existing)
            for row_idx, position in enumerate(indices):
                if position == 0:
                    continue
                new_values[row_idx] = 1 if operation.value is None else update_values[position - 1]
        columns[name] = new_values
        if name not in tdc_names:
            tdc_names.append(name)
        return columns, counts

    if operation.kind == "cumtdc":
        if name in event_censors:
            raise ValueError(f"attempt to turn event variable {name!r} into a cumtdc")
        increments = [
            1.0 if operation.value is None else _tmerge_numeric_value(value, name)
            for value in update_values
        ]
        existing = columns.get(name)
        if existing is None:
            default = (
                0.0
                if operation.value is None
                else (options["tdcstart"] if operation.default is None else operation.default)
            )
            initial = [float(default)] * len(base_ids)
        else:
            initial = [_tmerge_numeric_value(value, name) for value in existing]
        columns[name] = _core.tmerge(
            base_codes,
            starts,
            initial,
            update_codes,
            update_times,
            increments,
        )
        if name not in tdc_names:
            tdc_names.append(name)
        return columns, counts

    if name in tdc_names:
        raise ValueError(f"attempt to turn time-dependent covariate {name!r} into an event")
    raw_event_values = [1 if operation.value is None else value for value in update_values]
    if operation.kind == "cumevent":
        cumulative: dict[Any, float] = {}
        event_values: list[Any] = []
        for code, value in zip(update_codes, raw_event_values, strict=True):
            cumulative[code] = cumulative.get(code, 0.0) + _tmerge_numeric_value(value, name)
            event_values.append(cumulative[code])
    else:
        event_values = raw_event_values
    censor = event_censors.get(
        name,
        _tmerge_default_censor(
            event_values,
            _tmerge_operation_categories(operation, data2),
        ),
    )
    if name not in event_censors:
        columns[name] = [censor] * len(base_ids)
    target = {
        (code, float(stop)): idx
        for idx, (code, stop) in enumerate(zip(base_codes, columns[stopname], strict=True))
    }
    valid = {3, 4, 6}
    for update, kind, value, raw_value in zip(
        updates,
        classifications,
        event_values,
        raw_event_values,
        strict=True,
    ):
        if kind not in valid or (
            operation.kind == "cumevent"
            and not _is_missing_value(raw_value)
            and float(raw_value) == 0.0
        ):
            continue
        code, time, _unused, _source = update
        row_idx = target.get((code, time))
        if row_idx is not None:
            columns[name][row_idx] = value
    event_censors[name] = censor
    return columns, counts


def tmerge(
    data1: Any,
    data2: Any,
    id: Any,
    *,
    tstart: Any | None = None,
    tstop: Any | None = None,
    options: Mapping[str, Any] | None = None,
    operations: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    **updates: Any,
) -> TMergeFrame:
    """Create start/stop data with time-dependent covariates and events."""

    if data1 is None or data2 is None or id is None:
        raise TypeError("data1, data2, and id are required")
    operation_values = {} if operations is None else dict(operations)
    overlap = set(operation_values) & set(updates)
    if overlap:
        raise TypeError(f"duplicate tmerge operation(s): {', '.join(sorted(overlap))}")
    operation_values.update(updates)
    if any(not isinstance(name, str) or not name for name in operation_values):
        raise ValueError("all tmerge operations must have a name")
    parsed_operations = {
        name: _tmerge_operation(value, name) for name, value in operation_values.items()
    }
    retained_tname, retained_events, retained_tdc, retained_counts = _tmerge_retained_metadata(
        data1,
        metadata,
    )
    option_values = _tmerge_options(options, id)
    if retained_tname is not None:
        for key in ("idname", "tstartname", "tstopname"):
            if options is None or key not in options:
                option_values[key] = retained_tname[key]
        if tstart is not None or tstop is not None:
            raise ValueError("tstart and tstop only apply to the first tmerge call")
        columns = _tmerge_data_columns(data1, "data1")
    else:
        columns = _tmerge_initial_frame(
            data1,
            data2,
            id,
            tstart,
            tstop,
            parsed_operations,
            option_values,
        )

    data2_columns = _tmerge_data_columns(data2, "data2")
    n2 = _tmerge_row_count(data2_columns)
    if isinstance(id, str):
        if id not in data2_columns:
            raise KeyError(f"id column {id!r} not found in data2")
        data2_ids = list(data2_columns[id])
    else:
        data2_ids = _tmerge_resolve_vector(id, data2, n2, "id")
    _tmerge_unique_levels(data2_ids, "id")

    event_censors = dict(retained_events)
    tdc_names = list(retained_tdc)
    counts = {name: dict(values) for name, values in retained_counts.items()}
    for name, operation in parsed_operations.items():
        columns, operation_counts = _tmerge_apply_operation(
            columns,
            data2,
            data2_ids,
            name,
            operation,
            option_values,
            event_censors,
            tdc_names,
        )
        counts[name] = operation_counts
    tname = {
        "idname": str(option_values["idname"]),
        "tstartname": str(option_values["tstartname"]),
        "tstopname": str(option_values["tstopname"]),
    }
    return TMergeFrame(
        columns=columns,
        tname=tname,
        tevent=event_censors,
        tdcvar=tuple(tdc_names),
        tcount=counts,
    )


def _survsplit_output_name(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a variable name")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a variable name")
    return stripped


def _survsplit_data_columns(data: Any | None, n: int) -> dict[str, list[Any]]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping of column names to values")
    columns: dict[str, list[Any]] = {}
    for key, values in data.items():
        column_name = str(key)
        column = _materialize_1d(values, column_name)
        if len(column) != n:
            raise ValueError(f"data column {column_name!r} must have length {n}")
        columns[column_name] = column
    return columns


def survSplit(
    response: Surv,
    data: Any | None = None,
    *,
    cut: Any,
    start: str = "tstart",
    end: str = "tstop",
    event: str = "event",
    episode: str | None = None,
    id: str | None = None,
    zero: Any = 0,
) -> dict[str, list[Any]]:
    """Split right or counting-process survival data at fixed cut points."""

    if not isinstance(response, Surv):
        raise TypeError("survSplit response must be a Surv object")
    if response.type not in {"right", "mright", "counting", "mcounting"}:
        raise ValueError(f"not valid for {response.type} censored survival data")

    cut_values = _float_vector(cut, "cut")
    if any(not math.isfinite(value) for value in cut_values):
        raise ValueError("cut must be a vector of finite numbers")
    start_name = _survsplit_output_name(start, "start")
    end_name = _survsplit_output_name(end, "end")
    event_name = _survsplit_output_name(event, "event")
    episode_name = _survsplit_output_name(episode, "episode") if episode is not None else None
    id_name = _survsplit_output_name(id, "id") if id is not None else None

    n = len(response)
    frame = _survsplit_data_columns(data, n)
    if id_name is not None and id_name in frame:
        raise ValueError("the suggested id name is already present")

    if response.start is None:
        zero_value = _finite_float(zero, "zero")
        stop_values = list(response.time)
        observed_times = [value for value in stop_values if not math.isnan(value)]
        if any(value <= zero_value for value in observed_times):
            raise ValueError("'zero' parameter must be less than any observed times")
        start_values = [zero_value] * n
    else:
        start_values = list(response.start)
        stop_values = list(response.time)

    split = _core.survsplit(start_values, stop_values, cut_values)
    row_indices = [int(row) - 1 for row in split.row]

    result: dict[str, list[Any]] = {
        name: [column[row_idx] for row_idx in row_indices] for name, column in frame.items()
    }
    if id_name is not None:
        result[id_name] = [row_idx + 1 for row_idx in row_indices]

    original_status = list(response.event)
    result[start_name] = [float(value) for value in split.start]
    result[end_name] = [float(value) for value in split.end]
    result[event_name] = [
        0 if censor else int(original_status[row_idx])
        for censor, row_idx in zip(split.censor, row_indices, strict=True)
    ]
    if episode_name is not None:
        result[episode_name] = [int(value) for value in split.interval]
    return result
