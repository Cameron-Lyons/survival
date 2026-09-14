"""``cch`` case-cohort models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _encode_labels,
    _integer_scalar,
    _label_levels,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option_with_default,
    _pop_dotted_keyword,
)
from ._fit import _formula_design_output_names
from ._formula import (
    _apply_formula_na_action,
    _column_or_values,
    _design_rows_from_spec,
    _fit_formula_design,
    _formula_response_spec,
    _parse_formula,
    _subset_formula_inputs,
)
from ._types import CchModelResult


def _cch_stratified_cohort_sizes(value: Any, levels: Sequence[Any]) -> list[int]:
    if isinstance(value, Mapping):
        missing = [level for level in levels if level not in value]
        extra = [key for key in value if key not in levels]
        if missing or extra:
            raise ValueError("cohort_size mapping keys must match the stratum levels")
        raw_sizes = [value[level] for level in levels]
    else:
        raw_sizes = _materialize_1d(value, "cohort_size")
        if len(raw_sizes) != len(levels):
            raise ValueError("cohort_size and stratum levels must have the same length")
    sizes = [_integer_scalar(item, "cohort_size") for item in raw_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError("cohort_size values must be positive")
    return sizes


def cch(
    formula: str,
    data: Any,
    *,
    subcoh: Any,
    id: Any,
    cohort_size: Any | None = None,
    stratum: Any | None = None,
    method: str = "Prentice",
    robust: Any = False,
    subset: Any | None = None,
    na_action: str | None = "fail",
    **kwargs: Any,
) -> CchModelResult:
    """Fit an unstratified or sampling-stratified case-cohort Cox model."""

    cohort_size = _pop_dotted_keyword(
        kwargs,
        "cohort.size",
        "cohort_size",
        cohort_size,
        None,
    )
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"cch got unexpected keyword argument(s): {unexpected}")
    if not isinstance(formula, str):
        raise TypeError("cch formula must be a string")
    if data is None:
        raise ValueError("cch formula requires data")
    if cohort_size is None:
        raise TypeError("cohort_size is required")

    normalized_method = _match_string_arg(
        method,
        "method",
        ("prentice", "selfprentice", "linying", "i.borgan", "ii.borgan"),
        "cch method must be 'Prentice', 'SelfPrentice', 'LinYing', 'I.Borgan', or 'II.Borgan'",
    )
    method_name = {
        "prentice": "Prentice",
        "selfprentice": "SelfPrentice",
        "linying": "LinYing",
        "i.borgan": "I.Borgan",
        "ii.borgan": "II.Borgan",
    }[normalized_method]
    stratified = method_name in {"I.Borgan", "II.Borgan"}
    robust_value = _normalize_bool_option_with_default(robust, "robust", False)
    if stratified and robust_value:
        warnings.warn(
            "robust variance is not implemented for stratified cch analysis",
            RuntimeWarning,
            stacklevel=2,
        )
        robust_value = False
    elif robust_value and method_name != "LinYing":
        warnings.warn(
            f"robust ignored for method ({method_name})",
            RuntimeWarning,
            stacklevel=2,
        )
        robust_value = False

    subcohort_values = _column_or_values(data, subcoh, "subcoh")
    id_values = _column_or_values(data, id, "id")
    stratum_values = _column_or_values(data, stratum, "stratum") if stratum is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            subcohort=subcohort_values,
            id=id_values,
            stratum=stratum_values,
        )
        subcohort_values = aligned["subcohort"]
        id_values = aligned["id"]
        stratum_values = aligned["stratum"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        subcohort=subcohort_values,
        id=id_values,
        stratum=stratum_values,
    )
    subcohort_values = aligned["subcohort"]
    id_values = aligned["id"]
    stratum_values = aligned["stratum"]

    response_spec = _formula_response_spec(formula)
    response, terms = _parse_formula(formula, data)
    if response.type not in {"right", "counting"}:
        raise NotImplementedError("cch supports right-censored and counting Surv responses")
    if stratified and stratum_values is None:
        raise ValueError(f"method ({method_name}) requires stratum")
    if not stratified and (terms.strata or stratum_values is not None):
        warnings.warn(
            f"stratum ignored for method ({method_name})",
            RuntimeWarning,
            stacklevel=2,
        )
    if terms.offsets:
        warnings.warn("Offset term ignored", RuntimeWarning, stacklevel=2)
    if terms.clusters:
        raise ValueError("cluster() terms are not supported by cch")

    design = _fit_formula_design(data, response_spec, terms, len(response))
    rows = _design_rows_from_spec(data, design, len(response))
    if not rows or not rows[0]:
        raise ValueError("cch formula must contain at least one covariate")
    coefficient_names = tuple(_formula_design_output_names(design))

    raw_subcohort = _materialize_1d(subcohort_values, "subcoh")
    if len(raw_subcohort) != len(response):
        raise ValueError("subcoh must have the same length as the Surv response")
    fit_subcohort: list[int] = []
    for row_idx, value in enumerate(raw_subcohort):
        if isinstance(value, bool):
            fit_subcohort.append(int(value))
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("subcoh must contain only 0/1 or boolean values") from exc
        if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
            raise ValueError(
                f"subcoh must contain only 0/1 or boolean values; got {value!r} at row {row_idx}"
            )
        fit_subcohort.append(int(numeric))

    id_labels = _materialize_labels(id_values, "id")
    if len(id_labels) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if len(_label_levels(id_labels, "id")) != len(id_labels):
        raise ValueError("multiple records per id are not allowed")
    id_codes = _encode_labels(id_labels, "id")
    stratum_labels: list[Any] | None = None
    if stratified:
        stratum_labels = _materialize_labels(stratum_values, "stratum")
        if len(stratum_labels) != len(response):
            raise ValueError("stratum must have the same length as the Surv response")
        stratum_levels = _label_levels(stratum_labels, "stratum")
        stratum_codes = _encode_labels(stratum_labels, "stratum")
        cohort_sizes = _cch_stratified_cohort_sizes(cohort_size, stratum_levels)
        fit = _core.cch_borgan_fit(
            list(response.time),
            list(response.event),
            rows,
            fit_subcohort,
            id_codes,
            stratum_codes,
            cohort_sizes,
            start=list(response.start) if response.start is not None else None,
            method=method_name,
        )
    else:
        cohort_size_value = _integer_scalar(cohort_size, "cohort_size")
        if cohort_size_value <= 0:
            raise ValueError("cohort_size must be positive")
        cohort_sizes = [cohort_size_value]
        fit = _core.cch_fit(
            list(response.time),
            list(response.event),
            rows,
            fit_subcohort,
            id_codes,
            cohort_size_value,
            start=list(response.start) if response.start is not None else None,
            method=method_name,
            robust=robust_value,
        )
    return CchModelResult(
        fit=fit,
        design=design,
        formula=formula,
        coefficient_names=coefficient_names,
        response=response,
        id_values=id_labels,
        subcohort=fit_subcohort,
        stratum_values=stratum_labels,
        cohort_sizes=cohort_sizes,
    )
