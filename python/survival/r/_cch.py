"""``cch`` case-cohort models (R/cch.R): the argument checks and the model frame;
the Prentice/SelfPrentice/LinYing/Borgan estimators run in Rust."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _integer_scalar,
    _label_levels,
    _match_string_arg,
    _materialize_1d,
    _normalize_bool_option_with_default,
    _pop_dotted_keyword,
)
from ._fit import _model_frame, _strata_factor
from ._types import CchModelResult

_METHODS = {
    "prentice": "Prentice",
    "selfprentice": "SelfPrentice",
    "linying": "LinYing",
    "i.borgan": "I.Borgan",
    "ii.borgan": "II.Borgan",
}


def _subcohort_indicator(values: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for value in values:
        if isinstance(value, bool):
            out.append(int(value))
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Permissible values for subcohort indicator are 0/1 or TRUE/FALSE"
            ) from exc
        if numeric not in (0.0, 1.0):
            raise ValueError("Permissible values for subcohort indicator are 0/1 or TRUE/FALSE")
        out.append(int(numeric))
    return out


def _stratified_cohort_sizes(cohort_size: Any, levels: Sequence[str]) -> list[int]:
    if isinstance(cohort_size, Mapping):
        sizes = {
            str(key): _integer_scalar(value, "cohort_size") for key, value in cohort_size.items()
        }
        if len(sizes) != len(levels):
            raise ValueError("cohort.size and stratum do not match")
        if any(level not in sizes for level in levels):
            warnings.warn(
                "stratum levels and names(cohort.size) do not agree", RuntimeWarning, stacklevel=3
            )
            return list(sizes.values())
        return [sizes[level] for level in levels]
    values = [
        _integer_scalar(value, "cohort_size")
        for value in _materialize_1d(cohort_size, "cohort_size")
    ]
    if len(values) != len(levels):
        raise ValueError("cohort.size and stratum do not match")
    return values


def cch(
    formula: str,
    data: Any = None,
    subcoh: Any = None,
    id: Any = None,
    stratum: Any | None = None,
    cohort_size: Any | None = None,
    method: str = "Prentice",
    robust: Any = False,
    *,
    subset: Any | None = None,
    na_action: str | None = "fail",
    **kwargs: Any,
) -> CchModelResult:
    """Fit a case-cohort Cox model (R's ``cch``).

    ``subcoh``, ``id`` and ``stratum`` are vectors or column names of ``data``;
    ``cohort_size`` is one number, or one per stratum (a mapping keyed by stratum
    level) for the Borgan estimators.
    """

    cohort_size = _pop_dotted_keyword(kwargs, "cohort.size", "cohort_size", cohort_size, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        raise TypeError(f"cch got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if subcoh is None or id is None:
        raise TypeError("subcoh and id are required")
    if cohort_size is None:
        raise TypeError("cohort.size is required")
    method_name = _METHODS[
        _match_string_arg(
            method,
            "method",
            tuple(_METHODS),
            "method must be one of Prentice, SelfPrentice, LinYing, I.Borgan, II.Borgan",
        )
    ]
    stratified = method_name in {"I.Borgan", "II.Borgan"}
    robust_value = _normalize_bool_option_with_default(robust, "robust", False)
    if stratified:
        if robust_value:
            warnings.warn(
                "`robust' not implemented for stratified analysis.", RuntimeWarning, stacklevel=2
            )
        if stratum is None:
            raise ValueError(f"method ({method_name}) requires 'stratum'")
    else:
        if method_name != "LinYing" and robust_value:
            warnings.warn(
                f"`robust' ignored for  method ({method_name})", RuntimeWarning, stacklevel=2
            )
        if stratum is not None:
            warnings.warn(
                f"'stratum' ignored for method ({method_name})", RuntimeWarning, stacklevel=2
            )
        stratum = None
    robust_value = robust_value and method_name == "LinYing"

    frame = _model_frame(
        formula,
        data,
        subset=subset,
        na_action=na_action,
        id=id,
        extra={"subcoh": subcoh, "stratum": stratum},
    )
    y = frame.y
    if y.type not in {"right", "counting"}:
        raise ValueError(f'Cox model doesn\'t support "{y.type}" survival data')
    if frame.terms.offsets:
        warnings.warn("Offset term ignored", RuntimeWarning, stacklevel=2)
    if not frame.names:
        raise ValueError("cch formula must contain at least one covariate")
    id_values = list(frame.id or [])
    if len(_label_levels(id_values, "id")) != len(id_values):
        raise ValueError("Multiple records per id not allowed")
    subcohort = _subcohort_indicator(frame.extra["subcoh"])
    outside = sum(1 for sub, event in zip(subcohort, y.event, strict=True) if not sub and not event)
    if outside:
        raise ValueError(f"{outside} censored observations not in subcohort")
    id_codes = list(range(len(id_values)))
    start = None if y.start is None else list(y.start)
    status = [int(value) for value in y.event]
    stratum_labels: tuple[Any, ...] | None = None
    if stratified:
        factor = _strata_factor({"stratum": frame.extra["stratum"]}, frame.n, shortlabel=True)
        stratum_labels = tuple(frame.extra["stratum"])
        levels = list(factor.levels)
        codes = [int(code) for code in factor.codes]
        sizes = _stratified_cohort_sizes(cohort_size, levels)
        counts = list(factor.counts)
        if len(id_values) > sum(sizes):
            raise ValueError("Number of records greater than cohort size")
        if any(count > size for count, size in zip(counts, sizes, strict=True)):
            raise ValueError("Population smaller than sample in some strata")
        fit = _core.cch_borgan_fit(
            list(y.time),
            status,
            frame.x,
            subcohort,
            id_codes,
            codes,
            sizes,
            start=start,
            method=method_name,
        )
        subcohort_size = tuple(counts)
        cohort_sizes = tuple(sizes)
    else:
        if isinstance(cohort_size, Mapping) or (
            isinstance(cohort_size, Sequence) and not isinstance(cohort_size, str)
        ):
            raise ValueError("cohort size must be a scalar for unstratified analysis")
        size = _integer_scalar(cohort_size, "cohort_size")
        if len(id_values) > size:
            raise ValueError("Number of records greater than cohort size")
        fit = _core.cch_fit(
            list(y.time),
            status,
            frame.x,
            subcohort,
            id_codes,
            size,
            start=start,
            method=method_name,
            robust=robust_value,
        )
        subcohort_size = (sum(subcohort),)
        cohort_sizes = (size,)
    return CchModelResult(
        fit=fit,
        formula=formula,
        design=frame.design,
        coef_names=tuple(frame.names),
        y=y,
        id=tuple(id_values),
        subcoh=tuple(subcohort),
        stratum=stratum_labels,
        cohort_size=cohort_sizes,
        subcohort_size=subcohort_size,
    )


def summary_cch(fit: CchModelResult) -> dict[str, Any]:
    """R's ``summary.cch``: ``Value``/``SE``/``Z``/``p`` per coefficient."""

    rows = []
    for idx, (name, coef) in enumerate(zip(fit.coef_names, fit.coefficients, strict=True)):
        se = math.sqrt(fit.var[idx][idx])
        z = abs(coef / se) if se > 0.0 else math.nan
        p = 2.0 * (1.0 - 0.5 * math.erfc(-z / math.sqrt(2.0)))  # R: 2*(1-pnorm(Z))
        rows.append({"name": name, "coef": coef, "value": coef, "se": se, "z": z, "p": p})
    return {
        "model_type": "cch",
        "method": fit.method,
        "cohort_size": list(fit.cohort_size),
        "subcohort_size": list(fit.subcohort_size),
        "stratified": fit.stratified,
        "coefficient_names": list(fit.coef_names),
        "coefficient_columns": ["Value", "SE", "Z", "p"],
        "coefficients": rows,
    }
