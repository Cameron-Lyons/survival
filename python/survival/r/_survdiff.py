"""``survdiff`` log-rank family tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _encode_groups,
    _finite_float,
    _normalize_bool_option,
    _pop_dotted_keyword,
    _r_formula_ordered_levels,
    _subset_indices,
    _subset_optional_sequence,
)
from ._formula import (
    _apply_formula_na_action,
    _combined_columns,
    _combined_formula_groups,
    _offset_vector,
    _parse_formula,
    _reject_formula_clusters,
    _subset_formula_inputs,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._types import _FormulaTerms


def _survdiff_weight_type(rho: float) -> str:
    return "LogRank" if rho == 0.0 else f"FlemingHarrington(p={rho}, q=0)"


def _survdiff_r_ordered_levels(values: list[Any]) -> tuple[Any, ...]:
    return _r_formula_ordered_levels(values, "survdiff formula groups")


def _survdiff_formula_groups(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[Any], tuple[Any, ...], list[Any] | None]:
    if not terms.covariates:
        if terms.strata:
            raise ValueError("survdiff formula has no groups to test")
        raise ValueError("survdiff formula requires at least one grouping term")
    group = _combined_formula_groups(data, [], terms.covariates, n)
    group_levels = _survdiff_r_ordered_levels(group)
    strata = _combined_columns(data, terms.strata, n) if terms.strata else None
    return group, group_levels, strata


def _survdiff_offset_formula_values(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> list[float] | None:
    if not terms.offsets:
        return None
    if terms.covariates or terms.strata:
        raise ValueError("Cannot have both an offset and groups")
    values = _offset_vector(data, terms.offsets, n)
    if values is None:
        raise ValueError("offset formula did not produce values")
    return values


def _survdiff_result_from_components(components: Any, rho: float) -> Any:
    statistic = float(components.chi_squared)
    df = int(components.degrees_of_freedom)
    p_value = 1.0 if df == 0 else float(_core.lrt_test(statistic / 2.0, 0.0, df).p_value)
    variance = (
        float(components.variance[0][0]) if components.variance and components.variance[0] else 0.0
    )
    return _core.LogRankResult(
        statistic,
        p_value,
        df,
        [float(value) for value in components.observed],
        [float(value) for value in components.expected],
        variance,
        _survdiff_weight_type(rho),
    )


def _survdiff_offset_expected(offsets: Sequence[float]) -> float:
    total = 0.0
    for value in offsets:
        if value == 0.0:
            return math.inf
        total += -math.log(value)
    return total


def _survdiff_divide_statistic(numerator: float, variance: float) -> float:
    squared = numerator * numerator
    if variance == 0.0:
        return math.nan if squared == 0.0 else math.inf
    return squared / variance


def _survdiff_chisq_p_value(statistic: float) -> float:
    return math.erfc(math.sqrt(statistic / 2.0))


def _survdiff_offset_result(response: Surv, offsets: Sequence[float], rho: float) -> Any:
    if response.type != "right":
        raise NotImplementedError("survdiff offset formulas require right-censored Surv responses")
    if len(offsets) != len(response):
        raise ValueError("offset must have the same length as the Surv response")
    if any(value < 0.0 or value > 1.0 for value in offsets):
        raise ValueError("The offset must be a survival probability")

    observed = float(sum(response.event))
    expected = _survdiff_offset_expected(offsets)
    if rho == 0.0:
        variance = expected
        numerator = observed - variance
    else:
        inverse_rho = 1.0 / rho
        numerator = sum(
            inverse_rho - ((inverse_rho + float(event)) * (offset**rho))
            for offset, event in zip(offsets, response.event, strict=True)
        )
        variance = sum((1.0 - (offset ** (2.0 * rho))) / (2.0 * rho) for offset in offsets)
    statistic = _survdiff_divide_statistic(numerator, variance)
    return _core.LogRankResult(
        statistic,
        _survdiff_chisq_p_value(statistic),
        1,
        [observed],
        [expected],
        variance,
        _survdiff_weight_type(rho),
    )


def _stratified_survdiff(
    response: Surv,
    group: Any,
    strata: Any,
    rho: float,
    timefix: bool,
    group_levels: Sequence[Any] | None = None,
) -> Any:
    n = len(response)
    group_codes = _encode_groups(group, n, levels=group_levels)
    strata_codes = _encode_groups(strata, n)
    times = list(response.time)
    if response.start is not None:
        components = _core.stratified_counting_logrank_components(
            times,
            list(response.event),
            [code + 1 for code in group_codes],
            list(response.start),
            strata_codes,
            rho,
            timefix,
        )
        return _survdiff_result_from_components(components, rho)

    components = _core.stratified_logrank_components(
        times,
        list(response.event),
        [code + 1 for code in group_codes],
        strata_codes,
        rho,
        timefix,
    )
    return _survdiff_result_from_components(components, rho)


def survdiff(
    response: Surv | str,
    data: Any | None = None,
    *,
    group: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    rho: float = 0.0,
    timefix: bool = True,
    **kwargs: Any,
):
    """Compare survival curves with R's G-rho family for common survdiff use."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", timefix, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survdiff got unexpected keyword argument(s): {unexpected}")

    formula_strata: Any | None = None
    formula_group_levels: Sequence[Any] | None = None
    formula_offsets: list[float] | None = None
    if isinstance(response, str):
        if subset is not None:
            data, _aligned = _subset_formula_inputs(response, data, subset)
            subset = None
        data, _aligned = _apply_formula_na_action(response, data, na_action)
        na_action = "pass"
        response, terms = _parse_formula(response, data)
        _reject_formula_clusters("survdiff", terms)
        formula_offsets = _survdiff_offset_formula_values(data, terms, len(response))
        if formula_offsets is None:
            group, formula_group_levels, formula_strata = _survdiff_formula_groups(
                data,
                terms,
                len(response),
            )

    if not isinstance(response, Surv):
        raise TypeError("survdiff response must be a Surv object or formula")
    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "survdiff inputs",
        group=group,
        strata=formula_strata,
    )
    group = aligned["group"]
    formula_strata = aligned["strata"]
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survdiff currently supports right-censored and counting Surv responses"
        )
    rho_value = _finite_float(rho, "rho")
    fix_time = _normalize_bool_option(timefix, "timefix")
    if formula_offsets is not None:
        return _survdiff_offset_result(response, formula_offsets, rho_value)
    if group is None:
        raise ValueError("group is required")
    if formula_strata is not None:
        return _stratified_survdiff(
            response,
            group,
            formula_strata,
            rho_value,
            fix_time,
            formula_group_levels,
        )

    groups = _encode_groups(group, len(response), levels=formula_group_levels)
    group_codes = [code + 1 for code in groups]
    if response.start is not None:
        components = _core.compute_counting_logrank_components(
            list(response.time),
            list(response.event),
            group_codes,
            list(response.start),
            None,
            rho_value,
            fix_time,
        )
    else:
        components = _core.survdiff2(
            list(response.time),
            list(response.event),
            group_codes,
            None,
            rho_value,
            fix_time,
        )
    return _survdiff_result_from_components(components, rho_value)
