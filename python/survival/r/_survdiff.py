"""``survdiff``: the G-rho family of tests, mirroring R's ``survdiff.R``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _first_argument,
    _is_bool_like,
    _pop_dotted_keyword,
    _subset_indices,
    _subset_optional_sequence,
)
from ._formula import (
    _apply_formula_na_action,
    _column_source,
    _covariate_term_name,
    _formula_response_spec,
    _formula_rhs_terms,
    _na_action_record,
    _offset_vector,
    _prepare_formula_inputs,
    _strata_keep,
    _strata_specs,
    _subset_formula_inputs,
    _surv_from_spec,
    _term_values,
    _without_survival_namespace,
)
from ._surv import Surv, _apply_surv_na_action, _complete_codes, _subset_surv
from ._survfit import _curve_factor
from ._types import (
    _MISSING,
    NaAction,
    StrataFactor,
    SurvdiffResult,
    _InteractionTerm,
    _ModelCovariateTerm,
)


def _response_check(y: Surv) -> None:
    if y.type in {"mright", "mcounting"}:
        raise ValueError("survdiff not defined for multi-state data")
    if y.type == "counting":
        raise ValueError("survdiff not defined for counting process data")
    if y.type != "right":
        raise ValueError("Right censored data only")


def _formula_inputs(
    formula: str, data: Any, subset: Any | None, na_action: str | None
) -> tuple[
    Surv,
    dict[str, Any],
    StrataFactor | None,
    list[float] | None,
    NaAction | None,
]:
    """The model frame: the response, the group columns, the strata factor and the offset."""

    # survdiff's only grouping special is strata(). cluster(x) is an ordinary group.
    terms = _formula_rhs_terms(formula, data, special_cluster=False)
    if subset is not None:
        data, _aligned = _subset_formula_inputs(formula, data, subset, terms)
    data, _aligned, removed = _apply_formula_na_action(formula, data, na_action, terms)
    y = _surv_from_spec(data, _formula_response_spec(formula))
    n = len(y)
    columns: dict[str, Any] = {}
    for model_term in terms.model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            term = model_term.term
            if isinstance(term, _InteractionTerm):
                raise ValueError("Interaction terms are not valid for this function")
            plain = term.transform is None and term.arithmetic is None and term.call is None
            values = _column_source(data, term.column) if plain else _term_values(data, term, n)
            columns[_covariate_term_name(term)] = values
    strata_terms = _strata_specs(terms)
    strata = _strata_keep(data, strata_terms, drop_unused=False) if strata_terms else None
    offset = _offset_vector(data, terms.offsets, n) if terms.offsets else None
    if offset is not None and (columns or strata is not None):
        raise ValueError("Cannot have both an offset and groups")
    return y, columns, strata, offset, _na_action_record(na_action, removed)


def _one_sample(y: Surv, offset: list[float], rho: float) -> SurvdiffResult:
    """``survdiff`` with an ``offset()`` term: the observed against the expected events."""

    if any(value < 0.0 or value > 1.0 for value in offset):
        raise ValueError("The offset must be a survival probability")
    fit = _core.survdiff_one_sample(y._event_codes(), offset, rho)
    return SurvdiffResult(
        n=[len(y)],
        obs=[fit.obs[0][0]],
        exp=[fit.exp[0][0]],
        var=fit.var,
        chisq=fit.chisq,
        pvalue=fit.pvalue,
        df=1,
        groups=[],
    )


def _k_sample(
    y: Surv,
    group_codes: list[int],
    group_levels: list[str],
    strata: StrataFactor | None,
    rho: float,
    timefix: bool,
) -> SurvdiffResult:
    strata_codes = (
        None
        if strata is None
        else _complete_codes(strata, "missing values in the grouping variables")
    )
    fit = _core.survdiff(
        y.time,
        y._event_codes(),
        group_codes,
        strata=strata_codes,
        rho=rho,
        timefix=timefix,
    )
    strata_counts = fit.strata
    multiple_strata = strata_counts is not None and len(strata_counts) > 1
    return SurvdiffResult(
        n=[int(value) for value in fit.n],
        obs=fit.obs if multiple_strata else [row[0] for row in fit.obs],
        exp=fit.exp if multiple_strata else [row[0] for row in fit.exp],
        var=fit.var,
        chisq=fit.chisq,
        pvalue=fit.pvalue,
        df=int(fit.df),
        groups=group_levels,
        strata=None if strata is None else dict(zip(strata.levels, strata.counts, strict=True)),
    )


def survdiff(
    response: Any = _MISSING,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.omit",
    rho: Any = 0,
    timefix: Any = True,
    *,
    group: Any | None = None,
    **kwargs: Any,
) -> SurvdiffResult:
    """R's ``survdiff``: the log-rank (``rho = 0``) and G-rho family tests.

    ``response`` is a formula string (``"Surv(time, status) ~ sex + strata(inst)"``) evaluated
    in ``data``; an ``offset(expected)`` term of expected survival probabilities gives the
    one-sample test.  A ``Surv`` object with ``group`` compares its groups.
    """

    response = _first_argument(response, kwargs, "response", "formula", "survdiff")
    response = _without_survival_namespace(response)

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.omit")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survdiff got unexpected keyword argument(s): {unexpected}")
    if not _is_bool_like(timefix):
        raise ValueError("invalid value for timefix option")
    rho = _finite_float(rho, "rho")
    if isinstance(response, str):
        y, columns, strata, offset, omitted = _formula_inputs(response, data, subset, na_action)
        if group is not None:
            raise ValueError("group is only used with a Surv response")
        _response_check(y)
        if offset is not None:
            return replace(_one_sample(y, offset, rho), na_action=omitted)
        if not columns:
            raise ValueError("No groups to test")
        codes, levels = _curve_factor(columns, len(y))
    elif isinstance(response, Surv):
        data, supplied = _prepare_formula_inputs(data, group=group, subset=subset)
        group = supplied["group"]
        subset = supplied["subset"]
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            group = _subset_optional_sequence(group, indices, "group")
        y, aligned, removed = _apply_surv_na_action(
            response, na_action, "survdiff inputs", group=group
        )
        omitted = _na_action_record(na_action, removed)
        _response_check(y)
        if aligned["group"] is None:
            raise ValueError("No groups to test")
        codes, levels = _curve_factor({"group": aligned["group"]}, len(y), shortlabel=True)
        strata = None
    else:
        raise TypeError("The 'formula' argument is not a formula")
    result = _k_sample(y, codes, levels, strata, rho, bool(timefix))
    return replace(result, na_action=omitted)
