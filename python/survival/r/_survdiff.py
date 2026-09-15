"""``survdiff``: the G-rho family of tests, mirroring R's ``survdiff.R``."""

from __future__ import annotations

from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _is_bool_like,
    _pop_dotted_keyword,
    _r_factor,
    _subset_indices,
    _subset_optional_sequence,
)
from ._formula import (
    _apply_formula_na_action,
    _column_source,
    _covariate_term_name,
    _offset_vector,
    _parse_formula,
    _subset_formula_inputs,
    _term_values,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._survfit import _curve_factor, _strata_factor, _strata_term_values
from ._types import SurvDiffResult, _InteractionTerm, _ModelCovariateTerm, _ModelStrataTerm


def _response_check(y: Surv) -> None:
    if y.type in {"mright", "mcounting"}:
        raise ValueError("survdiff not defined for multi-state data")
    if y.type == "counting":
        raise ValueError("survdiff not defined for counting process data")
    if y.type != "right":
        raise ValueError("Right censored data only")


def _strata_values(data: Any, terms: list[_ModelStrataTerm], n: int) -> Any:
    """``strata.keep``: the ``strata()`` column, or ``strata(m[, vars], shortlabel = TRUE)``."""

    columns = {
        f"strata({', '.join(term.columns)})": _strata_term_values(data, term.columns, n)
        for term in terms
    }
    if len(columns) == 1:
        return next(iter(columns.values()))
    factor = _strata_factor(columns, shortlabel=True)
    return _r_factor(
        [None if code is None else factor.levels[code] for code in factor.codes], factor.levels
    )


def _formula_inputs(
    formula: str, data: Any, subset: Any | None, na_action: str | None
) -> tuple[Surv, dict[str, Any], list[str] | None, list[float] | None]:
    """The model frame: the response, the group columns, the strata factor and the offset."""

    if subset is not None:
        data, _aligned = _subset_formula_inputs(formula, data, subset)
    data, _aligned = _apply_formula_na_action(formula, data, na_action)
    y, terms = _parse_formula(formula, data)
    n = len(y)
    if terms.clusters:
        raise ValueError("cluster() terms are not valid for this function")
    columns: dict[str, Any] = {}
    for model_term in terms.model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            term = model_term.term
            if isinstance(term, _InteractionTerm):
                raise ValueError("Interaction terms are not valid for this function")
            plain = term.transform is None and term.arithmetic is None
            values = _column_source(data, term.column) if plain else _term_values(data, term, n)
            columns[_covariate_term_name(term)] = values
    strata_terms = [term for term in terms.model_terms if isinstance(term, _ModelStrataTerm)]
    strata = _strata_values(data, strata_terms, n) if strata_terms else None
    offset = _offset_vector(data, terms.offsets, n) if terms.offsets else None
    if offset is not None and (columns or strata is not None):
        raise ValueError("Cannot have both an offset and groups")
    return y, columns, strata, offset


def _one_sample(y: Surv, offset: list[float], rho: float) -> SurvDiffResult:
    """``survdiff`` with an ``offset()`` term: the observed against the expected events."""

    if any(value < 0.0 or value > 1.0 for value in offset):
        raise ValueError("The offset must be a survival probability")
    fit = _core.survdiff_one_sample([int(value) for value in y.event], offset, rho)
    return SurvDiffResult(
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
    strata: Any | None,
    rho: float,
    timefix: bool,
) -> SurvDiffResult:
    n = len(y)
    strata_codes = strata_levels = None
    if strata is not None:
        strata_codes, _labels = _curve_factor({"strata": strata}, n)
        strata_levels = list(_strata_factor({"strata": strata}, shortlabel=True).levels)
    fit = _core.survdiff(
        list(y.time),
        [int(value) for value in y.event],
        group_codes,
        strata=strata_codes,
        rho=rho,
        timefix=timefix,
    )
    stratified = fit.strata is not None
    return SurvDiffResult(
        n=[int(value) for value in fit.n],
        obs=fit.obs if stratified else [row[0] for row in fit.obs],
        exp=fit.exp if stratified else [row[0] for row in fit.exp],
        var=fit.var,
        chisq=fit.chisq,
        pvalue=fit.pvalue,
        df=int(fit.df),
        groups=group_levels,
        strata=(
            None
            if not stratified or strata_levels is None
            else dict(zip(strata_levels, [int(v) for v in fit.strata], strict=True))
        ),
    )


def survdiff(
    response: Any,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.omit",
    rho: Any = 0,
    timefix: Any = True,
    *,
    group: Any | None = None,
    **kwargs: Any,
) -> SurvDiffResult:
    """R's ``survdiff``: the log-rank (``rho = 0``) and G-rho family tests.

    ``response`` is a formula string (``"Surv(time, status) ~ sex + strata(inst)"``) evaluated
    in ``data``; an ``offset(expected)`` term of expected survival probabilities gives the
    one-sample test.  A ``Surv`` object with ``group`` compares its groups.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.omit")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survdiff got unexpected keyword argument(s): {unexpected}")
    if not _is_bool_like(timefix):
        raise ValueError("invalid value for timefix option")
    rho = _finite_float(rho, "rho")
    if isinstance(response, str):
        y, columns, strata, offset = _formula_inputs(response, data, subset, na_action)
        if group is not None:
            raise ValueError("group is only used with a Surv response")
        _response_check(y)
        if offset is not None:
            return _one_sample(y, offset, rho)
        if not columns:
            raise ValueError("No groups to test")
        codes, levels = _curve_factor(columns, len(y))
    elif isinstance(response, Surv):
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            group = _subset_optional_sequence(group, indices, "group")
        y, aligned = _apply_surv_na_action(response, na_action, "survdiff inputs", group=group)
        _response_check(y)
        if aligned["group"] is None:
            raise ValueError("No groups to test")
        columns = {"group": aligned["group"]}
        codes, _labels = _curve_factor(columns, len(y))
        # a bare vector has no variable name to label its levels with
        levels = list(_strata_factor(columns, shortlabel=True).levels)
        strata = None
    else:
        raise TypeError("The 'formula' argument is not a formula")
    return _k_sample(y, codes, levels, strata, rho, bool(timefix))
