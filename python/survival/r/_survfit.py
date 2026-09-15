"""``survfit`` and its methods, mirroring R's ``survfit.formula`` and its helpers.

``survfit`` builds the model frame (response, curve factor, weights, id, cluster, istate) the
way ``survfit.formula`` does and hands it to one of the three engines: ``survfitKM`` (right
censored or counting-process data), ``survfitAJ`` (multi-state data) or ``survfitTurnbull``
(interval censored data).  ``survfit0``, ``summary.survfit``, ``quantile.survfit``,
``aggregate.survfit`` and ``survfit_confint`` are thin wrappers over their Rust ports.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _encode_labels,
    _finite_float,
    _float_vector,
    _is_bool_like,
    _is_missing_value,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _mstate_categories,
    _mstate_event_label,
    _pop_dotted_keyword,
    _r_factor,
    _strata_level_sort_key,
    _strata_value_label,
    _subset_indices,
    _subset_optional_sequence,
)
from ._coxph import ClogitModel, CoxphModel, survfit_coxph
from ._formula import (
    _apply_formula_na_action,
    _column,
    _column_source,
    _covariate_term_name,
    _cox_survfit_model_frame,
    _formula_columns,
    _formula_response_spec,
    _parse_formula,
    _subset_formula_inputs,
    _surv_response_model_name,
    _term_values,
)
from ._surv import Surv, _apply_surv_na_action, _subset_surv
from ._types import (
    NamedMatrix,
    SummarySurvfitResult,
    SurvfitCall,
    SurvfitMultiStateResult,
    SurvfitQuantileResult,
    SurvfitResult,
    _InteractionTerm,
    _ModelClusterTerm,
    _ModelCovariateTerm,
    _ModelOffsetTerm,
    _ModelStrataTerm,
)

_CONF_TYPES = ("log", "log-log", "plain", "none", "logit", "arcsin")
_CONF_LOWER = ("usual", "peto", "modified")
_SURVFIT_TYPES = ("kaplan-meier", "fleming-harrington", "fh2")
_SPECIALS = ("weights", "id", "cluster", "istate")


# ---------------------------------------------------------------------------
# The model frame of a survfit call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SurvfitData:
    """What ``survfit.formula`` extracts from its model frame before dispatching.

    ``x_codes`` is the 0-based curve of each row and ``x_levels`` the curve labels (R's
    ``X <- strata(mf[ll])``); ``model`` is the model frame itself, stored on the result so
    that ``residuals.survfit`` can re-read it.
    """

    y: Surv
    x_codes: list[int]
    x_levels: list[str]
    weights: list[float] | None
    id: list[Any] | None
    cluster: list[Any] | None
    istate: list[Any] | None
    model: dict[str, Any]
    terms: tuple[str, ...]

    @property
    def n_curves(self) -> int:
        return len(self.x_levels)

    @property
    def strata_codes(self) -> list[int] | None:
        return self.x_codes if self.n_curves > 1 else None

    def id_codes(self) -> list[int] | None:
        """``factor(id, unique(id))`` as 0-based codes: subjects in order of appearance."""

        return None if self.id is None else _encode_labels(self.id, "id")

    def cluster_codes(self) -> list[int] | None:
        return None if self.cluster is None else _encode_labels(self.cluster, "cluster")

    def istate_labels(self) -> tuple[list[str] | None, list[str] | None]:
        """The starting states as strings and, for a factor, its level order."""

        if self.istate is None:
            return None, None
        categories = _mstate_categories(self.istate)
        levels = None if categories is None else [_mstate_event_label(v) for v in categories]
        return [_mstate_event_label(value) for value in self.istate], levels


def _factor_levels(values: Any, materialized: Sequence[Any]) -> list[Any]:
    """R's ``factor(x)`` levels: the factor's own levels, else the sorted unique values."""

    categories = _mstate_categories(values)
    if categories is not None:
        return list(categories)
    unique = {value: None for value in materialized if not _is_missing_value(value)}
    return sorted(unique, key=_strata_level_sort_key)


def _level_codes(values: Any, name: str) -> tuple[list[int | None], list[str]]:
    """The 0-based level code of each value (``None`` when missing) and the level labels."""

    materialized = _materialize_labels(values, name)
    levels = _factor_levels(values, materialized)
    index = {level: code for code, level in enumerate(levels)}
    codes = [None if _is_missing_value(value) else index[value] for value in materialized]
    return codes, [_strata_value_label(level) for level in levels]


def _strata_factor(columns: dict[str, Any], shortlabel: bool) -> _core.StrataResult:
    """R's ``strata()`` on named columns (its label rule is the caller's)."""

    coded = [_level_codes(values, name) for name, values in columns.items()]
    return _core.strata(
        list(columns),
        [labels for _codes, labels in coded],
        [codes for codes, _labels in coded],
        shortlabel=shortlabel,
    )


def _strata_term_values(data: Any, columns: Sequence[str], n: int) -> Any:
    """The column a ``strata(a, b)`` term adds to the model frame: R's ``strata()`` factor."""

    raw = {column: _column_source(data, column) for column in columns}
    shortlabel = all(
        _mstate_categories(values) is not None
        or all(isinstance(value, str) for value in _column(data, column) if value is not None)
        for column, values in raw.items()
    )
    factor = _strata_factor(raw, shortlabel)
    if len(factor.codes) != n:
        raise ValueError("formula columns must have the same length as the Surv response")
    labels = [None if code is None else factor.levels[code] for code in factor.codes]
    return _r_factor(labels, factor.levels)


def _curve_factor(columns: dict[str, Any], n: int) -> tuple[list[int], list[str]]:
    """``X <- strata(mf[ll])``, or ``factor(rep(1, n))`` for ``~ 1``."""

    if not columns:
        return [0] * n, ["1"]
    factor = _strata_factor(columns, shortlabel=False)
    if len(factor.codes) != n:
        raise ValueError("formula columns must have the same length as the Surv response")
    codes = []
    for code in factor.codes:
        if code is None:
            raise ValueError("missing values in the grouping variables")
        codes.append(int(code))
    return codes, list(factor.levels)


def _case_weights(weights: Any | None, n: int) -> list[float] | None:
    if weights is None:
        return None
    try:
        values = _float_vector(weights, "weights")
    except (TypeError, ValueError) as exc:
        raise TypeError("weights must be numeric") from exc
    if len(values) != n:
        raise ValueError("weights must have the same length as the Surv response")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("weights must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("weights must be non-negative")
    return values


def _aligned(values: Any | None, n: int, name: str) -> list[Any] | None:
    if values is None:
        return None
    materialized = _materialize_labels(values, name)
    if len(materialized) != n:
        raise ValueError(f"{name} must have the same length as the Surv response")
    if any(_is_missing_value(value) for value in materialized):
        raise ValueError(f"{name} contains missing values")
    if _mstate_categories(values) is not None:
        return _r_factor(materialized, _mstate_categories(values))
    return materialized


def _survfit_data(
    response: Surv,
    response_name: str,
    columns: dict[str, Any],
    extras: dict[str, Any],
    x_levels: list[str] | None = None,
) -> _SurvfitData:
    """Assemble the model frame and the curve factor from the response and its columns.

    ``extras`` holds the special arguments (``weights``, ``id``, ``cluster``, ``istate``),
    ``columns`` the grouping terms; ``x_levels`` overrides the curve labels.
    """

    n = len(response)
    if n == 0:
        raise ValueError("data set has no non-missing observations")
    x_codes, levels = _curve_factor(columns, n)
    model: dict[str, Any] = {response_name: response, **columns}
    aligned = {name: _aligned(extras[name], n, name) for name in _SPECIALS}
    for name, values in aligned.items():
        if values is not None:
            model[f"({name})"] = values
    return _SurvfitData(
        y=response,
        x_codes=x_codes,
        x_levels=levels if x_levels is None else x_levels,
        weights=_case_weights(aligned["weights"], n),
        id=aligned["id"],
        cluster=aligned["cluster"],
        istate=aligned["istate"],
        model=model,
        terms=tuple(columns),
    )


def _formula_model_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None,
    na_action: str | None,
    extras: dict[str, Any],
) -> _SurvfitData:
    """``model.frame(formula, data, weights, subset, na.action, id, cluster, istate)``."""

    extras = {
        name: _column_source(data, values) if isinstance(values, str) else values
        for name, values in extras.items()
    }
    if subset is not None:
        data, extras = _subset_formula_inputs(formula, data, subset, **extras)
    spec = _formula_response_spec(formula)
    # is.na(Surv): a missing endpoint of an interval-censored response is a censoring code
    exclude = set(spec.columns) if spec.type in {"interval", "interval2"} else set()
    if set(_formula_columns(formula, data)) - exclude or any(
        v is not None for v in extras.values()
    ):
        data, extras = _apply_formula_na_action(
            formula, data, na_action, exclude_columns=exclude, **extras
        )
    response, terms = _parse_formula(formula, data)
    n = len(response)

    cluster_terms = [term for term in terms.model_terms if isinstance(term, _ModelClusterTerm)]
    if cluster_terms:
        if extras["cluster"] is not None:
            raise ValueError("cluster appears as both an argument and a model term")
        if len(cluster_terms) > 1:
            raise ValueError("can not have two cluster terms")
        warnings.warn(
            "use of cluster() in a formula is deprecated; use the 'cluster' argument to the "
            "survfit function",
            DeprecationWarning,
            stacklevel=4,
        )
        extras["cluster"] = _column_source(data, cluster_terms[0].column)
    if terms.offsets:
        warnings.warn("Offset term ignored", stacklevel=4)

    columns: dict[str, Any] = {}
    for model_term in terms.model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            term = model_term.term
            if isinstance(term, _InteractionTerm):
                raise ValueError("Interaction terms are not valid for this function")
            plain = term.transform is None and term.arithmetic is None
            values = _column_source(data, term.column) if plain else _term_values(data, term, n)
            columns[_covariate_term_name(term)] = values
        elif isinstance(model_term, _ModelStrataTerm):
            name = f"strata({', '.join(model_term.columns)})"
            columns[name] = _strata_term_values(data, model_term.columns, n)
        elif not isinstance(model_term, _ModelOffsetTerm | _ModelClusterTerm):
            raise ValueError(f"unsupported survfit formula term {model_term!r}")
    return _survfit_data(response, _surv_response_model_name(spec), columns, extras)


def _surv_model_frame(
    response: Surv,
    *,
    group: Any | None,
    subset: Any | None,
    na_action: str | None,
    extras: dict[str, Any],
) -> _SurvfitData:
    """The model frame of ``survfit(Surv(...), group = )``: the Surv object is the response."""

    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
        extras = {
            name: _subset_optional_sequence(values, indices, name)
            for name, values in extras.items()
        }
    response, aligned = _apply_surv_na_action(
        response, na_action, "survfit inputs", group=group, **extras
    )
    group = aligned.pop("group")
    columns = {} if group is None else {"group": group}
    # a bare vector has no variable name to label its levels with
    levels = None if group is None else list(_strata_factor(columns, shortlabel=True).levels)
    return _survfit_data(response, "response", columns, aligned, levels)


def _survfit_data_from_fit(fit: SurvfitResult | SurvfitMultiStateResult) -> _SurvfitData:
    """Re-read the model frame of a fit, as ``residuals.survfit`` does with ``model.frame``."""

    model = fit.model
    if model is None:
        raise ValueError("the survfit object has no model frame")
    response_name = next((name for name, value in model.items() if isinstance(value, Surv)), None)
    if response_name is None:
        raise ValueError("the model frame of the survfit object has no Surv response")
    columns = {name: model[name] for name in fit.call.terms}
    extras: dict[str, Any] = {name: model.get(f"({name})") for name in _SPECIALS}
    data = _survfit_data(
        model[response_name], response_name, columns, extras, fit.strata_names or None
    )
    if len(fit.strata_names or ["1"]) != data.n_curves:
        raise ValueError("the model frame does not match the curves of the fit")
    return data


# ---------------------------------------------------------------------------
# Argument checks shared by the engines (survfitKM.R, survfitAJ.R, survfitTurnbull.R)
# ---------------------------------------------------------------------------


def _logical(value: Any, message: str) -> bool:
    if not _is_bool_like(value):
        raise ValueError(message)
    return bool(value)


def _survfit_type_codes(type_: Any, stype: Any, ctype: Any) -> tuple[int, int]:
    """R's old-style ``type`` or the ``stype``/``ctype`` arguments as ``(stype, ctype)``."""

    if type_ is not None:
        if not isinstance(type_, str):
            raise ValueError("type argument must be character")
        matched = _match_string_arg(type_, "type", _SURVFIT_TYPES, "invalid value for 'type'")
        return {"kaplan-meier": (1, 1), "fleming-harrington": (2, 1), "fh2": (2, 2)}[matched]
    if isinstance(ctype, bool) or ctype not in (1, 2):
        raise ValueError("ctype must be 1 or 2")
    if isinstance(stype, bool) or stype not in (1, 2):
        raise ValueError("stype must be 1 or 2")
    return int(stype), int(ctype)


def _match_arg(value: Any, name: str, choices: Sequence[str]) -> str:
    quoted = ", ".join(f'"{choice}"' for choice in choices)
    return _match_string_arg(value, name, choices, f"'{name}' should be one of {quoted}")


def _conf_arguments(conf_int: Any, conf_type: Any, conf_lower: Any) -> tuple[float, str, str]:
    """``match.arg`` on conf.type / conf.lower; ``conf.int = FALSE`` means no interval."""

    conf_type = _match_arg(conf_type, "conf.type", _CONF_TYPES)
    conf_lower = _match_arg(conf_lower, "conf.lower", _CONF_LOWER)
    if _is_bool_like(conf_int):
        if not conf_int:
            conf_type = "none"
        conf_int = 0.95
    return _finite_float(conf_int, "conf.int"), conf_type, conf_lower


def _influence_level(influence: Any) -> int:
    """``influence``: TRUE/FALSE (all or nothing) or 0..3."""

    if _is_bool_like(influence):
        return 3 if influence else 0
    if isinstance(influence, int | float):
        if influence not in (0, 1, 2, 3):
            raise ValueError("influence argument must be 0, 1, 2, or 3")
        return int(influence)
    raise ValueError("influence argument must be numeric or logical")


def _start_time_value(start_time: Any | None) -> float | None:
    if start_time is None:
        return None
    if isinstance(start_time, bool | str):
        raise ValueError("start.time must be a single numeric value")
    try:
        value = float(start_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("start.time must be a single numeric value") from exc
    if not math.isfinite(value):
        raise ValueError("start.time must be a single numeric value")
    return value


# ---------------------------------------------------------------------------
# survfit
# ---------------------------------------------------------------------------


def survfit(
    response: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.omit",
    stype: int | None = None,
    ctype: int | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    robust: Any | None = None,
    istate: Any | None = None,
    timefix: Any = True,
    etype: Any | None = None,
    model: Any = False,
    error: Any | None = None,
    entry: Any = False,
    time0: Any = False,
    *,
    group: Any | None = None,
    newdata: Any | None = None,
    se_fit: Any = True,
    conf_int: Any = 0.95,
    conf_type: str = "log",
    conf_lower: str = "usual",
    start_time: Any | None = None,
    influence: Any = False,
    p0: Any | None = None,
    type: str | None = None,
    reverse: Any = False,
    censor: Any = True,
    **kwargs: Any,
) -> Any:
    """R's ``survfit``: Kaplan-Meier / Fleming-Harrington, Aalen-Johansen or Turnbull curves.

    ``response`` is a formula string (``"Surv(time, status) ~ sex"``) evaluated in ``data``, a
    ``Surv`` object (``group`` gives the curves), or a fitted Cox model (``survfit.coxph``, with
    ``newdata`` and ``censor``).  The other arguments are those of ``survfit.formula`` and of
    the engine it dispatches to; the R spellings ``se.fit``, ``conf.int``, ``conf.type``,
    ``conf.lower``, ``start.time`` and ``na.action`` are accepted as keywords.  ``stype`` and
    ``ctype`` default per method as in R (1/1 for ``survfit.formula``, 2 and the tie method
    for ``survfit.coxph``).  ``reverse``
    estimates the censoring distribution (the engine's option).  The model frame is kept on
    the result for ``residuals.survfit`` / ``pseudo`` whatever ``model`` says, as R re-reads
    it through ``model.frame``.
    """

    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, True)
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    conf_lower = _pop_dotted_keyword(kwargs, "conf.lower", "conf_lower", conf_lower, "usual")
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.omit")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit got unexpected keyword argument(s): {unexpected}")
    if isinstance(response, ClogitModel):
        raise ValueError("predicted survival curves are not defined for a clogit model")
    if isinstance(response, CoxphModel):
        # survfit.coxph's second argument is newdata, so survfit(fit, frame) is R's spelling
        return _survfit_coxph(
            response,
            data if newdata is None else newdata,
            se_fit=se_fit,
            conf_int=conf_int,
            conf_type=conf_type,
            start_time=start_time,
            censor=censor,
            model=model,
            stype=2 if stype is None else stype,
            ctype=ctype,
            id=id,
        )
    if newdata is not None:
        raise ValueError("newdata is only used with a fitted Cox model")
    if type is not None and (stype is not None or ctype is not None):
        raise ValueError(
            "cannot have both an old-style 'type' argument and the stype/ctype arguments "
            "that replaced it"
        )
    if stype is None:
        stype = 1
    if ctype is None:
        ctype = 1
    if response is None:
        raise ValueError("a formula argument is required")
    if not _is_bool_like(timefix):
        raise ValueError("invalid value for timefix option")
    if etype is not None:
        raise ValueError(
            "the etype argument is no longer supported, use a factor as the status variable"
        )
    time0 = _logical(time0, "time0 must be TRUE/FALSE")
    entry = _logical(entry, "entry argument must be TRUE/FALSE")
    _logical(model, "model must be TRUE/FALSE")

    extras = {"weights": weights, "id": id, "cluster": cluster, "istate": istate}
    if isinstance(response, str):
        frame = _formula_model_frame(
            response, data, subset=subset, na_action=na_action, extras=extras
        )
    elif isinstance(response, Surv):
        frame = _surv_model_frame(
            response, group=group, subset=subset, na_action=na_action, extras=extras
        )
    else:
        raise TypeError("response must be a survival object")

    common = {
        "se_fit": se_fit,
        "conf_int": conf_int,
        "conf_type": conf_type,
        "conf_lower": conf_lower,
        "start_time": start_time,
        "type_": type,
        "robust": robust,
        "timefix": bool(timefix),
        "time0": time0,
        "id_name": id if isinstance(id, str) else None,
    }
    surv_type = frame.y.type
    if surv_type in {"left", "interval", "interval2"}:
        return _survfitTurnbull(frame, **common)
    if surv_type in {"right", "counting"}:
        return _survfitKM(
            frame,
            stype=stype,
            ctype=ctype,
            influence=influence,
            entry=entry,
            reverse=reverse,
            **common,
        )
    return _survfitAJ(
        frame, stype=stype, ctype=ctype, influence=influence, entry=entry, p0=p0, **common
    )


def _survfit_coxph(
    fit: Any,
    newdata: Any | None,
    *,
    se_fit: Any,
    conf_int: Any,
    conf_type: Any,
    start_time: Any | None,
    censor: Any,
    model: Any,
    stype: Any,
    ctype: Any,
    id: Any | None,
) -> Any:
    """``survfit.coxph``: the Cox module owns the curves, this is only the dispatch."""

    result = survfit_coxph(
        fit,
        newdata,
        se_fit=se_fit,
        conf_int=conf_int,
        conf_type=conf_type,
        censor=censor,
        start_time=_start_time_value(start_time),
        stype=stype,
        ctype=ctype,
        id=id,
    )
    if _logical(model, "model must be TRUE/FALSE") and hasattr(result, "model"):
        return dataclasses.replace(result, model=_cox_survfit_model_frame(fit, newdata))
    return result


# ---------------------------------------------------------------------------
# survfitKM
# ---------------------------------------------------------------------------


def _survfitKM(
    frame: _SurvfitData,
    *,
    stype: Any,
    ctype: Any,
    type_: Any,
    se_fit: Any,
    conf_int: Any,
    conf_type: Any,
    conf_lower: Any,
    start_time: Any,
    robust: Any,
    influence: Any,
    entry: bool,
    reverse: Any,
    timefix: bool,
    time0: bool,
    id_name: str | None,
) -> SurvfitResult:
    """``survfitKM``: the argument checks, then one call of the engine for all curves."""

    stype, ctype = _survfit_type_codes(type_, stype, ctype)
    conf_int, conf_type, conf_lower = _conf_arguments(conf_int, conf_type, conf_lower)
    se_fit = _logical(se_fit, "se.fit must be TRUE/FALSE")
    reverse = _logical(reverse, "reverse must be TRUE/FALSE")
    influence = _influence_level(influence)
    if robust is not None:
        robust = _logical(robust, "robust must be TRUE/FALSE")
        if frame.cluster is not None and not robust:
            warnings.warn("cluster specified with robust=FALSE, cluster ignored", stacklevel=4)
        if influence > 0 and not robust:
            warnings.warn("robust=FALSE implies influence=FALSE", stacklevel=4)
    start = _start_time_value(start_time)
    engine = _core.survfitkm(
        list(frame.y.time),
        [int(value) for value in frame.y.event],
        start=None if frame.y.start is None else list(frame.y.start),
        weights=frame.weights,
        strata=frame.strata_codes,
        id=frame.id_codes(),
        cluster=frame.cluster_codes(),
        stype=stype,
        ctype=ctype,
        se_fit=se_fit,
        conf_int=conf_int,
        conf_type=conf_type,
        conf_lower=conf_lower,
        start_time=start,
        robust=robust,
        influence=influence,
        entry=entry,
        timefix=timefix,
        reverse=reverse,
    )
    call = SurvfitCall(frame.terms, stype, ctype, timefix, start, id=id_name)
    labels = _curve_labels(engine, frame.x_levels)
    return _km_result(engine, labels, call, frame.model, se_fit, time0=time0)


def _curve_labels(
    engine: _core.SurvfitKMResult | _core.SurvfitAJResult, levels: Sequence[str]
) -> list[str]:
    """The label of each curve the engine fitted, in its order (``levels[strata_codes]``).

    A curve that ``start.time`` emptied keeps its ``n`` of 0 but is not fitted, so the
    engine's ``strata`` is shorter than ``levels``.
    """

    codes = engine.strata_codes
    return list(levels) if codes is None else [levels[code] for code in codes]


def _strata_table(
    engine: _core.SurvfitKMResult | _core.SurvfitAJResult, labels: Sequence[str]
) -> dict[str, int] | None:
    """R's named ``strata`` vector, ``temp$strata[temp$strata > 0]``, from the fitted curves."""

    if engine.strata is None:
        return None
    return {label: int(size) for label, size in zip(labels, engine.strata, strict=True) if size > 0}


def _km_result(
    engine: _core.SurvfitKMResult,
    labels: Sequence[str],
    call: SurvfitCall,
    model: dict[str, Any] | None,
    se_fit: bool,
    *,
    time0: bool,
) -> SurvfitResult:
    """A ``survfit`` object from the engine output; ``se.fit = FALSE`` drops the se parts."""

    return SurvfitResult(
        n=[int(value) for value in engine.n],
        time=engine.time,
        n_risk=engine.n_risk,
        n_event=engine.n_event,
        n_censor=engine.n_censor,
        surv=engine.surv,
        cumhaz=engine.cumhaz,
        type=engine.type,
        t0=engine.t0,
        n_enter=engine.n_enter,
        counts=engine.counts,
        std_err=engine.std_err if se_fit else None,
        std_chaz=engine.std_chaz if se_fit else None,
        lower=engine.lower if se_fit else None,
        upper=engine.upper if se_fit else None,
        strata=_strata_table(engine, labels),
        n_id=None if engine.n_id is None else [int(value) for value in engine.n_id],
        logse=engine.logse if se_fit else None,
        conf_int=engine.conf_int if se_fit else None,
        conf_type=engine.conf_type if se_fit else None,
        conf_lower=engine.conf_lower if se_fit and engine.conf_lower != "usual" else None,
        influence_surv=engine.influence_surv,
        influence_chaz=engine.influence_chaz,
        start_time=call.start_time,
        time0=time0,
        call=call,
        model=model,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# survfitAJ
# ---------------------------------------------------------------------------


def _survfitAJ(
    frame: _SurvfitData,
    *,
    stype: Any,
    ctype: Any,
    type_: Any,
    se_fit: Any,
    conf_int: Any,
    conf_type: Any,
    conf_lower: Any,
    start_time: Any,
    robust: Any,
    influence: Any,
    entry: bool,
    p0: Any | None,
    timefix: bool,
    time0: bool,
    id_name: str | None,
) -> SurvfitMultiStateResult:
    """``survfitAJ``: the Aalen-Johansen estimate of the probability in state."""

    stype, ctype = _survfit_type_codes(type_, stype, ctype)
    if stype != 1 or ctype != 1:
        warnings.warn("only stype=1, ctype=1 implimented for multi-state data", stacklevel=4)
    conf_int, conf_type, conf_lower = _conf_arguments(conf_int, conf_type, conf_lower)
    if conf_lower != "usual":
        warnings.warn("conf.lower is ignored for multi-state data", stacklevel=4)
    se_fit = _logical(se_fit, "se.fit must be TRUE/FALSE")
    if robust is not None and not _logical(robust, "robust must be TRUE/FALSE"):
        raise ValueError("multi-state survfit supports only a robust variance")
    if frame.id is None and frame.y.start is not None:
        raise ValueError("id statement is required")
    if p0 is not None:
        p0 = _float_vector(p0, "p0")
        if abs(sum(p0) - 1.0) > 1.5e-8 * max(1.0, abs(sum(p0))):
            raise ValueError("p0 must be a numeric vector that adds to 1")
    istate, istate_levels = frame.istate_labels()
    start = _start_time_value(start_time)
    engine = _core.survfitaj(
        list(frame.y.time),
        [int(value) for value in frame.y.event],
        list(frame.y.states),
        start=None if frame.y.start is None else list(frame.y.start),
        weights=frame.weights,
        strata=frame.strata_codes,
        id=frame.id_codes(),
        istate=istate,
        istate_levels=istate_levels,
        cluster=frame.cluster_codes(),
        se_fit=se_fit,
        conf_int=conf_int,
        conf_type=conf_type,
        influence=_influence_level(influence) > 0,
        start_time=start,
        p0=p0,
        entry=entry,
        time0=time0,
        timefix=timefix,
    )
    call = SurvfitCall(frame.terms, stype, ctype, timefix, start, p0=p0, id=id_name)
    labels = _curve_labels(engine, frame.x_levels)
    return _aj_result(engine, labels, call, frame.model, se_fit, time0=time0)


def _compact_transitions(table: Sequence[Sequence[float]], states: Sequence[str]) -> NamedMatrix:
    """``survcheck2``'s transitions table: drop empty columns and never-occurring states.

    The engine's table is ``states x (states + censored)``; R keeps the columns with a
    transition and the rows of states that occur at all (as a source or a target).
    """

    n_states = len(states)
    columns = [*states, "(censored)"]
    keep_columns = [j for j in range(n_states + 1) if any(row[j] > 0 for row in table)]
    keep_rows = [i for i in range(n_states) if sum(table[i]) + sum(row[i] for row in table) > 0]
    return NamedMatrix(
        rownames=[states[i] for i in keep_rows],
        colnames=[columns[j] for j in keep_columns],
        values=[[float(table[i][j]) for j in keep_columns] for i in keep_rows],
    )


def _aj_result(
    engine: _core.SurvfitAJResult,
    labels: Sequence[str],
    call: SurvfitCall,
    model: dict[str, Any] | None,
    se_fit: bool,
    *,
    time0: bool,
) -> SurvfitMultiStateResult:
    """A ``survfitms`` object from the engine output."""

    states = list(engine.states)
    with_ci = se_fit and engine.lower is not None
    return SurvfitMultiStateResult(
        n=[int(value) for value in engine.n],
        time=engine.time,
        n_risk=engine.n_risk,
        n_event=engine.n_event,
        n_censor=engine.n_censor,
        n_transition=engine.n_transition,
        pstate=engine.pstate,
        cumhaz=engine.cumhaz,
        p0=engine.p0,
        states=states,
        hazard_names=[
            f"{source + 1}:{target + 1}"
            for source, target in zip(engine.hazard_from, engine.hazard_to, strict=True)
        ],
        transitions=_compact_transitions(engine.transitions, states),
        n_id=[int(value) for value in engine.n_id],
        type=engine.type,
        t0=engine.t0,
        n_enter=engine.n_enter,
        counts=engine.counts,
        std_err=engine.std_err if se_fit else None,
        std_chaz=engine.std_chaz if se_fit else None,
        std_auc=engine.std_auc if se_fit else None,
        se0=engine.se0 if se_fit else None,
        lower=engine.lower if se_fit else None,
        upper=engine.upper if se_fit else None,
        strata=_strata_table(engine, labels),
        logse=False if se_fit else None,
        conf_int=engine.conf_int if with_ci else None,
        conf_type=engine.conf_type if with_ci else None,
        influence_pstate=engine.influence_pstate,
        start_time=engine.start_time,
        time0=time0,
        call=call,
        model=model,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# survfitTurnbull
# ---------------------------------------------------------------------------


def _interval_coding(y: Surv) -> tuple[list[float], list[float], list[int]]:
    """R's ``Surv(type = "interval")`` columns: time1, time2 (interval rows) and status."""

    status = [int(value) for value in y.event]
    if y.type == "left":
        return list(y.time), list(y.time), [2 if value == 0 else 1 for value in status]
    # Surv already stores a left-censored row's right end in time1 (time2 is R's placeholder 1)
    time2 = [math.nan if value is None else float(value) for value in y.time2 or ()]
    return list(y.time), time2, status


def _survfitTurnbull(
    frame: _SurvfitData,
    *,
    type_: Any,
    se_fit: Any,
    conf_int: Any,
    conf_type: Any,
    conf_lower: Any,
    start_time: Any,
    robust: Any,
    timefix: bool,
    time0: bool,
    id_name: str | None,
) -> SurvfitResult:
    """``survfitTurnbull``: the EM estimate for interval censored data, one curve per level."""

    if type_ is not None:
        _match_string_arg(type_, "type", _SURVFIT_TYPES, "invalid value for 'type'")
    conf_int, conf_type, _conf_lower = _conf_arguments(conf_int, conf_type, conf_lower)
    se_fit = _logical(se_fit, "se.fit must be TRUE/FALSE")
    if frame.y.start is not None:
        raise ValueError("survfitTurnbull not appropriate for counting process data")
    start = _start_time_value(start_time)
    time1, time2, status = _interval_coding(frame.y)
    rows = list(range(len(frame.y)))
    if start is not None:
        rows = [row for row in rows if time1[row] >= start]
        if not rows:
            label = _strata_value_label(start)
            raise ValueError(f"start.time = {label} is greater than all time points.")
    strata_codes = frame.strata_codes
    weights = frame.weights
    result = _core.turnbull(
        [time1[row] for row in rows],
        [time2[row] for row in rows],
        [status[row] for row in rows],
        weights=None if weights is None else [weights[row] for row in rows],
        group=None if strata_codes is None else [strata_codes[row] for row in rows],
        conf_level=conf_int,
        conf_type=conf_type,
        timefix=timefix,
    )
    curves = result.curves
    sizes = [0] * frame.n_curves
    for row in rows:
        sizes[frame.x_codes[row]] += 1
    levels = [level for level, size in zip(frame.x_levels, sizes, strict=True) if size > 0]
    with_ci = se_fit and conf_type != "none"

    def stack(name: str) -> list[float]:
        return [value for curve in curves for value in getattr(curve, name)]

    surv = stack("surv")
    return SurvfitResult(
        n=[int(curve.n) for curve in curves],
        time=stack("time"),
        n_risk=stack("n_risk"),
        n_event=stack("n_event"),
        n_censor=stack("n_censor"),
        surv=surv,
        cumhaz=[-math.log(value) if value > 0.0 else math.inf for value in surv],
        type="interval",
        t0=0.0 if start is None else start,
        std_err=stack("std_err") if se_fit else None,
        lower=stack("lower") if with_ci else None,
        upper=stack("upper") if with_ci else None,
        strata=(
            {level: len(curve.time) for level, curve in zip(levels, curves, strict=True)}
            if frame.n_curves > 1
            else None
        ),
        logse=True if se_fit else None,
        conf_int=conf_int if se_fit else None,
        conf_type=conf_type if se_fit else None,
        start_time=start,
        time0=time0,
        call=SurvfitCall(frame.terms, 1, 1, timefix, start, id=id_name),
        model=frame.model,
    )


# ---------------------------------------------------------------------------
# survfit0, summary.survfit, quantile.survfit, aggregate.survfit, survfit_confint
# ---------------------------------------------------------------------------


def _engine_of(x: Any) -> Any:
    if not isinstance(x, SurvfitResult | SurvfitMultiStateResult):
        raise TypeError("function requires a survfit object")
    if x.engine is None:
        raise NotImplementedError(
            "this method is not available for interval-censored (Turnbull) curves"
        )
    return x.engine


def survfit0(x: Any, *args: Any, **kwargs: Any) -> Any:
    """R's ``survfit0``: add the row at the starting time ``t0`` to every curve.

    A fit made with ``time0 = TRUE`` (or already processed) is returned as is.
    """

    if args or kwargs:
        raise TypeError("survfit0 takes a single survfit object")
    if not isinstance(x, SurvfitResult | SurvfitMultiStateResult):
        raise TypeError("function requires a survfit object")
    if x.time0:
        return x
    engine = _engine_of(x)
    if isinstance(x, SurvfitMultiStateResult):
        fit0 = _core.survfit0_aj(engine)
        return _aj_result(fit0, x.strata_names, x.call, x.model, x.std_err is not None, time0=True)
    return _km_result(
        _core.survfit0(engine), x.strata_names, x.call, x.model, x.std_err is not None, time0=True
    )


def _rmean_option(rmean: Any, fit: SurvfitResult) -> str:
    """``rmean``: ``"none"``, ``"common"``, ``"individual"`` or a truncation time."""

    if rmean is None:
        return "common"
    if isinstance(rmean, str):
        return _match_string_arg(
            rmean, "rmean", ("none", "common", "individual"), "Invalid value for rmean option"
        )
    value = _finite_float(rmean, "rmean")
    smallest = fit.start_time if fit.start_time is not None else min(fit.time)
    if value < smallest:
        raise ValueError("Truncation point for the mean time in state is < smallest survival")
    return repr(value)


def summary_survfit(
    object: Any,
    times: Any | None = None,
    censored: Any = False,
    scale: Any = 1,
    extend: Any = False,
    rmean: Any | None = None,
) -> SummarySurvfitResult:
    """R's ``summary.survfit``: the curves at their event times (or at ``times``) and the table.

    ``table`` is ``survmean``'s per-curve summary (records, n.max or n.id, n.start, events,
    the restricted mean and its se for ``rmean``, the median and its confidence limits).
    """

    if isinstance(object, SurvfitMultiStateResult):
        raise NotImplementedError("summary.survfitms is not available (no Rust kernel yet)")
    if not isinstance(object, SurvfitResult):
        raise TypeError("summary.survfit can only be used for survfit and survfit.coxph objects")
    censored = _logical(censored, "censored must be TRUE/FALSE")
    extend = _logical(extend, "extend must be TRUE/FALSE")
    scale = _finite_float(scale, "scale")
    engine = _engine_of(object)
    rmean_option = _rmean_option(rmean, object)
    table = _core.survmean(_core.survfit0(engine), scale, rmean_option)
    if times is None:
        rows = _core.summary_survfit(engine, censored=censored)
    else:
        times = _float_vector([times] if isinstance(times, int | float) else times, "times")
        if not times:
            raise ValueError("no values in times vector")
        if any(not math.isfinite(value) for value in times):
            raise ValueError("times contains missing values")
        rows = _core.summary_survfit(engine, times=times, extend=extend)
    strata_names = object.strata_names
    strata = None
    if rows.strata is not None:
        strata = [
            name for name, size in zip(strata_names, rows.strata, strict=True) for _ in range(size)
        ]
    return SummarySurvfitResult(
        time=[value / scale for value in rows.time],
        n_risk=rows.n_risk,
        n_event=rows.n_event,
        n_censor=rows.n_censor,
        surv=rows.surv,
        cumhaz=rows.cumhaz,
        strata=strata,
        table=_summary_table(table, strata_names, object),
        n=[int(value) for value in rows.n],
        n_enter=rows.n_enter,
        std_err=rows.std_err,
        std_chaz=rows.std_chaz,
        lower=rows.lower,
        upper=rows.upper,
        rmean_endtime=None if rmean_option == "none" else table.end_time,
        conf_int=object.conf_int,
        conf_type=object.conf_type,
    )


def _summary_table(
    table: _core.SurvmeanTable, strata_names: Sequence[str], fit: SurvfitResult
) -> NamedMatrix:
    """``survmean``'s matrix with R's column names."""

    columns: list[tuple[str, Sequence[float]]] = [
        ("records", table.records),
        ("n.id" if fit.n_id is not None else "n.max", table.n_max),
        ("n.start", table.n_start),
        ("events", table.events),
    ]
    if table.rmean is not None and table.se_rmean is not None:
        columns += [("rmean", table.rmean), ("se(rmean)", table.se_rmean)]
    columns.append(("median", table.median))
    if table.lower is not None and table.upper is not None:
        level = _strata_value_label(fit.conf_int if fit.conf_int is not None else 0.95)
        columns += [(f"{level}LCL", table.lower), (f"{level}UCL", table.upper)]
    return NamedMatrix(
        rownames=list(strata_names) if strata_names else None,
        colnames=[name for name, _values in columns],
        values=[
            [float(values[curve]) for _name, values in columns]
            for curve in range(len(table.records))
        ],
    )


def quantile_survfit(
    x: Any,
    probs: Any = (0.25, 0.5, 0.75),
    conf_int: Any = True,
    scale: Any = 1,
    tolerance: Any | None = None,
    **kwargs: Any,
) -> SurvfitQuantileResult:
    """R's ``quantile.survfit``: the quantiles of each curve and of its confidence bands."""

    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"quantile_survfit got unexpected keyword argument(s): {unexpected}")
    if isinstance(x, SurvfitMultiStateResult):
        raise ValueError("quantiles are not a well defined quantity for multi-state models")
    if not isinstance(x, SurvfitResult):
        raise TypeError("Must be a survfit object")
    probs = _float_vector([probs] if isinstance(probs, int | float) else probs, "probs")
    if any(math.isnan(value) for value in probs):
        raise ValueError("invalid probability")
    if any(value < 0.0 or value > 1.0 for value in probs):
        raise ValueError("Invalid probability")
    result = _core.quantile_survfit(
        _engine_of(x),
        probs,
        conf_int=_logical(conf_int, "conf.int must be TRUE/FALSE"),
        scale=_finite_float(scale, "scale"),
        tolerance=None if tolerance is None else _finite_float(tolerance, "tolerance"),
    )
    return SurvfitQuantileResult(
        probs=result.probs,
        quantile=result.quantile,
        strata=x.strata_names or None,
        lower=result.lower,
        upper=result.upper,
    )


def _grouping_factors(by: Any, n_data: int) -> list[_core.GroupingFactor]:
    """R's ``by`` argument as level codes: a vector, a list of vectors or a named mapping."""

    if by is None:
        return []
    if isinstance(by, dict):
        items: list[tuple[str | None, Any]] = [(str(name), values) for name, values in by.items()]
    elif isinstance(by, list | tuple) and by and isinstance(by[0], list | tuple):
        items = [(None, values) for values in by]
    else:
        items = [(None, by)]
    factors = []
    for name, values in items:
        codes, labels = _level_codes(values, "by")
        if len(codes) != n_data:
            raise ValueError("arguments must have the same length")
        if any(code is None for code in codes):
            raise ValueError("by contains missing values")
        factors.append(_core.GroupingFactor([int(code) for code in codes], labels, name))
    return factors


def aggregate_survfit(x: Any, by: Any | None = None, FUN: str = "mean") -> Any:
    """R's ``aggregate.survfit``: population-averaged curves of ``survfit(coxfit, newdata)``.

    ``x`` has a ``surv`` matrix (times x newdata rows) or a ``pstate`` array (times x rows x
    states); the rows are summarised within the groups of ``by`` (a vector, a list of vectors
    or a name -> vector mapping) with ``FUN``, one of ``"mean"`` (the default), ``"median"``,
    ``"min"`` or ``"max"``.  The components that do not collapse (``std_err``, ``lower``,
    ``upper``, ``cumhaz``, ...) are dropped as in R and ``newdata`` becomes the group labels.
    """

    surv = getattr(x, "surv", None)
    pstate = getattr(x, "pstate", None)
    surv = surv if surv and isinstance(surv[0], list | tuple) else None
    pstate = pstate if pstate and pstate[0] and isinstance(pstate[0][0], list | tuple) else None
    if surv is None and pstate is None:
        raise ValueError("survfit object does not have a 'data' margin")
    n_data = len(surv[0]) if surv is not None else len(pstate[0])  # type: ignore[index]
    if not isinstance(FUN, str):
        raise TypeError("FUN must be the name of a summary: mean, median, min or max")
    result = _core.aggregate_survfit(
        surv=surv, pstate=pstate, by=_grouping_factors(by, n_data), fun=FUN
    )
    if not dataclasses.is_dataclass(x) or isinstance(x, type):
        return result
    names = {field.name for field in dataclasses.fields(x)}
    updates: dict[str, Any] = {
        name: None
        for name in ("std_err", "std_chaz", "lower", "upper", "conf_int", "conf_type", "logse")
        if name in names
    }
    if "cumhaz" in names:
        updates["cumhaz"] = [] if isinstance(x.cumhaz, list) else None
    if result.surv is not None:
        updates["surv"] = result.surv
    if result.pstate is not None:
        updates["pstate"] = result.pstate
    if "newdata" in names:
        groups = result.newdata
        updates["newdata"] = (
            None
            if groups is None
            else {
                name: [labels[column] for labels in groups.labels]
                for column, name in enumerate(groups.names)
            }
        )
    return dataclasses.replace(x, **updates)


def aggregate_survfit_result(result: Any, groups: Any | None = None) -> Any:
    """The R bridge's entry point: ``groups`` are the integer codes it built from ``by``."""

    return aggregate_survfit(result, by=groups)


def survfit_confint(
    p: Any,
    se: Any,
    logse: Any = True,
    conf_type: str | None = None,
    conf_int: Any = 0.95,
    selow: Any | None = None,
    ulimit: Any = True,
    **kwargs: Any,
) -> _core.ConfidenceBands:
    """R's ``survfit_confint``: confidence limits for a survival estimate.

    ``se`` is the standard error of ``log(p)`` when ``logse`` is true and of ``p`` otherwise;
    ``selow`` replaces it for the lower limit (``conf.lower``).
    """

    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, None)
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit_confint got unexpected keyword argument(s): {unexpected}")
    if conf_type is None:
        raise TypeError('argument "conf.type" is missing, with no default')
    if not isinstance(conf_type, str) or conf_type not in _CONF_TYPES or conf_type == "none":
        raise ValueError("invalid conf.int type")
    return _core.survfit_confint(
        _float_vector(p, "p"),
        _float_vector(se, "se"),
        logse=_logical(logse, "logse must be TRUE/FALSE"),
        conf_type=conf_type,
        conf_int=_finite_float(conf_int, "conf.int"),
        selow=None if selow is None else _float_vector(selow, "selow"),
        ulimit=_logical(ulimit, "ulimit must be TRUE/FALSE"),
    )


# ---------------------------------------------------------------------------
# The influence matrices the R bridge's survfitKM asks for
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvfitKMInfluence:
    """The per-cluster influence on ``surv`` and on ``cumhaz`` (rows clusters, columns times)."""

    influence_surv: list[list[float]]
    influence_chaz: list[list[float]]


def _influence_matrices(engine: _core.SurvfitKMResult) -> SurvfitKMInfluence:
    surv = engine.influence_surv
    chaz = engine.influence_chaz
    if surv is None or chaz is None:
        raise RuntimeError("the engine did not return the influence matrices")
    return SurvfitKMInfluence(surv[0].values, chaz[0].values)


def survfitkm_influence(
    time: Any,
    status: Any,
    cluster: Any,
    weights: Any | None = None,
    stype: int = 1,
    ctype: int = 1,
    conf_level: float = 0.95,
    conf_type: str = "log",
) -> SurvfitKMInfluence:
    """``survfitKM(..., influence = 3)`` for right-censored data: the influence matrices."""

    return _influence_matrices(
        _core.survfitkm(
            _float_vector(time, "time"),
            [int(value) for value in _materialize_1d(status, "status")],
            weights=None if weights is None else _float_vector(weights, "weights"),
            cluster=_encode_labels(_materialize_labels(cluster, "cluster"), "cluster"),
            stype=stype,
            ctype=ctype,
            conf_int=conf_level,
            conf_type=conf_type,
            robust=True,
            influence=3,
        )
    )


def survfitkm_counting_influence(
    start: Any,
    stop: Any,
    status: Any,
    cluster: Any,
    weights: Any | None = None,
    stype: int = 1,
    ctype: int = 1,
    conf_level: float = 0.95,
    conf_type: str = "log",
    **kwargs: Any,
) -> SurvfitKMInfluence:
    """``survfitKM(..., influence = 3)`` for counting-process data: the influence matrices.

    The bridge also passes the curve it already holds (``curve_time``, ``curve_estimate``);
    the engine recomputes it, so those two are accepted and ignored.
    """

    kwargs.pop("curve_time", None)
    kwargs.pop("curve_estimate", None)
    if kwargs:
        raise TypeError(f"unexpected argument(s): {', '.join(sorted(kwargs))}")
    return _influence_matrices(
        _core.survfitkm(
            _float_vector(stop, "stop"),
            [int(value) for value in _materialize_1d(status, "status")],
            start=_float_vector(start, "start"),
            weights=None if weights is None else _float_vector(weights, "weights"),
            cluster=_encode_labels(_materialize_labels(cluster, "cluster"), "cluster"),
            stype=stype,
            ctype=ctype,
            conf_int=conf_level,
            conf_type=conf_type,
            robust=True,
            influence=3,
        )
    )


def _optional_float_list(result: Any, name: str) -> list[float] | None:
    # ``_models.as_data_frame`` still imports this for the pre-2.0 raw survfit outputs.
    values = getattr(result, name, None)
    return None if values is None else [float(value) for value in values]
