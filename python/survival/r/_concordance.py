"""``concordance``/``concordancefit`` (R/concordance.R) and the deprecated
``survConcordance`` entry points, on the Rust ``concordancefit`` kernel."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_matrix_rows,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _label_levels,
    _materialize_labels,
    _normalize_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
)
from ._coxph import CoxphModel, predict_coxph
from ._fit import _model_frame, _newdata_frame
from ._surv import Surv
from ._types import ConcordanceResult

_TIMEWT_CHOICES = ("n", "S", "S/G", "n/G2", "I")
_COUNT_NAMES = ("concordant", "discordant", "tied.x", "tied.y", "tied.xy")


def _timewt_name(timewt: Any) -> str:
    if not isinstance(timewt, str):
        raise TypeError("timewt must be a string")
    for choice in _TIMEWT_CHOICES:
        if choice.lower() == timewt.strip().lower():
            return choice
    raise ValueError("timewt must be one of n, S, S/G, n/G2, I")


def _time_bound(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool | str | bytes):
        raise TypeError(f"{name} must be a single number")
    try:
        return _finite_float(value, name)
    except TypeError as exc:
        raise TypeError(f"{name} must be a single number") from exc


def _cluster_codes(values: Sequence[Any]) -> list[int]:
    index = {value: idx for idx, value in enumerate(_label_levels(list(values), "cluster"))}
    return [index[value] for value in values]


def _count_row(counts: Any) -> dict[str, float]:
    return dict(
        zip(
            _COUNT_NAMES,
            (counts.concordant, counts.discordant, counts.tied_x, counts.tied_y, counts.tied_xy),
            strict=True,
        )
    )


def _ranks_rows(ranks: Any) -> list[dict[str, float]]:
    return [
        {"time": t, "rank": r, "timewt": tw, "casewt": cw}
        for t, r, tw, cw in zip(ranks.time, ranks.rank, ranks.timewt, ranks.casewt, strict=True)
    ]


def _result(
    cfit: _core.ConcordanceFit,
    names: Sequence[str] | None,
    strata_levels: Sequence[str],
    formula: str | None,
) -> ConcordanceResult:
    """Lay a ``ConcordanceFit`` out the way R's ``concordancefit`` list is."""

    nvar = len(cfit.concordance)
    single = nvar == 1
    count_rows = [_count_row(row) for row in cfit.count]
    count_names: list[str] | None = None
    if cfit.count_strata is not None:
        count_names = [str(strata_levels[int(code)]) for code in cfit.count_strata]
    elif not single:
        count_names = list(names) if names is not None else [f"X{i + 1}" for i in range(nvar)]
    var: Any = None
    if cfit.var is not None:
        var = cfit.var[0][0] if single else [list(row) for row in cfit.var]
    dfbeta: Any = None
    if cfit.dfbeta is not None:
        rows = [list(row) for row in cfit.dfbeta]
        dfbeta = [row[0] for row in rows] if single else rows
    influence: Any = None
    if cfit.influence is not None:
        matrices = [[list(row) for row in matrix] for matrix in cfit.influence]
        influence = matrices[0] if single else matrices
    ranks: Any = None
    if cfit.ranks is not None:
        tables = [_ranks_rows(table) for table in cfit.ranks]
        ranks = tables[0] if single else tables
    return ConcordanceResult(
        concordance=cfit.concordance[0] if single else list(cfit.concordance),
        count=count_rows[0] if len(count_rows) == 1 and count_names is None else count_rows,
        n=int(cfit.n),
        names=count_names if count_names is not None else (list(names) if names else None),
        var=var,
        cvar=None if cfit.cvar is None else (cfit.cvar[0] if single else list(cfit.cvar)),
        dfbeta=dfbeta,
        influence=influence,
        ranks=ranks,
        formula=formula,
    )


def concordancefit(
    y: Any,
    x: Any,
    strata: Any | None = None,
    weights: Any | None = None,
    ymin: Any | None = None,
    ymax: Any | None = None,
    timewt: Any = "n",
    cluster: Any | None = None,
    influence: Any = 0,
    ranks: Any = False,
    reverse: Any = False,
    timefix: Any = True,
    keepstrata: Any = 10,
    std_err: Any = True,
    *,
    names: Sequence[str] | None = None,
    strata_levels: Sequence[str] | None = None,
    formula: str | None = None,
    **kwargs: Any,
) -> ConcordanceResult:
    """R's ``concordancefit``: the concordance of ``y`` (a Surv, or a numeric
    vector) with one or more predictor columns ``x``."""

    std_err = _pop_dotted_keyword(kwargs, "std.err", "std_err", std_err, True)
    if kwargs:
        raise TypeError(f"concordancefit got unexpected argument(s): {', '.join(sorted(kwargs))}")
    if not isinstance(y, Surv):
        y = Surv(_float_vector(y, "y"))
    if y.type in {"left", "interval"}:
        raise ValueError("left or interval censored data is not supported")
    if y.type in {"mright", "mcounting"}:
        raise ValueError("multiple state survival is not supported")
    n = len(y)
    raw = (
        _as_matrix_rows(x, "x", allow_empty_columns=False)
        if _is_matrix(x)
        else [[v] for v in _float_vector(x, "x")]
    )
    if len(raw) != n:
        raise ValueError("x and y are not the same length")
    nvar = len(raw[0]) if raw else 0
    timewt_name = _timewt_name(timewt)
    if y.start is not None and timewt_name in {"S/G", "n/G2"}:
        raise ValueError(f"{timewt_name} timewt option not supported for (time1, time2) data")
    strata_codes: list[int] | None = None
    levels: list[str] = []
    if strata is not None:
        labels = _materialize_labels(strata, "strata")
        if len(labels) != n:
            raise ValueError("y and strata are not the same length")
        raw_levels = list(strata_levels) if strata_levels else list(_label_levels(labels, "strata"))
        levels = [str(level) for level in raw_levels]
        index = {level: idx for idx, level in enumerate(raw_levels)}
        strata_codes = [index[label] for label in labels]
    weight_values = _optional_float_vector(weights, "weights", n)
    if weight_values is not None and len(weight_values) != n:
        raise ValueError("y and weights are not the same length")
    influence_value = _integer_scalar(influence, "influence")
    if influence_value not in (0, 1, 2, 3):
        raise ValueError("influence must be 0, 1, 2 or 3")
    if isinstance(keepstrata, bool):
        keep = 10**9 if keepstrata else 0
    else:
        keep = _integer_scalar(keepstrata, "keepstrata")
    if not _normalize_bool_option(std_err, "std_err"):
        ranks, influence_value = False, 0
    common: dict[str, Any] = {
        "weights": None if weight_values is None else _core.Weights(weight_values),
        "strata": strata_codes,
        "cluster": None
        if cluster is None
        else _cluster_codes(_materialize_labels(cluster, "cluster")),
        "timewt": timewt_name,
        "ymin": _time_bound(ymin, "ymin"),
        "ymax": _time_bound(ymax, "ymax"),
        "influence": influence_value,
        "ranks": _normalize_bool_option(ranks, "ranks"),
        "reverse": _normalize_bool_option(reverse, "reverse"),
        "timefix": _normalize_bool_option(timefix, "timefix"),
        "keepstrata": keep,
        "std_err": _normalize_bool_option(std_err, "std_err"),
    }
    matrix = _core.CovariateMatrix([value for row in raw for value in row], n, nvar)
    if y.start is None:
        cfit = _core.concordancefit(
            _core.SurvivalData(list(y.time), list(y.event)), matrix, **common
        )
    else:
        cfit = _core.concordancefit_counting(
            _core.CountingProcessData(list(y.start), list(y.time), list(y.event)), matrix, **common
        )
    return _result(cfit, names, levels, formula)


def _is_matrix(x: Any) -> bool:
    if hasattr(x, "shape") and len(getattr(x, "shape", ())) == 2:
        return True
    return isinstance(x, list | tuple) and bool(x) and isinstance(x[0], list | tuple)


def _concordance_formula(
    formula: str,
    data: Any,
    *,
    weights: Any,
    subset: Any,
    na_action: str | None,
    cluster: Any,
    options: dict[str, Any],
) -> ConcordanceResult:
    """R's ``concordance.formula``; a numeric response ``y ~ x`` is read as ``Surv(y)``."""

    lhs, _sep, rhs = formula.partition("~")
    if not lhs.strip().startswith("Surv("):
        formula = f"Surv({lhs.strip()}) ~ {rhs.strip()}"
    frame = _model_frame(
        formula, data, subset=subset, na_action=na_action, weights=weights, cluster=cluster
    )
    if frame.terms.offsets:
        raise ValueError("Offset terms not allowed")
    if not frame.names:
        raise ValueError("the formula needs at least one predictor")
    strata = None if frame.strata is None else [frame.strata_levels[c] for c in frame.strata]
    return concordancefit(
        frame.y,
        frame.x,
        strata=strata,
        weights=frame.weights,
        cluster=frame.cluster,
        names=frame.names,
        strata_levels=frame.strata_levels,
        formula=formula,
        **options,
    )


@dataclass(frozen=True)
class _FitData:
    """R's ``cord.getdata``: the response, linear predictor and specials of one fit."""

    y: Surv
    x: list[float]
    strata: list[str] | None
    strata_levels: tuple[str, ...]
    weights: list[float] | None
    cluster: list[Any] | None


def _fit_data(fit: Any, newdata: Any | None, need_weights: bool) -> _FitData:
    if isinstance(fit, CoxphModel):
        if fit.tt:
            raise ValueError("cannot yet handle models with tt terms")
        if newdata is None:
            return _FitData(
                y=fit.y,
                x=fit.linear_predictors,
                strata=fit.strata,
                strata_levels=fit.strata_levels,
                weights=fit.weights if need_weights else None,
                cluster=None if fit.cluster is None else list(fit.cluster),
            )
        new = _newdata_frame(
            fit.design,
            fit.terms.strata,
            fit.strata_levels,
            newdata,
            need_strata=True,
            need_response=True,
        )
        if new.y is None:
            raise ValueError("newdata must contain the response variables")
        strata = None
        if new.strata is not None:
            strata = [fit.strata_levels[code] for code in new.strata]
        return _FitData(
            y=new.y,
            x=predict_coxph(fit, newdata, type="lp"),
            strata=strata,
            strata_levels=fit.strata_levels,
            weights=None,
            cluster=None,
        )
    y = getattr(fit, "y", None)
    lp = getattr(fit, "linear_predictors", None)
    if not isinstance(y, Surv) or lp is None or newdata is not None:
        raise TypeError("object is not an appropriate fit object")
    strata = getattr(fit, "strata", None)
    return _FitData(
        y=y,
        x=[float(value) for value in lp],
        strata=strata,
        strata_levels=tuple(getattr(fit, "strata_levels", ()) or ()),
        weights=getattr(fit, "weights", None) if need_weights else None,
        cluster=None,
    )


def _concordance_fits(
    fits: Sequence[Any],
    *,
    newdata: Any | None,
    cluster: Any | None,
    options: dict[str, Any],
) -> ConcordanceResult:
    """R's ``concordance.coxph``/``concordance.survreg`` (``cord.work``)."""

    is_cox = isinstance(fits[0], CoxphModel)
    for fit in fits[1:]:
        if isinstance(fit, CoxphModel) != is_cox:
            raise TypeError("argument is not an appropriate fit object")
    need_weights = any(getattr(fit, "weights", None) is not None for fit in fits)
    data = [_fit_data(fit, newdata, need_weights) for fit in fits]
    first = data[0]
    for other in data[1:]:
        if len(other.x) != len(first.x):
            raise ValueError("all models must have the same sample size")
        if other.y.time != first.y.time or other.y.event != first.y.event:
            warnings.warn(
                "models do not have the same response vector", RuntimeWarning, stacklevel=3
            )
        if other.weights != first.weights:
            raise ValueError("all models must have the same weight vector")
        if other.strata != first.strata:
            raise ValueError("all models must have the same strata")
    names = [f"fit{idx + 1}" for idx in range(len(fits))] if len(fits) > 1 else None
    options = {**options, "reverse": is_cox}
    return concordancefit(
        first.y,
        [[column[row] for column in (d.x for d in data)] for row in range(len(first.x))],
        strata=first.strata,
        strata_levels=first.strata_levels or None,
        weights=first.weights,
        cluster=cluster if cluster is not None else first.cluster,
        names=names,
        formula=getattr(fits[0], "formula", None),
        **options,
    )


def concordance(
    object: Any,  # noqa: A002 - R's argument name
    *more: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    cluster: Any | None = None,
    ymin: Any | None = None,
    ymax: Any | None = None,
    timewt: Any = "n",
    influence: Any = 0,
    ranks: Any = False,
    reverse: Any = False,
    timefix: Any = True,
    keepstrata: Any = 10,
    newdata: Any | None = None,
    scores: Any | None = None,
    strata: Any | None = None,
    **kwargs: Any,
) -> ConcordanceResult:
    """R's ``concordance``: for a formula (``Surv(time, status) ~ x + strata(g)``),
    for one or more ``coxph``/``survreg`` fits (their linear predictors, with
    ``reverse=TRUE`` for Cox models), or for a ``Surv`` plus ``scores``."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    object = _pop_dotted_keyword(kwargs, "response", "object", object, None)  # noqa: A001
    object = _pop_dotted_keyword(kwargs, "formula", "object", object, None)  # noqa: A001
    scores = _pop_dotted_keyword(kwargs, "risk_scores", "scores", scores, None)
    if kwargs:
        raise TypeError(f"concordance got unexpected argument(s): {', '.join(sorted(kwargs))}")
    options = {
        "ymin": ymin,
        "ymax": ymax,
        "timewt": timewt,
        "influence": influence,
        "ranks": ranks,
        "timefix": timefix,
        "keepstrata": keepstrata,
    }
    if isinstance(object, str):
        if len(more) == 1 and data is None:
            data, more = more[0], ()
        if more or scores is not None or newdata is not None:
            raise TypeError("a formula cannot be combined with fits, scores or newdata")
        return _concordance_formula(
            object,
            data,
            weights=weights,
            subset=subset,
            na_action=na_action,
            cluster=cluster,
            options={**options, "reverse": reverse},
        )
    if isinstance(object, Surv):
        if scores is None:
            raise ValueError("scores are required with a Surv response")
        return concordancefit(
            object,
            scores,
            strata=strata,
            weights=weights,
            cluster=cluster,
            reverse=reverse,
            **options,
        )
    if object is None:
        raise TypeError("a formula argument is required")
    return _concordance_fits([object, *more], newdata=newdata, cluster=cluster, options=options)


def survConcordance(
    formula: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    **kwargs: Any,
) -> ConcordanceResult:
    """Deprecated: ``concordance(formula, data, reverse=TRUE)``."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        raise TypeError(f"survConcordance got unexpected argument(s): {', '.join(sorted(kwargs))}")
    warnings.warn(
        "survConcordance is deprecated; use concordance instead", DeprecationWarning, stacklevel=2
    )
    return concordance(
        formula, data=data, weights=weights, subset=subset, na_action=na_action, reverse=True
    )


def survConcordance_fit(
    y: Any,
    x: Any,
    strata: Any | None = None,
    weight: Any | None = None,
) -> dict[str, float]:
    """Deprecated ``survConcordance.fit``: the ``concordancefit`` counts as
    ``concordant``/``discordant``/``tied.risk``/``tied.time``/``std(c-d)``."""

    warnings.warn(
        "survConcordance.fit is deprecated; use concordancefit instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(y, Surv):
        raise TypeError("y must be a Surv object")
    result = concordancefit(y, x, strata=strata, weights=weight, reverse=True)
    rows = result.count if isinstance(result.count, list) else [result.count]
    totals = {name: sum(row[name] for row in rows) for name in _COUNT_NAMES}
    npair = totals["concordant"] + totals["discordant"] + totals["tied.x"]
    std = math.sqrt(result.var) if isinstance(result.var, float) else math.nan
    return {
        "concordant": totals["concordant"],
        "discordant": totals["discordant"],
        "tied.risk": totals["tied.x"],
        "tied.time": totals["tied.y"],
        "std(c-d)": 2.0 * std * npair,
    }
