"""``concordance``/``concordancefit`` (R/concordance.R) and the deprecated
``survConcordance`` entry points (R/survConcordance.R, survConcordance.fit.R), on the
Rust ``concordancefit`` kernel."""

from __future__ import annotations

import math
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _DEFAULT_NA_ACTION,
    _as_matrix_rows,
    _categories,
    _factor,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _is_bool_like,
    _is_missing_value,
    _materialize_labels,
    _normalize_bool_option,
    _optional_float_vector,
    _pop_dotted_keyword,
    _r_factor,
)
from ._coxph import CoxphModel, predict_coxph
from ._fit import _model_frame, _ModelFrame, _newdata_frame
from ._formula import (
    _column_source,
    _data_column_names,
    _formula_name,
    _response_arg_columns,
    _response_arg_values,
)
from ._surv import Surv
from ._survreg import SurvregModelResult, predict_survreg
from ._types import ConcordanceResult, SurvConcordanceResult

_TIMEWT_CHOICES = ("n", "S", "S/G", "n/G2", "I")
_COUNT_NAMES = ("concordant", "discordant", "tied.x", "tied.y", "tied.xy")
_RANK_NAMES = ("time", "rank", "timewt", "casewt")
_SURVCONCORDANCE_NAMES = ("concordant", "discordant", "tied.risk", "tied.time", "std(c-d)")
_RESPONSE_ERROR = (
    "left hand side of the formula must be a numeric vector, survival object, "
    "or an orderable factor"
)


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


def _count_row(counts: Any) -> dict[str, float]:
    return dict(
        zip(
            _COUNT_NAMES,
            (counts.concordant, counts.discordant, counts.tied_x, counts.tied_y, counts.tied_xy),
            strict=True,
        )
    )


def _ranks_columns(ranks: Any) -> dict[str, list[float]]:
    """R's ``ranks`` data frame, column by column."""

    return {name: getattr(ranks, name) for name in _RANK_NAMES}


def _result(
    cfit: _core.ConcordanceFit,
    names: Sequence[str] | None,
    strata_levels: Sequence[str],
    formula: str | None,
) -> ConcordanceResult:
    """Lay a ``ConcordanceFit`` out the way R's ``concordancefit`` list is.

    Each Rust getter builds a fresh Python structure, so every one is read once.
    """

    concordance = cfit.concordance
    single = len(concordance) == 1
    count_rows = [_count_row(row) for row in cfit.count]
    count_strata = cfit.count_strata
    count_names: list[str] | None = None
    if count_strata is not None:
        count_names = [str(strata_levels[int(code)]) for code in count_strata]
    elif not single:
        count_names = (
            list(names) if names is not None else [f"X{i + 1}" for i in range(len(concordance))]
        )
    var: Any = cfit.var
    if var is not None and single:
        var = var[0][0]
    cvar: Any = cfit.cvar
    if cvar is not None and single:
        cvar = cvar[0]
    dfbeta: Any = cfit.dfbeta
    if dfbeta is not None and single:
        dfbeta = [row[0] for row in dfbeta]
    influence: Any = cfit.influence
    if influence is not None and single:
        influence = influence[0]
    ranks: Any = cfit.ranks
    if ranks is not None:
        tables = [_ranks_columns(table) for table in ranks]
        ranks = tables[0] if single else tables
    return ConcordanceResult(
        concordance=concordance[0] if single else concordance,
        count=count_rows[0] if len(count_rows) == 1 and count_names is None else count_rows,
        n=int(cfit.n),
        names=count_names if count_names is not None else (list(names) if names else None),
        var=var,
        cvar=cvar,
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
    _strata_levels: Sequence[str] | None = None,
    _formula: str | None = None,
    **kwargs: Any,
) -> ConcordanceResult:
    """R's ``concordancefit``: the concordance of ``y`` (a Surv, or a numeric
    vector) with one or more predictor columns ``x``.

    ``names`` labels the columns of ``x`` (R reads ``colnames(x)``, else ``X1``, ``X2``, ...).
    ``concordance`` passes the levels of its strata and the formula it was called with
    through the private ``_strata_levels`` and ``_formula``.
    """

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
    if _is_matrix(x):
        rows = _as_matrix_rows(x, "x", allow_empty_columns=False)
        nrow, nvar = len(rows), len(rows[0])
        values = [value for row in rows for value in row]
    else:
        values = _float_vector(x, "x")
        nrow, nvar = len(values), 1
    if nrow != n:
        raise ValueError("x and y are not the same length")
    timewt_name = _timewt_name(timewt)
    if y.start is not None and timewt_name in {"S/G", "n/G2"}:
        raise ValueError(f"{timewt_name} timewt option not supported for (time1, time2) data")
    strata_codes: list[int | None] | None = None
    levels: list[str] = []
    if strata is not None:
        labels = _materialize_labels(strata, "strata")
        if len(labels) != n:
            raise ValueError("y and strata are not the same length")
        # R's as.factor(strata): the declared levels in their order, else the sorted values
        declared = _strata_levels or _categories(strata)
        strata_codes, levels = _factor(
            labels if declared is None else _r_factor(labels, declared), "strata"
        )
        if None in strata_codes:
            raise ValueError("strata contains missing values")
    cluster_codes: list[int | None] | None = None
    if cluster is not None:
        # R's rowsum(dfbeta, cluster) orders the clusters as sort(unique(cluster))
        cluster_codes = _factor(cluster, "cluster")[0]
        if len(cluster_codes) != n:
            raise ValueError("y and cluster are not the same length")
        if None in cluster_codes:
            raise ValueError("cluster contains missing values")
    weight_values = _optional_float_vector(weights, "weights", n)
    if weight_values is not None and len(weight_values) != n:
        raise ValueError("y and weights are not the same length")
    influence_value = _influence_option(influence)
    if isinstance(keepstrata, bool):
        keep = 10**9 if keepstrata else 0
    else:
        keep = _integer_scalar(keepstrata, "keepstrata")
    if not _normalize_bool_option(std_err, "std_err"):
        ranks, influence_value = False, 0
    common: dict[str, Any] = {
        "weights": None if weight_values is None else _core.Weights(weight_values),
        "strata": strata_codes,
        "cluster": cluster_codes,
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
    matrix = _core.CovariateMatrix(values, n, nvar)
    if y.start is None:
        cfit = _core.concordancefit(
            _core.SurvivalData(list(y.time), list(y.event)), matrix, **common
        )
    else:
        cfit = _core.concordancefit_counting(
            _core.CountingProcessData(list(y.start), list(y.time), list(y.event)), matrix, **common
        )
    return _result(cfit, names, levels, _formula)


def _influence_option(value: Any) -> int:
    influence = _integer_scalar(value, "influence")
    if influence not in (0, 1, 2, 3):
        raise ValueError("influence must be 0, 1, 2 or 3")
    return influence


def _is_matrix(x: Any) -> bool:
    if hasattr(x, "shape") and len(getattr(x, "shape", ())) == 2:
        return True
    return isinstance(x, list | tuple) and bool(x) and isinstance(x[0], list | tuple)


def _is_surv_response(lhs: str) -> bool:
    return lhs.startswith(("Surv(", "survival::Surv("))


def _is_ordered(column: Any) -> bool:
    ordered = getattr(column, "ordered", None)
    if ordered is None:
        ordered = getattr(getattr(column, "dtype", None), "ordered", False)
    return bool(ordered)


def _orderable_response(data: Any, lhs: str) -> tuple[Any, str]:
    """R's ``concordance.formula`` for a response that is not ``Surv(...)``: ``data`` and
    a left-hand side naming a numeric response.

    A numeric response is kept.  A logical response, or an ordered or two-level factor
    (a factor column, or ``factor(x)``), becomes R's ``as.numeric`` of it; any other
    response is R's error.  A coerced expression goes in the column ``(response)``, and
    the columns it reads are dropped so that a ``.`` leaves them out, as R's does.
    """

    columns = _data_column_names(data)
    if columns is None:
        return data, lhs
    name, _quoted = _formula_name(lhs)
    source = _column_source(data, name) if name in columns else _response_arg_values(data, lhs)
    if _categories(source) is not None or lhs.startswith(("factor(", "as.factor(")):
        codes, levels = _factor(source, name)  # R's as.factor: declared or sorted levels
        if not _is_ordered(source) and len(levels) != 2:
            raise ValueError(_RESPONSE_ERROR)
        values = [None if code is None else code + 1.0 for code in codes]
    else:
        labels = _materialize_labels(source, name)
        present = [label for label in labels if not _is_missing_value(label)]
        if any(isinstance(label, str) for label in present):
            raise ValueError(_RESPONSE_ERROR)
        if not present or not all(_is_bool_like(label) for label in present):
            return data, lhs
        values = [None if _is_missing_value(label) else float(label) for label in labels]
    if name in columns:
        return {**{key: _column_source(data, key) for key in columns}, name: values}, lhs
    reads = set(_response_arg_columns(lhs))
    kept = {key: _column_source(data, key) for key in columns if key not in reads}
    return {**kept, "(response)": values}, "`(response)`"


def _concordance_frame(formula: str, data: Any, **arguments: Any) -> _ModelFrame:
    """The model frame of ``concordance.formula`` and ``survConcordance``: a numeric
    response is read as ``Surv(y)``; offsets are not allowed and at least one predictor
    is needed."""

    lhs, sep, rhs = formula.partition("~")
    if sep and not _is_surv_response(lhs.strip()):
        formula = f"Surv({lhs.strip()}) ~ {rhs.strip()}"
    frame = _model_frame(formula, data, **arguments)
    if frame.terms.offsets:
        raise ValueError("Offset terms not allowed")
    if not frame.names:
        raise ValueError("the formula needs at least one predictor")
    return frame


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
    """R's ``concordance.formula``.

    A response that is not ``Surv(...)`` (numeric, logical, or an ordered or two-level
    factor) is read as ``Surv(as.numeric(y))`` with ``timewt = "n"``.
    """

    frame_formula = formula
    lhs, sep, rhs = formula.partition("~")
    if sep and not _is_surv_response(lhs.strip()):
        data, response = _orderable_response(data, lhs.strip())
        frame_formula = f"{response} ~ {rhs.strip()}"
        options = {**options, "timewt": "n"}
    frame = _concordance_frame(
        frame_formula, data, subset=subset, na_action=na_action, weights=weights, cluster=cluster
    )
    return concordancefit(
        frame.y,
        frame.x,
        strata=frame.strata_labels(),
        weights=frame.weights,
        cluster=frame.cluster,
        names=frame.names,
        _strata_levels=frame.strata_levels,
        _formula=formula,
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
    cluster: Sequence[Any] | None


def _fit_data(fit: Any, newdata: Any | None, need_weights: bool, cluster: Any | None) -> _FitData:
    """``cord.getdata``.  An explicit ``cluster`` replaces the fit's own; a survreg
    fit's ``cluster()`` term is not used, as in R."""

    if isinstance(fit, CoxphModel):
        if fit.tt:
            raise ValueError("cannot yet handle models with tt terms")
        if newdata is not None:
            return _newdata_fit_data(fit, newdata, fit.terms.strata, predict_coxph, cluster)
        return _FitData(
            y=fit.y,
            x=fit.linear_predictors,
            strata=fit.strata,
            strata_levels=fit.strata_levels,
            weights=fit.weights if need_weights else None,
            cluster=cluster if cluster is not None else fit.cluster,
        )
    if isinstance(fit, SurvregModelResult):
        if newdata is not None:
            return _newdata_fit_data(fit, newdata, fit.strata_columns, predict_survreg, cluster)
        if fit.y is None:
            raise ValueError("the survreg fit has no response: refit it with y=True")
        levels = fit.strata_levels
        return _FitData(
            y=fit.y,
            x=list(fit.linear_predictors),
            strata=[levels[int(code)] for code in fit.fit.strata] if levels else None,
            strata_levels=levels,
            weights=fit.weights if need_weights else None,
            cluster=cluster,
        )
    raise TypeError("object is not an appropriate fit object")


def _newdata_fit_data(
    fit: Any, newdata: Any, strata_columns: Sequence[str], predict: Any, cluster: Any | None
) -> _FitData:
    """``cord.getdata`` with ``newdata``: the response and strata of the rows
    ``model.frame(Terms, newdata)`` keeps under R's default ``na.omit`` (no response,
    covariate, offset or strata variable missing; coxph and survreg move a
    ``cluster()`` term out of ``Terms``, so it is not checked), and the fit's linear
    predictor on them (no case weights)."""

    new = _newdata_frame(
        fit.design,
        strata_columns,
        fit.strata_levels,
        newdata,
        need_strata=True,
        need_response=True,
        na_action=_DEFAULT_NA_ACTION,
    )
    if new.y is None:
        raise ValueError("newdata must contain the response variables")
    return _FitData(
        y=new.y,
        x=predict(fit, new.data, type="lp"),
        strata=None if new.strata is None else [fit.strata_levels[code] for code in new.strata],
        strata_levels=fit.strata_levels,
        weights=None,
        cluster=cluster,
    )


def _fit_concordance(data: _FitData, options: dict[str, Any]) -> ConcordanceResult:
    return concordancefit(
        data.y,
        data.x,
        strata=data.strata,
        weights=data.weights,
        cluster=data.cluster,
        _strata_levels=data.strata_levels or None,
        **options,
    )


def _count_total(count: dict[str, float] | list[dict[str, float]]) -> dict[str, float]:
    """One fit's count row, summed over kept strata (R's ``colSums(x$count)``)."""

    rows = count if isinstance(count, list) else [count]
    return {name: sum(row[name] for row in rows) for name in _COUNT_NAMES}


def _concordance_fits(
    fits: Sequence[Any],
    *,
    newdata: Any | None,
    cluster: Any | None,
    options: dict[str, Any],
) -> ConcordanceResult:
    """R's ``concordance.coxph``/``concordance.survreg`` and ``cord.work``: each fit is
    scored with its own response, strata, weights and clusters."""

    is_cox = isinstance(fits[0], CoxphModel)
    for fit in fits[1:]:
        if isinstance(fit, CoxphModel) != is_cox:
            raise TypeError("argument is not an appropriate fit object")
    need_weights = any(getattr(fit, "weights", None) is not None for fit in fits)
    data = [_fit_data(fit, newdata, need_weights, cluster) for fit in fits]
    formula = getattr(fits[0], "formula", None)
    options = {**options, "reverse": is_cox}
    if len(data) == 1:
        return _fit_concordance(data[0], {**options, "_formula": formula})

    first = data[0]
    for other in data[1:]:
        if len(other.x) != len(first.x):
            raise ValueError("all models must have the same sample size")
        if (other.y.start, other.y.time, other.y.event) != (
            first.y.start,
            first.y.time,
            first.y.event,
        ):
            warnings.warn(
                "models do not have the same response vector", RuntimeWarning, stacklevel=3
            )
        if other.weights != first.weights:
            raise ValueError("all models must have the same weight vector")
    influence = _influence_option(options["influence"])
    options["influence"] = 3 if influence == 2 else 1
    # each fit has one predictor: scalar concordance/cvar, a dfbeta vector
    results: list[Any] = [_fit_concordance(d, options) for d in data]
    dfbeta = [result.dfbeta for result in results]
    if any(len(column) != len(dfbeta[0]) for column in dfbeta[1:]):
        raise ValueError("models must have identical clustering")
    names = [f"fit{idx + 1}" for idx in range(len(fits))]
    ranks: dict[str, list[Any]] | None = None
    if results[0].ranks is not None:
        ranks = {"fit": [], **{column: [] for column in _RANK_NAMES}}
        for name, result in zip(names, results, strict=True):
            ranks["fit"].extend([name] * len(result.ranks["time"]))
            for column in _RANK_NAMES:
                ranks[column].extend(result.ranks[column])
    return ConcordanceResult(
        concordance=[result.concordance for result in results],
        count=[_count_total(result.count) for result in results],
        n=results[0].n,
        names=names,
        # crossprod(dfbeta): the single-fit variance on the diagonal.  R's
        # t(wt * dfbeta) %*% dfbeta applies the case weights a second time.
        var=[[math.fsum(map(operator.mul, u, v)) for v in dfbeta] for u in dfbeta],
        cvar=[result.cvar for result in results],
        dfbeta=[list(row) for row in zip(*dfbeta, strict=True)] if influence == 1 else None,
        influence=[result.influence for result in results] if influence == 2 else None,
        ranks=ranks,
        formula=formula,
    )


def concordance(
    object: Any,  # noqa: A002 - R's argument name
    *more: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
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

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
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
    # R's concordance.coxph/.survreg take any other argument as one more fit
    for name, given in (
        ("data", data is not None),
        ("weights", weights is not None),
        ("subset", subset is not None),
        ("strata", strata is not None),
        ("scores", scores is not None),
        ("reverse", reverse is not False),
        ("na.action", na_action != _DEFAULT_NA_ACTION),
    ):
        if given:
            raise TypeError(f"{name} argument is not an appropriate fit object")
    return _concordance_fits([object, *more], newdata=newdata, cluster=cluster, options=options)


def _survconcordance_row(y: Surv, x: list[float], weights: list[float] | None) -> dict[str, float]:
    """One row of ``survConcordance.fit``.  Its old kernels count every pair tied on time
    as ``tied.time`` and report the Cox-model standard deviation of ``C - D``; that is
    ``concordancefit(reverse = TRUE)`` without timefix, with ``tied.y + tied.xy`` and
    ``2 * npair * sqrt(cvar)``."""

    fit: Any = concordancefit(y, x, weights=weights, reverse=True, timefix=False)
    count = fit.count
    concordant, discordant, tied_x = count["concordant"], count["discordant"], count["tied.x"]
    npair = concordant + discordant + tied_x
    std = 0.0 if concordant + discordant == 0 else 2.0 * npair * math.sqrt(fit.cvar)
    return dict(
        zip(
            _SURVCONCORDANCE_NAMES,
            (concordant, discordant, tied_x, count["tied.y"] + count["tied.xy"], std),
            strict=True,
        )
    )


def _survconcordance_strata(
    y: Surv,
    x: list[float],
    weights: list[float] | None,
    codes: Sequence[int | None],
    levels: Sequence[str],
) -> dict[str, dict[str, float]]:
    """``survConcordance.fit`` with strata: one row per non-empty level, in level order."""

    rows_of: dict[int, list[int]] = {}
    for row, code in enumerate(codes):
        if code is not None:
            rows_of.setdefault(code, []).append(row)
    return {
        levels[code]: _survconcordance_row(
            y.subset(rows),
            [x[row] for row in rows],
            None if weights is None else [weights[row] for row in rows],
        )
        for code, rows in sorted(rows_of.items())
    }


def survConcordance(
    formula: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = _DEFAULT_NA_ACTION,
    **kwargs: Any,
) -> SurvConcordanceResult:
    """Deprecated R ``survConcordance``: the concordance of a single predictor with
    ``survConcordance.fit``'s counts and standard error."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
    if kwargs:
        raise TypeError(f"survConcordance got unexpected argument(s): {', '.join(sorted(kwargs))}")
    warnings.warn(
        "survConcordance is deprecated; use concordance instead", DeprecationWarning, stacklevel=2
    )
    if not isinstance(formula, str):
        raise TypeError("a formula argument is required")
    frame = _concordance_frame(formula, data, subset=subset, na_action=na_action, weights=weights)
    if len(frame.names) > 1:
        raise ValueError("Only one predictor variable allowed")
    x = [row[0] for row in frame.x]
    stats: dict[str, float] | dict[str, dict[str, float]]
    if frame.strata is None:
        stats = _survconcordance_row(frame.y, x, frame.weights)
        rows = [stats]
    else:
        stats = _survconcordance_strata(
            frame.y, x, frame.weights, frame.strata, frame.strata_levels
        )
        rows = list(stats.values())
    total = {name: sum(row[name] for row in rows) for name in _SURVCONCORDANCE_NAMES}
    npair = total["concordant"] + total["discordant"] + total["tied.risk"]
    if npair == 0:  # no comparable pairs: R's 0/0
        npair = math.nan
    return SurvConcordanceResult(
        concordance=(total["concordant"] + total["tied.risk"] / 2.0) / npair,
        stats=stats,
        n=frame.n,
        std_err=total["std(c-d)"] / (2.0 * npair),
    )


def survConcordance_fit(
    y: Any,
    x: Any,
    strata: Any | None = None,
    weight: Any | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Deprecated R ``survConcordance.fit``: ``concordant``, ``discordant``, ``tied.risk``,
    ``tied.time`` and ``std(c-d)``, per stratum level (in factor order) when ``strata`` is
    given."""

    warnings.warn(
        "survConcordance.fit is deprecated; use concordancefit instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(y, Surv):
        raise TypeError("y must be a Surv object")
    n = len(y)
    values = _float_vector(x, "x")
    if len(values) != n:
        raise ValueError("x and y are not the same length")
    weights = _optional_float_vector(weight, "weight", n)
    if strata is None:
        return _survconcordance_row(y, values, weights)
    codes, levels = _factor(strata, "strata")  # R's as.factor(strata)
    if len(codes) != n:
        raise ValueError("y and strata are not the same length")
    return _survconcordance_strata(y, values, weights, codes, levels)
