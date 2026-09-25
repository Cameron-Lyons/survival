"""``coxph``/``clogit`` and the Cox model methods (R/coxph.R, predict.coxph.R,
residuals.coxph.R, survfit.coxph.R, basehaz.R, cox.zph.R, coxph.detail.R,
anova.coxph.R, anova.coxphlist.R, summary.coxph.R, coxph.wtest.R).

Every number comes from the Rust ``CoxPHFit``; this module does what R's R code
does: the model frame, argument checking, dispatch and result labelling.
"""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _apply_coxph_control,
    _as_matrix_rows,
    _as_rows,
    _coerce_array_like,
    _cox_tie_method,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _is_missing_value,
    _label_levels,
    _match_string_arg,
    _materialize_labels,
    _matrix_input_column_names,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
    _normalize_numeric_sequence_or_none,
    _normalize_optional_bool_option,
    _pop_dotted_keyword,
    _subset_data,
)
from ._fit import (
    _design_names_and_assign,
    _model_frame,
    _ModelFrame,
    _NewData,
    _newdata_frame,
    _tt_terms,
)
from ._formula import _column, _column_or_values, _design_rows_from_spec, _response_arg_columns
from ._surv import Surv
from ._types import (
    CoxBaseHazardResult,
    CoxPHDetailResult,
    CoxPHWTestResult,
    CoxSurvfitResult,
    CoxZPHResult,
    PredictResult,
    _CovariateTerm,
    _FormulaDesign,
    _FormulaTerms,
    _PenaltyDesignTerm,
)

_TIE_METHOD_NAMES = ("breslow", "efron", "exact")
_LOG_DOUBLE_MAX = math.log(sys.float_info.max)


# ---------------------------------------------------------------------------
# the coxph object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxphModel:
    """R's ``coxph`` object: the engine fit plus what the R list keeps around it.

    The numeric components (``coefficients``, ``var``, ``loglik``, ``residuals``,
    ...) are read through from :class:`survival._survival.CoxPHFit`; ``formula``,
    ``design``, ``assign``, ``coef_names``, ``y``, ``strata_levels`` and ``id`` are
    what ``predict``/``survfit``/``residuals`` need to rebuild the model frame.
    """

    fit: _core.CoxPHFit
    formula: str
    design: _FormulaDesign
    terms: _FormulaTerms
    coef_names: tuple[str, ...]
    assign: dict[str, tuple[int, ...]]
    y: Surv
    strata_levels: tuple[str, ...]
    concordance: dict[str, float]
    n: int
    timefix: bool
    tt: bool
    id: tuple[Any, ...] | None = None
    cluster: tuple[Any, ...] | None = None
    model: dict[str, Any] | None = None
    # the columns the call's weights= / id= named, for re-evaluation on newdata
    weights_column: str | None = None
    id_column: str | None = None
    # R's coxph returns a skeleton fit when the data has no events: NA
    # coefficients, a zero variance, loglik c(0, 0) and no iterations.
    no_events: bool = False
    penalized: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self.penalized is not None and name in {
            "df",
            "var2",
            "frail",
            "fvar",
            "history",
            "penalty",
            "pterms",
        }:
            return getattr(self.penalized, name)
        raise AttributeError(name)

    @property
    def coefficients(self) -> list[float]:
        if self.no_events:
            return [math.nan] * len(self.coef_names)
        return [float(value) for value in self.fit.coefficients]

    @property
    def var(self) -> list[list[float]]:
        return [list(row) for row in self.fit.var]

    @property
    def naive_var(self) -> list[list[float]] | None:
        naive = self.fit.naive_var
        return None if naive is None else [list(row) for row in naive]

    @property
    def robust(self) -> bool:
        return self.fit.naive_var is not None

    @property
    def loglik(self) -> list[float]:
        """``fit$loglik``: null and fitted values (one value for a null model, as R)."""

        values = list(self.fit.loglik)
        return values if self.coef_names else values[:1]

    @property
    def score(self) -> float | None:
        return float(self.fit.score) if self.coef_names else None

    @property
    def rscore(self) -> float | None:
        return self.fit.rscore

    @property
    def wald_test(self) -> float | None:
        if not self.coef_names:
            return None
        return 0.0 if self.no_events else float(self.fit.wald_test)

    @property
    def iter(self) -> int | list[int] | None:
        if self.penalized is not None:
            return list(self.penalized.iter)
        if not self.coef_names:
            return None
        return 0 if self.no_events else int(self.fit.iter)

    @property
    def linear_predictors(self) -> list[float]:
        return list(self.fit.linear_predictors)

    @property
    def residuals(self) -> list[float]:
        return list(self.fit.residuals)

    @property
    def means(self) -> list[float]:
        return list(self.fit.means)

    @property
    def method(self) -> str:
        return _TIE_METHOD_NAMES[int(self.fit.method)]

    @property
    def nevent(self) -> int:
        return int(self.fit.nevent)

    @property
    def nvar(self) -> int:
        return len(self.coef_names)

    @property
    def x(self) -> list[list[float]]:
        return [list(row) for row in self.fit.x]

    @property
    def weights(self) -> list[float] | None:
        """``fit$weights``: the case weights, present only when some differ from 1."""

        values = list(self.fit.weights)
        return values if any(value != 1.0 for value in values) else None

    @property
    def offset(self) -> list[float] | None:
        values = list(self.fit.offset)
        return values if any(value != 0.0 for value in values) else None

    @property
    def strata(self) -> list[str] | None:
        """``fit$strata``: the stratum label of every row, ``None`` when unstratified."""

        codes = self.fit.strata
        if codes is None or not self.strata_levels:
            return None
        return [self.strata_levels[int(code)] for code in codes]

    def predict(self, newdata: Any | None = None, **kwargs: Any) -> Any:
        return predict_coxph(self, newdata, **kwargs)

    def survfit(self, newdata: Any | None = None, **kwargs: Any) -> CoxSurvfitResult:
        return survfit_coxph(self, newdata, **kwargs)

    def summary(self, conf_int: float = 0.95, scale: float = 1.0) -> dict[str, Any]:
        return summary_coxph(self, conf_int=conf_int, scale=scale)


@dataclass(frozen=True)
class ClogitModel(CoxphModel):
    """R's ``clogit`` object (class ``c("clogit", "coxph")``)."""


def _has_strata(fit: CoxphModel) -> bool:
    return bool(fit.strata_levels)


def _aliased(fit: CoxphModel) -> list[bool]:
    return [math.isnan(value) for value in fit.coefficients]


def _active_assign(fit: CoxphModel) -> list[list[int]]:
    """``fit$assign`` restricted to the estimable columns, one entry per term."""

    aliased = _aliased(fit)
    return [[col for col in cols if not aliased[col]] for cols in fit.assign.values()]


# ---------------------------------------------------------------------------
# coxph
# ---------------------------------------------------------------------------


def _aeq_surv(y: Surv) -> Surv:
    """``aeqSurv``: snap times that are equal up to floating-point noise together."""

    if y.start is None:
        fixed = _core.aeq_surv(list(y.time))
        return Surv(list(fixed.time), list(y.event), type=y.type)
    fixed = _core.aeq_surv(list(y.start), list(y.time))
    time2 = fixed.time2 if fixed.time2 is not None else list(y.time)
    return Surv(list(fixed.time), list(time2), list(y.event), type=y.type)


def _obrien_time_transform(
    x: Sequence[float],
    time: Sequence[float],
    riskset: Sequence[int],
    weights: Sequence[float] | None,
) -> list[float]:
    """R's default ``tt``: O'Brien's logit rank within each risk set."""

    del time, weights
    out = [0.0] * len(x)
    groups: dict[int, list[int]] = {}
    for idx, group in enumerate(riskset):
        groups.setdefault(group, []).append(idx)
    for members in groups.values():
        order = sorted(members, key=lambda idx: x[idx])
        size = len(order)
        pos = 0
        while pos < size:  # average ranks over ties, as R's rank()
            end = pos
            while end + 1 < size and x[order[end + 1]] == x[order[pos]]:
                end += 1
            rank = (pos + end) / 2.0 + 1.0
            for k in range(pos, end + 1):
                out[order[k]] = (rank - 0.5) / (0.5 + size - rank)
            pos = end + 1
    return out


def _tt_functions(tt: Any, count: int) -> list[Callable[..., Any]]:
    if tt is None:
        return [_obrien_time_transform] * count
    functions = list(tt) if isinstance(tt, list | tuple) else [tt]
    if any(not callable(function) for function in functions):
        raise TypeError("The tt argument must contain a function or list of functions")
    if len(functions) != count:
        if len(functions) == 1:
            return functions * count
        raise ValueError("Wrong length for tt argument")
    return functions


@dataclass(frozen=True)
class _CoxData:
    """The rows handed to the engine (after the tt() expansion, when there is one)."""

    y: Surv
    x: list[list[float]]
    strata: list[int] | None
    weights: list[float] | None
    offset: list[float] | None
    cluster: list[Any] | None
    id: list[Any] | None


def _tt_expand(frame: _ModelFrame, tt: Any, tt_terms: list[_CovariateTerm]) -> _CoxData:
    """coxph.R's tt() section: one row per (risk set, subject at risk)."""

    y = frame.y
    if y.start is None:
        counts = _core.coxcount1(_core.SurvivalData(list(y.time), list(y.event)), frame.strata)
    else:
        counts = _core.coxcount2(
            _core.CountingProcessData(list(y.start), list(y.time), list(y.event)),
            frame.strata,
        )
    tindex = [int(idx) for idx in counts.index]
    nrisk = [int(value) for value in counts.nrisk]
    new_time = [time for time, size in zip(counts.time, nrisk, strict=True) for _ in range(size)]
    new_y = Surv(new_time, [int(value) for value in counts.status])
    riskset = [group for group, size in enumerate(nrisk) for _ in range(size)]
    data = _subset_data(frame.data, tindex)
    weights = None if frame.weights is None else [frame.weights[idx] for idx in tindex]
    transformed: dict[_CovariateTerm, list[float]] = {}
    for term, function in zip(tt_terms, _tt_functions(tt, len(tt_terms)), strict=True):
        values = [float(value) for value in _column(data, term.column)]
        transformed[term] = [
            float(value) for value in function(values, list(new_y.time), riskset, weights)
        ]
        if len(transformed[term]) != len(tindex):
            raise ValueError("the tt function must return one value per expanded row")
    return _CoxData(
        y=new_y,
        x=_design_rows_from_spec(
            data, frame.design, len(tindex), time_transform_values=transformed
        ),
        strata=riskset,
        weights=weights,
        offset=None if frame.offset is None else [frame.offset[idx] for idx in tindex],
        cluster=None if frame.cluster is None else [frame.cluster[idx] for idx in tindex],
        id=None if frame.id is None else [frame.id[idx] for idx in tindex],
    )


def _check_init(init: Any, x: list[list[float]], offset: list[float] | None) -> list[float]:
    values = _float_vector(init, "init")
    nvar = len(x[0]) if x else 0
    if len(values) != nvar:
        raise ValueError("wrong length for init argument")
    n = len(x)
    means = [sum(row[col] for row in x) / n for col in range(nvar)] if n else []
    center = sum(mean * value for mean, value in zip(means, values, strict=True))
    risks = []
    for idx, row in enumerate(x):
        eta = sum(a * b for a, b in zip(row, values, strict=True)) - center
        if offset is not None:
            eta += offset[idx]
        try:
            risks.append(math.exp(eta))
        except OverflowError:
            risks.append(math.inf)
    if any(math.isinf(risk) for risk in risks) or (risks and all(risk == 0.0 for risk in risks)):
        raise ValueError("initial values lead to overflow or underflow of the exp function")
    return values


def _robust_default(
    data: _CoxData,
    has_cluster: bool,
) -> bool:
    """coxph.R: robust when a cluster, non-integer weights, or an id with >1 event."""

    has_rwt = data.weights is not None and any(w != math.floor(w) for w in data.weights)
    has_id_events = False
    if data.id is not None:
        seen: set[Any] = set()
        for id_value, event in zip(data.id, data.y.event, strict=True):
            if event == 1:
                if id_value in seen:
                    has_id_events = True
                    break
                seen.add(id_value)
    return has_cluster or has_rwt or has_id_events


def _cluster_codes(values: Sequence[Any]) -> list[int]:
    """``match(cluster, unique(cluster))`` as 0-based codes."""

    codes = {value: idx for idx, value in enumerate(_label_levels(list(values), "cluster"))}
    return [codes[value] for value in values]


def _fit_concordance(
    fit: _core.CoxPHFit, data: _CoxData, cluster: list[int] | None
) -> dict[str, float]:
    """``fit$concordance``: counts, C and its se from ``concordancefit(reverse=TRUE)``."""

    y = data.y
    x = _core.CovariateMatrix(list(fit.linear_predictors), len(y), 1)
    weights = None if data.weights is None else _core.Weights(list(data.weights))
    kwargs: dict[str, Any] = {
        "weights": weights,
        "strata": data.strata,
        "cluster": cluster,
        "reverse": True,
        "timefix": False,
    }
    if y.start is None:
        cfit = _core.concordancefit(_core.SurvivalData(list(y.time), list(y.event)), x, **kwargs)
    else:
        cfit = _core.concordancefit_counting(
            _core.CountingProcessData(list(y.start), list(y.time), list(y.event)), x, **kwargs
        )
    counts = cfit.count
    variance = cfit.var[0][0] if cfit.var is not None else math.nan
    return {
        "concordant": sum(row.concordant for row in counts),
        "discordant": sum(row.discordant for row in counts),
        "tied.x": sum(row.tied_x for row in counts),
        "tied.y": sum(row.tied_y for row in counts),
        "tied.xy": sum(row.tied_xy for row in counts),
        "concordance": cfit.concordance[0],
        "std": math.sqrt(variance),
    }


def _cox_fit_diagnostic_messages(
    fit: Any, iter_max: int, eps: float | None, toler_inf: float | None
) -> list[str]:
    """R's ``coxph.fit`` convergence warnings for an engine fit (also the R bridge's).

    ``infs = |u %*% var|``: after the iterations ran out the fit may be infinite; a
    converged fit whose score still moves a coefficient by more than ``toler.inf``
    of its size converged before that variable did.
    """

    coef = list(fit.coefficients)
    nvar = len(coef)
    if nvar == 0 or iter_max <= 1:
        return []
    eps_value = 1e-9 if eps is None else float(eps)
    toler = math.sqrt(eps_value) if toler_inf is None else float(toler_inf)
    u = list(fit.first)
    var = fit.var
    infs = [
        abs(sum(u[i] * var[i][j] for i in range(nvar)) if var else math.nan) for j in range(nvar)
    ]
    messages: list[str] = []
    if fit.flag == 1000:
        messages.append("Ran out of iterations and did not converge")
        if max(fit.linear_predictors, default=0.0) > 500 or any(
            not math.isfinite(value) for value in infs
        ):
            messages.append("one or more coefficients may be infinite")
        return messages
    which = [
        j + 1
        for j in range(nvar)
        if not math.isfinite(u[j]) or (infs[j] > eps_value and infs[j] > toler * abs(coef[j]))
    ]
    if which:
        messages.append(
            "Loglik converged before variable "
            + ",".join(str(index) for index in which)
            + "; coefficient may be infinite. "
        )
    return messages


def _coxph_fit_frame(
    frame: _ModelFrame,
    *,
    method: str,
    init: Any | None,
    iter_max: int,
    eps: float | None,
    toler_chol: float | None,
    timefix: bool,
    robust: bool | None,
    singular_ok: bool,
    nocenter: list[float] | None,
    tt: Any,
    keep_model: bool,
    outer_max: int | None = None,
) -> CoxphModel:
    """coxph.R after the model frame: timefix, tt(), robust/cluster, the fit, the
    Wald test and concordance."""

    if frame.y.type not in {"right", "counting"}:
        raise ValueError(f'Cox model doesn\'t support "{frame.y.type}" survival data')
    y = _aeq_surv(frame.y) if timefix else frame.y
    tt_terms = _tt_terms(frame.design)
    if tt_terms:
        if keep_model:
            raise ValueError("'model=TRUE' not supported for models with tt terms")
        data = _tt_expand(replace(frame, y=y), tt, tt_terms)
    else:
        data = _CoxData(
            y=y,
            x=frame.x,
            strata=frame.strata,
            weights=frame.weights,
            offset=frame.offset,
            cluster=frame.cluster,
            id=frame.id,
        )
    if any(not math.isfinite(value) for row in data.x for value in row):
        raise ValueError("data contains an infinite predictor")
    if data.offset is not None and any(
        not math.isfinite(value) or value > _LOG_DOUBLE_MAX for value in data.offset
    ):
        raise ValueError("offsets must lead to a finite risk score")

    has_cluster = data.cluster is not None
    use_robust = _robust_default(data, has_cluster) if robust is None else robust
    cluster: list[int] | None = None
    if has_cluster and not use_robust:
        warnings.warn(
            "cluster specified with robust=FALSE, cluster ignored", RuntimeWarning, stacklevel=3
        )
    elif has_cluster:
        cluster = _cluster_codes(data.cluster or [])
    elif use_robust and data.id is not None:
        cluster = _cluster_codes(data.id)
    if use_robust and cluster is None:
        if data.y.start is None or robust is None:
            cluster = list(range(len(data.y)))
        else:
            raise ValueError("one of cluster or id is needed")
    init_values = None if init is None else _check_init(init, data.x, data.offset)
    no_events = not any(int(value) for value in data.y.event)
    if no_events:
        # R returns the fit without iterating (coefficients NA, variance 0)
        iter_max = 0
    penalized_terms = [
        term for term in frame.design.covariates if isinstance(term, _PenaltyDesignTerm)
    ]
    penalized = None
    if penalized_terms:
        if use_robust:
            warnings.warn(
                "the robust variance is not defined for a penalized model, option ignored",
                RuntimeWarning,
                stacklevel=3,
            )
            use_robust = False
        penalized = _core.coxpenal_fit(
            list(data.y.time),
            [int(value) for value in data.y.event],
            data.x,
            penalties=[term.penalty for term in penalized_terms],
            pcols=[list(frame.assign[term.term.call]) for term in penalized_terms],
            assign=[list(columns) for columns in frame.assign.values()],
            entry=None if data.y.start is None else list(data.y.start),
            strata=data.strata,
            weights=data.weights,
            offset=data.offset,
            method=method,
            init=init_values,
            iter_max=iter_max,
            outer_max=outer_max,
            eps=eps,
            toler_chol=toler_chol,
            nocenter=nocenter,
        )
        fit = penalized.coxph
        dense = [
            i
            for i, term in enumerate(frame.design.covariates)
            if not (isinstance(term, _PenaltyDesignTerm) and term.penalty.sparse)
        ]
        design = replace(
            frame.design,
            covariates=tuple(frame.design.covariates[i] for i in dense),
            term_assignments=tuple(frame.design.term_assignments[i] for i in dense),
        )
        names, assign = _design_names_and_assign(design)
    else:
        fit = _core.coxph_fit(
            list(data.y.time),
            [int(value) for value in data.y.event],
            data.x,
            entry=None if data.y.start is None else list(data.y.start),
            strata=data.strata,
            weights=data.weights,
            offset=data.offset,
            method=method,
            init=init_values,
            iter_max=iter_max,
            eps=eps,
            toler_chol=toler_chol,
            nocenter=nocenter,
            cluster=cluster,
            robust=use_robust,
        )
        design, names, assign = frame.design, frame.names, frame.assign
    aliased = [idx for idx, value in enumerate(fit.coefficients) if math.isnan(value)]
    if aliased and not singular_ok:
        columns = " ".join(str(idx + 1) for idx in aliased)
        raise ValueError(f"X matrix deemed to be singular; variable {columns}")
    if penalized is None:
        for message in _cox_fit_diagnostic_messages(fit, iter_max, eps, None):
            warnings.warn(message, RuntimeWarning, stacklevel=3)
    return CoxphModel(
        fit=fit,
        no_events=no_events,
        formula=frame.formula,
        design=design,
        terms=frame.terms,
        coef_names=tuple(names),
        assign=dict(assign),
        y=y,
        strata_levels=frame.strata_levels,
        concordance=_fit_concordance(fit, data, cluster),
        n=frame.n,
        timefix=timefix,
        tt=bool(tt_terms),
        id=None if frame.id is None else tuple(frame.id),
        cluster=None if frame.cluster is None else tuple(frame.cluster),
        model=frame.model_frame() if keep_model else None,
        weights_column=frame.weights_column,
        id_column=frame.id_column,
        penalized=penalized,
    )


def _surv_design_formula(response: Surv, design: Any) -> tuple[str, dict[str, Any]]:
    """A formula and data for ``coxph(<Surv>, x = <matrix or data frame>)``: the response
    columns and the design as named columns."""

    if isinstance(design, bool) or design is None:
        raise TypeError("a design matrix x is required with a Surv response")
    if response.type not in {"right", "counting"}:
        raise ValueError("a Surv response with a design must be right censored")
    rows = _as_rows(design, "x")
    if len(rows) != len(response):
        raise ValueError("x must have the same number of rows as the Surv response")
    names = _matrix_input_column_names(design)
    if names is None or len(names) != len(rows[0]):
        names = tuple(f"x{idx + 1}" for idx in range(len(rows[0])))
    if any(not name.isidentifier() for name in names):
        raise ValueError("the columns of x must have syntactic names")
    data: dict[str, Any] = {
        "survival_time_": list(response.time),
        "survival_status_": [int(value) for value in response.event],
    }
    surv = "Surv(survival_time_, survival_status_)"
    if response.start is not None:
        data["survival_start_"] = list(response.start)
        surv = "Surv(survival_start_, survival_time_, survival_status_)"
    for col, name in enumerate(names):
        data[name] = [row[col] for row in rows]
    return f"{surv} ~ {' + '.join(names)}", data


def coxph(
    formula: str | Surv | None = None,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    init: Any | None = None,
    control: Any | None = None,
    ties: str | None = None,
    method: str | None = None,
    singular_ok: Any = True,
    robust: Any | None = None,
    model: Any = False,
    x: Any = False,
    y: Any = True,
    tt: Any | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    istate: Any | None = None,
    statedata: Any | None = None,
    nocenter: Any = (-1, 0, 1),
    offset: Any | None = None,
    strata: Any | None = None,
    iter_max: Any | None = None,
    eps: Any | None = None,
    toler_chol: Any | None = None,
    timefix: Any | None = None,
    **kwargs: Any,
) -> CoxphModel:
    """Fit a Cox proportional hazards model (R's ``coxph``).

    ``formula`` is an R formula string with a ``Surv`` response; ``strata()``,
    ``cluster()``, ``offset()`` and ``tt()`` terms are honoured, as are the
    ``weights``/``offset``/``strata``/``cluster``/``id`` arguments given as vectors
    or as column names of ``data``.  ``eps``/``toler_chol``/``iter_max``/``timefix``
    are ``coxph.control`` options and may also be given through ``control``.
    """

    formula = _pop_dotted_keyword(kwargs, "response", "formula", formula, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    singular_ok = _pop_dotted_keyword(kwargs, "singular.ok", "singular_ok", singular_ok, True)
    iter_max = _pop_dotted_keyword(kwargs, "iter.max", "iter_max", iter_max, None)
    toler_chol = _pop_dotted_keyword(kwargs, "toler.chol", "toler_chol", toler_chol, None)
    # the R bridge evaluates weights= / id= itself and names the columns they came from
    weights_column = kwargs.pop("_weights_column", None)
    id_column = kwargs.pop("_id_column", None)
    kwargs.pop("survcheckallow", None)
    if isinstance(control, Mapping):
        control = {key: value for key, value in control.items() if key != "survcheckallow"}
    if kwargs:
        raise ValueError(f"Argument {', '.join(sorted(kwargs))} not matched")
    if formula is None:
        raise TypeError("a formula argument is required")
    if isinstance(formula, Surv):
        # coxph(<Surv>, x = <design>): the R bridge's matrix interface, as survreg has
        formula, data = _surv_design_formula(formula, x)
        x = False
    _ = _normalize_bool_option_with_default(x, "x", False)
    _ = _normalize_bool_option_with_default(y, "y", True)

    method_name = _cox_tie_method(method, ties)
    max_iter = 20 if iter_max is None else _integer_scalar(iter_max, "iter_max")
    eps_value = None if eps is None else _finite_float(eps, "eps")
    toler_value = None if toler_chol is None else _finite_float(toler_chol, "toler_chol")
    max_iter, eps_value, toler_value, fix_time = _apply_coxph_control(
        control, max_iter, eps_value, toler_value
    )
    if timefix is not None:
        fix_time = _normalize_bool_option(timefix, "timefix")

    frame = _model_frame(
        formula,
        data,
        subset=subset,
        na_action=na_action,
        weights=weights,
        offset=offset,
        strata_arg=strata,
        cluster=cluster,
        id=id,
        istate=istate,
    )
    if weights_column is not None or id_column is not None:
        frame = replace(
            frame,
            weights_column=frame.weights_column or weights_column,
            id_column=frame.id_column or id_column,
        )
    # istate/statedata only matter for a multi-state response (R keeps istate in the
    # model frame of an ordinary fit)
    if frame.y.type in {"mright", "mcounting"}:
        raise NotImplementedError("multi-state coxph models are not implemented")
    return _coxph_fit_frame(
        frame,
        method=method_name,
        init=init,
        iter_max=max_iter,
        eps=eps_value,
        toler_chol=toler_value,
        timefix=fix_time,
        robust=_normalize_optional_bool_option(robust, "robust"),
        singular_ok=_normalize_bool_option_with_default(singular_ok, "singular_ok", True),
        nocenter=[]
        if nocenter is None
        else _normalize_numeric_sequence_or_none(nocenter, "nocenter"),
        tt=tt,
        keep_model=_normalize_bool_option_with_default(model, "model", False),
        outer_max=(
            control.get("outer.max", control.get("outer_max"))
            if isinstance(control, Mapping)
            else None
        ),
    )


def clogit(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    method: str = "exact",
    **kwargs: Any,
) -> ClogitModel:
    """Conditional logistic regression as a stratified Cox model (R's ``clogit``)."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if not isinstance(formula, str):
        raise TypeError("A formula argument is required")
    response, separator, rhs = formula.partition("~")
    response = response.strip()
    if not separator or not response or not rhs.strip():
        raise ValueError("clogit formula must contain a response and '~'")
    method_name = _match_string_arg(
        method,
        "method",
        ("exact", "approximate", "efron", "breslow"),
        "method must be one of exact, approximate, efron, breslow",
    )
    cox_method = "breslow" if method_name == "approximate" else method_name
    if cox_method == "exact":
        if "cluster(" in rhs:
            raise ValueError("robust variance plus the exact method is not supported")
        if weights is not None:
            warnings.warn(
                "weights ignored: not possible for the exact method", RuntimeWarning, stacklevel=2
            )
            weights = None
    columns = _response_arg_columns(response)
    if not columns:
        raise ValueError("clogit response must name a column of data")
    n = len(_column(data, columns[0]))
    fit = coxph(
        f"Surv(rep(1, {n}), {response}) ~ {rhs.strip()}",
        data=data,
        weights=weights,
        subset=subset,
        na_action=na_action,
        method=cox_method,
        **kwargs,
    )
    return ClogitModel(**fit.__dict__)


# ---------------------------------------------------------------------------
# summary.coxph / coxph.wtest
# ---------------------------------------------------------------------------


def _pchisq_upper(statistic: float, df: int) -> float:
    """``pchisq(x, df, lower.tail=FALSE)``: the regularised upper incomplete gamma
    function Q(df/2, x/2) (series / Lentz continued fraction; no Python binding of
    R's pchisq exists yet)."""

    if math.isnan(statistic) or df <= 0:
        return math.nan
    if statistic <= 0.0:
        return 1.0
    if math.isinf(statistic):
        return 0.0
    a, x = df / 2.0, statistic / 2.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for k in range(1, 1000):
            term *= x / (a + k)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - math.exp(log_prefactor) * total)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for k in range(1, 1000):
        an = -k * (k - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(log_prefactor) * h


def _coefficient_table(
    fit: CoxphModel, scale: float
) -> tuple[list[str], list[dict[str, float | str]]]:
    beta = [value * scale for value in fit.coefficients]
    var = fit.var
    naive = fit.naive_var
    se = [math.sqrt(var[idx][idx]) * scale for idx in range(len(beta))]
    rows: list[dict[str, float | str]] = []
    for idx, (name, value) in enumerate(zip(fit.coef_names, beta, strict=True)):
        z = value / se[idx] if se[idx] > 0.0 else math.nan
        row: dict[str, float | str] = {
            "name": name,
            "coef": value,
            "exp_coef": math.exp(value) if not math.isnan(value) else math.nan,
            "se": se[idx],
            "z": z,
            "p": _pchisq_upper(z * z, 1) if not math.isnan(z) else math.nan,
        }
        if naive is not None:
            row["naive_se"] = math.sqrt(naive[idx][idx])
            row["robust_se"] = se[idx]
        rows.append(row)
    columns = ["coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)"]
    if naive is not None:
        columns.insert(3, "robust se")
    return columns, rows


def summary_coxph(fit: CoxphModel, conf_int: Any = 0.95, scale: Any = 1.0) -> Any:
    """R's ``summary.coxph`` as a dict keyed like the R list (the fit itself for a null
    model, as R returns the object unchanged)."""

    scale_value = _finite_float(scale, "scale")
    beta = fit.coefficients
    if not beta:
        return fit
    df = (
        sum(fit.df)
        if fit.penalized is not None
        else sum(1 for value in beta if not math.isnan(value))
    )
    loglik = fit.loglik
    score = fit.score if fit.score is not None else math.nan
    logtest = -2.0 * (loglik[0] - loglik[1])
    columns, rows = _coefficient_table(fit, scale_value)
    result: dict[str, Any] = {
        "model_type": "coxph",
        "n": fit.n,
        "nevent": fit.nevent,
        "n_event": fit.nevent,
        "loglik": loglik[1],
        "null_loglik": loglik[0],
        "df": df,
        "coefficient_names": list(fit.coef_names),
        "coefficient_columns": columns,
        "coefficients": rows,
        "logtest": {"test": logtest, "df": df, "pvalue": _pchisq_upper(logtest, df)},
        "sctest": {"test": score, "df": df, "pvalue": _pchisq_upper(score, df)},
        "score_test": score,
        "rsq": {
            "rsq": 1.0 - math.exp(-logtest / fit.n),
            "maxrsq": 1.0 - math.exp(2.0 * loglik[0] / fit.n),
        },
        "used_robust": fit.robust,
        "robust": fit.robust,
        "method": fit.method,
        "concordance": {"C": fit.concordance["concordance"], "se(C)": fit.concordance["std"]},
    }
    if conf_int:
        level = _normalize_conf_level(conf_int, "conf_int")
        z = NormalDist().inv_cdf((1.0 + level) / 2.0)
        result["conf_int"] = [
            {
                "name": name,
                "exp(coef)": math.exp(b),
                "exp(-coef)": math.exp(-b),
                "lower": math.exp(b - z * float(row["se"])),
                "upper": math.exp(b + z * float(row["se"])),
            }
            for name, b, row in zip(
                fit.coef_names, [v * scale_value for v in beta], rows, strict=True
            )
        ]
    wald = fit.wald_test
    if wald is not None:
        result["waldtest"] = {"test": round(wald, 2), "df": df, "pvalue": _pchisq_upper(wald, df)}
    if fit.rscore is not None:
        result["robscore"] = {
            "test": fit.rscore,
            "df": df,
            "pvalue": _pchisq_upper(fit.rscore, df),
        }
    return result


def _wtest_b(b: Any) -> tuple[list[list[float | None]], bool]:
    raw = _coerce_array_like(b, "b")
    if raw and isinstance(raw[0], list | tuple):
        width = len(raw[0])
        rows: list[list[float | None]] = []
        for row in raw:
            if not isinstance(row, list | tuple) or len(row) != width:
                raise ValueError("b matrix rows must be rectangular")
            rows.append([None if _is_missing_value(value) else float(value) for value in row])
        return rows, True
    return [[None if _is_missing_value(value) else float(value)] for value in raw], False


def coxph_wtest(var: Any, b: Any, toler_chol: Any = 1e-9) -> CoxPHWTestResult:
    """R's ``coxph.wtest``: the Wald statistic ``b' var^-1 b`` for each column of ``b``."""

    toler = _finite_float(toler_chol, "toler_chol")
    b_rows, b_is_matrix = _wtest_b(b)
    keep = [idx for idx, row in enumerate(b_rows) if all(value is not None for value in row)]
    raw_var = _coerce_array_like(var, "var")
    if raw_var and isinstance(raw_var[0], list | tuple):
        matrix = _as_matrix_rows(raw_var, "var", allow_empty_columns=False)
        var_length = len(matrix) * len(matrix[0])
    else:
        matrix = [[float(value)] for value in raw_var]
        var_length = len(raw_var)
    if len(keep) < len(b_rows):
        b_rows = [b_rows[idx] for idx in keep]
        matrix = [[matrix[row][col] for col in keep] for row in keep] if var_length > 1 else matrix
        var_length = len(matrix) * (len(matrix[0]) if matrix else 0)
    nvar = len(b_rows)
    ntest = len(b_rows[0]) if b_rows else 1
    b_values = [[float(value) for value in row] for row in b_rows]
    if var_length == 0:
        if nvar == 0:
            return CoxPHWTestResult(test=[], df=0, solve=0.0)
        raise ValueError("Argument lengths do not match")
    if var_length == 1:
        if nvar != 1:
            raise ValueError("Argument lengths do not match")
        variance = matrix[0][0]
        if not math.isfinite(variance):
            raise ValueError("infinite argument in coxph.wtest")
        values = b_values[0]
        return CoxPHWTestResult(
            test=[value * value / variance for value in values],
            df=1,
            solve=[value / variance for value in values],
        )
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("First argument must be a square matrix")
    if len(matrix) != nvar:
        raise ValueError("Argument lengths do not match")
    if any(not math.isfinite(value) for row in b_values for value in row) or any(
        not math.isfinite(value) for row in matrix for value in row
    ):
        raise ValueError("infinite argument in coxph.wtest")
    tests = [[b_values[row][col] for row in range(nvar)] for col in range(ntest)]
    result = _core.coxph_wtest(matrix, tests, toler)
    solve_rows = [list(row) for row in result.solve]
    solve: list[float] | list[list[float]] = (
        solve_rows if b_is_matrix and ntest > 1 else [row[0] for row in solve_rows]
    )
    return CoxPHWTestResult(test=list(result.test), df=int(result.df), solve=solve)


# ---------------------------------------------------------------------------
# predict.coxph
# ---------------------------------------------------------------------------


def _prediction_newdata(
    fit: CoxphModel, newdata: Any, *, need_strata: bool, need_response: bool
) -> _NewData:
    return _newdata_frame(
        fit.design,
        fit.terms.strata,
        fit.strata_levels,
        newdata,
        need_strata=need_strata,
        need_response=need_response,
    )


def _terms_selection(terms: Any | None, names: Sequence[str]) -> list[int]:
    """R's ``terms=`` argument of ``predict``: names or 1-based indices of ``assign``."""

    if terms is None:
        return list(range(len(names)))
    values = [terms] if isinstance(terms, str | int) else list(terms)
    selected: list[int] = []
    for value in values:
        if isinstance(value, str):
            if value not in names:
                raise ValueError("a name given in the terms argument not found in the model")
            selected.append(list(names).index(value))
        else:
            idx = _integer_scalar(value, "terms")
            if idx < 1 or idx > len(names):
                raise ValueError("Invalid terms argument")
            selected.append(idx - 1)
    return selected


def _rowsum(values: list[Any], groups: Sequence[Any], *, squares: bool = False) -> list[Any]:
    """R's ``rowsum``: sums per group, groups in sorted order."""

    labels = _materialize_labels(groups, "collapse")
    if len(labels) != len(values):
        raise ValueError("Collapse vector is the wrong length")
    order = sorted(_label_levels(labels, "collapse"), key=lambda v: (isinstance(v, str), v))
    index = {label: idx for idx, label in enumerate(order)}
    matrix = bool(values) and isinstance(values[0], list)
    width = len(values[0]) if matrix else 1
    sums = [[0.0] * width for _ in order]
    for value, label in zip(values, labels, strict=True):
        row = value if matrix else [value]
        for col, item in enumerate(row):
            sums[index[label]][col] += item * item if squares else item
    if squares:
        sums = [[math.sqrt(item) for item in row] for row in sums]
    return sums if matrix else [row[0] for row in sums]


def predict_coxph(
    fit: CoxphModel,
    newdata: Any | None = None,
    *,
    type: str = "lp",
    se_fit: Any = False,
    terms: Any | None = None,
    collapse: Any | None = None,
    reference: str | None = None,
    **kwargs: Any,
) -> Any:
    """R's ``predict.coxph``: ``lp``, ``risk``, ``expected``, ``terms`` or ``survival``.

    Returns the predictions (a list, or one row per observation for ``terms``), or a
    :class:`PredictResult` of predictions and standard errors when ``se_fit``.
    """

    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    if kwargs:
        raise TypeError(f"predict got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if fit.tt:
        raise ValueError("function not defined for models with tt() terms")
    predict_type = _match_string_arg(
        type,
        "type",
        ("lp", "risk", "expected", "terms", "survival"),
        "type must be one of lp, risk, expected, terms, survival",
    )
    include_se = _normalize_bool_option(se_fit, "se_fit")
    if reference is None:
        reference_name = "sample" if predict_type == "terms" else "strata"
    else:
        reference_name = _match_string_arg(
            reference,
            "reference",
            ("strata", "sample", "zero"),
            "reference must be one of strata, sample, zero",
        )
    if predict_type in {"expected", "survival"}:
        reference_name = "sample"

    new: _NewData | None = None
    if newdata is not None:
        need_response = predict_type in {"expected", "survival"}
        new = _prediction_newdata(
            fit, newdata, need_strata=_has_strata(fit), need_response=need_response
        )
        if (
            _has_strata(fit)
            and new.strata is None
            and (reference_name == "strata" or need_response or include_se)
        ):
            raise ValueError("New data must contain the strata variable(s) of the model")
        if need_response and new.y is None:
            raise ValueError("newdata must contain the response variables for type = 'expected'")
        if need_response and new.y is not None and new.y.type != fit.y.type:
            raise ValueError("New data has a different survival type than the model")

    if predict_type == "terms":
        selected = _terms_selection(terms, list(fit.assign))
        result = fit.fit.predict_terms(
            newdata=None if new is None else new.x,
            new_strata=None if new is None else new.strata,
            new_offset=None if new is None else new.offset,
            se_fit=include_se,
            reference=reference_name,
            assign=_active_assign(fit),
        )
        pred: Any = [[row[idx] for idx in selected] for row in result.fit]
        se: Any = (
            None
            if result.se_fit is None
            else [[row[idx] for idx in selected] for row in result.se_fit]
        )
    else:
        result = fit.fit.predict(
            predict_type,
            newdata=None if new is None else new.x,
            new_strata=None if new is None else new.strata,
            new_offset=None if new is None else new.offset,
            new_time=None if new is None or new.y is None else list(new.y.time),
            new_entry=None
            if new is None or new.y is None or new.y.start is None
            else list(new.y.start),
            se_fit=include_se,
            reference=reference_name,
        )
        pred, se = list(result.fit), (None if result.se_fit is None else list(result.se_fit))

    if collapse is not None and collapse is not False:
        pred = _rowsum(pred, collapse)
        if se is not None:
            se = _rowsum(se, collapse, squares=True)
    return PredictResult(pred, se) if include_se else pred


def predict_terms_constant(fit: CoxphModel) -> float:
    """``attr(predict(fit, type='terms'), 'constant')``: ``sum(coef * means)``."""

    return sum(
        coefficient * mean
        for coefficient, mean in zip(fit.coefficients, fit.means, strict=True)
        if not math.isnan(coefficient)
    )


# ---------------------------------------------------------------------------
# residuals.coxph
# ---------------------------------------------------------------------------

_RESIDUAL_TYPES = (
    "martingale",
    "deviance",
    "score",
    "schoenfeld",
    "dfbeta",
    "dfbetas",
    "scaledsch",
    "partial",
)


def _collapse_codes(fit: CoxphModel, collapse: Any) -> list[int] | None:
    """The engine's ``collapse`` groups: ``TRUE`` means the cluster (or id)."""

    if collapse is None or collapse is False:
        return None
    if collapse is True:
        labels = fit.cluster if fit.cluster is not None else fit.id
        if labels is None:
            return None
        labels = list(labels)
    else:
        labels = _materialize_labels(collapse, "collapse")
        if len(labels) != len(fit.residuals):
            raise ValueError("Wrong length for 'collapse'")
    order = sorted(_label_levels(labels, "collapse"), key=lambda v: (isinstance(v, str), v))
    index = {label: idx for idx, label in enumerate(order)}
    return [index[label] for label in labels]


def _drop_single_column(rows: list[list[float]], nvar: int) -> Any:
    return [row[0] for row in rows] if nvar == 1 else rows


def residuals_coxph(
    fit: CoxphModel,
    *,
    type: str = "martingale",
    collapse: Any | None = None,
    weighted: Any | None = None,
    **kwargs: Any,
) -> Any:
    """R's ``residuals.coxph``.

    Score, Schoenfeld and dfbeta residuals are matrices (one row per observation
    or event) that drop to a vector for a one-variable model, as in R.
    """

    if kwargs:
        raise TypeError(
            f"residuals got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    otype = _match_string_arg(
        type, "type", _RESIDUAL_TYPES, f"type must be one of {', '.join(_RESIDUAL_TYPES)}"
    )
    weighted_value = _normalize_optional_bool_option(weighted, "weighted")
    if weighted_value is None:
        weighted_value = otype in {"dfbeta", "dfbetas"}
    if fit.method == "exact" and otype in {"score", "schoenfeld", "scaledsch", "dfbeta", "dfbetas"}:
        raise ValueError(f"{otype} residuals are not available for the exact method")
    codes = _collapse_codes(fit, collapse)
    engine = fit.fit
    nvar = fit.nvar
    if otype == "martingale":
        return list(engine.martingale_residuals(weighted=weighted_value, collapse=codes))
    if otype == "deviance":
        return list(engine.deviance_residuals(weighted=weighted_value, collapse=codes))
    if otype == "score":
        return _drop_single_column(
            engine.score_residuals(weighted=weighted_value, collapse=codes), nvar
        )
    if otype == "dfbeta":
        return _drop_single_column(engine.dfbeta(weighted=weighted_value, collapse=codes), nvar)
    if otype == "dfbetas":
        return _drop_single_column(engine.dfbetas(weighted=weighted_value, collapse=codes), nvar)
    if otype == "partial":
        rows = engine.partial_residuals(
            assign=_active_assign(fit), weighted=weighted_value, collapse=codes
        )
        return [list(row) for row in rows]
    if codes is not None:
        raise ValueError("collapse is not defined for Schoenfeld residuals")
    residuals = (
        engine.schoenfeld_residuals(weighted=weighted_value)
        if otype == "schoenfeld"
        else engine.scaled_schoenfeld_residuals(weighted=weighted_value)
    )
    return _drop_single_column([list(row) for row in residuals.residuals], nvar)


# ---------------------------------------------------------------------------
# survfit.coxph / basehaz
# ---------------------------------------------------------------------------


def _curve_columns(values: list[list[float]]) -> Any:
    """A curve block as R stores it: a vector for one curve, ``ntime x ncurve`` rows otherwise."""

    if values and len(values[0]) == 1:
        return [row[0] for row in values]
    return [list(row) for row in values]


def _confidence_limits(surv: Any, std_err: Any, conf_type: str, conf_int: float) -> tuple[Any, Any]:
    if surv and isinstance(surv[0], list):
        columns = list(zip(*surv, strict=True))
        se_columns = list(zip(*std_err, strict=True))
        bands = [
            _core.survfit_confint(list(p), list(se), True, conf_type, conf_int)
            for p, se in zip(columns, se_columns, strict=True)
        ]
        lower = [list(row) for row in zip(*(band.lower for band in bands), strict=True)]
        upper = [list(row) for row in zip(*(band.upper for band in bands), strict=True)]
        return lower, upper
    band = _core.survfit_confint(list(surv), list(std_err), True, conf_type, conf_int)
    return list(band.lower), list(band.upper)


def _survfit_id_codes(newdata: Any, id: Any, n: int) -> list[int]:
    labels = _materialize_labels(_column_or_values(newdata, id, "id"), "id")
    if len(labels) != n:
        raise ValueError("id must have one value per newdata row")
    index = {label: idx for idx, label in enumerate(_label_levels(labels, "id"))}
    return [index[label] for label in labels]


def _survfit_curves(
    fit: CoxphModel,
    newdata: Any | None,
    *,
    individual: bool,
    id: Any | None,
    stype: int,
    ctype: int,
    se_fit: bool,
    censor: bool,
) -> tuple[list[Any], list[str]]:
    """The engine curves for ``survfit.coxph`` and the name of each block (R's
    ``names(fit$strata)``: the strata levels, or the newdata row numbers)."""

    engine = fit.penalized if fit.penalized is not None else fit.fit
    if newdata is None:
        if any(":" in name for name in fit.assign):
            warnings.warn(
                "the model contains interactions; the default curve based on columm means "
                "of the X matrix is almost certainly not useful. Consider adding a newdata "
                "argument.",
                RuntimeWarning,
                stacklevel=3,
            )
        curves = engine.survfit(stype=stype, ctype=ctype, se_fit=se_fit, censor=censor)
        return curves, [fit.strata_levels[c.stratum] for c in curves] if _has_strata(fit) else []
    new = _prediction_newdata(fit, newdata, need_strata=_has_strata(fit), need_response=individual)
    if individual:
        if new.y is None:
            raise ValueError("newdata must contain the response variables when id is given")
        if new.y.type != fit.y.type:
            raise ValueError("Survival type of newdata does not match the fitted model")
        if new.y.start is None:
            raise ValueError("Individual=TRUE is only valid for counting process data")
        curves = engine.survfit_individual(
            new.x,
            list(new.y.start),
            list(new.y.time),
            _survfit_id_codes(newdata, id, new.n) if id is not None else [0] * new.n,
            new_strata=new.strata,
            new_offset=new.offset,
            stype=stype,
            ctype=ctype,
            se_fit=se_fit,
            censor=censor,
        )
        return curves, [str(idx + 1) for idx in range(len(curves))] if len(curves) > 1 else []
    curves = engine.survfit(
        newdata=new.x,
        new_strata=new.strata,
        new_offset=new.offset,
        stype=stype,
        ctype=ctype,
        se_fit=se_fit,
        censor=censor,
    )
    if new.strata is not None:
        return curves, [str(idx + 1) for idx in range(len(curves))]
    return curves, [fit.strata_levels[c.stratum] for c in curves] if _has_strata(fit) else []


def survfit_coxph(
    fit: CoxphModel,
    newdata: Any | None = None,
    *,
    se_fit: Any = True,
    conf_int: Any = 0.95,
    individual: Any = False,
    stype: Any = 2,
    ctype: Any | None = None,
    conf_type: str = "log",
    censor: Any = True,
    start_time: Any | None = None,
    id: Any | None = None,
    **kwargs: Any,
) -> CoxSurvfitResult:
    """R's ``survfit.coxph``: predicted survival curves from a Cox model.

    Without ``newdata`` the curve is for the average covariate (``fit$means``); with
    ``newdata`` there is one curve per row (per row in its own stratum when the
    strata variables are present, otherwise every stratum for every row).  ``id``
    (with counting-process ``newdata``) gives one time-dependent curve per subject.
    """

    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, True)
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    if kwargs:
        raise TypeError(f"survfit got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if isinstance(fit, ClogitModel):
        raise ValueError("predicted survival curves are not defined for a clogit model")
    if fit.tt:
        raise ValueError("The survfit function can not process coxph models with a tt term")
    if start_time is not None:
        raise NotImplementedError("survfit(start.time=) is not available for Cox models")
    include_se = _normalize_bool_option(se_fit, "se_fit")
    stype_value = _integer_scalar(stype, "stype")
    if stype_value not in (1, 2):
        raise ValueError("stype must be 1 or 2")
    if ctype is None:
        ctype_value = 2 if fit.method == "efron" else 1
    else:
        ctype_value = _integer_scalar(ctype, "ctype")
        if ctype_value not in (1, 2):
            raise ValueError("ctype must be 1 or 2")
    conf_type_name = "none"
    if include_se:
        conf_type_name = _match_string_arg(
            conf_type,
            "conf_type",
            ("log", "log-log", "plain", "none", "logit", "arcsin"),
            "conf.type must be one of log, log-log, plain, none, logit, arcsin",
        )
    level = _normalize_conf_level(conf_int, "conf_int")
    censor_value = _normalize_bool_option(censor, "censor")
    individual_value = _normalize_bool_option(individual, "individual") or id is not None
    if individual_value and newdata is None:
        raise ValueError("the id option only makes sense with new data")

    curves, strata_names = _survfit_curves(
        fit,
        newdata,
        individual=individual_value,
        id=id,
        stype=stype_value,
        ctype=ctype_value,
        se_fit=include_se,
        censor=censor_value,
    )
    surv_rows = [row for curve in curves for row in curve.surv]
    cumhaz_rows = [row for curve in curves for row in curve.cumhaz]
    std_rows = [row for curve in curves for row in (curve.std_err or [])] if include_se else []
    surv = _curve_columns(surv_rows)
    cumhaz = _curve_columns(cumhaz_rows)
    std_err = _curve_columns(std_rows) if include_se else None
    lower = upper = None
    if include_se and conf_type_name != "none":
        lower, upper = _confidence_limits(surv, std_err, conf_type_name, level)
    return CoxSurvfitResult(
        n=[int(curve.n) for curve in curves],
        time=[float(t) for curve in curves for t in curve.time],
        n_risk=[float(v) for curve in curves for v in curve.n_risk],
        n_event=[float(v) for curve in curves for v in curve.n_event],
        n_censor=[float(v) for curve in curves for v in curve.n_censor],
        surv=surv,
        cumhaz=cumhaz,
        type=fit.y.type,
        strata={name: len(curve.time) for name, curve in zip(strata_names, curves, strict=True)}
        if strata_names
        else None,
        std_err=std_err,
        std_chaz=std_err if stype_value == 2 else None,
        lower=lower,
        upper=upper,
        logse=True,
        conf_type=conf_type_name,
        conf_int=level if conf_type_name != "none" else None,
        newdata=newdata,
    )


def basehaz(fit: Any, newdata: Any | None = None, centered: Any = True) -> CoxBaseHazardResult:
    """R's ``basehaz``: the cumulative hazard of ``survfit(fit)`` as a data frame."""

    if not isinstance(fit, CoxphModel):
        raise TypeError("must be a coxph object")
    if isinstance(fit, ClogitModel):
        raise ValueError("predicted survival curves are not defined for a clogit model")
    sfit = survfit_coxph(fit, newdata, se_fit=False)
    hazard: Any = sfit.cumhaz
    if newdata is None and not _normalize_bool_option(centered, "centered"):
        offset = math.exp(-predict_terms_constant(fit))
        hazard = [value * offset for value in hazard]
    strata = None
    if sfit.strata is not None:
        strata = [name for name, count in sfit.strata.items() for _ in range(count)]
    return CoxBaseHazardResult(hazard=hazard, time=list(sfit.time), strata=strata)


# ---------------------------------------------------------------------------
# cox.zph / coxph.detail
# ---------------------------------------------------------------------------


def cox_zph(
    fit: Any,
    transform: Any = "km",
    terms: Any = True,
    singledf: Any = False,
    global_test: Any = True,
    **kwargs: Any,
) -> CoxZPHResult:
    """R's ``cox.zph``: test the proportional hazards assumption of a Cox model."""

    global_test = _pop_dotted_keyword(kwargs, "global", "global_test", global_test, True)
    if kwargs:
        raise TypeError(f"cox_zph got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if not isinstance(fit, CoxphModel):
        raise TypeError("argument must be the result of a coxph fit")
    if not fit.coef_names:
        raise ValueError("there are no score residuals for a Null model")
    if fit.tt:
        raise ValueError("function not defined for models with tt() terms")
    if not isinstance(transform, str):
        raise TypeError("transform must be one of km, rank, identity, log")
    transform_name = _match_string_arg(
        transform, "transform", ("km", "rank", "identity", "log"), "Unrecognized transform"
    )
    use_terms = _normalize_bool_option(terms, "terms")
    aliased = _aliased(fit)
    if use_terms:
        assign = [[col for col in cols if not aliased[col]] for cols in fit.assign.values()]
        names = [name for name, cols in zip(fit.assign, assign, strict=True) if cols]
        assign = [cols for cols in assign if cols]
    else:
        names = [name for name, alias in zip(fit.coef_names, aliased, strict=True) if not alias]
        assign = [[col] for col, alias in enumerate(aliased) if not alias]
    result = _core.cox_zph(
        fit.fit,
        transform=transform_name,
        terms=use_terms,
        singledf=_normalize_bool_option(singledf, "singledf"),
        global_test=_normalize_bool_option(global_test, "global"),
        assign=assign,
    )
    table: list[dict[str, float | int | str]] = [
        {"name": name, "chisq": float(row.chisq), "df": int(row.df), "p": float(row.p)}
        for name, row in zip(names, result.table, strict=True)
    ]
    if result.global_test is not None:
        table.append(
            {
                "name": "GLOBAL",
                "chisq": float(result.global_test.chisq),
                "df": int(result.global_test.df),
                "p": float(result.global_test.p),
            }
        )
    strata = None
    if result.strata is not None and fit.strata_levels:
        strata = [fit.strata_levels[int(code)] for code in result.strata]
    return CoxZPHResult(
        table=table,
        x=list(result.x),
        time=list(result.time),
        y=[list(row) for row in result.y],
        var=[list(row) for row in result.var],
        transform=result.transform,
        names=list(names),
        strata=strata,
    )


def _detail_response(fit: CoxphModel) -> list[list[float]]:
    """``coxph.detail``'s ``y``: always in (start, stop, status) form."""

    y = fit.y
    if y.start is not None:
        return [[s, t, float(e)] for s, t, e in zip(y.start, y.time, y.event, strict=True)]
    mintime = min(y.time) if y.time else 0.0
    start = 2 * mintime - 1 if mintime < 0 else -1.0
    return [[start, t, float(e)] for t, e in zip(y.time, y.event, strict=True)]


def coxph_detail(fit: Any, riskmat: Any = False, rorder: str = "data") -> CoxPHDetailResult:
    """R's ``coxph.detail``: the per-event-time pieces of the Cox partial likelihood."""

    if not isinstance(fit, CoxphModel):
        raise TypeError("coxph_detail requires a fitted coxph model")
    if fit.method not in {"breslow", "efron"}:
        raise ValueError(f"Detailed output is not available for the {fit.method} method")
    order_name = _match_string_arg(
        rorder, "rorder", ("data", "time"), "rorder must be 'data' or 'time'"
    )
    include_riskmat = _normalize_bool_option(riskmat, "riskmat")
    detail = _core.coxph_detail(fit.fit, riskmat=include_riskmat)
    y = _detail_response(fit)
    x = fit.x
    n = len(y)
    strata_codes = fit.fit.strata or [0] * n
    order = sorted(range(n), key=lambda idx: (strata_codes[idx], y[idx][1], -y[idx][2]))
    weights = list(fit.fit.weights)
    weighted = any(value != 1.0 for value in weights)
    strata_table: dict[str, int] | None = None
    if detail.strata is not None and fit.strata_levels:
        strata_table = {}
        for code in detail.strata:
            label = fit.strata_levels[int(code)]
            strata_table[label] = strata_table.get(label, 0) + 1
    risk_rows = None if detail.riskmat is None else [list(row) for row in detail.riskmat]
    if order_name == "time":
        x = [x[idx] for idx in order]
        y = [y[idx] for idx in order]
        if risk_rows is not None:
            risk_rows = [risk_rows[idx] for idx in order]
    return CoxPHDetailResult(
        time=list(detail.time),
        nevent=[int(value) for value in detail.nevent],
        nrisk=[int(value) for value in detail.nrisk],
        hazard=list(detail.hazard),
        varhaz=list(detail.varhaz),
        wtrisk=list(detail.wtrisk),
        means=[list(row) for row in detail.means],
        score=[list(row) for row in detail.score],
        imat=[[list(row) for row in layer] for layer in detail.imat],
        x=x,
        y=y,
        strata=strata_table,
        riskmat=risk_rows,
        sortorder=order if order_name == "time" and include_riskmat else None,
        weights=[weights[idx] for idx in order] if weighted else None,
        nevent_wt=list(detail.nevent_wt) if weighted else None,
        nrisk_wt=list(detail.wtrisk) if weighted else None,
    )


# ---------------------------------------------------------------------------
# anova.coxph / anova.coxphlist
# ---------------------------------------------------------------------------


def _anova_test_name(test: Any) -> str | None:
    if test is None or test is False:
        return None
    if isinstance(test, str) and test.strip().lower() in {"chisq", "chi"}:
        return "Chisq"
    raise ValueError("test must be 'Chisq' or None")


def _nested_frame(fit: CoxphModel, columns: Sequence[int]) -> _ModelFrame:
    """The reduced model frame anova.coxph refits: ``Y ~ X[, columns] + strata + offset``."""

    names = [fit.coef_names[col] for col in columns]
    return _ModelFrame(
        formula=fit.formula,
        data=None,
        y=fit.y,
        x=[[row[col] for col in columns] for row in fit.x],
        design=fit.design,
        terms=fit.terms,
        names=names,
        assign={name: (idx,) for idx, name in enumerate(names)},
        strata=fit.fit.strata,
        strata_levels=fit.strata_levels,
        offset=fit.offset,
        weights=None,
        cluster=None,
        id=None,
        istate=None,
    )


def _anova_single(fit: CoxphModel, test: str | None) -> Any:
    if fit.rscore is not None:
        raise ValueError("Can't do anova tables with robust variances")
    aliased = _aliased(fit)
    term_names = list(fit.assign)
    logliks = [fit.loglik[0]]
    dfs = [0]
    for term_idx in range(len(term_names) - 1):
        columns = [col for name in term_names[: term_idx + 1] for col in fit.assign[name]]
        nested = _coxph_fit_frame(
            _nested_frame(fit, columns),
            method=fit.method,
            init=None,
            iter_max=20,
            eps=None,
            toler_chol=None,
            timefix=False,
            robust=False,
            singular_ok=True,
            nocenter=[-1.0, 0.0, 1.0],
            tt=None,
            keep_model=False,
        )
        logliks.append(nested.loglik[1])
        dfs.append(sum(1 for value in nested.coefficients if not math.isnan(value)))
    if term_names:
        logliks.append(fit.loglik[1])
        dfs.append(sum(1 for value in aliased if not value))
    return _core.anova_coxph(logliks, dfs, ["NULL", *term_names], sequential=True, test=test)


def _anova_list(fits: Sequence[CoxphModel], test: str | None) -> Any:
    if any(fit.rscore is not None for fit in fits):
        raise ValueError("Can't do anova tables with robust variances")
    if any(fit.method != fits[0].method for fit in fits):
        raise ValueError("all models must have the same ties option")
    if any(len(fit.residuals) != len(fits[0].residuals) for fit in fits):
        raise ValueError("models were not all fit to the same size of dataset")
    if any(fit.terms.strata != fits[0].terms.strata for fit in fits):
        raise ValueError("models do not have the same strata")
    responses = [fit.formula.split("~", 1)[0].strip() for fit in fits]
    keep = [idx for idx, response in enumerate(responses) if response == responses[0]]
    if len(keep) < len(fits):
        warnings.warn(
            "Models with a different response were removed because response differs from model 1",
            RuntimeWarning,
            stacklevel=3,
        )
        fits = [fits[idx] for idx in keep]
    if len(fits) == 1:
        return _anova_single(fits[0], test)
    logliks = [fit.loglik[-1] for fit in fits]
    dfs = [sum(1 for value in fit.coefficients if not math.isnan(value)) for fit in fits]
    return _core.anova_coxph(logliks, dfs, None, sequential=False, test=test)


def anova(*fits: Any, test: Any = "Chisq") -> Any:
    """R's ``anova.coxph``: sequential terms of one model, or a list of nested
    models (survreg fits go to ``anova_survreg``)."""

    if len(fits) == 1 and isinstance(fits[0], list | tuple):
        fits = tuple(fits[0])
    if not fits:
        raise TypeError("anova requires at least one fitted model")
    if not isinstance(fits[0], CoxphModel):
        from ._survreg import SurvregModelResult, anova_survreg

        if not isinstance(fits[0], SurvregModelResult):
            raise TypeError("anova requires fitted coxph or survreg models")
        return anova_survreg(*fits, test=test)
    if any(not isinstance(fit, CoxphModel) for fit in fits):
        raise TypeError("All arguments must be Cox models")
    test_name = _anova_test_name(test)
    if len(fits) == 1:
        return _anova_single(fits[0], test_name)
    return _anova_list(list(fits), test_name)
