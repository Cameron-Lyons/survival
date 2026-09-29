"""``coxph``/``clogit`` and the Cox model methods (R/coxph.R, predict.coxph.R,
residuals.coxph.R, survfit.coxph.R, basehaz.R, cox.zph.R, coxph.detail.R,
anova.coxph.R, anova.coxphlist.R, summary.coxph.R, coxph.wtest.R).

Every number comes from the Rust ``CoxPHFit``; this module does what R's R code
does: the model frame, argument checking, dispatch and result labelling.
"""

from __future__ import annotations

import math
import numbers
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _DEFAULT_NA_ACTION,
    _as_character,
    _as_matrix_rows,
    _as_rows,
    _coerce_array_like,
    _control_mapping,
    _cox_tie_method,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _is_bool_like,
    _is_missing_value,
    _label_levels,
    _match_string_arg,
    _materialize_labels,
    _matrix_input_column_names,
    _normalize_bool_option,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
    _normalize_na_action,
    _normalize_numeric_sequence_or_none,
    _normalize_optional_bool_option,
    _pop_dotted_keyword,
    _r_format_number,
    _start_time_value,
    _subset_indices,
    _subset_optional_sequence,
    _warn_outside_package,
)
from ._data_prep import aeqSurv
from ._fit import (
    _design_names_and_assign,
    _excluded_rows,
    _model_frame,
    _ModelFrame,
    _NewData,
    _newdata_columns,
    _newdata_frame,
    _pad_rows,
    _rowsum_excluded,
    _tt_terms,
)
from ._formula import (
    _column,
    _column_or_values,
    _data_row_count,
    _data_rows,
    _design_rows_from_spec,
    _formula_data_rows,
    _formula_design_row_count,
    _response_arg_columns,
    _strata_specs,
    _timeline_counting,
    _timeline_model_frame,
    _timeline_response,
)
from ._names import _make_unique
from ._penalties import _pspline_cbase
from ._surv import Surv
from ._types import (
    CoxBaseHazardResult,
    CoxPHDetailResult,
    CoxPHWTestResult,
    CoxSurvfitMultiStateResult,
    CoxSurvfitResult,
    CoxZPHResult,
    NaAction,
    PredictResult,
    _CovariateTerm,
    _DesignTerm,
    _FormulaDesign,
    _FormulaTerms,
    _InteractionDesignTerm,
    _InteractionTerm,
    _ModelCovariateTerm,
    _PenaltyDesignTerm,
)

_TIE_METHOD_NAMES = ("breslow", "efron", "exact")
_LOG_DOUBLE_MAX = math.log(sys.float_info.max)
# coxph.control's default toler.chol, .Machine$double.eps ^ .75
_TOLER_CHOL = sys.float_info.epsilon**0.75


# ---------------------------------------------------------------------------
# the coxph object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxphModel:
    """R's ``coxph`` object: the engine fit plus what the R list keeps around it.

    The numeric components (``coefficients``, ``var``, ``loglik``, ``residuals``,
    ...) are read through from :class:`survival._survival.CoxPHFit`; ``formula``,
    ``design``, ``assign``, ``coef_names``, ``y``, ``strata_levels`` and ``id`` are
    what ``predict``/``survfit``/``residuals`` need to rebuild the model frame, and
    ``na_action`` (``fit$na.action``) the rows the ``na.action`` removed.
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
    penalized: Any | None = None
    na_action: NaAction | None = None
    # the model frame the fit was made from (after subset and na.action), which
    # model.frame(fit) rebuilds when the fit did not keep it; the design rows are
    # dropped, since model.frame() does not use them
    _frame: _ModelFrame | None = field(default=None, repr=False, compare=False)

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
        """``fit$loglik``: null and fitted values (one value for a null model, as R;
        a penalized fit without coefficients, a frailty alone, keeps both)."""

        values = list(self.fit.loglik)
        return values if self.coef_names or self.penalized is not None else values[:1]

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
        return float(self.fit.wald_test)

    @property
    def iter(self) -> int | list[int] | None:
        if self.penalized is not None:
            return list(self.penalized.iter)
        if not self.coef_names:
            return None
        return int(self.fit.iter)

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
    def strata(self) -> list[str | None] | None:
        """``fit$strata``: the stratum label of every row, ``None`` when unstratified
        (a multi-state fit to a formula list may keep a row without one)."""

        codes = self.fit.strata
        if codes is None or not self.strata_levels:
            return None
        return [self.strata_levels[int(code)] for code in codes]

    def predict(self, newdata: Any | None = None, **kwargs: Any) -> Any:
        return predict_coxph(self, newdata, **kwargs)

    def survfit(self, newdata: Any | None = None, **kwargs: Any) -> CoxSurvfitResult:
        return survfit_coxph(self, newdata, **kwargs)

    def summary(
        self, conf_int: float = 0.95, scale: float = 1.0, terms: bool = False
    ) -> dict[str, Any]:
        return summary_coxph(self, conf_int=conf_int, scale=scale, terms=terms)


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


def _fit_frame(fit: CoxphModel) -> _ModelFrame:
    if fit._frame is None:
        raise TypeError("the fit keeps no model frame")
    return fit._frame


def _model_terms(fit: CoxphModel) -> list[tuple[str, _DesignTerm]]:
    """The model's covariate terms, labelled (R's ``names(fit$pterms)``): those of
    ``fit.assign`` plus a sparse frailty, which has no coefficients."""

    frame = _fit_frame(fit)
    return list(zip(frame.assign, frame.design.covariates, strict=True))


def _term_labels(fit: CoxphModel) -> list[str]:
    return [label for label, _term in _model_terms(fit)]


def _sparse_term(fit: CoxphModel) -> int | None:
    """The position among :func:`_model_terms` of a sparse penalized term."""

    if fit.penalized is None or 2 not in fit.penalized.pterms:
        return None
    return list(fit.penalized.pterms).index(2)


def _coxph_df(fit: CoxphModel) -> float:
    """The model degrees of freedom of R's summary, anova and logLik methods:
    ``sum(fit$df)`` for a penalized fit, else the number of non-NA coefficients."""

    if fit.penalized is not None:
        return float(sum(fit.penalized.df))
    return sum(1 for value in fit.coefficients if not math.isnan(value))


# ---------------------------------------------------------------------------
# coxph
# ---------------------------------------------------------------------------


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
    data = _formula_data_rows(frame.formula, frame.data, tindex, frame.n)
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
        x=_design_rows_from_spec(data, frame.design, len(tindex), evaluated=transformed),
        strata=riskset,
        weights=weights,
        offset=None if frame.offset is None else [frame.offset[idx] for idx in tindex],
        cluster=None if frame.cluster is None else [frame.cluster[idx] for idx in tindex],
        id=None if frame.id is None else [frame.id[idx] for idx in tindex],
    )


def _check_init(init: Any, x: list[list[float]], offset: list[float] | None) -> list[float]:
    """coxph.R's check of ``init``: ``exp(X %*% init - sum(colMeans(X) * init) + offset)``
    at the centred offset must neither overflow nor underflow everywhere."""

    values = _float_vector(init, "init")
    nvar = len(x[0]) if x else 0
    if len(values) != nvar:
        raise ValueError("wrong length for init argument")
    n = len(x)
    means = [sum(row[col] for row in x) / n for col in range(nvar)] if n else []
    center = sum(mean * value for mean, value in zip(means, values, strict=True))
    offset_mean = sum(offset) / n if offset is not None else 0.0
    risks = []
    for idx, row in enumerate(x):
        eta = sum(a * b for a, b in zip(row, values, strict=True)) - center
        if offset is not None:
            eta += offset[idx] - offset_mean
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


def _concordance_summary(cfit: Any) -> dict[str, float]:
    """``fit$concordance``: the summed counts, C and its se of the fit's
    ``concordancefit(reverse=TRUE)`` (NA C and se for data without events)."""

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
    """The convergence warnings of R's Cox fitters for an engine fit (also the R bridge's).

    ``infs = |u %*% imat|``, with the fitter's model-based variance (the naive one of a
    robust fit): after the iterations ran out the fit may be infinite; a converged fit
    whose score still moves a coefficient by more than ``toler.inf`` of its size
    converged before that variable did.  ``coxph.fit`` (right-censored
    Breslow/Efron) also flags a non-finite score; ``agreg.fit`` ((start, stop]
    Breslow/Efron) flags a non-finite score or ``infs > toler.inf * (1 + |coef|)``
    without the ``eps`` floor and stops on an overflowed fit; ``coxexact.fit`` and
    ``agexact.fit`` keep only the ``eps`` and ``toler.inf`` tests.
    """

    coef = list(fit.coefficients)
    nvar = len(coef)
    if nvar == 0 or iter_max <= 1:
        return []
    eps_value = 1e-9 if eps is None else float(eps)
    toler = math.sqrt(eps_value) if toler_inf is None else float(toler_inf)
    u = list(fit.first)
    var = fit.var if fit.naive_var is None else fit.naive_var
    infs = [abs(sum(u[i] * var[i][j] for i in range(nvar))) for j in range(nvar)]
    info = fit.info
    if info is not None:  # agreg.fit
        # the fitter's coefficients, before an aliased one is marked NA
        raw = [0.0 if math.isnan(b) and var[j][j] == 0.0 else b for j, b in enumerate(coef)]
        if not all(math.isfinite(value) for value in [*raw, *(v for row in var for v in row)]):
            raise ValueError(
                "routine failed due to numeric overflow."
                "This should never happen.  Please contact the author."
            )
        if info[3] > 0:
            return ["Ran out of iterations and did not converge"]
        which = [
            j + 1
            for j in range(nvar)
            if not math.isfinite(u[j]) or infs[j] > toler * (1.0 + abs(raw[j]))
        ]
        suffix = "; beta may be infinite. "
    elif _TIE_METHOD_NAMES[int(fit.method)] == "exact":  # coxexact.fit, agexact.fit
        if fit.flag == 1000:
            return ["Ran out of iterations and did not converge"]
        which = [
            j + 1 for j in range(nvar) if infs[j] > eps_value and infs[j] > toler * abs(coef[j])
        ]
        suffix = "; beta may be infinite. "
    else:  # coxph.fit
        if fit.flag == 1000:
            messages = ["Ran out of iterations and did not converge"]
            # coxph.fit's lp is at coxph()'s centred offset
            offset = list(fit.offset)
            lp_max = max(fit.linear_predictors) - sum(offset) / len(offset)
            if lp_max > 500 or any(not math.isfinite(value) for value in infs):
                messages.append("one or more coefficients may be infinite")
            return messages
        which = [
            j + 1
            for j in range(nvar)
            if not math.isfinite(u[j]) or (infs[j] > eps_value and infs[j] > toler * abs(coef[j]))
        ]
        suffix = "; coefficient may be infinite. "
    if not which:
        return []
    return ["Loglik converged before variable " + ",".join(map(str, which)) + suffix]


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
    toler_inf: float | None = None,
    outer_max: int | None = None,
) -> CoxphModel:
    """coxph.R after the model frame: timefix, tt(), robust/cluster, the fit (or the
    fit of data without events), the convergence warnings and the concordance."""

    if frame.y.type not in {"right", "counting"}:
        raise ValueError(f'Cox model doesn\'t support "{frame.y.type}" survival data')
    y = aeqSurv(frame.y) if timefix else frame.y
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
    if data.offset is not None and any(
        not math.isfinite(value) or value > _LOG_DOUBLE_MAX for value in data.offset
    ):
        raise ValueError("offsets must lead to a finite risk score")

    has_cluster = data.cluster is not None
    use_robust = _robust_default(data, has_cluster) if robust is None else robust
    cluster: list[int] | None = None
    if has_cluster:
        if not use_robust:
            # coxph() still hands the cluster to the concordance
            warnings.warn(
                "cluster specified with robust=FALSE, cluster ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        cluster = _cluster_codes(data.cluster or [])
    elif use_robust and data.id is not None:
        cluster = _cluster_codes(data.id)
    if use_robust and cluster is None:
        if data.y.start is None or robust is None:
            cluster = list(range(len(data.y)))
        else:
            raise ValueError("one of cluster or id is needed")
    # without events coxph() returns before checking the predictors or init and
    # before fitting anything, penalized terms included
    no_events = not any(int(value) for value in data.y.event)
    if not no_events and any(not math.isfinite(value) for row in data.x for value in row):
        raise ValueError("data contains an infinite predictor")
    init_values = None if init is None or no_events else _check_init(init, data.x, data.offset)
    penalized_terms = (
        []
        if no_events
        else [
            term
            for term in frame.design.covariates
            if isinstance(term, _PenaltyDesignTerm) and term.penalized
        ]
    )
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
            cluster=cluster,
        )
        fit = penalized.coxph
        dense = [
            i
            for i, term in enumerate(frame.design.covariates)
            if not (isinstance(term, _PenaltyDesignTerm) and term.penalized and term.penalty.sparse)
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
    if not no_events:
        aliased = [idx for idx, value in enumerate(fit.coefficients) if math.isnan(value)]
        if aliased and not singular_ok:
            columns = " ".join(str(idx + 1) for idx in aliased)
            raise ValueError(f"X matrix deemed to be singular; variable {columns}")
        if penalized is None:
            for message in _cox_fit_diagnostic_messages(fit, iter_max, eps, toler_inf):
                warnings.warn(message, RuntimeWarning, stacklevel=3)
        elif iter_max > 1 and penalized.inner_failures:
            # coxpenal.fit's warning, R's spelling kept
            warnings.warn(
                "Inner loop failed to coverge for iterations "
                + " ".join(map(str, penalized.inner_failures)),
                RuntimeWarning,
                stacklevel=3,
            )
    return CoxphModel(
        fit=fit,
        formula=frame.formula,
        design=design,
        terms=frame.terms,
        coef_names=tuple(names),
        assign=dict(assign),
        y=y,
        strata_levels=frame.strata_levels,
        concordance=_concordance_summary(fit.concordance),
        n=frame.n,
        timefix=timefix,
        tt=bool(tt_terms),
        id=None if frame.id is None else tuple(frame.id),
        cluster=None if frame.cluster is None else tuple(frame.cluster),
        model=frame.model_frame() if keep_model else None,
        weights_column=frame.weights_column,
        id_column=frame.id_column,
        penalized=penalized,
        na_action=frame.na_action,
        _frame=replace(frame, x=[]),
    )


def _coxph_model_frame(fit: CoxphModel) -> dict[str, Any]:
    """R's ``model.frame(fit)`` for a Cox model: ``fit$model`` when the fit kept it
    (``model=TRUE``), else the model frame rebuilt from the data it was fitted to."""

    if fit.model is not None:
        return fit.model
    return _timeline_model_frame(_fit_frame(fit).model_frame(), fit.formula)


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


# the coxph.control formals coxph() can only receive through **kwargs
_COXPH_CONTROL_KWARGS = ("iter.max", "toler.chol", "toler.inf", "outer.max")


def _control_number(value: Any, message: str, *, zero_ok: bool = False) -> float:
    """coxph.control's ``if (!is.numeric(x) || x <= 0) stop(message)`` (``x < 0`` when
    ``zero_ok``)."""

    if not isinstance(value, numbers.Real) or _is_bool_like(value):
        raise TypeError(message)
    numeric = float(value)
    if not (numeric >= 0.0 if zero_ok else numeric > 0.0):
        raise ValueError(message)
    return numeric


def _control_integer(value: Any, message: str, *, zero_ok: bool = False) -> int:
    """A checked coxph.control option through ``as.integer()``: truncated, and refused
    with the option's message where as.integer gives ``NA`` (outside R's integer
    range)."""

    numeric = _control_number(value, message, zero_ok=zero_ok)
    if numeric >= 2.0**31:
        raise ValueError(message)
    return int(numeric)


def coxph_control(
    eps: Any = 1e-9,
    toler_chol: Any = _TOLER_CHOL,
    iter_max: Any = 20,
    toler_inf: Any | None = None,
    outer_max: Any = 10,
    timefix: Any = True,
    survcheckallow: Any = "gap",
    **kwargs: Any,
) -> dict[str, Any]:
    """R's ``coxph.control``: the checked fitting options under R's names.

    ``toler_inf`` defaults to ``sqrt(eps)``; ``iter_max`` and ``outer_max`` are
    truncated to integers; ``toler.chol``, ``iter.max``, ``toler.inf`` and
    ``outer.max`` may also be given with their dotted names.  Warns, as R does, when
    ``eps`` is not above ``toler_chol``.  ``survcheckallow`` names the survcheck flags
    (``overlap``, ``gap``, ``jump``, ``teleport``) a multi-state fit lets through.
    """

    toler_chol = _pop_dotted_keyword(kwargs, "toler.chol", "toler_chol", toler_chol, _TOLER_CHOL)
    iter_max = _pop_dotted_keyword(kwargs, "iter.max", "iter_max", iter_max, 20)
    toler_inf = _pop_dotted_keyword(kwargs, "toler.inf", "toler_inf", toler_inf, None)
    outer_max = _pop_dotted_keyword(kwargs, "outer.max", "outer_max", outer_max, 10)
    if kwargs:
        raise TypeError(f"unused argument(s): {', '.join(sorted(kwargs))}")
    iterations = _control_integer(iter_max, "Invalid value for iterations", zero_ok=True)
    eps_value = _control_number(eps, "Invalid convergence criteria")
    toler_value = _control_number(toler_chol, "invalid value for toler.chol")
    if eps_value <= toler_value:
        _warn_outside_package("For numerical accuracy, tolerance should be < eps", RuntimeWarning)
    inf_value = (
        math.sqrt(eps_value)
        if toler_inf is None
        else _control_number(toler_inf, "The toler.inf setting must be >0")
    )
    if not _is_bool_like(timefix):
        raise TypeError("timefix must be TRUE or FALSE")
    outer = _control_integer(outer_max, "invalid value for outer.max")
    from ._coxphms import _survcheckallow

    _survcheckallow(survcheckallow)
    return {
        "eps": eps_value,
        "toler.chol": toler_value,
        "iter.max": iterations,
        "toler.inf": inf_value,
        "outer.max": outer,
        "timefix": bool(timefix),
        "survcheckallow": survcheckallow,
    }


def coxph(
    formula: str | Surv | list[str] | tuple[str, ...] | None = None,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
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
    toler_inf: Any | None = None,
    outer_max: Any | None = None,
    timefix: Any | None = None,
    survcheckallow: Any | None = None,
    **kwargs: Any,
) -> CoxphModel:
    """Fit a Cox proportional hazards model (R's ``coxph``).

    ``formula`` is an R formula string with a ``Surv`` response; ``strata()``,
    ``cluster()``, ``offset()`` and ``tt()`` terms are honoured, as are the
    ``weights``/``offset``/``strata``/``cluster``/``id`` arguments given as vectors
    or as column names of ``data``.  As in R, the :func:`coxph_control` options
    (``eps``, ``toler_chol``, ``iter_max``, ``toler_inf``, ``outer_max``, ``timefix``,
    ``survcheckallow``, dotted or not) are used when no ``control`` is given, and
    ignored otherwise.

    A multi-state response (``Surv(time, state)`` or ``Surv(start, stop, state)`` with a
    factor ``state``) fits R's multi-state model, a :class:`CoxphmsModel`; it needs
    ``id``, and ``istate`` gives the state each row starts in.  ``formula`` may then be
    R's list of formulas (a list of strings): the first with the response and the
    default covariates, then lines ``from:to ~ covariates / options`` (options
    ``common`` and ``shared``), whose state names ``statedata`` may extend.
    """

    formula = _pop_dotted_keyword(kwargs, "response", "formula", formula, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
    singular_ok = _pop_dotted_keyword(kwargs, "singular.ok", "singular_ok", singular_ok, True)
    # the R bridge evaluates weights= / id= itself and names the columns they came from
    weights_column = kwargs.pop("_weights_column", None)
    id_column = kwargs.pop("_id_column", None)
    # coxph.R hands its ... to coxph.control
    control_args = {
        name: value
        for name, value in (
            ("iter_max", iter_max),
            ("eps", eps),
            ("toler_chol", toler_chol),
            ("toler_inf", toler_inf),
            ("outer_max", outer_max),
            ("timefix", timefix),
            ("survcheckallow", survcheckallow),
        )
        if value is not None
    }
    control_args.update({key: kwargs.pop(key) for key in _COXPH_CONTROL_KWARGS if key in kwargs})
    if kwargs:
        raise ValueError(f"Argument {', '.join(sorted(kwargs))} not matched")
    if formula is None:
        raise TypeError("a formula argument is required")
    formulas = None
    if isinstance(formula, list | tuple):
        from ._coxphms import _formula_list

        formulas = _formula_list(formula, statedata)
        formula = formulas.master
    elif isinstance(formula, Surv):
        # coxph(<Surv>, x = <design>): the R bridge's matrix interface, as survreg has
        formula, data = _surv_design_formula(formula, x)
        x = False
    _ = _normalize_bool_option_with_default(x, "x", False)
    _ = _normalize_bool_option_with_default(y, "y", True)

    method_name = _cox_tie_method(method, ties)
    options = (
        coxph_control(**control_args)
        if control is None
        else coxph_control(**_control_mapping(control, "control"))
    )

    arguments = {
        "weights": weights,
        "offset": offset,
        "strata": strata,
        "cluster": cluster,
        "id": id,
        "istate": istate,
    }
    fit_formula = formula
    timeline = _timeline_response(formula)
    if timeline:
        # coxph.R converts timeline data (surv2counting) before its na.action; a
        # cluster() term is by then its cluster argument, which is not carried forward
        weights_column = weights_column or (weights if isinstance(weights, str) else None)
        id_column = id_column or (id if isinstance(id, str) else None)
        fit_formula, data, arguments = _timeline_counting(
            formula, data, subset, arguments, carry_clusters=False
        )
        subset = None
    elif subset is not None and data is not None:
        subset = _subset_indices(subset, _data_row_count(data, fit_formula))
    # a formula list defers its missing values until the transitions are known
    frame = _model_frame(
        fit_formula,
        data,
        subset=subset,
        na_action=na_action if formulas is None else "na.pass",
        weights=arguments["weights"],
        offset=arguments["offset"],
        strata_arg=arguments["strata"],
        cluster=arguments["cluster"],
        id=arguments["id"],
        istate=arguments["istate"],
        deferred_na=formulas is not None,
    )
    if weights_column is not None or id_column is not None:
        frame = replace(
            frame,
            weights_column=frame.weights_column or weights_column,
            id_column=frame.id_column or id_column,
        )
    fit_options: dict[str, Any] = {
        "init": init,
        "iter_max": options["iter.max"],
        "eps": options["eps"],
        "toler_chol": options["toler.chol"],
        "toler_inf": options["toler.inf"],
        "timefix": options["timefix"],
        "robust": _normalize_optional_bool_option(robust, "robust"),
        "singular_ok": _normalize_bool_option_with_default(singular_ok, "singular_ok", True),
        "nocenter": []
        if nocenter is None
        else _normalize_numeric_sequence_or_none(nocenter, "nocenter"),
        "keep_model": _normalize_bool_option_with_default(model, "model", False),
    }
    # istate/statedata only matter for a multi-state response (R keeps istate in the
    # model frame of an ordinary fit)
    if frame.y.type in {"mright", "mcounting"}:
        if strata is not None:
            raise ValueError("use strata() terms in the formula for multi-state models")
        from ._coxphms import _survcheckallow, fit_multistate

        # rownames(mf) before the na.action, which the residuals and predictions carry
        source_rows = range(_data_row_count(data, fit_formula)) if subset is None else subset
        fit: CoxphModel = fit_multistate(
            frame,
            row_labels=_row_names(data, source_rows),
            formulas=formulas,
            na_action=na_action,
            # coxph.R: breslow when neither ties nor method was given
            method="breslow" if ties is None and method is None else method_name,
            survcheckallow=_survcheckallow(options["survcheckallow"]),
            **fit_options,
        )
    elif formulas is not None:
        raise ValueError("formula is a list but the response is not multi-state")
    else:
        fit = _coxph_fit_frame(
            frame, method=method_name, outer_max=options["outer.max"], tt=tt, **fit_options
        )
    if not timeline:
        return fit
    model = None if fit.model is None else _timeline_model_frame(fit.model, formula)
    return replace(fit, formula=formula, model=model)


def clogit(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
    method: str = "exact",
    **kwargs: Any,
) -> ClogitModel:
    """Conditional logistic regression as a stratified Cox model (R's ``clogit``)."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
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
    if not _response_arg_columns(response):
        raise ValueError("clogit response must name a column of data")
    # R's Surv(1 + 0*case, case): a constant time as long as the response after
    # subset and na.action
    fit = coxph(
        f"Surv(rep(1, length({response})), {response}) ~ {rhs.strip()}",
        data=data,
        weights=weights,
        subset=subset,
        na_action=na_action,
        method=cox_method,
        **kwargs,
    )
    return ClogitModel(**fit.__dict__)


# ---------------------------------------------------------------------------
# summary.coxph / summary.coxph.penal / coxph.wtest
# ---------------------------------------------------------------------------


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
            "p": _core.pchisq(z * z, 1.0, lower_tail=False),
        }
        if naive is not None:
            row["naive_se"] = math.sqrt(naive[idx][idx])
            row["robust_se"] = se[idx]
        rows.append(row)
    columns = ["coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)"]
    if naive is not None:
        columns.insert(3, "robust se")
    return columns, rows


def _conf_int_rows(
    names: Sequence[str], beta: Sequence[float], se: Sequence[float], conf_int: Any
) -> list[dict[str, Any]]:
    """The ``conf.int`` table of the summaries: ``exp(coef)``, ``exp(-coef)`` and the
    limits, from the scaled coefficients and standard errors."""

    level = _normalize_conf_level(conf_int, "conf_int")
    z = NormalDist().inv_cdf((1.0 + level) / 2.0)
    return [
        {
            "name": name,
            "exp(coef)": math.exp(b),
            "exp(-coef)": math.exp(-b),
            "lower": math.exp(b - z * error),
            "upper": math.exp(b + z * error),
        }
        for name, b, error in zip(names, beta, se, strict=True)
    ]


def summary_coxph(
    fit: CoxphModel, conf_int: Any = 0.95, scale: Any = 1.0, terms: Any = False
) -> Any:
    """R's ``summary.coxph`` as a dict keyed like the R list (the fit itself for a null
    model, as R returns the object unchanged); a penalized fit dispatches to
    :func:`summary_coxph_penal`, the only method that reads ``terms``."""

    if fit.penalized is not None:
        return summary_coxph_penal(fit, conf_int=conf_int, scale=scale, terms=terms)
    scale_value = _finite_float(scale, "scale")
    beta = fit.coefficients
    if not beta:
        return fit
    df = _coxph_df(fit)
    loglik = fit.loglik
    score = fit.score if fit.score is not None else math.nan
    logtest = -2.0 * (loglik[0] - loglik[1])
    columns, rows = _coefficient_table(fit, scale_value)
    result: dict[str, Any] = {
        "model_type": "coxph",
        "n": fit.n,
        "nevent": fit.nevent,
        "na_action": fit.na_action,
        "loglik": loglik[1],
        "null_loglik": loglik[0],
        "df": df,
        "coefficient_names": list(fit.coef_names),
        "coefficient_columns": columns,
        "coefficients": rows,
        "logtest": {
            "test": logtest,
            "df": df,
            "pvalue": _core.pchisq(logtest, df, lower_tail=False),
        },
        "sctest": {"test": score, "df": df, "pvalue": _core.pchisq(score, df, lower_tail=False)},
        "rsq": {
            "rsq": 1.0 - math.exp(-logtest / fit.n),
            "maxrsq": 1.0 - math.exp(2.0 * loglik[0] / fit.n),
        },
        "used_robust": fit.robust,
        "method": fit.method,
        "concordance": {"C": fit.concordance["concordance"], "se(C)": fit.concordance["std"]},
    }
    if conf_int:
        result["conf_int"] = _conf_int_rows(
            fit.coef_names,
            [value * scale_value for value in beta],
            [float(row["se"]) for row in rows],
            conf_int,
        )
    from ._coxphms import CoxphmsModel

    if isinstance(fit, CoxphmsModel):
        # summary.coxph adds the multi-state maps
        result["cmap"] = fit.cmap
        result["states"] = list(fit.states)
    wald = fit.wald_test
    if wald is not None:
        result["waldtest"] = {
            "test": round(wald, 2),
            "df": df,
            "pvalue": _core.pchisq(wald, df, lower_tail=False),
        }
    if fit.rscore is not None:
        result["robscore"] = {
            "test": fit.rscore,
            "df": df,
            "pvalue": _core.pchisq(fit.rscore, df, lower_tail=False),
        }
    return result


def _penal_row(
    name: str, coef: float, se: float, se2: float, chisq: float, df: float, p: float
) -> dict[str, Any]:
    return {"name": name, "coef": coef, "se": se, "se2": se2, "chisq": chisq, "df": df, "p": p}


def _wald_row(name: str, coef: float, var: float, var2: float) -> dict[str, Any]:
    """A coefficient's row of summary.coxph.penal: ``Chisq = coef^2 / var`` on 1 df."""

    chisq = coef * coef / var if var > 0.0 else math.nan
    return _penal_row(
        name,
        coef,
        math.sqrt(var),
        math.sqrt(var2),
        chisq,
        1.0,
        _core.pchisq(chisq, 1.0, lower_tail=False),
    )


def _block(matrix: list[list[float]], index: Sequence[int]) -> list[list[float]]:
    return [[matrix[i][j] for j in index] for i in index]


def _quadratic_form(x: Sequence[float], matrix: list[list[float]]) -> float:
    return sum(a * row[j] * x[j] for a, row in zip(x, matrix, strict=True) for j in range(len(x)))


def _pspline_print(
    label: str,
    term: _PenaltyDesignTerm,
    coef: list[float],
    var: list[list[float]],
    var2: list[list[float]],
    df: float,
    history: Any,
    digits: int = 7,
) -> tuple[list[dict[str, Any]], str]:
    """pspline()'s ``printfun``: the spline's linear trend (a weighted regression of
    the coefficients on the basis centres ``cbase``) and the test of the rest on
    ``df - 1`` degrees of freedom; theta is formatted to the ``digits`` of the caller's
    ``options(digits)``.  ``cbase`` has a centre for every basis column but the first,
    so as in R a pspline that keeps its intercept column fails coxph.wtest's length
    check."""

    nvar = len(coef) + (0 if term.intercept else 1)
    cbase = _pspline_cbase(term.nterm, term.degree, term.boundary, nvar)
    test1 = coxph_wtest(var, coef).test[0]
    # xmat = cbind(1, cbase) and xsig = V X, for V a g-inverse of var
    xmat = [[1.0, centre] for centre in cbase]
    xsig = coxph_wtest(var, xmat).solve
    # the slope's weights: the second row of [X' V X]^- X' V
    xvx = [
        [sum(x[a] * v[b] for x, v in zip(xmat, xsig, strict=True)) for b in (0, 1)] for a in (0, 1)
    ]
    cmat = coxph_wtest(xvx, [list(row) for row in zip(*xsig, strict=True)]).solve[1]
    linear = sum(c * b for c, b in zip(cmat, coef, strict=True))
    lvar1 = _quadratic_form(cmat, var)
    test2 = linear * linear / lvar1 if lvar1 > 0.0 else math.nan
    nonlinear = test1 - test2
    rows = [
        _penal_row(
            f"{label}, linear",
            linear,
            math.sqrt(lvar1),
            math.sqrt(_quadratic_form(cmat, var2)),
            test2,
            1.0,
            _core.pchisq(test2, 1.0, lower_tail=False),
        ),
        # max(.5, df - 1) stops silly p-values for a chisq of 0 on 0 df
        _penal_row(
            f"{label}, nonlin",
            math.nan,
            math.nan,
            math.nan,
            nonlinear,
            df - 1.0,
            _core.pchisq(nonlinear, max(0.5, df - 1.0), lower_tail=False),
        ),
    ]
    return rows, f"Theta= {_r_format_number(history.theta, digits)}"


def _frailty_print(
    label: str, term: _PenaltyDesignTerm, test: float, df: float, history: Any
) -> tuple[dict[str, Any], str]:
    """The frailty distributions' ``printfun``: the Wald test of the random effects on
    the term's df, and the variance of the random effect."""

    theta = history.history[-1][0] if history.history else history.theta
    text = f"Variance of random effect= {_r_format_number(theta)}"
    if term.penalty.distribution == "gamma":
        text += f"   I-likelihood = {_r_format_number(round(history.c_loglik, 1), 10)}"
    # max(df, .5) stops silly p-values
    p = _core.pchisq(test, max(df, 0.5), lower_tail=False)
    return _penal_row(label, math.nan, math.nan, math.nan, test, df, p), text


def summary_coxph_penal(
    fit: CoxphModel, conf_int: Any = 0.95, scale: Any = 1.0, terms: Any = False
) -> dict[str, Any]:
    """R's ``summary.coxph.penal`` as a dict keyed like the R list.

    ``coefficients`` has one row per term with columns ``coef``, ``se(coef)``,
    ``se2`` (from the sandwich variance ``var2``), ``Chisq``, ``DF`` and ``p``: a
    pspline gives its linear and nonlinear parts, a frailty the Wald test of its
    random effects on the term's df (``print2`` holds their theta), and any other
    coefficient a Wald test on 1 df; ``terms`` makes a multi-column unpenalized
    term one row.  There is no score, Wald or R-squared test.
    """

    penalized = fit.penalized
    if penalized is None:
        raise TypeError("summary_coxph_penal requires a penalized Cox fit")
    scale_value = _finite_float(scale, "scale")
    term_tests = _normalize_bool_option(terms, "terms")
    beta = fit.coefficients
    if not beta and penalized.frail is None:
        raise ValueError("Penalized summary function can't be used for a null model")
    var, var2 = fit.var, fit.var2
    histories = {history.term: history for history in penalized.history}
    rows: list[dict[str, Any]] = []
    print2: list[str] = []
    for i, (label, term) in enumerate(_model_terms(fit)):
        columns, df = penalized.assign2[i], penalized.df[i]
        penalty = term.kind if isinstance(term, _PenaltyDesignTerm) and term.penalized else None
        coef = [] if penalized.pterms[i] == 2 else [beta[col] for col in columns]
        if penalty == "pspline":
            spline_rows, text = _pspline_print(
                label, term, coef, _block(var, columns), _block(var2, columns), df, histories[i]
            )
            rows.extend(spline_rows)
            print2.append(text)
        elif penalty == "frailty":
            if penalized.pterms[i] == 2:
                test = sum(b * b / v for b, v in zip(penalized.frail, penalized.fvar, strict=True))
            else:
                test = coxph_wtest(_block(var, columns), coef).test[0]
            row, text = _frailty_print(label, term, test, df, histories[i])
            rows.append(row)
            print2.append(text)
        elif term_tests and len(columns) > 1:
            test = coxph_wtest(_block(var, columns), coef).test[0]
            p = _core.pchisq(test, 1.0, lower_tail=False)
            rows.append(_penal_row(label, math.nan, math.nan, math.nan, test, df, p))
        else:
            rows.extend(
                _wald_row(fit.coef_names[col], beta[col], var[col][col], var2[col][col])
                for col in columns
            )
    logtest = -2.0 * (fit.loglik[0] - fit.loglik[1])
    df_total = _coxph_df(fit)
    result: dict[str, Any] = {
        "model_type": "coxph.penal",
        "n": fit.n,
        "nevent": fit.nevent,
        "na_action": fit.na_action,
        "loglik": fit.loglik[1],
        "null_loglik": fit.loglik[0],
        "iter": fit.iter,
        "df": list(penalized.df),
        "coefficient_columns": ["coef", "se(coef)", "se2", "Chisq", "DF", "p"],
        "coefficients": rows,
        "print2": print2,
        "logtest": {
            "test": logtest,
            "df": df_total,
            "pvalue": _core.pchisq(logtest, df_total, lower_tail=False),
        },
        "concordance": {"C": fit.concordance["concordance"], "se(C)": fit.concordance["std"]},
    }
    if conf_int and beta:
        result["conf_int"] = _conf_int_rows(
            fit.coef_names,
            [value * scale_value for value in beta],
            [math.sqrt(var[idx][idx]) * scale_value for idx in range(len(beta))],
            conf_int,
        )
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
    fit: CoxphModel,
    newdata: Any,
    *,
    need_strata: bool,
    need_response: bool,
    na_action: str,
    allow_missing_predictors: bool = False,
) -> _NewData:
    return _newdata_frame(
        fit.design,
        _strata_specs(fit.terms),
        fit.strata_levels,
        newdata,
        need_strata=need_strata,
        need_response=need_response,
        na_action=na_action,
        allow_missing_predictors=allow_missing_predictors,
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
    na_action: str | None = "na.pass",
    terms: Any | None = None,
    collapse: Any | None = None,
    reference: str | None = None,
    **kwargs: Any,
) -> Any:
    """R's ``predict.coxph``: ``lp``, ``risk``, ``expected``, ``terms`` or ``survival``.

    Returns the predictions (a list, or one row per observation for ``terms``), or a
    :class:`PredictResult` of predictions and standard errors when ``se_fit``.  Without
    ``newdata`` a ``na.exclude`` fit's predictions are NaN at the rows it removed
    (``napredict``); ``na_action`` applies to ``newdata``, whose incomplete rows are NaN
    (``na.pass``, ``na.exclude``), dropped (``na.omit``) or refused (``na.fail``).  A
    multi-state fit's predictions are :func:`survival.r._coxphms.predict_coxphms`'s.
    """

    from ._coxphms import CoxphmsModel, predict_coxphms

    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "na.pass")
    if kwargs:
        raise TypeError(f"predict got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if isinstance(fit, CoxphmsModel):
        return predict_coxphms(
            fit,
            newdata,
            type=type,
            se_fit=se_fit,
            na_action=na_action,
            terms=terms,
            collapse=collapse,
            reference=reference,
        )
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

    action = _normalize_na_action(na_action)
    new: _NewData | None = None
    if newdata is not None:
        need_response = predict_type in {"expected", "survival"}
        # predict.coxph keeps the strata in Terms2 only when the prediction uses them
        need_strata = _has_strata(fit) and (
            include_se
            or predict_type in {"terms", "expected", "survival"}
            or reference_name == "strata"
            or (reference_name == "zero" and any(value != 0.0 for value in fit.means))
        )
        new = _prediction_newdata(
            fit,
            newdata,
            need_strata=need_strata,
            need_response=need_response,
            na_action=action,
            allow_missing_predictors=True,
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

    pred: Any
    se: Any
    if predict_type == "terms":
        selected = _terms_selection(terms, _term_labels(fit))
    if new is not None and new.n == 0:  # no complete newdata row
        pred, se = [], ([] if include_se else None)
    elif (
        predict_type in {"lp", "risk", "terms"} and _sparse_term(fit) is not None and not fit.assign
    ):
        pred, se = _frailty_prediction(fit, new, include_se)
        if predict_type == "risk":
            pred = [math.exp(value) for value in pred]
    elif predict_type == "terms":
        pred, se = _predict_terms(fit, new, include_se, reference_name, selected)
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

    # napredict restores omitted rows. Under na.pass the numeric kernel already
    # propagated covariate/offset NaNs; gaps here need a missing stratum or time.
    if new is None:
        gaps = _excluded_rows(fit.na_action)
    else:
        gaps = [] if action == "omit" else list(new.missing)
        if new.missing and action == "omit" and collapse is not None and collapse is not False:
            missing = set(new.missing)
            kept = [row for row in range(new.n + len(missing)) if row not in missing]
            collapse = _subset_optional_sequence(collapse, kept, "collapse")
    width = len(selected) if predict_type == "terms" else None
    pred = _pad_rows(pred, gaps, width)
    se = None if se is None else _pad_rows(se, gaps, width)

    if collapse is not None and collapse is not False:
        pred = _rowsum(pred, collapse)
        if se is not None:
            se = _rowsum(se, collapse, squares=True)
    return PredictResult(pred, se) if include_se else pred


def _frailty_prediction(
    fit: CoxphModel, new: _NewData | None, se_fit: bool
) -> tuple[list[float], list[float] | None]:
    """predict.coxph.penal for a model of a sparse frailty alone: the linear predictor
    (for types lp, risk and terms), with the frailties' standard errors ``sqrt(fvar)``
    (not rescaled for the risk, as in R), and 0 for new data."""

    if new is not None:
        return [0.0] * new.n, [0.0] * new.n if se_fit else None
    penalized = fit.penalized
    se = [math.sqrt(penalized.fvar[group]) for group in penalized.frail_index]
    return fit.linear_predictors, se if se_fit else None


def _predict_terms(
    fit: CoxphModel, new: _NewData | None, se_fit: bool, reference: str, selected: list[int]
) -> tuple[list[list[float]], list[list[float]] | None]:
    """The ``terms`` predictions of the ``selected`` model terms (positions among
    :func:`_model_terms`).  As in predict.coxph.penal, a sparse frailty's column holds
    the subjects' frailties (with standard errors ``sqrt(fvar)``), and 0 for new
    data."""

    active = _active_assign(fit)
    position = _sparse_term(fit)
    # the engine's terms are the model terms without the sparse one
    engine_terms = [
        idx if position is None or idx < position else idx - 1
        for idx in selected
        if idx != position
    ]
    result = fit.fit.predict_terms(
        newdata=None if new is None else new.x,
        new_strata=None if new is None else new.strata,
        new_offset=None if new is None else new.offset,
        se_fit=se_fit,
        reference=reference,
        assign=[active[idx] for idx in engine_terms],
    )
    rows, se_rows = result.fit, result.se_fit
    # the frailty column goes in at each place the selection names it, left to right
    columns = [column for column, idx in enumerate(selected) if idx == position]
    if columns:
        penalized = fit.penalized
        for i, row in enumerate(rows):
            group = penalized.frail_index[i] if new is None else None
            value = 0.0 if group is None else penalized.frail[group]
            se = 0.0 if group is None else math.sqrt(penalized.fvar[group])
            for column in columns:
                row.insert(column, value)
                if se_rows is not None:
                    se_rows[i].insert(column, se)
    return rows, se_rows


def predict_terms_constant(fit: CoxphModel) -> float:
    """``attr(predict(fit, type='terms'), 'constant')``: ``sum(coef * means)``.  A
    multi-state fit has no terms prediction (predict.coxphms)."""

    from ._coxphms import _PREDICT_INCOMPLETE, CoxphmsModel

    if isinstance(fit, CoxphmsModel):
        raise ValueError(_PREDICT_INCOMPLETE)
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


def _collapse_codes(fit: CoxphModel, collapse: Any, n: int) -> list[int] | None:
    """The groups of R's ``rowsum(rr, collapse)``: ``TRUE`` means the fit's cluster
    (or id), and a vector must have the ``n`` rows of the residuals."""

    if collapse is None or collapse is False:
        return None
    if collapse is True:
        labels = fit.cluster if fit.cluster is not None else fit.id
        if labels is None:
            return None
        labels = list(labels)
    else:
        labels = _materialize_labels(collapse, "collapse")
        if len(labels) != n:
            raise ValueError("Wrong length for 'collapse'")
    order = sorted(_label_levels(labels, "collapse"), key=lambda v: (isinstance(v, str), v))
    index = {label: idx for idx, label in enumerate(order)}
    return [index[label] for label in labels]


def _drop_single_column(rows: list[list[float]], nvar: int) -> Any:
    return [row[0] for row in rows] if nvar == 1 else rows


@dataclass(frozen=True)
class CoxSchoenfeldResiduals:
    """``residuals(fit, type = "schoenfeld" | "scaledsch")``: one row of ``values`` per
    death (a vector for a one-variable model, as in R), labelled as R labels the matrix:
    ``time`` holds the death times (its row names), ``colnames`` the coefficient names,
    and ``strata`` the deaths per stratum in level order (``attr(, "strata")``, R's
    ``table(strata[deaths])``), ``None`` for an unstratified fit."""

    values: Any = field(repr=False)
    time: list[float] = field(repr=False)
    strata: dict[str, int] | None
    colnames: list[str]


def _schoenfeld_result(fit: CoxphModel, residuals: Any) -> CoxSchoenfeldResiduals:
    strata: dict[str, int] | None = None
    if residuals.strata is not None and fit.strata_levels:
        strata = dict.fromkeys(fit.strata_levels, 0)
        for code in residuals.strata:
            strata[fit.strata_levels[code]] += 1
    return CoxSchoenfeldResiduals(
        values=_drop_single_column(residuals.residuals, fit.nvar),
        time=residuals.time,
        strata=strata,
        colnames=list(fit.coef_names),
    )


def residuals_coxph(
    fit: CoxphModel,
    *,
    type: str = "martingale",
    collapse: Any | None = None,
    weighted: Any | None = None,
    **kwargs: Any,
) -> Any:
    """R's ``residuals.coxph``.

    Score and dfbeta residuals are matrices (one row per observation) that drop to a
    vector for a one-variable model, as in R; Schoenfeld residuals come as a
    :class:`CoxSchoenfeldResiduals`, one row per death.  A ``na.exclude`` fit's
    residuals other than Schoenfeld's are NaN at the rows it removed (``naresid``),
    and a ``collapse`` vector then covers those rows too.  A multi-state fit's
    residuals are :func:`survival.r._coxphms.residuals_coxphms`'s, which also takes
    ``na_action``.
    """

    from ._coxphms import CoxphmsModel, residuals_coxphms

    if isinstance(fit, CoxphmsModel):
        # residuals.coxphms's na.action overrides the kind of the fit's na.action
        na_action = _pop_dotted_keyword(
            kwargs, "na.action", "na_action", kwargs.pop("na_action", None), None
        )
        if kwargs:
            raise TypeError(
                f"residuals got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
            )
        return residuals_coxphms(
            fit, type=type, collapse=collapse, weighted=weighted, na_action=na_action
        )
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
    # residuals.coxph.null
    if not fit.coef_names and fit.penalized is None and otype not in {"martingale", "deviance"}:
        raise ValueError(f"'{otype}' residuals are not defined for a null model")
    if fit.method == "exact" and otype in {"score", "schoenfeld", "scaledsch", "dfbeta", "dfbetas"}:
        raise ValueError(f"{otype} residuals are not available for the exact method")
    excluded = _excluded_rows(fit.na_action)
    codes = _collapse_codes(fit, collapse, len(fit.residuals) + len(excluded))
    engine = fit.fit
    nvar = fit.nvar
    if otype in {"schoenfeld", "scaledsch"}:
        if codes is not None:
            raise ValueError("collapse is not defined for Schoenfeld residuals")
        residuals = (
            engine.schoenfeld_residuals(weighted=weighted_value)
            if otype == "schoenfeld"
            else engine.scaled_schoenfeld_residuals(weighted=weighted_value)
        )
        return _schoenfeld_result(fit, residuals)
    # naresid comes before the collapse: the engine sums the fit's rows, and a group
    # holding a row na.exclude removed sums to NA
    padded_codes = codes if excluded and collapse is not True else None
    fit_codes = codes
    if padded_codes is not None:
        gaps = set(excluded)
        fit_codes = [code for row, code in enumerate(padded_codes) if row not in gaps]
    values: list[Any]
    if otype == "martingale":
        values = list(engine.martingale_residuals(weighted=weighted_value, collapse=fit_codes))
    elif otype == "deviance":
        values = list(engine.deviance_residuals(weighted=weighted_value, collapse=fit_codes))
    elif otype == "partial":
        rows = engine.partial_residuals(
            assign=_active_assign(fit), weighted=weighted_value, collapse=fit_codes
        )
        values = [list(row) for row in rows]
    else:
        method = {
            "score": engine.score_residuals,
            "dfbeta": engine.dfbeta,
            "dfbetas": engine.dfbetas,
        }[otype]
        values = [list(row) for row in method(weighted=weighted_value, collapse=fit_codes)]
    if codes is None:
        values = _pad_rows(values, excluded)
    elif padded_codes is not None:
        values = _rowsum_excluded(values, padded_codes, excluded)
    if otype in {"martingale", "deviance", "partial"}:
        return values
    return _drop_single_column(values, nvar)


# ---------------------------------------------------------------------------
# survfit.coxph / basehaz
# ---------------------------------------------------------------------------


# survfit.coxph's old-style ``type`` values and the stype and ctype each stands for
_SURVFIT_TYPES = (
    "kalbfleisch-prentice",
    "aalen",
    "efron",
    "kaplan-meier",
    "breslow",
    "fleming-harrington",
    "greenwood",
    "tsiatis",
    "exact",
)
_SURVFIT_TYPE_STYPE = (1, 2, 2, 1, 2, 2, 2, 2, 2)
_SURVFIT_TYPE_CTYPE = (1, 1, 2, 1, 1, 2, 1, 1, 1)


def _survfit_types(fit: CoxphModel, type_: Any, stype: Any, ctype: Any) -> tuple[int, int]:
    """``survfit.coxph``'s ``stype`` and ``ctype``: those of the old-style ``type`` when
    neither is given, else stype 2 and the ctype of the fit's ties (2 for Efron)."""

    if type_ is not None:
        if stype is not None or ctype is not None:
            _warn_outside_package("type argument ignored", RuntimeWarning)
        else:
            choices = ", ".join(f'"{name}"' for name in _SURVFIT_TYPES)
            matched = _match_string_arg(
                type_, "type", _SURVFIT_TYPES, f"'type' should be one of {choices}"
            )
            index = _SURVFIT_TYPES.index(matched)
            stype = _SURVFIT_TYPE_STYPE[index]
            if stype != 1:
                ctype = _SURVFIT_TYPE_CTYPE[index]
    if ctype is None:
        ctype_value = 2 if fit.method == "efron" else 1
    else:
        ctype_value = _integer_scalar(ctype, "ctype")
        if ctype_value not in (1, 2):
            raise ValueError("ctype must be 1 or 2")
    stype_value = 2 if stype is None else _integer_scalar(stype, "stype")
    if stype_value not in (1, 2):
        raise ValueError("stype must be 1 or 2")
    return stype_value, ctype_value


def _check_interaction_margins(fit: CoxphModel) -> None:
    """``survfit.coxph`` refuses a model with an interaction whose lower-order terms are
    not all in it (a 2 in ``attr(Terms, "factors")``); strata terms do not count."""

    terms = [
        frozenset(term.term.factors)
        if isinstance(term.term, _InteractionTerm)
        else frozenset([term.term])
        for term in fit.terms.model_terms
        if isinstance(term, _ModelCovariateTerm)
        and not any(
            factor.strata
            for factor in (
                term.term.factors if isinstance(term.term, _InteractionTerm) else [term.term]
            )
        )
    ]
    present = set(terms)
    if any(len(term) > 1 and any(term - {v} not in present for v in term) for term in terms):
        raise ValueError(
            "not able to create a curve for models that contain an interaction without "
            "the lower order effect"
        )


def _curve_block(curves: list[Any], name: str) -> Any:
    """One matrix of the curves (``surv``, ``cumhaz`` or ``std_err``) as R stores it: the
    curves' rows end to end, ``ntime x ncurve`` rows, or a vector for one column."""

    rows = [row for curve in curves for row in getattr(curve, name)]
    if rows and len(rows[0]) == 1:
        return [row[0] for row in rows]
    return rows


def _confidence_limits(surv: Any, std_err: Any, conf_type: str, conf_int: float) -> tuple[Any, Any]:
    """``survfit_confint`` on the whole ``surv`` matrix at once, as R calls it (it works
    elementwise), with the limits cut back into rows."""

    if not (surv and isinstance(surv[0], list)):
        band = _core.survfit_confint(surv, std_err, True, conf_type, conf_int)
        return band.lower, band.upper
    width = len(surv[0])
    band = _core.survfit_confint(
        list(chain.from_iterable(surv)),
        list(chain.from_iterable(std_err)),
        True,
        conf_type,
        conf_int,
    )
    lower, upper = band.lower, band.upper
    starts = range(0, len(lower), width)
    return [lower[i : i + width] for i in starts], [upper[i : i + width] for i in starts]


def _row_names(data: Any, rows: Sequence[int]) -> list[str]:
    """R's ``row.names`` of ``data`` at the 0-based ``rows``: a data frame's own index
    labels when they can be R row names (none missing, no two alike under
    ``as.character``), else the 1-based row numbers (R's automatic row names, which is
    also what ``rbind`` gives two data frames that have them)."""

    index = None if isinstance(data, Mapping) else getattr(data, "index", None)
    if index is not None and not (
        type(index).__name__ == "RangeIndex" and index.start == 0 and index.step == 1
    ):
        labels = [_as_character(label) for label in index]
        if len(set(labels)) == len(labels) and not any(map(_is_missing_value, index)):
            return [labels[row] for row in rows]
    return [str(row + 1) for row in rows]


def _survfit_newdata(
    fit: CoxphModel, newdata: Any, *, individual: bool, id: Any | None, na_action: str
) -> tuple[_NewData, list[int], list[Any] | None]:
    """R's ``model.frame(Terms2, newdata, id = id, na.action = na.omit)`` (``na_action``):
    the newdata pieces at the rows without a missing value in a variable the curves read
    (the ``id`` included), those rows (0-based, for the curve names) and their ``id``."""

    n = _formula_design_row_count(newdata, fit.design)
    rows = list(range(n))
    ids = None
    if id is not None:
        ids = _materialize_labels(_column_or_values(newdata, id, "id"), "id")
        if len(ids) != n:
            raise ValueError("id must have one value per newdata row")
        rows = [row for row in rows if not _is_missing_value(ids[row])]
        if len(rows) < n:
            newdata = _data_rows(newdata, _newdata_columns(newdata), rows, n)
    new = _prediction_newdata(
        fit, newdata, need_strata=_has_strata(fit), need_response=individual, na_action=na_action
    )
    if new.missing:
        dropped = set(new.missing)
        rows = [row for position, row in enumerate(rows) if position not in dropped]
    if not rows:
        raise ValueError("all rows of newdata have missing values")
    return new, rows, None if ids is None else [ids[row] for row in rows]


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
    start_time: float | None = None,
    na_action: str = "na.omit",
) -> tuple[list[Any], list[str], list[str] | None]:
    """The engine curves for ``survfit.coxph``, the name of each block (R's
    ``names(fit$strata)``: the strata levels, the id values or the newdata row names) and
    the name of each column when every block holds a curve per newdata row (the row names,
    R's ``colnames(fit$surv)``).  ``na_action = "na.fail"`` refuses the newdata rows
    ``na.omit`` would leave out."""

    _check_interaction_margins(fit)
    if newdata is None and any(
        isinstance(term, _InteractionDesignTerm)
        and any(factor.term.strata for factor in term.factors)
        for term in fit.design.covariates
    ):
        raise ValueError("Models with strata by covariate interaction terms require newdata")
    engine = fit.penalized if fit.penalized is not None else fit.fit
    options: dict[str, Any] = {
        "stype": stype,
        "ctype": ctype,
        "se_fit": se_fit,
        "censor": censor,
        "start_time": start_time,
    }
    if newdata is None:
        if any(":" in name for name in fit.assign):
            _warn_outside_package(
                "the model contains interactions; the default curve based on columm means "
                "of the X matrix is almost certainly not useful. Consider adding a newdata "
                "argument.",
                RuntimeWarning,
            )
        curves = engine.survfit(**options)
        names = [fit.strata_levels[c.stratum] for c in curves] if _has_strata(fit) else []
        return curves, names, None
    new, rows, ids = _survfit_newdata(
        fit, newdata, individual=individual, id=id, na_action=na_action
    )
    if individual:
        if new.y is None:
            raise ValueError("newdata must contain the response variables when id is given")
        if new.y.type != fit.y.type:
            raise ValueError("Survival type of newdata does not match the fitted model")
        if new.y.start is None:
            raise ValueError("Individual=TRUE is only valid for counting process data")
        if ids is None:  # individual = TRUE: one subject
            codes, labels = [0] * new.n, []
        else:
            # coxsurv.fit's curves are the unique ids in order of first appearance, named
            # by as.character; make.unique keeps apart distinct ids that print alike
            # (0.1 + 0.2 and 0.3), whose repeated names R keeps but a dict cannot
            levels = _label_levels(ids, "id")
            index = {label: code for code, label in enumerate(levels)}
            codes = [index[label] for label in ids]
            labels = _make_unique([_as_character(label) for label in levels])
        curves = engine.survfit_individual(
            new.x,
            list(new.y.start),
            list(new.y.time),
            codes,
            new_strata=new.strata,
            new_offset=new.offset,
            **options,
        )
        return curves, labels if len(curves) > 1 else [], None
    curves = engine.survfit(newdata=new.x, new_strata=new.strata, new_offset=new.offset, **options)
    if new.strata is not None:
        return curves, _row_names(newdata, rows), None
    names = [fit.strata_levels[c.stratum] for c in curves] if _has_strata(fit) else []
    return curves, names, _row_names(newdata, rows)


def survfit_coxph(
    fit: CoxphModel,
    newdata: Any | None = None,
    *,
    se_fit: Any | None = None,
    conf_int: Any = 0.95,
    individual: Any | None = None,
    stype: Any | None = None,
    ctype: Any | None = None,
    conf_type: str = "log",
    censor: Any = True,
    start_time: Any | None = None,
    id: Any | None = None,
    type: str | None = None,
    **kwargs: Any,
) -> CoxSurvfitResult | CoxSurvfitMultiStateResult:
    """R's ``survfit.coxph``: predicted survival curves from a Cox model.

    Without ``newdata`` the curve is for the average covariate (``fit$means``); with
    ``newdata`` there is one curve per row (per row in its own stratum when the
    strata variables are present, otherwise every stratum for every row), and rows
    with a missing value are left out (R's ``na.omit``).  ``id`` (with
    counting-process ``newdata``) gives one time-dependent curve per subject.
    ``stype``/``ctype`` default to 2 and the tie method; the old-style ``type``
    (``"kalbfleisch-prentice"``, ``"aalen"``, ``"efron"``, ...) sets them when neither is
    given.  ``start_time`` builds the curves from the rows still at risk at that time.
    ``se_fit`` defaults to true.  A multi-state fit goes to
    :func:`survival.r._coxphms.survfit_coxphms`, with the further keywords of that method.
    """

    from ._coxphms import CoxphmsModel, survfit_coxphms

    if isinstance(fit, CoxphmsModel):
        if se_fit is not None:
            kwargs["se_fit"] = se_fit
        return survfit_coxphms(
            fit,
            newdata,
            conf_int=conf_int,
            individual=False if individual is None else individual,
            stype=stype,
            ctype=ctype,
            conf_type=conf_type,
            censor=censor,
            start_time=start_time,
            id=id,
            type=type,
            **kwargs,
        )
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, None)
    se_fit = True if se_fit is None else se_fit
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    if kwargs:
        raise TypeError(f"survfit got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if isinstance(fit, ClogitModel):
        raise ValueError("predicted survival curves are not defined for a clogit model")
    if fit.tt:
        raise ValueError("The survfit function can not process coxph models with a tt term")
    stype_value, ctype_value = _survfit_types(fit, type, stype, ctype)
    include_se = _normalize_bool_option(se_fit, "se_fit")
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
    individual_value = id is not None
    if individual is not None:
        _warn_outside_package("the `id' option supersedes `individual'", RuntimeWarning)
        individual_value = _normalize_bool_option(individual, "individual") or individual_value
    if individual_value and newdata is None:
        raise ValueError("the id option only makes sense with new data")
    start = _start_time_value(start_time)

    curves, strata_names, column_names = _survfit_curves(
        fit,
        newdata,
        individual=individual_value,
        id=id,
        stype=stype_value,
        ctype=ctype_value,
        se_fit=include_se,
        censor=censor_value,
        start_time=start,
    )
    surv = _curve_block(curves, "surv")
    std_err = _curve_block(curves, "std_err") if include_se else None
    lower = upper = None
    if include_se and conf_type_name != "none":
        lower, upper = _confidence_limits(surv, std_err, conf_type_name, level)
    return CoxSurvfitResult(
        n=[curve.n for curve in curves],
        time=[t for curve in curves for t in curve.time],
        n_risk=[v for curve in curves for v in curve.n_risk],
        n_event=[v for curve in curves for v in curve.n_event],
        n_censor=[v for curve in curves for v in curve.n_censor],
        surv=surv,
        cumhaz=_curve_block(curves, "cumhaz"),
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
        start_time=start,
        newdata=newdata,
        colnames=column_names if surv and isinstance(surv[0], list) else None,
    )


def basehaz(fit: Any, newdata: Any | None = None, centered: Any = True) -> CoxBaseHazardResult:
    """R's ``basehaz``: the cumulative hazard of ``survfit(fit)`` as a data frame."""

    from ._coxphms import CoxphmsModel

    if not isinstance(fit, CoxphModel):
        raise TypeError("must be a coxph object")
    if isinstance(fit, ClogitModel):
        raise ValueError("predicted survival curves are not defined for a clogit model")
    if isinstance(fit, CoxphmsModel):
        raise ValueError("the basehaz function is not implemented for multi-state models")
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
    """R's ``cox.zph``: test the proportional hazards assumption of a Cox model.

    ``transform`` is ``"km"``, ``"rank"``, ``"identity"``, ``"log"`` or a function
    of the (stop) times, which receives them as a list; the result is labelled
    by the function's name, or ``"user"`` for an anonymous one.  For a penalized
    fit the penalty enters the information matrix and the degrees of freedom are
    ``fit$df``, as in R.

    A multi-state fit is tested on its stacked data (coxph.getdata stacks it), with
    one term per model term and transition; its ``strata`` are the stacked strata
    ``"1"``, ``"2"``, ...  Unlike R, this works for models with ``strata()`` terms or
    ``ph()`` coefficients.
    """

    global_test = _pop_dotted_keyword(kwargs, "global", "global_test", global_test, True)
    if kwargs:
        raise TypeError(f"cox_zph got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    from ._coxphms import CoxphmsModel, _stacked_strata_labels, _zph_assign

    if not isinstance(fit, CoxphModel):
        raise TypeError("argument must be the result of a coxph fit")
    multistate = isinstance(fit, CoxphmsModel)
    if not fit.coef_names:
        raise ValueError("there are no score residuals for a Null model")
    if fit.tt:
        raise ValueError("function not defined for models with tt() terms")
    transform_arg: str | list[float]
    if isinstance(transform, str):
        transform_name = _match_string_arg(
            transform, "transform", ("km", "rank", "identity", "log"), "Unrecognized transform"
        )
        transform_arg = transform_name
    elif callable(transform):
        name = getattr(transform, "__name__", "")
        transform_name = name if name.isidentifier() else "user"
        times = fit.fit.time if multistate else fit.y.time
        transform_arg = [float(value) for value in transform(list(times))]
    else:
        raise TypeError("transform must be one of km, rank, identity, log, or a function")
    use_terms = _normalize_bool_option(terms, "terms")
    groups: list[tuple[str, Sequence[int]]]
    if not use_terms:
        groups = [(name, [col]) for col, name in enumerate(fit.coef_names)]
    elif isinstance(fit, CoxphmsModel):  # narrows fit, which multistate does not
        groups = list(_zph_assign(fit))
    else:
        groups = list(fit.assign.items())
    aliased = _aliased(fit)
    groups = [(name, [col for col in cols if not aliased[col]]) for name, cols in groups]
    names = [name for name, cols in groups if cols]
    assign = [list(cols) for _name, cols in groups if cols]
    result = _core.cox_zph(
        fit.penalized if fit.penalized is not None else fit.fit,
        transform=transform_arg,
        terms=use_terms,
        singledf=_normalize_bool_option(singledf, "singledf"),
        global_test=_normalize_bool_option(global_test, "global"),
        assign=assign,
    )
    table: list[dict[str, float | str]] = [
        {"name": name, "chisq": float(row.chisq), "df": float(row.df), "p": float(row.p)}
        for name, row in zip(names, result.table, strict=True)
    ]
    if result.global_test is not None:
        table.append(
            {
                "name": "GLOBAL",
                "chisq": float(result.global_test.chisq),
                "df": float(result.global_test.df),
                "p": float(result.global_test.p),
            }
        )
    strata = None
    if result.strata is not None and multistate:
        strata = _stacked_strata_labels(result.strata)
    elif result.strata is not None and fit.strata_levels:
        strata = [fit.strata_levels[int(code)] for code in result.strata]
    return CoxZPHResult(
        table=table,
        x=list(result.x),
        time=list(result.time),
        y=[list(row) for row in result.y],
        var=[list(row) for row in result.var],
        transform=transform_name,
        names=list(names),
        strata=strata,
    )


def _detail_response(
    start: Sequence[float] | None, time: Sequence[float], status: Sequence[Any]
) -> list[list[float]]:
    """``coxph.detail``'s ``y``: always in (start, stop, status) form."""

    if start is not None:
        return [[s, t, float(e)] for s, t, e in zip(start, time, status, strict=True)]
    mintime = min(time) if time else 0.0
    begin = 2 * mintime - 1 if mintime < 0 else -1.0
    return [[begin, t, float(e)] for t, e in zip(time, status, strict=True)]


def coxph_detail(fit: Any, riskmat: Any = False, rorder: str = "data") -> CoxPHDetailResult:
    """R's ``coxph.detail``: the per-event-time pieces of the Cox partial likelihood.

    A multi-state fit is described on its stacked data (coxph.getdata stacks it): ``x``
    and ``y`` have one row per stacked row, and ``strata`` counts the times of each
    stacked stratum ``"1"``, ``"2"``, ...  Unlike R, this works for models with
    ``strata()`` terms.
    """

    from ._coxphms import CoxphmsModel, _stacked_strata_labels

    if not isinstance(fit, CoxphModel):
        raise TypeError("coxph_detail requires a fitted coxph model")
    multistate = isinstance(fit, CoxphmsModel)
    if fit.method not in {"breslow", "efron"}:
        raise ValueError(f"Detailed output is not available for the {fit.method} method")
    order_name = _match_string_arg(
        rorder, "rorder", ("data", "time"), "rorder must be 'data' or 'time'"
    )
    include_riskmat = _normalize_bool_option(riskmat, "riskmat")
    detail = _core.coxph_detail(fit.fit, riskmat=include_riskmat)
    if multistate:
        engine = fit.fit
        y = _detail_response(engine.entry, engine.time, engine.status)
        x = [list(row) for row in engine.x]
    else:
        y = _detail_response(fit.y.start, fit.y.time, fit.y.event)
        x = fit.x
    n = len(y)
    strata_codes = fit.fit.strata or [0] * n
    order = sorted(range(n), key=lambda idx: (strata_codes[idx], y[idx][1], -y[idx][2]))
    weights = list(fit.fit.weights)
    weighted = any(value != 1.0 for value in weights)
    strata_table: dict[str, int] | None = None
    if detail.strata is not None and (multistate or fit.strata_levels):
        labels = (
            _stacked_strata_labels(detail.strata)
            if multistate
            else [fit.strata_levels[int(code)] for code in detail.strata]
        )
        strata_table = {}
        for label in labels:
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


def _model_matrix_by_term(fit: CoxphModel) -> list[tuple[list[str], list[list[float]]]]:
    """``model.matrix(fit)`` split by term: each term's column names and columns.  A
    penalized term enters with its basis columns, a sparse frailty with its group
    codes (``as.numeric(factor(x))``), which is how R's model matrix holds them."""

    x = fit.x
    blocks = [
        ([fit.coef_names[col] for col in cols], [[row[col] for row in x] for col in cols])
        for cols in fit.assign.values()
    ]
    position = _sparse_term(fit)
    if position is not None:
        codes = [float(group + 1) for group in fit.penalized.frail_index]
        blocks.insert(position, ([_term_labels(fit)[position]], [codes]))
    return blocks


def _nested_frame(fit: CoxphModel, names: list[str], columns: list[list[float]]) -> _ModelFrame:
    """The reduced model anova.coxph refits, ``Y ~ X[, assign <= k] + strata + offset``:
    plain numeric columns, so every term is refitted unpenalized."""

    return _ModelFrame(
        formula=fit.formula,
        data=None,
        y=fit.y,
        x=[list(row) for row in zip(*columns, strict=True)],
        design=replace(fit.design, covariates=(), term_assignments=()),
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
    """anova.coxph for one model.  As in R, where anova.coxph.penal is not registered,
    the leading terms of a penalized model are refitted unpenalized and the full model
    counts ``sum(fit$df)``, so a step's Df can be fractional or negative."""

    if fit.rscore is not None:
        raise ValueError("Can't do anova tables with robust variances")
    blocks = _model_matrix_by_term(fit)
    logliks = [fit.loglik[0]]
    dfs = [0.0]
    for k in range(1, len(blocks)):
        nested = _coxph_fit_frame(
            _nested_frame(
                fit,
                [name for names, _ in blocks[:k] for name in names],
                [column for _, columns in blocks[:k] for column in columns],
            ),
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
        dfs.append(_coxph_df(nested))
    if blocks:
        logliks.append(fit.loglik[1])
        dfs.append(_coxph_df(fit))
    return _core.anova_coxph(logliks, dfs, ["NULL", *_term_labels(fit)], sequential=True, test=test)


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
    dfs = [_coxph_df(fit) for fit in fits]
    return _core.anova_coxph(logliks, dfs, None, sequential=False, test=test)


def anova(*fits: Any, test: Any = "Chisq") -> Any:
    """R's ``anova.coxph``: sequential terms of one model, or a list of nested
    models (survreg fits go to ``anova_survreg``)."""

    if len(fits) == 1 and isinstance(fits[0], list | tuple):
        fits = tuple(fits[0])
    if not fits:
        raise TypeError("anova requires at least one fitted model")
    from ._coxphms import CoxphmsModel

    if any(isinstance(fit, CoxphmsModel) for fit in fits):
        # anova.coxphms (reached through anova.coxph) stops here
        raise NotImplementedError("anova not yet available for multistate")
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
