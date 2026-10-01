"""Model generics (``coef``, ``vcov``, ``predict``, ``residuals``, ``model_summary``,
``model_frame``, ``as_data_frame``, ...): R's S3 generics as
:func:`functools.singledispatch` functions.

Each generic registers R's methods for the classes that have one (coxph and clogit,
cch, aareg, survreg, concordance, survfit, pyears); any other object raises
``TypeError``.  The survreg methods are the ``*_survreg`` functions of
:mod:`survival.r._survreg` (``predict_survreg``, ``residuals_survreg``,
``model_summary_survreg``, ...).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from functools import singledispatch
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from .. import _survival as _core
from ._aareg import summary_aareg
from ._cch import summary_cch
from ._coerce import (
    _coefficient_selection,
    _coerce_array_like,
    _integer_scalar,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
)
from ._coxph import (
    CoxphModel,
    _coxph_df,
    _coxph_model_frame,
    _has_strata,
    _prediction_newdata,
    _term_labels,
    _terms_selection,
    predict_coxph,
    residuals_coxph,
    summary_coxph,
)
from ._coxph import predict_terms_constant as predict_terms_constant  # re-exported by survival.r
from ._coxphms import CoxphmsModel, coef_coxphms, vcov_coxphms
from ._data_prep import summary_tmerge
from ._finegray import _finegray_frame
from ._formula import _column as _formula_column
from ._formula import _formula_columns, _formula_model_term_degree
from ._formula import model_frame as _formula_model_frame
from ._names import _make_unique
from ._pyears import (
    RateTableMatch,
    RateTableSummary,
    _pyears_result_frame,
    _survexp_frame,
    summary_pyears,
    summary_ratetable,
    summary_survexp,
)
from ._surv import Surv
from ._surv_summary_print import SurvivalTablePrint
from ._survfit import (
    _derived_survfit,
    _engine_of,
    median_surv,
    median_survfit,
    quantile_surv,
    quantile_survfit,
    summary_survfit,
)
from ._survfit_print import SurvfitPrint
from ._survfit_residuals import survfit_residuals
from ._survpenal_print import SurvregPenalPrint
from ._survreg import (
    SurvregAnovaResult,
    SurvregModelResult,
    coef_names_survreg,
    confint_survreg,
    model_matrix_survreg,
    model_summary_survreg,
    model_term_names_survreg,
    predict_survreg,
    residuals_survreg,
    survreg_df,
    vcov_survreg,
)
from ._types import (
    AaregModelResult,
    CchModelResult,
    ConcordanceResult,
    CoxBaseHazardResult,
    CoxPHDetailResult,
    CoxSurvfitMultiStateResult,
    CoxSurvfitResult,
    CoxZPHResult,
    ModelPrint,
    PyearsResult,
    RateTablePrint,
    ResponsePrint,
    SummarySurvfitCoxmsResult,
    SurvDiffResult,
    SurvExpResult,
    SurvExpSummary,
    SurvfitMultiStateResult,
    SurvfitQuantileResult,
    SurvfitResult,
    TMergeFrame,
    YatesPrint,
    _ModelCovariateTerm,
    _ModelStrataTerm,
)

_SurvfitCurves = SurvfitResult | SurvfitMultiStateResult | CoxSurvfitResult


@singledispatch
def quantile(x: Any, probs: Any = (0.25, 0.5, 0.75), **kwargs: Any) -> SurvfitQuantileResult:
    """R's quantile methods for ``Surv`` responses and fitted survival curves.

    ``probs`` defaults to the quartiles; ``conf_int``, ``scale`` and ``tolerance``
    are the curve quantile options. Responses additionally accept ``na_rm``.
    """

    raise TypeError("quantile requires a Surv response or survfit object")


@singledispatch
def median(x: Any, **kwargs: Any) -> SurvfitQuantileResult:
    """R's median methods for responses and survival curves.

    A response includes confidence bounds by default; a fitted curve returns
    only its median. Both return a ``SurvfitQuantileResult`` with ``probs=[0.5]``.
    """

    raise TypeError("median requires a Surv response or survfit object")


quantile.register(Surv, quantile_surv)
median.register(Surv, median_surv)
quantile.register(
    SurvfitResult | CoxSurvfitResult | SurvfitMultiStateResult | CoxSurvfitMultiStateResult,
    quantile_survfit,
)
median.register(
    SurvfitResult | CoxSurvfitResult | SurvfitMultiStateResult | CoxSurvfitMultiStateResult,
    median_survfit,
)


def _no_method(generic: str) -> TypeError:
    return TypeError(f"{generic} requires a fitted coxph or survreg model")


# ---------------------------------------------------------------------------
# coefficients and likelihoods
# ---------------------------------------------------------------------------


@singledispatch
def coef(fit: Any, *, matrix: Any = False) -> Any:
    """``coef``: ``fit$coefficients`` (``NaN`` marks an aliased coefficient, like R's
    ``NA``; a survreg fit's location coefficients), or a concordance's estimate.
    ``matrix`` is ``coef.coxphms``'s: a multi-state fit's coefficients laid out like its
    ``cmap``; other fits ignore it."""

    raise _no_method("coef")


@coef.register(CoxphModel | CchModelResult | SurvregModelResult)
def _coef_fit(
    fit: CoxphModel | CchModelResult | SurvregModelResult, *, matrix: Any = False
) -> list[float]:
    return fit.coefficients


@coef.register(CoxphmsModel)
def _coef_coxphms(fit: CoxphmsModel, *, matrix: Any = False) -> Any:
    return coef_coxphms(fit, matrix=_normalize_bool_option_with_default(matrix, "matrix", False))


@coef.register(ConcordanceResult)
def _coef_concordance(fit: ConcordanceResult, *, matrix: Any = False) -> Any:
    # coef.concordance
    return fit.concordance


@singledispatch
def coef_names(fit: Any, *, complete: Any | None = None) -> list[str]:
    """``names(coef(fit))``; ``complete=False`` drops aliased coefficients."""

    raise _no_method("coef_names")


@coef_names.register(CoxphModel | CchModelResult)
def _coef_names_cox(fit: CoxphModel | CchModelResult, *, complete: Any | None = None) -> list[str]:
    include = _normalize_bool_option_with_default(complete, "complete", True)
    names = list(fit.coef_names)
    if include:
        return names
    return [name for name, b in zip(names, fit.coefficients, strict=True) if not math.isnan(b)]


coef_names.register(SurvregModelResult, coef_names_survreg)


@singledispatch
def vcov(fit: Any, *, complete: Any = True, matrix: Any = False) -> Any:
    """``vcov``: the robust variance when the fit used one, else the model-based one;
    a concordance's variance.  ``matrix`` is ``vcov.coxphms``'s: a multi-state fit's
    variance per transition (see :func:`survival.r._coxphms.vcov_coxphms`); other fits
    ignore it."""

    raise _no_method("vcov")


@vcov.register(CoxphModel | CchModelResult)
def _vcov_cox(
    fit: CoxphModel | CchModelResult, *, complete: Any = True, matrix: Any = False
) -> list[list[float]]:
    include = _normalize_bool_option_with_default(complete, "complete", True)
    var = fit.var
    if include:
        return var
    keep = [i for i, b in enumerate(fit.coefficients) if not math.isnan(b)]
    return [[var[i][j] for j in keep] for i in keep]


@vcov.register(CoxphmsModel)
def _vcov_coxphms(fit: CoxphmsModel, *, complete: Any = True, matrix: Any = False) -> Any:
    return vcov_coxphms(
        fit,
        complete=_normalize_bool_option_with_default(complete, "complete", True),
        matrix=_normalize_bool_option_with_default(matrix, "matrix", False),
    )


@vcov.register(SurvregModelResult)
def _vcov_survreg(fit: SurvregModelResult, *, complete: Any = True, matrix: Any = False) -> Any:
    return vcov_survreg(fit, complete=complete)


@vcov.register(ConcordanceResult)
def _vcov_concordance(fit: ConcordanceResult, *, complete: Any = True, matrix: Any = False) -> Any:
    # vcov.concordance(object, ...): complete is one of the ignored arguments
    return fit.var


@singledispatch
def loglik(fit: Any) -> float:
    """``logLik``: the fitted log-likelihood ``fit$loglik[2]`` (``loglik[1]`` for a null
    Cox model, as logLik.coxph.null)."""

    raise _no_method("loglik")


@loglik.register(CoxphModel | SurvregModelResult)
def _loglik_fit(fit: CoxphModel | SurvregModelResult) -> float:
    return fit.loglik[-1]


@singledispatch
def nobs(fit: Any) -> int:
    """``nobs``: the number of events for a Cox model (the ``nobs`` attribute of its
    logLik), the number of observations for survreg."""

    raise _no_method("nobs")


@nobs.register(CoxphModel)
def _nobs_cox(fit: CoxphModel) -> int:
    return fit.nevent


@nobs.register(SurvregModelResult)
def _nobs_survreg(fit: SurvregModelResult) -> int:
    # nobs.survreg: length(fit$linear.predictors)
    return fit.n


@singledispatch
def degrees_freedom(fit: Any) -> float:
    """The ``df`` attribute of ``logLik``: the number of estimated coefficients
    (``sum(fit$df)`` for a penalized fit, the scales included for survreg)."""

    raise _no_method("degrees_freedom")


@degrees_freedom.register(CoxphModel)
def _degrees_freedom_cox(fit: CoxphModel) -> float:
    return _coxph_df(fit)


@degrees_freedom.register(SurvregModelResult)
def _degrees_freedom_survreg(fit: SurvregModelResult) -> float:
    # logLik.survreg: sum(object$df)
    return survreg_df(fit)


@singledispatch
def df_residual(fit: Any) -> float:
    """``df.residual`` (survreg only)."""

    raise _no_method("df_residual")


@df_residual.register(CoxphModel)
def _df_residual_cox(fit: CoxphModel) -> float:
    raise TypeError("df_residual is only defined for fitted survreg models")


@df_residual.register(SurvregModelResult)
def _df_residual_survreg(fit: SurvregModelResult) -> float:
    return fit.df_residual


def aic(fit: Any, *, k: Any = 2.0) -> float:
    """``AIC``: ``-2 logLik + k df``."""

    return -2.0 * loglik(fit) + float(k) * degrees_freedom(fit)


def bic(fit: Any) -> float:
    """``BIC``: ``AIC`` with ``k = log(nobs)``."""

    count = nobs(fit)
    return math.nan if count == 0 else aic(fit, k=math.log(count))


def extract_aic(fit: Any, *, scale: Any = 0.0, k: Any = 2.0) -> list[float]:
    """``extractAIC``: ``c(df, AIC)``."""

    del scale
    return [float(degrees_freedom(fit)), aic(fit, k=k)]


# ---------------------------------------------------------------------------
# formula, terms, weights, model matrix and frame
# ---------------------------------------------------------------------------

_FormulaFits = (
    CoxphModel
    | CchModelResult
    | AaregModelResult
    | SurvregModelResult
    | PyearsResult
    | SurvExpResult
)


@singledispatch
def model_formula(fit: Any) -> str:
    """``formula(fit)``."""

    raise _no_method("model_formula")


@model_formula.register(_FormulaFits)
def _model_formula_fit(fit: _FormulaFits) -> str:
    formula = fit.formula
    if formula is None:
        raise TypeError("model_formula requires a formula-based fitted model")
    return formula


@singledispatch
def model_term_names(fit: Any, terms: Any | None = None) -> list[str]:
    """``labels(fit)``: ``attr(terms(fit), 'term.labels')``, optionally the subset
    ``terms`` selects."""

    raise _no_method("model_term_names")


@model_term_names.register(CoxphModel)
def _model_term_names_cox(fit: CoxphModel, terms: Any | None = None) -> list[str]:
    names = _term_labels(fit)
    if fit.terms.model_terms:
        covariates = iter(names)
        ordered = sorted(
            (
                item
                for item in fit.terms.model_terms
                if isinstance(item, _ModelCovariateTerm | _ModelStrataTerm)
            ),
            key=_formula_model_term_degree,
        )
        names = [
            item.spec.call if isinstance(item, _ModelStrataTerm) else next(covariates)
            for item in ordered
        ]
    return [names[idx] for idx in _terms_selection(terms, names)]


@model_term_names.register(AaregModelResult | PyearsResult | SurvExpResult)
def _model_term_names_stored(
    fit: AaregModelResult | PyearsResult | SurvExpResult, terms: Any | None = None
) -> list[str]:
    # labels.aareg
    names = list(fit.term_labels)
    return [names[idx] for idx in _terms_selection(terms, names)]


model_term_names.register(SurvregModelResult, model_term_names_survreg)


@singledispatch
def model_weights(fit: Any) -> list[float] | None:
    """``weights(fit)``: the case weights, ``None`` when none were given."""

    raise _no_method("model_weights")


@model_weights.register(CoxphModel | AaregModelResult | SurvregModelResult)
def _model_weights_fit(
    fit: CoxphModel | AaregModelResult | SurvregModelResult,
) -> list[float] | None:
    return None if fit.weights is None else list(fit.weights)


@singledispatch
def model_matrix(fit: Any, data: Any | None = None) -> dict[str, Any]:
    """``model.matrix(fit)``: the design matrix, its column names and ``assign``."""

    raise _no_method("model_matrix")


@model_matrix.register(CoxphModel)
def _model_matrix_cox(fit: CoxphModel, data: Any | None = None) -> dict[str, Any]:
    """R's ``model.matrix.coxph``: the fit's design, or with ``data`` the design of
    those rows (``model.frame(Terms, data)``, whose default ``na.omit`` leaves out
    the incomplete ones).  ``assign`` numbers each column's term by its position in
    the model's term labels, which count ``strata()`` terms (the columns keep R's
    numbering "wrt the original model matrix") but not ``cluster()``; ``strata`` is
    ``attr(X, "strata")``, each row's stratum, ``None`` for an unstratified fit.

    A multi-state fit's design is the unstacked one, a column per covariate (NaN
    where a formula list left a covariate missing)."""

    names = list(fit.ms.x_names) if isinstance(fit, CoxphmsModel) else list(fit.coef_names)
    assign = [0] * len(names)
    for term_idx, columns in zip(fit.design.term_assignments, fit.assign.values(), strict=True):
        for col in columns:
            assign[col] = term_idx
    if data is None:
        rows, strata = fit.x, fit.strata
    else:
        new = _prediction_newdata(
            fit, data, need_strata=_has_strata(fit), need_response=False, na_action="na.omit"
        )
        rows, strata = new.x, None
        if _has_strata(fit):
            if new.strata is None:
                raise ValueError("data must contain the strata variable(s) of the model")
            strata = [fit.strata_levels[code] for code in new.strata]
    return {"data": rows, "columns": names, "assign": assign, "strata": strata}


model_matrix.register(SurvregModelResult, model_matrix_survreg)


@singledispatch
def model_frame(formula: Any, data: Any | None = None, **kwargs: Any) -> dict[str, list[Any]]:
    """``model.frame``: the model frame of a formula string and ``data`` (see
    :func:`survival.r._formula.model_frame` for the arguments; R's ``na.action`` spelling
    is accepted) or of a fitted model, as columns: a ``Surv`` response split into
    ``time``/``status`` (``start``/``stop``/``status`` for counting data), then the
    formula's variables and the ``(weights)``, ``(id)``, ... arguments.

    A Cox model's frame is rebuilt when the fit did not keep it; the other fits need
    ``model=TRUE``.
    """

    raise TypeError("model_frame requires a formula or a fitted model")


@model_frame.register(str)
def _model_frame_formula(
    formula: str, data: Any | None = None, **kwargs: Any
) -> dict[str, list[Any]]:
    if "na.action" in kwargs:
        kwargs["na_action"] = kwargs.pop("na.action")
    # a Surv2 response has no time/status columns here, so survSplit's timeline
    # switch stays internal
    frame = _formula_model_frame(formula, data, **kwargs, timeline=False)
    columns: dict[str, Any] = {}
    response_columns: tuple[str, ...] = ()
    if frame.response is not None:
        columns[frame.response_name or "response"] = frame.response
        response_columns = frame.response_columns
    elif isinstance(frame.y, np.ndarray):
        columns[frame.response_name or "response"] = frame.y.tolist()
        response_columns = frame.response_columns
    for name in _formula_columns(formula, frame.data):
        if name not in response_columns:
            columns[name] = _formula_column(frame.data, name)
    for name in ("weights", "offset", "id", "cluster", "istate"):
        values = getattr(frame, name)
        if values is not None:
            columns[f"({name})"] = values
    columns.update(frame.extra)
    return _plain_model_frame(columns)


@model_frame.register(CoxphModel)
def _model_frame_cox(fit: CoxphModel) -> dict[str, list[Any]]:
    return _plain_model_frame(_coxph_model_frame(fit))


@model_frame.register(
    AaregModelResult
    | SurvregModelResult
    | SurvfitResult
    | SurvfitMultiStateResult
    | PyearsResult
    | SurvExpResult
)
def _model_frame_stored(
    fit: AaregModelResult
    | SurvregModelResult
    | SurvfitResult
    | SurvfitMultiStateResult
    | PyearsResult
    | SurvExpResult,
) -> dict[str, list[Any]]:
    if fit.model is None:
        raise TypeError("model_frame requires a fit made with model=TRUE")
    return _plain_model_frame(fit.model)


@model_frame.register(Mapping)
def _model_frame_grouped(fit: Mapping[Any, Any]) -> dict[str, list[Any]]:
    # the bridge's grouped survfit curves share the model frame of the call
    if not fit:
        raise TypeError("model_frame requires a non-empty grouped survfit result")
    return model_frame(next(iter(fit.values())))


def _surv_columns(response: Surv, existing: set[str]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    if response.start is not None:
        if "start" not in existing:
            columns["start"] = list(response.start)
        if "stop" not in existing:
            columns["stop"] = list(response.time)
    elif "time" not in existing:
        columns["time"] = list(response.time)
    if response.time2 is not None and "time2" not in existing:
        columns["time2"] = list(response.time2)
    if "status" not in existing:
        columns["status"] = list(response.event)
    return columns


def _plain_model_frame(frame: Mapping[str, Any]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    for name, values in frame.items():
        if isinstance(values, Surv):
            columns.update(_surv_columns(values, set(columns)))
            continue
        if isinstance(values, Mapping):
            continue
        text_name = str(name)
        materialized = _coerce_array_like(values, text_name)
        if text_name in {"group", "(id)", "(cluster)", "(strata)"} and (
            not materialized or not isinstance(materialized[0], list)
        ):
            columns[text_name] = _materialize_labels(values, text_name)
            continue
        if materialized and isinstance(materialized[0], list | tuple):
            columns[text_name] = [list(row) for row in materialized]
            continue
        columns[text_name] = list(materialized)
    return columns


# ---------------------------------------------------------------------------
# predict / residuals / summaries
# ---------------------------------------------------------------------------


def predict(fit: Any, newdata: Any | None = None, **kwargs: Any) -> Any:
    """``predict``: see :func:`survival.r._coxph.predict_coxph` and
    :func:`survival.r._survreg.predict_survreg`.  R's ``se.fit`` and ``na.action``
    spellings are accepted. For a ``PsplineResult``, evaluate its basis on
    ``newdata`` (or R's ``newx``); omitting both returns the original basis."""

    for dotted, name in (("se.fit", "se_fit"), ("na.action", "na_action")):
        if dotted in kwargs:
            kwargs[name] = kwargs.pop(dotted)
    return _predict(fit, newdata, **kwargs)


@singledispatch
def _predict(fit: Any, newdata: Any | None = None, **kwargs: Any) -> Any:
    raise TypeError("predict requires a fitted coxph or survreg model, or a PsplineResult")


_predict.register(CoxphModel, predict_coxph)
_predict.register(SurvregModelResult, predict_survreg)


@singledispatch
def fitted(fit: Any, **kwargs: Any) -> Any:
    """``fitted``: ``predict`` on the training data; for a Cox model R's
    ``fitted.coxph``, the fit's linear predictors."""

    return predict(fit, None, **kwargs)


@fitted.register(CoxphModel)
def _fitted_cox(fit: CoxphModel, **_kwargs: Any) -> list[float]:
    # fitted.coxph(object, ...) is object$linear.predictors: centred at the overall
    # means, not padded by naresid, other arguments ignored
    return fit.linear_predictors


@singledispatch
def residuals(fit: Any, **kwargs: Any) -> Any:
    """``residuals``: see :func:`survival.r._coxph.residuals_coxph`,
    :func:`survival.r._survreg.residuals_survreg` and
    :func:`survival.r._survfit_residuals.survfit_residuals`; without ``type`` each
    method uses its own default (martingale for Cox models, response for survreg,
    pstate for survival curves)."""

    raise _no_method("residuals")


residuals.register(CoxphModel, residuals_coxph)
residuals.register(SurvregModelResult, residuals_survreg)
# residuals.survfit; a survfit.coxph curve gets R's "method not found" error from it
residuals.register(_SurvfitCurves, survfit_residuals)


@residuals.register(CoxSurvfitMultiStateResult)
def _residuals_coxms(fit: CoxSurvfitMultiStateResult, **kwargs: Any) -> Any:
    # R's residuals.survfit returns the Aalen-Johansen residuals of the data, which
    # ignore the Cox model and the newdata
    raise TypeError("residuals are not defined for multi-state Cox curves")


@singledispatch
def confint(
    fit: Any, parm: Any | None = None, *, level: Any = 0.95
) -> list[dict[str, float | str]]:
    """``confint``: normal-approximation intervals for the coefficients."""

    raise _no_method("confint")


@confint.register(CoxphModel | CchModelResult)
def _confint_cox(
    fit: CoxphModel | CchModelResult, parm: Any | None = None, *, level: Any = 0.95
) -> list[dict[str, float | str]]:
    z = NormalDist().inv_cdf(1.0 - (1.0 - _normalize_conf_level(level, "level")) / 2.0)
    names = coef_names(fit)
    coefficients = coef(fit)
    variance = vcov(fit)
    return [
        {
            "name": names[idx],
            "lower": coefficients[idx] - z * math.sqrt(variance[idx][idx]),
            "upper": coefficients[idx] + z * math.sqrt(variance[idx][idx]),
        }
        for idx in _coefficient_selection(parm, names)
    ]


confint.register(SurvregModelResult, confint_survreg)


@singledispatch
def model_summary(fit: Any, **kwargs: Any) -> Any:
    """``summary``: R's summary of a coxph, clogit, cch, aareg or survreg fit, a survival
    curve, an expected-survival result, a population rate/person-years table,
    or merged event data."""

    raise TypeError(
        "model_summary requires a fitted model, survival curve, population result, or TMergeFrame"
    )


model_summary.register(CoxphModel, summary_coxph)
model_summary.register(AaregModelResult, summary_aareg)
model_summary.register(SurvregModelResult, model_summary_survreg)
model_summary.register(_SurvfitCurves | CoxSurvfitMultiStateResult, summary_survfit)
model_summary.register(PyearsResult, summary_pyears)
model_summary.register(SurvExpResult, summary_survexp)
model_summary.register(_core.RateTable, summary_ratetable)
model_summary.register(TMergeFrame, summary_tmerge)


@model_summary.register(CchModelResult)
def _model_summary_cch(fit: CchModelResult, **kwargs: Any) -> dict[str, Any]:
    # summary.cch(object, ...): the further arguments are ignored
    return summary_cch(fit)


# ---------------------------------------------------------------------------
# as_data_frame
# ---------------------------------------------------------------------------


def _add_optional_survfit_column(
    frame: dict[str, list[Any]],
    name: str,
    values: Sequence[Any],
    row_count: int,
) -> None:
    column = list(values)
    if not column:
        return
    if len(column) != row_count:
        raise ValueError(f"survfit column {name!r} must match time length")
    frame[name] = column


def _survfit_frame(result: SurvfitResult) -> dict[str, list[Any]]:
    """``survfit`` as R's ``summary(fit, censored = TRUE, data.frame = TRUE)``: one row
    per time with ``std.err`` on the survival scale, plus a ``strata`` column."""

    row_count = len(result.time)
    frame: dict[str, list[Any]] = {
        "time": [float(value) for value in result.time],
        "n.risk": [float(value) for value in result.n_risk],
        "n.event": [float(value) for value in result.n_event],
        "n.censor": [float(value) for value in result.n_censor],
        "surv": [float(value) for value in result.surv],
        "cumhaz": [float(value) for value in result.cumhaz],
    }
    std_err = result.std_err
    if std_err is not None and result.logse:
        std_err = [float(se) * float(surv) for se, surv in zip(std_err, result.surv, strict=True)]
    # once a curve reaches 0 its standard error and limits are undefined
    terminal = [value <= 0.0 for value in frame["surv"]]
    for name, values in (
        ("std.err", std_err),
        ("lower", result.lower),
        ("upper", result.upper),
    ):
        if values is not None:
            column = [
                math.nan if done else float(value)
                for value, done in zip(values, terminal, strict=True)
            ]
            _add_optional_survfit_column(frame, name, column, row_count)
    if result.std_chaz is not None:
        _add_optional_survfit_column(frame, "std.chaz", result.std_chaz, row_count)
    if result.n_enter is not None:
        frame["n.enter"] = [float(value) for value in result.n_enter]
    if result.strata:
        frame["strata"] = [name for name, count in result.strata.items() for _ in range(count)]
    return frame


def _matrix_column(values: Sequence[Sequence[Any]], index: int) -> list[float]:
    return [float(row[index]) for row in values]


def _survfit_multistate_frame(result: SurvfitMultiStateResult) -> dict[str, list[Any]]:
    """``survfitms`` state by state: the ``time x state`` matrices as long columns."""

    row_count = len(result.time)
    columns: dict[str, Sequence[Sequence[Any]] | None] = {
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "pstate": result.pstate,
        "std.err": result.std_err,
        "lower": result.lower,
        "upper": result.upper,
    }
    present = [name for name, values in columns.items() if values is not None]
    frame: dict[str, list[Any]] = {"time": [], **{name: [] for name in present}}
    if result.strata:
        frame["strata"] = []
    frame["state"] = []
    for state_index, state in enumerate(result.states):
        frame["time"].extend(float(value) for value in result.time)
        for name in present:
            frame[name].extend(_matrix_column(columns[name] or (), state_index))
        if result.strata:
            frame["strata"].extend(
                name for name, count in result.strata.items() for _ in range(count)
            )
        frame["state"].extend([state] * row_count)
    return frame


def _summary_coxms_frame(summary: SummarySurvfitCoxmsResult) -> dict[str, list[Any]]:
    """``summary(fit, data.frame = TRUE)`` of multi-state Cox curves
    (summary.survfitms.R): the (time, stratum) rows vary fastest, then the newdata
    rows, then the states; the counts repeat for every newdata row."""

    nt, nd, ns = summary.pstate.shape
    per_state = nd * nt

    def counts(values: list[list[float]]) -> list[float]:
        return [float(values[t][s]) for s in range(ns) for _ in range(nd) for t in range(nt)]

    frame: dict[str, list[Any]] = {
        "time": list(summary.time) * (nd * ns),
        "n.risk": counts(summary.n_risk),
        "n.event": counts(summary.n_event),
        "n.censor": counts(summary.n_censor),
        "pstate": summary.pstate.ravel(order="F").tolist(),
    }
    if summary.strata is not None:
        frame["strata"] = list(summary.strata) * (nd * ns)
    frame["state"] = [state for state in summary.states for _ in range(per_state)]
    for name, values in (summary.newdata or {}).items():
        frame[name] = [values[i] for _ in range(ns) for i in range(nd) for _ in range(nt)]
    return frame


def _coxms_curves_frame(result: CoxSurvfitMultiStateResult) -> dict[str, list[Any]]:
    """Multi-state Cox curves as R's ``summary(fit, censored = TRUE, data.frame = TRUE)``."""

    return _summary_coxms_frame(summary_survfit(result, censored=True))


def _positions(selection: Any, count: int, labels: Sequence[str], name: str) -> list[int]:
    """0-based positions among ``count`` items from indices or, when the items have
    ``labels``, labels."""

    positions = []
    for value in _materialize_1d(selection, name):
        if isinstance(value, str):
            if value not in labels:
                raise ValueError(f"{name} {value!r} is not one of {', '.join(labels)}")
            positions.append(labels.index(value))
        else:
            position = _integer_scalar(value, name)
            if not 0 <= position < count:
                raise IndexError(f"{name} index {position} is out of bounds")
            positions.append(position)
    if not positions:
        raise ValueError(f"select at least one {name} value")
    return positions


def _subset_coxms_curves(
    result: CoxSurvfitMultiStateResult,
    strata: Any | None = None,
    data: Any | None = None,
    states: Any | None = None,
) -> CoxSurvfitMultiStateResult:
    """``fit[strata, data, states]`` of multi-state Cox curves (``[.survfitms``):
    each argument ``None`` (keep all) or 0-based indices, stratum labels or state names.

    Strata select their time rows and their ``n``, ``n_id`` and ``p0`` rows; ``data``
    the newdata rows.  A state subset keeps those columns of ``pstate``, ``n_risk``,
    ``n_event``, ``n_censor`` and ``p0``, drops ``cumhaz`` and ``n_transition`` and
    records ``oldstate``. The engine retains the same selected counts for summary
    and initial-row operations. ``transitions`` is always dropped. Unlike R,
    ``n_id`` follows the selected strata and censor counts follow the states.
    """

    names = result.strata_names
    sizes = list(result.strata.values()) if result.strata else [len(result.time)]
    starts = np.cumsum([0, *sizes])
    if strata is None:
        kept_strata = list(range(len(sizes)))
    elif not names:
        raise ValueError("the curves have no strata to select")
    else:
        kept_strata = _positions(strata, len(names), names, "strata")
    ndata = result.pstate.shape[1]
    kept_data = list(range(ndata)) if data is None else _positions(data, ndata, (), "data")
    nstate = len(result.states)
    kept_states = (
        list(range(nstate))
        if states is None
        else _positions(states, nstate, result.states, "states")
    )
    every_stratum = kept_strata == list(range(len(sizes)))
    every_state = kept_states == list(range(nstate))
    rows = [row for s in kept_strata for row in range(starts[s], starts[s + 1])]

    def pick(values: list[list[float]], columns: list[int] | None = None) -> list[list[float]]:
        if columns is None:
            return [list(values[row]) for row in rows]
        return [[values[row][c] for c in columns] for row in rows]

    state_columns = None if every_state else kept_states
    engine = result.engine
    if engine is not None and not every_stratum:
        engine = engine.select_curves(kept_strata)
    if engine is not None and not every_state:
        engine = engine.select_states(kept_states)
    cumhaz = None
    if every_state and result.cumhaz is not None:
        cumhaz = result.cumhaz[np.ix_(rows, kept_data, range(result.cumhaz.shape[2]))]
    return dataclasses.replace(
        result,
        n=[result.n[s] for s in kept_strata],
        n_id=[result.n_id[s] for s in kept_strata],
        time=[result.time[row] for row in rows],
        n_risk=pick(result.n_risk, state_columns),
        n_event=pick(result.n_event, state_columns),
        n_censor=pick(result.n_censor, state_columns),
        n_transition=None
        if not every_state or result.n_transition is None
        else pick(result.n_transition),
        pstate=result.pstate[np.ix_(rows, kept_data, kept_states)],
        cumhaz=cumhaz,
        p0=[[result.p0[s][c] for c in kept_states] for s in kept_strata],
        states=[result.states[c] for c in kept_states],
        oldstate=result.oldstate if every_state else tuple(result.oldstate or result.states),
        transitions=None,
        strata=None
        if not names
        else dict(
            zip(
                _make_unique([names[s] for s in kept_strata]),
                [sizes[s] for s in kept_strata],
                strict=True,
            )
        ),
        newdata=None
        if result.newdata is None
        else {name: [values[i] for i in kept_data] for name, values in result.newdata.items()},
        engine=engine,
    )


# --- the R bridge's grouped view of a stratified curve set --------------------------------


def _bare_strata_label(name: str) -> str:
    """``rx=1`` -> ``1`` and ``a=1, b=x`` -> ``1, x`` (the labels the bridge names curves by)."""

    return re.sub(r"(^|, )[^=,]+=", r"\1", name)


def _survfit_stratum(
    result: SurvfitResult | SurvfitMultiStateResult, index: int
) -> SurvfitResult | SurvfitMultiStateResult:
    """Curve ``index`` of ``result`` as its own unstratified result."""

    return _derived_survfit(result, _engine_of(result).select_curves([index]), time0=result.time0)


def _survfit_strata_curves(result: Any) -> Any:
    """The bridge's shape for a stratified ``survfit``: ``{bare level label: curve}``.

    ``survfit.formula`` results lay their strata end to end with ``strata`` naming the
    blocks (R's layout); the R bridge represents them as a named list of per-stratum
    curves keyed by the bare levels.  Unstratified results are returned as they are.
    """

    if not isinstance(result, SurvfitResult | SurvfitMultiStateResult) or not result.strata:
        return result
    return {
        _bare_strata_label(name): _survfit_stratum(result, index)
        for index, name in enumerate(result.strata)
    }


def _subset_survfit_multistate(
    result: SurvfitMultiStateResult,
    state_indices: Any,
    keep_n_id: bool | None = None,
) -> SurvfitMultiStateResult:
    """``fit[, states]`` (``[.survfitms``): keep the selected states (0-based indices).

    The transition parts go unless every state is kept in its original order, which is
    also when R records no ``oldstate``.  R keeps ``n.id`` for a stratified object and drops
    it otherwise (``[.survfitms`` reads ``x$id`` there); ``keep_n_id`` overrides that for
    the bridge's split curves.
    """

    if not isinstance(result, SurvfitMultiStateResult):
        raise TypeError("multi-state survfit subsetting requires a multi-state result")
    indices = [
        _integer_scalar(value, "state_indices")
        for value in _materialize_1d(state_indices, "state_indices")
    ]
    if not indices:
        raise ValueError("multi-state survfit subsetting must select at least one state")
    if any(index < 0 or index >= len(result.states) for index in indices):
        raise IndexError("multi-state survfit state index is out of bounds")
    subset = _derived_survfit(result, _engine_of(result).select_states(indices), time0=result.time0)
    if keep_n_id is None:
        keep_n_id = bool(result.strata)
    every_state = indices == list(range(len(result.states)))
    return dataclasses.replace(
        subset,
        n_id=subset.n_id if keep_n_id else None,
        oldstate=None if every_state else tuple(result.states),
    )


def _survfit_multistate_structure(
    result: SurvfitMultiStateResult | Mapping[Any, Any],
) -> dict[str, Any]:
    """R's ``survfitms`` list for the bridge: one result, or the bridge's grouped curves."""

    if isinstance(result, SurvfitMultiStateResult):
        curves: list[tuple[Any, SurvfitMultiStateResult]] = [(None, result)]
    elif (
        isinstance(result, Mapping)
        and result
        and all(isinstance(curve, SurvfitMultiStateResult) for curve in result.values())
    ):
        curves = list(result.items())
    else:
        raise TypeError("survfit structure requires a multi-state result")
    grouped = curves[0][0] is not None
    first = curves[0][1]
    for _label, curve in curves:
        if curve.states != first.states:
            raise ValueError("grouped multi-state results must share state columns")
        if curve.hazard_names != first.hazard_names:
            raise ValueError("grouped multi-state results must share transition columns")

    def combined_matrix(name: str) -> list[list[float]] | None:
        matrices = [getattr(curve, name) for _label, curve in curves]
        if all(matrix is None for matrix in matrices):
            return None
        if any(matrix is None for matrix in matrices):
            raise ValueError(f"grouped multi-state results must share {name} output")
        return [[float(value) for value in row] for matrix in matrices for row in matrix]

    def flat(name: str) -> list[Any]:
        return [value for _label, curve in curves for value in getattr(curve, name)]

    # R's survfitms component order
    structure: dict[str, Any] = {
        "n": flat("n"),
        "time": [float(value) for value in flat("time")],
        "n.risk": combined_matrix("n_risk"),
        "n.event": combined_matrix("n_event"),
        "n.censor": combined_matrix("n_censor"),
        "pstate": combined_matrix("pstate"),
    }
    if first.hazard_names:
        structure["n.transition"] = combined_matrix("n_transition")
    if first.n_id is not None:
        structure["n.id"] = flat("n_id")
    if first.hazard_names:
        structure["cumhaz"] = combined_matrix("cumhaz")
    n_enter = combined_matrix("n_enter")
    if n_enter is not None:
        structure["n.enter"] = n_enter
    p0_rows = []
    for _label, curve in curves:
        # A homogeneous p0 is either one state vector or one vector per stratum.
        initial = (
            curve.p0[0]
            if curve.p0 and isinstance(curve.p0[0], list | tuple)
            else cast(list[float], curve.p0)
        )
        p0_rows.append([float(value) for value in initial])
    structure["p0"] = p0_rows if grouped else p0_rows[0]
    if grouped:
        structure["strata"] = {str(label): len(curve.time) for label, curve in curves}
    elif first.strata:
        structure["strata"] = dict(first.strata)
        structure["p0"] = [[float(value) for value in row] for row in first.p0]
    for field_name, attribute in (
        ("std.err", "std_err"),
        ("std.chaz", "std_chaz"),
        ("std.auc", "std_auc"),
    ):
        values = combined_matrix(attribute)
        if values is not None and (field_name != "std.chaz" or first.hazard_names):
            structure[field_name] = values
    structure["logse"] = bool(first.logse)
    if first.transitions is not None:
        transition_names = first.transitions.rownames
        if transition_names is None:
            raise ValueError("multi-state transition counts require row names")
        totals = [[0.0] * len(first.transitions.colnames) for _row in transition_names]
        for _label, curve in curves:
            if curve.transitions is None:
                continue
            transition_rows: Sequence[Sequence[float]] = curve.transitions.values
            for row_index, row in enumerate(transition_rows):
                for col_index, value in enumerate(row):
                    totals[row_index][col_index] += float(value)
        structure["transitions"] = {
            "values": totals,
            "rows": list(transition_names),
            "columns": list(first.transitions.colnames),
        }
    for field_name, attribute in (("lower", "lower"), ("upper", "upper")):
        values = combined_matrix(attribute)
        if values is not None:
            structure[field_name] = values
    structure.update(
        {
            "conf.type": first.conf_type,
            "conf.int": first.conf_int,
            "states": list(first.states),
            "type": first.type,
            "t0": first.t0,
            "_transition_names": list(first.hazard_names),
        }
    )
    if first.oldstate is not None:
        structure["oldstate"] = list(first.oldstate)
    return structure


def _grouped_survfit_frame(result: Mapping[Any, Any]) -> dict[str, list[Any]]:
    """The bridge's grouped curves as one table with a ``strata`` column (state-major
    for multi-state curves, as R's ``summary(fit, data.frame = TRUE)`` lays them out)."""

    if result and all(isinstance(curve, SurvfitMultiStateResult) for curve in result.values()):
        curve_frames = {label: _survfit_multistate_frame(curve) for label, curve in result.items()}
        columns = list(next(iter(curve_frames.values())))
        states = next(iter(result.values())).states
        if any(curve.states != states for curve in result.values()):
            raise ValueError("grouped multi-state results must share state columns")
        frame: dict[str, list[Any]] = {
            name: [] for name in [*[name for name in columns if name != "state"], "strata", "state"]
        }
        for state in states:
            for label, curve_frame in curve_frames.items():
                if list(curve_frame) != columns:
                    raise ValueError("grouped multi-state results must share tabular columns")
                indices = [
                    index for index, value in enumerate(curve_frame["state"]) if value == state
                ]
                frame["strata"].extend([str(label)] * len(indices))
                frame["state"].extend([state] * len(indices))
                for name in columns:
                    if name != "state":
                        frame[name].extend(curve_frame[name][index] for index in indices)
        return frame

    frame = {}
    for label, curve in result.items():
        curve_frame = as_data_frame(curve)
        if not curve_frame:
            continue
        n_rows = len(next(iter(curve_frame.values())))
        if not frame:
            frame = {"strata": []}
            for name in curve_frame:
                frame[name] = []
        elif set(curve_frame) != set(frame) - {"strata"}:
            raise ValueError("grouped survfit results must share tabular columns")
        frame["strata"].extend([str(label)] * n_rows)
        for name, values in curve_frame.items():
            frame[name].extend(values)
    return frame


def _cox_basehaz_frame(result: CoxBaseHazardResult) -> dict[str, list[Any]]:
    hazard = result.hazard
    frame: dict[str, list[Any]] = {}
    if hazard and isinstance(hazard[0], list):
        for col in range(len(hazard[0])):
            frame[f"hazard.{col + 1}"] = [row[col] for row in hazard]
    else:
        frame["hazard"] = list(hazard)
    frame["time"] = list(result.time)
    if result.strata is not None:
        frame["strata"] = list(result.strata)
    return frame


def _cox_survfit_frame(result: CoxSurvfitResult) -> dict[str, list[Any]]:
    """``summary(survfit)``-style columns: one row per (curve, time)."""

    ncurve = result.ncurve
    ntime = len(result.time)

    def column(values: Any, curve: int) -> list[float]:
        # a one-column matrix (aggregate()'s result) is still a matrix
        if values and isinstance(values[0], list):
            return [float(row[curve]) for row in values]
        return [float(value) for value in values]

    frame: dict[str, list[Any]] = {
        name: [] for name in ("curve", "time", "n.risk", "n.event", "n.censor", "surv")
    }
    if result.strata is not None:
        frame["strata"] = []
    optional = {
        # aggregate() leaves the cumulative hazard out, as R does
        "cumhaz": result.cumhaz or None,
        "std.err": result.std_err,
        "std.chaz": result.std_chaz,
        "lower": result.lower,
        "upper": result.upper,
    }
    for name, values in optional.items():
        if values is not None:
            frame[name] = []
    strata = [name for name, count in (result.strata or {}).items() for _ in range(count)]
    for curve in range(ncurve):
        frame["curve"].extend([curve + 1] * ntime)
        frame["time"].extend(result.time)
        frame["n.risk"].extend(result.n_risk)
        frame["n.event"].extend(result.n_event)
        frame["n.censor"].extend(result.n_censor)
        frame["surv"].extend(column(result.surv, curve))
        if result.strata is not None:
            frame["strata"].extend(strata)
        for name, values in optional.items():
            if values is not None:
                frame[name].extend(column(values, curve))
    return frame


def _survdiff_frame(result: SurvDiffResult) -> dict[str, list[Any]]:
    """One row per group: the observed and expected counts (summed over strata) and
    the diagonal of the variance."""

    def totals(values: Sequence[Any]) -> list[float]:
        return [float(sum(row)) if isinstance(row, list | tuple) else float(row) for row in values]

    observed = totals(result.obs)
    expected = totals(result.exp)
    variance = result.var
    if len(variance) == len(observed):
        variance_diag = [float(row[idx]) for idx, row in enumerate(variance)]
    else:
        variance_diag = [math.nan] * len(observed)
    groups = (
        list(result.groups) if result.groups else [str(idx + 1) for idx in range(len(observed))]
    )
    return {
        "group": groups,
        "observed": observed,
        "expected": expected,
        "variance": variance_diag,
    }


def _cox_zph_frame(result: CoxZPHResult) -> dict[str, list[Any]]:
    return {
        "name": [str(row["name"]) for row in result.table],
        "chisq": [float(row["chisq"]) for row in result.table],
        "df": [float(row["df"]) for row in result.table],
        "p": [float(row["p"]) for row in result.table],
    }


def _coxph_detail_frame(result: CoxPHDetailResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "nevent": result.nevent,
        "nrisk": result.nrisk,
        "hazard": result.hazard,
        "varhaz": result.varhaz,
        "cumhaz": result.cumhaz,
        "wtrisk": result.wtrisk,
    }
    if result.nevent_wt is not None:
        frame["nevent.wt"] = result.nevent_wt
    if result.nrisk_wt is not None:
        frame["nrisk.wt"] = result.nrisk_wt
    if result.strata is not None:
        frame["strata"] = [name for name, count in result.strata.items() for _ in range(count)]
    return frame


def _anova_frame(result: _core.AnovaCoxphResult) -> dict[str, list[Any]]:
    rows = list(result.rows)
    return {
        "model": [str(row.name) for row in rows],
        "loglik": [float(row.loglik) for row in rows],
        "Chisq": [math.nan if row.chisq is None else float(row.chisq) for row in rows],
        "Df": [math.nan if row.df is None else int(row.df) for row in rows],
        "Pr(>|Chi|)": [math.nan if row.p_value is None else float(row.p_value) for row in rows],
    }


def _concordance_frame(result: ConcordanceResult) -> dict[str, list[Any]]:
    rows = result.count if isinstance(result.count, list) else [result.count]
    names = result.names or [f"X{idx + 1}" for idx in range(len(rows))]
    if isinstance(result.concordance, list):
        concordance = list(result.concordance)
        var = (
            [result.var[idx][idx] for idx in range(len(result.var))]
            if isinstance(result.var, list)
            else [math.nan] * len(rows)
        )
    else:
        concordance = [result.concordance] * len(rows)
        if isinstance(result.var, list):
            raise ValueError("a scalar concordance requires a scalar variance")
        var = [math.nan if result.var is None else float(result.var)] * len(rows)
    frame: dict[str, list[Any]] = {
        "score": list(names[: len(rows)]),
        "concordance": concordance[: len(rows)],
    }
    for name in ("concordant", "discordant", "tied.x", "tied.y", "tied.xy"):
        frame[name] = [row[name] for row in rows]
    frame["n"] = [result.n] * len(rows)
    frame["var"] = var[: len(rows)]
    return frame


def _surv_response_frame(response: Surv) -> dict[str, list[Any]]:
    if response.start is not None:
        frame: dict[str, list[Any]] = {
            "start": list(response.start),
            "stop": list(response.time),
            "status": list(response.event),
        }
    else:
        frame = {"time": list(response.time), "status": list(response.event)}
        if response.time2 is not None:
            frame["time2"] = list(response.time2)
    frame["type"] = [response.type] * len(response)
    return frame


@singledispatch
def as_data_frame(result: Any) -> dict[str, list[Any]]:
    """``as.data.frame``: a result object as a plain column-oriented table."""

    raise TypeError("as_data_frame requires a survival result object")


as_data_frame.register(Surv, _surv_response_frame)
as_data_frame.register(CoxSurvfitResult, _cox_survfit_frame)
as_data_frame.register(CoxBaseHazardResult, _cox_basehaz_frame)
as_data_frame.register(SurvfitMultiStateResult, _survfit_multistate_frame)
as_data_frame.register(CoxSurvfitMultiStateResult, _coxms_curves_frame)
as_data_frame.register(SummarySurvfitCoxmsResult, _summary_coxms_frame)
as_data_frame.register(SurvfitResult, _survfit_frame)
as_data_frame.register(CoxZPHResult, _cox_zph_frame)
as_data_frame.register(CoxPHDetailResult, _coxph_detail_frame)
as_data_frame.register(ConcordanceResult, _concordance_frame)
as_data_frame.register(PyearsResult, _pyears_result_frame)
as_data_frame.register(SurvExpResult | SurvExpSummary, _survexp_frame)
as_data_frame.register(_core.FineGrayOutput, _finegray_frame)
as_data_frame.register(SurvDiffResult, _survdiff_frame)
as_data_frame.register(_core.AnovaCoxphResult, _anova_frame)


@as_data_frame.register(SurvfitPrint)
def _survfit_print_frame(result: SurvfitPrint) -> dict[str, list[Any]]:
    table = result.table
    frame: dict[str, list[Any]] = {}
    if table.rownames is not None:
        frame["curve"] = list(table.rownames)
    frame.update((name, [row[j] for row in table.values]) for j, name in enumerate(table.colnames))
    return frame


@as_data_frame.register(ModelPrint)
def _model_print_frame(result: ModelPrint) -> dict[str, list[Any]]:
    table = result.tables.get(result.primary_table)
    if table is None:
        return {}
    frame: dict[str, list[Any]] = {}
    if table.rownames is not None:
        frame[result.row_label] = list(table.rownames)
    frame.update((name, [row[j] for row in table.values]) for j, name in enumerate(table.colnames))
    return frame


@as_data_frame.register(YatesPrint)
def _yates_print_frame(result: YatesPrint) -> dict[str, list[Any]]:
    return {name: list(values) for name, values in result.estimates.items()}


@as_data_frame.register(ResponsePrint)
def _response_print_frame(result: ResponsePrint) -> dict[str, list[Any]]:
    return {name: list(values) for name, values in result.data.items()}


@as_data_frame.register(RateTablePrint)
def _ratetable_print_frame(result: RateTablePrint) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {}
    stride = 1
    names = _make_unique([*result.dimid, "rate"])
    for name, extent, levels in zip(names[:-1], result.dims, result.dimnames, strict=True):
        frame[name] = [levels[(i // stride) % extent] for i in range(len(result.rates))]
        stride *= extent
    frame[names[-1]] = list(result.rates)
    return frame


@as_data_frame.register(RateTableMatch)
def _ratetable_match_frame(result: RateTableMatch) -> dict[str, list[Any]]:
    return {name: [row[j] for row in result.r] for j, name in enumerate(result.dimid)}


@as_data_frame.register(SurvregPenalPrint)
def _survreg_penal_print_frame(result: SurvregPenalPrint) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {"term": list(result.rownames)}
    frame.update((name, [row[j] for row in result.rows]) for j, name in enumerate(result.columns))
    return frame


@as_data_frame.register(SurvivalTablePrint)
def _survival_table_print_frame(result: SurvivalTablePrint) -> dict[str, list[Any]]:
    if not result.tables:
        return {}
    columns = result.tables[0].colnames
    if any(table.colnames != columns for table in result.tables):
        raise ValueError("report tables must have matching columns")
    frame: dict[str, list[Any]] = {name: [] for name in columns}
    if any(group is not None for group in result.groups):
        frame["strata"] = []
    for table, group in zip(result.tables, result.groups, strict=True):
        for j, name in enumerate(columns):
            frame[name].extend(row[j] for row in table.values)
        if "strata" in frame:
            frame["strata"].extend([group] * len(table.values))
    return frame


@as_data_frame.register(RateTableSummary)
def _ratetable_summary_frame(result: RateTableSummary) -> dict[str, list[Any]]:
    return {name: list(values) for name, values in result.dimensions.items()}


@as_data_frame.register(_core.RateTable)
def _ratetable_frame(result: Any) -> dict[str, list[Any]]:
    return _ratetable_summary_frame(summary_ratetable(result))


@as_data_frame.register(SurvregAnovaResult)
def _survreg_anova_frame(result: SurvregAnovaResult) -> dict[str, list[Any]]:
    return result.frame()


@as_data_frame.register(Mapping)
def _mapping_frame(result: Mapping[Any, Any]) -> dict[str, list[Any]]:
    # the bridge's grouped survfit curves, or a data frame held as columns (tmerge,
    # survSplit, finegray)
    if result and all(isinstance(curve, _SurvfitCurves) for curve in result.values()):
        return _grouped_survfit_frame(result)
    columns: dict[str, list[Any]] = {}
    for name, values in result.items():
        if isinstance(values, str | bytes | Mapping | _SurvfitCurves) or not hasattr(
            values, "__iter__"
        ):
            raise TypeError("as_data_frame requires a survival result object")
        columns[str(name)] = list(values)
    return columns
