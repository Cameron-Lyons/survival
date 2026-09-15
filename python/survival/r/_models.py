"""Model generics (``coef``, ``vcov``, ``predict``, ``residuals``, ``model_summary``,
``as_data_frame``, ...): R-style S3 dispatch on the fitted object.

Cox, clogit, cch and aareg fits dispatch here; survreg fits go to the matching
``*_survreg`` function of :mod:`survival.r._survreg` (``predict_survreg``,
``residuals_survreg``, ``summary_survreg``, ...).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

from .. import _survival as _core
from ._aareg import summary_aareg
from ._cch import summary_cch
from ._coerce import (
    _integer_scalar,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option_with_default,
    _normalize_conf_level,
)
from ._coxph import CoxphModel, predict_coxph, residuals_coxph, summary_coxph
from ._coxph import predict_terms_constant as predict_terms_constant  # re-exported by survival.r
from ._pyears import _finegray_frame, _pyears_result_frame
from ._surv import Surv
from ._types import (
    AaregModelResult,
    CchModelResult,
    ConcordanceResult,
    CoxBaseHazardResult,
    CoxPHDetailResult,
    CoxSurvfitResult,
    CoxZPHResult,
    FineGrayFrame,
    FineGrayOutput,
    PyearsResult,
    SurvDiffResult,
    SurvfitMultiStateResult,
    SurvfitResult,
)

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def _survreg_method(generic: str) -> Any:
    """``<generic>.survreg``: the survreg method of a generic, from ``_survreg``."""

    from . import _survreg

    method = getattr(_survreg, f"{generic}_survreg", None)
    if method is None:
        raise TypeError(f"{generic} requires a fitted coxph or survreg model")
    return method


def _dispatch(generic: str, fit: Any, *args: Any, **kwargs: Any) -> Any:
    from ._survreg import SurvregModelResult

    if not isinstance(fit, SurvregModelResult):
        raise TypeError(f"{generic} requires a fitted coxph or survreg model")
    return _survreg_method(generic)(fit, *args, **kwargs)


# ---------------------------------------------------------------------------
# coefficients and likelihoods
# ---------------------------------------------------------------------------


def coef(fit: Any) -> list[float]:
    """``fit$coefficients`` (``NaN`` marks an aliased Cox coefficient, like R's ``NA``)."""

    if isinstance(fit, CoxphModel | CchModelResult):
        return fit.coefficients
    return _dispatch("coef", fit)


def coef_names(fit: Any, *, complete: Any | None = None) -> list[str]:
    """``names(coef(fit))``; ``complete=False`` drops aliased Cox coefficients."""

    if isinstance(fit, CoxphModel | CchModelResult):
        include = _normalize_bool_option_with_default(complete, "complete", True)
        names = list(fit.coef_names)
        if include:
            return names
        return [name for name, b in zip(names, fit.coefficients, strict=True) if not math.isnan(b)]
    return _dispatch("coef_names", fit, complete=complete)


def vcov(fit: Any, *, complete: Any = True) -> list[list[float]]:
    """``vcov``: the robust variance when the fit used one, else the model-based one."""

    if isinstance(fit, CoxphModel | CchModelResult):
        include = _normalize_bool_option_with_default(complete, "complete", True)
        var = fit.var
        if include:
            return var
        keep = [i for i, b in enumerate(fit.coefficients) if not math.isnan(b)]
        return [[var[i][j] for j in keep] for i in keep]
    return _dispatch("vcov", fit, complete=complete)


def loglik(fit: Any) -> float:
    """``logLik``: the fitted partial log-likelihood ``fit$loglik[2]``."""

    if isinstance(fit, CoxphModel):
        return fit.loglik[1]
    return _dispatch("loglik", fit)


def nobs(fit: Any) -> int:
    """``nobs``: the number of events for a Cox model (the ``nobs`` attribute of its logLik)."""

    if isinstance(fit, CoxphModel):
        return fit.nevent
    return _dispatch("nobs", fit)


def degrees_freedom(fit: Any) -> int:
    """The ``df`` attribute of ``logLik``: the number of estimated coefficients."""

    if isinstance(fit, CoxphModel):
        return sum(1 for value in fit.coefficients if not math.isnan(value))
    return _dispatch("degrees_freedom", fit)


def df_residual(fit: Any) -> int:
    """``df.residual`` (survreg only)."""

    if isinstance(fit, CoxphModel):
        raise TypeError("df_residual is only defined for fitted survreg models")
    return _dispatch("df_residual", fit)


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


def model_formula(fit: Any) -> str:
    """``formula(fit)``."""

    if isinstance(fit, CoxphModel | CchModelResult | AaregModelResult):
        formula = fit.formula
        if formula is None:
            raise TypeError("model_formula requires a formula-based fitted model")
        return formula
    return _dispatch("model_formula", fit)


def model_term_names(fit: Any, terms: Any | None = None) -> list[str]:
    """``attr(terms(fit), 'term.labels')``, optionally the subset ``terms`` selects."""

    if isinstance(fit, CoxphModel):
        from ._coxph import _terms_selection

        names = list(fit.assign)
        return [names[idx] for idx in _terms_selection(terms, names)]
    return _dispatch("model_term_names", fit, terms)


def model_weights(fit: Any) -> list[float] | None:
    """``weights(fit)``: the case weights, ``None`` when none were given."""

    if isinstance(fit, CoxphModel):
        return fit.weights
    if isinstance(fit, AaregModelResult):
        return None if fit.weights is None else list(fit.weights)
    return _dispatch("model_weights", fit)


def model_matrix(fit: Any) -> dict[str, Any]:
    """``model.matrix(fit)``: the design matrix, its column names and ``assign``."""

    if isinstance(fit, CoxphModel):
        assign = [0] * len(fit.coef_names)
        for term_idx, columns in enumerate(fit.assign.values(), start=1):
            for col in columns:
                assign[col] = term_idx
        return {"data": fit.x, "columns": list(fit.coef_names), "assign": assign}
    return _dispatch("model_matrix", fit)


def model_frame(fit: Any) -> dict[str, list[Any]]:
    """``model.frame(fit)`` for a fit made with ``model=TRUE``."""

    if isinstance(fit, Mapping):
        if not fit:
            raise TypeError("model_frame requires a non-empty grouped survfit result")
        return model_frame(next(iter(fit.values())))
    if isinstance(fit, CoxphModel | AaregModelResult):
        frame = fit.model
        if frame is None:
            raise TypeError("model_frame requires a fit made with model=TRUE")
        return _plain_model_frame(frame)
    frame = getattr(fit, "model", None)
    if isinstance(frame, Mapping):
        return _plain_model_frame(frame)
    return _dispatch("model_frame", fit)


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
        if text_name in {"group", "(id)", "(cluster)", "(strata)"}:
            columns[text_name] = _materialize_labels(values, text_name)
            continue
        materialized = _materialize_1d(values, text_name)
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns[text_name] = list(materialized)
    return columns


# ---------------------------------------------------------------------------
# predict / residuals / summaries
# ---------------------------------------------------------------------------


def predict(fit: Any, newdata: Any | None = None, **kwargs: Any) -> Any:
    """``predict``: see :func:`survival.r._coxph.predict_coxph` and the survreg method.
    R's ``se.fit`` spelling is accepted."""

    if "se.fit" in kwargs:
        kwargs["se_fit"] = kwargs.pop("se.fit")
    if isinstance(fit, CoxphModel):
        return predict_coxph(fit, newdata, **kwargs)
    return _dispatch("predict", fit, newdata, **kwargs)


def fitted(fit: Any, **kwargs: Any) -> Any:
    """``fitted``: ``predict`` on the training data."""

    return predict(fit, None, **kwargs)


def residuals(fit: Any, *, type: str = "martingale", **kwargs: Any) -> Any:
    """``residuals``: see :func:`survival.r._coxph.residuals_coxph` and the survreg method."""

    if isinstance(fit, CoxphModel):
        return residuals_coxph(fit, type=type, **kwargs)
    return _dispatch("residuals", fit, type=type, **kwargs)


def _coefficient_selection(parm: Any, names: list[str]) -> list[int]:
    if parm is None:
        return list(range(len(names)))
    values = [parm] if isinstance(parm, str | int) else list(_materialize_1d(parm, "parm"))
    indices: list[int] = []
    for value in values:
        if isinstance(value, str):
            if value not in names:
                raise ValueError(f"unknown coefficient name {value!r}")
            indices.append(names.index(value))
        else:
            idx = _integer_scalar(value, "parm") - 1
            if idx < 0 or idx >= len(names):
                raise IndexError("parm index out of range")
            indices.append(idx)
    return indices


def confint(
    fit: Any, parm: Any | None = None, *, level: Any = 0.95
) -> list[dict[str, float | str]]:
    """``confint``: normal-approximation intervals for the coefficients."""

    if not isinstance(fit, CoxphModel | CchModelResult):
        return _dispatch("confint", fit, parm, level=level)
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


def model_summary(fit: Any, **kwargs: Any) -> dict[str, Any]:
    """``summary``: R's summary list of a coxph, clogit, cch, aareg or survreg fit."""

    if isinstance(fit, CoxphModel):
        return summary_coxph(fit, **kwargs)
    if isinstance(fit, CchModelResult):
        return summary_cch(fit)
    if isinstance(fit, AaregModelResult):
        return summary_aareg(fit, **kwargs)
    return _dispatch("model_summary", fit, **kwargs)


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


# --- the R bridge's grouped view of a stratified curve set --------------------------------

_PER_STRATUM_FIELDS = frozenset(
    {"n", "n_id", "p0", "se0", "influence_pstate", "influence_surv", "influence_chaz"}
)


def _bare_strata_label(name: str) -> str:
    """``rx=1`` -> ``1`` and ``a=1, b=x`` -> ``1, x`` (the labels the bridge names curves by)."""

    return re.sub(r"(^|, )[^=,]+=", r"\1", name)


def _survfit_stratum(result: Any, index: int, rows: slice) -> Any:
    """One stratum of ``result`` as its own unstratified result."""

    changes: dict[str, Any] = {"strata": None}
    row_count = len(result.time)
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if field.name == "strata" or not isinstance(value, list | tuple):
            continue
        if field.name in _PER_STRATUM_FIELDS:
            if field.name == "p0" and value and not isinstance(value[0], list | tuple):
                continue
            changes[field.name] = [value[index]] if field.name in {"n", "n_id"} else value[index]
        elif len(value) == row_count:
            changes[field.name] = list(value[rows])
    counts = getattr(result, "counts", None)
    if counts is not None:
        changes["counts"] = _core.SurvfitCounts(
            list(counts.n_risk[rows]),
            list(counts.n_event[rows]),
            list(counts.n_censor[rows]),
            None if counts.n_enter is None else list(counts.n_enter[rows]),
        )
    return dataclasses.replace(result, **changes)


def _survfit_strata_curves(result: Any) -> Any:
    """The bridge's shape for a stratified ``survfit``: ``{bare level label: curve}``.

    ``survfit.formula`` results lay their strata end to end with ``strata`` naming the
    blocks (R's layout); the R bridge represents them as a named list of per-stratum
    curves keyed by the bare levels.  Unstratified results are returned as they are.
    """

    if not isinstance(result, SurvfitResult | SurvfitMultiStateResult) or not result.strata:
        return result
    curves: dict[str, Any] = {}
    start = 0
    for index, (name, count) in enumerate(result.strata.items()):
        curves[_bare_strata_label(name)] = _survfit_stratum(
            result, index, slice(start, start + count)
        )
        start += count
    return curves


def _subset_survfit_multistate(
    result: SurvfitMultiStateResult,
    state_indices: Any,
    keep_n_id: bool | None = None,
) -> SurvfitMultiStateResult:
    """``fit[, states]``: keep the selected state columns (the transition columns go).

    R keeps ``n.id`` for a stratified object and drops it otherwise (``[.survfitms``
    reads ``x$id`` there); ``keep_n_id`` overrides that for the bridge's split curves.
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

    def select_columns(values: Sequence[Sequence[Any]] | None) -> list[list[float]] | None:
        if values is None:
            return None
        return [[float(row[index]) for index in indices] for row in values]

    def select_p0(values: Sequence[Any]) -> list[Any]:
        if values and isinstance(values[0], list | tuple):
            return [[float(row[index]) for index in indices] for row in values]
        return [float(values[index]) for index in indices]

    empty_transitions: list[list[float]] = [[] for _ in result.time]
    if keep_n_id is None:
        keep_n_id = bool(result.strata)
    return dataclasses.replace(
        result,
        n_id=result.n_id if keep_n_id else None,
        n_risk=select_columns(result.n_risk) or [],
        n_event=select_columns(result.n_event) or [],
        n_censor=select_columns(result.n_censor) or [],
        n_transition=empty_transitions,
        pstate=select_columns(result.pstate) or [],
        cumhaz=empty_transitions,
        p0=select_p0(result.p0),
        states=tuple(result.states[index] for index in indices),
        hazard_names=(),
        transitions=None,
        std_err=select_columns(result.std_err),
        std_chaz=None,
        std_auc=select_columns(result.std_auc),
        lower=select_columns(result.lower),
        upper=select_columns(result.upper),
        influence_pstate=None,
        oldstate=result.oldstate or tuple(result.states),
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
    p0_rows = [
        [float(value) for value in (curve.p0[0] if grouped_p0 else curve.p0)]
        for _label, curve in curves
        for grouped_p0 in [bool(curve.p0) and isinstance(curve.p0[0], list | tuple)]
    ]
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
        totals = [[0.0] * len(first.transitions.colnames) for _row in first.transitions.rownames]
        for _label, curve in curves:
            if curve.transitions is None:
                continue
            for row_index, row in enumerate(curve.transitions.values):
                for col_index, value in enumerate(row):
                    totals[row_index][col_index] += float(value)
        structure["transitions"] = {
            "values": totals,
            "rows": list(first.transitions.rownames),
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
        frame = {
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
        "df": [int(row["df"]) for row in result.table],
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


def _anova_frame(result: Any) -> dict[str, list[Any]]:
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


def as_data_frame(result: Any) -> dict[str, list[Any]]:
    """Return a plain column-oriented table for common R-style result objects."""

    if isinstance(result, Surv):
        return _surv_response_frame(result)
    if isinstance(result, CoxSurvfitResult):
        return _cox_survfit_frame(result)
    if isinstance(result, CoxBaseHazardResult):
        return _cox_basehaz_frame(result)
    if isinstance(result, SurvfitMultiStateResult):
        return _survfit_multistate_frame(result)
    if isinstance(result, SurvfitResult):
        return _survfit_frame(result)
    if isinstance(result, CoxZPHResult):
        return _cox_zph_frame(result)
    if isinstance(result, CoxPHDetailResult):
        return _coxph_detail_frame(result)
    if isinstance(result, ConcordanceResult):
        return _concordance_frame(result)
    if isinstance(result, PyearsResult):
        return _pyears_result_frame(result)
    if isinstance(result, FineGrayFrame):
        return {name: list(values) for name, values in result.items()}
    if isinstance(result, FineGrayOutput):
        return _finegray_frame(result)
    if isinstance(result, Mapping):
        return _grouped_survfit_frame(result)
    if isinstance(result, SurvDiffResult):
        return _survdiff_frame(result)
    if hasattr(result, "rows") and hasattr(result, "test"):
        return _anova_frame(result)
    if hasattr(result, "frame") and hasattr(result, "heading"):  # anova.survreg
        return result.frame()
    raise TypeError("as_data_frame requires a survival result object")
