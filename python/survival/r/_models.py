"""Model generics (``coef``, ``vcov``, ``predict``, ``residuals``, ``model_summary``,
``as_data_frame``, ...): R-style S3 dispatch on the fitted object.

Cox, clogit, cch and aareg fits dispatch here; survreg fits go to the matching
``*_survreg`` function of :mod:`survival.r._survreg` (``predict_survreg``,
``residuals_survreg``, ``summary_survreg``, ...).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any

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
from ._survfit import _optional_float_list
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
    SurvfitMultiStateResult,
    SurvfitResult,
    TurnbullSurvfitResult,
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
    """``predict``: see :func:`survival.r._coxph.predict_coxph` and the survreg method."""

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


def _empty_columns(names: tuple[str, ...]) -> dict[str, list[Any]]:
    return {name: [] for name in names}


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
    std_err = list(result.std_err)
    conf_lower = list(result.conf_lower)
    conf_upper = list(result.conf_upper)
    for idx, survival in enumerate(result.estimate):
        if survival <= 0.0:
            if idx < len(std_err):
                std_err[idx] = math.nan
            if idx < len(conf_lower):
                conf_lower[idx] = math.nan
            if idx < len(conf_upper):
                conf_upper[idx] = math.nan
    row_count = len(result.time)
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "surv": result.estimate,
        "cumhaz": result.cumhaz,
    }
    _add_optional_survfit_column(frame, "std.err", std_err, row_count)
    _add_optional_survfit_column(frame, "lower", conf_lower, row_count)
    _add_optional_survfit_column(frame, "upper", conf_upper, row_count)
    _add_optional_survfit_column(frame, "std.chaz", result.std_chaz, row_count)
    if result.n_enter is not None:
        frame["n.enter"] = result.n_enter
    return frame


def _survfit_multistate_column(
    values: Sequence[Sequence[Any]],
    state_index: int,
    row_count: int,
    state_count: int,
    name: str,
) -> list[Any]:
    if len(values) != row_count:
        raise ValueError(f"multi-state survfit column {name!r} must match time length")
    column: list[Any] = []
    for row in values:
        if len(row) != state_count:
            raise ValueError(f"multi-state survfit column {name!r} must have one value per state")
        column.append(row[state_index])
    return column


def _survfit_multistate_frame(result: SurvfitMultiStateResult) -> dict[str, list[Any]]:
    row_count = len(result.time)
    state_count = len(result.states)
    required = {
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "pstate": result.pstate,
    }
    optional = {
        "std.err": result.std_err,
        "lower": result.conf_lower,
        "upper": result.conf_upper,
    }
    frame: dict[str, list[Any]] = {
        "time": [],
        **{name: [] for name in required},
        **{name: [] for name, values in optional.items() if values is not None},
        "state": [],
    }
    for state_index, state in enumerate(result.states):
        frame["time"].extend(float(value) for value in result.time)
        for name, values in required.items():
            frame[name].extend(
                _survfit_multistate_column(
                    values,
                    state_index,
                    row_count,
                    state_count,
                    name,
                )
            )
        for name, values in optional.items():
            if values is not None:
                frame[name].extend(
                    _survfit_multistate_column(
                        values,
                        state_index,
                        row_count,
                        state_count,
                        name,
                    )
                )
        frame["state"].extend([state] * row_count)
    return frame


def _subset_survfit_multistate(
    result: SurvfitMultiStateResult,
    state_indices: Any,
) -> SurvfitMultiStateResult:
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

    def select_columns(values: list[list[float]]) -> list[list[float]]:
        return [[float(row[index]) for index in indices] for row in values]

    def select_optional(
        values: list[list[float]] | None,
    ) -> list[list[float]] | None:
        return None if values is None else select_columns(values)

    empty_transitions = [[] for _ in result.time]
    return SurvfitMultiStateResult(
        time=[float(value) for value in result.time],
        n_risk=select_columns(result.n_risk),
        n_event=select_columns(result.n_event),
        n_censor=select_columns(result.n_censor),
        pstate=select_columns(result.pstate),
        cumhaz=empty_transitions,
        states=tuple(result.states[index] for index in indices),
        transitions=(),
        p0=[float(result.p0[index]) for index in indices],
        t0=result.t0,
        n=result.n,
        n_id=result.n_id,
        std_err=select_optional(result.std_err),
        std_err0=(
            None
            if result.std_err0 is None
            else [float(result.std_err0[index]) for index in indices]
        ),
        std_chaz=None if result.std_chaz is None else empty_transitions,
        std_auc=select_optional(result.std_auc),
        conf_lower=select_optional(result.conf_lower),
        conf_upper=select_optional(result.conf_upper),
        n_risk_count=select_optional(result.n_risk_count),
        n_event_count=select_optional(result.n_event_count),
        n_censor_count=select_optional(result.n_censor_count),
        n_enter=select_optional(result.n_enter),
        n_enter_count=select_optional(result.n_enter_count),
        n_transition=empty_transitions,
        n_transition_count=(None if result.n_transition_count is None else empty_transitions),
        model=result.model,
        surv_type=result.surv_type,
        conf_type=result.conf_type,
        conf_level=result.conf_level,
        oldstate=result.states if result.oldstate is None else result.oldstate,
        p0_fixed=result.p0_fixed,
        timefix=result.timefix,
    )


def _survfit_multistate_structure(
    result: SurvfitMultiStateResult | Mapping[Any, Any],
) -> dict[str, Any]:
    if isinstance(result, SurvfitMultiStateResult):
        curves = [(None, result)]
        grouped = False
    elif (
        isinstance(result, Mapping)
        and result
        and all(isinstance(curve, SurvfitMultiStateResult) for curve in result.values())
    ):
        curves = list(result.items())
        grouped = True
    else:
        raise TypeError("survfit structure requires a multi-state result")

    first = curves[0][1]
    for _label, curve in curves:
        if curve.states != first.states:
            raise ValueError("grouped multi-state results must share state columns")
        if curve.transitions != first.transitions:
            raise ValueError("grouped multi-state results must share transition columns")
        if curve.surv_type != first.surv_type:
            raise ValueError("grouped multi-state results must share a response type")

    def combined_matrix(name: str) -> list[list[float]] | None:
        matrices = [getattr(curve, name) for _label, curve in curves]
        if all(matrix is None for matrix in matrices):
            return None
        if any(matrix is None for matrix in matrices):
            raise ValueError(f"grouped multi-state results must share {name} output")
        return [[float(value) for value in row] for matrix in matrices for row in matrix]

    transition_names = [f"{source + 1}:{target + 1}" for source, target in first.transitions]
    structure: dict[str, Any] = {
        "n": [curve.n for _label, curve in curves] if grouped else first.n,
        "time": [float(value) for _label, curve in curves for value in curve.time],
        "n.risk": combined_matrix("n_risk"),
        "n.event": combined_matrix("n_event"),
        "n.censor": combined_matrix("n_censor"),
        "pstate": combined_matrix("pstate"),
    }
    if first.transitions:
        structure["n.transition"] = combined_matrix("n_transition")
    if grouped or first.oldstate is None:
        structure["n.id"] = [curve.n_id for _label, curve in curves] if grouped else first.n_id
    if first.transitions:
        structure["cumhaz"] = combined_matrix("cumhaz")
    n_enter = combined_matrix("n_enter")
    if n_enter is not None:
        structure["n.enter"] = n_enter
    structure["p0"] = (
        [[float(value) for value in curve.p0] for _label, curve in curves]
        if grouped
        else [float(value) for value in first.p0]
    )
    if grouped:
        structure["strata"] = {str(label): len(curve.time) for label, curve in curves}
    for field_name, attribute in (
        ("std.err", "std_err"),
        ("std.chaz", "std_chaz"),
        ("std.auc", "std_auc"),
    ):
        values = combined_matrix(attribute)
        if values is not None and (field_name != "std.chaz" or first.transitions):
            structure[field_name] = values
    structure["logse"] = False

    if first.transitions:
        target_states = list(dict.fromkeys(target for _source, target in first.transitions))
        target_columns = {state: index for index, state in enumerate(target_states)}
        transition_table = [[0.0] * (len(target_states) + 1) for _state in first.states]
        for _label, curve in curves:
            transition_values = (
                curve.n_transition_count
                if curve.n_transition_count is not None
                else curve.n_transition
            )
            for row in transition_values:
                for transition_index, (source, target) in enumerate(curve.transitions):
                    transition_table[source][target_columns[target]] += float(row[transition_index])
            censor_values = (
                curve.n_censor_count if curve.n_censor_count is not None else curve.n_censor
            )
            for row in censor_values:
                for state, value in enumerate(row):
                    transition_table[state][-1] += float(value)
        structure["transitions"] = {
            "values": transition_table,
            "rows": list(first.states),
            "columns": [first.states[state] for state in target_states] + ["(censored)"],
        }

    for field_name, attribute in (("lower", "conf_lower"), ("upper", "conf_upper")):
        values = combined_matrix(attribute)
        if values is not None:
            structure[field_name] = values
    structure.update(
        {
            "conf.type": first.conf_type,
            "conf.int": first.conf_level,
            "states": list(first.states),
            "type": first.surv_type,
            "t0": first.t0,
            "_transition_names": transition_names,
        }
    )
    if first.oldstate is not None:
        structure["oldstate"] = list(first.oldstate)
    return structure


def _turnbull_survfit_frame(result: TurnbullSurvfitResult) -> dict[str, list[Any]]:
    return {
        "time": result.time_points,
        "surv": result.survival,
        "lower": result.survival_lower,
        "upper": result.survival_upper,
    }


def _grouped_survfit_frame(result: Mapping[Any, Any]) -> dict[str, list[Any]]:
    if result and all(isinstance(curve, SurvfitMultiStateResult) for curve in result.values()):
        curve_frames = {label: _survfit_multistate_frame(curve) for label, curve in result.items()}
        columns = list(next(iter(curve_frames.values())))
        if "state" not in columns:
            raise ValueError("multi-state survfit frame is missing its state column")
        frame = {
            name: [] for name in [*[name for name in columns if name != "state"], "strata", "state"]
        }
        states = next(iter(result.values())).states
        if any(curve.states != states for curve in result.values()):
            raise ValueError("grouped multi-state results must share state columns")
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

    frame: dict[str, list[Any]] = {}
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


def _raw_survfit_frame(result: Any) -> dict[str, list[Any]]:
    return _survfit_frame(
        SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[float(value) for value in result.std_err],
            conf_lower=[float(value) for value in result.conf_lower],
            conf_upper=[float(value) for value in result.conf_upper],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[float(value) for value in result.std_chaz],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
        )
    )


def _raw_turnbull_survfit_frame(result: Any) -> dict[str, list[Any]]:
    return _turnbull_survfit_frame(
        TurnbullSurvfitResult(
            time_points=[float(value) for value in result.time_points],
            survival=[float(value) for value in result.survival],
            survival_lower=[float(value) for value in result.survival_lower],
            survival_upper=[float(value) for value in result.survival_upper],
            n_iter=int(result.n_iter),
            converged=bool(result.converged),
        )
    )


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
        return [row[curve] for row in values] if ncurve > 1 else list(values)

    frame: dict[str, list[Any]] = {
        name: [] for name in ("curve", "time", "n.risk", "n.event", "n.censor", "surv", "cumhaz")
    }
    if result.strata is not None:
        frame["strata"] = []
    optional = {
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
        frame["cumhaz"].extend(column(result.cumhaz, curve))
        if result.strata is not None:
            frame["strata"].extend(strata)
        for name, values in optional.items():
            if values is not None:
                frame[name].extend(column(values, curve))
    return frame


def _survdiff_frame(result: Any) -> dict[str, list[Any]]:
    observed = [float(value) for value in result.observed]
    expected = [float(value) for value in result.expected]
    variance = getattr(result, "variance", None)
    if isinstance(variance, int | float):
        variance_diag = [float(variance)] * len(observed)
    elif variance is not None:
        variance_diag = [float(row[idx]) for idx, row in enumerate(variance)]
    else:
        variance_diag = [math.nan] * len(observed)
    return {
        "group": [idx + 1 for idx in range(len(observed))],
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
    if isinstance(result, TurnbullSurvfitResult):
        return _turnbull_survfit_frame(result)
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return _raw_survfit_frame(result)
    if all(
        hasattr(result, name)
        for name in ("time_points", "survival", "survival_lower", "survival_upper")
    ):
        return _raw_turnbull_survfit_frame(result)
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
    if hasattr(result, "observed") and hasattr(result, "expected") and hasattr(result, "variance"):
        return _survdiff_frame(result)
    if hasattr(result, "rows") and hasattr(result, "test"):
        return _anova_frame(result)
    if hasattr(result, "frame") and hasattr(result, "heading"):  # anova.survreg
        return result.frame()
    raise TypeError("as_data_frame requires a survival result object")
