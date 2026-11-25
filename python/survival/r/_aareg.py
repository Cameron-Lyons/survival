"""``aareg`` Aalen additive regression (R/aareg.R, summary.aareg.R): the model
frame and the summary table; the fit runs in Rust."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _DEFAULT_NA_ACTION,
    _finite_float,
    _float_vector,
    _integer_scalar,
    _label_levels,
    _match_string_arg,
    _normalize_bool_option,
    _pop_dotted_keyword,
)
from ._fit import _model_frame
from ._formula import _without_survival_namespace
from ._types import AaregModel, SummaryAaregResult

_TESTS = ("aalen", "variance", "nrisk")


def _taper_values(value: Any) -> list[float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        values = [_finite_float(value, "taper")]
    else:
        values = _float_vector(value, "taper")
    if not values or any(item <= 0.0 for item in values):
        raise ValueError("taper must contain positive finite values")
    return values


def aareg(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
    qrtol: Any = 1e-7,
    nmin: Any | None = None,
    dfbeta: Any = False,
    taper: Any = 1.0,
    test: Any = "aalen",
    cluster: Any | None = None,
    model: Any = False,
    x: Any = False,
    y: Any = False,
    **kwargs: Any,
) -> AaregModel:
    """Fit Aalen's additive regression model (R's ``aareg``)."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, _DEFAULT_NA_ACTION)
    if kwargs:
        raise TypeError(f"aareg got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    test_name = _match_string_arg(
        test, "test", _TESTS, "test must be one of aalen, variance, nrisk"
    )
    formula = _without_survival_namespace(formula)
    frame = _model_frame(
        formula, data, subset=subset, na_action=na_action, weights=weights, cluster=cluster
    )
    response = frame.y
    if response.type not in {"right", "counting"}:
        raise ValueError(f'Aalen model doesn\'t support "{response.type}" survival data')
    if frame.terms.strata:
        raise ValueError("Strata terms not allowed")
    cluster_values = frame.cluster
    cluster_codes = None
    if cluster_values is not None:
        levels = _label_levels(cluster_values, "cluster")
        index = {level: idx for idx, level in enumerate(levels)}
        cluster_codes = [index[value] for value in cluster_values]
    keep_dfbeta = _normalize_bool_option(dfbeta, "dfbeta") or cluster_codes is not None
    qrtol_value = _finite_float(qrtol, "qrtol")
    if qrtol_value <= 0.0:
        raise ValueError("qrtol must be positive")
    nmin_value = None if nmin is None else _integer_scalar(nmin, "nmin")
    events = response._event_codes()
    raw = _core.aareg_fit(
        list(response.time),
        events,
        frame.x,
        start=None if response.start is None else list(response.start),
        weights=frame.weights,
        cluster=cluster_codes,
        qrtol=qrtol_value,
        nmin=nmin_value,
        dfbeta=keep_dfbeta,
        taper=_taper_values(taper),
        test=test_name,
    )
    coefficient_names = ["Intercept", *frame.names]
    # The getter builds new lists, so the reordered influences are stored without a copy.
    influences = raw.dfbeta
    if influences is not None:
        order = sorted(range(len(response.time)), key=lambda i: (response.time[i], -events[i]))
        codes = list(range(len(order))) if cluster_codes is None else cluster_codes
        cluster_order = list(dict.fromkeys(codes[i] for i in order))
        influences = [influences[i] for i in cluster_order]
    return AaregModel(
        n=[int(value) for value in raw.n],
        times=list(raw.times),
        n_risk=list(raw.n_risk),
        coefficient=[list(row) for row in raw.coefficient],
        coefficient_names=coefficient_names,
        test_statistic=list(raw.test_statistic),
        test_statistic_names=coefficient_names,
        test_variance=[list(row) for row in raw.test_variance],
        test=str(raw.test),
        time_weights=[list(row) for row in raw.time_weights],
        dfbeta=influences,
        robust_test_variance=(
            None
            if raw.robust_test_variance is None
            else [list(row) for row in raw.robust_test_variance]
        ),
        formula=formula,
        weights=frame.weights,
        cluster=cluster_values,
        cluster_levels=None
        if cluster_values is None
        else list(_label_levels(cluster_values, "cluster")),
        model=frame.model_frame() if _normalize_bool_option(model, "model") else None,
        x=frame.matrix_rows() if _normalize_bool_option(x, "x") else None,
        y=response if _normalize_bool_option(y, "y") else None,
        term_labels=tuple(frame.assign),
        na_action=frame.na_action,
    )


def _cumsum(values: list[float]) -> list[float]:
    out: list[float] = []
    total = 0.0
    for value in values:
        total += value
        out.append(total)
    return out


def _summary_influence_covariance(
    fit: AaregModel, times: list[float], weights: list[list[float]]
) -> list[list[float]]:
    """Reweight stored group influences with a bounded conversion buffer.

    Influences already aggregate observations within clusters. Use the weights
    at the first event at each distinct time, as in R's summary.aareg.
    """
    indices = [i for i, time in enumerate(times) if i == 0 or time != times[i - 1]]
    twt = np.asarray([weights[i] for i in indices], dtype=float).T
    width, count = twt.shape
    covariance = np.zeros((width, width))
    influences = fit.dfbeta
    if influences is None:
        raise ValueError("robust summaries require stored influences")
    block_size = max(1, 131072 // max(1, width * count))
    for start in range(0, len(influences), block_size):
        if isinstance(influences, np.ndarray):
            block = np.asarray(influences[start : start + block_size, :, :count], dtype=float)
        else:
            block = np.asarray(
                [
                    [column if len(column) == count else column[:count] for column in group]
                    for group in influences[start : start + block_size]
                ],
                dtype=float,
            )
        weighted = np.einsum("gpt,pt->gp", block, twt)
        covariance += weighted.T @ weighted
        del block, weighted
    return covariance.tolist()


def summary_aareg(
    fit: AaregModel,
    maxtime: Any | None = None,
    test: Any | None = None,
    scale: Any = 1.0,
) -> SummaryAaregResult:
    """R's ``summary.aareg``: the slope of each coefficient curve, the test statistic
    per covariate and the overall chi-square (which excludes the intercept)."""

    test_name = _match_string_arg(
        fit.test if test is None else test,
        "test",
        ("aalen", "nrisk"),
        "test must be aalen or nrisk",
    )
    scale_value = _finite_float(scale, "scale")
    if scale_value == 0:
        raise ValueError("scale must be nonzero")
    cutoff = None if maxtime is None else _finite_float(maxtime, "maxtime")
    ntime = len(fit.times) if cutoff is None else sum(1 for t in fit.times if t <= cutoff)
    if ntime == 0:
        raise ValueError("maxtime must include at least one event")
    times = fit.times[:ntime]
    nvar = len(fit.coefficient_names)
    times_arr = np.asarray(times, dtype=np.float64)
    coef_arr = np.asarray(fit.coefficient[:ntime], dtype=np.float64)
    if test_name == "aalen":
        twt_arr = np.asarray(fit.time_weights[:ntime], dtype=np.float64)
        scales = (twt_arr.sum(axis=0) / scale_value).tolist()
    else:
        twt_arr = np.broadcast_to(
            np.asarray(fit.n_risk[:ntime], dtype=np.float64)[:, None], (ntime, nvar)
        )
        scales = [ntime / scale_value] * nvar
    tx_arr = twt_arr * coef_arr
    ctx_arr = np.cumsum(tx_arr, axis=0)
    tempwt = np.sum(twt_arr * (times_arr[:, None] ** 2), axis=0)
    numerator = np.sum(ctx_arr * times_arr[:, None], axis=0)
    with np.errstate(all="ignore"):
        slope = np.divide(numerator, tempwt).tolist()
    if maxtime is not None or fit.test != test_name:
        test_stat = tx_arr.sum(axis=0).tolist()
        test_var = (tx_arr.T @ tx_arr).tolist()
        test_var2 = (
            None
            if fit.dfbeta is None
            else _summary_influence_covariance(fit, times, twt_arr.tolist())
        )
    else:
        test_stat = list(fit.test_statistic)
        test_var = fit.test_variance
        test_var2 = fit.robust_test_variance
    variance = test_var if test_var2 is None else test_var2
    se = [math.sqrt(variance[k][k]) for k in range(nvar)]
    rows = []
    for k, name in enumerate(fit.coefficient_names):
        z = test_stat[k] / se[k]
        row: dict[str, Any] = {
            "name": name,
            "slope": slope[k],
            "coef": test_stat[k] / scales[k],
            "se(coef)": math.sqrt(test_var[k][k]) / scales[k],
        }
        if test_var2 is not None:
            row["robust se"] = se[k] / scales[k]
        row["z"] = z
        row["p"] = 2.0 * _pnorm_lower(-abs(z))
        rows.append(row)
    sub_var = [[variance[j][k] for k in range(1, nvar)] for j in range(1, nvar)]
    chisq = _core.coxph_wtest(sub_var, [test_stat[1:]]).test[0] if nvar > 1 else math.nan
    return SummaryAaregResult(
        table=rows,
        test=test_name,
        test_statistic=test_stat,
        test_var=test_var,
        test_var2=test_var2,
        chisq=chisq,
        n=[fit.n[0], len(set(times)), fit.n[2]],
        df=nvar - 1,
        p=_core.pchisq(chisq, nvar - 1.0, lower_tail=False),
    )


def _pnorm_lower(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))
