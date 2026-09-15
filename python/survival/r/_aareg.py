"""``aareg`` Aalen additive regression (R/aareg.R, summary.aareg.R): the model
frame and the summary table; the fit runs in Rust."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _float_vector,
    _integer_scalar,
    _label_levels,
    _match_string_arg,
    _normalize_bool_option,
    _pop_dotted_keyword,
)
from ._coxph import _pchisq_upper
from ._fit import _model_frame
from ._types import AaregModelResult

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
    na_action: str | None = "fail",
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
) -> AaregModelResult | _core.AaregResult:
    """Fit Aalen's additive regression model (R's ``aareg``).

    ``survival.aareg`` is also the package-level name of the engine's option-driven
    ``aareg(AaregOptions)``; an ``AaregOptions`` first argument is handed to it.
    """

    if isinstance(formula, _core.AaregOptions):
        return _core.aareg(formula)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        raise TypeError(f"aareg got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    test_name = _match_string_arg(
        test, "test", _TESTS, "test must be one of aalen, variance, nrisk"
    )
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
    raw = _core.aareg_fit(
        list(response.time),
        [int(value) for value in response.event],
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
    return AaregModelResult(
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
        dfbeta=(
            None
            if raw.dfbeta is None
            else [[list(values) for values in rows] for rows in raw.dfbeta]
        ),
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
        x=frame.x if _normalize_bool_option(x, "x") else None,
        y=response if _normalize_bool_option(y, "y") else None,
    )


def _cumsum(values: list[float]) -> list[float]:
    out: list[float] = []
    total = 0.0
    for value in values:
        total += value
        out.append(total)
    return out


def summary_aareg(
    fit: AaregModelResult,
    maxtime: Any | None = None,
    test: Any | None = None,
    scale: Any = 1.0,
) -> dict[str, Any]:
    """R's ``summary.aareg``: the slope of each coefficient curve, the test statistic
    per covariate and the overall chi-square (which excludes the intercept)."""

    test_name = (
        fit.test
        if test is None
        else _match_string_arg(test, "test", ("aalen", "nrisk"), "test must be aalen or nrisk")
    )
    scale_value = _finite_float(scale, "scale")
    ntime = (
        len(fit.times)
        if maxtime is None
        else sum(1 for t in fit.times if t <= _finite_float(maxtime, "maxtime"))
    )
    times = fit.times[:ntime]
    nvar = len(fit.coefficient_names)
    if test_name == "aalen":
        twt = [list(row) for row in fit.time_weights[:ntime]]
        scales = [sum(row[k] for row in twt) / scale_value for k in range(nvar)]
    else:
        twt = [[fit.n_risk[i]] * nvar for i in range(ntime)]
        scales = [ntime / scale_value] * nvar
    tx = [[twt[i][k] * fit.coefficient[i][k] for k in range(nvar)] for i in range(ntime)]
    slope: list[float] = []
    for k in range(nvar):
        ctx = _cumsum([row[k] for row in tx])
        tempwt = sum(twt[i][k] * times[i] ** 2 for i in range(ntime))
        slope.append(sum(c * t for c, t in zip(ctx, times, strict=True)) / tempwt)
    if maxtime is not None or fit.test != test_name:
        test_stat = [sum(row[k] for row in tx) for k in range(nvar)]
        test_var = [[sum(row[j] * row[k] for row in tx) for k in range(nvar)] for j in range(nvar)]
        test_var2 = None
    else:
        test_stat = list(fit.test_statistic)
        test_var = fit.test_variance
        test_var2 = fit.robust_test_variance
    variance = test_var if test_var2 is None else test_var2
    se = [math.sqrt(variance[k][k]) for k in range(nvar)]
    columns = ["slope", "coef", "se(coef)", "z", "p"]
    rows = []
    for k, name in enumerate(fit.coefficient_names):
        z = test_stat[k] / se[k]
        row: dict[str, Any] = {
            "name": name,
            "slope": slope[k],
            "coef": test_stat[k] / scales[k],
            "se": math.sqrt(test_var[k][k]) / scales[k],
            "z": z,
            "p": 2.0 * _pnorm_lower(-abs(z)),
        }
        if test_var2 is not None:
            row["robust_se"] = se[k] / scales[k]
        rows.append(row)
    if test_var2 is not None:
        columns = ["slope", "coef", "se(coef)", "robust se", "z", "p"]
    sub_var = [[variance[j][k] for k in range(1, nvar)] for j in range(1, nvar)]
    chisq = _core.coxph_wtest(sub_var, [test_stat[1:]]).test[0] if nvar > 1 else math.nan
    return {
        "model_type": "aareg",
        "table": rows,
        "columns": columns,
        "test": test_name,
        "test_statistic": test_stat,
        "test_var": test_var,
        "test_var2": test_var2,
        "chisq": chisq,
        "df": nvar - 1,
        "p": _pchisq_upper(chisq, nvar - 1),
        "n": [fit.n[0], len(set(times)), fit.n[2]],
    }


def _pnorm_lower(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))
