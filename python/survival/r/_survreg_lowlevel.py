"""Bare parametric fitting on prepared matrices, as in R's survreg.fit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _finite_float,
    _float_vector,
    _integer_scalar,
    _materialize_labels,
    _matrix_input_column_names,
    _numeric_design_matrix,
    _warn_outside_package,
)
from ._surv import Surv
from ._survreg import _parms_vector, _resolve_control, _resolve_distribution, survreg_distributions


@dataclass(frozen=True)
class SurvregFitResult:
    """Bare fit components, without a retained response, design or distribution.

    Coefficients include estimated log-scales. The score retains the internal
    covariate scaling used by R. Numeric properties return independent copies.
    """

    _fit: _core.SurvregFitResult = field(repr=False)
    coefficient_names: tuple[str, ...]
    icoef_names: tuple[str, ...]
    variance_names: tuple[str, ...] | None

    @property
    def coefficients(self) -> list[float]:
        return self._fit.coefficients

    @property
    def icoef(self) -> list[float]:
        return self._fit.icoef

    @property
    def var(self) -> list[list[float]]:
        return self._fit.var

    @property
    def loglik(self) -> list[float]:
        return self._fit.loglik

    @property
    def iter(self) -> int:
        return self._fit.iter

    @property
    def linear_predictors(self) -> list[float]:
        return self._fit.linear_predictors

    @property
    def df(self) -> int:
        return self._fit.df

    @property
    def score(self) -> list[float]:
        return self._fit.score


def _unused_distribution_component(*args: Any) -> Any:
    raise ValueError("distribution component is unavailable for this bare fit")


def _fitting_distribution(dist: Any, parms: Any) -> _core.SurvregDistribution:
    if isinstance(dist, str):
        key = dist
        if key not in survreg_distributions:
            raise ValueError("Unrecognized distribution")
        dist = survreg_distributions[key]
        if isinstance(dist, _core.SurvregDistribution):
            if key == "t" and dist.family == _core.SurvregFamily.T and not _parms_vector(parms):
                raise ValueError("Student-t distribution requires an explicit parms value")
            # R's transformed built-in entries have no density of their own.
            # Registered custom definitions supply their own density instead.
            if (
                key
                in {"weibull", "exponential", "rayleigh", "loggaussian", "lognormal", "loglogistic"}
                and dist.transform != _core.SurvregTransform.Identity
                and dist.family != _core.SurvregFamily.Custom
            ):
                raise ValueError("Missing density function in the definition of the distribution")
    if isinstance(dist, Mapping):
        if not callable(dist.get("density")):
            raise ValueError("Missing density function in the definition of the distribution")
        values = parms
        return _core.SurvregDistribution.from_callbacks(
            dist.get("name", "Custom fitting distribution"),
            dist.get("init", _unused_distribution_component),
            dist["density"],
            _unused_distribution_component,
            _unused_distribution_component,
            variance=dist.get("variance"),
            fitting_variance=dist.get("fitting_variance"),
            parms=_parms_vector(values),
            parm_names=list(values) if isinstance(values, Mapping) else None,
        )
    if not isinstance(dist, _core.SurvregDistribution):
        raise TypeError("Invalid distribution object")
    # The three fixed built-in densities ignore their parms argument in R.
    optional = int(dist.family) in {int(_core.SurvregFamily.T), int(_core.SurvregFamily.Custom)}
    return _resolve_distribution(dist, parms if optional else None, probe=False)


def _response(y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(y, Surv):
        if y.type not in {"right", "left", "interval"}:
            raise ValueError("Invalid survival response")
        time = np.asarray(y.time, dtype=float)
        status = np.asarray(y.event, dtype=float)
        time2 = None if y.time2 is None else np.asarray(y.time2, dtype=float)
    else:
        matrix = np.asarray(y)
        if matrix.ndim != 2 or matrix.shape[1] not in {2, 3}:
            raise ValueError("Invalid survival response")
        if matrix.dtype.kind not in "biuf":
            raise TypeError("y must be a numeric matrix")
        time, status = matrix[:, 0], matrix[:, -1]
        time2 = matrix[:, 1] if matrix.shape[1] == 3 else None
    if not np.all(np.isfinite(status)) or not np.all(np.isin(status, [0, 1, 2, 3])):
        raise ValueError("y status must contain only 0, 1, 2 or 3")
    return time, status.astype(np.int32), time2


def _prepared_data(
    x: Any,
    y: Any,
    weights: Any,
    offset: Any,
    scale: Any,
    nstrat: Any,
    strata: Any,
    column_names: Sequence[str] | None,
) -> tuple[np.ndarray, _core.SurvregData, float, int, tuple[str, ...]]:
    """Validate the common prepared inputs without materializing Python rows."""
    time, status, time2 = _response(y)
    n = len(time)
    matrix = _numeric_design_matrix(x, n)
    scale_value = _finite_float(scale, "scale")
    if scale_value < 0:
        raise ValueError("Invalid scale")
    count = _integer_scalar(nstrat, "nstrat")
    if count < 1:
        raise ValueError("nstrat must be positive")
    if scale_value > 0 and count > 1:
        raise ValueError("Cannot have both a fixed scale and strata")
    codes: list[int] | None = None
    if count > 1:
        values = None if strata is None else _float_vector(strata, "strata")
        if (
            values is None
            or len(values) != n
            or any(
                not np.isfinite(value) or value != int(value) or not 1 <= value <= count
                for value in values
            )
        ):
            raise ValueError("Invalid strata variable")
        codes = [int(value) - 1 for value in values]
    names = (
        _matrix_input_column_names(x)
        if column_names is None
        else tuple(_materialize_labels(column_names, "column_names"))
    )
    if names is not None and (
        len(names) != matrix.shape[1] or any(not isinstance(v, str) for v in names)
    ):
        raise ValueError("column_names must contain one string per design column")
    names = (
        tuple(names) if names is not None else tuple(f"x {i + 1}" for i in range(matrix.shape[1]))
    )
    weight_values = None if weights is None else _float_vector(weights, "weights")
    if weight_values is not None and any(value <= 0 for value in weight_values):
        raise ValueError("Invalid weights, must be >0")
    data = _core.SurvregData(
        time,
        status,
        matrix,
        time2=time2,
        weights=weight_values,
        offset=None if offset is None else _float_vector(offset, "offset"),
        strata=codes,
    )
    return matrix, data, scale_value, count, names


def survreg_fit(
    x: Any,
    y: Any,
    weights: Any = None,
    offset: Any = None,
    init: Any = None,
    controlvals: Any = None,
    dist: Any = "extreme",
    scale: Any = 0,
    nstrat: Any = 1,
    strata: Any = None,
    parms: Any = None,
    assign: Any = None,
    *,
    column_names: Sequence[str] | None = None,
) -> SurvregFitResult:
    """R's bare AFT matrix fitter, including estimated log-scales in coefficients.

    Supply the response on its fitting scale and a base density (``extreme``,
    ``logistic``, ``gaussian`` or ``t``). Numeric response codes are 0 right,
    1 exact, 2 left and 3 interval censored. No response transform, status
    recoding, missing-row omission or robust variance is applied. ``assign``
    is accepted and unused, as in R. Stratum codes are one-based.
    """
    matrix, data, scale_value, count, names = _prepared_data(
        x, y, weights, offset, scale, nstrat, strata, column_names
    )
    if scale_value == 0:
        names += ("Log(scale)",) * count
    control = _resolve_control(controlvals, {})
    result = _core.survreg_fit_raw(
        data,
        _fitting_distribution(dist, parms),
        init=None if init is None else _float_vector(init, "init"),
        scale=scale_value,
        control=control,
        nstrat=count,
    )
    if control.iter_max > 1 and not result.converged:
        _warn_outside_package("Ran out of iterations and did not converge", RuntimeWarning)
    mean_only = matrix.shape[1] == 1 and bool(np.all(matrix == 1))
    icoef_names = names if mean_only else ("Intercept",) + ("Log(scale)",) * (len(result.icoef) - 1)
    return SurvregFitResult(result, names, icoef_names, None if result.rescaled else names)
