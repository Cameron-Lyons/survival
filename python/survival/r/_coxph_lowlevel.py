"""Matrix interfaces for R's coxph.fit, agreg.fit and agexact.fit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _control_mapping,
    _encode_labels,
    _float_vector,
    _is_missing_value,
    _materialize_labels,
    _matrix_input_column_names,
    _normalize_bool_option,
    _warn_outside_package,
)
from ._coxph import _cox_fit_diagnostic_messages, coxph_control
from ._surv import Surv


@dataclass(frozen=True)
class CoxFitResult:
    """Bare fit components with R's output shapes and explicit row/column names.

    The native result retains no design matrix or response data. It owns the
    numerical arrays; property access returns copies.
    Null models have no coefficient, variance, iteration or score components.
    """

    _fit: _core.CoxphFitResult = field(repr=False)
    fitter: str
    method: str
    coefficient_names: tuple[str, ...] | None
    row_names: tuple[str, ...] | None
    null_model: bool

    @property
    def coefficients(self) -> list[float] | None:
        return None if self.null_model else self._fit.coefficients

    @property
    def var(self) -> list[list[float]] | None:
        return None if self.null_model else self._fit.var

    @property
    def loglik(self) -> list[float]:
        values = self._fit.loglik
        return [values[1 if self.fitter == "agreg" else 0]] if self.null_model else values

    @property
    def score(self) -> float | None:
        return None if self.null_model else self._fit.score

    @property
    def iter(self) -> int | None:
        return None if self.null_model else self._fit.iter

    @property
    def linear_predictors(self) -> list[float] | list[list[float]]:
        values = self._fit.linear_predictors
        return [[value] for value in values] if self.fitter == "agexact" else values

    @property
    def residuals(self) -> list[float] | None:
        return self._fit.residuals

    @property
    def means(self) -> list[float] | None:
        return None if self.null_model else self._fit.means

    @property
    def first(self) -> list[float] | None:
        return self._fit.first if self.fitter == "agreg" and not self.null_model else None

    @property
    def info(self) -> dict[str, int] | None:
        values = self._fit.info
        return (
            dict(zip(("rank", "rescale", "step halving", "convergence"), values, strict=True))
            if self.fitter == "agreg" and not self.null_model and values is not None
            else None
        )

    @property
    def classes(self) -> tuple[str, ...]:
        if self.fitter == "agexact":
            return ()
        return ("coxph.null", "coxph") if self.null_model else ("coxph",)


def _design(x: Any, n: int, kind: str) -> np.ndarray:
    if isinstance(x, Mapping):
        values = np.column_stack(list(x.values())) if x else np.empty((n, 0))
    else:
        values = x.to_numpy() if hasattr(x, "to_numpy") else x
    matrix = np.asarray(values)
    if matrix.ndim == 1 and matrix.size == 0:
        matrix = np.empty((n, 0))
    elif matrix.ndim == 1 and kind == "coxph":
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("Invalid formula for cox fitting function")
    if matrix.shape[0] != n:
        raise ValueError("x and y have different numbers of rows")
    if matrix.dtype.kind not in "biuf":
        raise TypeError("x must be a numeric matrix")
    return matrix.astype(np.float64, copy=False)


def _fit(
    kind: str,
    x: Any,
    y: Surv,
    strata: Any,
    offset: Any,
    init: Any,
    control: Any,
    weights: Any,
    method: Any,
    rownames: Any,
    resid: Any,
    nocenter: Any,
    column_names: Any,
) -> CoxFitResult:
    if not isinstance(y, Surv):
        raise TypeError("y must be a Surv object")
    allowed = (
        {"counting"} if kind == "agreg" else {"right"} if kind == "coxph" else {"right", "counting"}
    )
    if y.type not in allowed:
        raise ValueError(f"{kind}_fit requires {' or '.join(sorted(allowed))} survival data")
    if any(value is None for value in y.event):
        raise ValueError("y contains missing status values")
    n = len(y)
    matrix = _design(x, n, kind)
    p = matrix.shape[1]
    options = coxph_control(**({} if control is None else _control_mapping(control, "control")))
    residuals = _normalize_bool_option(resid, "resid")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if method not in {"efron", "breslow", "exact"}:
        raise ValueError("method must be 'efron', 'breslow' or 'exact'")
    # The bare R fitters use the test method == 'efron'. Only agexact.fit
    # selects exact likelihood; even the label 'exact' otherwise means Breslow.
    engine_method = "exact" if kind == "agexact" else "efron" if method == "efron" else "breslow"
    strata_values = None if strata is None else _materialize_labels(strata, "strata")
    if strata_values:
        if len(strata_values) != n:
            raise ValueError("strata and y have different numbers of rows")
        if any(_is_missing_value(value) for value in strata_values):
            raise ValueError("strata contains missing values")
        strata_codes = _encode_labels(strata_values, "strata")
    else:
        strata_codes = None
    arrays = {}
    for name, value in (("offset", offset), ("weights", weights)):
        values = None if value is None else _float_vector(value, name)
        if values is not None and len(values) != n:
            raise ValueError(f"{name} and y have different numbers of rows")
        arrays[name] = values
    names = (
        _matrix_input_column_names(x)
        if column_names is None
        else tuple(_materialize_labels(column_names, "column_names"))
    )
    if names is not None and (len(names) != p or any(not isinstance(name, str) for name in names)):
        raise ValueError("column_names must contain one string per design column")
    row_names = None if rownames is None else tuple(_materialize_labels(rownames, "rownames"))
    if row_names is not None and (
        len(row_names) != n or any(not isinstance(name, str) for name in row_names)
    ):
        raise ValueError("rownames must contain one string per response row")
    initial = None if init is None else _float_vector(init, "init")
    if kind == "coxph" and (p == 0 or initial == []):
        initial = None
    entry = list(y.start) if y.start is not None else [0.0] * n if kind == "agexact" else None
    fitted = _core.coxph_fit_raw(
        y.time,
        [int(value) for value in y.event if value is not None],
        matrix,
        entry=entry,
        strata=strata_codes,
        weights=arrays["weights"],
        offset=arrays["offset"],
        method=engine_method,
        init=initial,
        iter_max=options["iter.max"],
        eps=options["eps"],
        toler_chol=options["toler.chol"],
        nocenter=None if nocenter is None else _float_vector(nocenter, "nocenter"),
        resid=residuals,
    )
    for message in _cox_fit_diagnostic_messages(
        fitted, options["iter.max"], options["eps"], options["toler.inf"], offset_centered=False
    ):
        _warn_outside_package(message, RuntimeWarning)
    return CoxFitResult(
        fitted,
        kind,
        "coxph" if kind == "agexact" else method,
        names,
        row_names,
        p == 0 and kind != "agexact",
    )


def coxph_fit(
    x: Any,
    y: Surv,
    strata: Any = None,
    offset: Any = None,
    init: Any = None,
    control: Any = None,
    weights: Any = None,
    method: str = "efron",
    rownames: Sequence[str] | None = None,
    resid: Any = True,
    nocenter: Any = None,
    *,
    column_names: Sequence[str] | None = None,
) -> CoxFitResult:
    """R's bare right-censored fitter: original offsets and optional residuals.

    As in R, this fitter uses Efron only for ``method='efron'`` and Breslow
    otherwise. Use ``agexact_fit`` for exact likelihood, or ``coxph`` for a
    full model with high-level tie selection, time fixing and diagnostics.
    """
    return _fit(
        "coxph",
        x,
        y,
        strata,
        offset,
        init,
        control,
        weights,
        method,
        rownames,
        resid,
        nocenter,
        column_names,
    )


def agreg_fit(
    x: Any,
    y: Surv,
    strata: Any = None,
    offset: Any = None,
    init: Any = None,
    control: Any = None,
    weights: Any = None,
    method: str = "efron",
    rownames: Sequence[str] | None = None,
    resid: Any = True,
    nocenter: Any = None,
    *,
    column_names: Sequence[str] | None = None,
) -> CoxFitResult:
    """R's bare counting-process fitter, including the final score and iteration info."""
    return _fit(
        "agreg",
        x,
        y,
        strata,
        offset,
        init,
        control,
        weights,
        method,
        rownames,
        resid,
        nocenter,
        column_names,
    )


def agexact_fit(
    x: Any,
    y: Surv,
    strata: Any = None,
    offset: Any = None,
    init: Any = None,
    control: Any = None,
    weights: Any = None,
    method: str = "exact",
    rownames: Sequence[str] | None = None,
    resid: Any = True,
    nocenter: Any = None,
    *,
    column_names: Sequence[str] | None = None,
) -> CoxFitResult:
    """R's bare exact fitter, using entry zero for a right-censored response.

    R's single-column predictor matrix and historical ``method='coxph'`` result
    label are preserved. Unit weights are required; ``method`` does not select
    the numerical algorithm.
    """
    return _fit(
        "agexact",
        x,
        y,
        strata,
        offset,
        init,
        control,
        weights,
        method,
        rownames,
        resid,
        nocenter,
        column_names,
    )
