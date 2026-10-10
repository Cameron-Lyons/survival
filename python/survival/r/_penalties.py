"""Design matrices for R's ridge, pspline and frailty formula terms.

The formula parser supplies column names and literal options.  R evaluates these
functions in ``model.frame``, before ``subset`` and ``na.action``: the pspline knots,
the ridge variances and the frailty groups come from that data, the design columns from
the rows fitted.  Numerical basis construction, penalty selection and optimization remain
in the Rust engine.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _categories,
    _coerce_array_like,
    _float_or_nan,
    _float_vector,
    _integer_scalar,
    _is_missing_value,
    _materialize_1d,
    _materialize_labels,
    _matrix_input_column_names,
    _normalize_bool_option,
    _scalar_or_vector,
    _strata_level_sort_key,
    _strata_value_label,
)
from ._types import CoxPenaltyBasis, _CovariateTerm, _PenaltyDesignTerm

PENALTY_FUNCTIONS = (
    "ridge",
    "pspline",
    "frailty",
    "frailty.gamma",
    "frailty.gaussian",
    "frailty.t",
)

# The coefficient names of a dense frailty term: frailty.gamma.R, frailty.gaussian.R and
# frailty.t.R paste these before the levels.
_FRAILTY_PREFIX = {"gamma": "gamma", "gaussian": "gauss", "t": "t"}


@dataclass(frozen=True)
class RidgeResult(CoxPenaltyBasis):
    """A ridge basis and native penalty, with variances from the original inputs.

    ``scale_values`` excludes missing values separately in each column, before
    any subsequent row selection. ``column_names`` gives complete coefficient
    labels; Python cannot recover R's unevaluated argument expressions.
    """

    scale_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class FrailtyResult(CoxPenaltyBasis):
    """A sparse group column or dense indicator basis with its native penalty.

    ``levels`` retains the used factor levels in R order. ``codes`` contains
    their one-based indices, with ``None`` for missing observations. Dense
    coefficient labels include the distribution and original group labels.
    """

    levels: tuple[str, ...] = ()
    codes: tuple[int | None, ...] = ()


def ridge(
    *x: Any,
    theta: float | None = None,
    df: float | None = None,
    eps: float = 0.1,
    scale: bool = True,
    column_names: Sequence[str] | None = None,
) -> RidgeResult:
    """Construct R's ``ridge`` basis and penalty for a Cox time-transform.

    Inputs are numeric vectors or matrices combined like R's ``cbind``. Their
    sample variances are computed now, so later subsetting retains the original
    scaling. A supplied ``theta`` fixes the penalty; otherwise ``df`` defaults
    to half the number of columns. Explicit ``column_names`` are full fitted
    labels; named matrix inputs use ``ridge(name)`` and other columns use
    ``ridge(x1)``, ``ridge(x2)``, ... .
    """

    if not x:
        raise ValueError("ridge requires at least one numeric input")
    pieces: list[tuple[list[list[float]], int, bool, tuple[str, ...] | None]] = []
    for value in x:
        names = _matrix_input_column_names(value)
        try:
            values = _coerce_array_like(value, "x")
        except TypeError:
            if isinstance(value, str | bytes):
                raise TypeError("ridge inputs must be numeric") from None
            values = [value]
        matrix = (
            isinstance(value, Mapping)
            or getattr(value, "ndim", None) == 2
            or (
                bool(values)
                and (
                    (isinstance(values[0], Sequence) and not isinstance(values[0], str))
                    or getattr(values[0], "ndim", None) == 1
                )
            )
        )
        if matrix:
            values = _coerce_array_like(value, "x")
            width = len(values[0]) if values else getattr(value, "shape", (0, 0))[1]
            if any(len(row) != width for row in values):
                raise ValueError("ridge matrix rows must have the same length")
            rows = [[_ridge_number(item) for item in row] for row in values]
        else:
            width = 1
            rows = [[_ridge_number(item)] for item in values]
        pieces.append((rows, width, matrix, names))
    matrix_sizes = {len(rows) for rows, _, matrix, _ in pieces if matrix}
    if len(matrix_sizes) > 1:
        raise ValueError("ridge matrices must have the same number of rows")
    n = next(iter(matrix_sizes)) if matrix_sizes else max(len(rows) for rows, *_ in pieces)
    columns: list[list[float]] = []
    labels: list[str] = []
    for rows, width, matrix, names in pieces:
        if not rows and not matrix and n:
            continue
        if rows and not matrix and (len(rows) > n or n % len(rows)):
            warnings.warn(
                "number of rows of result is not a multiple of vector length",
                RuntimeWarning,
                stacklevel=2,
            )
        for col in range(width):
            columns.append([rows[row % len(rows)][col] for row in range(n)])
            labels.append(
                f"ridge({names[col]})" if names is not None else f"ridge(x{len(labels) + 1})"
            )
    if not columns:
        raise ValueError("ridge requires at least one numeric column")
    if column_names is not None:
        labels = [str(name) for name in _materialize_1d(column_names, "column_names")]
        if len(labels) != len(columns):
            raise ValueError("ridge column_names must match the number of columns")
    variances = tuple(_r_var(_observed(column)) for column in columns)
    penalty = _core.CoxPenalty.ridge(
        theta=theta,
        df=len(columns) / 2 if theta is None and df is None else df,
        eps=eps,
        scale=_normalize_bool_option(scale, "scale"),
        scale_values=variances,
    )
    return RidgeResult(
        basis=[[column[row] for column in columns] for row in range(n)],
        penalty=penalty,
        column_names=tuple(labels),
        scale_values=variances,
    )


def _ridge_number(value: Any) -> float:
    if isinstance(value, str | bytes):
        raise TypeError("ridge inputs must be numeric")
    try:
        return _float_or_nan(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("ridge inputs must be numeric") from exc


def frailty(
    x: Any,
    distribution: str = "gamma",
    *,
    sparse: bool | None = None,
    theta: float | None = None,
    df: float | None = None,
    eps: float | None = None,
    method: str | None = None,
    tdf: float = 5.0,
    caic: bool = False,
    init: Any | None = None,
) -> FrailtyResult:
    """Construct R's ``frailty`` basis and native gamma, Gaussian or t penalty.

    Factor inputs retain their declared order and drop unused levels. Other
    inputs use R's numeric/logical or character ordering. ``sparse`` defaults
    to more than five observed groups; dense bases keep one indicator column
    for every group. The result can be returned directly from ``coxph(tt=)``.
    """

    declared = _categories(x)
    if x is None:
        raise ValueError("x is required")
    try:
        values = _materialize_labels(x, "x")
    except TypeError:
        values = [x]
    present = {value for value in values if not _is_missing_value(value)}
    groups = tuple(
        sorted(present, key=_strata_level_sort_key)
        if declared is None
        else (level for level in declared if level in present)
    )
    lookup = {level: index + 1 for index, level in enumerate(groups)}
    if any(value not in lookup for value in present):
        raise ValueError("frailty input contains values outside its factor levels")
    codes = tuple(None if _is_missing_value(value) else lookup[value] for value in values)
    sparse_value = len(groups) > 5 if sparse is None else _normalize_bool_option(sparse, "sparse")
    if not sparse_value and len(groups) < 2:
        raise ValueError("not enough degrees of freedom to define contrasts")
    penalty = _frailty_penalty(
        len(values),
        distribution=distribution,
        sparse=sparse_value,
        theta=theta,
        df=df,
        eps=eps,
        method=method,
        tdf=tdf,
        caic=_normalize_bool_option(caic, "caic"),
        init=init,
    )
    levels = tuple(_strata_value_label(level) for level in groups)
    family = penalty.distribution
    if family is None:
        raise RuntimeError("native frailty penalty has no distribution")
    basis = (
        [[math.nan if code is None else float(code)] for code in codes]
        if sparse_value
        else [
            [
                math.nan if code is None else float(code == group)
                for group in range(1, len(groups) + 1)
            ]
            for code in codes
        ]
    )
    return FrailtyResult(
        basis=basis,
        penalty=penalty,
        column_names=None
        if sparse_value
        else tuple(f"{_FRAILTY_PREFIX[family]}:{level}" for level in levels),
        levels=levels,
        codes=codes,
    )


def _frailty_penalty(n: int, **options: Any) -> _core.CoxPenalty:
    init = options.pop("init", None)
    penalty = _core.CoxPenalty.frailty(n=n, **options)
    family = penalty.distribution
    if family is None:
        raise RuntimeError("native frailty penalty has no distribution")
    methods = {
        "gamma": ("em", "aic", "df", "fixed"),
        "gaussian": ("reml", "aic", "df", "fixed"),
        "t": ("aic", "df", "fixed"),
    }[family]
    method = options.get("method")
    theta, df = options.get("theta"), options.get("df")
    method_value = (
        next(name for name in methods if name.startswith(method.lower()))
        if method is not None
        else "fixed"
        if theta is not None
        else "aic"
        if df == 0 and family != "gamma"
        else "df"
        if df is not None
        else methods[0]
    )
    # AIC prepends init=(.1, 1) before ... . Gamma EM's c(list(eps), ...)
    # flattens a vector into init1/init2, so $init finds no usable vector.
    # Other fixed/df searches do not consume init either.
    effective_init = (
        [0.1, 1.0]
        if method_value == "aic"
        else _float_vector(_scalar_or_vector(init, "init"), "init")
        if init is not None and method_value in {"em", "reml"}
        else None
    )
    if (
        family == "gamma"
        and method_value == "em"
        and effective_init is not None
        and len(effective_init) != 1
    ):
        effective_init = None
    return (
        penalty
        if effective_init is None
        else _core.CoxPenalty.frailty(n=n, init=effective_init, **options)
    )


def frailty_gamma(x: Any, **kwargs: Any) -> FrailtyResult:
    """The gamma-family form of :func:`frailty` (R's ``frailty.gamma``)."""
    return frailty(x, distribution="gamma", **kwargs)


def frailty_gaussian(x: Any, **kwargs: Any) -> FrailtyResult:
    """The Gaussian-family form of :func:`frailty` (R's ``frailty.gaussian``)."""
    return frailty(x, distribution="gaussian", **kwargs)


def frailty_t(x: Any, **kwargs: Any) -> FrailtyResult:
    """The Student-t-family form of :func:`frailty` (R's ``frailty.t``)."""
    return frailty(x, distribution="t", **kwargs)


def _penalty_control(
    options: Mapping[str, Any],
    iteration: int,
    old: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One R/Python crossing per search step; all numerical work stays in Rust."""
    controller = _core.PenaltyController(**options)
    if iteration == 0:
        state = controller.initial()
    else:
        if old is None:
            raise ValueError("old state is required after iteration zero")
        state = controller.step(_core.PenaltyControlState(**old), iteration, **(inputs or {}))
    history = state.history
    return {
        "theta": state.theta,
        "done": state.done,
        "row": history[-1] if history else None,
        "columns": controller.columns,
        "c_loglik": state.c_loglik,
        "half": state.half,
        "theta_history_index": state.theta_history_index,
    }


def fit_penalty(
    term: _CovariateTerm,
    columns: Sequence[str],
    values: Mapping[str, Sequence[Any]],
    options: Mapping[str, Any],
    levels: Sequence[Any] | None = None,
) -> _PenaltyDesignTerm:
    """Evaluate a penalty function on its data *values* before ``subset`` and ``na.action``;
    *levels* are the categories of a factor frailty column."""

    if term.call is None:
        raise ValueError("a penalty term must have a function call")
    kind = term.call.split("(", 1)[0]
    kwargs = dict(options)
    if kind == "ridge":
        # ridge.R: vars <- apply(x, 2, function(z) var(z[!is.na(z)]))
        scale_values = [_r_var(_observed(values[name])) for name in columns]
        penalty = _core.CoxPenalty.ridge(scale_values=scale_values, **kwargs)
        return _PenaltyDesignTerm(
            term, tuple(columns), tuple(f"ridge({name})" for name in columns), penalty
        )
    if len(columns) != 1:
        raise ValueError(f"{kind}() requires exactly one data column")
    column = columns[0]
    if kind == "pspline":
        return _fit_pspline(term, column, values[column], kwargs)
    return _fit_frailty(term, kind, column, values[column], kwargs, levels)


def _fit_pspline(
    term: _CovariateTerm, column: str, x: Sequence[Any], kwargs: dict[str, Any]
) -> _PenaltyDesignTerm:
    """pspline.R: the knots span ``range(x[!is.na(x)])`` unless ``Boundary.knots`` is given;
    ``combine`` sums basis columns and ``penalty = FALSE`` leaves an ordinary matrix term."""

    degree = _integer_scalar(kwargs.pop("degree", 3), "degree")
    boundary = kwargs.pop("Boundary.knots", kwargs.pop("boundary_knots", None))
    combine = kwargs.pop("combine", None)
    penalized = _normalize_bool_option(kwargs.pop("penalty", True), "penalty")
    intercept = _normalize_bool_option(kwargs.pop("intercept", False), "intercept")
    penalty = _core.CoxPenalty.pspline(intercept=intercept, **kwargs)
    knots = _pspline_boundary(boundary, _observed(x))
    nterm = penalty.nterm
    if nterm is None:
        raise RuntimeError("native pspline penalty has no basis dimension")
    groups = None if combine is None else _pspline_combine(combine, nterm + degree, intercept)
    nvar = nterm + degree if groups is None else len(set(groups))
    ncol = nvar if intercept else nvar - 1
    if penalized:
        # ps(x)1..nvar, or ps(x)3..nvar+1 after the first column is dropped
        first = 1 if intercept else 3
        names = tuple(f"ps({column}){first + j}" for j in range(ncol))
    else:
        # model.matrix numbers the columns of an unnamed matrix term
        names = tuple(f"{term.call}{j}" for j in range(1, ncol + 1))
    return _PenaltyDesignTerm(
        term,
        (column,),
        names,
        penalty if penalized else None,
        degree,
        knots,
        intercept=intercept,
        nterm=nterm,
        combine=groups,
    )


def _fit_frailty(
    term: _CovariateTerm,
    kind: str,
    column: str,
    x: Sequence[Any],
    kwargs: dict[str, Any],
    levels: Sequence[Any] | None,
) -> _PenaltyDesignTerm:
    """frailty.gamma/gaussian/t: the groups are the levels of ``factor(x)``, their number
    sets the ``sparse`` default and ``length(x)`` the start of the ``df`` search."""

    present = {value for value in x if not _is_missing_value(value)}
    groups = tuple(
        sorted(present, key=_strata_level_sort_key)
        if levels is None
        else (level for level in levels if level in present)
    )
    if "dist" in kwargs:
        if "distribution" in kwargs:
            raise ValueError("use only one of dist or distribution")
        kwargs["distribution"] = kwargs.pop("dist")
    if "." in kind:
        kwargs.setdefault("distribution", kind.split(".", 1)[1])
    kwargs.setdefault("sparse", len(groups) > 5)
    penalty = _frailty_penalty(len(x), **kwargs)
    if term.call is None:
        raise ValueError("a penalty term must have a function call")
    distribution = penalty.distribution
    if distribution is None:
        raise RuntimeError("native frailty penalty has no distribution")
    if penalty.sparse:
        names: tuple[str, ...] = (term.call,)
    else:
        prefix = _FRAILTY_PREFIX[distribution]
        names = tuple(f"{prefix}:{_strata_value_label(level)}" for level in groups)
    return _PenaltyDesignTerm(term, (column,), names, penalty, levels=groups)


def _observed(values: Sequence[Any]) -> list[float]:
    """``x[!is.na(x)]`` as numbers."""

    return [float(value) for value in values if not _is_missing_value(value)]


def _r_var(values: Sequence[float]) -> float:
    """R's ``var(x)``: the ``n - 1`` denominator, ``NA`` below two values."""

    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        return math.nan
    array = np.asarray(values, dtype=np.longdouble)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mean = np.sum(array) / len(values)
        mean += np.sum(array - mean) / len(values)
        return float(np.sum((array - mean) ** 2) / (len(values) - 1))


def _pspline_boundary(boundary: Any, observed: Sequence[float]) -> tuple[float, float]:
    """pspline.R's ``Boundary.knots``: ``range(x)`` over the non-missing values unless
    given, when it must be two increasing numbers."""

    if boundary is None:
        return (min(observed), max(observed))
    given = [float(value) for value in _scalar_or_vector(boundary, "Boundary.knots")]
    if len(given) != 2 or not given[0] < given[1]:
        raise ValueError("Invalid values for Boundary.knots")
    return (given[0], given[1])


def _pspline_combine(combine: Any, ncol: int, intercept: bool) -> tuple[int, ...]:
    """pspline.R's checks of ``combine``: the group of each of the ``ncol`` basis columns,
    with the first column (dropped unless ``intercept``) coded 0 in front of them."""

    codes = _float_vector(_scalar_or_vector(combine, "combine"), "combine")
    if any(not code.is_integer() or code < 0 for code in codes) or any(
        b < a for a, b in pairwise(codes)
    ):
        raise ValueError("combine must be an increasing vector of positive integers")
    groups = tuple(int(code) for code in codes)
    if not intercept:
        groups = (0, *groups)
    if len(groups) != ncol:
        raise ValueError("wrong length for combine")
    return groups


def _combine_basis(rows: Sequence[Sequence[float]], groups: Sequence[int]) -> list[list[float]]:
    """``newx %*% tmat``: the sums of each row over the columns of every group, in
    increasing group order."""

    index = {group: i for i, group in enumerate(sorted(set(groups)))}
    target = [index[group] for group in groups]
    combined = []
    for row in rows:
        sums = [0.0] * len(index)
        for value, i in zip(row, target, strict=True):
            sums[i] += value
        combined.append(sums)
    return combined


def _pspline_cbase(
    nterm: int, degree: int, boundary: tuple[float, float], nvar: int
) -> list[float]:
    """pspline.R's ``cbase``, the centres of the basis functions its printfun regresses
    the coefficients on: ``knots[2:nvar] + (Boundary.knots[1] - knots[1])`` for the knots
    ``Boundary.knots[1] + dx * ((-degree):(nterm - 1))``, where ``nvar`` counts the basis
    columns after ``combine`` and before the first one is dropped."""

    lower, upper = boundary
    dx = (upper - lower) / nterm
    knots = [lower + dx * k for k in range(-degree, nvar - degree)]
    return [knot + (lower - knots[0]) for knot in knots[1:]]


def penalty_columns(
    spec: _PenaltyDesignTerm, values: Mapping[str, Sequence[Any]]
) -> list[list[float]]:
    if spec.kind == "ridge":
        return [[float(value) for value in values[column]] for column in spec.columns]
    x = values[spec.columns[0]]
    if spec.kind == "pspline":
        if spec.boundary is None:
            raise ValueError("spline prediction requires basis boundaries")
        basis = _core.pspline_basis(
            [float(value) for value in x], spec.nterm, spec.degree, spec.boundary
        ).basis
        if spec.combine is not None:
            basis = _combine_basis(basis, spec.combine)
        first = 0 if spec.intercept else 1
        return [list(column) for column in zip(*(row[first:] for row in basis), strict=True)]
    lookup = {level: i + 1 for i, level in enumerate(spec.levels)}
    if any(value not in lookup for value in x):
        raise ValueError(f"newdata column {spec.columns[0]!r} contains an unknown frailty level")
    if spec.penalty.sparse:
        return [[float(lookup[value]) for value in x]]
    return [[float(value == level) for value in x] for level in spec.levels]
