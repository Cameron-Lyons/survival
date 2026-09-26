"""Design matrices for R's ridge, pspline and frailty formula terms.

The formula parser supplies column names and literal options.  R evaluates these
functions in ``model.frame``, before ``subset`` and ``na.action``: the pspline knots,
the ridge variances and the frailty groups come from that data, the design columns from
the rows fitted.  Numerical basis construction, penalty selection and optimization remain
in the Rust engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _float_vector,
    _integer_scalar,
    _is_missing_value,
    _normalize_bool_option,
    _strata_level_sort_key,
    _strata_value_label,
)
from ._types import _CovariateTerm, _PenaltyDesignTerm

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


def fit_penalty(
    term: _CovariateTerm,
    columns: Sequence[str],
    values: Mapping[str, Sequence[Any]],
    options: Mapping[str, Any],
    levels: Sequence[Any] | None = None,
) -> _PenaltyDesignTerm:
    """Evaluate a penalty function on its data *values* before ``subset`` and ``na.action``;
    *levels* are the categories of a factor frailty column."""

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
    if boundary is None:
        observed = _observed(x)
        knots = (min(observed), max(observed))
    else:
        given = _float_vector(boundary, "Boundary.knots")
        if len(given) != 2 or not given[0] < given[1]:
            raise ValueError("Invalid values for Boundary.knots")
        knots = (given[0], given[1])
    nterm = penalty.nterm
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
    penalty = _core.CoxPenalty.frailty(n=len(x), **kwargs)
    if penalty.sparse:
        names: tuple[str, ...] = (term.call,)
    else:
        prefix = _FRAILTY_PREFIX[penalty.distribution]
        names = tuple(f"{prefix}:{_strata_value_label(level)}" for level in groups)
    return _PenaltyDesignTerm(term, (column,), names, penalty, levels=groups)


def _observed(values: Sequence[Any]) -> list[float]:
    """``x[!is.na(x)]`` as numbers."""

    return [float(value) for value in values if not _is_missing_value(value)]


def _r_var(values: Sequence[float]) -> float:
    """R's ``var(x)``: the ``n - 1`` denominator, ``NA`` below two values."""

    n = len(values)
    if n < 2:
        return math.nan
    mean = math.fsum(values) / n
    return math.fsum((value - mean) ** 2 for value in values) / (n - 1)


def _pspline_combine(combine: Any, ncol: int, intercept: bool) -> tuple[int, ...]:
    """pspline.R's checks of ``combine``: the group of each of the ``ncol`` basis columns,
    with the first column (dropped unless ``intercept``) coded 0 in front of them."""

    codes = _float_vector(combine, "combine")
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


def penalty_columns(
    spec: _PenaltyDesignTerm, values: Mapping[str, Sequence[Any]]
) -> list[list[float]]:
    # an unpenalized basis is pspline(penalty = FALSE)
    kind = "pspline" if spec.penalty is None else spec.penalty.kind
    if kind == "ridge":
        return [[float(value) for value in values[column]] for column in spec.columns]
    x = values[spec.columns[0]]
    if kind == "pspline":
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
