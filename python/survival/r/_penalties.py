"""Design matrices for R's ridge, pspline and frailty formula terms.

The formula parser supplies column names and literal options. Numerical basis
construction, penalty selection and optimization remain in the Rust engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .. import _survival as _core
from ._coerce import _integer_scalar, _strata_level_sort_key, _strata_value_label
from ._types import _CovariateTerm, _PenaltyDesignTerm

PENALTY_FUNCTIONS = (
    "ridge",
    "pspline",
    "frailty",
    "frailty.gamma",
    "frailty.gaussian",
    "frailty.t",
)


def fit_penalty(
    term: _CovariateTerm,
    columns: Sequence[str],
    values: Mapping[str, Sequence[Any]],
    options: Mapping[str, Any],
    levels: Sequence[Any] | None = None,
) -> _PenaltyDesignTerm:
    kind = term.call.split("(", 1)[0]
    kwargs = dict(options)
    if kind == "ridge":
        penalty = _core.CoxPenalty.ridge(**kwargs)
        return _PenaltyDesignTerm(
            term, tuple(columns), tuple(f"ridge({name})" for name in columns), penalty
        )
    if len(columns) != 1:
        raise ValueError(f"{kind}() requires exactly one data column")
    column = columns[0]
    if kind == "pspline":
        degree = _integer_scalar(kwargs.pop("degree", 3), "degree")
        boundary = kwargs.pop("Boundary.knots", kwargs.pop("boundary_knots", None))
        x = [float(value) for value in values[column]]
        boundary = (min(x), max(x)) if boundary is None else tuple(boundary)
        if len(boundary) != 2:
            raise ValueError("Boundary.knots must contain two values")
        penalty = _core.CoxPenalty.pspline(**kwargs)
        intercept = kwargs.get("intercept", False)
        names = tuple(
            f"ps({column}){j + 2}" for j in range(0 if intercept else 1, penalty.nterm + degree)
        )
        return _PenaltyDesignTerm(
            term, (column,), names, penalty, degree, boundary, intercept=intercept
        )
    present = set(values[column])
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
    penalty = _core.CoxPenalty.frailty(**kwargs)
    names = (
        (term.call,)
        if penalty.sparse
        else tuple(f"gamma:{_strata_value_label(level)}" for level in groups)
    )
    return _PenaltyDesignTerm(term, (column,), names, penalty, levels=groups)


def penalty_columns(
    spec: _PenaltyDesignTerm, values: Mapping[str, Sequence[Any]]
) -> list[list[float]]:
    if spec.penalty.kind == "ridge":
        return [[float(value) for value in values[column]] for column in spec.columns]
    x = values[spec.columns[0]]
    if spec.penalty.kind == "pspline":
        basis = _core.pspline_basis(
            [float(value) for value in x], spec.penalty.nterm, spec.degree, spec.boundary
        ).basis
        first = 0 if spec.intercept else 1
        return [list(column) for column in zip(*(row[first:] for row in basis), strict=True)]
    lookup = {level: i + 1 for i, level in enumerate(spec.levels)}
    if any(value not in lookup for value in x):
        raise ValueError(f"newdata column {spec.columns[0]!r} contains an unknown frailty level")
    if spec.penalty.sparse:
        return [[float(lookup[value]) for value in x]]
    return [[float(value == level) for value in x] for level in spec.levels]
