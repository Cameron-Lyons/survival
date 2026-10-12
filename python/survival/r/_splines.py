"""R's matrix-valued basis terms ``ns()``, ``bs()``, ``poly()`` and ``nsk()``.

A term is evaluated on its data column before ``subset`` and ``na.action``, as
``model.frame`` does, and keeps the knots, boundary and polynomial coefficients that
R's ``makepredictcall`` methods store, so new data reuse the fitted basis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import _floats_or_nan, _normalize_bool_option, _warn_outside_package
from ._types import _CovariateTerm, _PenaltyDesignTerm

BASIS_FUNCTIONS = ("ns", "bs", "poly", "nsk")

_BASIS_FORMALS = {
    "ns": ("x", "df", "knots", "intercept", "Boundary.knots"),
    "bs": ("x", "df", "knots", "degree", "intercept", "Boundary.knots", "warn.outside"),
    "poly": ("x", "degree", "coefs", "raw", "simple"),
    "nsk": ("x", "df", "knots", "intercept", "b", "Boundary.knots"),
}


def basis_arguments(call: str) -> tuple[str, dict[str, Any]]:
    """The data column and literal options of a basis call, matched as R matches them."""

    from ._formula import (
        _formula_name,
        _formula_response_parts,
        _literal_vector,
        _match_arguments,
        _parse_formula_literal,
        _unsupported_formula_name,
    )

    function, _sep, inner = call.partition("(")
    arguments = _match_arguments(
        function, _formula_response_parts(inner[:-1]), _BASIS_FORMALS[function]
    )
    if "x" not in arguments:
        raise ValueError(f"{function}() requires a variable")
    column, quoted = _formula_name(arguments.pop("x"))
    if _unsupported_formula_name(column, quoted):
        raise ValueError(f"{function}() requires a data column; transform the column in the data")
    options: dict[str, Any] = {}
    for name, value in arguments.items():
        if value == "NULL":
            options[name] = None
            continue
        try:
            options[name] = _parse_formula_literal(value)
        except ValueError:
            values = _literal_vector(value)
            options[name] = values[0] if len(values) == 1 else values
    return column, options


def _count(value: Any, name: str) -> int | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return int(number)


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    values = value if isinstance(value, list | tuple) else [value]
    return [float(item) for item in values]


def _boundary(value: Any) -> tuple[float, float] | None:
    values = _vector(value)
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("Boundary.knots must have two values")
    low, high = sorted(values)
    return (low, high)


def _r_mean(values: np.ndarray) -> float:
    """R's ``mean``: a long-double sum refined by the mean residual."""

    array = values.astype(np.longdouble)
    mean = np.sum(array) / len(array)
    mean += np.sum(array - mean) / len(array)
    return float(mean)


def _poly_fit(x: np.ndarray, degree: int) -> tuple[np.ndarray, list[float], list[float]]:
    """``poly(x, degree)``: orthonormal polynomial columns and R's ``coefs``."""

    if np.isnan(x).any():
        raise ValueError("missing values are not allowed in 'poly'")
    if degree >= len(np.unique(x)):
        raise ValueError("'degree' must be less than number of unique points")
    xbar = _r_mean(x)
    centered = x - xbar
    q, r = np.linalg.qr(centered[:, None] ** np.arange(degree + 1))
    z = q * np.diag(r)
    norm2 = (z**2).sum(axis=0)
    alpha = ((centered[:, None] * z**2).sum(axis=0) / norm2 + xbar)[:degree]
    full_norm2 = [1.0, *norm2.tolist()]
    return z[:, 1:] / np.sqrt(norm2[1:]), alpha.tolist(), full_norm2


def _poly_predict(x: np.ndarray, alpha: Sequence[float], norm2: Sequence[float]) -> np.ndarray:
    degree = len(alpha)
    z = np.ones((len(x), degree + 1))
    z[:, 1] = x - alpha[0]
    for i in range(2, degree + 1):
        z[:, i] = (x - alpha[i - 1]) * z[:, i - 1] - (norm2[i] / norm2[i - 1]) * z[:, i - 2]
    return z[:, 1:] / np.sqrt(np.asarray(norm2[2:]))


def _warn_beyond(x: np.ndarray, boundary: tuple[float, float] | None) -> None:
    if boundary is not None and ((x < boundary[0]) | (x > boundary[1])).any():
        _warn_outside_package(
            "some 'x' values beyond boundary knots may cause ill-conditioned bases"
        )


def _columns(matrix: np.ndarray) -> list[list[float]]:
    return matrix.T.tolist()


def _native(result: Any) -> np.ndarray:
    return np.asarray(result.basis, dtype=float).reshape(result.n_rows, result.n_cols)


def fit_basis(
    term: _CovariateTerm, column: str, values: Sequence[Any], options: dict[str, Any]
) -> _PenaltyDesignTerm:
    """Evaluate a basis call on all of its column's *values*."""

    from ._expression import _expression_label

    if term.call is None:
        raise ValueError("a basis term must have a function call")
    function = term.call.split("(", 1)[0]
    x = np.asarray(_floats_or_nan(values), dtype=float)
    intercept = _normalize_bool_option(options.get("intercept", False), "intercept")
    knots = _vector(options.get("knots"))
    boundary = _boundary(options.get("Boundary.knots"))
    df = _count(options.get("df"), "df")
    if function == "poly":
        if options.get("coefs") is not None:
            raise ValueError("poly(coefs=) is not supported in formulas")
        degree = _count(options.get("degree", 1), "degree") or 0
        if degree < 1:
            raise ValueError("'degree' must be at least 1")
        raw = _normalize_bool_option(options.get("raw", False), "raw")
        if raw:
            matrix = x[:, None] ** np.arange(1, degree + 1)
            coefs = None
        else:
            matrix, alpha, norm2 = _poly_fit(x, degree)
            coefs = (tuple(alpha), tuple(norm2))
        spec_knots: tuple[float, ...] = ()
        spec_boundary: tuple[float, float] | None = None
    else:
        degree = 3
        if function == "bs":
            degree = _count(options.get("degree", 3), "degree") or 0
            if _normalize_bool_option(options.get("warn.outside", True), "warn.outside"):
                _warn_beyond(x, boundary)
            result = _core.bs_basis(x.tolist(), df, knots, degree, intercept, boundary)
        elif function == "ns":
            result = _core.ns_basis(x.tolist(), df, knots, intercept, boundary)
        else:
            from ._misc import nsk

            nsk_options = (
                {"Boundary_knots": options["Boundary.knots"]} if "Boundary.knots" in options else {}
            )
            result = nsk(x.tolist(), df, knots, intercept, options.get("b", 0.05), **nsk_options)
        matrix = _native(result)
        spec_knots = tuple(result.knots)
        low, high = result.boundary_knots
        spec_boundary = (float(low), float(high))
        coefs = None
        raw = False
    label = _expression_label(term.call)
    names = tuple(f"{label}{j}" for j in range(1, matrix.shape[1] + 1))
    return _PenaltyDesignTerm(
        term,
        (column,),
        names,
        None,
        degree,
        spec_boundary,
        intercept=intercept,
        basis=function,
        knots=spec_knots,
        coefs=coefs,
        raw=raw,
    )


def basis_columns(spec: _PenaltyDesignTerm, values: Sequence[Any]) -> list[list[float]]:
    """The fitted basis of *spec* at new *values*, as R's ``makepredictcall`` rebuilds it."""

    x = np.asarray(_floats_or_nan(values), dtype=float)
    if not len(x):
        return [[] for _ in spec.names]
    if spec.basis == "poly":
        degree = len(spec.names)
        if spec.raw:
            return _columns(x[:, None] ** np.arange(1, degree + 1))
        if spec.coefs is None:
            raise ValueError("poly prediction requires its fitted coefficients")
        return _columns(_poly_predict(x, *spec.coefs))
    if spec.boundary is None:
        raise ValueError("spline prediction requires its boundary knots")
    knots = list(spec.knots)
    if spec.basis == "bs":
        _warn_beyond(x, spec.boundary)
        result = _core.bs_basis(x.tolist(), None, knots, spec.degree, spec.intercept, spec.boundary)
    elif spec.basis == "ns":
        result = _core.ns_basis(x.tolist(), None, knots, spec.intercept, spec.boundary)
    else:
        result = _core.NaturalSplineKnot(knots or None, spec.boundary, None, spec.intercept).basis(
            x.tolist()
        )
    return _columns(_native(result))
