"""Default orthogonal polynomial coding for R's ordered factors."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from ._coerce import _as_character


def _column_norm(values: np.ndarray) -> float:
    """Sequential BLAS norm, scaling large columns before squaring them."""

    medium = large = 0.0
    for value in values:
        absolute = abs(float(value))
        if absolute > 2.0**486:
            scaled = absolute * 2.0**-538
            large += scaled * scaled
        else:
            medium += absolute * absolute
    if large:
        return 2.0**538 * math.sqrt(large + (medium * 2.0**-538) * 2.0**-538)
    return math.sqrt(medium)


def _polynomial_basis(n: int) -> np.ndarray:
    """R's limited-pivot LINPACK QR followed by its rank-limited ``qr.qy``.

    A LAPACK complete Q differs once the Vandermonde loses numerical rank.
    The small (at most 95 square) design is cached by ``_contr_poly``.
    """

    scores = np.arange(1, n + 1, dtype=float) - (n + 1) / 2
    design = np.power(scores[:, None], np.arange(n)[None, :])
    auxiliary = np.asarray([_column_norm(design[:, column]) for column in range(n)])
    original = auxiliary.copy()
    rank = n
    for column in range(n):
        while column < rank and auxiliary[column] < original[column] * 1e-7:
            # dqrdc2 cycles negligible columns to the right, retaining order.
            design[:, column:] = np.roll(design[:, column:], -1, axis=1)
            auxiliary[column:] = np.roll(auxiliary[column:], -1)
            original[column:] = np.roll(original[column:], -1)
            rank -= 1
        if column == n - 1:
            continue
        vector = design[column:, column]
        length = _column_norm(vector)
        if length == 0.0:
            continue
        if vector[0] != 0.0:
            length = math.copysign(length, vector[0])
        vector *= 1 / length
        vector[0] += 1
        for following in range(column + 1, n):
            other = design[column:, following]
            other += (-np.dot(vector, other) / vector[0]) * vector
            if auxiliary[following] != 0.0:
                remainder = max(1 - (abs(other[0]) / auxiliary[following]) ** 2, 0.0)
                auxiliary[following] = (
                    auxiliary[following] * math.sqrt(remainder)
                    if remainder >= 1e-6
                    else _column_norm(other[1:])
                )
        auxiliary[column] = vector[0]
        vector[0] = -length

    # Applying Q to signs of the diagonal gives the normalized result without
    # multiplying by large Vandermonde diagonal entries first.
    basis = np.diag(np.sign(np.diag(design)))
    for column in reversed(range(min(rank, n - 1))):
        if auxiliary[column] == 0.0:
            continue
        vector = design[column:, column].copy()
        vector[0] = auxiliary[column]
        for following in range(n):
            other = basis[column:, following]
            other += (-np.dot(vector, other) / vector[0]) * vector
    basis /= np.linalg.norm(basis, axis=0)
    return basis[:, 1:]


@lru_cache(maxsize=32)
def _contr_poly(n: int) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]:
    """R's ``contr.poly(n)`` at equally spaced level scores, as an owned basis."""

    if n < 2:
        raise ValueError(f"contrasts not defined for {n - 1} degrees of freedom")
    if n > 95:
        raise ValueError(
            "orthogonal polynomials cannot be represented accurately enough "
            f"for {n - 1} degrees of freedom"
        )
    basis = _polynomial_basis(n)
    names = tuple([".L", ".Q", ".C"][: n - 1] + [f"^{degree}" for degree in range(4, n)])
    return tuple(tuple(float(value) for value in row) for row in basis), names


def _named_factor_contrast(name: str, levels: tuple[Any, ...]) -> dict[str, Any] | None:
    """Standard R named contrasts evaluated at a prepared frame's own levels."""

    n = len(levels)
    if name == "contr.poly":
        rows, columns = _contr_poly(n)
        return {"data": rows, "columns": columns, "label": name}
    if name not in {"contr.sum", "contr.helmert", "contr.SAS", "contr.treatment"}:
        return None
    if name == "contr.helmert":
        matrix = np.zeros((n, n - 1))
        for column in range(n - 1):
            matrix[: column + 1, column] = -1.0
            matrix[column + 1, column] = column + 1
    else:
        matrix = np.eye(n)[:, 1:] if name == "contr.treatment" else np.eye(n)[:, :-1]
        if name == "contr.sum":
            matrix[-1] = -1.0
    names = (
        tuple(
            _as_character(level)
            for level in (levels[1:] if name == "contr.treatment" else levels[:-1])
        )
        if name in {"contr.SAS", "contr.treatment"}
        else tuple(str(column + 1) for column in range(n - 1))
    )
    return {"data": matrix, "columns": names, "label": name}
