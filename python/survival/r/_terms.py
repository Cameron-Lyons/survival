"""Prepared term metadata and R's column/special-term helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, NotRequired, TypedDict

from ._coerce import _integer_scalar, _materialize_1d


def _strings(value: Any, name: str) -> tuple[str, ...]:
    values = _materialize_1d(value, name)
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{name} must contain strings")
    return tuple(values)


@dataclass(frozen=True)
class TermMetadata:
    """Prepared attributes of an R ``terms`` object, without data or an environment.

    ``term_labels`` names the factor-matrix columns; ``variables`` names its
    rows, including the response. ``factors`` contains R's 0/1/2 term codes,
    ``order`` gives each term's degree, and ``response`` is 0 or 1. ``specials``
    maps registered function names to **one-based variable positions**.
    Only ``term_labels`` is needed by :func:`attrassign`.

    Inputs are copied to immutable tuples and a read-only mapping. This is a
    metadata container; it does not parse, evaluate or reconstruct a formula.
    """

    term_labels: Sequence[str]
    variables: Sequence[str] = ()
    factors: Sequence[Sequence[int]] | None = None
    order: Sequence[int] | None = None
    response: int = 0
    specials: Mapping[str, Sequence[int] | None] | None = None

    def __post_init__(self) -> None:
        labels = _strings(self.term_labels, "term_labels")
        variables = _strings(self.variables, "variables")
        response = _integer_scalar(self.response, "response")
        if response not in (0, 1):
            raise ValueError("response must be 0 or 1")
        factors = None
        if self.factors is not None:
            factors = tuple(
                tuple(_integer_scalar(code, "factors") for code in row) for row in self.factors
            )
            if len(factors) != len(variables) or any(len(row) != len(labels) for row in factors):
                raise ValueError("factors must have one row per variable and one column per term")
            if any(code not in (0, 1, 2) for row in factors for code in row):
                raise ValueError("factors must contain only 0, 1, or 2")
        order = None
        if self.order is not None:
            order = tuple(_integer_scalar(value, "order") for value in self.order)
            if len(order) != len(labels) or any(value < 1 for value in order):
                raise ValueError("order must contain one positive degree per term")
        specials = {}
        for name, positions in (self.specials or {}).items():
            if not isinstance(name, str):
                raise TypeError("special names must be strings")
            codes = (
                ()
                if positions is None
                else tuple(_integer_scalar(code, "specials") for code in positions)
            )
            if any(code < 1 or code > len(variables) for code in codes):
                raise ValueError("specials must contain one-based variable positions")
            specials[name] = codes
        object.__setattr__(self, "term_labels", labels)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "specials", MappingProxyType(specials))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.term_labels,
                self.variables,
                self.factors,
                self.order,
                self.response,
                dict(self.specials or {}),
            ),
        )


class SpecialTerms(TypedDict):
    """R's special-variable names and one-based term positions."""

    vars: list[str]
    terms: list[int]
    tvar: NotRequired[list[int]]


def _metadata(tt: Any) -> TermMetadata:
    if isinstance(tt, TermMetadata):
        return tt
    if isinstance(tt, Mapping):
        labels = tt.get("term_labels", tt.get("term.labels"))
        if labels is not None:
            return TermMetadata(
                labels,
                tt.get("variables", ()),
                tt.get("factors"),
                tt.get("order"),
                tt.get("response", 0),
                tt.get("specials"),
            )
    raise TypeError("need TermMetadata or a mapping of terms attributes")


def _column_groups(
    assign: Sequence[int], labels: Sequence[str], *, offset: int = 0, exclude: int | None = None
) -> dict[str, list[int]]:
    """Group already-validated column codes in one pass, preserving first appearance."""
    groups: dict[str, list[int]] = {}
    for column, code in enumerate(assign, start=offset):
        if code != exclude:
            label = labels[code]
            group = groups.get(label)
            if group is None:
                groups[label] = [column]
            else:
                group.append(column)
    return groups


def attrassign(object: Any, tt: Any) -> dict[str, list[int]]:
    """Map term labels to **one-based** model-matrix columns, as in R.

    ``object`` is the mapping returned by :func:`model_matrix`, or an object
    with an ``assign`` vector. Codes are 0 for the intercept and 1..N for the
    labels in ``tt``. ``tt`` accepts :class:`TermMetadata`, a mapping with
    ``term_labels``/``term.labels``, or a fitted model accepted by
    :func:`model_term_names`. Groups follow their first matrix-column occurrence.
    No matrix data are read or copied. Internal fitted-model ``assign`` mappings
    continue to use zero-based columns.
    """
    if isinstance(tt, TermMetadata):
        labels = tt.term_labels
    elif isinstance(tt, Mapping):
        names = tt.get("term_labels", tt.get("term.labels"))
        if names is None:
            raise TypeError("need term metadata with term_labels")
        labels = _strings(names, "term_labels")
    else:
        from ._models import model_term_names

        try:
            labels = model_term_names(tt)
        except TypeError as exc:
            raise TypeError("need term metadata or a fitted model with term labels") from exc
    raw = object.get("assign") if isinstance(object, Mapping) else getattr(object, "assign", None)
    if raw is None or isinstance(raw, Mapping):
        raise TypeError("argument is not really a model matrix: an assign vector is required")
    assign = [_integer_scalar(value, "assign") for value in _materialize_1d(raw, "assign")]
    if any(code < 0 or code > len(labels) for code in assign):
        raise ValueError("assign codes must be between 0 and the number of terms")
    return _column_groups(assign, ("(Intercept)", *labels), offset=1)


def untangle_specials(tt: Any, special: str, order: Any = 1) -> SpecialTerms:
    """Find registered special variables and the terms containing them.

    ``tt`` is :class:`TermMetadata` or a mapping of its fields. The requested
    ``order`` is an integer or sequence of degrees. Output matches R's
    ``untangle.specials``: ``vars`` retains every registered special, ``tvar``
    subtracts the response from its one-based variable position, and ``terms``
    selects one-based term columns with a matching degree. An absent special
    returns only empty ``vars`` and ``terms`` lists. Membership uses metadata,
    so nested calls and namespace-qualified functions are not guessed from text.
    R's single-term ``seq`` behavior is preserved: factor codes greater than
    one can yield extra term indices (see the term-helper compatibility guide).
    """
    metadata = _metadata(tt)
    if not isinstance(special, str):
        raise TypeError("special must be a string")
    positions = (metadata.specials or {}).get(special, ())
    if not positions:
        return {"vars": [], "terms": []}
    if metadata.factors is None or metadata.order is None:
        raise ValueError("factors and order are required for a present special")
    values = [order] if isinstance(order, Real) else _materialize_1d(order, "order")
    degrees = {_integer_scalar(value, "order") for value in values}
    selected = [False] * len(metadata.term_labels)
    for position in positions:
        for column, code in enumerate(metadata.factors[position - 1]):
            selected[column] |= code != 0
    if len(selected) == 1:
        # R calls seq(ff), not seq_along(ff). For one term, ff is a scalar
        # factor-code sum, so interaction-only codes can produce extra indices.
        count = sum(metadata.factors[position - 1][0] for position in positions)
        terms = list(range(1, count + 1)) if metadata.order[0] in degrees else []
    else:
        terms = [
            column
            for column, (present, degree) in enumerate(
                zip(selected, metadata.order, strict=True), start=1
            )
            if present and degree in degrees
        ]
    return {
        "vars": [metadata.variables[position - 1] for position in positions],
        "tvar": [position - metadata.response for position in positions],
        "terms": terms,
    }
