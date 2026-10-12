"""R factor columns as Python lists that carry their levels."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, SupportsIndex, overload


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _level_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, str):
        return (1, value)
    return (0, value)


class Factor(list[Any]):
    """An R factor: a list of labels (``None`` for NA) with R's level order.

    ``categories`` holds the levels in R's order and ``ordered`` marks an ordered
    factor; ``levels`` is R's name for ``categories``. As in R's ``factor()``, the
    levels default to the sorted distinct values and a value outside the given
    levels becomes NA. The formula functions read the levels from ``categories``,
    as they do for a pandas ``Categorical``.
    """

    categories: tuple[Any, ...]
    ordered: bool

    def __init__(
        self,
        values: Iterable[Any] = (),
        levels: Sequence[Any] | None = None,
        ordered: bool = False,
    ) -> None:
        values = list(values)
        if levels is None:
            levels = sorted({value for value in values if not _missing(value)}, key=_level_key)
        else:
            declared = set(levels)
            values = [
                None if _missing(value) or value not in declared else value for value in values
            ]
        super().__init__(values)
        self.categories = tuple(levels)
        self.ordered = bool(ordered)

    @property
    def levels(self) -> list[Any]:
        return list(self.categories)

    @overload
    def __getitem__(self, index: SupportsIndex) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> Factor: ...

    def __getitem__(self, index: SupportsIndex | slice) -> Any:
        if isinstance(index, slice):
            return Factor(super().__getitem__(index), self.categories, self.ordered)
        return super().__getitem__(index)

    def copy(self) -> Factor:
        return Factor(self, self.categories, self.ordered)

    def __reduce__(self) -> tuple[Any, ...]:
        return (Factor, (list(self), self.categories, self.ordered))

    def __repr__(self) -> str:
        return f"Factor({list.__repr__(self)}, levels={list(self.categories)!r})"
