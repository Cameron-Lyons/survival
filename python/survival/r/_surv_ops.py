"""Operation guards for survival responses, matching R's Math/Ops/Summary groups."""

from __future__ import annotations

import math
from dataclasses import Field, fields
from typing import Any, ClassVar, Never


def _invalid_operation(*args: Any, **kwargs: Any) -> Never:
    raise TypeError("Invalid operation on a survival time")


def _same_value(left: Any, right: Any) -> bool:
    # Tuple equality stays in C for the usual complete-data comparison.
    if left is right or left == right:
        return True
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    )


class _SurvivalOperations:
    """Responses are structured observations, not numeric operands.

    Explicit matrix/column extraction is required before numeric operations.
    NumPy must not reduce a response as a zero-dimensional object array: that
    can silently return the input, or a truth value based on its row count.
    """

    __slots__ = ()
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def equals(self, other: object) -> bool:
        """Compare response columns and metadata, treating matching missing values as equal.

        This is an explicit structural comparison, like R's ``identical``.
        Arithmetic comparisons, including ``==`` and ``!=``, are invalid.
        """
        if type(self) is not type(other):
            return False
        return all(
            _same_value(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
        )

    __eq__ = _invalid_operation
    __ne__ = _invalid_operation
    __lt__ = _invalid_operation
    __le__ = _invalid_operation
    __gt__ = _invalid_operation
    __ge__ = _invalid_operation
    __add__ = __radd__ = _invalid_operation
    __sub__ = __rsub__ = _invalid_operation
    __mul__ = __rmul__ = _invalid_operation
    __truediv__ = __rtruediv__ = _invalid_operation
    __floordiv__ = __rfloordiv__ = _invalid_operation
    __mod__ = __rmod__ = _invalid_operation
    __pow__ = __rpow__ = _invalid_operation
    __divmod__ = __rdivmod__ = _invalid_operation
    __matmul__ = __rmatmul__ = _invalid_operation
    __and__ = __rand__ = _invalid_operation
    __or__ = __ror__ = _invalid_operation
    __xor__ = __rxor__ = _invalid_operation
    __lshift__ = __rlshift__ = _invalid_operation
    __rshift__ = __rrshift__ = _invalid_operation
    __neg__ = __pos__ = __abs__ = __invert__ = _invalid_operation
    __round__ = __floor__ = __ceil__ = __trunc__ = _invalid_operation
    __bool__ = __float__ = __int__ = __complex__ = _invalid_operation
    __iter__ = _invalid_operation
    __array__ = __array_ufunc__ = __array_function__ = _invalid_operation
