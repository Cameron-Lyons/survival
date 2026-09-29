"""Shared plotting controls; Matplotlib is imported only when axes are needed."""

from __future__ import annotations

from operator import index
from typing import Any

import numpy as np


def axes(ax: Any, overlay: bool) -> Any:
    if ax is not None:
        return ax
    try:
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("Rendering requires 'pip install survival[plot]'") from exc
    return plt.gca() if overlay else plt.subplots()[1]


def styles(value: Any, default: Any) -> list[Any]:
    if value is None:
        return [default]
    if isinstance(value, str) or np.isscalar(value):
        return [value]
    result = list(value)
    if not result:
        raise ValueError("style sequences must not be empty")
    return result


def panel_axes(ax: Any, count: int) -> tuple[Any, ...]:
    if ax is not None:
        result = (ax,) if hasattr(ax, "plot") else tuple(np.asarray(ax, dtype=object).ravel())
        if len(result) != count:
            raise ValueError(f"ax must contain one axis per plotted panel ({count} required)")
        return result
    first = axes(None, False)
    if count == 1:
        return (first,)
    figure = first.figure
    first.remove()
    figure.set_size_inches(7, 3 * count)
    figure.set_layout_engine("constrained")
    return tuple(figure.subplots(count, 1, squeeze=False).ravel())


def variables(var: Any, names: list[str]) -> list[int]:
    if var is None:
        return list(range(len(names)))
    values: list[Any] = [var] if isinstance(var, str) or np.isscalar(var) else list(var)
    try:
        selected = [
            names.index(value) if isinstance(value, str) else index(value) - 1 for value in values
        ]
    except (ValueError, TypeError) as exc:
        raise ValueError("var must contain term names or one-based column indices") from exc
    if not selected or any(i < 0 or i >= len(names) for i in selected):
        raise ValueError("invalid variable requested")
    return selected
