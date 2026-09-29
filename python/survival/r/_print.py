"""Shared text layout for numeric report tables, without global print options."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._coerce import _finite_float, _integer_scalar, _r_format_numbers
from ._types import NamedMatrix


def print_options(
    scale: Any, digits: Any, width: Any, default_digits: int = 3
) -> tuple[float, int, int]:
    scale = _finite_float(scale, "scale")
    if scale <= 0:
        raise ValueError("scale must be finite and positive")
    digits = default_digits if digits is None else _integer_scalar(digits, "digits")
    if not 1 <= digits <= 22:
        raise ValueError("digits must be between 1 and 22")
    width = _integer_scalar(width, "width")
    if not 10 <= width <= 10000:
        raise ValueError("width must be between 10 and 10000")
    return scale, digits, width


def numeric_vector_lines(
    names: Sequence[str], values: Sequence[float], digits: int, width: int
) -> list[str]:
    """Named vectors share precision and cell width across all entries."""

    cells = _r_format_numbers(values, digits)
    cell_width = max(max(map(len, names), default=0), max(map(len, cells), default=0))
    per_line = max(1, width // (cell_width + 1))
    lines = []
    for start in range(0, len(cells), per_line):
        lines.append(" ".join(name.rjust(cell_width) for name in names[start : start + per_line]))
        lines.append(" ".join(cell.rjust(cell_width) for cell in cells[start : start + per_line]))
    return lines


def numeric_matrix_lines(table: NamedMatrix, digits: int, width: int) -> list[str]:
    """R's numeric matrix layout: right-aligned columns and wrapped column blocks.

    Each numeric column chooses its own fixed/scientific precision. Explicit row
    names align left; implicit ``[i,]`` names align right. Trailing spaces are removed.
    """

    count = len(table.values)
    names = table.rownames
    if names is None:
        names = [f"[{i + 1},]" for i in range(count)]
    if len(names) != count or any(len(row) != len(table.colnames) for row in table.values):
        raise ValueError("report table labels must match its shape")
    name_width = max(map(len, names), default=4 if not count else 0)
    names = [
        name.rjust(name_width) if table.rownames is None else name.ljust(name_width)
        for name in names
    ]
    columns = [
        _r_format_numbers([row[j] for row in table.values], digits)
        for j in range(len(table.colnames))
    ]
    widths = [
        max(len(title), max(map(len, column), default=0))
        for title, column in zip(table.colnames, columns, strict=True)
    ]
    lines = []
    start = 0
    while start < len(columns):
        end = start + 1
        used = name_width + 1 + widths[start]
        while end < len(columns) and used + 1 + widths[end] <= width:
            used += 1 + widths[end]
            end += 1
        block = range(start, end)
        lines.append(
            (
                " " * name_width + "".join(" " + table.colnames[j].rjust(widths[j]) for j in block)
            ).rstrip()
        )
        lines.extend(
            (name + "".join(" " + columns[j][i].rjust(widths[j]) for j in block)).rstrip()
            for i, name in enumerate(names)
        )
        start = end
    return lines
