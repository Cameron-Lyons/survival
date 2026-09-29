"""Shared text layout for numeric report tables, without global print options."""

from __future__ import annotations

import math
import sys
import textwrap
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


def numeric_matrix_lines(
    table: NamedMatrix, digits: int, width: int, *, row_title: str | None = None
) -> list[str]:
    """R's numeric matrix layout: right-aligned columns and wrapped column blocks.

    Each numeric column chooses its own fixed/scientific precision. Explicit row
    names align left; implicit ``[i,]`` names align right. Trailing spaces are removed.
    """

    columns = [
        _r_format_numbers([row[j] for row in table.values], digits)
        for j in range(len(table.colnames))
    ]
    return character_matrix_lines(table, columns, width, row_title=row_title)


def character_matrix_lines(
    table: NamedMatrix,
    columns: Sequence[Sequence[str]],
    width: int,
    *,
    right: bool = True,
    row_title: str | None = None,
) -> list[str]:
    """Lay out preformatted columns using R's matrix labels and column blocks."""
    count = len(table.values)
    names = table.rownames
    if names is None:
        names = [f"[{i + 1},]" for i in range(count)]
    if len(names) != count or any(len(row) != len(table.colnames) for row in table.values):
        raise ValueError("report table labels must match its shape")
    name_width = max(map(len, names), default=4 if not count else 0)
    if row_title is not None:
        indent = max(2, len(row_title) - name_width)
        names = [" " * indent + name.ljust(name_width) for name in names]
        name_width += indent
    else:
        names = [
            name.rjust(name_width) if table.rownames is None else name.ljust(name_width)
            for name in names
        ]
    widths = [
        max(len(title), max(map(len, column), default=0))
        for title, column in zip(table.colnames, columns, strict=True)
    ]
    lines = []
    align = str.rjust if right else str.ljust
    start = 0
    while start < len(columns):
        end = start + 1
        used = name_width + 1 + widths[start]
        while end < len(columns) and used + 1 + widths[end] < width:
            used += 1 + widths[end]
            end += 1
        block = range(start, end)
        if row_title is not None:
            lines.append("")
        lines.append(
            (
                (row_title or "").ljust(name_width)
                + "".join(" " + align(table.colnames[j], widths[j]) for j in block)
            ).rstrip()
        )
        lines.extend(
            (name + "".join(" " + align(columns[j][i], widths[j]) for j in block)).rstrip()
            for i, name in enumerate(names)
        )
        start = end
    return lines


def format_pvalues(values: Sequence[float], digits: int) -> list[str]:
    """R's vector ``format.pval``, including mixed precision and tiny values."""
    result = ["NA"] * len(values)
    groups: list[list[int]] = [[], []]
    zeros = []
    for i, value in enumerate(values):
        if math.isnan(value):
            continue
        if value < sys.float_info.epsilon:
            zeros.append(i)
        else:
            exponent = math.floor(math.log10(value)) if math.isfinite(value) else math.inf
            fixed = exponent >= -3 or (exponent == -4 and digits > 1)
            groups[int(fixed)].append(i)
    for group in groups:
        for i, cell in zip(
            group, _r_format_numbers([values[i] for i in group], digits), strict=True
        ):
            result[i] = cell
    if zeros:
        precision = max(1, digits - 2)
        ordinary = groups[0] + groups[1]
        if ordinary:
            size = max(len(result[i]) for i in ordinary)
            if precision > 1 and precision + 6 > size:
                precision = max(1, size - 7)
            separator = "" if precision == 1 and size <= 6 else " "
        else:
            separator = "" if precision == 1 else " "
        cell = "<" + separator + _r_format_numbers([sys.float_info.epsilon], precision)[0]
        for i in zeros:
            result[i] = cell
    return result


def coefficient_matrix_lines(
    table: NamedMatrix,
    digits: int,
    width: int,
    *,
    signif_stars: bool = False,
    na_print: str = "NA",
    row_title: str | None = None,
) -> list[str]:
    """R's ``printCoefmat`` for model tables ending in a test and p-value.

    Coefficients and standard errors share decimal precision across columns.
    NaN represents unavailable model estimates (R's NA), as in native summaries.
    """
    count, nc = len(table.values), len(table.colnames)
    test_digits = max(1, min(5, digits - 1))
    values = [row[j] for j in range(nc - 2) for row in table.values]
    finite = [abs(value) for value in values if math.isfinite(value) and value != 0]
    places = max(1, digits - (1 + math.floor(math.log10(min(finite))) if finite else 1))
    cells = _r_format_numbers([round(value, places) for value in values], digits)
    # Avoid replacing a nonzero estimate with a displayed zero after rounding.
    lost = [
        i
        for i, (value, cell) in enumerate(zip(values, cells, strict=True))
        if math.isfinite(value) and value != 0 and float(cell) == 0
    ]
    for i, cell in zip(
        lost, _r_format_numbers([values[i] for i in lost], max(1, digits - 1)), strict=True
    ):
        cells[i] = cell
    columns = [cells[j * count : (j + 1) * count] for j in range(nc - 2)]
    columns.append(_r_format_numbers([round(row[-2], test_digits) for row in table.values], digits))
    pvalues = [row[-1] for row in table.values]
    columns.append(format_pvalues(pvalues, test_digits))
    for j, column in enumerate(columns):
        for i in range(count):
            if math.isnan(table.values[i][j]):
                column[i] = na_print
    signif_stars = signif_stars and any(value < 0.1 for value in pvalues)
    if signif_stars:
        symbols = [
            next(
                (
                    symbol
                    for cutoff, symbol in ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, "."))
                    if value <= cutoff
                ),
                " ",
            )
            for value in pvalues
        ]
        size = max(map(len, symbols), default=0)
        columns.append([symbol.ljust(size) for symbol in symbols])
        table = NamedMatrix(
            table.rownames,
            [*table.colnames, ""],
            [[float(value) for value in row] + [0.0] for row in table.values],
        )
        # R's cbind of significance symbols discards dimnames' dimension titles.
        row_title = None
    lines = character_matrix_lines(table, columns, width, row_title=row_title)
    if signif_stars:
        legend = "0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"
        lines.append("---")
        if width < len(legend):
            lines.append("Signif. codes:")
            lines.extend(
                textwrap.wrap(legend, width=width - 2, initial_indent="  ", subsequent_indent="  ")
            )
        elif len(legend) + 15 > width + 4:
            lines.extend(["Signif. codes:", legend])
        else:
            lines.append("Signif. codes:  " + legend)
    return lines
