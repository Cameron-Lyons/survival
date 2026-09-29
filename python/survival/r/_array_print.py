"""R-style raw response vectors and rate-table arrays, with explicit display limits."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from typing import Any

from ._coerce import (
    _integer_scalar,
    _materialize_1d,
    _normalize_bool_option,
    _pop_dotted_keyword,
    _r_format_numbers,
)
from ._print import print_options
from ._surv import Surv, Surv2, as_character_surv
from ._types import RateTable, RateTablePrint, ResponsePrint

_MAX_NOTE = " [ reached 'max' / getOption(\"max.print\") -- omitted "


def _max_print(value: Any) -> int:
    value = _integer_scalar(value, "max_print")
    if not 0 <= value <= 2147483646:
        raise ValueError("max_print must be between 0 and 2147483646")
    return value


def _encoded(value: str, quote: bool = False) -> str:
    replacements = {
        "\\": "\\\\",
        "\a": "\\a",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\v": "\\v",
    }
    if quote:
        replacements['"'] = '\\"'
    cells = []
    for char in value:
        if char in replacements:
            cells.append(replacements[char])
        elif ord(char) < 32 or ord(char) == 127:
            cells.append(f"\\{ord(char):03o}")
        else:
            cells.append(char)
    text = "".join(cells)
    return '"' + text + '"' if quote else text


def _width(value: str) -> int:
    return sum(
        0
        if unicodedata.combining(char)
        else 2
        if unicodedata.east_asian_width(char) in {"F", "W"}
        else 1
        for char in value
    )


def _pad(value: str, width: int, *, right: bool = True) -> str:
    spaces = " " * max(0, width - _width(value))
    return spaces + value if right else value + spaces


def _vector_lines(
    cells: list[str], width: int, *, names: list[str] | None = None, right: bool = False
) -> list[str]:
    size = max(map(_width, cells), default=0)
    if names is not None:
        labels = [_encoded(name) for name in names]
        size = max(size, max(map(_width, labels), default=0))
        per_line = max(1, width // (size + 1))
        lines = []
        for start in range(0, len(cells), per_line):
            lines.append(
                " ".join(_pad(name, size) for name in labels[start : start + per_line]).rstrip()
            )
            # R's named character vectors always align values right.
            lines.append(
                " ".join(_pad(cell, size) for cell in cells[start : start + per_line]).rstrip()
            )
        return lines or [""]
    label_width = len(str(len(cells))) + 2
    per_line = max(1, (width - label_width) // (size + 1))
    lines = []
    for start in range(0, len(cells), per_line):
        label = f"[{start + 1}]".rjust(label_width)
        lines.append(
            (
                label
                + " "
                + " ".join(
                    _pad(cell, size, right=right) for cell in cells[start : start + per_line]
                )
            ).rstrip()
        )
    return lines or [" [1]"]


def _response_report(
    x: Surv | Surv2, quote: Any, right: Any, names: Any, width: Any, max_print: Any
) -> ResponsePrint:
    _, _, width = print_options(1, None, width, 7)
    quote = _normalize_bool_option(quote, "quote")
    right = _normalize_bool_option(right, "right")
    limit = _max_print(max_print)
    labels = as_character_surv(x)
    n = len(labels)
    row_names = None
    if names is not None:
        row_names = _materialize_1d(names, "names")
        if len(row_names) != n or any(not isinstance(name, str) for name in row_names):
            raise ValueError("names must contain one string per response row")
    shown = n if n <= limit + 1 else limit
    if n == 0:
        lines = ["character(0)"]
    else:
        cells = [_encoded(label, quote) for label in labels[:shown]]
        lines = _vector_lines(
            cells, width, names=None if row_names is None else row_names[:shown], right=right
        )
        if shown < n:
            lines.append(_MAX_NOTE + f"{n - shown} entries ]")
    if isinstance(x, Surv2):
        data: dict[str, list[Any]] = {
            "time": list(x.time),
            "status": list(x.status),
            "type": ["Surv2"] * n,
        }
    else:
        data = {"time": list(x.time), "status": list(x.event), "type": [x.type] * n}
        if x.start is not None:
            data = {
                "start": list(x.start),
                "stop": data["time"],
                "status": data["status"],
                "type": data["type"],
            }
        if x.time2 is not None:
            data["time2"] = list(x.time2)
    if row_names is not None:
        data = {"name": row_names, **data}
    return ResponsePrint(data, labels, lines, shown)


def print_surv(
    x: Surv,
    quote: Any = False,
    *,
    right: Any = False,
    names: Sequence[str] | None = None,
    width: Any = 80,
    max_print: Any = 99999,
    **kwargs: Any,
) -> ResponsePrint:
    """Format a raw survival response; optional names supply R's response row names."""
    if not isinstance(x, Surv):
        raise TypeError("print_surv requires a Surv response")
    max_print = _pop_dotted_keyword(kwargs, "max", "max_print", max_print, 99999)
    if kwargs:
        raise TypeError("print_surv got unexpected arguments: " + ", ".join(sorted(kwargs)))
    return _response_report(x, quote, right, names, width, max_print)


def print_surv2(
    x: Surv2,
    quote: Any = False,
    *,
    right: Any = False,
    names: Sequence[str] | None = None,
    width: Any = 80,
    max_print: Any = 99999,
    **kwargs: Any,
) -> ResponsePrint:
    """Format a timeline response, preserving its event/censor state labels."""
    if not isinstance(x, Surv2):
        raise TypeError("print_surv2 requires a Surv2 response")
    max_print = _pop_dotted_keyword(kwargs, "max", "max_print", max_print, 99999)
    if kwargs:
        raise TypeError("print_surv2 got unexpected arguments: " + ", ".join(sorted(kwargs)))
    return _response_report(x, quote, right, names, width, max_print)


def _matrix_slice(
    rates: list[float],
    offset: int,
    nrow: int,
    rows: list[str],
    columns: list[str],
    titles: list[str],
    shown_rows: int,
    shown_columns: int,
    digits: int,
    width: int,
) -> list[str]:
    # Precision and row-label widths use all rows, including rows hidden by max.print.
    row_names = [_encoded(name) for name in rows]
    names = [_encoded(name) for name in columns[:shown_columns]]
    row_width = max(map(_width, row_names), default=0)
    row_width += max(2, _width(titles[0]) - row_width)
    indent = row_width - max(map(_width, row_names), default=0)
    cells = [
        _r_format_numbers(rates[offset + j * nrow : offset + (j + 1) * nrow], digits)
        for j in range(shown_columns)
    ]
    sizes = [
        max(_width(name), max(map(len, column), default=0))
        for name, column in zip(names, cells, strict=True)
    ]
    lines = []
    start = 0
    while start < shown_columns:
        end = start + 1
        used = row_width + sizes[start] + 1
        while end < shown_columns and used + sizes[end] + 1 < width:
            used += sizes[end] + 1
            end += 1
        # The row-axis title is printed by R's byte-counted printf field.
        row_title = titles[0] + " " * max(0, row_width - len(titles[0].encode("utf-8")))
        lines.extend(
            [
                " " * row_width + titles[1],
                (
                    row_title + "".join(" " + _pad(names[j], sizes[j]) for j in range(start, end))
                ).rstrip(),
            ]
        )
        for i in range(shown_rows):
            label = " " * indent + _pad(row_names[i], row_width - indent, right=False)
            lines.append(
                (
                    label + "".join(" " + cells[j][i].rjust(sizes[j]) for j in range(start, end))
                ).rstrip()
            )
        start = end
    if shown_columns == 0:
        lines = [
            " " * row_width + titles[1],
            titles[0].rstrip(),
            *(" " * indent + name for name in row_names),
        ]
    return lines


def _quantity(count: int, name: str) -> str:
    return f"{count} {name}" + ("s" if count != 1 else "")


def print_ratetable(
    x: RateTable, digits: Any = None, *, width: Any = 80, max_print: Any = 99999, **kwargs: Any
) -> RateTablePrint:
    """Format rate arrays as vectors, matrices or column-major matrix slices.

    Rendering stops at ``max_print`` using R's row/slice rules. The report still
    owns all rates at full precision, without expanding a dense coordinate grid.
    """
    if not isinstance(x, RateTable):
        raise TypeError("print_ratetable requires a rate table")
    _, digits, width = print_options(1, digits, width, 7)
    max_print = _pop_dotted_keyword(kwargs, "max", "max_print", max_print, 99999)
    if kwargs:
        raise TypeError("print_ratetable got unexpected arguments: " + ", ".join(sorted(kwargs)))
    limit = _max_print(max_print)
    dims, dimid, dimnames, rates = x.dims, x.dimid, x.dimnames, x.rates
    lines = ["Rate table with dimension(s): " + " ".join(dimid)]
    displayed = 0
    if len(dims) == 1:
        shown = len(rates) if len(rates) <= limit + 1 else limit
        cells = _r_format_numbers(rates[:shown], digits)
        # A named one-dimensional array prints its dimension title, then its labels.
        lines.append(dimid[0])
        lines.extend(_vector_lines(cells, width, names=dimnames[0][:shown], right=True))
        displayed = shown
        if shown < len(rates):
            lines.append(_MAX_NOTE + f"{len(rates) - shown} entries ]")
    else:
        nr, nc = dims[:2]
        size = nr * nc
        count = math.prod(dims[2:])
        if len(dims) == 2:
            shown_columns = min(nc, limit)
            shown_rows = min(nr, limit // nc)
            if shown_columns < nc and shown_rows < 1:
                shown_rows = 1
            lines.extend(
                _matrix_slice(
                    rates,
                    0,
                    nr,
                    dimnames[0],
                    dimnames[1],
                    dimid[:2],
                    shown_rows,
                    shown_columns,
                    digits,
                    width,
                )
            )
            displayed = shown_rows * shown_columns
            omitted = []
            if shown_rows < nr:
                omitted.append(_quantity(nr - shown_rows, "row"))
            if shown_columns < nc:
                omitted.append(_quantity(nc - shown_columns, "column"))
            if omitted:
                lines.append(_MAX_NOTE + " and ".join(omitted) + " ]")
        else:
            slices = min(count, (limit + size - 1) // size)
            last_rows, last_columns = nr, nc
            for page in range(slices):
                available = min(size, limit - page * size)
                last_columns = min(nc, available)
                last_rows = 1 if available < nc else available // nc
                parts = []
                stride = 1
                for name, levels, extent in zip(dimid[2:], dimnames[2:], dims[2:], strict=True):
                    parts.append(name + " = " + levels[(page // stride) % extent])
                    stride *= extent
                lines += [", , " + ", ".join(parts), ""]
                lines += _matrix_slice(
                    rates,
                    page * size,
                    nr,
                    dimnames[0],
                    dimnames[1],
                    dimid[:2],
                    last_rows,
                    last_columns,
                    digits,
                    width,
                )
                lines.append("")
                displayed += last_rows * last_columns
            if limit < len(rates):
                omitted = []
                if slices < count:
                    omitted.append(_quantity(count - slices, "slice"))
                else:
                    if last_rows < nr:
                        omitted.append(_quantity(nr - last_rows, "row"))
                    if last_columns < nc:
                        omitted.append(_quantity(nc - last_columns, "column"))
                lines.append(_MAX_NOTE + " ".join(omitted) + " ]")
    return RateTablePrint(dims, dimid, dimnames, rates, digits, lines, displayed)
