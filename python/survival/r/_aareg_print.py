"""Aalen additive model reports using the existing numerical summaries."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from ._aareg import summary_aareg
from ._coerce import _r_format_number
from ._print import format_pvalues, numeric_matrix_lines, print_options
from ._survpenal_print import _naprint, _signif
from ._types import AaregModelResult, ModelPrint, NamedMatrix


def _report(
    x: Mapping[str, Any], digits: int, width: int, *, test_label: str | None = None
) -> ModelPrint:
    rows = x["table"]
    keys = (
        ["slope", "coef", "se"]
        + (["robust_se"] if "robust se" in x["columns"] else [])
        + ["z", "p"]
    )
    table = NamedMatrix(
        [row["name"] for row in rows],
        list(x["columns"]),
        [[float(row[key]) for key in keys] for row in rows],
    )
    display = NamedMatrix(
        table.rownames,
        table.colnames,
        [[_signif(value, 3) for value in row] for row in table.values],
    )
    lines = numeric_matrix_lines(display, digits, width)
    p = format_pvalues([x["p"]], 3)[0]
    lines += [
        "",
        f"Chisq={_r_format_number(round(x['chisq'], 2), 7)} on {x['df']} df, p={p}; "
        f"test weights={x['test'] if test_label is None else test_label}",
    ]
    statistics = {
        key: copy.deepcopy(x[key])
        for key in ("n", "test", "test_statistic", "test_var", "test_var2", "chisq", "df", "p")
    }
    return ModelPrint({"coefficients": table}, statistics, digits, lines)


def print_aareg(
    x: AaregModelResult,
    maxtime: Any = None,
    test: Any = None,
    scale: Any = 1,
    *,
    width: Any = 80,
) -> ModelPrint:
    """Return ``print.aareg``, including sample and unique-event counts.

    ``maxtime``, ``test`` and ``scale`` select the numerical summary. R rounds
    entries to three significant figures before printing with seven digits.
    The footer retains the fit's test label even when ``test`` overrides it.
    """
    if not isinstance(x, AaregModelResult):
        raise TypeError("print_aareg requires an Aalen fit")
    _, digits, width = print_options(1, None, width, 7)
    summary = summary_aareg(x, maxtime=maxtime, test=test, scale=scale)
    report = _report(summary, digits, width, test_label=x.test)
    count = f"  n={x.n[0]} ({_naprint(x.na_action)})" if x.na_action else f"  n= {x.n[0]}"
    return ModelPrint(
        report.tables,
        report.statistics,
        digits,
        [
            count,
            f"    {summary['n'][1]} out of {x.n[2]} unique event times used",
            "",
            *report.lines,
        ],
    )


def print_summary_aareg(x: Mapping[str, Any], *, width: Any = 80) -> ModelPrint:
    """Return ``print.summary.aareg`` with full-precision coefficients and tests."""
    if not isinstance(x, Mapping) or x.get("model_type") != "aareg":
        raise TypeError("print_summary_aareg requires an Aalen model summary")
    _, digits, width = print_options(1, None, width, 7)
    return _report(x, digits, width)
