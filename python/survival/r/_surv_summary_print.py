"""Detailed survival and expected-survival tables, with optional grouped text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._coerce import _normalize_bool_option, _r_format_number
from ._print import numeric_matrix_lines, numeric_vector_lines, print_options
from ._types import (
    NamedMatrix,
    SummarySurvfitCoxmsResult,
    SummarySurvfitResult,
    SurvExpResult,
    SurvExpSummary,
)


@dataclass(frozen=True)
class SurvivalTablePrint:
    """Detailed tables at full precision and their formatted text.

    ``tables`` contains one matrix per group, with matching ``groups`` labels
    (``None`` for an ungrouped table). ``lines`` contains the rendered report;
    ``str(report)`` adds a final newline. Construction never writes to stdout.
    ``as_data_frame`` combines the groups as columns without rounding.
    """

    tables: list[NamedMatrix]
    groups: list[str | None]
    digits: int
    lines: list[str]

    def __str__(self) -> str:
        return "\n".join(self.lines) + "\n"


def _matrix(values: Any, rows: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) != rows:
        raise ValueError(f"{name} must have one row per summary time")
    return array


def _tables(
    values: NDArray[np.float64],
    columns: list[str],
    groups: list[str] | None,
    levels: list[str] | None = None,
) -> tuple[list[NamedMatrix], list[str | None]]:
    if groups is None:
        return [NamedMatrix([""] * len(values), columns, values.tolist())], [None]
    if len(groups) != len(values):
        raise ValueError("strata must have one label per summary time")
    # One pass through labels avoids rescanning the full matrix for each group.
    indices: dict[str, list[int]] = {label: [] for label in levels or ()}
    for row, label in enumerate(groups):
        indices.setdefault(label, []).append(row)
    return [
        NamedMatrix([""] * len(rows), list(columns), values[rows].tolist())
        for rows in indices.values()
    ], list(indices)


def _report(
    tables: list[NamedMatrix],
    groups: list[str | None],
    digits: int,
    width: int,
    *,
    leading_blank: bool = False,
) -> SurvivalTablePrint:
    lines: list[str] = [""] if leading_blank else []
    for table, group in zip(tables, groups, strict=True):
        if group is not None:
            lines.append(" " * 16 + group)
        if group is not None and len(table.values) == 1:
            lines.extend(numeric_vector_lines(table.colnames, table.values[0], digits, width))
        else:
            lines.extend(numeric_matrix_lines(table, digits, width))
        if group is not None:
            lines.append("")
    return SurvivalTablePrint(tables, groups, digits, lines)


def _confidence(x: SummarySurvfitResult, rows: int) -> tuple[list[NDArray[np.float64]], list[str]]:
    values, columns = [], []
    if x.std_err is not None:
        values.append(_matrix(x.std_err, rows, "std_err"))
        columns.append("std.err")
        if x.lower is not None:
            if x.upper is None or x.conf_int is None:
                raise ValueError("confidence limits require upper bounds and a confidence level")
            values.extend((_matrix(x.lower, rows, "lower"), _matrix(x.upper, rows, "upper")))
            level = _r_format_number(100 * x.conf_int, 15)
            columns.extend((f"lower {level}% CI", f"upper {level}% CI"))
    if any(value.shape[1] != 1 for value in values):
        raise ValueError("single-curve confidence columns must be vectors")
    return values, columns


def print_summary_survfit(
    x: SummarySurvfitResult | SummarySurvfitCoxmsResult,
    digits: Any = None,
    *,
    width: Any = 80,
) -> SurvivalTablePrint:
    """Format event-time or requested-time rows from ``summary_survfit``.

    One ordinary curve includes standard errors and confidence limits when
    available; several Cox predictions print one survival column each. Multistate
    summaries dispatch to ``print_summary_survfitms``. Strata form separate tables.
    Call ``summary_survfit(..., censored=True)`` to report censor-only observations.
    """

    if isinstance(x, SummarySurvfitCoxmsResult) or (
        isinstance(x, SummarySurvfitResult) and x.pstate is not None
    ):
        return print_summary_survfitms(x, digits, width=width)
    if not isinstance(x, SummarySurvfitResult):
        raise TypeError("print_summary_survfit requires a survival curve summary")
    _, digits, width = print_options(1, digits, width)
    rows = len(x.time)
    if not rows:
        raise ValueError("There are no events to print; use censored=True in summary_survfit")
    if x.surv is None:
        raise ValueError("summary is missing survival probabilities")
    values = [
        np.asarray(x.time)[:, None],
        _matrix(x.n_risk, rows, "n_risk"),
        _matrix(x.n_event, rows, "n_event"),
    ]
    columns = ["time", "n.risk", "n.event"]
    if x.n_enter is not None and x.type != "right":
        values.append(_matrix(x.n_censor, rows, "n_censor"))
        columns.append("censored")
    probabilities = _matrix(x.surv, rows, "surv")
    ncurve = probabilities.shape[1]
    values.append(probabilities)
    columns.extend(["survival"] if ncurve == 1 else [f"survival{i + 1}" for i in range(ncurve)])
    if ncurve == 1:
        confidence, headers = _confidence(x, rows)
        values.extend(confidence)
        columns.extend(headers)
    matrix = np.column_stack(values)
    groups = x.strata
    if x.start_time is not None:
        keep = np.asarray(x.time) >= x.start_time
        matrix = matrix[keep]
        groups = (
            None
            if groups is None
            else [group for group, use in zip(groups, keep, strict=True) if use]
        )
    tables, labels = _tables(matrix, columns, groups, x.strata_levels)
    return _report(tables, labels, digits, width)


def print_summary_survfitms(
    x: SummarySurvfitResult | SummarySurvfitCoxmsResult,
    digits: Any = None,
    *,
    width: Any = 80,
) -> SurvivalTablePrint:
    """Format state probabilities with total risk and event counts at each time.

    Multistate Cox prediction rows form ``data 1``, ``data 2``, ... groups, each
    subdivided by stratum. A one-state summary also prints its available standard
    errors and confidence limits. Conditional cutoffs use the summary's time units.
    """

    if not isinstance(x, SummarySurvfitResult | SummarySurvfitCoxmsResult) or x.pstate is None:
        raise TypeError("print_summary_survfitms requires a multistate curve summary")
    _, digits, width = print_options(1, digits, width)
    rows = len(x.time)
    if not rows:
        raise ValueError("There are no events to print; use censored=True in summary_survfit")
    probability = np.asarray(x.pstate, dtype=float)
    if probability.ndim not in (2, 3) or len(probability) != rows:
        raise ValueError("pstate must have one row per summary time")
    nstate = probability.shape[-1]
    states = x.states or []
    if len(states) != nstate:
        raise ValueError("state labels must match the probability columns")
    columns = ["time", "n.risk", "n.event"]
    columns += [f"Pr({state})" for state in states] if nstate > 1 else ["P"]
    counts = np.column_stack(
        (
            x.time,
            _matrix(x.n_risk, rows, "n_risk").sum(axis=1),
            _matrix(x.n_event, rows, "n_event").sum(axis=1),
        )
    )
    keep = np.ones(rows, dtype=bool) if x.start_time is None else np.asarray(x.time) >= x.start_time
    if not keep.any():
        raise ValueError(f"No rows remain using start_time = {x.start_time}")
    tables, labels = [], []
    ndata = probability.shape[1] if probability.ndim == 3 else 1
    for col in range(ndata):
        probabilities = probability[:, col, :] if probability.ndim == 3 else probability
        values = [counts, probabilities]
        names = list(columns)
        if nstate == 1 and isinstance(x, SummarySurvfitResult):
            confidence, headers = _confidence(x, rows)
            values.extend(confidence)
            names.extend(headers)
        groups = x.strata
        if probability.ndim == 3:
            groups = (
                [f"{group}, data {col + 1}" for group in groups]
                if groups is not None
                else [f"data {col + 1}"] * rows
            )
        groups = (
            None
            if groups is None
            else [group for group, use in zip(groups, keep, strict=True) if use]
        )
        block, block_labels = _tables(np.column_stack(values)[keep], names, groups)
        tables.extend(block)
        labels.extend(block_labels)
    return _report(tables, labels, digits, width)


def _expected_table(
    x: SurvExpResult | SurvExpSummary, scale: float, *, summary: bool, omit_missing: bool = False
) -> NamedMatrix:
    rows = len(x.time)
    ncurve = len(x.strata or [""])
    raw_survival = np.asarray(x.surv, dtype=float)
    raw_risk = np.asarray(x.n_risk, dtype=float)
    survival = _matrix(raw_survival, rows, "surv") if rows else np.empty((0, ncurve))
    risk = _matrix(raw_risk, rows, "n_risk") if rows else np.empty((0, ncurve))
    ncurve = survival.shape[1]
    columns = ["time"]
    columns += (
        [f"nrisk{i + 1}" for i in range(risk.shape[1])]
        if raw_risk.ndim == 2 or risk.shape[1] > 1
        else ["n.risk"]
    )
    if ncurve == 1 and (summary or raw_survival.ndim != 2):
        columns.append("survival")
    elif summary:
        columns.extend(f"survival{i + 1}" for i in range(ncurve))
    else:
        if x.strata is None or len(x.strata) != ncurve:
            raise ValueError("expected-survival labels must match the curve columns")
        columns.extend(x.strata)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.column_stack((np.asarray(x.time, dtype=float) / scale, risk, survival))
    if omit_missing:
        matrix = matrix[np.isnan(matrix).sum(axis=1) < len(columns) - 2]
    return NamedMatrix([""] * len(matrix), columns, matrix.tolist())


def print_survexp(
    x: SurvExpResult,
    scale: Any = 1,
    digits: Any = None,
    naprint: Any = False,
    *,
    width: Any = 80,
) -> SurvivalTablePrint:
    """Format expected curves as time, per-curve risk counts and survival columns.

    Names on ``SurvExpResult.strata`` label columns, not blocks of rows. As in R,
    rows with too many missing cells are omitted unless ``naprint=True``. Stored
    expected curves have no standard errors. Original call and rate-table prose
    are not reconstructed. Individual-method vectors are not curve results.
    """

    if not isinstance(x, SurvExpResult):
        raise TypeError("print_survexp requires an expected-survival curve")
    scale, digits, width = print_options(scale, digits, width)
    show_na = _normalize_bool_option(naprint, "naprint")
    table = _expected_table(x, scale, summary=False, omit_missing=not show_na)
    return _report([table], [None], digits, width, leading_blank=True)


def print_summary_survexp(
    x: SurvExpSummary,
    digits: Any = None,
    *,
    width: Any = 80,
) -> SurvivalTablePrint:
    """Format selected expected-survival times, retaining one risk column per curve."""

    if not isinstance(x, SurvExpSummary):
        raise TypeError("print_summary_survexp requires an expected-survival summary")
    _, digits, width = print_options(1, digits, width)
    return _report([_expected_table(x, 1, summary=True)], [None], digits, width)
