"""Person-years, survival-data consistency and marginal-means reports."""

from __future__ import annotations

import copy
import math
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_character,
    _finite_float,
    _integer_scalar,
    _is_missing_value,
    _r_format_number,
    _r_format_numbers,
)
from ._print import (
    character_matrix_lines,
    format_pvalues,
    numeric_matrix_lines,
    numeric_vector_lines,
    print_options,
)
from ._survpenal_print import _naprint
from ._types import ModelPrint, NamedMatrix, PyearsResult, SurvCheckResult, YatesPrint, YatesResult


def _total(values: Any) -> float:
    # Traverse stored cells without copying or flattening a multidimensional table.
    if isinstance(values, (list, tuple)):
        return math.fsum(_total(value) for value in values)
    return float(values)


def print_pyears(x: PyearsResult) -> ModelPrint:
    """Return person-years/event totals and the retained rate-table match summary."""
    if not isinstance(x, PyearsResult):
        raise TypeError("print_pyears requires a person-years result")
    years = x.pyears if x.data is None else x.data["pyears"]
    events = x.event if x.data is None else x.data.get("event")
    statistics = {"pyears": _total(years), "offtable": x.offtable, "observations": x.observations}
    lines = []
    if events is not None:
        statistics["event"] = _total(events)
        lines.append("Total number of events: " + _r_format_number(statistics["event"], 7))
    lines += [
        "Total number of person-years tabulated: " + _r_format_number(statistics["pyears"], 7),
        "Total number of person-years off table: " + _r_format_number(x.offtable, 7),
    ]
    # cat() inserts a separator before the original summary, which starts with a space.
    text = "" if x.summary is None else "Matches to the chosen rate table:\n   " + x.summary
    text += f"Observations in the data set: {x.observations}\n"
    lines += [line.rstrip() for line in text.splitlines()]
    if x.na_action:
        lines.append("  (" + _naprint(x.na_action) + ")")
    lines.append("")
    table = NamedMatrix(None, list(statistics), [list(statistics.values())])
    return ModelPrint(
        {"totals": table}, dict(statistics, summary=x.summary), 7, lines, primary_table="totals"
    )


def _table_lines(table: NamedMatrix, width: int, row_title: str, column_title: str) -> list[str]:
    lines = numeric_matrix_lines(table, 7, width, row_title=row_title)
    indent = max(len(row_title), max(map(len, table.rownames or []), default=0) + 2)
    return [" " * indent + column_title if line == "" else line for line in lines]


def print_survcheck(x: SurvCheckResult, *, width: Any = 80) -> ModelPrint:
    """Return transition/subject counts and all reported data-consistency problems."""
    if not isinstance(x, SurvCheckResult):
        raise TypeError("print_survcheck requires a survival-data consistency result")
    _, digits, width = print_options(1, None, width, 7)
    n = [x.n_id, x.n_observations, x.n_transitions]
    lines = numeric_vector_lines(
        ["Unique identifiers", "Observations", "Transitions"], n, digits, width
    )
    if x.na_action:
        lines.append(f"{len(x.na_action)} observations removed due to missing")
    transitions = NamedMatrix(
        list(x.transitions.from_states),
        [
            f"({getattr(x.y, 'clabel', None) or 'censor'})" if name == "(censored)" else name
            for name in x.transitions.to_states
        ],
        [list(row) for row in x.transitions.counts],
    )
    tables = {"transitions": transitions}
    # R's table class prints both dimension titles, unlike a plain numeric matrix.
    lines += ["", "Transitions table:", *_table_lines(transitions, width, "from", "to"), ""]
    if x.events is not None:
        events = NamedMatrix(
            list(x.events.states),
            [str(value) for value in x.events.count],
            [list(row) for row in x.events.subjects],
        )
        tables["events"] = events
        lines += [
            "Number of subjects with 0, 1, ... transitions to each state:",
            *(
                _table_lines(events, width, "state", "count")
                if "(any)" in (events.rownames or [])
                else numeric_matrix_lines(events, digits, width)
            ),
            "",
        ]
    problems = {}
    for name, label in [
        ("overlap", "Overlap"),
        ("gap", "Gap"),
        ("teleport", "Teleport"),
        ("jump", "Jump"),
    ]:
        problem = getattr(x, name)
        if getattr(x.flag, name) > 0 and problem is not None:
            count = len(problem.id)
            lines.append(
                f"{label} check: {count} "
                + ("id" if count == 1 else "ids")
                + f" ({len(problem.row)} rows)"
            )
            problems[name] = {"id": list(problem.id), "row": list(problem.row)}
    statistics = {
        "n": dict(x.n),
        "flags": {
            name: getattr(x.flag, name)
            for name in ("overlap", "gap", "jump", "teleport", "duplicate")
        },
        "problems": problems,
    }
    return ModelPrint(
        tables, statistics, digits, lines, primary_table="transitions", row_label="from"
    )


def _level_cells(values: list[Any], digits: int) -> list[str]:
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool) or _is_missing_value(value)
        for value in values
    ):
        return _r_format_numbers(
            [math.nan if _is_missing_value(value) else float(value) for value in values], digits
        )
    return ["NA" if _is_missing_value(value) else _as_character(value) for value in values]


def print_yates(
    x: YatesResult, digits: Any = None, dig_tst: Any = None, eps: Any = 1e-8, *, width: Any = 80
) -> YatesPrint:
    """Return marginal means alongside global, trend, pairwise or SAS tests.

    Numeric estimates and tests retain full precision; blank rows used to align
    the two display blocks do not become observations in ``as_data_frame``.
    """
    if not isinstance(x, YatesResult):
        raise TypeError("print_yates requires a marginal-means result")
    _, digits, width = print_options(1, digits, width, 5)
    dig_tst = max(1, min(5, digits - 1)) if dig_tst is None else _integer_scalar(dig_tst, "dig_tst")
    if not 1 <= dig_tst <= 22:
        raise ValueError("dig_tst must be between 1 and 22")
    eps = _finite_float(eps, "eps")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    estimates = copy.deepcopy(x.estimate)
    names = list(estimates)
    columns = [
        _level_cells(estimates[name], digits if name in {"pmm", "std"} else 7) for name in names
    ]
    has_ss = any(row.ss is not None for row in x.test)
    test_columns = ["chisq", "df"] + (["ss"] if has_ss else []) + ["Pr"]
    values = []
    for row in x.test:
        df = math.nan if row.df is None else float(row.df)
        p = (
            math.nan
            if math.isnan(df) or math.isnan(row.chisq)
            else _core.pchisq(row.chisq, df, lower_tail=False)
        )
        values.append(
            [row.chisq, df] + ([math.nan if row.ss is None else row.ss] if has_ss else []) + [p]
        )
    tests = NamedMatrix([row.name for row in x.test], test_columns, values)
    columns.append(["     " + row.name for row in x.test])
    names.append("test")
    for j, name in enumerate(test_columns):
        col = [row[j] for row in values]
        columns.append(
            format_pvalues(col, dig_tst, eps=eps)
            if name == "Pr"
            else _r_format_numbers(
                col, dig_tst if name == "chisq" else digits if name == "ss" else 7
            )
        )
        names.append(name)
    count = max(map(len, columns), default=0)
    for column in columns:
        column.extend([""] * (count - len(column)))
    # Only the shape is used by the character formatter; share its placeholder row.
    display = NamedMatrix([""] * count, names, [[0.0] * len(names)] * count)
    means = NamedMatrix(
        None,
        ["pmm", "std"],
        [
            [float(mean), float(std)]
            for mean, std in zip(estimates["pmm"], estimates["std"], strict=True)
        ],
    )
    return YatesPrint(
        {"estimates": means, "tests": tests},
        {"dig_tst": dig_tst, "eps": eps},
        digits,
        character_matrix_lines(display, columns, width),
        primary_table="estimates",
        estimates=estimates,
    )
