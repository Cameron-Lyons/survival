"""Structured Cox model reports, with R-compatible tables and text."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .. import _survival as _core
from ._coerce import _integer_scalar, _normalize_bool_option, _r_format_number, _r_format_numbers
from ._coxph import CoxphModel, _coefficient_table, summary_coxph_penal
from ._coxphms import CoxphmsModel
from ._print import (
    character_matrix_lines,
    coefficient_matrix_lines,
    format_pvalues,
    numeric_matrix_lines,
    print_options,
)
from ._survpenal_print import _naprint, _signif
from ._types import ModelPrint, NamedMatrix


def _table(summary: Mapping[str, Any], *, direct: bool = False) -> NamedMatrix:
    rows = summary["coefficients"]
    penal = summary["model_type"] == "coxph.penal"
    robust = "robust se" in summary["coefficient_columns"]
    keys = (
        ["coef", "se", "se2", "chisq", "df", "p"]
        if penal
        else ["coef", "exp_coef", "naive_se", "robust_se", "z", "p"]
        if robust
        else ["coef", "exp_coef", "se", "z", "p"]
    )
    columns = list(summary["coefficient_columns"])
    if direct:
        columns[-1] = "p"
    return NamedMatrix(
        [row["name"] for row in rows], columns, [[float(row[key]) for key in keys] for row in rows]
    )


def _tables(summary: Mapping[str, Any]) -> dict[str, NamedMatrix]:
    tables = {"coefficients": _table(summary)}
    if summary.get("conf_int"):
        level = "." + _r_format_number(round(100 * float(summary.get("conf_level", 0.95)), 2), 7)
        rows = summary["conf_int"]
        tables["conf_int"] = NamedMatrix(
            [row["name"] for row in rows],
            ["exp(coef)", "exp(-coef)", f"lower {level}", f"upper {level}"],
            [
                [float(row[key]) for key in ("exp(coef)", "exp(-coef)", "lower", "upper")]
                for row in rows
            ],
        )
    return tables


def _report(
    tables: dict[str, NamedMatrix], summary: Mapping[str, Any], digits: int, lines: list[str]
) -> ModelPrint:
    fields = (
        "n",
        "nevent",
        "n_id",
        "loglik",
        "null_loglik",
        "iter",
        "df",
        "logtest",
        "waldtest",
        "sctest",
        "robscore",
        "concordance",
        "used_robust",
        "conf_level",
    )
    return ModelPrint(
        tables,
        {key: copy.deepcopy(summary[key]) for key in fields if key in summary},
        digits,
        lines,
    )


def _counts(summary: Mapping[str, Any], *, direct: bool = False) -> list[str]:
    line = ("n= " if direct else "  n= ") + str(summary["n"])
    if direct and summary.get("n_id") is not None:
        line += f", unique id= {summary['n_id']}"
    if summary.get("nevent") is not None:
        line += f", number of events= {summary['nevent']}"
    result = [line]
    if summary.get("na_action"):
        result.append("   (" + _naprint(summary["na_action"]) + ")")
    return result


def _concordance(summary: Mapping[str, Any], digits: int) -> list[str]:
    value = summary.get("concordance")
    if value is None:
        return []
    c = _r_format_number(round(value["C"], 3), digits)
    se = _r_format_number(round(value["se(C)"], 3), digits)
    return [f"Concordance= {c}  (se = {se} )"]


def _test_line(
    label: str,
    test: Mapping[str, float],
    digits: int,
    *,
    direct: bool = False,
    full_p: bool = False,
) -> str:
    value = _r_format_number(round(test["test"], 2), digits)
    df = _r_format_number(test["df"], digits)
    p = format_pvalues([test["pvalue"]], digits if full_p else max(1, digits - 4))[0]
    gap, pgap = ("", " ") if direct else (" ", "   ")
    return f"{label}={gap}{value}  on {df} df,{pgap}p={p}"


def _transition_tables(
    table: NamedMatrix, cmap: NamedMatrix, *, share: Any = None, drop_empty: bool = False
) -> list[tuple[str, NamedMatrix]]:
    groups: dict[tuple[int, ...], list[str]] = {}
    for j, name in enumerate(cmap.colnames):
        indices = tuple(int(row[j]) for row in cmap.values)
        if drop_empty and not any(indices):
            continue
        groups.setdefault(indices, []).append(name)
    names = cmap.rownames or []
    if share is not None:
        names = [
            name + ("*" if kind == 2 else "") for name, kind in zip(names, share.vtype, strict=True)
        ]
    return [
        (
            ", ".join(labels),
            NamedMatrix(
                [names[i] for i, index in enumerate(indices) if index],
                list(table.colnames),
                [list(table.values[index - 1]) for index in indices if index],
            ),
        )
        for indices, labels in groups.items()
    ]


def _states(states: Any, *, direct: bool = False) -> str:
    return (" States:  " if direct else " States: ") + ", ".join(
        f"{i}= {name}" for i, name in enumerate(states, 1)
    )


def print_coxph(
    x: CoxphModel, digits: Any = None, signif_stars: Any = False, *, width: Any = 80
) -> ModelPrint:
    """Return ``print.coxph`` tables and text; dispatch penalized and null fits.

    ``digits`` defaults to 4 (3 for penalized fits). Multistate fits group shared
    coefficients by transition. ``width`` controls column-block wrapping.
    """
    if not isinstance(x, CoxphModel):
        raise TypeError("print_coxph requires a Cox fit")
    stars = _normalize_bool_option(signif_stars, "signif_stars")
    if x.penalized is not None:
        return print_coxph_penal(x, digits=digits, width=width)
    if not x.coefficients:
        return print_coxph_null(x, digits=digits, width=width)
    _, digits, width = print_options(1, digits, width, 4)
    columns, rows = _coefficient_table(x, 1)
    df = sum(not math.isnan(value) for value in x.coefficients)
    statistic = 2 * (x.loglik[1] - x.loglik[0])
    summary: dict[str, Any] = {
        "model_type": "coxph",
        "coefficient_columns": columns,
        "coefficients": rows,
        "n": x.n,
        "nevent": x.nevent,
        "na_action": x.na_action,
        "df": df,
        "loglik": x.loglik[1],
        "null_loglik": x.loglik[0],
        "logtest": {
            "test": statistic,
            "df": df,
            "pvalue": _core.pchisq(statistic, df, lower_tail=False),
        },
    }
    table = _table(summary, direct=True)
    lines = []
    if isinstance(x, CoxphmsModel):
        summary["n_id"] = x.n_id
        selected = [j for j, name in enumerate(table.colnames) if name != "se(coef)"]
        table = NamedMatrix(
            table.rownames,
            [table.colnames[j] for j in selected],
            [[row[j] for j in selected] for row in table.values],
        )
        extra: list[str] = []
        for label, group in _transition_tables(table, x.cmap, share=x.share, drop_empty=True):
            if len(label) > 20:
                short = chr(65 + len(extra))
                extra.extend([f"{short}: {label}", f"{short}:"])
                label = short
            lines.extend(
                coefficient_matrix_lines(group, digits, width, signif_stars=stars, row_title=label)
            )
            lines.append("")
        lines.append(_states(x.states, direct=True))
        lines.extend(extra)
        if x.share is not None and any(kind == 2 for kind in x.share.vtype):
            lines.append(" (*) = coef for proportional baselines")
    else:
        lines = coefficient_matrix_lines(table, digits, width, signif_stars=stars)
    lines += [
        "",
        _test_line("Likelihood ratio test", summary["logtest"], digits, direct=True, full_p=True),
    ]
    lines.extend(_counts(summary, direct=True))
    return _report({"coefficients": table}, summary, digits, lines)


def print_coxph_null(x: CoxphModel, digits: Any = None, *, width: Any = 80) -> ModelPrint:
    """Return the null-model report. R ignores ``digits`` here and uses 7."""
    if not isinstance(x, CoxphModel) or x.coefficients or x.penalized is not None:
        raise TypeError("print_coxph_null requires an unpenalized null Cox fit")
    print_options(1, digits, width, 3)
    line = f"  n= {x.n}"
    if x.na_action:
        line = f"  n={x.n} ({_naprint(x.na_action)})"
    return ModelPrint(
        {},
        {"n": x.n, "loglik": x.loglik[0]},
        7,
        ["Null model", "  log likelihood= " + _r_format_number(x.loglik[0], 7), line],
    )


def print_summary_coxph(
    x: Mapping[str, Any] | CoxphModel,
    digits: Any = None,
    signif_stars: Any = True,
    expand: Any = False,
    *,
    width: Any = 80,
) -> ModelPrint:
    """Render ``model_summary(fit)`` like ``print.summary.coxph``.

    ``expand=True`` groups multistate tables by transition; expanded tables omit
    significance marks, as R does. A null summary is the unchanged null fit.
    """
    if isinstance(x, CoxphModel) and not x.coefficients and x.penalized is None:
        return print_coxph_null(x, digits=digits, width=width)
    if not isinstance(x, Mapping) or x.get("model_type") not in {"coxph", "coxph.penal"}:
        raise TypeError("print_summary_coxph requires a Cox model summary")
    stars = _normalize_bool_option(signif_stars, "signif_stars")
    expanded = _normalize_bool_option(expand, "expand")
    if x["model_type"] == "coxph.penal":
        return print_summary_coxph_penal(x, digits=digits, signif_stars=stars, width=width)
    _, digits, width = print_options(1, digits, width, 4)
    tables = _tables(x)
    lines = _counts(x)
    if not tables["coefficients"].values:
        return _report(tables, x, digits, [*lines, "   Null model"])
    if expanded and x.get("cmap") is not None:
        coefficients = _transition_tables(tables["coefficients"], x["cmap"])
        intervals = (
            _transition_tables(tables["conf_int"], x["cmap"]) if "conf_int" in tables else []
        )
        for i, (label, table) in enumerate(coefficients):
            lines.extend(coefficient_matrix_lines(table, digits, width, row_title=label))
            if intervals:
                lines.extend(numeric_matrix_lines(intervals[i][1], digits, width, row_title=label))
        lines += ["", _states(x["states"])]
    else:
        lines += [
            "",
            *coefficient_matrix_lines(tables["coefficients"], digits, width, signif_stars=stars),
        ]
        if "conf_int" in tables:
            lines += ["", *numeric_matrix_lines(tables["conf_int"], digits, width)]
    lines += ["", *_concordance(x, digits)]
    for key, label in (
        ("logtest", "Likelihood ratio test"),
        ("waldtest", "Wald test            "),
        ("sctest", "Score (logrank) test "),
    ):
        if key in x:
            lines.append(_test_line(label, x[key], digits))
    if x.get("robscore"):
        robust = x["robscore"]
        lines[-1] += ",   Robust = " + _r_format_number(round(robust["test"], 2), digits)
        lines[-1] += "  p=" + format_pvalues([robust["pvalue"]], max(1, digits - 4))[0]
    lines.append("")
    if x.get("used_robust"):
        lines += [
            "  (Note: the likelihood ratio and score tests assume independence of",
            "     observations within a cluster, the Wald and robust score tests do not).",
        ]
    return _report(tables, x, digits, lines)


def _maxlabel(value: Any) -> int:
    value = _integer_scalar(value, "maxlabel")
    if value < 1:
        raise ValueError("maxlabel must be positive")
    return value


def _penal_footer(x: Mapping[str, Any], digits: int, *, direct: bool) -> list[str]:
    outer, inner = x["iter"]
    lines = ["", f"Iterations: {outer} outer, {inner} Newton-Raphson"]
    lines.extend("     " + line for line in x["print2"])
    lines.append(
        "Degrees of freedom for terms= "
        + " ".join(_r_format_numbers([round(df, 1) for df in x["df"]], digits))
    )
    if not direct:
        lines.extend(_concordance(x, digits))
    test = dict(x["logtest"])
    test["df"] = round(sum(x["df"]), 2)
    test["pvalue"] = _core.pchisq(test["test"], test["df"], lower_tail=False)
    lines.append(_test_line("Likelihood ratio test", test, digits, direct=direct))
    lines.extend(_counts(x, direct=True) if direct else [""])
    return lines


def print_coxph_penal(
    x: CoxphModel, terms: Any = False, maxlabel: Any = 25, digits: Any = None, *, width: Any = 80
) -> ModelPrint:
    """Return penalized Cox term tests, iteration counts, effective df and history.

    ``terms=True`` combines ordinary multicolumn terms into Wald-test rows.
    Only displayed coefficient labels are truncated to ``maxlabel`` characters.
    """
    if not isinstance(x, CoxphModel) or x.penalized is None:
        raise TypeError("print_coxph_penal requires a penalized Cox fit")
    maxlabel = _maxlabel(maxlabel)
    _, digits, width = print_options(1, digits, width, 3)
    summary = summary_coxph_penal(x, conf_int=False, terms=terms, _print_digits=digits)
    table = _table(summary)
    display = NamedMatrix(
        [name[:maxlabel] for name in table.rownames or []], table.colnames, table.values
    )
    lines = coefficient_matrix_lines(display, digits, width, na_print="")
    lines += _penal_footer(summary, digits, direct=True)
    return _report({"coefficients": table}, summary, digits, lines)


def print_summary_coxph_penal(
    x: Mapping[str, Any],
    digits: Any = None,
    signif_stars: Any = True,
    maxlabel: Any = 25,
    *,
    width: Any = 80,
) -> ModelPrint:
    """Render a penalized Cox summary. R accepts but ignores ``signif_stars``."""
    if not isinstance(x, Mapping) or x.get("model_type") != "coxph.penal":
        raise TypeError("print_summary_coxph_penal requires a penalized Cox summary")
    _normalize_bool_option(signif_stars, "signif_stars")
    maxlabel = _maxlabel(maxlabel)
    _, digits, width = print_options(1, digits, width, 4)
    tables = _tables(x)
    table = tables["coefficients"]
    display = NamedMatrix(
        [name[:maxlabel] for name in table.rownames or []], table.colnames, table.values
    )
    columns = []
    for j in range(6):
        values = [row[j] for row in table.values]
        rounded = [
            _signif(value, 2) if j == 5 else round(value, 2) if j >= 3 else value
            for value in values
        ]
        columns.append(
            [
                "" if math.isnan(value) else cell
                for value, cell in zip(values, _r_format_numbers(rounded, digits), strict=True)
            ]
        )
    lines = [*_counts(x), "", *character_matrix_lines(display, columns, width, right=False)]
    if "conf_int" in tables:
        lines += ["", *numeric_matrix_lines(tables["conf_int"], digits, width)]
    lines += _penal_footer(x, digits, direct=False)
    return _report(tables, x, digits, lines)
