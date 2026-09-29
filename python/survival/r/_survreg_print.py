"""Accelerated-failure-time model and summary reports."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .. import _survival as _core
from ._coerce import _normalize_bool_option, _r_format_number, _r_format_numbers
from ._print import (
    character_matrix_lines,
    coefficient_matrix_lines,
    format_pvalues,
    numeric_vector_lines,
    print_options,
)
from ._survpenal_print import SurvregPenalPrint, _naprint, _signif, print_survreg_penal
from ._survreg import SurvregModelResult, survreg_summary
from ._types import ModelPrint, NamedMatrix


def _scales(
    x: Mapping[str, Any], digits: int, width: int, *, vector_digits: int | None = None
) -> list[str]:
    scales = list(x["scales"])
    if x["fixed_scale"]:
        return ["", "Scale fixed at " + " ".join(_r_format_numbers(scales, digits))]
    if len(scales) == 1:
        return ["", "Scale= " + _r_format_number(scales[0], digits)]
    return [
        "",
        "Scale:",
        *numeric_vector_lines(
            x["scale_names"], scales, digits if vector_digits is None else vector_digits, width
        ),
    ]


def _likelihood(x: Mapping[str, Any], *, summary: bool) -> list[str]:
    null, full = x["loglik"]
    lines = [
        f"Loglik(model)= {_r_format_number(round(full, 1), 7)}   "
        f"Loglik(intercept only)= {_r_format_number(round(null, 1), 7)}"
    ]
    df, chi = x["chi_df"], x["chi"]
    if df > 0:
        p = _core.pchisq(chi, df, lower_tail=False)
        text = _r_format_number(_signif(p, 2), 7) if summary else format_pvalues([p], 3)[0]
        lines.append(
            f"\tChisq= {_r_format_number(round(chi, 2), 7)} on "
            f"{_r_format_number(round(df, 1), 7)} degrees of freedom, p= {text}"
        )
    return lines


def _sample_size(x: Mapping[str, Any]) -> str:
    if x.get("na_action"):
        return f"n={x['n']} ({_naprint(x['na_action'])})"
    return f"n= {x['n']}"


def _statistics(x: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "n",
        "df",
        "idf",
        "loglik",
        "chi",
        "chi_df",
        "iter",
        "robust",
        "scales",
        "scale_names",
        "fixed_scale",
        "distribution",
        "distribution_parameters",
    )
    values = {key: copy.deepcopy(x[key]) for key in keys if key in x}
    values["pvalue"] = (
        _core.pchisq(x["chi"], x["chi_df"], lower_tail=False) if x["chi_df"] > 0 else math.nan
    )
    return values


def print_survreg(
    x: SurvregModelResult,
    digits: Any = None,
    *,
    width: Any = 80,
) -> ModelPrint | SurvregPenalPrint:
    """Return ``print.survreg``; penalized fits dispatch to ``print_survreg_penal``.

    Ordinary coefficients default to seven digits. As in R, ``digits`` changes
    coefficient and named-scale vectors; likelihood and scalar scale displays
    use seven digits. Reports do not print automatically.
    """
    if not isinstance(x, SurvregModelResult):
        raise TypeError("print_survreg requires an AFT fit")
    if x.penalized is not None:
        return print_survreg_penal(x, digits=digits, width=width)
    _, digits, width = print_options(1, digits, width, 7)
    values = x.coefficients
    names = list(x.coefficient_names)
    table = NamedMatrix(names, ["coef"], [[value] for value in values])
    missing = sum(math.isnan(value) for value in values)
    label = (
        f"Coefficients: ({missing} not defined because of singularities)"
        if missing
        else "Coefficients:"
    )
    summary = survreg_summary(x)
    summary.update(n=x.n, na_action=x.na_action, df=x.df, robust=x.robust)
    lines = ["", label, *numeric_vector_lines(names, values, digits, width)]
    lines += _scales(summary, 7, width, vector_digits=digits)
    lines += ["", *_likelihood(summary, summary=False), _sample_size(summary)]
    return ModelPrint({"coefficients": table}, _statistics(summary), digits, lines)


def _correlation_lines(table: NamedMatrix, digits: int, width: int) -> list[str]:
    size = len(table.values)
    if size <= 1:
        return []
    selected = [(i, j) for j in range(size - 1) for i in range(j + 1, size)]
    cells = _r_format_numbers([round(table.values[i][j], digits) for i, j in selected], 7)
    columns = [[""] * (size - 1) for _ in range(size - 1)]
    for (i, j), cell in zip(selected, cells, strict=True):
        columns[j][i - 1] = cell
    display = NamedMatrix(
        (table.rownames or [])[1:],
        table.colnames[:-1],
        [[float(value) for value in row[:-1]] for row in table.values[1:]],
    )
    return [
        "",
        "Correlation of Coefficients:",
        *character_matrix_lines(display, columns, width, right=False),
    ]


def print_summary_survreg(
    x: Mapping[str, Any],
    digits: Any = None,
    signif_stars: Any = False,
    *,
    width: Any = 80,
) -> ModelPrint:
    """Render an AFT ``model_summary`` with scales, likelihoods and correlations.

    The default is three digits. Robust fits show both standard errors and the
    likelihood's independence assumption. The complete correlation matrix is
    retained in ``tables['correlation']``; its lower triangle is displayed.
    """
    if not isinstance(x, Mapping) or x.get("model_type") != "survreg":
        raise TypeError("print_summary_survreg requires an AFT model summary")
    _, digits, width = print_options(1, digits, width, 3)
    stars = _normalize_bool_option(signif_stars, "signif_stars")
    robust = bool(x["robust"])
    columns = (
        ["Value", "Std. Err", "(Naive SE)", "z", "p"]
        if robust
        else ["Value", "Std. Error", "z", "p"]
    )
    keys = ["coef", "se", "naive_se", "z", "p"] if robust else ["coef", "se", "z", "p"]
    rows = x["coefficients"]
    table = NamedMatrix(
        [row["name"] for row in rows], columns, [[float(row[key]) for key in keys] for row in rows]
    )
    tables = {"coefficients": table}
    lines = coefficient_matrix_lines(table, digits, width, signif_stars=stars)
    # Older summaries can infer scale metadata from location/table dimensions.
    metadata = dict(x)
    metadata.setdefault("fixed_scale", len(x["var"]) == len(x["location_coefficients"]))
    metadata.setdefault(
        "scale_names", [row["name"] for row in rows[len(x["location_coefficients"]) :]]
    )
    lines += _scales(metadata, digits, width)
    parameters = x.get("distribution_parameters")
    description = (
        "".join(
            f"{x['distribution']} distribution: parmameters= {_r_format_number(value, 7)}"
            for value in parameters
        )
        if parameters
        else x["parms"]
    )
    lines += ["", description, *_likelihood(x, summary=True)]
    if robust:
        lines.append("(Loglikelihood assumes independent observations)")
    iterations = x["iter"] if isinstance(x["iter"], list | tuple) else [x["iter"]]
    lines.append(
        "Number of Newton-Raphson Iterations: "
        + " ".join(_r_format_numbers([math.trunc(value) for value in iterations], 7))
    )
    lines.append(_sample_size(x))
    if x.get("correlation") is not None:
        names = [row["name"] for row in rows if not math.isnan(float(row["coef"]))]
        correlation = NamedMatrix(names, list(names), [list(row) for row in x["correlation"]])
        tables["correlation"] = correlation
        lines.extend(_correlation_lines(correlation, digits, width))
    lines.append("")
    return ModelPrint(tables, _statistics(metadata), digits, lines)
