"""Case-cohort, proportional-hazards, concordance and survival-test reports."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, cast

from .. import _survival as _core
from ._cch import summary_cch
from ._coerce import _normalize_bool_option, _r_format_number, _r_format_numbers
from ._coxph import ClogitModel
from ._coxph_print import print_coxph
from ._print import (
    coefficient_matrix_lines,
    format_pvalues,
    numeric_matrix_lines,
    numeric_vector_lines,
    print_options,
)
from ._survpenal_print import _naprint, _signif
from ._types import (
    CchModelResult,
    ConcordanceResult,
    CoxZPHResult,
    ModelPrint,
    NamedMatrix,
    SurvConcordanceResult,
    SurvDiffResult,
)


def _rounded(table: NamedMatrix, digits: int) -> NamedMatrix:
    return NamedMatrix(
        table.rownames,
        table.colnames,
        [[round(value, digits) for value in row] for row in table.values],
    )


def _sample_size(n: int, omitted: Any, prefix: str = "") -> str:
    return f"{prefix}n={n} ({_naprint(omitted)})" if omitted else f"{prefix}n= {n}"


def _exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def print_clogit(
    x: ClogitModel, digits: Any = None, signif_stars: Any = False, *, width: Any = 80
) -> ModelPrint:
    """Return the conditional-logistic report, using the Cox report layout."""
    if not isinstance(x, ClogitModel):
        raise TypeError("print_clogit requires a conditional-logistic fit")
    return print_coxph(x, digits=digits, signif_stars=signif_stars, width=width)


def _case_cohort_report(x: Mapping[str, Any], width: int, *, summary: bool) -> ModelPrint:
    tables: dict[str, NamedMatrix] = {}
    if x["stratified"]:
        lines = [f"Exposure-stratified case-cohort analysis, {x['method']} method."]
        names = x.get("stratum_names") or [f"[,{i + 1}]" for i in range(len(x["cohort_size"]))]
        sizes = NamedMatrix(
            ["subcohort", "cohort"],
            list(names),
            [list(x["subcohort_size"]), list(x["cohort_size"])],
        )
        tables["sizes"] = sizes
        lines += numeric_matrix_lines(sizes, 7, width)
    else:
        lines = [
            f"Case-cohort analysis,x$method, {x['method']}",
            f" with subcohort of {x['subcohort_size'][0]} from cohort of {x['cohort_size'][0]}",
            "",
        ]
    rows = x["coefficients"]
    if summary:
        columns = ["Coef", "HR", "(95%", "CI)", "p"]
        values = [
            [
                row["coef"],
                _exp(row["coef"]),
                _exp(row["coef"] - 1.96 * row["se"]),
                _exp(row["coef"] + 1.96 * row["se"]),
                row["p"],
            ]
            for row in rows
        ]
    else:
        columns = ["Value", "SE", "Z", "p"]
        values = [[float(row[key]) for key in ("coef", "se", "z", "p")] for row in rows]
    table = NamedMatrix([row["name"] for row in rows], columns, values)
    tables["coefficients"] = table
    display = _rounded(table, 3) if summary else table
    # R drops the coefficient name when its case-cohort fit has one column.
    if len(rows) == 1:
        display = NamedMatrix(["Value"] if summary else None, display.colnames, display.values)
    lines += ["", "Coefficients:", *numeric_matrix_lines(display, 7, width)]
    statistics = {
        key: copy.deepcopy(x[key])
        for key in ("method", "stratified", "cohort_size", "subcohort_size")
    }
    return ModelPrint(tables, statistics, 3 if summary else 7, lines)


def print_cch(x: CchModelResult, *, width: Any = 80) -> ModelPrint:
    """Return R's case-cohort coefficient report and sampling counts."""
    if not isinstance(x, CchModelResult):
        raise TypeError("print_cch requires a case-cohort fit")
    _, _, width = print_options(1, None, width, 7)
    return _case_cohort_report(summary_cch(x), width, summary=False)


def print_summary_cch(x: Mapping[str, Any], digits: Any = 3, *, width: Any = 80) -> ModelPrint:
    """Return case-cohort hazard ratios and 1.96-SE confidence limits.

    R accepts but ignores ``digits`` here, rounding the display to three decimal
    places. The report's numeric table retains unrounded values.
    """
    if not isinstance(x, Mapping) or x.get("model_type") != "cch":
        raise TypeError("print_summary_cch requires a case-cohort summary")
    _, _, width = print_options(1, digits, width, 3)
    return _case_cohort_report(x, width, summary=True)


def print_cox_zph(
    x: CoxZPHResult, digits: Any = None, signif_stars: Any = False, *, width: Any = 80
) -> ModelPrint:
    """Return coefficient/term and global proportional-hazards test tables."""
    if not isinstance(x, CoxZPHResult):
        raise TypeError("print_cox_zph requires a proportional-hazards diagnostic")
    _, digits, width = print_options(1, digits, width, 3)
    stars = _normalize_bool_option(signif_stars, "signif_stars")
    table = NamedMatrix(
        [str(row["name"]) for row in x.table],
        ["chisq", "df", "p"],
        [[float(row[key]) for key in ("chisq", "df", "p")] for row in x.table],
    )
    return ModelPrint(
        {"tests": table},
        {"transform": x.transform},
        digits,
        coefficient_matrix_lines(table, digits, width, signif_stars=stars),
        primary_table="tests",
    )


def _count_table(
    count: dict[str, float] | list[dict[str, float]], names: list[str] | None
) -> tuple[NamedMatrix, bool]:
    vector = isinstance(count, dict)
    rows = [count] if isinstance(count, dict) else count
    columns = list(rows[0]) if rows else []
    return NamedMatrix(
        None if vector else (list(names) if names is not None else None),
        columns,
        [[float(row[key]) for key in columns] for row in rows],
    ), vector


def _number(value: float, digits: int) -> str:
    return "NaN" if math.isnan(value) else _r_format_number(value, digits)


def print_concordance(x: ConcordanceResult, digits: Any = None, *, width: Any = 80) -> ModelPrint:
    """Return concordance estimates, standard errors and pair-count tables.

    ``digits`` defaults to four: scalar estimates use significant figures,
    multiple estimates round to decimal places. Counts round to two decimals,
    then use R's seven-digit numeric layout. Kept strata remain separate rows.
    """
    if not isinstance(x, ConcordanceResult):
        raise TypeError("print_concordance requires a concordance result")
    _, digits, width = print_options(1, digits, width, 4)
    estimates = x.concordance if isinstance(x.concordance, list) else [x.concordance]
    std = x.std
    errors = std if isinstance(std, list) else [] if std is None else [std] * len(estimates)
    table = NamedMatrix(
        list(x.names) if len(estimates) > 1 and x.names is not None else None,
        ["concordance"] + (["se"] if std is not None else []),
        [
            [value] + ([float(errors[i])] if std is not None else [])
            for i, value in enumerate(estimates)
        ],
    )
    lines = [_sample_size(x.n, x.na_action)]
    if len(estimates) > 1 and std is not None:
        lines += [*numeric_matrix_lines(_rounded(table, digits), 7, width, nan_print="NaN"), ""]
    else:
        cells = _r_format_numbers(estimates, digits)
        value = " ".join(
            "NaN" if math.isnan(estimate) else cell
            for estimate, cell in zip(estimates, cells, strict=True)
        )
        if std is None:
            lines.append("Concordance=  " + value)
        else:
            lines.append("Concordance= " + value + " se= " + _number(float(errors[0]), digits))
    counts, vector = _count_table(x.count, x.names)
    display = _rounded(counts, 2)
    lines += (
        numeric_vector_lines(counts.colnames, display.values[0], 7, width)
        if vector
        else numeric_matrix_lines(display, 7, width)
    )
    return ModelPrint(
        {"concordance": table, "counts": counts},
        {"n": x.n, "variance": copy.deepcopy(x.var)},
        digits,
        lines,
        primary_table="concordance",
        row_label="predictor",
    )


def print_survConcordance(x: SurvConcordanceResult, *, width: Any = 80) -> ModelPrint:
    """Render R's deprecated concordance report without emitting a warning."""
    if not isinstance(x, SurvConcordanceResult):
        raise TypeError("print_survConcordance requires a legacy concordance result")
    _, digits, width = print_options(1, None, width, 7)
    stratified = any(isinstance(value, dict) for value in x.stats.values())
    counts, vector = (
        _count_table(list(cast(dict[str, dict[str, float]], x.stats).values()), list(x.stats))
        if stratified
        else _count_table(cast(dict[str, float], x.stats), None)
    )
    lines = [
        _sample_size(x.n, x.na_action, "  "),
        f"Concordance= {_number(x.concordance, 7)} se= {_number(x.std_err, 7)}",
    ]
    lines += (
        numeric_vector_lines(counts.colnames, counts.values[0], 7, width)
        if vector
        else numeric_matrix_lines(counts, 7, width)
    )
    return ModelPrint(
        {"counts": counts},
        {"n": x.n, "concordance": x.concordance, "std_err": x.std_err},
        digits,
        lines,
        primary_table="counts",
        row_label="strata",
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan if numerator == 0 else math.inf


def print_survdiff(x: SurvDiffResult, digits: Any = None, *, width: Any = 80) -> ModelPrint:
    """Return grouped log-rank/G-rho or one-sample expected-survival tests."""
    if not isinstance(x, SurvDiffResult):
        raise TypeError("print_survdiff requires a survival-difference test")
    _, digits, width = print_options(1, digits, width, 3)
    lines = []
    if x.na_action:
        lines += [f"n={sum(x.n)}, {_naprint(x.na_action)}.", ""]
    observed = [sum(value) if isinstance(value, list) else float(value) for value in x.obs]
    expected = [sum(value) if isinstance(value, list) else float(value) for value in x.exp]
    if len(x.n) == 1:
        difference = expected[0] - observed[0]
        sign = 1 if difference > 0 else -1 if difference < 0 else 0
        p = _core.pchisq(x.chisq, 1, lower_tail=False)
        values = [observed[0], expected[0], sign * math.sqrt(x.chisq), p]
        table = NamedMatrix(None, ["Observed", "Expected", "Z", "p"], [values])
        lines += numeric_vector_lines(
            table.colnames, [*values[:3], _signif(p, digits)], digits, width
        )
        df = 1
    else:
        table = NamedMatrix(
            list(x.groups),
            ["N", "Observed", "Expected", "(O-E)^2/E", "(O-E)^2/V"],
            [
                [float(n), o, e, _ratio((o - e) ** 2, e), _ratio((o - e) ** 2, x.var[i][i])]
                for i, (n, o, e) in enumerate(zip(x.n, observed, expected, strict=True))
            ],
        )
        lines += numeric_matrix_lines(table, digits, width, nan_print="NaN")
        df = sum(value > 0 for value in expected) - 1
        p = _core.pchisq(x.chisq, df, lower_tail=False)
        lines += [
            "",
            f" Chisq= {_r_format_number(round(x.chisq, 1), digits)}  on {df} degrees of freedom, "
            f"p= {format_pvalues([p], max(1, digits - 2))[0]}",
        ]
    return ModelPrint(
        {"test": table},
        {"chisq": x.chisq, "df": df, "pvalue": p},
        digits,
        lines,
        primary_table="test",
        row_label="group",
    )
