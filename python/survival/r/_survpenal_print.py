"""R's ``print.survreg.penal`` (``print.survreg.penal.R``): the per-term table of a
penalized ``survreg`` fit and the lines R prints under it.  R has no
``summary.survreg.penal``; this table is what ``print(fit)`` shows.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _integer_scalar,
    _normalize_bool_option,
    _r_format_number,
    _r_format_numbers,
)
from ._coxph import _block, _pspline_print, _wald_row, coxph_wtest
from ._formula import _design_term_name
from ._survreg import SurvregModelResult, survreg_df
from ._types import NaAction, _PenaltyDesignTerm

_COLUMNS = ("coef", "se(coef)", "se2", "Chisq", "DF", "p")
_ROW_KEYS = ("coef", "se", "se2", "chisq", "df", "p")
_WIDTH = 80  # options(width)


@dataclass(frozen=True)
class SurvregPenalPrint:
    """``print.survreg.penal``'s output.

    ``rows`` holds the numbers of the table (one list per row of ``rownames``, in the
    order of ``columns``; NaN where R prints a blank), ``history`` the printfun lines
    (``"Theta= 0.924"``), ``logtest`` the likelihood ratio test on ``logtest_df``
    (``sum(df) - idf``) degrees of freedom and ``lines`` what R prints from the table
    header to the ``n=`` line, formatted to ``digits``.  ``str()`` adds the Call block,
    ``survreg(formula = ...)`` only: the fit does not record R's other arguments.
    """

    formula: str | None
    rownames: list[str]
    columns: tuple[str, ...]
    rows: list[list[float]]
    history: list[str]
    scale: list[float]
    scale_names: list[str]
    fixed_scale: bool
    iter: list[int]
    df: list[float]
    logtest: float
    logtest_df: float
    logtest_p: float
    n: int
    na_action: NaAction | None
    digits: int
    lines: list[str]

    def __str__(self) -> str:
        call = [] if self.formula is None else ["Call:", f"survreg(formula = {self.formula})", ""]
        return "\n".join([*call, *self.lines]) + "\n"


def _term_table(
    fit: SurvregModelResult, penalized: Any, terms: bool, digits: int
) -> tuple[list[str], list[list[float]], list[str]]:
    """print.survreg.penal's loop over ``pterms`` (lines 40-76) of the fit's
    ``SurvpenalFit``: the row names, the rows and the printfun history lines."""

    coef = fit.coefficients
    var, var2 = fit.var, penalized.var2
    splines = {
        _design_term_name(term): term
        for term in (fit.design.covariates if fit.design is not None else ())
        if isinstance(term, _PenaltyDesignTerm) and term.penalized and term.kind == "pspline"
    }
    histories = {entry.term: entry for entry in penalized.history}
    names: list[str] = []
    rows: list[list[float]] = []
    history: list[str] = []
    for i, pterm in enumerate(penalized.pterms):
        label, columns, df = fit.assign2_labels[i], penalized.assign2[i], penalized.df[i]
        term_coef = [coef[j] for j in columns]
        if pterm and label in splines:
            spline_rows, text = _pspline_print(
                label,
                splines[label],
                term_coef,
                _block(var, columns),
                _block(var2, columns),
                df,
                histories[i],
                digits,
            )
            names.extend(row["name"] for row in spline_rows)
            rows.extend([row[key] for key in _ROW_KEYS] for row in spline_rows)
            history.append(text)
        elif terms and len(columns) > 1:
            # the p-value is on 1 df whatever the DF column says, as in R
            test = coxph_wtest(_block(var, columns), term_coef).test[0]
            names.append(label)
            rows.append(
                [math.nan, math.nan, math.nan, test, df, _core.pchisq(test, 1.0, lower_tail=False)]
            )
        else:
            for j in columns:
                row = _wald_row(fit.coefficient_names[j], coef[j], var[j][j], var2[j][j])
                names.append(row["name"])
                rows.append([row[key] for key in _ROW_KEYS])
    return names, rows, history


def _pow_di(x: float, n: int) -> float:
    """R's ``R_pow_di`` (arithmetic.c): ``x^n`` by repeated squaring."""

    result = 1.0
    negative = n < 0
    n = abs(n)
    while True:
        if n & 1:
            result *= x
        n >>= 1
        if not n:
            break
        x *= x
    return 1.0 / result if negative else result


def _signif(value: float, digits: int) -> float:
    """R's ``signif``, ported from ``fprec`` (nmath/fprec.c): scale by a power of
    ten and round half to even, so ``signif(0.000125, 2)`` is ``0.00012``.  Within
    ``1e-306 < |value| < 1e306`` it matches R bit for bit; beyond that R's powers of
    ten differ from these in the last bits, which can move the result by an ulp or,
    next to ``DBL_MAX``, by one in the last kept digit."""

    if not math.isfinite(value) or value == 0.0:
        return value
    max10e = sys.float_info.max_10_exp
    dig = min(max(digits, 1), 22)
    sign = -1.0 if value < 0.0 else 1.0
    x = abs(value)
    l10 = math.log10(x)
    e10 = dig - 1 - math.floor(l10)
    if l10 < max10e - 2:
        p10 = 1.0
        if e10 > max10e:
            p10 = _pow_di(10.0, e10 - max10e)
            e10 = max10e
        if e10 > 0:
            pow10 = _pow_di(10.0, e10)
            return sign * (round(x * pow10 * p10) / pow10) / p10
        pow10 = _pow_di(10.0, -e10)
        return sign * round(x / pow10) * pow10
    do_round = max10e - l10 >= _pow_di(10.0, -dig)
    e2 = dig + (1 if e10 > 0 else 6) - 22
    p10 = _pow_di(10.0, e2)
    big = _pow_di(10.0, e10 - e2)
    x = x * p10 * big
    if do_round:
        x += 0.5
    return sign * (math.floor(x) / p10) / big


def _format_column(values: Sequence[float], digits: int) -> list[str]:
    """``format(column)`` with the NA cells blanked (``ifelse(is.na(print1), "", temp)``)."""

    return [
        "" if math.isnan(value) else cell
        for value, cell in zip(values, _r_format_numbers(values, digits), strict=True)
    ]


def _character_matrix_lines(
    rownames: Sequence[str], header: Sequence[str], columns: Sequence[Sequence[str]]
) -> list[str]:
    """``print(<character matrix>, quote = FALSE)``: left-justified row names and
    columns, one space apart.  The columns go into blocks that fit ``options(width)``;
    each block prints the header and every row."""

    name_width = max(map(len, rownames), default=0)
    widths = [
        max(len(title), *(len(cell) for cell in cells))
        for title, cells in zip(header, columns, strict=True)
    ]
    lines: list[str] = []
    start = 0
    while start < len(columns):
        end, used = start + 1, name_width + 1 + widths[start]
        while end < len(columns) and used + 1 + widths[end] < _WIDTH:
            used += 1 + widths[end]
            end += 1
        block = range(start, end)
        lines.append(" " * name_width + "".join(" " + header[j].ljust(widths[j]) for j in block))
        lines.extend(
            name.ljust(name_width) + "".join(" " + columns[j][row].ljust(widths[j]) for j in block)
            for row, name in enumerate(rownames)
        )
        start = end
    return lines


def _named_vector_lines(names: Sequence[str], values: Sequence[float], digits: int) -> list[str]:
    """``print(<named numeric>)``: right-justified names over their values, as many
    per line as ``options(width)`` holds."""

    cells = _r_format_numbers(values, digits)
    width = max(*map(len, names), *map(len, cells))
    per_line = max(1, _WIDTH // (width + 1))
    lines: list[str] = []
    for start in range(0, len(cells), per_line):
        lines.append(" ".join(name.rjust(width) for name in names[start : start + per_line]))
        lines.append(" ".join(cell.rjust(width) for cell in cells[start : start + per_line]))
    return lines


def _format_pval(p: float, digits: int) -> str:
    """R's ``format.pval`` of one p-value (``eps = .Machine$double.eps``)."""

    if math.isnan(p):
        return "NA"
    if p < sys.float_info.epsilon:
        digits = max(1, digits - 2)
        return "<" + ("" if digits == 1 else " ") + _r_format_number(sys.float_info.epsilon, digits)
    return _r_format_number(p, digits)


def _naprint(na_action: NaAction) -> str:
    """``naprint`` of an ``na.omit`` or ``na.exclude`` record."""

    count = len(na_action)
    noun = "observation" if count == 1 else "observations"
    return f"{count} {noun} deleted due to missingness"


def print_survreg_penal(
    fit: SurvregModelResult, terms: Any = False, maxlabel: Any = 25, digits: Any | None = None
) -> SurvregPenalPrint:
    """R's ``print.survreg.penal``: a row per coefficient (coef, se from ``var``, se2 from
    ``var2``, a Wald chi-square on 1 df), pspline()'s linear and nonlinear parts, and with
    ``terms`` one Wald row for every other multi-column term.  Row names are cut at
    ``maxlabel`` characters; ``digits`` defaults to R's ``max(options()$digits - 4, 3)``.
    """

    penalized = fit.penalized if isinstance(fit, SurvregModelResult) else None
    if penalized is None:
        raise TypeError("Invalid object")
    term_rows = _normalize_bool_option(terms, "terms")
    label_width = _integer_scalar(maxlabel, "maxlabel")
    digits = 3 if digits is None else _integer_scalar(digits, "digits")
    if not fit.coefficients:
        raise ValueError("Penalized fits must have an intercept!")

    names, rows, history = _term_table(fit, penalized, term_rows, digits)
    rownames = [name[:label_width] for name in names]
    coef, se, se2, chisq, df_column, p = (list(column) for column in zip(*rows, strict=True))
    cells = [
        _format_column(coef, digits),
        _format_column(se, digits),
        _format_column(se2, digits),
        _format_column([round(value, 2) for value in chisq], digits),
        _format_column([round(value, 2) for value in df_column], digits),
        _format_column([_signif(value, 2) for value in p], digits),
    ]
    lines = _character_matrix_lines(rownames, _COLUMNS, cells)

    scale = list(fit.scale)
    scale_names = list(fit.strata_levels) if len(scale) > 1 else []
    fixed_scale = len(fit.var) == len(fit.coefficients)
    if fixed_scale:
        lines += ["", "Scale fixed at " + " ".join(_r_format_numbers(scale, digits))]
    elif len(scale) == 1:
        lines += ["", "Scale= " + _r_format_number(scale[0], digits)]
    else:
        lines += ["", "Scale:", *_named_vector_lines(scale_names, scale, digits)]

    outer, inner = penalized.iter
    lines += ["", f"Iterations: {outer} outer, {inner} Newton-Raphson"]
    lines += ["     " + text for text in history]
    df = list(penalized.df)
    lines.append(
        "Degrees of freedom for terms= "
        + " ".join(_r_format_numbers([round(value, 1) for value in df], digits))
    )
    loglik = fit.loglik
    logtest = -2.0 * (loglik[0] - loglik[1])
    logtest_df = survreg_df(fit) - fit.idf
    logtest_p = _core.pchisq(logtest, logtest_df, lower_tail=False)
    # format.pval at pdig = max(1, digits - 4) (print.survreg.penal.R:114)
    test = (
        f"Likelihood ratio test={_r_format_number(round(logtest, 2), digits)}"
        f"  on {_r_format_number(round(logtest_df, 1))} df,"
        f" p={_format_pval(logtest_p, max(1, digits - 4))}"
    )
    n = fit.n
    if fit.na_action is not None and len(fit.na_action):
        lines += [test, f"  n={n} ({_naprint(fit.na_action)})"]
    else:
        lines.append(f"{test}  n= {n}")

    return SurvregPenalPrint(
        formula=fit.formula,
        rownames=rownames,
        columns=_COLUMNS,
        rows=rows,
        history=history,
        scale=scale,
        scale_names=scale_names,
        fixed_scale=fixed_scale,
        iter=[outer, inner],
        df=df,
        logtest=logtest,
        logtest_df=logtest_df,
        logtest_p=logtest_p,
        n=n,
        na_action=fit.na_action,
        digits=digits,
        lines=[line.rstrip() for line in lines],
    )
