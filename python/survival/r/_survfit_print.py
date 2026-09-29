"""Compact survival-curve reports using the native mean/median kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import _survival as _core
from ._coerce import _finite_float, _integer_scalar, _normalize_bool_option, _r_format_number
from ._print import numeric_matrix_lines
from ._survfit import (
    _cox_curve_labels,
    _cox_engines,
    _coxms_engine,
    _coxms_table_labels,
    _engine_of,
    _rmean_option,
    _summary_table,
)
from ._types import (
    CoxSurvfitMultiStateResult,
    CoxSurvfitResult,
    NamedMatrix,
    SurvfitMultiStateResult,
    SurvfitResult,
)

_MULTISTATE = (SurvfitMultiStateResult, CoxSurvfitMultiStateResult)
_CurveFit = SurvfitResult | CoxSurvfitResult | SurvfitMultiStateResult | CoxSurvfitMultiStateResult


@dataclass(frozen=True)
class SurvfitPrint:
    """R's compact curve report, with numbers available separately from text.

    ``table`` contains the displayed rows and columns at full precision;
    ``rmean_endtime`` contains the scaled integration cutoffs, when requested.
    ``lines`` holds the formatted table and footnote. ``str(report)`` renders
    them with a trailing newline. Construction does not write to stdout.
    Original R call expressions and omission notices are not reconstructed.
    """

    table: NamedMatrix
    rmean_endtime: list[float] | None
    digits: int
    lines: list[str]

    def __str__(self) -> str:
        return "\n".join(self.lines) + "\n"


def _options(scale: Any, digits: Any, width: Any, default_digits: int) -> tuple[float, int, int]:
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


def print_survfit(
    x: _CurveFit,
    scale: Any = 1,
    digits: Any = None,
    print_rmean: Any = None,
    rmean: Any = None,
    *,
    width: Any = 80,
) -> SurvfitPrint:
    """Return R's compact ``print.survfit`` table and formatted text.

    Ordinary curves show sample sizes, events, medians and their fitted limits.
    ``rmean`` adds restricted means and standard errors: ``"common"`` uses the
    latest time, ``"individual"`` each stratum's endpoint, and a number supplies
    the cutoff. ``print_rmean=True`` is the legacy spelling for ``rmean="common"``.
    The default is ``"none"``. ``scale`` divides reported times, including the
    cutoff footnote; numeric cutoffs remain in original units.

    Multistate inputs dispatch to ``print_survfitms`` (restricted means by default).
    ``digits`` defaults to 3 for ordinary curves and 7 for multistate curves,
    matching R with default options. ``width`` controls column-block wrapping.
    The native kernels compute the compact table without expanding summary rows.
    """

    if isinstance(x, _MULTISTATE):
        return print_survfitms(x, scale=scale, rmean=rmean, digits=digits, width=width)
    if not isinstance(x, SurvfitResult | CoxSurvfitResult):
        raise TypeError("print_survfit requires a fitted survival curve")
    scale, digits, width = _options(scale, digits, width, 3)
    if rmean is None:
        enabled = (
            False if print_rmean is None else _normalize_bool_option(print_rmean, "print_rmean")
        )
        rmean = "common" if enabled else "none"
    option = _rmean_option(rmean, x)
    engines = _cox_engines(x) if isinstance(x, CoxSurvfitResult) else [_engine_of(x)]
    if (
        isinstance(x, CoxSurvfitResult)
        and option not in {"none", "common", "individual"}
        and float(option) < min(x.time)
    ):
        # A conditional Cox curve allows a cutoff at start.time before its
        # first observation. Add the zero-time row the native KM validator
        # needs; counts and the mean table otherwise remain unchanged.
        engines = [_core.survfit0(engine) for engine in engines]
    means = [_core.survmean(engine, scale, option) for engine in engines]
    labels = _cox_curve_labels(x) if isinstance(x, CoxSurvfitResult) else x.strata_names
    table = _summary_table(
        means, labels, n_id=getattr(x, "n_id", None) is not None, conf_int=x.conf_int
    )
    columns = list(table.colnames)
    selected = list(range(len(columns)))
    values = table.values
    # R drops redundant count columns, but retains the record count when n.id
    # is present: subjects may contribute multiple observation intervals.
    if columns[1] == "n.id":
        selected.remove(2)
        columns[1] = "n"
    else:
        if all(row[1] == row[2] for row in values):
            selected.remove(2)
            columns[1] = "n"
        if all(row[0] == row[1] for row in values):
            selected.remove(0)
    table = NamedMatrix(
        table.rownames,
        ["rmean*" if columns[j] == "rmean" else columns[j] for j in selected],
        [[row[j] for j in selected] for row in values],
    )
    lines = numeric_matrix_lines(table, digits, width)
    ends = None if option == "none" else list(means[0].end_time)
    if ends is not None:
        if option == "individual":
            lines.append("   * restricted mean with variable upper limit")
        else:
            lines.append(
                "    * restricted mean with upper limit =  " + _r_format_number(ends[0], digits)
            )
    return SurvfitPrint(table, ends, digits, lines)


def print_survfitms(
    x: SurvfitMultiStateResult | CoxSurvfitMultiStateResult,
    scale: Any = 1,
    rmean: Any = None,
    *,
    digits: Any = None,
    width: Any = 80,
) -> SurvfitPrint:
    """Return counts and restricted mean time in each state, as ``print.survfitms``.

    Rows vary over strata first, prediction rows second, then states. The default
    cutoff is the latest time across curves. ``rmean="none"`` reports counts only;
    ``"individual"`` and numeric cutoffs work as in ``print_survfit``. Standard
    errors appear only when the fit stores area-under-curve uncertainty.
    """

    if not isinstance(x, _MULTISTATE):
        raise TypeError("print_survfitms requires fitted multistate curves")
    scale, digits, width = _options(scale, digits, width, 7)
    # R validates the cutoff after adding the origin row with survfit0. The
    # native table kernel already inserts it; avoid copying a whole fit here.
    option = _rmean_option(rmean, x, include_origin=True)
    if isinstance(x, CoxSurvfitMultiStateResult):
        engine = _coxms_engine(x, "print")
        values, ends, columns = engine.mean_table_data(x.pstate, x.p0, scale=scale, rmean=option)
        labels = _coxms_table_labels(x)
    else:
        values, ends, columns = _engine_of(x).mean_table(scale=scale, rmean=option)
        labels = (
            [f"{group}, {state}" for state in x.states for group in x.strata_names]
            if x.strata and len(x.strata) > 1
            else list(x.states)
        )
    if ends:
        columns[-1] += "*"
    table = NamedMatrix(labels, columns, values)
    lines = numeric_matrix_lines(table, digits, width)
    if ends:
        lines.append(
            "   *restricted mean time in state (max time = "
            + _r_format_number(ends[0], digits)
            + " )"
            if len(ends) == 1
            else "   *restricted mean time in state (per curve cutoff)"
        )
    return SurvfitPrint(table, ends or None, digits, lines)
