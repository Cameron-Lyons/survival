"""``pyears``, ``survexp`` and the rate-table helpers.

Ports of the R code of ``R/pyears.R``, ``R/survexp.R``, ``R/ratetableDate.R`` and
``R/is.ratetable.R``: the model frame, the ``rmap`` expansion, the term
categories (factors, ``tcut``, ``cut``) and the result labelling; the person-years
tabulation, ``match.ratetable`` and the expected-survival curves are the
``pyears``, ``match_ratetable`` and ``survexp`` kernels.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC
from datetime import date as _Date
from datetime import datetime as _DateTime
from datetime import timedelta as _TimeDelta
from itertools import pairwise
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _as_character,
    _categories,
    _factor,
    _factor_levels,
    _finite_float,
    _float_vector,
    _floats_or_nan,
    _is_missing_value,
    _match_string_arg,
    _materialize_1d,
    _materialize_labels,
    _normalize_bool_option,
    _normalize_positive_scale,
    _pop_dotted_keyword,
    _r_factor,
    _r_format_number,
    _rows_of,
    _scalar_or_vector,
    _subset_sequence,
)
from ._fit import _excluded_rows, _pad_rows
from ._formula import (
    _arithmetic_expression_values,
    _call_arguments,
    _column,
    _column_source,
    _covariate_term_name,
    _data_column_names,
    _data_row_count,
    _expression_columns,
    _expression_values,
    _formula_columns,
    _formula_name,
    _formula_rhs_terms,
    _literal_vector,
    _model_strata,
    _model_variables,
    _numeric_scalar,
    _numeric_vector,
    _r_literal,
    _response_spec,
    _seq_length,
    _term_values,
    _unsupported_formula_name,
    model_frame,
)
from ._surv import Surv
from ._types import (
    ModelFrame,
    NaAction,
    PyearsResult,
    RateTable,
    StrataFactor,
    SurvExpResult,
    SurvExpSummary,
    TcutResult,
    _CovariateTerm,
    _InteractionTerm,
    _ModelCovariateTerm,
    _ModelStrataTerm,
)

# ---------------------------------------------------------------------------
# Rate tables
# ---------------------------------------------------------------------------

_RATETABLE_ATTRIBUTES = ("dims", "dimid", "dimnames", "cutpoints", "types")


@dataclass(frozen=True)
class RateTableSummary:
    """Rate-table dimensions, canonical attributes and the native summary text."""

    dimensions: dict[str, list[Any]]
    attributes: dict[str, Any]
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class RateTableMatch:
    """Matched rows in rate-table dimension order, cutpoints and a built-in summary.

    Categorical positions are one-based; continuous positions keep their units
    and dates count days since 1970-01-01. US calendar positions are unadjusted.
    """

    r: list[list[float]]
    dimid: list[str]
    cutpoints: list[list[float] | None]
    summary: str | None


def match_ratetable(data: Any, ratetable: RateTable) -> RateTableMatch:
    """Match named data columns to a rate table, validating declared factor levels.

    Accept a mapping or data frame. Required columns may be numeric, categorical,
    labels or dates; unrelated columns are ignored. Missing values must be removed
    before matching, as required by the numerical population kernels.
    """
    if not isinstance(ratetable, RateTable):
        raise TypeError("Invalid rate table")
    matched = _match_rate_columns(data, ratetable)
    positions = matched.r
    return RateTableMatch(
        positions,
        ratetable.dimid,
        matched.cutpoints,
        _population_match_summary(ratetable, positions),
    )


def summary_ratetable(object: RateTable, **_kwargs: Any) -> RateTableSummary:
    """Describe each rate-table dimension and retain its canonical attributes.

    The dimension table gives levels for factors and lower/upper cutpoints
    otherwise. Date boundaries are ISO dates; numeric boundaries keep the
    original units. ``str(result)`` is the Rust rate-table summary text.
    """

    if not isinstance(object, RateTable):
        raise TypeError("Argument is not a rate table")
    dims, names, labels = object.dims, object.dimid, object.dimnames
    cuts, types = object.cutpoints, object.type_codes()
    lower: list[Any] = []
    upper: list[Any] = []
    for kind, values in zip(types, cuts, strict=True):
        bounds: list[Any] = [None, None] if not values else [values[0], values[-1]]
        if values and kind > 2:
            dates = (_core.days_to_date(value) for value in bounds)
            bounds = [f"{day.year:04d}-{day.month:02d}-{day.day:02d}" for day in dates]
        lower.append(bounds[0])
        upper.append(bounds[1])
    return RateTableSummary(
        dimensions={
            "dimension": names,
            "type": types,
            "categories": dims,
            "levels": [
                level if kind == 1 else None for kind, level in zip(types, labels, strict=True)
            ],
            "lower": lower,
            "upper": upper,
        },
        attributes={
            "dim": dims,
            "dimid": names,
            "dimnames": labels,
            "cutpoints": cuts,
            "type": types,
            "class": "ratetable",
        },
        text=str(object),
    )


def is_ratetable(x: Any, verbose: bool = False) -> bool | list[str]:
    """R's ``is.ratetable``: a ``RateTable``, or a mapping of R's ratetable attributes.

    With ``verbose`` the structural problems are returned instead of ``False``.
    """

    _normalize_bool_option(verbose, "verbose")
    if isinstance(x, RateTable):
        return True
    if isinstance(x, Mapping) and set(_RATETABLE_ATTRIBUTES) <= set(x):
        rates = x.get("rates")
        check = _core.is_ratetable(
            [int(value) for value in x["dims"]],
            [str(value) for value in x["dimid"]],
            [[str(label) for label in labels] for labels in x["dimnames"]],
            [None if cuts is None else [float(value) for value in cuts] for cuts in x["cutpoints"]],
            [int(value) for value in x["types"]],
            int(x.get("n_rates", 0 if rates is None else len(rates))),
        )
        if check.valid:
            return True
        return list(check.messages) if verbose else False
    return ["wrong class"] if verbose else False


_EPOCH_ORDINAL = _Date(1970, 1, 1).toordinal()
_ONE_DAY = _TimeDelta(days=1)


def _is_numpy_scalar(value: Any, name: str) -> bool:
    value_type = type(value)
    return value_type.__module__ == "numpy" and value_type.__name__ == name


def _day_count(value: Any) -> float:
    """A number as ``match.ratetable`` ``unclass``es it: a time difference counts days."""

    if type(value) is float or type(value) is int:
        return float(value)
    if isinstance(value, _TimeDelta):
        return value / _ONE_DAY
    if _is_missing_value(value):
        return math.nan
    if _is_numpy_scalar(value, "timedelta64"):
        return float(value / type(value)(1, "D"))
    return float(value)


def _ratetable_day(value: Any) -> float:
    """``ratetableDate`` of one value: dates become days since 1970-01-01, numbers pass."""

    # pandas' NaT is a date instance, so only a plain date skips the missing check
    if type(value) is _Date:
        return float(value.toordinal() - _EPOCH_ORDINAL)
    if _is_missing_value(value):
        return math.nan
    if isinstance(value, _DateTime) and value.utcoffset() is not None:
        # R's as.Date.POSIXct uses the UTC calendar date by default.
        value = value.astimezone(UTC)
    if isinstance(value, _Date):
        return float(value.toordinal() - _EPOCH_ORDINAL)
    if _is_numpy_scalar(value, "datetime64"):
        return float(value.astype("datetime64[D]").astype("int64"))
    if isinstance(value, str):
        return float(_Date.fromisoformat(value[:10]).toordinal() - _EPOCH_ORDINAL)
    return _day_count(value)


def ratetableDate(x: Any) -> float | list[float]:
    """R's ``ratetableDate``: dates (``date``, ``datetime``, ISO strings) to day counts.

    Numbers are R's default method and pass through unchanged.
    """

    if isinstance(x, str | _Date | _DateTime) or not hasattr(x, "__iter__"):
        return _ratetable_day(x)
    return [_ratetable_day(value) for value in _materialize_1d(x, "x")]


def survexp_us() -> RateTable:
    """R's ``survexp.us`` rate table."""

    return _core.survexp_us()


def survexp_usr() -> RateTable:
    """R's ``survexp.usr`` rate table."""

    return _core.survexp_usr()


def survexp_mn() -> RateTable:
    """R's ``survexp.mn`` rate table."""

    return _core.survexp_mn()


def _rmap_columns(
    rmap: Mapping[str, Any] | None, ratetable: RateTable, data: Any
) -> dict[str, Any]:
    """R's ``rcall`` expansion: every rate-table dimension from ``rmap`` or a same-named column."""

    return _mapped_columns(rmap, ratetable.dimid, data)


def _mapped_columns(
    rmap: Mapping[str, Any] | None, names: Sequence[str], data: Any
) -> dict[str, Any]:
    """``rmap``'s entries, then a same-named column for each other variable in *names*.

    A string naming a column of *data* stays that column.  An R constant (``21915``,
    ``"white"``) is its value, and an expression reading columns of *data*
    (``ageyr * 365.25``, or ``accept_dt - birth_dt`` with the dates as days since
    1970-01-01) is evaluated there, as R evaluates ``rmap`` in the model frame.  Any
    other string (a word such as ``white``, or ``1995-03-01``) is a constant label, and
    so is any other scalar.
    """

    columns: dict[str, Any] = {}
    n = _data_row_count(data)
    available = set(_data_column_names(data) or ())
    for name, value in ({} if rmap is None else rmap).items():
        if str(name) not in names:
            raise ValueError(f"Variable not found in the ratetable:{name}")
        if isinstance(value, str) and value not in available:
            value = _rmap_value(str(name), value, data, n)
        elif not isinstance(value, str):
            if not hasattr(value, "__iter__"):
                value = [value] * n
            else:
                raw = _materialize_1d(value, str(name))
                value = _rows_of(value, raw * n if len(raw) == 1 else raw)
        columns[str(name)] = value
    for dimid in names:
        columns.setdefault(dimid, dimid)
    return columns


def _rmap_value(name: str, text: str, data: Any, n: int) -> list[Any]:
    """The values of the ``rmap`` entry *name*, a string *text* naming no column."""

    literal = _r_literal(text)
    sign, rest = text.strip()[:1], text.strip()[1:]
    if literal is None and sign in {"-", "+"}:
        # a signed R number, such as -365.25, is a constant too
        number = _r_literal(rest)
        if isinstance(number, float):
            literal = -number if sign == "-" else number
    if literal is not None:
        return [literal] * n
    word, quoted = _formula_name(text)
    if not quoted and not _unsupported_formula_name(word, quoted):
        return [text] * n
    try:
        used = _expression_columns(text)
    except ValueError as exc:
        raise ValueError(f"rmap {name} = {text}: {exc}") from exc
    if not used:
        # a string reading no column, such as "1995-03-01", is a label as a quoted R
        # string is (not the arithmetic 1995 - 3 - 1)
        return [text] * n
    values = {}
    for column in used:
        raw = _column(data, column)
        values[column] = [_ratetable_day(value) for value in raw] if _is_date_column(raw) else raw
    return _expression_values(values, text, n)


def _is_date_column(values: Sequence[Any]) -> bool:
    """``match.ratetable``'s ``datecheck``: the column holds dates or date-times."""

    first = next((value for value in values if not _is_missing_value(value)), None)
    return isinstance(first, _Date) or _is_numpy_scalar(first, "datetime64")


def _rate_positions(mf: ModelFrame, ratetable: RateTable) -> list[list[float]]:
    """R's ``match.ratetable(rdata, ratetable)$R`` for the model frame's rate variables.

    Labels go to the kernel as strings.  As in R, dates are an error in a factor or
    continuous dimension (type 1 or 2) and become days since 1970-01-01 in a date
    dimension (type 3 or 4); time differences (R's ``difftime``) count days.
    """

    return _match_rate_columns(mf.extra, ratetable).r


def _match_rate_columns(data: Any, ratetable: RateTable) -> _core.MatchRatetableResult:
    """Shared date/factor coercion for public matching and population model frames."""
    names = ratetable.dimid
    available = _data_column_names(data)
    if available is None:
        raise TypeError("data must be a mapping or data frame with named columns")
    for name in names:
        if name not in available:
            raise ValueError(f"Argument '{name}' needed by the ratetable was not found in the data")
        if available.count(name) > 1 or names.count(name) > 1:
            raise ValueError("A ratetable argument appears twice in the data")
    types = ratetable.type_codes()
    sources = [_column_source(data, name) for name in names]
    values = [_materialize_1d(source, name) for name, source in zip(names, sources, strict=True)]
    misplaced = [
        name
        for name, code, column in zip(names, types, values, strict=True)
        if code < 3 and _is_date_column(column)
    ]
    if misplaced:
        raise ValueError(
            "Data has a date type variable, but the reference ratetable is not a date "
            "variable: " + " ".join(misplaced)
        )
    columns: list[list[str] | list[float]] = []
    for dimension, (name, code, source, column) in enumerate(
        zip(names, types, sources, values, strict=True)
    ):
        if _categories(source) is not None:
            codes, labels = _factor(source, name)
            matched = ratetable.match_levels(dimension, labels)
            columns.append(
                [math.nan if index is None else float(matched[index]) for index in codes]
            )
        elif all(isinstance(value, str) or _is_missing_value(value) for value in column) and any(
            isinstance(value, str) for value in column
        ):
            if any(_is_missing_value(value) for value in column):
                raise ValueError(f"The variable {name} contains missing values")
            columns.append([str(value) for value in column])
        elif code > 2:
            columns.append([_ratetable_day(value) for value in column])
        else:
            columns.append([_day_count(value) for value in column])
    return _core.match_ratetable(ratetable, names, columns)


def _ratetable_argument(ratetable: Any) -> RateTable:
    if ratetable is None:
        return _core.survexp_us()
    if isinstance(ratetable, RateTable):
        return ratetable
    from ._coxphms import CoxphmsModel

    if isinstance(ratetable, CoxphmsModel):
        # pyears.R and survexp.R refuse a coxphms fit before their coxph branch
        raise ValueError("Invalid rate table")
    if hasattr(ratetable, "coefficients") or hasattr(ratetable, "linear_predictors"):
        raise NotImplementedError("a coxph fit as the ratetable is not supported yet")
    raise ValueError("Invalid rate table")


# ---------------------------------------------------------------------------
# pyears
# ---------------------------------------------------------------------------


def _breaks_vector(expression: str, data: Any) -> list[float]:
    """A ``tcut()``/``cut()`` ``breaks`` argument as numbers: a literal vector,
    ``as.Date(...)`` as days since 1970-01-01, or a column of *data* read whole (R
    evaluates the terms before ``subset`` and ``na.action`` remove rows)."""

    text = expression.strip()
    if text.startswith("as.Date(") and text.endswith(")"):
        return [_ratetable_day(value) for value in _literal_vector(text[8:-1])]
    name, quoted = _formula_name(text)
    if quoted or name in (_data_column_names(data) or ()):
        return _floats_or_nan(_column(data, name))
    return _numeric_vector(_literal_vector(text))


def _logical_argument(arguments: Mapping[str, str], name: str, default: bool) -> bool:
    """A logical argument of a formula call, as R's ``if`` reads it."""

    if name not in arguments:
        return default
    values = _literal_vector(arguments[name])
    if len(values) != 1 or isinstance(values[0], str) or math.isnan(values[0]):
        raise ValueError(f"'{name}' must be TRUE or FALSE")
    return bool(values[0])


def _required_arguments(call: str) -> dict[str, str]:
    arguments = _call_arguments(call)
    for name in ("x", "breaks"):
        if name not in arguments:
            raise ValueError(f'argument "{name}" is missing, with no default')
    return arguments


def _format_break(value: float, digits: int) -> str:
    """``formatC(0 + value, digits, width = 1)``: C's ``%g``, R's ``Inf``."""

    if math.isinf(value):
        return " Inf" if value > 0 else "-Inf"
    return f"{0.0 + value:.{digits}g}"


def _r_cut(
    x: Sequence[float],
    breaks: Sequence[float],
    labels: Sequence[Any] | None,
    include_lowest: bool,
    right: bool,
    dig_lab: int,
) -> tuple[list[float], list[str] | None]:
    """R's ``cut.default``: the one-based interval code of each value (NaN outside the
    breaks) and the level labels, ``None`` for ``labels = FALSE``.

    A single ``breaks`` value is the number of intervals, spread over the range of *x*
    widened by a thousandth; default labels take the fewest digits from ``dig.lab`` up
    that tell the breaks apart.
    """

    if len(breaks) == 1:
        if math.isnan(breaks[0]) or breaks[0] < 2:
            raise ValueError("invalid number of intervals")
        count = int(breaks[0] + 1)
        present = [value for value in x if not math.isnan(value)]
        if not present:
            raise ValueError("'from' must be a finite number")
        low, high = min(present), max(present)
        width = high - low
        if width == 0.0:
            width = abs(low) if low != 0.0 else 1.0
            breaks = _seq_length(low - width / 1000, high + width / 1000, count)
        else:
            breaks = _seq_length(low, high, count)
            breaks[0], breaks[-1] = low - width / 1000, high + width / 1000
    else:
        breaks = sorted(value for value in breaks if not math.isnan(value))
    count = len(breaks)
    if len(set(breaks)) < count:
        raise ValueError("'breaks' are not unique")
    if labels is None:
        levels = [f"Range_{k}" for k in range(1, count)]
        for digits in range(dig_lab, max(12, dig_lab) + 1):
            formatted = [_format_break(value, digits) for value in breaks]
            if all(a != b for a, b in pairwise(formatted)):
                left, closing = ("(", "]") if right else ("[", ")")
                levels = [f"{left}{a},{b}{closing}" for a, b in pairwise(formatted)]
                if include_lowest and right:
                    levels[0] = "[" + levels[0][1:]
                elif include_lowest:
                    levels[-1] = levels[-1][:-1] + "]"
                break
    elif len(labels) == 1 and labels[0] is False:
        levels = None
    elif len(labels) != count - 1:
        raise ValueError("number of intervals and length of 'labels' differ")
    else:
        levels = [_as_character(label) for label in labels]
    # .bincode: (b[k-1], b[k]] when right, [b[k-1], b[k]) otherwise; include.lowest
    # closes the first (right) or last interval
    locate = bisect_left if right else bisect_right
    lowest, highest = breaks[0], breaks[-1]
    codes: list[float] = []
    for value in x:
        if math.isnan(value) or value < lowest or value > highest:
            codes.append(math.nan)
            continue
        code = locate(breaks, value)
        if code == 0:  # the first break, left out of (b[0], b[1]]
            code = 1 if include_lowest else 0
        elif code == count:  # the last break, left out of [b[-2], b[-1])
            code = count - 1 if include_lowest else 0
        codes.append(float(code) if code else math.nan)
    if levels is not None and len(set(levels)) < len(levels):
        # factor() merges intervals that share a label
        merged = list(dict.fromkeys(levels))
        position = [merged.index(level) + 1.0 for level in levels]
        codes = [code if math.isnan(code) else position[int(code) - 1] for code in codes]
        levels = merged
    return codes, levels


@dataclass(frozen=True)
class _CallTerm:
    """A ``tcut()`` or ``cut()`` term evaluated on the whole data, as R's
    ``model.frame`` evaluates it before ``subset`` and ``na.action``: the scaled times
    of a ``tcut`` with its level labels and cutpoints, or the interval codes of a
    ``cut`` (NaN outside the breaks: a missing value) with its labels, ``None`` for
    ``labels = FALSE``."""

    values: list[float]
    levels: list[str] | None
    cuts: list[float] | None = None


def _tcut_call(call: str, data: Any, n: int) -> _CallTerm:
    """R's ``tcut(x, breaks, labels, scale = 1)``."""

    arguments = _required_arguments(call)
    labels = (
        [_as_character(value) for value in _literal_vector(arguments["labels"])]
        if "labels" in arguments
        else None
    )
    scale = _numeric_scalar(_literal_vector(arguments.get("scale", "1")), "scale")
    result = _core.tcut(
        _arithmetic_expression_values(data, arguments["x"], n),
        _breaks_vector(arguments["breaks"], data),
        labels,
        scale,
    )
    return _CallTerm(list(result.values), list(result.labels), list(result.cutpoints))


def _cut_call(call: str, data: Any, n: int) -> _CallTerm:
    """R's ``cut(x, breaks, labels = NULL, include.lowest = FALSE, right = TRUE,
    dig.lab = 3, ordered_result = FALSE)``; the order of an ordered result does not
    change ``pyears``' table."""

    arguments = _required_arguments(call)
    labels = _literal_vector(arguments.get("labels", "NULL")) or None
    dig_lab = _numeric_scalar(_literal_vector(arguments.get("dig.lab", "3")), "dig.lab")
    _logical_argument(arguments, "ordered_result", False)
    codes, levels = _r_cut(
        _arithmetic_expression_values(data, arguments["x"], n),
        _breaks_vector(arguments["breaks"], data),
        labels,
        _logical_argument(arguments, "include.lowest", False),
        _logical_argument(arguments, "right", True),
        int(dig_lab),
    )
    return _CallTerm(codes, levels)


def _pyears_calls(formula: str, data: Any) -> dict[str, _CallTerm]:
    """The ``tcut()`` and ``cut()`` terms of *formula*, evaluated on the whole *data*."""

    terms = _formula_rhs_terms(formula, data).covariates
    if any(isinstance(term, _InteractionTerm) for term in terms):
        raise ValueError("Pyears cannot have interaction terms")
    texts = [term.call for term in terms if isinstance(term, _CovariateTerm) and term.call]
    n = _data_row_count(data, formula) if texts else 0
    calls: dict[str, _CallTerm] = {}
    for text in texts:
        function = text.partition("(")[0]
        if function == "tcut":
            calls[text] = _tcut_call(text, data, n)
        elif function == "cut":
            calls[text] = _cut_call(text, data, n)
        else:
            raise ValueError(f"unsupported pyears term {text}")
    return calls


@dataclass(frozen=True)
class _PyearsTerm:
    """One right-hand-side term of ``pyears`` as a category dimension.

    ``factor`` is 0 for a time-based ``tcut`` term (``cuts`` holds its cutpoints
    and ``values`` the raw times) and 1 for a factor (``values`` its one-based codes).
    """

    label: str
    factor: int
    values: list[float]
    levels: list[str]
    cuts: list[float]


def _factor_call_term(mf: ModelFrame, term: _CovariateTerm, label: str, data: Any) -> _PyearsTerm:
    """A ``factor(x)`` or ``as.factor(x)`` formula term.  model.frame evaluates the call
    on the whole *data* before ``subset`` and ``na.action`` remove rows, so the levels
    are the whole column's: every declared one for ``as.factor``, those in use for
    ``factor``."""

    whole = _column_source(data, term.column)
    levels = _factor_levels(whole, label)
    if term.categorical_wrapper == "factor":
        used = {
            value for value in _materialize_labels(whole, label) if not _is_missing_value(value)
        }
        levels = [level for level in levels if level in used]
    index = {level: code + 1.0 for code, level in enumerate(levels)}
    codes = [
        math.nan if _is_missing_value(value) else index[value]
        for value in _column(mf.data, term.column)
    ]
    return _PyearsTerm(label, 1, codes, [_as_character(level) for level in levels], [])


def _pyears_term(
    mf: ModelFrame, term: _CovariateTerm, data: Any, calls: Mapping[str, _CallTerm]
) -> _PyearsTerm:
    label = _covariate_term_name(term)
    if term.call is not None:
        call = calls[term.call]
        # the model frame carries the values at the rows subset and na.action kept
        values = _floats_or_nan(mf.extra[term.call])
        if call.cuts is not None:
            return _PyearsTerm(label, 0, values, call.levels or [], call.cuts)
        if call.levels is not None:
            return _PyearsTerm(label, 1, values, call.levels, [])
        return _factor_term(label, values)  # cut(labels = FALSE): as.factor of the codes
    if term.categorical_wrapper is not None:
        return _factor_call_term(mf, term, label, data)
    source = _column_source(mf.data, term.column) if term.arithmetic is None else None
    if isinstance(source, TcutResult):
        return _PyearsTerm(
            label, 0, list(source.values), list(source.labels), list(source.cutpoints)
        )
    # pyears' as.factor keeps every declared level of a factor column, empty or not
    if source is None or term.transform is not None:
        return _factor_term(label, _term_values(mf.data, term, mf.n))
    return _factor_term(label, source)


def _factor_term(label: str, values: Any) -> _PyearsTerm:
    """``as.factor(values)`` as a category dimension."""

    codes, levels = _factor(values, label)
    return _PyearsTerm(label, 1, [math.nan if c is None else c + 1.0 for c in codes], levels, [])


def _pyears_terms(mf: ModelFrame, data: Any, calls: Mapping[str, _CallTerm]) -> list[_PyearsTerm]:
    """The category dimensions (the model frame has no interaction terms)."""

    for columns in mf.terms.strata:
        raise ValueError(f"unsupported pyears term strata({columns})")
    return [
        _pyears_term(mf, term, data, calls)
        for term in mf.terms.covariates
        if isinstance(term, _CovariateTerm)
    ]


def _pyears_followup(
    mf: ModelFrame, has_ratetable: bool
) -> tuple[
    list[float] | np.ndarray, list[float] | np.ndarray | None, list[float] | np.ndarray | None
]:
    """R's ``Y`` checks: (stop, start, event) of the follow-up."""

    response = mf.response
    if response is None:
        if mf.y is None:
            raise ValueError("Follow-up time must appear in the formula")
        if isinstance(mf.y, np.ndarray):
            if np.any(mf.y < 0):
                raise ValueError("Negative follow up time")
            if mf.y.shape[1] > 2:
                raise ValueError("Y has too many columns")
            if mf.y.shape[1] == 0:
                raise ValueError("Y must have at least one column")
            if mf.y.shape[1] == 2:
                if has_ratetable:
                    return mf.y[:, 1], mf.y[:, 0], None
                return mf.y[:, 0], None, mf.y[:, 1]
            return mf.y[:, 0], None, None
        if any(value < 0.0 for value in mf.y):
            raise ValueError("Negative follow up time")
        return list(mf.y), None, None
    if not isinstance(response, Surv):
        raise ValueError("Only right-censored and counting process survival types are supported")
    if response.type == "right":
        if any(value < 0.0 for value in response.time):
            raise ValueError("Negative survival time")
        nzero = sum(
            1 for t, s in zip(response.time, response.event, strict=True) if t == 0 and s == 1
        )
        if nzero > 0:
            warnings.warn(
                f"{nzero} observations with an event and 0 follow-up time, any rate "
                "calculations are statistically questionable",
                stacklevel=3,
            )
    elif response.type != "counting":
        raise ValueError("Only right-censored and counting process survival types are supported")
    event = [math.nan if value is None else float(value) for value in response.event]
    return list(response.time), None if response.start is None else list(response.start), event


def _row_major(cells: Sequence[Any], dims: Sequence[int]) -> Any:
    """R's array over ``dims`` (column-major ``cells``) as a row-major nested list."""

    if not dims:
        return cells[0]
    strides = [1]
    for extent in dims[:-1]:
        strides.append(strides[-1] * extent)

    def build(prefix: int, depth: int) -> Any:
        if depth == len(dims):
            return cells[prefix]
        return [build(prefix + k * strides[depth], depth + 1) for k in range(dims[depth])]

    return build(0, 0)


def _reshape(values: Sequence[float] | None, dims: Sequence[int]) -> Any:
    """:func:`_row_major` of a numeric table, ``None`` passing through."""

    return None if values is None else _row_major([float(value) for value in values], dims)


def _cell_frame(
    tables: Mapping[str, Sequence[Any]], dimnames: Mapping[str, Sequence[str]]
) -> dict[str, list[Any]]:
    """R's ``data.frame = TRUE`` layout of the column-major ``tables``: one row per cell
    with person-years (every cell without terms), a column per term then the tables
    (pyears.R: ``expand.grid(dimnames)[pyears > 0, ]``)."""

    pyears = tables["pyears"]
    cells = (
        [cell for cell, value in enumerate(pyears) if value > 0.0]
        if dimnames
        else list(range(len(pyears)))
    )
    frame: dict[str, list[Any]] = {}
    stride = 1
    for label, levels in dimnames.items():
        frame[label] = [levels[(cell // stride) % len(levels)] for cell in cells]
        stride *= len(levels)
    for name, values in tables.items():
        frame[name] = [values[cell] for cell in cells]
    return frame


def _pyears_frame(result: Any, terms: Sequence[_PyearsTerm]) -> dict[str, list[Any]]:
    """The ``data.frame = TRUE`` layout of the kernel's result."""

    # each getter copies the whole table out of the kernel result: read it once
    tables = {"pyears": result.pyears, "n": result.n}
    if result.expected is not None:
        tables["expected"] = result.expected
    if result.event is not None:
        tables["event"] = result.event
    return _cell_frame(tables, {term.label: term.levels for term in terms})


def _population_retention_options(model: Any, x: Any, y: Any) -> tuple[bool, bool, bool]:
    """R keeps either the complete frame or the requested X/Y components."""

    if _normalize_bool_option(model, "model"):
        return True, False, False
    return False, _normalize_bool_option(x, "x"), _normalize_bool_option(y, "y")


def _population_retention_rows(extra: dict[str, Any], n: int, keep: bool) -> str | None:
    """Carry original row indices through the existing subset/NA path."""

    if not keep:
        return None
    name = "_population_model_rows"
    while name in extra:
        name += "_"
    extra[name] = list(range(n))
    return name


def _population_term_labels(mf: ModelFrame) -> list[str]:
    labels = []
    for term in mf.terms.model_terms:
        if isinstance(term, _ModelCovariateTerm):
            labels.append(_covariate_term_name(term.term))
        elif isinstance(term, _ModelStrataTerm):
            labels.append(term.spec.call)
    return labels


def _rmap_source_names(
    rmap: Mapping[str, Any] | None, names: Sequence[str], data: Any
) -> list[str]:
    """The columns R adds to the model frame via ``all.vars(rcall)``."""

    entries = dict(rmap or {})
    for name in names:
        entries.setdefault(name, name)
    available = set(_data_column_names(data) or ())
    sources: dict[str, None] = {}
    for value in entries.values():
        if not isinstance(value, str):
            continue
        if value in available:
            sources[value] = None
            continue
        try:
            used = _expression_columns(value)
        except ValueError:
            # A mapping may contain a literal category or date label.
            continue
        sources.update((name, None) for name in used if name in available)
    return list(sources)


def _population_variable_overrides(
    mf: ModelFrame,
    data: Any,
    calls: Mapping[str, _CallTerm] | None = None,
) -> dict[str, Any]:
    """Preserve factor levels and tcut metadata in evaluated formula columns."""

    overrides: dict[str, Any] = {}
    for term in mf.terms.covariates:
        if not isinstance(term, _CovariateTerm):
            continue
        label = _covariate_term_name(term)
        if term.call is not None and calls is not None and term.call in calls:
            call = calls[term.call]
            values = mf.extra[term.call]
            if call.cuts is not None:
                overrides[label] = _core.tcut(values, call.cuts, call.levels, 1.0)
            elif call.levels is not None:
                overrides[label] = _r_factor(
                    [None if _is_missing_value(v) else call.levels[int(v) - 1] for v in values],
                    call.levels,
                )
            else:
                overrides[label] = list(values)
        elif term.categorical_wrapper is not None:
            factor = _factor_call_term(mf, term, label, data)
            overrides[label] = _r_factor(
                [None if math.isnan(v) else factor.levels[int(v) - 1] for v in factor.values],
                factor.levels,
            )
        elif term.transform is None and term.arithmetic is None:
            source = _column_source(mf.data, term.column)
            overrides[label] = _rows_of(source, _materialize_labels(source, label))
    return overrides


def _population_model_frame(
    mf: ModelFrame,
    data: Any,
    row_key: str,
    rmap: Mapping[str, Any] | None,
    rate_names: Sequence[str],
    calls: Mapping[str, _CallTerm] | None = None,
) -> dict[str, Any]:
    """Snapshot the evaluated formula columns and the original rmap source columns."""

    overrides = _population_variable_overrides(mf, data, calls)
    frame: dict[str, Any] = {}
    if mf.response is not None:
        frame[mf.response_name or "response"] = mf.response
    elif mf.y is not None:
        frame[mf.response_name or "response"] = (
            mf.y.tolist() if isinstance(mf.y, np.ndarray) else list(mf.y)
        )
    frame.update(_model_variables(mf, overrides))
    rows = [int(row) for row in mf.extra[row_key]]
    for name in _rmap_source_names(rmap, rate_names, data):
        if name not in frame:
            frame[name] = _subset_sequence(_column_source(data, name), rows, name)
    if mf.weights is not None:
        frame["(weights)"] = list(mf.weights)
    return frame


def _pyears_result(
    result: Any, terms: Sequence[_PyearsTerm], data_frame: bool, na_action: NaAction | None
) -> PyearsResult:
    dims = list(result.dims) if terms else []
    dimnames = {term.label: list(term.levels) for term in terms}
    has_tcut = any(term.factor == 0 for term in terms)
    if data_frame:
        return PyearsResult(
            pyears=None,
            n=None,
            offtable=float(result.offtable),
            observations=int(result.observations),
            tcut=has_tcut,
            dim=dims,
            dimnames=dimnames,
            event=None,
            expected=None,
            data=_pyears_frame(result, terms),
            na_action=na_action,
        )
    return PyearsResult(
        pyears=_reshape(result.pyears, dims),
        n=_reshape(result.n, dims),
        offtable=float(result.offtable),
        observations=int(result.observations),
        tcut=has_tcut,
        dim=dims,
        dimnames=dimnames,
        event=_reshape(result.event, dims),
        expected=_reshape(result.expected, dims),
        na_action=na_action,
    )


def _population_match_summary(table: RateTable, positions: list[list[float]]) -> str | None:
    """The built-in table's summary of matched data, before US birthday adjustment."""
    source = table.source
    if source not in {"survexp.us", "survexp.usr", "survexp.mn"}:
        return None
    age_low = year_low = math.inf
    age_high = year_high = -math.inf
    male = female = white = black = 0
    for row in positions:
        age_low, age_high = min(age_low, row[0]), max(age_high, row[0])
        year_low, year_high = min(year_low, row[-1]), max(year_high, row[-1])
        male += row[1] == 1
        female += row[1] == 2
        if source == "survexp.usr":
            white += row[2] == 1
            black += row[2] == 2
    if positions:
        dates = [_core.days_to_date(math.floor(value)) for value in (year_low, year_high)]
        first, last = [f"{day.year:04d}-{day.month:02d}-{day.day:02d}" for day in dates]
    else:
        first, last = "Inf", "-Inf"
    low, high = [_r_format_number(round(value / 365.25, 1), 7) for value in (age_low, age_high)]
    indent = "  " if source == "survexp.mn" else "    "
    text = (
        f" age ranges from {low} to {high} years\n"
        f"{indent}male: {male}  female: {female} \n"
        f"{indent}date of entry from {first} to {last} \n"
    )
    if source == "survexp.usr":
        text += f"    white: {white}  black: {black} \n"
    return text


def _pyears_direct(
    response: Any,
    time: Any,
    start: Any,
    stop: Any,
    event: Any,
    group: Any,
    weights: Any,
    subset: Any,
    na_action: Any,
    scale: float,
    data_frame: bool,
    retention: tuple[bool, bool, bool],
) -> PyearsResult:
    """``pyears`` on vectors (the reticulate bridge's entry): one ``group`` category."""

    columns: dict[str, Any] = {}
    if isinstance(response, Surv):
        if response.type not in {"right", "counting"}:
            raise ValueError(
                "Only right-censored and counting process survival types are supported"
            )
        columns["time"] = list(response.time)
        columns["event"] = list(response.event)
        if response.start is not None:
            columns["start"] = list(response.start)
    else:
        follow_up = response if response is not None else (stop if stop is not None else time)
        if follow_up is None:
            raise ValueError("Follow-up time must appear in the formula")
        columns["time"] = _float_vector(follow_up, "time")
        if start is not None and stop is not None:
            columns["start"] = _float_vector(start, "start")
        if event is not None:
            columns["event"] = event
    n = len(columns["time"])
    if "event" in columns:
        lhs = "Surv(start, time, event)" if "start" in columns else "Surv(time, event)"
    else:
        if "start" in columns:
            columns["time"] = [
                b - a for a, b in zip(columns["start"], columns["time"], strict=True)
            ]
        lhs = "time"
    if group is None:
        rhs = "1"
    else:
        columns["group"] = _materialize_labels(group, "group")
        if len(columns["group"]) != n:
            raise ValueError("group must have the same length as the response")
        rhs = "group"
    return pyears(
        f"{lhs} ~ {rhs}",
        columns,
        weights=weights,
        subset=subset,
        na_action=na_action,
        scale=scale,
        data_frame=data_frame,
        model=retention[0],
        x=retention[1],
        y=retention[2],
    )


def pyears(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    rmap: Mapping[str, Any] | None = None,
    ratetable: Any | None = None,
    scale: Any = 365.25,
    expect: str = "event",
    model: bool = False,
    x: bool = False,
    y: bool = False,
    data_frame: bool = False,
    *,
    time: Any = None,
    start: Any = None,
    stop: Any = None,
    event: Any = None,
    group: Any = None,
    **kwargs: Any,
) -> PyearsResult:
    """R's ``pyears``: person-years, events and expected events over a category table.

    ``formula`` is ``Surv(time, status) ~ tcut(...) + factor`` (or ``time ~ ...``);
    ``rmap`` maps rate-table dimensions to columns or vectors.  A ``Surv`` object or
    time vector as ``formula`` with ``group``/``time``/``start``/``stop``/``event``
    keywords tabulates plain vectors (the reticulate bridge's call).
    ``model=True`` retains the evaluated model frame. Otherwise, ``x=True``
    retains the grouping codes and raw ``tcut`` times (ones without groups),
    and ``y=True`` retains the ``Surv`` response or a numeric matrix.

    A numeric matrix column (``Y ~ group``) or ``cbind(time, event)`` supplies
    one or two response columns. Without a rate table, two columns are time and
    numeric event count; with a rate table, they are start and stop without
    events. Matrix responses are not recoded as binary ``Surv`` statuses.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, None)
    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        raise TypeError(f"pyears got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    scale_value = _normalize_positive_scale(scale)
    data_frame_value = _normalize_bool_option(data_frame, "data.frame")
    retention = _population_retention_options(model, x, y)
    expect_value = _match_string_arg(
        expect, "expect", ("event", "pyears"), "expect must be event or pyears"
    )
    if not isinstance(formula, str):
        return _pyears_direct(
            formula,
            time,
            start,
            stop,
            event,
            group,
            weights,
            subset,
            na_action,
            scale_value,
            data_frame_value,
            retention,
        )
    table = None if ratetable is None and rmap is None else _ratetable_argument(ratetable)
    if rmap is not None and ratetable is None:
        raise ValueError("No rate table specified")
    calls = _pyears_calls(formula, data)
    # the rate variables and the tcut()/cut() values go through subset and na.action
    # with the formula's variables, so a cut() value outside the breaks drops its row
    extra = {} if table is None else _rmap_columns(rmap, table, data)
    extra.update((call, term.values) for call, term in calls.items())
    row_key = _population_retention_rows(extra, _data_row_count(data, formula), retention[0])
    mf = model_frame(
        formula, data, subset=subset, na_action=na_action or "omit", weights=weights, extra=extra
    )
    if mf.n == 0:
        raise ValueError("Data set has 0 observations")
    stop_values, start_values, event_values = _pyears_followup(mf, table is not None)
    terms = _pyears_terms(mf, data, calls)
    categories = np.column_stack([term.values for term in terms]) if terms else np.empty((mf.n, 0))
    positions = None if table is None else _rate_positions(mf, table)
    result = _core.pyears(
        stop_values,
        start_values,
        event_values,
        None if mf.weights is None else [float(value) for value in mf.weights],
        [term.factor for term in terms],
        [len(term.levels) for term in terms],
        [term.cuts for term in terms],
        categories,
        table,
        positions,
        expect_value,
        scale_value,
    )
    output = _pyears_result(result, terms, data_frame_value, mf.na_action)
    retained_y: Surv | list[list[float]] | None = None
    if retention[2]:
        if isinstance(mf.response, Surv):
            retained_y = mf.response
        elif isinstance(mf.y, np.ndarray):
            retained_y = mf.y.tolist()
        elif mf.y is not None:
            retained_y = [[value] for value in mf.y]
    return replace(
        output,
        formula=formula,
        term_labels=[term.label for term in terms],
        summary=None
        if table is None or positions is None
        else _population_match_summary(table, positions),
        model=None
        if row_key is None
        else _population_model_frame(
            mf, data, row_key, rmap, [] if table is None else table.dimid, calls
        ),
        x=((categories.tolist() if terms else [1.0] * mf.n) if retention[1] else None),
        y=retained_y,
    )


def _pyears_result_frame(result: PyearsResult) -> dict[str, list[Any]]:
    """``as_data_frame`` of a ``pyears`` result: its ``data.frame = TRUE`` layout."""

    if result.data is not None:
        return {name: list(values) for name, values in result.data.items()}
    tables = {
        name: _flatten(getattr(result, name), result.dim)
        for name in ("pyears", "n", "expected", "event")
        if getattr(result, name) is not None
    }
    return _cell_frame(tables, result.dimnames)


def _flatten(values: Any, dims: Sequence[int]) -> list[float]:
    """The column-major cells of a row-major nested list (inverse of :func:`_reshape`)."""

    if not dims:
        return [float(values)]
    if len(dims) == 1:
        return [float(value) for value in values]
    flat = [0.0] * math.prod(dims)
    strides = [1]
    for extent in dims[:-1]:
        strides.append(strides[-1] * extent)

    def walk(node: Any, depth: int, offset: int) -> None:
        if depth == len(dims):
            flat[offset] = float(node)
            return
        for k, child in enumerate(node):
            walk(child, depth + 1, offset + k * strides[depth])

    walk(values, 0, 0)
    return flat


@dataclass(frozen=True)
class PyearsSummary:
    """The tables R's ``summary.pyears`` prints.

    Each table is a row-major nested list over ``dim`` like :class:`PyearsResult`'s
    (a scalar without terms); with ``totals`` the first two dimensions end in a
    ``"Total"`` level, whose ``n`` is missing (``nan``) when a ``tcut`` term makes it
    meaningless.  ``ci_r`` and ``ci_rr`` hold one ``(lower, upper)`` pair per cell
    (R's ``cipoisson`` array).  A statistic that was not requested, or that the
    ``pyears`` object cannot give (no events, no expected counts), is ``None``; an
    empty cell has missing (``nan``) rates, ratios and limits.
    """

    n: Any
    pyears: Any
    offtable: float
    observations: int
    dim: list[int]
    dimnames: dict[str, list[str]]
    event: Any = None
    expected: Any = None
    rate: Any = None
    ci_r: Any = None
    rr: Any = None
    ci_rr: Any = None


def _pyears_tables(result: PyearsResult) -> dict[str, list[float]]:
    """The column-major cells of ``pyears``, ``n`` and, when present, ``event`` and
    ``expected``.

    A ``data.frame = TRUE`` result's rows are put back into the full table, with zero
    cells elsewhere, as ``summary.pyears`` does.
    """

    names = ("pyears", "n", "event", "expected")
    if result.data is None:
        arrays = {name: getattr(result, name) for name in names}
        return {
            name: _flatten(array, result.dim) for name, array in arrays.items() if array is not None
        }
    data = result.data
    cells = [0] * len(data["pyears"])
    stride = 1
    for (label, levels), extent in zip(result.dimnames.items(), result.dim, strict=True):
        position = {level: k for k, level in enumerate(levels)}
        cells = [
            cell + position[level] * stride for cell, level in zip(cells, data[label], strict=True)
        ]
        stride *= extent
    tables: dict[str, list[float]] = {}
    for name in names:
        if name in data:
            table = [0.0] * stride
            for cell, value in zip(cells, data[name], strict=True):
                table[cell] = float(value)
            tables[name] = table
    return tables


def summary_pyears(
    object: PyearsResult,
    totals: bool = False,
    rate: bool = False,
    ci_r: bool = False,
    rr: bool = True,
    ci_rr: bool = False,
    conf_level: Any = 0.95,
    scale: Any = 1,
    **kwargs: Any,
) -> PyearsSummary:
    """R's ``summary.pyears``: the tables of a ``pyears`` result, with rates and ratios.

    ``rate`` is ``scale * event / pyears`` and ``rr`` the observed/expected ratio (R's
    default ``rr = expected`` is on whenever the object has expected counts); ``ci_r``
    and ``ci_rr`` add their exact Poisson limits at ``conf_level``, and ``totals``
    appends the margins of the first two dimensions.  R's printing switches
    (``header``, ``vline``, ``nastring``, ...) have no counterpart: the tables are
    returned instead.
    """

    ci_r = _pop_dotted_keyword(kwargs, "ci.r", "ci_r", ci_r, False)
    ci_rr = _pop_dotted_keyword(kwargs, "ci.rr", "ci_rr", ci_rr, False)
    conf_level = _pop_dotted_keyword(kwargs, "conf.level", "conf_level", conf_level, 0.95)
    if kwargs:
        raise TypeError(
            f"summary_pyears got unexpected keyword argument(s): {', '.join(sorted(kwargs))}"
        )
    if not isinstance(object, PyearsResult):
        raise TypeError("input must be a pyears object")
    options = {
        name: _normalize_bool_option(value, name.replace("_", "."))
        for name, value in (
            ("totals", totals),
            ("rate", rate),
            ("ci_r", ci_r),
            ("rr", rr),
            ("ci_rr", ci_rr),
        )
    }
    if options["totals"] and not object.dim:
        raise ValueError("totals need a pyears table with at least one term")
    tables = _pyears_tables(object)
    summary = _core.summary_pyears(
        tables["pyears"],
        tables["n"],
        tables.get("event"),
        tables.get("expected"),
        list(object.dim),
        float(object.offtable),
        int(object.observations),
        bool(object.tcut),
        conf_level=_finite_float(conf_level, "conf.level"),
        scale=_finite_float(scale, "scale"),
        **options,
    )
    dims = list(summary.dims) if object.dim else []
    dimnames = {label: list(levels) for label, levels in object.dimnames.items()}
    if options["totals"]:
        for label in list(dimnames)[:2]:
            dimnames[label].append("Total")

    def limits(lower: Sequence[float] | None, upper: Sequence[float] | None) -> Any:
        if lower is None or upper is None:
            return None
        return _row_major(list(zip(lower, upper, strict=True)), dims)

    return PyearsSummary(
        n=_reshape(summary.n, dims),
        pyears=_reshape(summary.pyears, dims),
        offtable=float(summary.offtable),
        observations=int(summary.observations),
        dim=dims,
        dimnames=dimnames,
        event=_reshape(summary.event, dims),
        expected=_reshape(summary.expected, dims),
        rate=_reshape(summary.rate, dims),
        ci_r=limits(summary.ci_r_lower, summary.ci_r_upper),
        rr=_reshape(summary.rr, dims),
        ci_rr=limits(summary.ci_rr_lower, summary.ci_rr_upper),
    )


# ---------------------------------------------------------------------------
# survexp
# ---------------------------------------------------------------------------

_SURVEXP_METHODS = ("ederer", "hakulinen", "conditional", "individual.h", "individual.s")


def _survexp_times(times: Any | None) -> list[float] | None:
    if times is None:
        return None
    values = _float_vector(_scalar_or_vector(times, "times"), "times")
    if any(value < 0.0 for value in values):
        raise ValueError("Invalid time point requested")
    if any(b < a for a, b in zip(values[:-1], values[1:], strict=True)):
        raise ValueError("Times must be in increasing order")
    return values


def _survexp_response(mf: ModelFrame) -> list[float] | None:
    if mf.response is not None:
        if mf.response.type != "right":
            raise ValueError("Illegal response value")
        values = list(mf.response.time)
    elif mf.y is not None:
        if isinstance(mf.y, np.ndarray):
            raise ValueError("Illegal response value")
        values = list(mf.y)
    else:
        return None
    if any(value < 0.0 for value in values):
        raise ValueError("Negative follow up time")
    return values


def _is_tcut(mf: ModelFrame, term: _CovariateTerm) -> bool:
    """R's class check of a model-frame variable: a ``tcut()`` call, or a data column
    holding a ``tcut``."""

    name = _covariate_term_name(term)
    return name.startswith("tcut(") or (
        name == term.column and isinstance(_column_source(mf.data, name), TcutResult)
    )


def _survexp_groups(mf: ModelFrame, data: Any) -> tuple[list[int] | None, list[str] | None]:
    """R's ``strata(mf[ovars])``: zero-based curve of each row and the curve labels."""

    if any(isinstance(term, _InteractionTerm) for term in mf.terms.covariates):
        raise ValueError("Survexp cannot have interaction terms")
    if any(_is_tcut(mf, term) for term in mf.terms.covariates if isinstance(term, _CovariateTerm)):
        raise ValueError("Can't use tcut variables in expected survival")
    groups = _model_strata(mf, _population_variable_overrides(mf, data))
    if groups is None:
        return None, None
    if any(code is None for code in groups.codes):
        raise ValueError("missing values in the grouping variables")
    return [int(code) for code in groups.codes if code is not None], list(groups.levels)


def _survexp_method(method: str | None, cohort: Any, conditional: Any, has_response: bool) -> str:
    """R's ``method`` resolution: ``match.arg`` or the historical defaults."""

    if method is not None:
        return _match_string_arg(method, "method", _SURVEXP_METHODS, "invalid method")
    if not _normalize_bool_option(cohort, "cohort"):
        return "individual.s"
    if _normalize_bool_option(conditional, "conditional"):
        return "conditional"
    return "hakulinen" if has_response else "ederer"


def _survexp_vectors(
    time: Any, age: Any, year: Any, sex: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    """The reticulate bridge's vector call as data and ``rmap`` for ``time ~ 1``."""

    if time is None or age is None or year is None:
        raise ValueError("the direct survexp call requires time, age and year")
    columns: dict[str, Any] = {"time": _float_vector(time, "time"), "age": age, "year": year}
    rmap = {"age": "age", "year": "year"}
    if sex is not None:
        columns["sex"] = sex
        rmap["sex"] = "sex"
    return columns, rmap


def _survexp_result(result: Any, levels: list[str] | None, n: int) -> SurvExpResult:
    if levels is None:
        return SurvExpResult(
            time=list(result.time),
            surv=[row[0] for row in result.surv],
            n_risk=[row[0] for row in result.n_risk],
            method=str(result.method),
            n=n,
        )
    return SurvExpResult(
        time=list(result.time),
        surv=[list(row) for row in result.surv],
        n_risk=[list(row) for row in result.n_risk],
        method=str(result.method),
        n=n,
        strata=levels,
    )


def _survexp_retained(
    output: SurvExpResult,
    mf: ModelFrame,
    data: Any,
    rmap: Mapping[str, Any] | None,
    rate_names: Sequence[str],
    row_key: str | None,
    retention: tuple[bool, bool, bool],
    groups: list[int] | None,
    levels: list[str] | None,
    response: list[float] | None,
    default_followup: float | None = None,
) -> SurvExpResult:
    categories: StrataFactor | list[float] | None = None
    if retention[1]:
        if groups is None or levels is None:
            categories = [1.0] * mf.n
        else:
            counts = [0] * len(levels)
            for code in groups:
                counts[code] += 1
            categories = StrataFactor(
                codes=list(groups),
                levels=list(levels),
                labels=[levels[code] for code in groups],
                counts=counts,
            )
    followup = None
    if retention[2]:
        if response is not None:
            followup = list(response)
        elif default_followup is not None:
            followup = [default_followup] * mf.n
    return replace(
        output,
        formula=mf.formula,
        term_labels=_population_term_labels(mf),
        model=None
        if row_key is None
        else _population_model_frame(mf, data, row_key, rmap, rate_names),
        x=categories,
        y=followup,
    )


def summary_survexp(
    object: SurvExpResult, times: Any | None = None, scale: Any = 1, **_kwargs: Any
) -> SurvExpSummary:
    """R's ``summary.survexp``, using the native expected-curve time selector.

    Requested times are sorted, retaining duplicates and removing missing or
    out-of-range times. Survival is taken from the previous observation and
    risk counts from the next. ``scale`` divides output times; omitted times
    keep every source row. Curve labels name matrix columns, as on the fit.
    """

    if not isinstance(object, SurvExpResult):
        raise TypeError("Invalid data")
    matrix = bool(object.surv and isinstance(object.surv[0], list))
    surv = object.surv if matrix else [[value] for value in object.surv]
    n_risk = object.n_risk if matrix else [[value] for value in object.n_risk]
    requested = None if times is None else _floats_or_nan(_scalar_or_vector(times, "times"))
    result = _core.summary_survexp(
        object.time,
        surv,
        n_risk,
        requested,
        _numeric_scalar(_scalar_or_vector(scale, "scale"), "scale"),
        object.method,
    )
    ncols = len(surv[0]) if surv else len(object.strata or [""])
    return SurvExpSummary(
        time=list(result.time),
        surv=[row[0] for row in result.surv] if ncols == 1 else result.surv,
        n_risk=[row[0] for row in result.n_risk] if ncols == 1 else result.n_risk,
        method=result.method,
        strata=None if object.strata is None else list(object.strata),
    )


def _survexp_frame(result: SurvExpResult | SurvExpSummary) -> dict[str, list[Any]]:
    """Expected curves as one row per (curve, time)."""

    matrix = bool(result.surv and isinstance(result.surv[0], list))
    ncols = len(result.surv[0]) if matrix else len(result.strata or [""])
    if result.strata is not None and len(result.strata) != ncols:
        raise ValueError("curve labels must match the survival columns")
    frame: dict[str, list[Any]] = {"time": result.time * ncols, "surv": [], "n_risk": []}
    for col in range(ncols):
        for name in ("surv", "n_risk"):
            values = getattr(result, name)
            frame[name].extend((row[col] for row in values) if matrix else values)
    if result.strata is not None:
        frame["strata"] = [label for label in result.strata for _ in result.time]
    elif ncols > 1:
        frame["curve"] = [col + 1 for col in range(ncols) for _ in result.time]
    return frame


def survexp(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    rmap: Mapping[str, Any] | None = None,
    times: Any | None = None,
    method: str | None = None,
    cohort: bool = True,
    conditional: bool = False,
    ratetable: Any | None = None,
    scale: Any = 1,
    se_fit: bool | None = None,
    model: bool = False,
    x: bool = False,
    y: bool = False,
    *,
    time: Any = None,
    age: Any = None,
    year: Any = None,
    sex: Any = None,
    **kwargs: Any,
) -> SurvExpResult | list[float]:
    """R's ``survexp``: expected survival from a population rate table.

    ``formula`` is ``~ group``, ``time ~ group`` or ``Surv(time, status) ~ group``
    with ``rmap`` naming the rate-table variables.  The ``time``/``age``/``year``/
    ``sex`` keywords are the reticulate bridge's vector call (``survexp.us`` with
    ``rmap = list(age, sex, year)``).  The individual methods return one value per
    subject. ``model=True`` retains the evaluated model frame. Otherwise
    ``x=True`` retains the grouping factor (ones without groups), and ``y=True``
    retains follow-up times. Individual methods return a vector and ignore
    these retention flags, as in R.
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, None)
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, None)
    if kwargs:
        raise TypeError(f"survexp got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if time is not None or age is not None or year is not None:
        data, rmap = _survexp_vectors(time, age, year, sex)
        formula = "time ~ 1"
    if not isinstance(formula, str):
        raise ValueError("A formula argument is required")
    from ._coxph import CoxphModel, _check_interaction_margins, _survfit_newdata, predict_coxph
    from ._coxphms import CoxphmsModel

    method_value = _survexp_method(method, cohort, conditional, _response_spec(formula) is not None)
    retention = (
        (False, False, False)
        if method_value.startswith("individual")
        else _population_retention_options(model, x, y)
    )
    if isinstance(ratetable, CoxphmsModel):
        raise ValueError("Invalid rate table")
    if isinstance(ratetable, CoxphModel):
        names = _formula_columns("~" + ratetable.formula.split("~", 1)[1], data)
        extra = _mapped_columns(rmap, names, data)
        if method_value.startswith("individual"):
            # predict(type = "expected") also reads the Cox model's response: R's
            # survexp adds the data's remaining columns to the rate variables
            for name in _formula_columns(ratetable.formula, data):
                extra.setdefault(name, name)
        row_key = _population_retention_rows(extra, _data_row_count(data, formula), retention[0])
        mf = model_frame(
            formula,
            data,
            subset=subset,
            na_action=na_action or "omit",
            weights=weights,
            extra=extra,
        )
        if mf.n == 0:
            raise ValueError("Data set has 0 rows")
        response = _survexp_response(mf)
        mapped: dict[str, Any] = {
            name: _column(mf.data, name) for name in _data_column_names(mf.data) or []
        }
        mapped.update(mf.extra)
        if se_fit is not None and _normalize_bool_option(se_fit, "se.fit"):
            warnings.warn("se.fit value ignored", stacklevel=2)
        if method_value.startswith("individual"):
            if response is None:
                raise ValueError("for individual survival an observation time must be given")
            hazard = predict_coxph(ratetable, mapped, type="expected")
            values = (
                hazard if method_value == "individual.h" else [math.exp(-value) for value in hazard]
            )
            return _pad_rows(values, _excluded_rows(mf.na_action))
        # Every retained row needs valid Cox prediction terms. A transformation
        # that becomes missing (log(-1)) is an error, not a silently omitted row.
        _check_interaction_margins(ratetable)
        new, _, _ = _survfit_newdata(
            ratetable,
            mapped,
            individual=False,
            id=None,
            na_action="na.fail",
        )
        groups, levels = _survexp_groups(mf, data)
        engine = ratetable.penalized if ratetable.penalized is not None else ratetable.fit
        result = engine.expected_survival(
            new.x,
            groups or [0] * mf.n,
            mf.weights or [1.0] * mf.n,
            new_strata=new.strata,
            new_offset=new.offset,
            y=response,
            times=_survexp_times(times),
            method=method_value,
        )
        output = _survexp_result(result, levels, mf.n)
        divisor = _normalize_positive_scale(scale)
        output = replace(output, time=[value / divisor for value in output.time])
        return _survexp_retained(
            output, mf, data, rmap, names, row_key, retention, groups, levels, response
        )
    table = _ratetable_argument(ratetable)
    extra = _rmap_columns(rmap, table, data)
    row_key = _population_retention_rows(extra, _data_row_count(data, formula), retention[0])
    mf = model_frame(
        formula,
        data,
        subset=subset,
        na_action=na_action or "omit",
        weights=weights,
        extra=extra,
    )
    if mf.n == 0:
        raise ValueError("Data set has 0 rows")
    if se_fit is not None and _normalize_bool_option(se_fit, "se.fit"):
        warnings.warn("se.fit value ignored", stacklevel=2)
    if mf.weights is not None and any(float(value) != 1.0 for value in mf.weights):
        warnings.warn("weights ignored", stacklevel=2)
    requested = _survexp_times(times)
    response = _survexp_response(mf)
    if response is None and requested is None:
        raise ValueError("either a times argument or a response is needed")
    if response is None and method_value != "ederer":
        raise ValueError("a response is required in the formula unless method='ederer'")
    groups, levels = _survexp_groups(mf, data)
    result = _core.survexp(
        table,
        _rate_positions(mf, table),
        response,
        groups,
        requested,
        method_value,
        _normalize_bool_option(cohort, "cohort"),
        _normalize_bool_option(conditional, "conditional"),
        _normalize_positive_scale(scale),
    )
    if method_value.startswith("individual"):
        return _pad_rows([row[0] for row in result.surv], _excluded_rows(mf.na_action))
    return _survexp_retained(
        _survexp_result(result, levels, mf.n),
        mf,
        data,
        rmap,
        table.dimid,
        row_key,
        retention,
        groups,
        levels,
        response,
        max(requested) if requested else None,
    )
