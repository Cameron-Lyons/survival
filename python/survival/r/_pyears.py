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
from dataclasses import dataclass
from datetime import date as _Date
from datetime import datetime as _DateTime
from datetime import timedelta as _TimeDelta
from itertools import pairwise
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_character,
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
    _scalar_or_vector,
)
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
    SurvExpResult,
    TcutResult,
    _CovariateTerm,
    _InteractionTerm,
)

# ---------------------------------------------------------------------------
# Rate tables
# ---------------------------------------------------------------------------

_RATETABLE_ATTRIBUTES = ("dims", "dimid", "dimnames", "cutpoints", "types")


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

    A string naming a column of *data* stays that column.  Any other string is R code:
    an R constant (``60``, ``"white"``) or a bare word (``white``) is a constant, and an
    expression
    (``ageyr * 365.25``, or ``accept_dt - birth_dt`` with the dates as days since
    1970-01-01) is evaluated in *data*, as R evaluates ``rmap`` in the model frame.
    Any other scalar is a constant.
    """

    columns: dict[str, Any] = {}
    n = _data_row_count(data)
    available = set(_data_column_names(data) or ())
    for name, value in ({} if rmap is None else rmap).items():
        if str(name) not in names:
            raise ValueError(f"Variable not found in the ratetable:{name}")
        if isinstance(value, str) and value not in available:
            value = _rmap_value(str(name), value, data, n)
        elif not isinstance(value, str) and not hasattr(value, "__iter__"):
            value = [value] * n
        columns[str(name)] = value
    for dimid in names:
        columns.setdefault(dimid, dimid)
    return columns


def _rmap_value(name: str, text: str, data: Any, n: int) -> list[Any]:
    """The values of the ``rmap`` entry *name*, a string *text* naming no column."""

    literal = _r_literal(text)
    if literal is not None:
        return [literal] * n
    word, quoted = _formula_name(text)
    if not quoted and not _unsupported_formula_name(word, quoted):
        return [text] * n
    try:
        used = _expression_columns(text)
    except ValueError as exc:
        raise ValueError(f"rmap {name} = {text}: {exc}") from exc
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

    names = list(ratetable.dimid)
    types = ratetable.type_codes()
    values = [mf.extra[name] for name in names]
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
    for code, column in zip(types, values, strict=True):
        if all(isinstance(value, str) or _is_missing_value(value) for value in column) and any(
            isinstance(value, str) for value in column
        ):
            columns.append([str(value) for value in column])
        elif code > 2:
            columns.append([_ratetable_day(value) for value in column])
        else:
            columns.append([_day_count(value) for value in column])
    return _core.match_ratetable(ratetable, names, columns).r


def _ratetable_argument(ratetable: Any) -> RateTable:
    if ratetable is None:
        return _core.survexp_us()
    if isinstance(ratetable, RateTable):
        return ratetable
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


def _pyears_followup(mf: ModelFrame) -> tuple[list[float], list[float] | None, list[float] | None]:
    """R's ``Y`` checks: (stop, start, event) of the follow-up."""

    response = mf.response
    if response is None:
        if mf.y is None:
            raise ValueError("Follow-up time must appear in the formula")
        if any(value < 0.0 for value in mf.y):
            raise ValueError("Negative follow up time")
        return list(mf.y), None, None
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


def _pyears_frame(result: Any, terms: Sequence[_PyearsTerm]) -> dict[str, list[Any]]:
    """R's ``data.frame = TRUE`` layout: one row per cell with person-years."""

    # each getter copies the whole table out of the kernel result: read it once
    tables = {"pyears": result.pyears, "n": result.n}
    if result.expected is not None:
        tables["expected"] = result.expected
    if result.event is not None:
        tables["event"] = result.event
    pyears = tables["pyears"]
    cells = (
        [cell for cell, value in enumerate(pyears) if value > 0.0]
        if terms
        else list(range(len(pyears)))
    )
    frame: dict[str, list[Any]] = {}
    for depth, term in enumerate(terms):
        stride = math.prod(len(other.levels) for other in terms[:depth])
        frame[term.label] = [term.levels[(cell // stride) % len(term.levels)] for cell in cells]
    for name, values in tables.items():
        frame[name] = [values[cell] for cell in cells]
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
    """

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, None)
    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        raise TypeError(f"pyears got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    scale_value = _normalize_positive_scale(scale)
    data_frame_value = _normalize_bool_option(data_frame, "data.frame")
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
        )
    table = None if ratetable is None and rmap is None else _ratetable_argument(ratetable)
    if rmap is not None and ratetable is None:
        raise ValueError("No rate table specified")
    calls = _pyears_calls(formula, data)
    # the rate variables and the tcut()/cut() values go through subset and na.action
    # with the formula's variables, so a cut() value outside the breaks drops its row
    extra = {} if table is None else _rmap_columns(rmap, table, data)
    extra.update((call, term.values) for call, term in calls.items())
    mf = model_frame(
        formula, data, subset=subset, na_action=na_action or "omit", weights=weights, extra=extra
    )
    if mf.n == 0:
        raise ValueError("Data set has 0 observations")
    stop_values, start_values, event_values = _pyears_followup(mf)
    terms = _pyears_terms(mf, data, calls)
    result = _core.pyears(
        stop_values,
        start_values,
        event_values,
        None if mf.weights is None else [float(value) for value in mf.weights],
        [term.factor for term in terms],
        [len(term.levels) for term in terms],
        [term.cuts for term in terms],
        [[term.values[row] for term in terms] for row in range(mf.n)],
        table,
        None if table is None else _rate_positions(mf, table),
        expect_value,
        scale_value,
    )
    return _pyears_result(result, terms, data_frame_value, mf.na_action)


def _pyears_result_frame(result: PyearsResult) -> dict[str, list[Any]]:
    """``as_data_frame`` of a ``pyears`` result (its ``data.frame = TRUE`` layout)."""

    if result.data is not None:
        return {name: list(values) for name, values in result.data.items()}
    cells = list(range(max(1, math.prod(result.dim))))
    labels = result.group
    frame: dict[str, list[Any]] = {"group": [labels[cell] for cell in cells]}
    flat = {
        name: _flatten(getattr(result, name), result.dim)
        for name in ("pyears", "n", "expected", "event")
        if getattr(result, name) is not None
    }
    frame.update(flat)
    return frame


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


def _finegray_frame(result: Any) -> dict[str, list[Any]]:
    """``as_data_frame`` of a raw ``FineGrayOutput``."""

    return {
        "row": [int(value) for value in result.row],
        "start": [float(value) for value in result.start],
        "end": [float(value) for value in result.end],
        "wt": [float(value) for value in result.wt],
        "add": [int(value) for value in result.add],
    }


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


def _survexp_groups(mf: ModelFrame) -> tuple[list[int] | None, list[str] | None]:
    """R's ``strata(mf[ovars])``: zero-based curve of each row and the curve labels."""

    if any(isinstance(term, _InteractionTerm) for term in mf.terms.covariates):
        raise ValueError("Survexp cannot have interaction terms")
    if any(_is_tcut(mf, term) for term in mf.terms.covariates if isinstance(term, _CovariateTerm)):
        raise ValueError("Can't use tcut variables in expected survival")
    groups = _model_strata(mf)
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
    subject.  ``model``, ``x`` and ``y`` are accepted for R compatibility; the
    result carries no model frame.
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
    from ._coxph import CoxphModel, _survfit_curves, predict_coxph

    if isinstance(ratetable, CoxphModel):
        method_value = _survexp_method(
            method, cohort, conditional, _response_spec(formula) is not None
        )
        names = _formula_columns("~" + ratetable.formula.split("~", 1)[1], data)
        extra = _mapped_columns(rmap, names, data)
        if method_value.startswith("individual"):
            # predict(type = "expected") also reads the Cox model's response: R's
            # survexp adds the data's remaining columns to the rate variables
            for name in _formula_columns(ratetable.formula, data):
                extra.setdefault(name, name)
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
        mapped = {name: _column(mf.data, name) for name in _data_column_names(mf.data) or []}
        mapped.update(mf.extra)
        if se_fit is not None and _normalize_bool_option(se_fit, "se.fit"):
            warnings.warn("se.fit value ignored", stacklevel=2)
        if method_value.startswith("individual"):
            if response is None:
                raise ValueError("for individual survival an observation time must be given")
            hazard = predict_coxph(ratetable, mapped, type="expected")
            return (
                hazard if method_value == "individual.h" else [math.exp(-value) for value in hazard]
            )
        curves, _ = _survfit_curves(
            ratetable,
            mapped,
            individual=False,
            id=None,
            stype=2,
            ctype=2 if ratetable.method == "efron" else 1,
            se_fit=False,
            censor=False,
        )
        groups, levels = _survexp_groups(mf)
        result = _core.survexp_cox(
            curves,
            groups or [0] * mf.n,
            mf.weights or [1.0] * mf.n,
            y=response,
            times=_survexp_times(times),
            method=method_value,
        )
        output = _survexp_result(result, levels, mf.n)
        divisor = _normalize_positive_scale(scale)
        return SurvExpResult(
            time=[value / divisor for value in output.time],
            surv=output.surv,
            n_risk=output.n_risk,
            method=output.method,
            n=output.n,
            strata=output.strata,
        )
    table = _ratetable_argument(ratetable)
    mf = model_frame(
        formula,
        data,
        subset=subset,
        na_action=na_action or "omit",
        weights=weights,
        extra=_rmap_columns(rmap, table, data),
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
    method_value = _survexp_method(method, cohort, conditional, response is not None)
    if response is None and method_value != "ederer":
        raise ValueError("a response is required in the formula unless method='ederer'")
    groups, levels = _survexp_groups(mf)
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
        return [row[0] for row in result.surv]
    return _survexp_result(result, levels, mf.n)


def survexp_individual(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
) -> list[float]:
    """Per-subject expected survival (``survexp(..., cohort = FALSE)``) from vectors."""

    values = survexp(time=time, age=age, year=year, sex=sex, ratetable=ratetable, cohort=False)
    return [float(value) for value in values]
