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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as _Date
from datetime import datetime as _DateTime
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _as_character,
    _factor,
    _float_vector,
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
    _column,
    _column_source,
    _covariate_term_name,
    _data_column_names,
    _data_row_count,
    _formula_columns,
    _formula_response_parts,
    _model_strata,
    _parse_formula_literal,
    _term_values,
    model_frame,
)
from ._surv import Surv
from ._types import (
    ModelFrame,
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


def _ratetable_day(value: Any) -> float:
    """``ratetableDate`` of one value: dates become days since 1970-01-01, numbers pass."""

    if _is_missing_value(value):
        return math.nan
    if isinstance(value, _DateTime | _Date):
        return _core.ratetable_date(value.year, value.month, value.day)
    if isinstance(value, str):
        parsed = _Date.fromisoformat(value[:10])
        return _core.ratetable_date(parsed.year, parsed.month, parsed.day)
    return float(value)


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
    columns: dict[str, Any] = {}
    n = _data_row_count(data)
    for name, value in ({} if rmap is None else rmap).items():
        if str(name) not in names:
            raise ValueError(f"Variable not found in the ratetable:{name}")
        is_constant = (
            isinstance(value, str) and value not in (_data_column_names(data) or [])
        ) or (not isinstance(value, str) and not hasattr(value, "__iter__"))
        if is_constant:
            value = [value] * n
        columns[str(name)] = value
    for dimid in names:
        columns.setdefault(dimid, dimid)
    return columns


def _rate_positions(mf: ModelFrame, ratetable: RateTable) -> list[list[float]]:
    """R's ``match.ratetable(rdata, ratetable)$R`` for the model frame's rate variables."""

    names = list(ratetable.dimid)
    columns: list[Any] = []
    for name in names:
        values = mf.extra[name]
        if all(isinstance(value, str) or _is_missing_value(value) for value in values) and any(
            isinstance(value, str) for value in values
        ):
            columns.append([str(value) for value in values])
        else:
            columns.append([_ratetable_day(value) for value in values])
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


def _r_vector_literal(expression: str, data: Any) -> list[float]:
    """A formula-level R vector: ``c(...)``, ``c(...) * k``, ``as.Date(c(...))`` or a column."""

    text = expression.strip()
    if text.startswith("as.Date(") and text.endswith(")"):
        return [_ratetable_day(value) for value in _r_string_literal(text[8:-1])]
    if text.startswith("c(") and text.endswith(")"):
        return [float(_parse_formula_literal(part)) for part in _formula_response_parts(text[2:-1])]
    for operator in ("*", "/"):
        head, sep, tail = text.rpartition(operator)
        if sep and head.strip().endswith(")"):
            values = _r_vector_literal(head, data)
            factor = float(_parse_formula_literal(tail))
            return [value * factor if operator == "*" else value / factor for value in values]
    try:
        return [float(_parse_formula_literal(text))]
    except ValueError:
        return [float(value) for value in _column(data, text)]


def _r_string_literal(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("c(") and text.endswith(")"):
        text = text[2:-1]
    return [str(_parse_formula_literal(part)) for part in _formula_response_parts(text)]


def _cut_labels(breaks: Sequence[float]) -> list[str]:
    """R's ``cut()`` default labels ``"(a,b]"``."""

    def label(value: float) -> str:
        return _as_character(value) if float(value).is_integer() else f"{value:.3g}"

    return [f"({label(a)},{label(b)}]" for a, b in zip(breaks[:-1], breaks[1:], strict=True)]


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


def _tcut_term(mf: ModelFrame, text: str) -> _PyearsTerm:
    """A ``tcut(x, breaks[, labels = c(...)])`` formula term."""

    arguments = _formula_response_parts(text[5:-1])
    x = _float_vector(_column(mf.data, arguments[0]), arguments[0])
    breaks = _r_vector_literal(arguments[1], mf.data)
    labels = None
    for argument in arguments[2:]:
        name, _sep, value = argument.partition("=")
        if name.strip() == "labels":
            labels = _r_string_literal(value)
    cut = _core.tcut(x, breaks, labels, 1.0)
    return _PyearsTerm(text, 0, list(cut.values), list(cut.labels), list(cut.cutpoints))


def _cut_term(mf: ModelFrame, text: str) -> _PyearsTerm:
    """A ``cut(x, breaks)`` formula term: R's right-closed intervals ``(a, b]``."""

    arguments = _formula_response_parts(text[4:-1])
    x = _arithmetic_expression_values(mf.data, arguments[0], mf.n)
    breaks = _r_vector_literal(arguments[1], mf.data)
    codes: list[float] = []
    for value in x:
        position = next(
            (k for k in range(1, len(breaks)) if breaks[k - 1] < value <= breaks[k]), None
        )
        codes.append(math.nan if position is None else float(position))
    return _PyearsTerm(text, 1, codes, _cut_labels(breaks), [])


def _pyears_term(mf: ModelFrame, term: _CovariateTerm) -> _PyearsTerm:
    label = _covariate_term_name(term)
    if term.call is not None:
        return (
            _tcut_term(mf, term.call) if term.call.startswith("tcut(") else _cut_term(mf, term.call)
        )
    source = _column_source(mf.data, term.column) if term.arithmetic is None else None
    if isinstance(source, TcutResult):
        return _PyearsTerm(
            label, 0, list(source.values), list(source.labels), list(source.cutpoints)
        )
    codes_raw, levels = _factor(_term_values(mf.data, term, mf.n), label)
    return _PyearsTerm(
        label, 1, [math.nan if c is None else c + 1.0 for c in codes_raw], levels, []
    )


def _pyears_terms(mf: ModelFrame) -> list[_PyearsTerm]:
    terms: list[_PyearsTerm] = []
    for term in mf.terms.covariates:
        if isinstance(term, _InteractionTerm):
            raise ValueError("Pyears cannot have interaction terms")
        terms.append(_pyears_term(mf, term))
    for columns in mf.terms.strata:
        raise ValueError(f"unsupported pyears term strata({columns})")
    return terms


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


def _reshape(values: Sequence[float] | None, dims: Sequence[int]) -> Any:
    """R's array over ``dims`` (column-major cells) as a row-major nested list."""

    if values is None:
        return None
    if not dims:
        return float(values[0])
    if len(dims) == 1:
        return [float(value) for value in values]
    strides = [1]
    for extent in dims[:-1]:
        strides.append(strides[-1] * extent)

    def build(prefix: int, depth: int) -> Any:
        if depth == len(dims):
            return float(values[prefix])
        return [build(prefix + k * strides[depth], depth + 1) for k in range(dims[depth])]

    return build(0, 0)


def _pyears_frame(result: Any, terms: Sequence[_PyearsTerm]) -> dict[str, list[Any]]:
    """R's ``data.frame = TRUE`` layout: one row per cell with person-years."""

    cells = list(range(len(result.pyears)))
    if terms:
        cells = [cell for cell in cells if result.pyears[cell] > 0.0]
    frame: dict[str, list[Any]] = {}
    for depth, term in enumerate(terms):
        stride = math.prod(len(other.levels) for other in terms[:depth])
        frame[term.label] = [term.levels[(cell // stride) % len(term.levels)] for cell in cells]
    frame["pyears"] = [result.pyears[cell] for cell in cells]
    frame["n"] = [result.n[cell] for cell in cells]
    if result.expected is not None:
        frame["expected"] = [result.expected[cell] for cell in cells]
    if result.event is not None:
        frame["event"] = [result.event[cell] for cell in cells]
    return frame


def _pyears_result(result: Any, terms: Sequence[_PyearsTerm], data_frame: bool) -> PyearsResult:
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
    extra = None if table is None else _rmap_columns(rmap, table, data)
    mf = model_frame(
        formula, data, subset=subset, na_action=na_action or "omit", weights=weights, extra=extra
    )
    if mf.n == 0:
        raise ValueError("Data set has 0 observations")
    stop_values, start_values, event_values = _pyears_followup(mf)
    terms = _pyears_terms(mf)
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
    return _pyears_result(result, terms, data_frame_value)


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


def _survexp_groups(mf: ModelFrame) -> tuple[list[int] | None, list[str] | None]:
    """R's ``strata(mf[ovars])``: zero-based curve of each row and the curve labels."""

    if any(isinstance(term, _InteractionTerm) for term in mf.terms.covariates):
        raise ValueError("Survexp cannot have interaction terms")
    for term in mf.terms.covariates:
        name = _covariate_term_name(term)
        if name.startswith("tcut("):
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
        names = _formula_columns("~" + ratetable.formula.split("~", 1)[1], data)
        mf = model_frame(
            formula,
            data,
            subset=subset,
            na_action=na_action or "omit",
            weights=weights,
            extra=_mapped_columns(rmap, names, data),
        )
        if mf.n == 0:
            raise ValueError("Data set has 0 rows")
        response = _survexp_response(mf)
        method_value = _survexp_method(method, cohort, conditional, response is not None)
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
