"""Formula tokenizer/parser, terms, design matrices, and model-frame builders."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from itertools import combinations, compress, product
from operator import add, ge, mul, sub, truediv
from typing import Any

import numpy as np

from .. import _survival as _core
from ._coerce import (
    _DEFAULT_NA_ACTION,
    _as_character,
    _coerce_array_like,
    _finite_float,
    _floats_or_nan,
    _hashable_group_value,
    _is_missing_value,
    _keep_rows_after_na_action,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _mstate_categories,
    _normalize_na_action,
    _numeric_ndarray,
    _r_factor,
    _RFactorVector,
    _rows_of,
    _strata_value_label,
    _subset_indices,
    _subset_optional_sequence,
    _subset_sequence,
    _warn_outside_package,
)
from ._penalties import PENALTY_FUNCTIONS, fit_penalty, penalty_columns
from ._surv import (
    Surv,
    Surv2,
    _formula_response_argument_name,
    _normalize_surv_type,
    _ordered_named_response_arguments,
    _repeated_option,
    _strata,
    _time_column,
)
from ._types import (
    _MISSING,
    ModelFrame,
    NaAction,
    StrataFactor,
    _CachedFormulaTerms,
    _CategoricalDesignTerm,
    _CovariateSpec,
    _CovariateTerm,
    _DesignTerm,
    _FormulaDesign,
    _FormulaModelTerm,
    _FormulaTerms,
    _InteractionDesignTerm,
    _InteractionTerm,
    _ModelClusterTerm,
    _ModelCovariateTerm,
    _ModelOffsetTerm,
    _ModelStrataTerm,
    _NumericDesignTerm,
    _PenaltyDesignTerm,
    _ResponseOperand,
    _SingleDesignTerm,
    _SurvResponseSpec,
)


def _column_source(data: Any, name: str) -> Any:
    if data is None:
        raise ValueError("data is required when using a formula")
    if isinstance(data, Mapping):
        try:
            return data[name]
        except KeyError as exc:
            raise KeyError(f"column {name!r} not found in data") from exc
    try:
        return data[name]
    except Exception as exc:
        raise KeyError(f"column {name!r} not found in data") from exc


def _column(data: Any, name: str) -> list[Any]:
    return _materialize_1d(_column_source(data, name), name)


def _comparison_column(data: Any, name: str) -> list[Any]:
    """A data column as R's relational operators read it: a factor is its labels as
    strings (``Ops.factor``), any other column its own values."""

    source = _column_source(data, name)
    values = _materialize_1d(source, name)
    if _mstate_categories(source) is None:
        return values
    return [None if _is_missing_value(value) else _as_character(value) for value in values]


def _as_numeric_column(data: Any, name: str) -> list[Any]:
    """R's ``as.numeric`` of a data column: a factor's 1-based level codes (NaN where
    missing), any other column's own values."""

    source = _column_source(data, name)
    values = _materialize_1d(source, name)
    categories = _mstate_categories(source)
    if categories is None:
        return values
    codes = {value: i + 1 for i, value in enumerate(categories)}
    return [math.nan if _is_missing_value(value) else codes[value] for value in values]


def _formula_name(name: str) -> tuple[str, bool]:
    name = name.strip()
    if name.startswith("`") and name.endswith("`") and len(name) >= 2:
        inner = name[1:-1]
        if not inner:
            raise ValueError("backtick formula names must not be empty")
        return inner, True
    return name, False


# ---------------------------------------------------------------------------
# The formula tokenizer: one scanner that knows about parentheses, backtick
# names and string literals, and the splitters built on it.
# ---------------------------------------------------------------------------


def _scan(text: str, *, quotes: bool = True) -> list[tuple[int, str, int]]:
    """Every character of *text* outside backtick names (and string literals) with its depth.

    Parentheses are reported with the depth outside them, so an item with depth 0
    is a top-level character; the scanner raises R's unterminated-backtick and
    unterminated-quote errors.
    """

    items: list[tuple[int, str, int]] = []
    depth = 0
    in_backtick = False
    quote: str | None = None
    for idx, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char == "`":
            in_backtick = not in_backtick
            continue
        if in_backtick:
            continue
        if quotes and char in {"'", '"'}:
            quote = char
            continue
        if char == "(":
            items.append((idx, char, depth))
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            items.append((idx, char, depth))
            continue
        items.append((idx, char, depth))
    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    if quote is not None:
        raise ValueError("unterminated quote in formula")
    return items


def _top_level(text: str, *, quotes: bool = True) -> list[tuple[int, str]]:
    """The top-level characters of *text* (depth 0, outside names and literals)."""

    return [(idx, char) for idx, char, depth in _scan(text, quotes=quotes) if depth == 0]


def _split_at(text: str, positions: Sequence[tuple[int, int]], *, keep_empty: bool) -> list[str]:
    """Split *text* around the ``(start, end)`` character spans in *positions*."""

    parts: list[str] = []
    cursor = 0
    for start, end in positions:
        parts.append(text[cursor:start].strip())
        cursor = end
    parts.append(text[cursor:].strip())
    return parts if keep_empty else [part for part in parts if part]


def _split_top_level(segment: str, separator: str) -> list[str]:
    """Split at every top-level *separator* character (empty pieces kept, as R's terms)."""

    positions = [
        (idx, idx + 1) for idx, char in _top_level(segment, quotes=False) if char == separator
    ]
    return _split_at(segment, positions, keep_empty=True)


def _split_top_level_token(segment: str, token: str) -> list[str]:
    """Split at every top-level occurrence of the multi-character *token* (``%in%``)."""

    positions: list[tuple[int, int]] = []
    for idx, char in _top_level(segment, quotes=False):
        if char == token[0] and segment.startswith(token, idx):
            if positions and idx < positions[-1][1]:
                continue
            positions.append((idx, idx + len(token)))
    return _split_at(segment, positions, keep_empty=True)


def _formula_name_items(segment: str) -> list[tuple[str, bool]]:
    """The comma-separated names of a ``strata(a, b)``-style argument list."""

    positions = [(idx, idx + 1) for idx, char in _top_level(segment, quotes=False) if char == ","]
    return [_formula_name(name) for name in _split_at(segment, positions, keep_empty=False)]


def _formula_response_parts(segment: str) -> list[str]:
    """The top-level comma-separated arguments of a ``Surv(...)`` call."""

    positions = [(idx, idx + 1) for idx, char in _top_level(segment) if char == ","]
    return _split_at(segment, positions, keep_empty=False)


def _formula_named_option(part: str) -> tuple[str, str] | None:
    """``name = value`` at the top level of *part*, ignoring ``==``, ``!=``, ``<=`` and ``>=``."""

    for idx, char in _top_level(part):
        if char != "=":
            continue
        previous = part[idx - 1] if idx > 0 else ""
        following = part[idx + 1] if idx + 1 < len(part) else ""
        if previous in {"=", "!", "<", ">"} or following == "=":
            continue
        return part[:idx].strip(), part[idx + 1 :].strip()
    return None


def _top_level_comparison(part: str) -> tuple[str, str, str] | None:
    """The first top-level comparison ``left <op> right`` of a response argument."""

    for idx, _char in _top_level(part):
        for operator in ("==", "!=", "<=", ">=", "<", ">"):
            if part.startswith(operator, idx):
                left = part[:idx].strip()
                right = part[idx + len(operator) :].strip()
                if not left or not right:
                    raise ValueError("formula response comparisons require both operands")
                return left, operator, right
    return None


def _formula_tokens(rhs: str) -> list[tuple[str, str]]:
    """The ``+``/``-`` separated terms of a right-hand side with their sign."""

    positions = [(idx, idx + 1) for idx, char in _top_level(rhs, quotes=False) if char in "+-"]
    parts = _split_at(rhs, positions, keep_empty=True)
    operators = ["+", *(rhs[idx] for idx, _end in positions)]
    return [(op, term) for op, term in zip(operators, parts, strict=True) if term]


def _find_top_level_arithmetic_operator(
    expression: str,
    operators: set[str],
    *,
    quotes: bool = False,
) -> tuple[str, str, str] | None:
    """The right-most top-level binary operator of *operators* (unary signs and the
    sign of a number's exponent skipped)."""

    for idx, char in reversed(_top_level(expression, quotes=quotes)):
        if char not in operators:
            continue
        previous = _previous_non_space(expression, idx)
        if previous is None or previous in "+-*/(^:<>=!&|" or _is_exponent_sign(expression, idx):
            continue
        left = expression[:idx].strip()
        right = expression[idx + 1 :].strip()
        if not left or not right:
            raise ValueError("formula arithmetic terms require both operands")
        return left, char, right
    return None


def _find_top_level_power_operator(
    expression: str, *, quotes: bool = False
) -> tuple[str, str, str] | None:
    """The left-most top-level ``^``."""

    for idx, char in _top_level(expression, quotes=quotes):
        if char != "^":
            continue
        left = expression[:idx].strip()
        right = expression[idx + 1 :].strip()
        if not left or not right:
            raise ValueError("formula arithmetic terms require both operands")
        return left, char, right
    return None


def _is_exponent_sign(text: str, idx: int) -> bool:
    """Whether the sign at *idx* belongs to a number's exponent (``1e-3``)."""

    if idx < 2 or text[idx - 1] not in "eE":
        return False
    start = idx - 1
    while start > 0 and (text[start - 1].isdigit() or text[start - 1] == "."):
        start -= 1
    mantissa = text[start : idx - 1]
    if not any(char.isdigit() for char in mantissa):
        return False
    # a mantissa that ends a name (x1e-3) is the name minus a number
    return start == 0 or not (text[start - 1].isalnum() or text[start - 1] in "._")


def _previous_non_space(text: str, idx: int) -> str | None:
    cursor = idx - 1
    while cursor >= 0:
        if not text[cursor].isspace():
            return text[cursor]
        cursor -= 1
    return None


def _strip_outer_formula_parentheses(term: str) -> str:
    """Remove every pair of parentheses that encloses the whole term."""

    value = term.strip()
    while value.startswith("(") and value.endswith(")"):
        items = _scan(value, quotes=False)
        if any(depth == 0 for idx, _char, depth in items if 0 < idx < len(value) - 1):
            break
        value = value[1:-1].strip()
    return value


def _parse_formula_literal(value: str) -> Any:
    """A constant of a ``Surv()`` response comparison, ``rep()`` or a penalty option,
    typed for Python: a whole number written without a point or exponent is an ``int``
    (``rep()``'s count must be one) and ``T``/``F`` are logical.  Unlike
    :func:`_r_literal`, which formula arithmetic and vectors read constants with, a
    value that is not a finite constant is an error."""

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "t"}:
        return True
    if lowered in {"false", "f"}:
        return False
    if lowered.endswith("l") and len(value) > 1:
        value = value[:-1]
        lowered = value.lower()

    try:
        if "_" in value:  # a Python digit separator, not R
            raise ValueError(value)
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(
            "formula response comparisons require a numeric, boolean, or quoted string literal"
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError("formula response comparison literals must be finite")
    if numeric.is_integer() and not any(char in value.lower() for char in {".", "e"}):
        return int(numeric)
    return numeric


def _unwrap_response_identity(expression: str) -> str:
    expression = expression.strip()
    for wrapper in ("I", "identity", "factor", "as.factor"):
        prefix = f"{wrapper}("
        if expression.startswith(prefix) and expression.endswith(")"):
            return expression[len(prefix) : -1].strip()
    return expression


def _response_rep_call(expression: str) -> tuple[Any, str] | None:
    expression = _unwrap_response_identity(expression)
    if not expression.startswith("rep(") or not expression.endswith(")"):
        return None

    arguments = _formula_response_parts(expression[4:-1])
    repeated_value: Any = _MISSING
    count_expression: str | None = None
    positional: list[str] = []
    for argument in arguments:
        option_spec = _formula_named_option(argument)
        if option_spec is None:
            positional.append(argument)
            continue
        option, value = option_spec
        option = option.strip().lower()
        if option in {"x", "value"}:
            if repeated_value is not _MISSING:
                raise ValueError("rep(...) formula response contains multiple value arguments")
            repeated_value = _parse_formula_literal(value)
        elif option in {"times", "length.out"}:
            if count_expression is not None:
                raise ValueError("rep(...) formula response contains multiple length arguments")
            count_expression = value
        else:
            raise ValueError("rep(...) formula response supports only value and times arguments")

    if positional:
        if repeated_value is _MISSING:
            repeated_value = _parse_formula_literal(positional.pop(0))
        if positional and count_expression is None:
            count_expression = positional.pop(0)
        if positional:
            raise ValueError("rep(...) formula response supports only value and length arguments")
    if repeated_value is _MISSING or count_expression is None:
        raise ValueError("rep(...) formula response requires value and length arguments")
    return repeated_value, count_expression


def _response_operand(expression: str, *, allow_literal: bool) -> _ResponseOperand:
    expression = _unwrap_response_identity(expression)
    if not expression:
        raise ValueError("formula response comparison operands must not be empty")
    if allow_literal:
        try:
            return _ResponseOperand(value=_parse_formula_literal(expression))
        except ValueError:
            pass

    column, quoted = _formula_name(expression)
    if not column:
        raise ValueError("Surv(...) formula response arguments must not be empty")
    if not quoted and any(token in column for token in "():*/"):
        raise ValueError(f"unsupported formula response expression: {expression}")
    return _ResponseOperand(column=column)


def _response_arg_columns(part: str) -> list[str]:
    part = _unwrap_response_identity(part)
    if _response_rep_call(part) is not None:
        return []
    if _is_formula_arithmetic_expression(part):
        return _arithmetic_expression_columns(part)
    comparison = _top_level_comparison(part)
    if comparison is None:
        operand = _response_operand(part, allow_literal=False)
        return [operand.column] if operand.column is not None else []

    columns: list[str] = []
    for expression in (comparison[0], comparison[2]):
        operand = _response_operand(expression, allow_literal=True)
        if operand.column is not None:
            _append_unique(columns, [operand.column])
    if not columns:
        raise ValueError("formula response comparisons require at least one data column")
    return columns


def _response_operand_values(
    data: Any,
    operand: _ResponseOperand,
) -> tuple[list[Any] | None, Any]:
    if operand.column is None:
        return None, operand.value
    return _comparison_column(data, operand.column), None


def _compare_response_values(left: Any, operator: str, right: Any) -> bool | None:
    """R's relational *operator* on one pair of values (``NA`` is ``None``).  As in R's
    ``relop``, a string operand makes both strings (``as.character``: ``2`` is ``"2"``,
    ``TRUE`` is ``"TRUE"``); other operands compare as numbers."""

    if _is_missing_value(left) or _is_missing_value(right):
        return None
    if isinstance(left, str) or isinstance(right, str):
        # R orders strings by the locale's collation (and a factor not at all)
        if operator not in {"==", "!="}:
            raise ValueError(
                f"formula comparison {operator!r} of character or factor values is not supported"
            )
        equal = _as_character(left) == _as_character(right)
        return equal if operator == "==" else not equal
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right

    try:
        left_numeric = float(left)
        right_numeric = float(right)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordered formula comparisons require numeric values") from exc

    if operator == "<=":
        return left_numeric <= right_numeric
    if operator == ">=":
        return left_numeric >= right_numeric
    if operator == "<":
        return left_numeric < right_numeric
    if operator == ">":
        return left_numeric > right_numeric
    raise ValueError(f"unsupported formula response comparison {operator!r}")


def _response_rep_count(count_expression: str, inferred_length: int | None) -> int:
    count_expression = count_expression.strip()
    try:
        count = _parse_formula_literal(count_expression)
    except ValueError:
        if inferred_length is None:
            raise ValueError(
                f"unsupported formula response rep(...) length expression: {count_expression}"
            ) from None
        return inferred_length
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("rep(...) formula response length must be a non-negative integer")
    return count


def _response_arg_values(data: Any, part: str, inferred_length: int | None = None) -> Any:
    part = _unwrap_response_identity(part)
    rep_call = _response_rep_call(part)
    if rep_call is not None:
        repeated_value, count_expression = rep_call
        return [repeated_value] * _response_rep_count(count_expression, inferred_length)

    if _is_formula_arithmetic_expression(part):
        columns = _arithmetic_expression_columns(part)
        n = len(_column(data, columns[0])) if columns else inferred_length
        if n is None:
            raise ValueError("formula response arithmetic requires a data column")
        return _arithmetic_expression_values(data, part, n)
    comparison = _top_level_comparison(part)
    if comparison is None:
        operand = _response_operand(part, allow_literal=False)
        if operand.column is None:
            raise ValueError("Surv(...) formula response arguments must be data columns")
        return _column_source(data, operand.column)

    left_operand = _response_operand(comparison[0], allow_literal=True)
    right_operand = _response_operand(comparison[2], allow_literal=True)
    left_values, left_literal = _response_operand_values(data, left_operand)
    right_values, right_literal = _response_operand_values(data, right_operand)
    operator = comparison[1]

    if left_values is None and right_values is None:
        raise ValueError("formula response comparisons require at least one data column")
    if left_values is not None and right_values is not None:
        if len(left_values) != len(right_values):
            raise ValueError("formula response comparison columns must have the same length")
        return [
            _compare_response_values(left, operator, right)
            for left, right in zip(left_values, right_values, strict=True)
        ]
    if left_values is not None:
        return [_compare_response_values(left, operator, right_literal) for left in left_values]
    if right_values is not None:
        return [_compare_response_values(left_literal, operator, right) for right in right_values]
    raise ValueError("formula response comparisons require at least one data column")


def _formula_response_values(data: Any, spec: _SurvResponseSpec) -> list[Any]:
    inferred_length: int | None = None
    if spec.columns:
        inferred_length = len(_column(data, spec.columns[0]))
    return [_response_arg_values(data, argument, inferred_length) for argument in spec.arguments]


def _parse_formula_type_option(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if value.strip().lower() == "mstate":
        return "mstate"
    return _normalize_surv_type(value)


def _parse_formula_origin_option(value: str) -> float:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return _finite_float(value, "origin")


_SURV2_CALLS = ("Surv2(", "survival::Surv2(")
_SURV_CALLS = ("Surv(", "survival::Surv(", *_SURV2_CALLS)


@lru_cache(maxsize=512)
def _formula_response_spec(formula: str) -> _SurvResponseSpec:
    lhs, sep, _rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")

    lhs = lhs.strip()
    if not (lhs.startswith(_SURV_CALLS) and lhs.endswith(")")):
        raise ValueError("formula response must be Surv(...)")
    response_inner = lhs.partition("(")[2][:-1]
    if lhs.startswith(_SURV2_CALLS):
        return _timeline_response_spec(response_inner)

    columns: list[str] = []
    surv_type: str | None = None
    origin = 0.0
    has_origin = False
    arguments: list[str] = []
    named_arguments: dict[str, str] = {}
    for part in _formula_response_parts(response_inner):
        option_spec = _formula_named_option(part)
        if option_spec is not None:
            option, value = option_spec
            if option == "type":
                if surv_type is not None:
                    raise ValueError("formula Surv(...) contains multiple type= arguments")
                surv_type = _parse_formula_type_option(value)
            elif option == "origin":
                if has_origin:
                    raise ValueError("formula Surv(...) contains multiple origin= arguments")
                origin = _parse_formula_origin_option(value)
                has_origin = True
            elif (argument_name := _formula_response_argument_name(option)) is not None:
                if argument_name in named_arguments:
                    raise ValueError(
                        f"formula Surv(...) contains multiple {argument_name}= arguments"
                    )
                named_arguments[argument_name] = value
                _append_unique(columns, _response_arg_columns(value))
            else:
                raise ValueError(
                    "formula Surv(...) supports only named time=, time2=, event=, type=, "
                    "and origin= arguments"
                )
            continue
        if named_arguments:
            raise ValueError(
                "Surv(...) formula response must not mix positional and named time/time2/event "
                "arguments"
            )
        arguments.append(part)
        _append_unique(columns, _response_arg_columns(part))

    if named_arguments:
        if arguments:
            raise ValueError(
                "Surv(...) formula response must not mix positional and named time/time2/event "
                "arguments"
            )
        arguments = _ordered_named_response_arguments(named_arguments)
    if len(arguments) not in {1, 2, 3}:
        raise ValueError("Surv(...) formula response must have 1, 2, or 3 column arguments")
    return _SurvResponseSpec(
        arguments=tuple(arguments),
        columns=tuple(columns),
        type=surv_type,
        origin=origin,
    )


def _timeline_response_spec(inner: str) -> _SurvResponseSpec:
    """A ``Surv2(time, event, repeated = FALSE)`` response (R/Surv2.R), its arguments
    matched as R matches them: by name, then the rest by position."""

    formals = ("time", "event", "repeated")
    named: dict[str, str] = {}
    positional: list[str] = []
    for part in _formula_response_parts(inner):
        option = _formula_named_option(part)
        if option is None:
            positional.append(part)
            continue
        name, value = option
        if name not in formals:
            raise ValueError(f"unused argument ({name} = {value}) in Surv2(...)")
        if name in named:
            raise ValueError(f"formula Surv2(...) contains multiple {name}= arguments")
        named[name] = value
    unmatched = [name for name in formals if name not in named]
    if len(positional) > len(unmatched):
        raise ValueError("unused argument in Surv2(...)")
    named.update(zip(unmatched, positional, strict=False))
    if "time" not in named:
        raise ValueError("must have a time argument")
    if "event" not in named:
        raise ValueError("must have an event argument")
    repeated: Any = False
    if "repeated" in named:
        repeated = _parse_formula_literal(named["repeated"])
        if not (isinstance(repeated, bool) or repeated == "first"):
            raise ValueError("invalid value for repeated option")
    arguments = (named["time"], named["event"])
    columns: list[str] = []
    for argument in arguments:
        _append_unique(columns, _response_arg_columns(argument))
    return _SurvResponseSpec(
        arguments=arguments,
        columns=tuple(columns),
        type=None,
        timeline=True,
        repeated=repeated,
    )


@lru_cache(maxsize=512)
def _response_spec(formula: str) -> _SurvResponseSpec | None:
    """The left-hand side of any survival formula.

    ``Surv(...)`` and ``Surv2(...)`` responses go through :func:`_formula_response_spec`; an empty
    left-hand side (``~ sex``) gives ``None`` and a plain expression (``time ~ 1``,
    ``stop / 365.25 ~ surgery``) a numeric response spec with ``surv=False``.
    """

    lhs, sep, _rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")
    lhs = lhs.strip()
    if not lhs:
        return None
    if lhs.startswith(_SURV_CALLS) and lhs.endswith(")"):
        return _formula_response_spec(formula)
    return _SurvResponseSpec(
        arguments=(lhs,),
        columns=tuple(_response_arg_columns(lhs)),
        type=None,
        origin=0.0,
        surv=False,
    )


class _FormulaRows(dict[str, Any]):
    """``data[rows, ]`` restricted to a formula's variables (see :func:`_formula_data_rows`).

    Like R's data frame it keeps its row count without any column, as for ``~ 1``.
    """

    __slots__ = ("nrow",)

    def __init__(self, columns: dict[str, Any], nrow: int) -> None:
        super().__init__(columns)
        self.nrow = nrow


def _data_row_count(data: Any, formula: str | None = None) -> int:
    """The number of rows of *data*: the first response column, else the first column."""

    if isinstance(data, _FormulaRows):
        return data.nrow
    spec = None if formula is None else _response_spec(formula)
    if spec is not None and spec.columns:
        return len(_column(data, spec.columns[0]))
    names = _data_column_names(data)
    if names:
        return len(_column(data, str(names[0])))
    try:
        return len(data)
    except TypeError as exc:
        raise ValueError("data must have at least one column") from exc


def _covariate_factors(term: _CovariateSpec) -> tuple[_CovariateTerm, ...]:
    if isinstance(term, _InteractionTerm):
        return term.factors
    return (term,)


def _covariate_columns(terms: Iterable[_CovariateSpec]) -> list[str]:
    columns: list[str] = []
    for term in terms:
        for factor in _covariate_factors(term):
            _append_unique(columns, _covariate_term_columns(factor))
    return columns


def _offset_columns(terms: Sequence[_CovariateTerm]) -> list[str]:
    columns: list[str] = []
    for term in terms:
        _append_unique(columns, _covariate_term_columns(term))
    return columns


def _arithmetic_literal(value: str) -> float | None:
    """The number R's constant *value* is in arithmetic (``TRUE`` counts one), or
    ``None`` when *value* is not a constant."""

    literal = _r_literal(value)
    if isinstance(literal, str):
        raise ValueError(f"non-numeric argument to binary operator: {value.strip()}")
    return None if literal is None else float(literal)


def _is_formula_arithmetic_expression(expression: str) -> bool:
    stripped_expression = _strip_outer_formula_parentheses(expression)
    return (
        stripped_expression.startswith(("+", "-"))
        or _find_top_level_arithmetic_operator(expression, {"+", "-"}) is not None
        or _find_top_level_arithmetic_operator(expression, {"*", "/"}) is not None
        or _find_top_level_power_operator(expression) is not None
    )


def _arithmetic_expression_columns(expression: str) -> list[str]:
    expression = _strip_outer_formula_parentheses(expression)
    split = _find_top_level_arithmetic_operator(expression, {"+", "-"})
    if split is None:
        split = _find_top_level_arithmetic_operator(expression, {"*", "/"})
    if split is not None:
        left, _operator, right = split
        columns = _arithmetic_expression_columns(left)
        _append_unique(columns, _arithmetic_expression_columns(right))
        return columns

    if expression.startswith(("+", "-")):
        return _arithmetic_expression_columns(expression[1:].strip())

    split = _find_top_level_power_operator(expression)
    if split is not None:
        left, _operator, right = split
        columns = _arithmetic_expression_columns(left)
        _append_unique(columns, _arithmetic_expression_columns(right))
        return columns

    if _arithmetic_literal(expression) is not None:
        return []
    if _top_level_comparison(expression) is not None:
        return _expression_columns(expression)

    call = _numeric_call(expression)
    if call is not None:
        return _expression_columns(call[1])

    column, quoted = _formula_name(expression)
    if _unsupported_formula_name(column, quoted):
        raise ValueError(f"unsupported formula arithmetic term: {expression}")
    return [column]


def _expression_columns(expression: str) -> list[str]:
    """The data columns an arithmetic expression or a comparison reads."""

    comparison = _top_level_comparison(_strip_outer_formula_parentheses(expression))
    if comparison is None:
        return _arithmetic_expression_columns(expression)
    columns: list[str] = []
    for operand in (comparison[0], comparison[2]):
        if _r_literal(operand) is None:
            _append_unique(columns, _arithmetic_expression_columns(operand))
    return columns


# The calls formula arithmetic evaluates: R's functions of one numeric argument and
# the wrappers that return theirs unchanged.
_NUMERIC_CALLS = ("log", "sqrt", "exp", "I", "identity", "as.numeric")


def _numeric_call(expression: str) -> tuple[str, str] | None:
    """``(function, argument)`` of a call such as ``log(x + 1)`` in formula arithmetic."""

    for function in _NUMERIC_CALLS:
        if expression.startswith(f"{function}(") and expression.endswith(")"):
            arguments = _formula_response_parts(expression[len(function) + 1 : -1])
            if len(arguments) != 1:
                raise ValueError(f"unsupported formula arithmetic term: {expression}")
            return function, arguments[0]
    return None


def _r_literal(text: str) -> Any:
    """R's constant *text* (a quoted string, ``TRUE``/``FALSE``, ``Inf`` or a number, as
    a double), or ``None`` when *text* is not one."""

    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    if text in {"TRUE", "FALSE"}:
        return text == "TRUE"
    if text == "Inf":
        return math.inf
    number = text.removesuffix("L")
    # float() also reads Python's digit separators (1_000), which R does not
    if not number or not (number[0].isdigit() or number[0] == ".") or "_" in number:
        return None
    try:
        return float(number)
    except ValueError:
        return None


def _r_divide(numerator: float, denominator: float) -> float:
    """R's ``numerator / denominator`` (IEEE): ``±Inf`` over zero, NaN for ``0/0``."""

    try:
        return numerator / denominator
    except ZeroDivisionError:
        if numerator == 0.0 or math.isnan(numerator):
            return math.nan
        return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)


def _r_pow(base: float, exponent: float) -> float:
    """R's ``base ^ exponent`` (``R_pow``): NaN for a negative base with a fractional
    exponent, ``Inf`` for zero to a negative power, ``±Inf`` on overflow."""

    try:
        return math.pow(base, exponent)
    except OverflowError:
        return -math.inf if base < 0.0 and exponent % 2.0 == 1.0 else math.inf
    except ValueError:
        return math.inf if base == 0.0 else math.nan


def _arithmetic_expression_values(data: Any, expression: str, n: int) -> list[float]:
    expression = _strip_outer_formula_parentheses(expression)
    additive = _find_top_level_arithmetic_operator(expression, {"+", "-"})
    if additive is not None:
        left, operator, right = additive
        left_values = _arithmetic_expression_values(data, left, n)
        right_values = _arithmetic_expression_values(data, right, n)
        if operator == "+":
            return [left + right for left, right in zip(left_values, right_values, strict=True)]
        return [left - right for left, right in zip(left_values, right_values, strict=True)]

    multiplicative = _find_top_level_arithmetic_operator(expression, {"*", "/"})
    if multiplicative is not None:
        left, operator, right = multiplicative
        left_values = _arithmetic_expression_values(data, left, n)
        right_values = _arithmetic_expression_values(data, right, n)
        if operator == "*":
            return [left * right for left, right in zip(left_values, right_values, strict=True)]
        try:
            return list(map(truediv, left_values, right_values))
        except ZeroDivisionError:
            return list(map(_r_divide, left_values, right_values))

    if expression.startswith(("+", "-")):
        values = _arithmetic_expression_values(data, expression[1:].strip(), n)
        if expression[0] == "-":
            return [-value for value in values]
        return values

    power = _find_top_level_power_operator(expression)
    if power is not None:
        left, _operator, right = power
        left_values = _arithmetic_expression_values(data, left, n)
        right_values = _arithmetic_expression_values(data, right, n)
        try:
            return list(map(math.pow, left_values, right_values))
        except (OverflowError, ValueError):
            return list(map(_r_pow, left_values, right_values))

    literal = _arithmetic_literal(expression)
    if literal is not None:
        return [literal] * n
    if _top_level_comparison(expression) is not None:
        # a parenthesised comparison, TRUE counting 1: I((age > 60) + 0)
        return _floats_or_nan(_expression_values(data, expression, n))

    call = _numeric_call(expression)
    if call is not None:
        function, argument = call
        column, quoted = _formula_name(argument)
        if function == "as.numeric" and not _unsupported_formula_name(column, quoted):
            return _numeric_column(expression, _as_numeric_column(data, column), n)
        # a logical argument counts TRUE as 1 (R's as.numeric)
        values = _floats_or_nan(_expression_values(data, argument, n))
        return _apply_numeric_transform(values, function, argument)

    column, quoted = _formula_name(expression)
    if _unsupported_formula_name(column, quoted):
        raise ValueError(f"unsupported formula arithmetic term: {expression}")
    return _numeric_column(expression, _column(data, column), n)


def _numeric_column(expression: str, values: list[Any], n: int) -> list[float]:
    """The column *values* an arithmetic *expression* reads, as numbers."""

    if len(values) != n:
        raise ValueError("formula columns must have the same length as the Surv response")
    try:
        return _floats_or_nan(values)  # R's arithmetic keeps an NA missing
    except (TypeError, ValueError) as exc:
        raise ValueError(f"I() formula term {expression!r} requires numeric values") from exc


def _expression_values(data: Any, expression: str, n: int) -> list[Any]:
    """A formula variable written as R arithmetic (floats) or as a comparison (R's
    logical: ``True``/``False``, ``None`` where an operand is missing)."""

    comparison = _top_level_comparison(_strip_outer_formula_parentheses(expression))
    if comparison is None:
        return _arithmetic_expression_values(data, expression, n)
    left, operator, right = comparison
    return [
        _compare_response_values(a, operator, b)
        for a, b in zip(
            _comparison_operand(data, left, n), _comparison_operand(data, right, n), strict=True
        )
    ]


def _comparison_operand(data: Any, text: str, n: int) -> list[Any]:
    """One side of a formula comparison: a literal, a column (a factor as its labels)
    or arithmetic."""

    literal = _r_literal(text)
    if literal is not None:
        return [literal] * n
    column, quoted = _formula_name(text)
    if _unsupported_formula_name(column, quoted):
        return _arithmetic_expression_values(data, text, n)
    values = _comparison_column(data, column)
    if len(values) != n:
        raise ValueError("formula columns must have the same length as the Surv response")
    return values


# ---------------------------------------------------------------------------
# Literal vectors: the R vectors formula arguments spell out (cut breaks and
# labels, penalty options), evaluated from literals only, never with eval.
# ---------------------------------------------------------------------------


def _literal_vector(expression: str) -> list[Any]:
    """The R vector *expression* writes with literals only.

    Constants (numbers, strings, ``TRUE``/``FALSE``/``T``/``F``, ``Inf``; ``NULL`` is
    empty), ``c(...)``, ``from:to``, ``seq(...)``, ``seq_len(n)`` and ``+ - * / ^`` of
    those, the shorter operand recycled as R does.  Computed numbers are doubles.
    """

    text = _strip_outer_formula_parentheses(expression)
    literal = _r_literal(text)
    if literal is not None:
        return [literal]
    if text in {"T", "F"}:
        return [text == "T"]
    if text == "NULL":
        return []
    # split at the operator R binds loosest: + -, then * /, :, a sign, ^
    for operators in ({"+", "-"}, {"*", "/"}):
        split = _find_top_level_arithmetic_operator(text, operators, quotes=True)
        if split is not None:
            left, operator, right = split
            return _vector_arithmetic(_literal_vector(left), operator, _literal_vector(right))
    colons = [idx for idx, char in _top_level(text) if char == ":"]
    if colons:
        return _r_colon(
            _literal_vector(text[: colons[-1]]), _literal_vector(text[colons[-1] + 1 :])
        )
    if text.startswith(("-", "+")):
        sign = -1.0 if text[0] == "-" else 1.0
        return _vector_arithmetic([sign], "*", _literal_vector(text[1:]))
    split = _find_top_level_power_operator(text, quotes=True)
    if split is not None:
        return _vector_arithmetic(_literal_vector(split[0]), "^", _literal_vector(split[2]))
    function, opening, inner = text.partition("(")
    if opening and text.endswith(")"):
        arguments = _formula_response_parts(inner[:-1])
        if function == "c":
            values: list[Any] = []
            for argument in arguments:
                named = _formula_named_option(argument)
                values.extend(_literal_vector(argument if named is None else named[1]))
            return values
        if function == "seq":
            return _r_seq(arguments)
        if function == "seq_len" and len(arguments) == 1:
            return _seq_len(_numeric_scalar(_literal_vector(arguments[0]), "length.out"))
    raise ValueError(f"unsupported formula vector expression: {expression.strip()}")


def _numeric_vector(values: Sequence[Any]) -> list[float]:
    """``as.numeric`` of literal values: ``TRUE`` counts one; strings are an error."""

    if any(isinstance(value, str) for value in values):
        raise ValueError("non-numeric argument to a formula vector expression")
    return [float(value) for value in values]


def _numeric_scalar(values: Sequence[Any], name: str) -> float:
    if len(values) != 1:
        raise ValueError(f"'{name}' must be of length 1")
    return _numeric_vector(values)[0]


_VECTOR_OPERATORS = {"+": add, "-": sub, "*": mul, "/": _r_divide, "^": _r_pow}


def _vector_arithmetic(left: Sequence[Any], operator: str, right: Sequence[Any]) -> list[float]:
    """R's elementwise arithmetic, the shorter operand recycled."""

    x, y = _numeric_vector(left), _numeric_vector(right)
    if not x or not y:
        return []
    function = _VECTOR_OPERATORS[operator]
    return [function(x[i % len(x)], y[i % len(y)]) for i in range(max(len(x), len(y)))]


# C's FLT_EPSILON, the fuzz seq.c's seq_colon adds to the length of from:to
_FLT_EPSILON = 2.0**-23


def _r_colon(start: Sequence[Any], end: Sequence[Any]) -> list[float]:
    """R's ``from:to`` (seq.c's ``seq_colon``): steps of one from the first element of
    ``from`` towards ``to``, ``|to - from| + 1 + FLT_EPSILON`` of them truncated."""

    if not start or not end:
        raise ValueError("argument of length 0")
    first, last = _numeric_vector(start[:1])[0], _numeric_vector(end[:1])[0]
    if not (math.isfinite(first) and math.isfinite(last)):
        raise ValueError("NA/NaN argument")
    step = 1.0 if first <= last else -1.0
    return [first + step * k for k in range(int(abs(last - first) + 1 + _FLT_EPSILON))]


def _seq_len(count: float) -> list[float]:
    """R's ``seq_len(count)``."""

    return [float(k) for k in range(1, int(count) + 1)]


def _seq_length(start: float, stop: float, count: int) -> list[float]:
    """R's ``seq(from, to, length.out = count)``: ``from + i * by`` inside, ``to`` last."""

    if count <= 2:
        return [start, stop][:count]
    by = (stop - start) / (count - 1)
    return [start, *(start + i * by for i in range(1, count - 1)), stop]


# seq.default's formals before its ``...``
_SEQ_FORMALS = ("from", "to", "by", "length.out", "along.with")


def _r_seq(arguments: Sequence[str]) -> list[float]:
    """R's ``seq.default`` of literal arguments."""

    extra: list[str] = []
    matched = _match_arguments("seq", arguments, _SEQ_FORMALS, dots=extra)
    given = {name: _literal_vector(value) for name, value in matched.items()}
    if extra:
        # R's chkDots: the arguments in seq.default's ... are dropped with a warning
        names = ", ".join(f"'{name}'" for name in extra)
        plural = "s" if len(extra) > 1 else ""
        _warn_outside_package(f"extra argument{plural} {names} will be disregarded")
    elif set(given) == {"from"}:
        # seq(n) is 1:n, seq(x) of a vector 1:length(x) (R's nargs() == 1)
        only = given["from"]
        if len(only) != 1:
            return _seq_len(len(only))
        if not math.isfinite(_numeric_scalar(only, "from")):
            raise ValueError("'from' must be a finite number")
        return _r_colon([1.0], only)
    length: float | None = None
    if "along.with" in given:
        length = float(len(given["along.with"]))
    elif "length.out" in given:
        length = float(math.ceil(_numeric_vector(given["length.out"][:1])[0]))
    ends: dict[str, float] = {}
    for name in ("from", "to"):
        if name in given:
            value = _numeric_scalar(given[name], name)
            if not math.isfinite(value):
                raise ValueError(f"'{name}' must be a finite number")
            ends[name] = value
    by = _numeric_scalar(given["by"], "by") if "by" in given else None
    if length is None:
        start, stop = ends.get("from", 1.0), ends.get("to", 1.0)
        if by is None:
            return _r_colon([start], [stop])
        delta = stop - start
        if delta == 0.0 and stop == 0.0:
            return [stop]
        steps = _r_divide(delta, by)
        if not math.isfinite(steps):
            if by == 0.0 and delta == 0.0:
                return [start]
            raise ValueError("invalid '(to - from)/by'")
        if steps < 0.0:
            raise ValueError("wrong sign in 'by' argument")
        if abs(delta) / max(abs(stop), abs(start)) < 100 * sys.float_info.epsilon:
            return [start]
        values = [start + k * by for k in range(int(steps + 1e-10) + 1)]
        return [min(value, stop) if by > 0 else max(value, stop) for value in values]
    if not math.isfinite(length) or length < 0:
        raise ValueError("'length.out' must be a non-negative number")
    count = int(length)
    if count == 0:
        return []
    if not ends and by is None:
        return _seq_len(count)
    if by is None:
        start = ends.get("from", ends.get("to", 1.0) - (count - 1))
        stop = ends.get("to", start + (count - 1))
        return [start] * count if start == stop else _seq_length(start, stop, count)
    if "to" not in ends:
        start = ends.get("from", 1.0)
        return [start + k * by for k in range(count)]
    if "from" not in ends:
        return [ends["to"] - k * by for k in range(count - 1, -1, -1)]
    raise ValueError("too many arguments")


def _match_arguments(
    function: str,
    arguments: Sequence[str],
    formals: Sequence[str],
    *,
    dots: list[str] | None = None,
) -> dict[str, str]:
    """R's matching of the *arguments* of a call to *function*'s *formals*.

    Names match exactly, then as the unique prefix of a formal; the unnamed arguments
    fill the remaining formals in order.  An argument that matches no formal goes to
    *dots* when the function has R's ``...`` (its name, empty when unnamed), and is an
    error (R's ``unused argument``) otherwise.
    """

    matched: dict[str, str] = {}
    partial: list[tuple[str, str]] = []
    positional: list[str] = []
    for argument in arguments:
        named = _formula_named_option(argument)
        if named is None:
            positional.append(argument)
        elif named[0] in formals:
            if named[0] in matched:
                raise ValueError(
                    f'{function}(): formal argument "{named[0]}" matched by multiple actual '
                    "arguments"
                )
            matched[named[0]] = named[1]
        else:
            partial.append(named)
    exact = set(matched)
    for name, value in partial:
        candidates = [
            formal for formal in formals if formal.startswith(name) and formal not in exact
        ]
        if not candidates:
            if dots is None:
                raise ValueError(f"{function}(): unused argument ({name} = {value})")
            dots.append(name)
            continue
        if len(candidates) > 1:
            raise ValueError(f"{function}(): argument {name} matches multiple formal arguments")
        if candidates[0] in matched:
            raise ValueError(
                f'{function}(): formal argument "{candidates[0]}" matched by multiple actual '
                "arguments"
            )
        matched[candidates[0]] = value
    remaining = [formal for formal in formals if formal not in matched]
    if len(positional) > len(remaining):
        if dots is None:
            raise ValueError(f"{function}(): unused argument ({positional[len(remaining)]})")
        dots.extend("" for _ in positional[len(remaining) :])
    matched.update(zip(remaining, positional, strict=False))
    return matched


def _covariate_term_columns(term: _CovariateTerm) -> list[str]:
    if term.call is not None and term.call.split("(", 1)[0] in PENALTY_FUNCTIONS:
        return _penalty_arguments(term.call)[0]
    if term.arithmetic is not None:
        return _expression_columns(term.arithmetic)
    return [term.column]


def _formula_rhs_terms(formula: str, data: Any) -> _FormulaTerms:
    """The terms of *formula*'s right-hand side, a ``.`` standing for the other columns
    of *data*."""

    spec = _response_spec(formula)
    _lhs, _sep, rhs = formula.partition("~")
    return _split_terms(rhs, _dot_terms(data, [] if spec is None else list(spec.columns)))


def _formula_columns(formula: str, data: Any) -> list[str]:
    spec = _response_spec(formula)
    terms = _formula_rhs_terms(formula, data)
    columns = (
        ([] if spec is None else list(spec.columns))
        + _covariate_columns(terms.covariates)
        + terms.strata
        + _offset_columns(terms.offsets)
        + terms.clusters
    )
    return list(dict.fromkeys(columns))


def _data_rows(
    data: Any,
    columns: Sequence[str],
    rows: Sequence[int],
    n: int,
) -> _FormulaRows:
    """``data[rows, columns]``, the columns in *data*'s order; factor and ``tcut`` columns
    keep their attributes, and numeric columns stay numpy arrays."""

    index = np.asarray(rows, dtype=np.intp)
    frame = {
        name: _column_rows(_column_source(data, name), name, rows, index, n)
        for name in _data_order(data, columns)
    }
    return _FormulaRows(frame, len(rows))


def _data_order(data: Any, columns: Sequence[str]) -> list[str]:
    """*columns* in *data*'s column order."""

    names = _data_column_names(data)
    if names is None:
        return list(columns)
    used = set(columns)
    return [name for name in names if name in used]


def _column_rows(source: Any, name: str, rows: Sequence[int], index: np.ndarray, n: int) -> Any:
    """``source[rows]`` of the data column *name* (*index* is *rows* as an array)."""

    array = _numeric_ndarray(source)
    if array is not None:
        if len(array) != n:
            raise ValueError(f"variable lengths differ (found for '{name}')")
        return array[index]
    values = _coerce_array_like(source, name)
    if len(values) != n:
        raise ValueError(f"variable lengths differ (found for '{name}')")
    return _rows_of(source, [values[row] for row in rows])


def _formula_data_rows(
    formula: str,
    data: Any,
    rows: list[int],
    n: int,
) -> _FormulaRows:
    """``data[rows, ]`` restricted to the variables *formula* uses.

    R's ``model.frame`` evaluates only the formula's variables, so ``subset`` and
    ``na.action`` never copy the other columns of *data* (nor require them to be
    row-aligned).  The columns keep *data*'s order, so a ``.`` expands to the same
    terms afterwards.
    """

    return _data_rows(data, _formula_columns(formula, data), rows, n)


def _subset_formula_inputs(
    formula: str,
    data: Any,
    subset: Any,
    **row_aligned: Any,
) -> tuple[_FormulaRows, dict[str, Any]]:
    n = _data_row_count(data, formula)
    indices = _subset_indices(subset, n)
    filtered = {
        name: _subset_optional_sequence(values, indices, name)
        for name, values in row_aligned.items()
    }
    return _formula_data_rows(formula, data, indices, n), filtered


def _backwards_interval_rows(formula: str, data: Any, n: int) -> list[int]:
    """The rows of a ``Surv(start, stop, event)`` response with ``start >= stop``.

    R's ``Surv`` turns their start into ``NA`` (with its warning) while ``model.frame``
    evaluates the response, so ``na.action`` treats them as missing.  A ``NaN``
    endpoint compares false; ``None``/``pd.NA`` go through ``Surv``'s conversion.
    """

    spec = _response_spec(formula)
    if (
        spec is None
        or not spec.surv
        or len(spec.arguments) != 3
        or spec.type not in {None, "counting", "mstate"}
    ):
        return []
    start = _materialize_1d(_response_arg_values(data, spec.arguments[0], n), "time")
    stop = _materialize_1d(_response_arg_values(data, spec.arguments[1], n), "time2")
    try:
        rows = list(compress(range(n), map(ge, start, stop)))
    except TypeError:
        start = _time_column(start, "time", "Time variable is not numeric")
        stop = _time_column(stop, "time2", "Stop time is not numeric")
        rows = list(compress(range(n), map(ge, start, stop)))
    if rows:
        _warn_outside_package("Stop time must be > start time, NA created")
    return rows


def _response_variables(spec: _SurvResponseSpec | None) -> list[_CovariateTerm]:
    """The arithmetic arguments of the response, as the variables ``na.action`` evaluates.

    ``is.na(Surv(...))`` is true where one of them is NaN (``Surv((time * z) / z,
    status)`` at ``z = 0``).  An interval-censored response is left to ``is.na(Surv)``,
    which reads a missing endpoint as a censoring code.
    """

    if spec is None or spec.type in {"interval", "interval2"}:
        return []
    arguments = [_unwrap_response_identity(argument) for argument in spec.arguments]
    return [
        _CovariateTerm(argument, arithmetic=argument)
        for argument in arguments
        if _is_formula_arithmetic_expression(argument)
    ]


def _made_nan_rows(
    data: Any,
    variables: Iterable[_CovariateTerm],
    missing: set[int],
    n: int,
) -> tuple[set[int], dict[_CovariateTerm, list[float]]]:
    """The rows outside ``missing`` at which one of the formula ``variables`` is NaN, and
    the values at the rows outside ``missing`` of the variables it evaluated.

    R's ``model.frame`` evaluates every variable before ``na.action`` scans it, so a NaN
    made from values that are present (``log``/``sqrt`` of a negative value, ``0/0``,
    ``Inf - Inf``) is missing as well.  Only arithmetic and ``log``/``sqrt`` make one, so
    no other variable is evaluated; an interaction's product is not a variable.
    """

    variables = [
        term
        for term in dict.fromkeys(variables)
        if term.call is None and (term.arithmetic is not None or term.transform in {"log", "sqrt"})
    ]
    if not variables:
        return set(), {}
    rows: Sequence[int] = range(n)
    if missing:
        rows = [row for row in rows if row not in missing]
        data = _data_rows(data, _covariate_columns(variables), rows, n)
    values = {term: _numeric_variable(data, term, len(rows)) for term in variables}
    made: set[int] = set()
    for column in values.values():
        made.update(compress(rows, map(math.isnan, column)))
    return made, values


def _apply_formula_na_action(
    formula: str,
    data: Any,
    na_action: str | None,
    *,
    exclude_columns: Sequence[str] = (),
    missing_rows: Iterable[int] = (),
    **row_aligned: Any,
) -> tuple[Any, dict[str, Any], list[int]]:
    """``na.action`` on the formula's variables and the row-aligned arguments together:
    the data and arguments at the kept rows, and the 0-based rows it removed.

    ``exclude_columns`` are not scanned; ``missing_rows`` are rows the caller found
    missing otherwise (``is.na`` of an interval-censored response).
    """

    action = _normalize_na_action(na_action)
    if action == "pass":
        return data, row_aligned, []

    excluded = set(exclude_columns)
    sources = {
        column: _column_source(data, column)
        for column in _formula_columns(formula, data)
        if column not in excluded
    }
    n = _data_row_count(data, formula)
    missing = _missing_row_indices(
        [
            *sources.items(),
            *((name, values) for name, values in row_aligned.items() if values is not None),
        ],
        n,
    )
    missing.update(missing_rows)
    missing.update(_backwards_interval_rows(formula, sources, n))
    terms = _formula_rhs_terms(formula, data)
    variables = [
        *_response_variables(_response_spec(formula)),
        *(factor for term in terms.covariates for factor in _covariate_factors(term)),
        *terms.offsets,
    ]
    made, _values = _made_nan_rows(data, variables, missing, n)
    missing.update(made)
    keep = _keep_rows_after_na_action(missing, n, action, "formula data")
    if keep is None:
        return data, row_aligned, []
    filtered = {
        name: _subset_optional_sequence(values, keep, name) for name, values in row_aligned.items()
    }
    return _formula_data_rows(formula, data, keep, n), filtered, sorted(missing)


def _na_action_record(na_action: str | None, removed: Sequence[int]) -> NaAction | None:
    """R's ``attr(mf, "na.action")`` for the 0-based rows an ``na.action`` removed
    (none, ``NULL`` in R, when it removed nothing)."""

    if not removed:
        return None
    return NaAction(tuple(row + 1 for row in removed), _normalize_na_action(na_action))


def _data_column_names(data: Any) -> list[Any] | None:
    if isinstance(data, Mapping):
        return list(data)
    columns = getattr(data, "columns", None)
    if columns is None:
        return None
    return list(columns)


def _dot_terms(data: Any, response_terms: list[str]) -> list[str] | None:
    names = _data_column_names(data)
    if names is None:
        return None

    excluded = set(response_terms)
    terms: list[str] = []
    unsupported: list[Any] = []
    for name in names:
        if name in excluded:
            continue
        if not isinstance(name, str) or not name:
            unsupported.append(name)
            continue
        terms.append(name)

    if unsupported:
        joined = ", ".join(repr(name) for name in unsupported)
        raise ValueError(f"unsupported formula column name(s): {joined}")
    return terms


def _append_unique(target: list[Any], values: list[Any]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _remove_values(target: list[Any], values: list[Any]) -> None:
    remove = set(values)
    target[:] = [value for value in target if value not in remove]


def _unsupported_formula_name(name: str, quoted: bool) -> bool:
    return not quoted and any(token in name for token in "():*/+-^%=<>!&|")


def _factor_column_items(term: str) -> tuple[str, list[tuple[str, bool]]] | None:
    for wrapper in ("factor", "as.factor"):
        prefix = f"{wrapper}("
        if term.startswith(prefix) and term.endswith(")"):
            return wrapper, _formula_name_items(term[len(prefix) : -1])
    return None


def _transform_argument(term: str) -> tuple[str, str] | None:
    """``(transform, argument)`` of a ``log``/``sqrt``/``exp``/``I``/``identity``/
    ``as.numeric``/``tt`` term."""

    for wrapper in (*_NUMERIC_CALLS, "tt"):
        prefix = f"{wrapper}("
        if term.startswith(prefix) and term.endswith(")"):
            arguments = _formula_response_parts(term[len(prefix) : -1])
            if len(arguments) != 1:
                raise ValueError(f"{wrapper}() requires exactly one column")
            return wrapper, arguments[0]
    return None


def _interaction_from_factors(factors: list[_CovariateTerm]) -> _CovariateSpec:
    unique: list[_CovariateTerm] = []
    for factor in factors:
        if factor not in unique:
            unique.append(factor)
    if len(unique) == 1:
        return unique[0]
    return _InteractionTerm(tuple(unique))


def _interaction_from_terms(terms: tuple[_CovariateSpec, ...]) -> _CovariateSpec:
    factors = [factor for term in terms for factor in _covariate_factors(term)]
    return _interaction_from_factors(factors)


def _dot_covariate_terms(dot_terms: Sequence[str] | None) -> list[_CovariateSpec]:
    if dot_terms is None:
        raise ValueError("formula '.' requires named tabular data")
    return [_CovariateTerm(column) for column in dot_terms]


def _parse_covariate_atom(term: str) -> _CovariateTerm:
    if not term:
        raise ValueError("formula interaction terms must not be empty")

    factor_items = _factor_column_items(term)
    if factor_items is not None:
        wrapper, column_items = factor_items
        columns = [column for column, _quoted in column_items]
        if len(columns) != 1:
            raise ValueError(f"{wrapper}() requires exactly one column")
        if any(_unsupported_formula_name(column, quoted) for column, quoted in column_items):
            raise ValueError(f"unsupported formula term(s): {columns[0]}")
        return _CovariateTerm(
            columns[0],
            categorical=True,
            categorical_wrapper=wrapper,
        )

    transform_argument = _transform_argument(term)
    if transform_argument is not None:
        transform, argument = transform_argument
        column, quoted = _formula_name(argument)
        if not _unsupported_formula_name(column, quoted):
            return _CovariateTerm(column, transform=transform)
        if transform == "tt":
            raise ValueError(f"unsupported formula term(s): {column}")
        _expression_columns(argument)
        # I() and identity() keep a comparison logical, which the design codes as a
        # factor (I(sex == 2)TRUE); the other transforms count TRUE as 1
        logical = (
            transform in {"I", "identity"}
            and _top_level_comparison(_strip_outer_formula_parentheses(argument)) is not None
        )
        return _CovariateTerm(
            argument, categorical=logical, transform=transform, arithmetic=argument
        )

    call_term = _parse_call_term(term)
    if call_term is not None:
        return call_term

    term_name, quoted = _formula_name(term)
    if _unsupported_formula_name(term_name, quoted):
        # a bare comparison (age > 60) is a logical term in R, coded like I(age > 60)
        if _top_level_comparison(term) is not None:
            _expression_columns(term)
            return _CovariateTerm(term, categorical=True, arithmetic=term)
        raise ValueError(f"unsupported formula term(s): {term_name}")
    return _CovariateTerm(term_name)


_CALL_TERMS = ("tcut", "cut", *PENALTY_FUNCTIONS)


def _penalty_arguments(call: str) -> tuple[list[str], dict[str, Any]]:
    """Parse data-column arguments and literal penalty options without eval."""

    columns = []
    options = {}
    for argument in _formula_response_parts(call.split("(", 1)[1][:-1]):
        named = _formula_named_option(argument)
        if named is None:
            name, quoted = _formula_name(argument)
            if _unsupported_formula_name(name, quoted):
                raise ValueError(f"unsupported penalty variable {argument!r}")
            columns.append(name)
            continue
        name, value = named
        if name in options:
            raise ValueError(f"duplicate penalty option {name!r}")
        if value == "NULL":
            options[name] = None
            continue
        try:
            options[name] = _parse_formula_literal(value)
        except ValueError:
            # R has no scalars: a vector of length one (df = c(4), theta = 1/2,
            # method = c("aic")) is the scalar the penalty takes
            values = _literal_vector(value)
            options[name] = values[0] if len(values) == 1 else values
    if not columns:
        raise ValueError("penalty terms require a data column")
    return columns, options


# The formals of the categorising calls pyears evaluates, in R's order
_CALL_FORMALS = {
    "tcut": ("x", "breaks", "labels", "scale"),
    "cut": ("x", "breaks", "labels", "include.lowest", "right", "dig.lab", "ordered_result"),
}


def _call_arguments(call: str) -> dict[str, str]:
    """The arguments of a ``tcut()``/``cut()`` term by formal name, matched as R does."""

    function, _sep, inner = call.partition("(")
    return _match_arguments(function, _formula_response_parts(inner[:-1]), _CALL_FORMALS[function])


def _parse_call_term(term: str) -> _CovariateTerm | None:
    """A ``tcut(x, ...)``/``cut(x, ...)`` or penalty term: categorical, reading the
    column of its variable ``x``."""

    for function in _CALL_TERMS:
        prefix = f"{function}("
        if not (term.startswith(prefix) and term.endswith(")")):
            continue
        if function in _CALL_FORMALS:
            x = _call_arguments(term).get("x")
            if x is None:
                raise ValueError(f"{function}() requires a variable")
        else:
            x = _penalty_arguments(term)[0][0]
        column, quoted = _formula_name(x)
        if not _unsupported_formula_name(column, quoted):
            return _CovariateTerm(column, categorical=True, call=term)
        columns = _expression_columns(x)
        if not columns:
            raise ValueError(f"{function}() requires a data column")
        return _CovariateTerm(columns[0], categorical=True, arithmetic=x, call=term)
    return None


def _parse_interaction_term(
    term: str,
    dot_terms: Sequence[str] | None,
) -> list[_CovariateSpec]:
    parts = _split_top_level(term, ":")
    if len(parts) == 1:
        if parts[0] == ".":
            return _dot_covariate_terms(dot_terms)
        return [_parse_covariate_atom(parts[0])]

    parsed_groups = [_parse_covariate_expression(part, dot_terms) for part in parts]
    interactions: list[_CovariateSpec] = []
    for term_combo in product(*parsed_groups):
        _append_unique(interactions, [_interaction_from_terms(term_combo)])
    return interactions


def _parse_offset_term(expression: str) -> _CovariateTerm:
    expression = expression.strip()
    if _is_formula_arithmetic_expression(expression):
        _arithmetic_expression_columns(expression)
        return _CovariateTerm(expression, arithmetic=expression)
    offset_term = _parse_covariate_atom(expression)
    if offset_term.categorical:
        raise ValueError("offset() requires a numeric column or transform")
    return offset_term


def _parse_formula_power_degree(value: str) -> int:
    text = value.strip()
    if not text.isdigit():
        raise ValueError("formula ^ degree must be a nonnegative integer")
    return int(text)


def _parse_formula_power_base_terms(
    term: str,
    dot_terms: Sequence[str] | None,
) -> list[_CovariateSpec]:
    expression = _strip_outer_formula_parentheses(term)
    terms: list[_CovariateSpec] = []
    for op, base_term in _formula_tokens(expression):
        if base_term in {"0", "1"}:
            continue
        parsed = _parse_covariate_expression(base_term, dot_terms)
        if op == "-":
            _remove_values(terms, parsed)
        else:
            _append_unique(terms, parsed)
    return terms


def _parse_formula_power_expression(
    term: str,
    dot_terms: Sequence[str] | None,
) -> list[_CovariateSpec] | None:
    parts = _split_top_level(term, "^")
    if len(parts) == 1:
        return None
    if len(parts) != 2:
        raise ValueError("formula ^ expressions must contain one degree")

    base_terms = _parse_formula_power_base_terms(parts[0], dot_terms)
    degree = _parse_formula_power_degree(parts[1])
    if degree == 0 or not base_terms:
        return []

    expanded: list[_CovariateSpec] = []
    for size in range(1, min(degree, len(base_terms)) + 1):
        for term_combo in combinations(base_terms, size):
            _append_unique(expanded, [_interaction_from_terms(term_combo)])
    return expanded


def _parse_parenthesized_formula_expression(
    term: str,
    dot_terms: Sequence[str] | None,
) -> list[_CovariateSpec] | None:
    stripped = _strip_outer_formula_parentheses(term)
    if stripped == term.strip():
        return None
    return _parse_formula_power_base_terms(stripped, dot_terms)


def _parse_covariate_expression(
    term: str,
    dot_terms: Sequence[str] | None,
) -> list[_CovariateSpec]:
    if term == ".":
        return _dot_covariate_terms(dot_terms)

    power_terms = _parse_formula_power_expression(term, dot_terms)
    if power_terms is not None:
        return power_terms

    grouped_terms = _parse_parenthesized_formula_expression(term, dot_terms)
    if grouped_terms is not None:
        return grouped_terms

    in_parts = _split_top_level_token(term, "%in%")
    if len(in_parts) > 1:
        # a %in% b is a:b; the fit orders each interaction's factors as R's terms() does
        parsed_groups = [_parse_interaction_term(part, dot_terms) for part in in_parts]
        nested_expanded = parsed_groups[0]
        for nested_group in parsed_groups[1:]:
            next_expanded: list[_CovariateSpec] = []
            for current, nested in product(nested_expanded, nested_group):
                _append_unique(next_expanded, [_interaction_from_terms((current, nested))])
            nested_expanded = next_expanded
        return nested_expanded

    nested_parts = _split_top_level(term, "/")
    if len(nested_parts) > 1:
        parsed_parts = [_parse_interaction_term(part, dot_terms) for part in nested_parts]
        slash_expanded: list[_CovariateSpec] = []
        current_group = parsed_parts[0]
        _append_unique(slash_expanded, current_group)
        for nested_group in parsed_parts[1:]:
            current_group = [
                _interaction_from_terms((current, nested))
                for current, nested in product(current_group, nested_group)
            ]
            _append_unique(slash_expanded, current_group)
        return slash_expanded

    parts = _split_top_level(term, "*")
    if len(parts) == 1:
        return _parse_interaction_term(parts[0], dot_terms)

    crossed_groups: list[list[_CovariateSpec]] = [
        _parse_covariate_expression(part, dot_terms) for part in parts
    ]
    crossed_expanded: list[_CovariateSpec] = []
    for group in crossed_groups:
        _append_unique(crossed_expanded, group)
    for size in range(2, len(crossed_groups) + 1):
        for group_combo in combinations(crossed_groups, size):
            for term_combo in product(*group_combo):
                _append_unique(crossed_expanded, [_interaction_from_terms(term_combo)])
    return crossed_expanded


def _materialize_formula_terms(terms: _CachedFormulaTerms) -> _FormulaTerms:
    return _FormulaTerms(
        covariates=list(terms.covariates),
        strata=list(terms.strata),
        offsets=list(terms.offsets),
        clusters=list(terms.clusters),
        model_terms=list(terms.model_terms),
        intercept=terms.intercept,
    )


@lru_cache(maxsize=512)
def _split_terms_cached(
    rhs: str,
    dot_terms: tuple[str, ...] | None = None,
) -> _CachedFormulaTerms:
    covariates: list[_CovariateSpec] = []
    strata: list[str] = []
    offsets: list[_CovariateTerm] = []
    clusters: list[str] = []
    model_terms: list[_FormulaModelTerm] = []
    unsupported: list[str] = []
    intercept = True

    for op, term in _formula_tokens(rhs):
        if not term:
            continue
        if term == "1":
            intercept = op != "-"
            continue
        if term == "0":
            intercept = op == "-"
            continue
        if term == ".":
            if dot_terms is None:
                raise ValueError("formula '.' requires named tabular data")
            terms = [_CovariateTerm(column) for column in dot_terms]
            model_items = [_ModelCovariateTerm(item) for item in terms]
            if op == "-":
                _remove_values(covariates, terms)
                _remove_values(model_terms, model_items)
            else:
                _append_unique(covariates, terms)
                _append_unique(model_terms, model_items)
            continue
        if term.startswith("strata(") and term.endswith(")"):
            column_items = _formula_name_items(term[7:-1])
            columns = [column for column, _quoted in column_items]
            if not columns:
                raise ValueError("strata() requires at least one column")
            unsupported.extend(
                column
                for column, quoted in column_items
                if _unsupported_formula_name(column, quoted)
            )
            if op == "-":
                _remove_values(strata, columns)
                _remove_values(model_terms, [_ModelStrataTerm(tuple(columns))])
            else:
                _append_unique(strata, columns)
                _append_unique(model_terms, [_ModelStrataTerm(tuple(columns))])
            continue
        if term.startswith("cluster(") and term.endswith(")"):
            column_items = _formula_name_items(term[8:-1])
            columns = [column for column, _quoted in column_items]
            if not columns:
                raise ValueError("cluster() requires at least one column")
            unsupported.extend(
                column
                for column, quoted in column_items
                if _unsupported_formula_name(column, quoted)
            )
            if op == "-":
                _remove_values(clusters, columns)
                _remove_values(model_terms, [_ModelClusterTerm(column) for column in columns])
            else:
                _append_unique(clusters, columns)
                _append_unique(model_terms, [_ModelClusterTerm(column) for column in columns])
            continue
        if term.startswith("offset(") and term.endswith(")"):
            offset_term = _parse_offset_term(term[7:-1])
            model_item = _ModelOffsetTerm(offset_term)
            if op == "-":
                _remove_values(offsets, [offset_term])
                _remove_values(model_terms, [model_item])
            else:
                _append_unique(offsets, [offset_term])
                _append_unique(model_terms, [model_item])
            continue
        covariate_terms = _parse_covariate_expression(term, dot_terms)
        model_items = [_ModelCovariateTerm(item) for item in covariate_terms]
        if op == "-":
            _remove_values(covariates, covariate_terms)
            _remove_values(model_terms, model_items)
        else:
            _append_unique(covariates, covariate_terms)
            _append_unique(model_terms, model_items)

    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"unsupported formula term(s): {joined}")
    return _CachedFormulaTerms(
        covariates=tuple(covariates),
        strata=tuple(strata),
        offsets=tuple(offsets),
        clusters=tuple(clusters),
        model_terms=tuple(model_terms),
        intercept=intercept,
    )


def _split_terms(rhs: str, dot_terms: list[str] | None = None) -> _FormulaTerms:
    # R's comparisons bind looser than + and -, so ~ age > 60 + sex is the single
    # term age > (60 + sex); refuse the mixed form rather than fit another model
    if len(_formula_tokens(rhs)) > 1 and any(
        rhs.startswith(op, idx) for idx, _char in _top_level(rhs) for op in ("==", "!=", "<", ">")
    ):
        raise ValueError(
            "a comparison in a formula with other terms must be wrapped in I(), "
            "e.g. ~ I(age > 60) + sex"
        )
    dot_key = None if dot_terms is None else tuple(dot_terms)
    return _materialize_formula_terms(_split_terms_cached(rhs, dot_key))


def _parse_formula(formula: str, data: Any) -> tuple[Surv, _FormulaTerms]:
    _lhs, sep, rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")

    response_spec = _formula_response_spec(formula)
    surv = _surv_from_spec(data, response_spec)
    terms = _split_terms(rhs, _dot_terms(data, response_spec.columns))
    return surv, terms


def _r_log(value: float) -> float:
    """R's ``log``: ``-Inf`` at zero and NaN below it."""

    if value > 0.0:
        return math.log(value)
    return -math.inf if value == 0.0 else math.nan


def _r_sqrt(value: float) -> float:
    """R's ``sqrt``: NaN below zero."""

    return math.sqrt(value) if value >= 0.0 else math.nan


def _apply_numeric_transform(values: list[float], transform: str | None, term: str) -> list[float]:
    if transform is None:
        return values
    if transform in {"log", "sqrt"}:
        try:
            return list(map(math.log if transform == "log" else math.sqrt, values))
        except ValueError:
            if any(value < 0.0 for value in values):
                _warn_outside_package(f"NaNs produced in {transform}({term})")
            return list(map(_r_log if transform == "log" else _r_sqrt, values))
    if transform == "exp":
        return [math.exp(value) for value in values]
    if transform in {"I", "identity", "as.numeric", "tt"}:
        return values
    raise ValueError(f"unsupported formula transform {transform!r}")


def _numeric_term_values(values: list[Any], term: _CovariateTerm) -> list[float]:
    try:
        numeric = _floats_or_nan(values)
    except (TypeError, ValueError) as exc:
        if term.transform is not None:
            raise ValueError(
                f"{term.transform}() formula term {term.column!r} requires numeric values"
            ) from exc
        raise
    return _apply_numeric_transform(numeric, term.transform, term.column)


def _numeric_variable(
    data: Any,
    term: _CovariateTerm,
    n: int,
    evaluated: Mapping[_CovariateTerm, list[float]] | None = None,
) -> list[float]:
    """The numeric formula variable ``term`` at the rows of *data*, taken from
    ``evaluated`` (variables already evaluated at those rows) when it is there."""

    if evaluated and term in evaluated:
        return evaluated[term]
    return _numeric_term_values(_term_raw_values(data, term, n), term)


def _term_raw_values(data: Any, term: _CovariateTerm, n: int) -> list[Any]:
    if term.call is not None:
        raise ValueError(f"unsupported formula term(s): {term.call}")
    if term.arithmetic is not None:
        return _expression_values(data, term.arithmetic, n)
    if term.transform == "as.numeric":
        values = _as_numeric_column(data, term.column)
    else:
        values = _column(data, term.column)
    if len(values) != n:
        raise ValueError("formula columns must have the same length as the Surv response")
    return values


def _term_values(data: Any, term: _CovariateSpec, n: int) -> list[Any]:
    if isinstance(term, _InteractionTerm):
        factor_values = [_term_values(data, factor, n) for factor in term.factors]
        if any(factor.categorical for factor in term.factors):
            return [tuple(values[idx] for values in factor_values) for idx in range(n)]
        try:
            numeric_values = [[float(value) for value in values] for values in factor_values]
        except (TypeError, ValueError):
            return [tuple(values[idx] for values in factor_values) for idx in range(n)]
        return [math.prod(values[idx] for values in numeric_values) for idx in range(n)]

    values = _term_raw_values(data, term, n)
    if term.transform is None or term.categorical:
        return values
    return _numeric_term_values(values, term)


def _categorical_levels(values: list[Any], column: str) -> tuple[Any, ...]:
    """The distinct values of a categorical term in order of appearance; a missing
    value is not a level (R's ``factor()``)."""

    labels: dict[Any, None] = {}
    for value in values:
        if _is_missing_value(value):
            continue
        try:
            labels.setdefault(value, None)
        except TypeError as exc:
            message = f"categorical formula term {column!r} contains unhashable values"
            raise TypeError(message) from exc
    levels = tuple(labels)
    if len(levels) < 2:
        raise ValueError(f"categorical formula term {column!r} must have at least two levels")
    return levels


def _fit_single_design_term(
    data: Any,
    term: _CovariateTerm,
    n: int,
    full_data: Any,
) -> _SingleDesignTerm:
    if term.call is not None and term.call.split("(", 1)[0] in PENALTY_FUNCTIONS:
        columns, options = _penalty_arguments(term.call)
        values = {column: _column(full_data, column) for column in columns}
        levels = _mstate_categories(_column_source(full_data, columns[0]))
        return fit_penalty(term, columns, values, options, levels)
    values = _term_raw_values(data, term, n)
    if not term.categorical and (
        term.transform is not None or _mstate_categories(_column_source(data, term.column)) is None
    ):
        if term.transform is not None:
            _numeric_term_values(values, term)
            return _NumericDesignTerm(term)
        try:
            _numeric_term_values(values, term)
        except (TypeError, ValueError):
            pass
        else:
            return _NumericDesignTerm(term)
    return _CategoricalDesignTerm(term, _categorical_levels(values, term.column))


def _fit_design_term(
    data: Any,
    term: _CovariateSpec,
    n: int,
    full_data: Any,
    factor_order: Mapping[_CovariateTerm, int] | None = None,
) -> _DesignTerm:
    if isinstance(term, _InteractionTerm):
        factors = term.factors
        if factor_order is not None:
            factors = tuple(sorted(factors, key=factor_order.__getitem__))
        fitted = tuple(_fit_single_design_term(data, factor, n, full_data) for factor in factors)
        if any(isinstance(factor, _PenaltyDesignTerm) and factor.penalized for factor in fitted):
            raise ValueError("Penalty terms cannot be in an interaction")
        return _InteractionDesignTerm(fitted)
    return _fit_single_design_term(data, term, n, full_data)


def _formula_factor_order(terms: Sequence[_CovariateSpec]) -> dict[_CovariateTerm, int]:
    order: dict[_CovariateTerm, int] = {}
    for term in terms:
        for factor in _covariate_factors(term):
            order.setdefault(factor, len(order))
    return order


def _formula_model_term_degree(term: _FormulaModelTerm) -> int:
    if isinstance(term, _ModelCovariateTerm):
        return len(_covariate_factors(term.term))
    if isinstance(term, _ModelStrataTerm):
        return 1
    return 0


def _categorical_design_factors(spec: _DesignTerm) -> list[_CategoricalDesignTerm]:
    if isinstance(spec, _CategoricalDesignTerm):
        return [spec]
    if isinstance(spec, _InteractionDesignTerm):
        return [factor for factor in spec.factors if isinstance(factor, _CategoricalDesignTerm)]
    return []


def _set_full_categorical_factors(
    spec: _DesignTerm,
    full_factors: set[_CovariateTerm],
) -> _DesignTerm:
    if isinstance(spec, _CategoricalDesignTerm):
        return _CategoricalDesignTerm(
            spec.term,
            spec.levels,
            full=spec.term in full_factors,
        )
    if isinstance(spec, _InteractionDesignTerm):
        return _InteractionDesignTerm(
            tuple(
                _CategoricalDesignTerm(
                    factor.term,
                    factor.levels,
                    full=factor.term in full_factors,
                )
                if isinstance(factor, _CategoricalDesignTerm)
                else factor
                for factor in spec.factors
            )
        )
    return spec


def _fit_formula_design(
    data: Any,
    response_spec: _SurvResponseSpec,
    terms: _FormulaTerms,
    n: int,
    *,
    include_intercept: bool = False,
    full_data: Any | None = None,
) -> _FormulaDesign:
    """The design of *terms* on the *n* rows of *data*.

    *full_data* is the data before ``subset`` and ``na.action`` (by default *data*): R's
    ``model.frame`` evaluates the ``pspline``, ``ridge`` and ``frailty`` terms on it.
    """

    if full_data is None:
        full_data = data
    factor_order = _formula_factor_order(terms.covariates)
    ordered_terms = sorted(terms.covariates, key=lambda term: len(_covariate_factors(term)))
    ordered_model_terms = sorted(
        (
            term
            for term in terms.model_terms
            if isinstance(term, _ModelCovariateTerm | _ModelStrataTerm)
        ),
        key=_formula_model_term_degree,
    )
    term_assignments: dict[_CovariateSpec, int] = {}
    for term_index, model_term in enumerate(ordered_model_terms, start=1):
        if isinstance(model_term, _ModelCovariateTerm):
            term_assignments[model_term.term] = term_index
    contrast_intercept = terms.intercept if include_intercept else True
    covered_terms: set[frozenset[_CovariateTerm]] = {frozenset()} if contrast_intercept else set()
    promoted_no_intercept_factor = contrast_intercept
    design_terms: list[_DesignTerm] = []
    for term in ordered_terms:
        fitted_term = _fit_design_term(data, term, n, full_data, factor_order)
        raw_factors = frozenset(_covariate_factors(term))
        categorical_factors = _categorical_design_factors(fitted_term)
        full_factors = {
            factor.term
            for factor in categorical_factors
            if raw_factors - {factor.term} not in covered_terms
        }
        if not promoted_no_intercept_factor and categorical_factors:
            full_factors.add(categorical_factors[0].term)
            promoted_no_intercept_factor = True
        fitted_term = _set_full_categorical_factors(fitted_term, full_factors)
        design_terms.append(fitted_term)
        if (
            len(raw_factors) == 1
            and isinstance(fitted_term, _CategoricalDesignTerm)
            and fitted_term.full
        ):
            covered_terms.add(frozenset())
        covered_terms.add(raw_factors)
    return _FormulaDesign(
        response=response_spec,
        covariates=tuple(design_terms),
        offsets=tuple(terms.offsets),
        term_assignments=tuple(term_assignments[term] for term in ordered_terms),
        strata=tuple(terms.strata),
        intercept=include_intercept and terms.intercept,
    )


def _single_design_columns(
    data: Any,
    spec: _SingleDesignTerm,
    n: int,
    evaluated: Mapping[_CovariateTerm, list[float]] | None = None,
) -> list[list[float]]:
    if isinstance(spec, _PenaltyDesignTerm):
        values = {column: _column(data, column) for column in spec.columns}
        if any(len(value) != n for value in values.values()):
            raise ValueError("formula columns must have the same length as the Surv response")
        return penalty_columns(spec, values)
    if isinstance(spec, _NumericDesignTerm):
        return [_numeric_variable(data, spec.term, n, evaluated)]

    values = _term_raw_values(data, spec.term, n)
    levels = spec.levels
    # model.matrix of an na.pass frame: a missing value is NA in every column
    missing: list[int] = []
    for row, value in enumerate(values):
        if all(value != level for level in levels):
            if not _is_missing_value(value):
                raise ValueError(
                    f"newdata column {spec.term.column!r} contains unknown level {value!r}"
                )
            missing.append(row)
    encoded_levels = levels if spec.full else levels[1:]
    columns = [[1.0 if value == level else 0.0 for value in values] for level in encoded_levels]
    for column in columns:
        for row in missing:
            column[row] = math.nan
    return columns


def _design_term_columns(
    data: Any,
    spec: _DesignTerm,
    n: int,
    evaluated: Mapping[_CovariateTerm, list[float]] | None = None,
) -> list[list[float]]:
    if isinstance(spec, _InteractionDesignTerm):
        factor_columns = [
            _single_design_columns(data, factor, n, evaluated) for factor in spec.factors
        ]
        interaction_columns: list[list[float]] = []
        for reversed_combo in product(*reversed(factor_columns)):
            column_combo = tuple(reversed(reversed_combo))
            interaction_columns.append(
                [math.prod(column[idx] for column in column_combo) for idx in range(n)]
            )
        return interaction_columns
    return _single_design_columns(data, spec, n, evaluated)


def _design_rows_from_spec(
    data: Any,
    design: _FormulaDesign,
    n: int,
    *,
    evaluated: Mapping[_CovariateTerm, list[float]] | None = None,
) -> list[list[float]]:
    """The rows of the design matrix of *data*.  ``evaluated`` holds numeric variables
    already evaluated at its rows (the ``tt()`` terms' values, the variables the
    ``na.action`` scan evaluated), which are not evaluated again."""

    columns = [
        column
        for term in design.covariates
        for column in _design_term_columns(data, term, n, evaluated)
    ]
    if design.intercept:
        columns.insert(0, [1.0] * n)
    if not columns:
        return [[] for _row in range(n)]
    return list(map(list, zip(*columns, strict=True)))


def _covariate_term_name(term: _CovariateTerm) -> str:
    if term.call is not None:
        return term.call
    if term.transform is not None:
        return f"{term.transform}({term.column})"
    if term.categorical_wrapper is not None:
        return f"{term.categorical_wrapper}({term.column})"
    return term.column


def _display_single_design_term(spec: _SingleDesignTerm) -> str:
    return _covariate_term_name(spec.term)


def _design_term_name(spec: _DesignTerm) -> str:
    if isinstance(spec, _InteractionDesignTerm):
        return ":".join(_display_single_design_term(factor) for factor in spec.factors)
    return _display_single_design_term(spec)


def _single_design_term_output_names(spec: _SingleDesignTerm) -> list[str]:
    if isinstance(spec, _PenaltyDesignTerm):
        return list(spec.names)
    term = spec.term
    if isinstance(spec, _CategoricalDesignTerm):
        prefix = _covariate_term_name(term)
        levels = spec.levels if spec.full else spec.levels[1:]
        return [f"{prefix}{_strata_value_label(level)}" for level in levels]
    return [_covariate_term_name(term)]


def _design_term_output_names(spec: _DesignTerm) -> list[str]:
    if isinstance(spec, _InteractionDesignTerm):
        factor_names = [_single_design_term_output_names(factor) for factor in spec.factors]
        return [":".join(reversed(combo)) for combo in product(*reversed(factor_names))]
    return _single_design_term_output_names(spec)


def _design_term_columns_used(spec: _DesignTerm) -> list[str]:
    if isinstance(spec, _InteractionDesignTerm):
        columns: list[str] = []
        for factor in spec.factors:
            _append_unique(columns, _covariate_term_columns(factor.term))
        return columns
    return _covariate_term_columns(spec.term)


def _formula_design_columns(design: _FormulaDesign) -> list[str]:
    columns = [column for term in design.covariates for column in _design_term_columns_used(term)]
    columns.extend(_offset_columns(design.offsets))
    return list(dict.fromkeys(columns))


def _formula_model_frame(
    data: Any,
    response: Surv,
    design: _FormulaDesign,
    *,
    extra_columns: Sequence[str] = (),
    weights: Any | None = None,
    offset: Any | None = None,
    offsets: Any | None = None,
    strata: Any | None = None,
    cluster: Any | None = None,
    id: Any | None = None,
    istate: Any | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {design.response.name: response}
    columns: list[str] = []
    _append_unique(columns, design.response.columns)
    _append_unique(columns, _formula_design_columns(design))
    _append_unique(columns, list(design.strata))
    _append_unique(columns, list(extra_columns))
    for column in columns:
        frame[column] = _column(data, column)
    for name, values in (
        ("(weights)", weights),
        ("(offset)", offsets if offsets is not None else offset),
        ("(strata)", strata),
        ("(cluster)", cluster),
        ("(id)", id),
        ("(istate)", istate),
    ):
        if values is not None:
            frame[name] = _materialize_1d(values, name)
    return frame


def _formula_design_row_count(data: Any, design: _FormulaDesign) -> int:
    columns = _formula_design_columns(design)
    if columns:
        return len(_column(data, columns[0]))
    if isinstance(data, Mapping) and data:
        name, values = next(iter(data.items()))
        return len(_materialize_1d(values, str(name)))
    raise ValueError("newdata must include at least one column")


def _combine_aligned_columns(columns: list[list[Any]], n: int) -> list[Any]:
    if any(len(column) != n for column in columns):
        raise ValueError("formula columns must have the same length as the Surv response")
    if len(columns) == 1:
        return columns[0]
    return [tuple(column[i] for column in columns) for i in range(n)]


def _combined_columns(data: Any, terms: list[str], n: int) -> list[Any]:
    return _combine_aligned_columns([_column(data, term) for term in terms], n)


def _offset_vector(
    data: Any,
    terms: Sequence[_CovariateTerm],
    n: int,
    evaluated: Mapping[_CovariateTerm, list[float]] | None = None,
) -> list[float] | None:
    if not terms:
        return None
    columns = [_numeric_variable(data, term, n, evaluated) for term in terms]
    return [sum(column[i] for column in columns) for i in range(n)]


def _column_or_values(data: Any, values: Any, name: str) -> Any:
    if isinstance(values, str):
        if data is None:
            raise ValueError(f"{name} column lookup requires data")
        return _column(data, values)
    return values


# ---------------------------------------------------------------------------
# model.frame: the one path from (formula, data, subset, na.action, weights,
# ...) to a row-aligned frame that every fitter starts from.
# ---------------------------------------------------------------------------

_MODEL_FRAME_ARGUMENTS = ("weights", "offset", "id", "cluster", "istate")


def _surv_from_spec(data: Any, spec: _SurvResponseSpec) -> Surv:
    """Evaluate a ``Surv(...)`` response spec against *data*."""

    if spec.timeline:
        raise ValueError("response must be a survival object")
    args = _formula_response_values(data, spec)
    if len(args) not in {1, 2, 3}:
        raise ValueError("Surv(...) formula response must have 1, 2, or 3 column arguments")
    return Surv(*args, type=spec.type, origin=spec.origin)


def _surv2_from_spec(data: Any, spec: _SurvResponseSpec) -> Surv2:
    """Evaluate a ``Surv2(...)`` response spec against *data*."""

    time, event = _formula_response_values(data, spec)
    return Surv2(time, event, spec.repeated)


# ---------------------------------------------------------------------------
# Timeline data: R's surv2counting (R/fromtimeline.R), which coxph, survfit,
# survcheck and fromtimeline run on the model frame of a Surv2 response
# before its na.action.
# ---------------------------------------------------------------------------

_TSTART, _TSTOP, _STATUS = "(tstart)", "(tstop)", "(status)"


def _timeline_counting(
    formula: str,
    data: Any,
    subset: Any | None,
    arguments: Mapping[str, Any],
    *,
    repeated: Any | None = None,
    lvcf: bool = True,
    require_repeats: bool = False,
    carry_clusters: bool = True,
) -> tuple[str, _FormulaRows, dict[str, Any]]:
    """R's ``surv2counting(mf)`` for the ``Surv2`` formula *formula* (or a ``Surv(time,
    status)`` one) on *data* after ``subset``, with the row-aligned *arguments* (vectors or
    column names of *data*).

    Each subject's rows pair up: row ``j + 1`` gives the end time and outcome of the
    interval that row ``j`` starts, and supplies everything else.  With *lvcf* a missing
    formula variable takes the subject's last value (the ``(...)`` arguments are left
    alone), and when every subject starts in a state the ``istate`` argument becomes that
    current state, which an ``istate`` given must agree with.  The result is a formula with
    the counting-process response ``Surv([(tstart), ](tstop), (status))`` (one interval per
    subject gives the right-censored form), the data it reads and the arguments at its
    rows, for the caller's ``na.action``.  *repeated* overrides the response's own;
    *require_repeats* refuses data in which no subject has two rows (fromtimeline's check).
    Without *carry_clusters* a ``cluster()`` term's variable is not carried forward either:
    coxph.R turns that term into its ``cluster`` argument before the model frame.
    """

    arguments = {
        name: None if value is None else _column_or_values(data, value, name)
        for name, value in arguments.items()
    }
    if subset is not None:
        data, arguments = _subset_formula_inputs(formula, data, subset, **arguments)
    spec = _formula_response_spec(formula)
    if spec.timeline:
        response: Surv | Surv2 = _surv2_from_spec(data, spec)
        time, status = response.time, response.status
        if repeated is None:
            repeated = response.repeated
    else:
        # fromtimeline's Surv(time, status) form of a timeline
        response = _surv_from_spec(data, spec)
        if response.type in {"counting", "mcounting"}:
            raise ValueError("response cannot be of counting process type")
        if response.type not in {"right", "mright"}:
            raise ValueError(f"not valid for {response.type} censored data")
        time, status = response.time, response.event
    n = len(time)
    ids = arguments.get("id")
    if ids is None or len(ids := _materialize_labels(ids, "id")) != n:
        raise ValueError("id statement is required")
    if require_repeats and len(set(map(_hashable_group_value, ids))) == n:
        raise ValueError("data does not appear to be timeline data")
    if any(map(_is_missing_value, ids)) or any(map(math.isnan, time)):
        raise ValueError("id and time cannot be missing")
    _lhs, _sep, rhs = formula.partition("~")
    terms = _formula_rhs_terms(formula, data)
    model_columns = (
        _covariate_columns(terms.covariates) + terms.strata + _offset_columns(terms.offsets)
    )
    variables = _data_order(data, model_columns + terms.clusters)
    sources = {name: _column_source(data, name) for name in variables}
    carried = set(variables if carry_clusters else model_columns)
    # the variables to carry forward: those with a missing value
    masks: dict[str, list[bool]] = {}
    if lvcf:
        for name, source in sources.items():
            if name not in carried:
                continue
            missing = _missing_row_indices([(name, source)], n)
            if missing:
                masks[name] = [row in missing for row in range(n)]
    result = _core.surv2counting(
        ids,
        list(time),
        list(status),
        bool(response.states),
        _repeated_option(repeated),
        list(masks.values()),
    )
    rows = list(result.row)
    carry_from = dict(zip(masks, result.carry_from, strict=True))
    columns: dict[str, Any] = {}
    for name, source in sources.items():
        take = carry_from.get(name, rows)
        columns[name] = _column_rows(source, name, take, np.asarray(take, dtype=np.intp), n)
    outcome: Any = list(result.status)
    if response.states:
        labels = ("censor", *response.states)
        outcome = _r_factor([labels[code] for code in outcome], labels)
    if result.counting:
        columns[_TSTART] = list(result.tstart)
    columns[_TSTOP] = list(result.tstop)
    columns[_STATUS] = outcome
    converted = {
        name: None if values is None else _subset_sequence(values, rows, name)
        for name, values in arguments.items()
    }
    if result.istate is not None:
        current = [response.states[code - 1] for code in result.istate]
        given = converted.get("istate")
        if given is not None and not _istate_agrees(given, result.istate, current):
            raise ValueError("istate argument does not agree with initial Surv2 values")
        converted["istate"] = _r_factor(current, response.states)
    response_columns = [_TSTART, _TSTOP, _STATUS] if result.counting else [_TSTOP, _STATUS]
    counting_formula = f"Surv({', '.join(f'`{name}`' for name in response_columns)}) ~{rhs}"
    return counting_formula, _FormulaRows(columns, len(rows)), converted


def _timeline_response(formula: Any) -> bool:
    """Whether *formula* is a formula string with a ``Surv2`` response."""

    if not isinstance(formula, str):
        return False
    spec = _response_spec(formula)
    return spec is not None and spec.timeline


def _timeline_model_frame(columns: dict[str, Any], formula: str) -> dict[str, Any]:
    """The model frame of a fit to the :func:`_timeline_counting` form of *formula*, as
    R's: the counting-process response (the first column) under the ``Surv2`` name,
    without the columns it was built from.  Other formulas' frames are returned as is."""

    if not _timeline_response(formula):
        return columns
    (_name, response), *rest = columns.items()
    reserved = {_TSTART, _TSTOP, _STATUS}
    return {
        _formula_response_spec(formula).name: response,
        **{name: values for name, values in rest if name not in reserved},
    }


def _istate_agrees(given: Any, codes: Sequence[int], current: Sequence[str]) -> bool:
    """Whether an ``istate`` argument names the current states of a timeline: as the
    state codes when it is numeric, else as the state names."""

    values = _materialize_labels(given, "istate")
    if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
        return all(value == code for value, code in zip(values, codes, strict=True))
    return all(_as_character(value) == state for value, state in zip(values, current, strict=True))


def _numeric_response(data: Any, spec: _SurvResponseSpec, n: int) -> list[float]:
    values = _response_arg_values(data, spec.arguments[0], n)
    try:
        return _floats_or_nan(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"formula response {spec.arguments[0]!r} must be numeric") from exc


def model_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None = None,
    na_action: str | None = _DEFAULT_NA_ACTION,
    weights: Any | None = None,
    offset: Any | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    istate: Any | None = None,
    extra: Mapping[str, Any] | None = None,
    timeline: bool = False,
) -> ModelFrame:
    """R's ``model.frame`` call every survival fitter starts with.

    The extra arguments may be column names of *data* or row-aligned vectors, as
    R evaluates ``weights = wt`` in the data; ``extra`` names further such
    columns (``pyears``' ``rmap`` variables and ``tcut``/``cut`` values), which a
    missing value removes like a formula variable.  ``subset`` (a mask or row indices)
    and then ``na_action`` (``"na.omit"``, R's default, ``"na.exclude"``,
    ``"na.pass"``, ``"na.fail"``, or ``None`` for none) are applied to the formula's
    variables and the arguments together, after which the response and the terms
    are evaluated.  A ``Surv2`` response is refused unless *timeline* says the caller
    takes one.
    """

    if not isinstance(formula, str):
        raise TypeError("a formula argument is required")
    if data is None:
        raise ValueError("a data argument is required")
    _lhs, sep, rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")
    action = _normalize_na_action(na_action)
    arguments: dict[str, Any] = {
        name: None if value is None else _column_or_values(data, value, name)
        for name, value in zip(
            _MODEL_FRAME_ARGUMENTS, (weights, offset, id, cluster, istate), strict=True
        )
    }
    extra_names = [] if extra is None else [str(name) for name in extra]
    if set(extra_names) & set(_MODEL_FRAME_ARGUMENTS):
        raise ValueError("extra columns must not be named like a model.frame argument")
    for name in extra_names:
        arguments[name] = _column_or_values(data, extra[name], name)
    if subset is not None:
        data, arguments = _subset_formula_inputs(formula, data, subset, **arguments)
    data, arguments, removed = _apply_formula_na_action(formula, data, action, **arguments)

    spec = _response_spec(formula)
    n = _data_row_count(data, formula)
    response: Surv | Surv2 | None = None
    y: list[float] | None = None
    if spec is not None and spec.timeline and timeline:
        response = _surv2_from_spec(data, spec)
        n = len(response)
    elif spec is not None and spec.surv:
        response = _surv_from_spec(data, spec)
        n = len(response)
    elif spec is not None:
        y = _numeric_response(data, spec, n)
        n = len(y)
    terms = _split_terms(rhs, _dot_terms(data, list(spec.columns) if spec else []))

    aligned: dict[str, list[Any] | None] = {}
    for name, values in arguments.items():
        if values is None:
            aligned[name] = None
            continue
        materialized = _materialize_labels(values, name)
        if len(materialized) == 1 and n != 1 and name in extra_names:
            materialized = materialized * n
        if len(materialized) != n:
            raise ValueError(f"{name} must have the same length as the response")
        aligned[name] = materialized
    formula_offset = _offset_vector(data, terms.offsets, n)
    offset_values = aligned["offset"]
    if offset_values is not None:
        offset_values = [float(value) for value in offset_values]
        if formula_offset is not None:
            offset_values = [a + b for a, b in zip(offset_values, formula_offset, strict=True)]
    else:
        offset_values = formula_offset
    return ModelFrame(
        formula=formula,
        data=data,
        n=n,
        spec=spec,
        response=response,
        y=y,
        terms=terms,
        weights=aligned["weights"],
        offset=offset_values,
        id=aligned["id"],
        cluster=aligned["cluster"],
        istate=aligned["istate"],
        na_action=_na_action_record(action, removed),
        extra={name: aligned[name] or [] for name in extra_names},
    )


def _strata_term(data: Any, columns: Sequence[str]) -> StrataFactor:
    """R's ``strata(a, b)`` formula term evaluated on *data* (R's default label rule)."""

    return _strata([(column, _column_source(data, column)) for column in columns])


def _strata_term_values(data: Any, columns: Sequence[str]) -> _RFactorVector:
    """The model-frame column of a ``strata(a, b)`` term: the factor ``strata()`` returns."""

    factor = _strata_term(data, columns)
    return _r_factor(factor.labels, factor.levels)


def _strata_keep(data: Any, terms: Sequence[Sequence[str]]) -> StrataFactor:
    """``strata.keep`` of coxph.R, survreg.R and survdiff.R for the ``strata()`` terms
    (each given by its columns): the one term's factor, else ``strata(m[, vars],
    shortlabel = TRUE)`` of the terms' factors."""

    if len(terms) == 1:
        return _strata_term(data, terms[0])
    return _strata(
        [(f"strata({', '.join(term)})", _strata_term_values(data, term)) for term in terms],
        shortlabel=True,
    )


def _strata_term_columns(terms: _FormulaTerms) -> tuple[tuple[str, ...], ...]:
    """The columns of each ``strata()`` term of *terms*, in formula order."""

    return tuple(term.columns for term in terms.model_terms if isinstance(term, _ModelStrataTerm))


def _model_variables(
    mf: ModelFrame, overrides: Mapping[str, Any] | None = None
) -> list[tuple[str, Any]]:
    """R's ``mf[-1]``: one evaluated column per formula term, in formula order.

    Interactions contribute their factors; ``strata()`` becomes the strata factor,
    ``offset()`` the numeric offset; ``cluster()`` terms are left out.
    ``overrides`` supplies already evaluated columns, such as population
    cut terms evaluated before subsetting and missing-value removal.
    """

    columns: list[tuple[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, values: Any) -> None:
        if name not in seen:
            seen.add(name)
            columns.append((name, values))

    terms = mf.terms
    model_terms: Sequence[_FormulaModelTerm]
    model_terms = terms.model_terms or [_ModelCovariateTerm(term) for term in terms.covariates]
    for model_term in model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            for factor in _covariate_factors(model_term.term):
                name = _covariate_term_name(factor)
                values = (
                    overrides[name]
                    if overrides is not None and name in overrides
                    else _term_values(mf.data, factor, mf.n)
                )
                add(name, values)
        elif isinstance(model_term, _ModelStrataTerm):
            name = f"strata({', '.join(model_term.columns)})"
            add(name, _strata_term_values(mf.data, model_term.columns))
        elif isinstance(model_term, _ModelOffsetTerm):
            term = model_term.term
            values = _numeric_term_values(_term_raw_values(mf.data, term, mf.n), term)
            add(f"offset({_covariate_term_name(term)})", values)
    return columns


def _model_strata(
    mf: ModelFrame, overrides: Mapping[str, Any] | None = None
) -> StrataFactor | None:
    """R's ``strata(mf[ll])`` over the term labels: the grouping factor, or ``None``.

    Used where a right-hand side only groups the observations (``rttright``,
    ``survexp``): every term variable, ``strata()`` included, is a component of
    the grouping factor, whose ``codes`` are zero based.
    """

    terms = mf.terms
    if any(isinstance(term, _InteractionTerm) for term in terms.covariates):
        raise ValueError("Interaction terms are not valid for this function")
    variables = [
        (name, values)
        for name, values in _model_variables(mf, overrides)
        if not name.startswith("offset(")
    ]
    if not variables:
        return None
    # strata(mf[ovars]) hands R a named list, so the labels are never shortened
    return _strata(variables, shortlabel=False)
