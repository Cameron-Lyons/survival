"""Formula tokenizer/parser, terms, design matrices, and model-frame builders."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import combinations, product
from typing import Any

from ._coerce import (
    _finite_float,
    _is_missing_value,
    _keep_rows_after_na_action,
    _label_levels,
    _materialize_1d,
    _materialize_labels,
    _missing_row_indices,
    _mstate_categories,
    _normalize_na_action,
    _strata_value_label,
    _subset_data,
    _subset_indices,
    _subset_optional_sequence,
)
from ._penalties import PENALTY_FUNCTIONS, fit_penalty, penalty_columns
from ._surv import (
    Surv,
    _formula_response_argument_name,
    _normalize_surv_type,
    _ordered_named_response_arguments,
)
from ._types import (
    _MISSING,
    ModelFrame,
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
) -> tuple[str, str, str] | None:
    """The right-most top-level binary operator of *operators* (unary signs skipped)."""

    for idx, char in reversed(_top_level(expression, quotes=False)):
        if char not in operators:
            continue
        previous = _previous_non_space(expression, idx)
        if previous is None or previous in "+-*/(^":
            continue
        left = expression[:idx].strip()
        right = expression[idx + 1 :].strip()
        if not left or not right:
            raise ValueError("formula arithmetic terms require both operands")
        return left, char, right
    return None


def _find_top_level_power_operator(expression: str) -> tuple[str, str, str] | None:
    """The left-most top-level ``^``."""

    for idx, char in _top_level(expression, quotes=False):
        if char != "^":
            continue
        left = expression[:idx].strip()
        right = expression[idx + 1 :].strip()
        if not left or not right:
            raise ValueError("formula arithmetic terms require both operands")
        return left, char, right
    return None


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
    return _column(data, operand.column), None


def _compare_response_values(left: Any, operator: str, right: Any) -> bool | None:
    if _is_missing_value(left) or _is_missing_value(right):
        return None
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right

    try:
        left_numeric = float(left)
        right_numeric = float(right)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordered formula response comparisons require numeric values") from exc

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


@lru_cache(maxsize=512)
def _formula_response_spec(formula: str) -> _SurvResponseSpec:
    lhs, sep, _rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")

    lhs = lhs.strip()
    if lhs.startswith("Surv(") and lhs.endswith(")"):
        response_inner = lhs[5:-1]
    elif lhs.startswith("survival::Surv(") and lhs.endswith(")"):
        response_inner = lhs[15:-1]
    else:
        raise ValueError("formula response must be Surv(...)")

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


@lru_cache(maxsize=512)
def _response_spec(formula: str) -> _SurvResponseSpec | None:
    """The left-hand side of any survival formula.

    ``Surv(...)`` responses go through :func:`_formula_response_spec`; an empty
    left-hand side (``~ sex``) gives ``None`` and a plain expression (``time ~ 1``,
    ``stop / 365.25 ~ surgery``) a numeric response spec with ``surv=False``.
    """

    lhs, sep, _rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")
    lhs = lhs.strip()
    if not lhs:
        return None
    if lhs.startswith(("Surv(", "survival::Surv(")) and lhs.endswith(")"):
        return _formula_response_spec(formula)
    return _SurvResponseSpec(
        arguments=(lhs,),
        columns=tuple(_response_arg_columns(lhs)),
        type=None,
        origin=0.0,
        surv=False,
    )


def _data_row_count(data: Any, formula: str | None = None) -> int:
    """The number of rows of *data*: the first response column, else the first column."""

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


def _covariate_columns(terms: list[_CovariateSpec]) -> list[str]:
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
    try:
        literal = float(value)
    except ValueError:
        return None
    if not math.isfinite(literal):
        raise ValueError("formula arithmetic literals must be finite")
    return literal


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

    column, quoted = _formula_name(expression)
    if _unsupported_formula_name(column, quoted):
        raise ValueError(f"unsupported formula arithmetic term: {expression}")
    return [column]


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
        if any(value == 0.0 for value in right_values):
            raise ValueError("formula arithmetic division by zero")
        return [left / right for left, right in zip(left_values, right_values, strict=True)]

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
        powered: list[float] = []
        for left_value, right_value in zip(left_values, right_values, strict=True):
            try:
                value = math.pow(left_value, right_value)
            except ValueError as exc:
                raise ValueError("formula arithmetic power produced a non-real value") from exc
            if not math.isfinite(value):
                raise ValueError("formula arithmetic power produced a non-finite value")
            powered.append(value)
        return powered

    literal = _arithmetic_literal(expression)
    if literal is not None:
        return [literal] * n

    column, quoted = _formula_name(expression)
    if _unsupported_formula_name(column, quoted):
        raise ValueError(f"unsupported formula arithmetic term: {expression}")
    values = _column(data, column)
    if len(values) != n:
        raise ValueError("formula columns must have the same length as the Surv response")
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"I() formula term {expression!r} requires numeric values") from exc


def _covariate_term_columns(term: _CovariateTerm) -> list[str]:
    if term.call is not None and term.call.split("(", 1)[0] in PENALTY_FUNCTIONS:
        return _penalty_arguments(term.call)[0]
    if term.arithmetic is not None:
        return _arithmetic_expression_columns(term.arithmetic)
    return [term.column]


def _formula_columns(formula: str, data: Any) -> list[str]:
    spec = _response_spec(formula)
    args = [] if spec is None else list(spec.columns)
    _lhs, _sep, rhs = formula.partition("~")
    terms = _split_terms(rhs, _dot_terms(data, args))
    columns = (
        args
        + _covariate_columns(terms.covariates)
        + terms.strata
        + _offset_columns(terms.offsets)
        + terms.clusters
    )
    return list(dict.fromkeys(columns))


def _subset_formula_inputs(
    formula: str,
    data: Any,
    subset: Any,
    **row_aligned: Any,
) -> tuple[Any, dict[str, Any]]:
    indices = _subset_indices(subset, _data_row_count(data, formula))
    filtered = {
        name: _subset_optional_sequence(values, indices, name)
        for name, values in row_aligned.items()
    }
    return _subset_data(data, indices), filtered


def _apply_formula_na_action(
    formula: str,
    data: Any,
    na_action: str | None,
    *,
    exclude_columns: Sequence[str] = (),
    **row_aligned: Any,
) -> tuple[Any, dict[str, Any]]:
    action = _normalize_na_action(na_action)
    if action == "pass":
        return data, row_aligned

    excluded = set(exclude_columns)
    columns = [column for column in _formula_columns(formula, data) if column not in excluded]
    n = _data_row_count(data, formula)
    missing = _missing_row_indices(
        [
            *[(column, _column(data, column)) for column in columns],
            *((name, values) for name, values in row_aligned.items() if values is not None),
        ],
        n,
    )
    keep = _keep_rows_after_na_action(missing, n, action, "formula data")
    if keep is None:
        return data, row_aligned
    filtered = {
        name: _subset_optional_sequence(values, keep, name) for name, values in row_aligned.items()
    }
    return _subset_data(data, keep), filtered


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
    return not quoted and any(token in name for token in "():*/+-^%")


def _factor_column_items(term: str) -> tuple[str, list[tuple[str, bool]]] | None:
    for wrapper in ("factor", "as.factor"):
        prefix = f"{wrapper}("
        if term.startswith(prefix) and term.endswith(")"):
            return wrapper, _formula_name_items(term[len(prefix) : -1])
    return None


def _transform_column_items(term: str) -> tuple[str, list[tuple[str, bool]]] | None:
    for wrapper in ("log", "sqrt", "exp", "I", "identity", "as.numeric", "tt"):
        prefix = f"{wrapper}("
        if term.startswith(prefix) and term.endswith(")"):
            return wrapper, _formula_name_items(term[len(prefix) : -1])
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

    for wrapper in ("I", "identity"):
        prefix = f"{wrapper}("
        if term.startswith(prefix) and term.endswith(")"):
            expression = term[len(prefix) : -1].strip()
            if _is_formula_arithmetic_expression(expression):
                _arithmetic_expression_columns(expression)
                return _CovariateTerm(expression, transform=wrapper, arithmetic=expression)

    transform_items = _transform_column_items(term)
    if transform_items is not None:
        transform, column_items = transform_items
        columns = [column for column, _quoted in column_items]
        if len(columns) != 1:
            raise ValueError(f"{transform}() requires exactly one column")
        if any(_unsupported_formula_name(column, quoted) for column, quoted in column_items):
            raise ValueError(f"unsupported formula term(s): {columns[0]}")
        return _CovariateTerm(columns[0], transform=transform)

    call_term = _parse_call_term(term)
    if call_term is not None:
        return call_term

    term_name, quoted = _formula_name(term)
    if _unsupported_formula_name(term_name, quoted):
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
        elif value.startswith("c(") and value.endswith(")"):
            options[name] = [
                _parse_formula_literal(v) for v in _formula_response_parts(value[2:-1])
            ]
        else:
            options[name] = _parse_formula_literal(value)
    if not columns:
        raise ValueError("penalty terms require a data column")
    return columns, options


def _parse_call_term(term: str) -> _CovariateTerm | None:
    """A ``tcut(x, ...)``/``cut(x, ...)`` term: categorical, reading the column of ``x``."""

    for function in _CALL_TERMS:
        prefix = f"{function}("
        if not (term.startswith(prefix) and term.endswith(")")):
            continue
        arguments = _formula_response_parts(term[len(prefix) : -1])
        if not arguments:
            raise ValueError(f"{function}() requires a variable")
        first = arguments[0]
        if _is_formula_arithmetic_expression(first):
            columns = _arithmetic_expression_columns(first)
            if not columns:
                raise ValueError(f"{function}() requires a data column")
            return _CovariateTerm(columns[0], categorical=True, arithmetic=first, call=term)
        column, quoted = _formula_name(first)
        if _unsupported_formula_name(column, quoted):
            raise ValueError(f"unsupported formula term(s): {term}")
        return _CovariateTerm(column, categorical=True, call=term)
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
        parsed_groups = [_parse_interaction_term(part, dot_terms) for part in in_parts]
        nested_expanded = parsed_groups[-1]
        for nested_group in reversed(parsed_groups[:-1]):
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


def _apply_numeric_transform(values: list[float], transform: str | None, term: str) -> list[float]:
    if transform is None:
        return values
    if transform == "log":
        if any(value <= 0.0 for value in values):
            raise ValueError(f"log() formula term {term!r} requires positive values")
        return [math.log(value) for value in values]
    if transform == "sqrt":
        if any(value < 0.0 for value in values):
            raise ValueError(f"sqrt() formula term {term!r} requires nonnegative values")
        return [math.sqrt(value) for value in values]
    if transform == "exp":
        return [math.exp(value) for value in values]
    if transform in {"I", "identity", "as.numeric", "tt"}:
        return values
    raise ValueError(f"unsupported formula transform {transform!r}")


def _numeric_term_values(values: list[Any], term: _CovariateTerm) -> list[float]:
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        if term.transform is not None:
            raise ValueError(
                f"{term.transform}() formula term {term.column!r} requires numeric values"
            ) from exc
        raise
    return _apply_numeric_transform(numeric, term.transform, term.column)


def _term_raw_values(data: Any, term: _CovariateTerm, n: int) -> list[Any]:
    if term.call is not None:
        raise ValueError(f"unsupported formula term(s): {term.call}")
    if term.arithmetic is not None:
        return _arithmetic_expression_values(data, term.arithmetic, n)
    values = _column(data, term.column)
    if term.transform == "as.numeric":
        categories = _mstate_categories(_column_source(data, term.column))
        if categories is not None:
            codes = {value: i + 1 for i, value in enumerate(categories)}
            values = [math.nan if _is_missing_value(value) else codes[value] for value in values]
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
    if term.transform is None:
        return values
    return _numeric_term_values(values, term)


def _term_columns(
    data: Any,
    term: _CovariateSpec,
    n: int,
) -> list[list[float]]:
    if isinstance(term, _InteractionTerm):
        factor_columns = [_term_columns(data, factor, n) for factor in term.factors]
        interaction_columns: list[list[float]] = []
        for column_combo in product(*factor_columns):
            interaction_columns.append(
                [math.prod(column[idx] for column in column_combo) for idx in range(n)]
            )
        return interaction_columns

    values = _term_raw_values(data, term, n)
    if not term.categorical:
        if term.transform is not None:
            return [_numeric_term_values(values, term)]
        try:
            numeric = _numeric_term_values(values, term)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            return [numeric]

    levels = _categorical_levels(values, term.column)
    return [[1.0 if value == level else 0.0 for value in values] for level in levels[1:]]


def _categorical_levels(values: list[Any], column: str) -> tuple[Any, ...]:
    labels: dict[Any, None] = {}
    for value in values:
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
) -> _SingleDesignTerm:
    if term.call is not None and term.call.split("(", 1)[0] in PENALTY_FUNCTIONS:
        columns, options = _penalty_arguments(term.call)
        values = {column: _column(data, column) for column in columns}
        if any(len(value) != n for value in values.values()):
            raise ValueError("formula columns must have the same length as the Surv response")
        levels = _mstate_categories(_column_source(data, columns[0]))
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
    factor_order: Mapping[_CovariateTerm, int] | None = None,
) -> _DesignTerm:
    if isinstance(term, _InteractionTerm):
        factors = term.factors
        if factor_order is not None:
            factors = tuple(sorted(factors, key=factor_order.__getitem__))
        fitted = tuple(_fit_single_design_term(data, factor, n) for factor in factors)
        if any(isinstance(factor, _PenaltyDesignTerm) for factor in fitted):
            raise ValueError("penalty terms cannot appear in interactions")
        return _InteractionDesignTerm(fitted)
    return _fit_single_design_term(data, term, n)


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
) -> _FormulaDesign:
    strata_values = _combined_columns(data, terms.strata, n) if terms.strata else []
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
        fitted_term = _fit_design_term(data, term, n, factor_order)
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
        strata_levels=_label_levels(strata_values, "strata") if terms.strata else (),
        intercept=include_intercept and terms.intercept,
    )


def _single_design_columns(
    data: Any,
    spec: _SingleDesignTerm,
    n: int,
    time_transform_values: Mapping[_CovariateTerm, Sequence[float]] | None = None,
) -> list[list[float]]:
    if isinstance(spec, _PenaltyDesignTerm):
        return penalty_columns(spec, {column: _column(data, column) for column in spec.columns})
    if (
        isinstance(spec, _NumericDesignTerm)
        and spec.term.transform == "tt"
        and time_transform_values is not None
    ):
        values = list(time_transform_values[spec.term])
        if len(values) != n:
            raise ValueError("tt transform result must match the expanded risk-set rows")
        return [values]
    values = _term_raw_values(data, spec.term, n)
    if isinstance(spec, _NumericDesignTerm):
        return [_numeric_term_values(values, spec.term)]

    levels = spec.levels
    for value in values:
        if all(value != level for level in levels):
            raise ValueError(
                f"newdata column {spec.term.column!r} contains unknown level {value!r}"
            )
    encoded_levels = levels if spec.full else levels[1:]
    return [[1.0 if value == level else 0.0 for value in values] for level in encoded_levels]


def _design_term_columns(
    data: Any,
    spec: _DesignTerm,
    n: int,
    time_transform_values: Mapping[_CovariateTerm, Sequence[float]] | None = None,
) -> list[list[float]]:
    if isinstance(spec, _InteractionDesignTerm):
        factor_columns = [
            _single_design_columns(data, factor, n, time_transform_values)
            for factor in spec.factors
        ]
        interaction_columns: list[list[float]] = []
        for reversed_combo in product(*reversed(factor_columns)):
            column_combo = tuple(reversed(reversed_combo))
            interaction_columns.append(
                [math.prod(column[idx] for column in column_combo) for idx in range(n)]
            )
        return interaction_columns
    return _single_design_columns(data, spec, n, time_transform_values)


def _design_rows_from_spec(
    data: Any,
    design: _FormulaDesign,
    n: int,
    *,
    time_transform_values: Mapping[_CovariateTerm, Sequence[float]] | None = None,
) -> list[list[float]]:
    columns = [
        column
        for term in design.covariates
        for column in _design_term_columns(data, term, n, time_transform_values)
    ]
    if design.intercept:
        columns.insert(0, [1.0] * n)
    return [[column[i] for column in columns] for i in range(n)]


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


def _surv_response_model_name(spec: _SurvResponseSpec) -> str:
    return f"Surv({', '.join(spec.arguments)})"


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
    frame: dict[str, Any] = {_surv_response_model_name(design.response): response}
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


def _cox_survfit_model_frame(fit: Any, newdata: Any | None) -> dict[str, Any]:
    frame: dict[str, Any] = {"fit": fit}
    model = getattr(fit, "model", None)
    if model is not None:
        frame["model"] = model
    if newdata is not None:
        frame["newdata"] = newdata
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


def _offset_vector(data: Any, terms: Sequence[_CovariateTerm], n: int) -> list[float] | None:
    if not terms:
        return None
    columns = [_numeric_term_values(_term_raw_values(data, term, n), term) for term in terms]
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

    args = _formula_response_values(data, spec)
    if len(args) not in {1, 2, 3}:
        raise ValueError("Surv(...) formula response must have 1, 2, or 3 column arguments")
    return Surv(*args, type=spec.type, origin=spec.origin)


def _numeric_response(data: Any, spec: _SurvResponseSpec, n: int) -> list[float]:
    values = _response_arg_values(data, spec.arguments[0], n)
    try:
        return [math.nan if _is_missing_value(value) else float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"formula response {spec.arguments[0]!r} must be numeric") from exc


def model_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None = None,
    na_action: str | None = None,
    weights: Any | None = None,
    offset: Any | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    istate: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> ModelFrame:
    """R's ``model.frame`` call every survival fitter starts with.

    The extra arguments may be column names of *data* or row-aligned vectors, as
    R evaluates ``weights = wt`` in the data; ``extra`` names further such
    columns (``pyears``' ``rmap`` variables).  ``subset`` (a mask or row indices)
    and then ``na_action`` (``"na.pass"``, ``"na.omit"``, ``"na.fail"``; R's
    ``model.frame`` default is ``na.omit``, each caller passes its own default) are
    applied to the data and the arguments together, after which the response and
    the terms are evaluated.
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
    data, arguments = _apply_formula_na_action(formula, data, action, **arguments)

    spec = _response_spec(formula)
    n = _data_row_count(data, formula)
    response: Surv | None = None
    y: list[float] | None = None
    if spec is not None and spec.surv:
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
        na_action=action,
        extra={name: aligned[name] or [] for name in extra_names},
    )


def _strata_term_values(mf: ModelFrame, columns: Sequence[str]) -> list[Any]:
    """R's ``strata(a, b)`` model-frame column: the stratum label of each row."""

    from ._surv import strata

    factor = strata(*[_column(mf.data, column) for column in columns], labels=list(columns))
    return list(factor.labels)


def _model_variables(mf: ModelFrame) -> list[tuple[str, list[Any]]]:
    """R's ``mf[-1]``: one evaluated column per formula term, in formula order.

    Interactions contribute their factors; ``strata()`` becomes the strata label,
    ``offset()`` the numeric offset; ``cluster()`` terms are left out.
    """

    columns: list[tuple[str, list[Any]]] = []
    seen: set[str] = set()

    def add(name: str, values: list[Any]) -> None:
        if name not in seen:
            seen.add(name)
            columns.append((name, values))

    terms = mf.terms
    model_terms: Sequence[_FormulaModelTerm]
    model_terms = terms.model_terms or [_ModelCovariateTerm(term) for term in terms.covariates]
    for model_term in model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            for factor in _covariate_factors(model_term.term):
                add(_covariate_term_name(factor), _term_values(mf.data, factor, mf.n))
        elif isinstance(model_term, _ModelStrataTerm):
            name = f"strata({', '.join(model_term.columns)})"
            add(name, _strata_term_values(mf, model_term.columns))
        elif isinstance(model_term, _ModelOffsetTerm):
            term = model_term.term
            values = _numeric_term_values(_term_raw_values(mf.data, term, mf.n), term)
            add(f"offset({_covariate_term_name(term)})", values)
    return columns


def _model_strata(mf: ModelFrame) -> StrataFactor | None:
    """R's ``strata(mf[ll])`` over the term labels: the grouping factor, or ``None``.

    Used where a right-hand side only groups the observations (``rttright``,
    ``survexp``): every term variable, ``strata()`` included, is a component of
    the grouping factor, whose ``codes`` are zero based.
    """

    from ._surv import strata

    terms = mf.terms
    if any(isinstance(term, _InteractionTerm) for term in terms.covariates):
        raise ValueError("Interaction terms are not valid for this function")
    variables = [
        (name, values) for name, values in _model_variables(mf) if not name.startswith("offset(")
    ]
    if not variables:
        return None
    # strata(mf[ovars]) hands R a named list, so the labels are never shortened
    return strata(
        *[values for _name, values in variables],
        labels=[n for n, _v in variables],
        shortlabel=False,
    )
