"""Restricted, typed R expressions shared by formula discovery and evaluation.

Formula operators outside calls remain terms algebra. This parser handles the
vector expressions inside calls, response arguments, and logical formula terms;
it never evaluates Python or accepts arbitrary function calls.
"""

from __future__ import annotations

import math
import numbers
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from operator import add, mul, sub, truediv
from typing import Any

from ._coerce import (
    _NA_REAL,
    _floats_or_nan,
    _is_bool_like,
    _is_missing_value,
    _materialize_1d,
    _mstate_categories,
    _numeric_ndarray,
    _strata_level_sort_key,
    _warn_outside_package,
)


class _ExpressionVector(list[Any]):
    """Values with R's declared type, including when no observed value remains."""

    def __init__(
        self,
        values: Iterable[Any],
        kind: str,
        categories: Iterable[Any] | None = None,
        *,
        ordered: bool = False,
        declared: bool = True,
        storage: str | None = None,
    ) -> None:
        super().__init__(values)
        self.kind = kind
        self.categories = None if categories is None else tuple(categories)
        self.ordered = ordered
        self.declared = declared
        self.storage = (storage or "double") if kind == "numeric" else None

    def take(self, rows: Iterable[int]) -> _ExpressionVector:
        missing = None if self.kind in {"logical", "character", "factor"} else _NA_REAL
        return _ExpressionVector(
            (missing if row < 0 else self[row] for row in rows),
            self.kind,
            self.categories,
            ordered=self.ordered,
            declared=self.declared,
            storage=getattr(self, "storage", None),
        )


def _expression_source(source: Any, name: str) -> _ExpressionVector:
    """A column's declared type takes precedence over inference from its values."""
    if isinstance(source, _ExpressionVector):
        return source
    values = _materialize_1d(source, name)
    levels = _mstate_categories(source)
    dtype = getattr(source, "dtype", None)
    kind = getattr(source, "kind", None)
    declared = (
        levels is not None
        or kind in {"logical", "numeric", "character"}
        or (
            getattr(dtype, "kind", None) in {"b", "i", "u", "f", "U", "S"}
            or str(dtype) == "boolean"
        )
    )
    if levels is not None:
        kind = "factor"
    elif kind not in {"logical", "numeric", "character"}:
        if getattr(dtype, "kind", None) == "b" or str(dtype) == "boolean":
            kind = "logical"
        elif getattr(dtype, "kind", None) in {"i", "u", "f"}:
            kind = "numeric"
        elif any(_is_bool_like(value) for value in values) and all(
            _is_bool_like(value) or _is_missing_value(value) for value in values
        ):
            kind = "logical"
        elif getattr(dtype, "kind", None) in {"U", "S"} or any(
            isinstance(value, str) for value in values
        ):
            kind = "character"
        else:
            kind = "numeric"
    storage = "double"
    numeric_array = (
        _numeric_ndarray(source)
        if kind == "numeric" and getattr(dtype, "kind", None) in {"i", "u"}
        else None
    )
    if numeric_array is not None:
        if not numeric_array.size or (
            numeric_array.min().item() > -(2**31) and numeric_array.max().item() < 2**31
        ):
            storage = "integer"
    elif kind == "numeric" and getattr(dtype, "kind", None) != "f":
        integer = getattr(dtype, "kind", None) in {"i", "u"}
        for value in values:
            if _is_missing_value(value):
                continue
            if (
                not isinstance(value, numbers.Integral)
                or _is_bool_like(value)
                or not (-(2**31) < int(value) < 2**31)
            ):
                integer = False
                break
            integer = True
        if integer:
            storage = "integer"
    return _ExpressionVector(
        values,
        kind,
        levels,
        ordered=bool(getattr(source, "ordered", getattr(dtype, "ordered", False))),
        declared=declared,
        storage=storage,
    )


@dataclass(frozen=True)
class _ExpressionNode:
    kind: str
    value: Any
    children: tuple[_ExpressionNode, ...] = ()
    columns: tuple[str, ...] = ()
    declared_kind: str | None = None
    declared_storage: str | None = None
    expression: str = ""


_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?L?")
_HEX = "0123456789abcdefABCDEF"
_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "`": "`",
    "\n": "\n",
}
_CALLS = {"I", "identity", "as.numeric", "log", "sqrt", "exp", "factor", "as.factor"}
_BINDING = {
    "|": 10,
    "&": 20,
    "==": 40,
    "!=": 40,
    "<": 40,
    "<=": 40,
    ">": 40,
    ">=": 40,
    "+": 50,
    "-": 50,
    "*": 60,
    "/": 60,
    "^": 80,
}


class _ExpressionParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.token: tuple[str, Any] = ("", "")
        self.advance()

    def error(self) -> ValueError:
        return ValueError(f"unsupported formula expression: {self.text}")

    def unicode_code(self, escape: str) -> int:
        braces = self.pos < len(self.text) and self.text[self.pos] == "{"
        self.pos += int(braces)
        digits = ""
        limit = 4 if escape == "u" else 8
        while len(digits) < limit and self.pos < len(self.text) and self.text[self.pos] in _HEX:
            digits += self.text[self.pos]
            self.pos += 1
        if braces:
            if self.pos == len(self.text) or self.text[self.pos] != "}":
                raise self.error()
            self.pos += 1
        if not digits or not 0 < (code := int(digits, 16)) <= 0x10FFFF:
            raise self.error()
        return code

    def quoted(self, quote: str) -> str:
        """Read R escapes, retaining UTF-8 text and refusing unsupported byte strings."""
        self.pos += 1
        contents = bytearray()
        byte_escape = unicode_escape = False
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            value = self.text[self.pos]
            self.pos += 1
            if value == "\\":
                if self.pos == len(self.text):
                    raise self.error()
                escape = self.text[self.pos]
                self.pos += 1
                if escape in _ESCAPES:
                    value = _ESCAPES[escape]
                elif escape in "01234567x":
                    byte_escape = True
                    digits = "" if escape == "x" else escape
                    limit, alphabet, base = (2, _HEX, 16) if escape == "x" else (3, "01234567", 8)
                    while (
                        len(digits) < limit
                        and self.pos < len(self.text)
                        and self.text[self.pos] in alphabet
                    ):
                        digits += self.text[self.pos]
                        self.pos += 1
                    if not digits or not 0 < (code := int(digits, base)) <= 255:
                        raise self.error()
                    contents.append(code)
                    value = ""
                elif escape in {"u", "U"}:
                    if quote == "`":
                        raise self.error()
                    unicode_escape = True
                    code = self.unicode_code(escape)
                    if 0xD800 <= code <= 0xDBFF:
                        if self.text[self.pos : self.pos + 2] not in {"\\u", "\\U"}:
                            raise self.error()
                        escape = self.text[self.pos + 1]
                        self.pos += 2
                        low = self.unicode_code(escape)
                        if not 0xDC00 <= low <= 0xDFFF:
                            raise self.error()
                        code = 0x10000 + ((code - 0xD800) << 10) + low - 0xDC00
                    elif 0xDC00 <= code <= 0xDFFF:
                        raise self.error()
                    value = chr(code)
                else:
                    raise self.error()
            if byte_escape and unicode_escape or "\0" in value:
                raise self.error()
            contents.extend(value.encode("utf-8"))
        if self.pos == len(self.text):
            raise self.error()
        self.pos += 1
        try:
            return contents.decode("utf-8")
        except UnicodeDecodeError:
            raise self.error() from None

    def advance(self) -> None:
        text = self.text
        while self.pos < len(text) and text[self.pos].isspace():
            self.pos += 1
        self.begin = self.pos
        if self.pos == len(text):
            self.token = ("end", "")
            return
        char = text[self.pos]
        if char in {"`", "'", '"'}:
            value = self.quoted(char)
            self.token = ("name", value) if char == "`" else ("literal", (value, "character"))
            return
        number = _NUMBER.match(text, self.pos)
        if number is not None:
            self.pos = number.end()
            number_value = float(number.group().removesuffix("L"))
            integer = (
                number.group().endswith("L")
                and number_value.is_integer()
                and -(2**31) < number_value < 2**31
            )
            self.token = (
                "literal",
                (
                    int(number_value) if integer else number_value,
                    "numeric",
                    "integer" if integer else "double",
                ),
            )
            return
        if char.isalpha() or char in "._":
            start = self.pos
            self.pos += 1
            while self.pos < len(text) and (text[self.pos].isalnum() or text[self.pos] in "._"):
                self.pos += 1
            value = text[start : self.pos]
            literal = {
                "TRUE": (True, "logical"),
                "FALSE": (False, "logical"),
                "Inf": (math.inf, "numeric"),
                "NA": (None, "logical"),
                "NA_real_": (_NA_REAL, "numeric"),
                "NA_integer_": (_NA_REAL, "numeric", "integer"),
                "NA_character_": (None, "character"),
                "NaN": (math.nan, "numeric"),
            }
            self.token = ("literal", literal[value]) if value in literal else ("name", value)
            return
        for operator in (
            "==",
            "!=",
            "<=",
            ">=",
            "|",
            "&",
            "!",
            "+",
            "-",
            "*",
            "/",
            "^",
            "<",
            ">",
            "(",
            ")",
        ):
            if text.startswith(operator, self.pos):
                self.pos += len(operator)
                self.token = (operator, operator)
                return
        raise self.error()

    def parse(self, minimum: int = 0) -> _ExpressionNode:
        begin = self.begin
        kind, value = self.token
        self.advance()
        if kind == "literal":
            left = _ExpressionNode(
                "literal",
                value[0],
                declared_kind=value[1],
                declared_storage=value[2] if len(value) > 2 else None,
            )
        elif kind == "name":
            if not value:
                raise self.error()
            if self.token[0] == "(":
                if value not in _CALLS:
                    raise self.error()
                self.advance()
                child = self.parse()
                if self.token[0] != ")":
                    raise self.error()
                self.advance()
                left = _ExpressionNode("call", value, (child,), child.columns)
            else:
                left = _ExpressionNode("column", value, columns=(value,))
        elif kind == "(":
            left = self.parse()
            if self.token[0] != ")":
                raise self.error()
            self.advance()
        elif kind in {"+", "-", "!"}:
            child = self.parse(30 if kind == "!" else 70)
            left = _ExpressionNode("unary", kind, (child,), child.columns)
        else:
            raise self.error()
        comparison_seen = False
        while (binding := _BINDING.get(self.token[0], -1)) >= minimum:
            operator = self.token[0]
            if binding == 40:
                if comparison_seen:
                    raise self.error()
                comparison_seen = True
            self.advance()
            right = self.parse(binding if operator == "^" else binding + 1)
            columns = tuple(dict.fromkeys((*left.columns, *right.columns)))
            left = _ExpressionNode("binary", operator, (left, right), columns)
        return replace(left, expression=self.text[begin : self.begin].strip())


@lru_cache(maxsize=1024)
def _parse_expression(expression: str) -> _ExpressionNode:
    parser = _ExpressionParser(expression.strip())
    tree = parser.parse()
    if parser.token[0] != "end":
        raise parser.error()
    return tree


@lru_cache(maxsize=1024)
def _expression_label(expression: str, *, keep_integer: bool = False) -> str:
    """R deparse uses double quotes for strings, independently of their syntax."""
    if "'" not in expression and "\\" not in expression and "L" not in expression:
        return expression
    parser = _ExpressionParser(expression)
    pieces: list[str] = []
    start = 0
    while parser.token[0] != "end":
        end = parser.pos
        if parser.token[0] == "literal" and parser.token[1][1] == "character":
            begin = parser.begin
            pieces.extend((expression[start:begin], _quoted_label(parser.token[1][0])))
            start = end
        elif (
            parser.token[0] == "literal"
            and parser.token[1][1] == "numeric"
            and expression[parser.begin : end].endswith("L")
        ):
            begin = parser.begin
            literal = parser.token[1][0]
            spelling = (
                str(literal) + ("L" if keep_integer else "")
                if isinstance(literal, int)
                else "Inf"
                if math.isinf(literal)
                else str(literal).removesuffix(".0")
            )
            pieces.extend((expression[start:begin], spelling))
            start = end
        elif parser.token[0] == "name" and expression[parser.begin] == "`":
            begin = parser.begin
            name = parser.token[1]
            syntactic = (
                name.replace(".", "_").isidentifier()
                and not (name.startswith(".") and len(name) > 1 and name[1].isdigit())
                and name
                not in {
                    "TRUE",
                    "FALSE",
                    "NA",
                    "NA_real_",
                    "NA_integer_",
                    "NA_character_",
                    "NaN",
                    "Inf",
                    "if",
                    "else",
                    "repeat",
                    "while",
                    "function",
                    "for",
                    "in",
                    "next",
                    "break",
                }
            )
            spelling = (
                name if syntactic else "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"
            )
            pieces.extend((expression[start:begin], spelling))
            start = end
        parser.advance()
        if start < end:
            pieces.append(expression[start:end])
            start = end
    pieces.append(expression[start:])
    return "".join(pieces)


def _quoted_label(value: str) -> str:
    escapes = {value: "\\" + key for key, value in _ESCAPES.items() if key in 'abfnrtv\\"'}
    pieces = []
    for char in value:
        code = ord(char)
        if char in escapes:
            pieces.append(escapes[char])
        elif code < 32 or code == 127:
            pieces.append(f"\\{code:03o}")
        elif 128 <= code < 160 or unicodedata.category(char) in {"Cn", "Zl", "Zp"}:
            pieces.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
        else:
            pieces.append(char)
    return '"' + "".join(pieces) + '"'


def _logical_value(value: Any) -> bool | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str):
        raise ValueError("formula logical operators require numeric or logical values")
    return bool(value)


def _logical_binary(left: Any, operator: str, right: Any) -> bool | None:
    a, b = _logical_value(left), _logical_value(right)
    if operator == "&":
        return False if a is False or b is False else None if a is None or b is None else True
    return True if a is True or b is True else None if a is None or b is None else False


def _numeric_values(values: _ExpressionVector, *, cast: bool = False) -> list[float]:
    if values.kind == "factor":
        if not cast:
            raise ValueError("formula arithmetic requires numeric or logical values")
        codes = {value: i + 1 for i, value in enumerate(values.categories or ())}
        return [_NA_REAL if _is_missing_value(value) else float(codes[value]) for value in values]
    if values.kind == "character":
        if not cast:
            raise ValueError("non-numeric argument to binary operator")
        result = []
        invalid = False
        for value in values:
            if _is_missing_value(value):
                result.append(_NA_REAL)
                continue
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(_NA_REAL)
                invalid = True
        if invalid:
            _warn_outside_package("NAs introduced by coercion")
        return result
    return _floats_or_nan(values)


def _integer_operand(values: _ExpressionVector) -> bool:
    return values.kind == "logical" or (values.kind == "numeric" and values.storage == "integer")


def _integer_result(values: Iterable[float]) -> _ExpressionVector:
    result: list[Any] = []
    overflow = False
    for value in values:
        if -(2**31) < value < 2**31:
            result.append(int(value))
        else:
            result.append(_NA_REAL)
            overflow |= not math.isnan(value)
    if overflow:
        _warn_outside_package("NAs produced by integer overflow")
    return _ExpressionVector(result, "numeric", storage="integer")


def _evaluate_expression(
    tree: _ExpressionNode,
    n: int,
    source: Callable[[str], Any],
    transform: Callable[[list[float], str, str], list[float]],
    compare: Callable[[Any, str, Any], bool | None],
    divide: Callable[[float, float], float],
    power: Callable[[float, float], float],
) -> _ExpressionVector:
    """Evaluate one cached tree over aligned vectors using R's vector precedence."""

    def evaluate(node: _ExpressionNode) -> _ExpressionVector:
        if n == 0 and not node.columns:
            # R computes constant subexpressions before recycling to an empty
            # vector, so their warnings survive an empty model frame.
            return _evaluate_expression(node, 1, source, transform, compare, divide, power).take(())
        operator = node.value
        if node.kind == "column":
            values = _expression_source(source(operator), operator)
            if len(values) != n:
                raise ValueError("formula columns must have the same length as the Surv response")
            return values
        if node.kind == "literal":
            return _ExpressionVector(
                [operator] * n,
                node.declared_kind or "numeric",
                storage=node.declared_storage,
            )
        left = evaluate(node.children[0])
        if node.kind == "call":
            if operator in {"I", "identity"}:
                return left
            if operator in {"factor", "as.factor"}:
                if operator == "as.factor" and left.kind == "factor":
                    return left
                present = {value for value in left if not _is_missing_value(value)}
                levels = (
                    [value for value in left.categories if value in present]
                    if left.categories is not None
                    else sorted(present, key=_strata_level_sort_key)
                )
                return _ExpressionVector(left, "factor", levels)
            numeric = _numeric_values(left, cast=operator == "as.numeric")
            label = node.children[0].expression or str(node.children[0].value)
            return _ExpressionVector(transform(numeric, operator, label), "numeric")
        if node.kind == "unary":
            if operator == "!":
                if left.kind == "factor":
                    _warn_outside_package("'!' not meaningful for factors")
                    return _ExpressionVector([None] * n, "logical")
                return _ExpressionVector(
                    (
                        None if (value := _logical_value(item)) is None else not value
                        for item in left
                    ),
                    "logical",
                )
            numeric = _numeric_values(left)
            unary_values = (
                numeric if operator == "+" else (x if math.isnan(x) else -x for x in numeric)
            )
            return (
                _integer_result(unary_values)
                if _integer_operand(left)
                else _ExpressionVector(unary_values, "numeric")
            )
        right = evaluate(node.children[1])
        pairs = zip(left, right, strict=True)
        if operator in {"&", "|"}:
            if "factor" in {left.kind, right.kind}:
                _warn_outside_package(f"'{operator}' not meaningful for factors")
                return _ExpressionVector([None] * n, "logical")
            return _ExpressionVector((_logical_binary(a, operator, b) for a, b in pairs), "logical")
        if operator in {"==", "!=", "<", "<=", ">", ">="}:
            from ._coerce import _as_character

            def labels(values: _ExpressionVector) -> Iterable[Any]:
                if values.kind != "factor":
                    return values
                return (None if _is_missing_value(v) else _as_character(v) for v in values)

            return _ExpressionVector(
                (compare(a, operator, b) for a, b in zip(labels(left), labels(right), strict=True)),
                "logical",
            )
        a_values, b_values = _numeric_values(left), _numeric_values(right)
        if operator in {"+", "-", "*"}:
            operation = {"+": add, "-": sub, "*": mul}[operator]
            arithmetic_values = map(operation, a_values, b_values)
            return (
                _integer_result(arithmetic_values)
                if _integer_operand(left) and _integer_operand(right)
                else _ExpressionVector(arithmetic_values, "numeric")
            )
        if operator == "/":
            try:
                return _ExpressionVector(map(truediv, a_values, b_values), "numeric")
            except ZeroDivisionError:
                return _ExpressionVector(map(divide, a_values, b_values), "numeric")
        try:
            return _ExpressionVector(map(math.pow, a_values, b_values), "numeric")
        except (OverflowError, ValueError):
            return _ExpressionVector(map(power, a_values, b_values), "numeric")

    return evaluate(tree)
