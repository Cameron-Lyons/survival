"""Restricted, typed R expressions shared by formula discovery and evaluation.

Formula operators outside calls remain terms algebra. This parser handles the
vector expressions inside calls, response arguments, and logical formula terms;
it never evaluates Python or accepts arbitrary function calls.
"""

from __future__ import annotations

import math
import numbers
import re
import struct
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from operator import add, mul, sub, truediv
from typing import Any

import numpy as np

from ._coerce import (
    _NA_REAL,
    _as_character,
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
        contrast: Any = None,
    ) -> None:
        super().__init__(values)
        self.kind = kind
        self.categories = None if categories is None else tuple(categories)
        self.ordered = ordered
        self.declared = declared
        self.storage = (storage or "double") if kind == "numeric" else None
        self.contrast = contrast

    def take(self, rows: Iterable[int]) -> _ExpressionVector:
        missing = None if self.kind in {"logical", "character", "factor"} else _NA_REAL
        return _ExpressionVector(
            (missing if row < 0 else self[row] for row in rows),
            self.kind,
            self.categories,
            ordered=self.ordered,
            declared=self.declared,
            storage=getattr(self, "storage", None),
            contrast=getattr(self, "contrast", None),
        )


def _all_logical(values: Sequence[Any]) -> bool:
    """Whether *values* hold a logical and nothing but logicals and missing values."""

    types = set(map(type, values))
    logical = {
        kind
        for kind in types
        if kind is bool or (kind.__module__ == "numpy" and kind.__name__ in {"bool", "bool_"})
    }
    if not logical:
        return False
    others = types - logical - {type(None)}
    return not others or all(_is_missing_value(value) for value in values if type(value) in others)


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
        elif _all_logical(values):
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
        contrast=getattr(source, "contrast", None),
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
    names: tuple[str | None, ...] = ()
    parens: int = 0


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
_FORMALS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "I": (("x",), None),
    "identity": (("x",), None),
    "as.numeric": (("x",), "after"),
    "log": (("x", "base"), None),
    "sqrt": (("x",), None),
    "exp": (("x",), None),
    "factor": (("x", "levels", "labels", "exclude", "ordered", "nmax"), None),
    "as.factor": (("x",), None),
    "c": ((), "before"),
    "pmin": (("na.rm",), "before"),
    "pmax": (("na.rm",), "before"),
    "ifelse": (("test", "yes", "no"), None),
    "round": (("x", "digits"), None),
    "signif": (("x", "digits"), None),
    "floor": (("x",), None),
    "ceiling": (("x",), None),
    "trunc": (("x",), "after"),
    "abs": (("x",), None),
    "sign": (("x",), None),
    "log1p": (("x",), None),
    "expm1": (("x",), None),
    "log2": (("x",), None),
    "log10": (("x",), None),
    "as.character": (("x",), "after"),
    "as.integer": (("x",), "after"),
    "as.logical": (("x",), "after"),
    "is.na": (("x",), None),
    "relevel": (("x", "ref"), "after"),
    "interaction": (("drop", "sep", "lex.order"), "before"),
    "scale": (("x", "center", "scale"), None),
    "ns": (("x", "df", "knots", "intercept", "Boundary.knots"), None),
    "bs": (("x", "df", "knots", "degree", "intercept", "Boundary.knots", "warn.outside"), None),
    "poly": (("x", "degree", "coefs", "raw", "simple"), None),
    "nsk": (("x", "df", "knots", "intercept", "b", "Boundary.knots"), None),
}
_MATRIX_CALLS = frozenset({"ns", "bs", "poly", "nsk"})
_CALLS = frozenset(_FORMALS)
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
    "%in%": 65,
    "^": 80,
}
_SPACED_OPERATORS = frozenset({"+", "-", "*", "==", "!=", "<", "<=", ">", ">=", "&", "|", "%in%"})


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
                "NULL": (None, "null"),
            }
            self.token = ("literal", literal[value]) if value in literal else ("name", value)
            return
        if char == "%":
            end = text.find("%", self.pos + 1)
            if text[self.pos : end + 1] != "%in%":
                raise self.error()
            self.pos = end + 1
            self.token = ("%in%", "%in%")
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
            ",",
            "=",
        ):
            if text.startswith(operator, self.pos):
                self.pos += len(operator)
                self.token = (operator, operator)
                return
        raise self.error()

    def argument_name(self) -> str | None:
        if self.token[0] != "name":
            return None
        state = (self.pos, self.token, self.begin)
        name = self.token[1]
        self.advance()
        if self.token[0] == "=":
            self.advance()
            return name
        self.pos, self.token, self.begin = state
        return None

    def call(self, function: str) -> _ExpressionNode:
        if function not in _CALLS:
            raise self.error()
        self.advance()
        children: list[_ExpressionNode] = []
        names: list[str | None] = []
        if self.token[0] != ")":
            while True:
                names.append(self.argument_name())
                children.append(self.parse())
                if self.token[0] != ",":
                    break
                self.advance()
        if self.token[0] != ")":
            raise self.error()
        self.advance()
        columns = tuple(dict.fromkeys(column for child in children for column in child.columns))
        return _ExpressionNode(
            "call",
            function,
            tuple(children),
            columns,
            names=tuple(names) if any(names) else (),
        )

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
                left = self.call(value)
            else:
                left = _ExpressionNode("column", value, columns=(value,))
        elif kind == "(":
            left = self.parse()
            if self.token[0] != ")":
                raise self.error()
            self.advance()
            left = replace(left, parens=left.parens + 1)
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


_RESERVED_NAMES = frozenset(
    {
        "TRUE",
        "FALSE",
        "NA",
        "NA_real_",
        "NA_integer_",
        "NA_character_",
        "NaN",
        "Inf",
        "NULL",
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


def _deparse_name(name: str) -> str:
    syntactic = (
        name.replace(".", "_").isidentifier()
        and not (name.startswith(".") and len(name) > 1 and name[1].isdigit())
        and name not in _RESERVED_NAMES
    )
    return name if syntactic else "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"


def _deparse_literal(node: _ExpressionNode, keep_integer: bool) -> str:
    value, kind = node.value, node.declared_kind
    if kind == "null":
        return "NULL"
    if kind == "character":
        return "NA_character_" if value is None else _quoted_label(value)
    if kind == "logical":
        return "NA" if value is None else "TRUE" if value else "FALSE"
    if node.declared_storage == "integer":
        if _is_na_real(value):
            return "NA_integer_"
        return f"{value}L" if keep_integer else str(value)
    if _is_na_real(value):
        return "NA_real_"
    return _as_character(float(value))


def _deparse(node: _ExpressionNode, keep_integer: bool = False) -> str:
    """R's ``deparse`` of a parsed expression: its spacing, quotes and number format."""
    if node.kind == "literal":
        text = _deparse_literal(node, keep_integer)
    elif node.kind == "column":
        text = _deparse_name(node.value)
    elif node.kind == "call":
        names = node.names or (None,) * len(node.children)
        arguments = (
            ("" if name is None else f"{_deparse_name(name)} = ") + _deparse(child, keep_integer)
            for name, child in zip(names, node.children, strict=True)
        )
        text = f"{node.value}({', '.join(arguments)})"
    elif node.kind == "unary":
        text = node.value + _deparse(node.children[0], keep_integer)
    else:
        operator = f" {node.value} " if node.value in _SPACED_OPERATORS else node.value
        text = (
            _deparse(node.children[0], keep_integer)
            + operator
            + _deparse(node.children[1], keep_integer)
        )
    return "(" * node.parens + text + ")" * node.parens


@lru_cache(maxsize=1024)
def _expression_label(expression: str, *, keep_integer: bool = False) -> str:
    """R's deparsed label of *expression*: ``I(age>60)`` is ``I(age > 60)``."""
    try:
        return _deparse(_parse_expression(expression), keep_integer)
    except ValueError:
        return _token_label(expression, keep_integer=keep_integer)


def _token_label(expression: str, *, keep_integer: bool = False) -> str:
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


def _is_na_real(value: Any) -> bool:
    """Whether *value* is R's ``NA_real_`` rather than another NaN."""
    return (
        isinstance(value, float)
        and math.isnan(value)
        and struct.unpack("=Q", struct.pack("=d", value))[0] & 0xFFFFFFFF == 1954
    )


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


def _comparison_values(
    left: _ExpressionVector,
    operator: str,
    right: _ExpressionVector,
    compare: Callable[[Any, str, Any], bool | None],
) -> _ExpressionVector:
    """Apply R's factor methods before comparing the underlying atomic values."""

    left_factor, right_factor = left.kind == "factor", right.kind == "factor"
    if not (left_factor or right_factor):
        return _ExpressionVector(
            (compare(a, operator, b) for a, b in zip(left, right, strict=True)), "logical"
        )

    def levels(values: _ExpressionVector) -> tuple[str, ...]:
        return tuple(_as_character(value) for value in values.categories or ())

    def labels(values: _ExpressionVector) -> Iterable[Any]:
        if values.kind != "factor":
            return values
        return (None if _is_missing_value(value) else _as_character(value) for value in values)

    def codes(values: Iterable[Any], factor_levels: tuple[str, ...]) -> Iterable[int | None]:
        index = {value: i + 1 for i, value in enumerate(factor_levels)}
        return (
            None if _is_missing_value(value) else index.get(_as_character(value))
            for value in values
        )

    if left_factor and right_factor and left.ordered != right.ordered:
        left_method = "Ops.ordered" if left.ordered else "Ops.factor"
        right_method = "Ops.ordered" if right.ordered else "Ops.factor"
        _warn_outside_package(
            f'Incompatible methods ("{left_method}", "{right_method}") for "{operator}"'
        )
        left_values, right_values = codes(left, levels(left)), codes(right, levels(right))
    elif operator in {"==", "!="}:
        if left_factor and right_factor and set(levels(left)) != set(levels(right)):
            raise ValueError("level sets of factors are different")
        left_values, right_values = labels(left), labels(right)
    else:
        ordered = left if left_factor else right
        if not ordered.ordered:
            _warn_outside_package(f"'{operator}' not meaningful for factors")
            return _ExpressionVector([None] * len(left), "logical")
        factor_levels = levels(ordered)
        if left_factor and right_factor and levels(right) != factor_levels:
            raise ValueError("level sets of factors are different")
        left_values, right_values = codes(left, factor_levels), codes(right, factor_levels)
    return _ExpressionVector(
        (compare(a, operator, b) for a, b in zip(left_values, right_values, strict=True)),
        "logical",
    )


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
            return _evaluate_expression(node, 1, source, transform, compare, divide, power).take(())
        operator = node.value
        if node.kind == "column":
            values = _expression_source(source(operator), operator)
            if len(values) != n:
                raise ValueError("formula columns must have the same length as the Surv response")
            return values
        if node.kind == "literal":
            if node.declared_kind == "null":
                raise ValueError("NULL is not a vector in a formula expression")
            return _ExpressionVector(
                [operator] * n,
                node.declared_kind or "numeric",
                storage=node.declared_storage,
            )
        if node.kind == "call":
            return _evaluate_call(node, n, evaluate, constant, transform)
        left = evaluate(node.children[0])
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
        if operator == "%in%":
            table = node.children[1]
            return _in_values(left, evaluate(table) if table.columns else constant(table))
        right = evaluate(node.children[1])
        pairs = zip(left, right, strict=True)
        if operator in {"&", "|"}:
            if "factor" in {left.kind, right.kind}:
                _warn_outside_package(f"'{operator}' not meaningful for factors")
                return _ExpressionVector([None] * n, "logical")
            return _ExpressionVector((_logical_binary(a, operator, b) for a, b in pairs), "logical")
        if operator in {"==", "!=", "<", "<=", ">", ">="}:
            return _comparison_values(left, operator, right, compare)
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

    def constant(node: _ExpressionNode) -> _ExpressionVector:
        if node.columns:
            raise ValueError(f"{_deparse(node)} must not refer to data columns")
        if node.kind == "literal" and node.declared_kind == "null":
            return _ExpressionVector([], "null")
        if node.kind == "call" and node.value == "c":
            return _concatenate([constant(child) for child in node.children])
        return _evaluate_expression(node, 1, source, transform, compare, divide, power)

    return evaluate(tree)


def _match_call(node: _ExpressionNode) -> tuple[dict[str, _ExpressionNode], list[_ExpressionNode]]:
    """R's matching of a call's arguments: exact names, unique prefixes of the formals
    before ``...``, then positions; the rest go to ``...``."""

    function = node.value
    formals, dots = _FORMALS[function]
    names = node.names or (None,) * len(node.children)
    matched: dict[str, _ExpressionNode] = {}
    extra: list[tuple[int, _ExpressionNode]] = []
    positional: list[tuple[int, _ExpressionNode]] = []
    partial: list[tuple[int, str, _ExpressionNode]] = []
    for index, (name, child) in enumerate(zip(names, node.children, strict=True)):
        if name is None:
            positional.append((index, child))
        elif name in formals:
            if name in matched:
                raise ValueError(
                    f'{function}(): formal argument "{name}" matched by multiple actual arguments'
                )
            matched[name] = child
        else:
            partial.append((index, name, child))
    for index, name, child in partial:
        candidates = (
            []
            if dots == "before"
            else [formal for formal in formals if formal.startswith(name) and formal not in matched]
        )
        if len(candidates) > 1:
            raise ValueError(f"{function}(): argument {name} matches multiple formal arguments")
        if candidates:
            matched[candidates[0]] = child
        elif dots is not None:
            extra.append((index, child))
        else:
            raise ValueError(f"{function}(): unused argument ({name} = {_deparse(child)})")
    if dots == "before":
        extra.extend(positional)
    else:
        remaining = [formal for formal in formals if formal not in matched]
        matched.update(
            (formal, child) for formal, (_index, child) in zip(remaining, positional, strict=False)
        )
        overflow = positional[len(remaining) :]
        if overflow and dots is None:
            raise ValueError(f"{function}(): unused argument ({_deparse(overflow[0][1])})")
        extra.extend(overflow)
    return matched, [child for _index, child in sorted(extra, key=lambda item: item[0])]


def _scalar(values: _ExpressionVector, function: str, name: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{function}(): {name} must be a single value")
    return values[0]


def _flag(values: _ExpressionVector, function: str, name: str) -> bool:
    value = _scalar(values, function, name)
    if _is_missing_value(value) or isinstance(value, str):
        raise ValueError(f"{function}(): {name} must be TRUE or FALSE")
    return bool(value)


def _recycled(values: _ExpressionVector, n: int, function: str) -> _ExpressionVector:
    if len(values) == n:
        return values
    if len(values) != 1:
        raise ValueError(f"{function}(): arguments must have length 1 or the number of rows")
    return _ExpressionVector(
        values * n,
        values.kind,
        values.categories,
        ordered=values.ordered,
        storage=values.storage,
        contrast=values.contrast,
    )


_KIND_RANK = {"logical": 0, "integer": 1, "numeric": 2, "character": 3}


def _kind_rank(values: _ExpressionVector) -> int:
    if values.kind == "numeric":
        return _KIND_RANK["integer" if values.storage == "integer" else "numeric"]
    return _KIND_RANK[values.kind]


def _as_kind(value: Any, rank: int) -> Any:
    """One non-factor value coerced as R's ``c()`` coerces it to the type of *rank*."""
    if rank == _KIND_RANK["character"]:
        if _is_missing_value(value) and not _is_nan_number(value):
            return None
        return value if isinstance(value, str) else _as_character(value)
    if rank == _KIND_RANK["logical"]:
        return value
    if _is_missing_value(value):
        return value if _is_nan_number(value) else _NA_REAL
    if rank == _KIND_RANK["integer"]:
        return int(value)
    return float(value)


def _is_nan_number(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value) and not _is_na_real(value)


def _with_rank(values: list[Any], rank: int) -> _ExpressionVector:
    if rank == _KIND_RANK["character"]:
        return _ExpressionVector(values, "character")
    if rank == _KIND_RANK["logical"]:
        return _ExpressionVector(values, "logical")
    storage = "integer" if rank == _KIND_RANK["integer"] else "double"
    return _ExpressionVector(values, "numeric", storage=storage)


def _concatenate(vectors: Sequence[_ExpressionVector]) -> _ExpressionVector:
    vectors = [vector for vector in vectors if vector.kind != "null"]
    if any(vector.kind == "factor" for vector in vectors):
        raise ValueError("c() of factors is not supported in formula expressions")
    if not vectors:
        return _ExpressionVector([], "null")
    rank = max(map(_kind_rank, vectors))
    return _with_rank([_as_kind(value, rank) for vector in vectors for value in vector], rank)


def _factor_codes(values: _ExpressionVector) -> _ExpressionVector:
    """A factor's integer codes, as R's ``unclass``."""
    codes = {level: i + 1 for i, level in enumerate(values.categories or ())}
    return _ExpressionVector(
        (_NA_REAL if _is_missing_value(value) else codes[value] for value in values),
        "numeric",
        storage="integer",
    )


def _labels(values: _ExpressionVector) -> list[str | None]:
    """R's ``as.character``: factor labels, ``"TRUE"``, ``"1"``, ``"NaN"``; NA stays NA."""
    return [
        None
        if _is_missing_value(value) and not _is_nan_number(value)
        else value
        if isinstance(value, str)
        else "NaN"
        if _is_nan_number(value)
        else _as_character(value)
        for value in values
    ]


def _factor_vector(
    source: _ExpressionVector,
    values: Iterable[Any],
    levels: Sequence[Any],
    *,
    ordered: bool,
) -> _ExpressionVector:
    """A factor with *levels*; a default contrast declared on *source* is recomputed
    for the new levels, as ``factor()`` reevaluates ``options(contrasts)``."""

    contrast = getattr(source, "contrast", None)
    default_contrast = None
    default_name = (
        contrast.get("default_name") or (contrast.get("label") if contrast.get("default") else None)
        if contrast is not None
        else None
    )
    if default_name is not None and len(levels) > 1:
        from ._contrasts import _named_factor_contrast

        default_contrast = _named_factor_contrast(default_name, tuple(levels))
        if default_contrast is None:
            raise ValueError(f"factor() requires reevaluated {default_name} contrasts")
        default_contrast["default"] = True
    return _ExpressionVector(values, "factor", levels, ordered=ordered, contrast=default_contrast)


def _factor(
    x: _ExpressionVector,
    levels: _ExpressionVector | None = None,
    labels: _ExpressionVector | None = None,
    exclude: _ExpressionVector | None = None,
    ordered: bool | None = None,
) -> _ExpressionVector:
    """R's ``factor(x, levels, labels, exclude, ordered)``."""

    ordered = x.ordered if ordered is None else ordered
    if levels is None and labels is None and exclude is None:
        present = {value for value in x if not _is_missing_value(value)}
        found = (
            [value for value in x.categories if value in present]
            if x.categories is not None
            else sorted(present, key=_strata_level_sort_key)
        )
        return _factor_vector(x, x, found, ordered=ordered)
    if levels is None:
        present = {value for value in x if not _is_missing_value(value)}
        levels = _ExpressionVector(
            [value for value in x.categories if value in present]
            if x.categories is not None
            else sorted(present, key=_strata_level_sort_key),
            "character",
        )
    if exclude is not None and exclude.kind == "null":
        raise ValueError("factor(exclude = NULL) is not supported in formula expressions")
    excluded = set() if exclude is None else set(_labels(exclude))
    level_values = [
        level
        for level, label in zip(levels, _labels(levels), strict=True)
        if label is not None and label not in excluded
    ]
    level_labels = _labels(_ExpressionVector(level_values, levels.kind))
    index: dict[str | None, int] = {}
    for position, label in enumerate(level_labels):
        if label in index:
            raise ValueError(f"factor level [{position + 1}] is duplicated")
        index[label] = position
    codes = [index.get(label) for label in _labels(x)]
    if labels is None or labels.kind == "null":
        names = level_values
    elif len(labels) == len(level_values):
        names = list(labels)
    elif len(labels) == 1:
        names = [f"{_labels(labels)[0]}{i}" for i in range(1, len(level_values) + 1)]
    else:
        raise ValueError(
            f"invalid 'labels'; length {len(labels)} should be 1 or {len(level_values)}"
        )
    categories = list(dict.fromkeys(names))
    return _factor_vector(
        x, (None if code is None else names[code] for code in codes), categories, ordered=ordered
    )


def _relevel(x: _ExpressionVector, ref: _ExpressionVector) -> _ExpressionVector:
    if x.kind != "factor":
        raise ValueError("'relevel' only for (unordered) factors")
    if x.ordered:
        raise ValueError("'relevel' only for unordered factors")
    if len(ref) != 1:
        raise ValueError("'ref' must be of length one")
    levels = list(x.categories or ())
    if ref.kind == "character":
        labels = _labels(_ExpressionVector(levels, "factor"))
        if ref[0] not in labels:
            raise ValueError("'ref' must be an existing level")
        position = labels.index(ref[0])
    else:
        number = float(ref[0])
        if not (1 <= number <= len(levels)):
            raise ValueError(f"ref = {_as_character(ref[0])} must be in 1L:{len(levels)}")
        position = int(number) - 1
    order = [levels[position], *levels[:position], *levels[position + 1 :]]
    return _factor_vector(x, x, order, ordered=False)


def _interaction(
    factors: Sequence[_ExpressionVector], *, drop: bool, sep: str, lex_order: bool
) -> _ExpressionVector:
    """R's ``interaction(...)``: the first factor varies fastest unless *lex_order*."""

    if not factors:
        raise ValueError("No factors specified")
    coded: list[tuple[list[str | None], list[int | None]]] = []
    for values in factors:
        factor = values if values.kind == "factor" else _factor(values)
        levels = list(factor.categories or ())
        if drop:
            present = {value for value in factor if not _is_missing_value(value)}
            levels = [level for level in levels if level in present]
        index = {level: i for i, level in enumerate(levels)}
        codes = [None if _is_missing_value(value) else index[value] for value in factor]
        coded.append((_labels(_ExpressionVector(levels, "factor")), codes))
    labels, answer = coded[-1][0], list(coded[-1][1])
    for level_labels, codes in reversed(coded[:-1]):
        width = len(labels) if lex_order else len(level_labels)
        answer = [
            None if a is None or c is None else a + width * c if lex_order else a * width + c
            for a, c in zip(answer, codes, strict=True)
        ]
        labels = (
            [f"{left}{sep}{right}" for left in level_labels for right in labels]
            if lex_order
            else [f"{left}{sep}{right}" for right in labels for left in level_labels]
        )
    combined = [None if code is None else labels[code] for code in answer]
    categories = list(dict.fromkeys(labels))
    if drop:
        used = {label for label in combined if label is not None}
        categories = [label for label in categories if label in used]
    return _ExpressionVector(combined, "factor", categories)


def _in_values(x: _ExpressionVector, table: _ExpressionVector) -> _ExpressionVector:
    """R's ``x %in% table``: ``match`` on the common type; NA matches NA."""

    character = bool({x.kind, table.kind} & {"character", "factor"})

    def keys(values: _ExpressionVector) -> list[Any]:
        if character:
            return [("NA",) if label is None else label for label in _labels(values)]
        return [
            ("NaN",)
            if _is_nan_number(value)
            else ("NA",)
            if _is_missing_value(value)
            else float(value)
            for value in values
        ]

    lookup = set(keys(table)) if table.kind != "null" else set()
    return _ExpressionVector((key in lookup for key in keys(x)), "logical")


def _logical_flags(values: _ExpressionVector) -> list[bool | None]:
    """R's ``as.logical`` of an ``ifelse`` test."""
    if values.kind == "factor":
        return [None if _is_missing_value(value) else True for value in values]
    if values.kind == "character":
        truth = {"TRUE": True, "true": True, "T": True, "True": True}
        truth.update({"FALSE": False, "false": False, "F": False, "False": False})
        return [None if value is None else truth.get(value) for value in values]
    return [None if _is_missing_value(value) else bool(value) for value in values]


def _ifelse(
    test: _ExpressionVector, yes: _ExpressionVector, no: _ExpressionVector
) -> _ExpressionVector:
    """R's ``ifelse``: the result takes the type of the branches it uses, and factor
    branches contribute their integer codes."""

    flags = _logical_flags(test)
    yes = _factor_codes(yes) if yes.kind == "factor" else yes
    no = _factor_codes(no) if no.kind == "factor" else no
    rank = _KIND_RANK["logical"]
    if any(flag is True for flag in flags):
        rank = max(rank, _kind_rank(yes))
    if any(flag is False for flag in flags):
        rank = max(rank, _kind_rank(no))
    missing = _as_kind(None, rank)
    values = [
        missing if flag is None else _as_kind(a if flag else b, rank)
        for flag, a, b in zip(flags, yes, no, strict=True)
    ]
    return _with_rank(values, rank)


def _extreme(
    function: str, vectors: Sequence[_ExpressionVector], na_rm: bool, n: int
) -> _ExpressionVector:
    """R's ``pmin``/``pmax`` of numeric and logical vectors."""

    if not vectors:
        raise ValueError(f"{function}(): no arguments")
    if any(vector.kind in {"factor", "character"} for vector in vectors):
        raise ValueError(f"{function}() requires numeric or logical arguments")
    stack = np.array([_floats_or_nan(vector) for vector in vectors], dtype=float).reshape(
        len(vectors), n
    )
    nan = np.isnan(stack)
    na = nan & ((stack.view(np.uint64) & np.uint64(0xFFFFFFFF)) == np.uint64(1954))
    with np.errstate(all="ignore"):
        if na_rm:
            reduce = np.fmin if function == "pmin" else np.fmax
            result = reduce.reduce(stack, axis=0)
            result[nan.all(axis=0)] = _NA_REAL
        else:
            result = (np.min if function == "pmin" else np.max)(stack, axis=0)
            result[nan.any(axis=0)] = math.nan
            result[na.any(axis=0)] = _NA_REAL
    if all(_integer_operand(vector) for vector in vectors):
        return _integer_result(result.tolist())
    return _ExpressionVector(result.tolist(), "numeric")


_MATH = {
    "floor": np.floor,
    "ceiling": np.ceil,
    "trunc": np.trunc,
    "abs": np.abs,
    "sign": np.sign,
    "log1p": np.log1p,
    "expm1": np.expm1,
    "log2": np.log2,
    "log10": np.log10,
}


def _math(
    function: str, values: _ExpressionVector, label: str, base: float | None = None
) -> _ExpressionVector:
    """R's Math group functions, with ``NaNs produced`` warnings and NA kept as NA."""

    if values.kind in {"factor", "character"}:
        raise ValueError(f"non-numeric argument to mathematical function {function}()")
    numeric = np.asarray(_floats_or_nan(values), dtype=float)
    with np.errstate(all="ignore"):
        if function == "log":
            result = (
                np.log10(numeric)
                if base == 10
                else np.log2(numeric)
                if base == 2
                else np.log(numeric) / math.log(base)
                if base is not None and base > 0
                else np.log(numeric) / np.log(float("nan") if base is None else base)
            )
        else:
            result = _MATH[function](numeric)
    nan = np.isnan(numeric)
    if np.any(np.isnan(result) & ~nan):
        _warn_outside_package(f"NaNs produced in {function}({label})")
    na = nan & ((numeric.view(np.uint64) & np.uint64(0xFFFFFFFF)) == np.uint64(1954))
    result[na] = _NA_REAL
    if function == "abs" and _integer_operand(values):
        return _integer_result(result.tolist())
    return _ExpressionVector(result.tolist(), "numeric")


def _rounded(function: str, values: _ExpressionVector, digits: float) -> _ExpressionVector:
    if values.kind in {"factor", "character"}:
        raise ValueError(f"non-numeric argument to mathematical function {function}()")
    if function == "round":
        places = int(digits)
        return _ExpressionVector(
            (
                value if not math.isfinite(value) else round(value, places)
                for value in _floats_or_nan(values)
            ),
            "numeric",
        )
    places = max(1, min(22, int(digits)))
    return _ExpressionVector(
        (
            value if not math.isfinite(value) or value == 0 else float(f"{value:.{places}g}")
            for value in _floats_or_nan(values)
        ),
        "numeric",
    )


def _as_integer(values: _ExpressionVector) -> _ExpressionVector:
    if values.kind == "factor":
        return _factor_codes(values)
    numbers = _numeric_values(values, cast=True)
    result: list[Any] = []
    overflow = False
    for value in numbers:
        if math.isnan(value):
            result.append(_NA_REAL)
        elif -(2**31) < value < 2**31:
            result.append(int(value))
        else:
            result.append(_NA_REAL)
            overflow = True
    if overflow:
        _warn_outside_package("NAs introduced by coercion to integer range")
    return _ExpressionVector(result, "numeric", storage="integer")


def _as_logical(values: _ExpressionVector) -> _ExpressionVector:
    if values.kind in {"factor", "character"}:
        truth = {"TRUE": True, "true": True, "T": True, "True": True}
        truth.update({"FALSE": False, "false": False, "F": False, "False": False})
        return _ExpressionVector(
            (None if label is None else truth.get(label) for label in _labels(values)), "logical"
        )
    return _ExpressionVector(
        (None if _is_missing_value(value) else bool(value) for value in values), "logical"
    )


def _evaluate_call(
    node: _ExpressionNode,
    n: int,
    evaluate: Callable[[_ExpressionNode], _ExpressionVector],
    constant: Callable[[_ExpressionNode], _ExpressionVector],
    transform: Callable[[list[float], str, str], list[float]],
) -> _ExpressionVector:
    function = node.value
    arguments, dots = _match_call(node)

    def argument(name: str) -> _ExpressionVector:
        if name not in arguments:
            raise ValueError(f'{function}(): argument "{name}" is missing, with no default')
        return evaluate(arguments[name])

    def option(name: str) -> _ExpressionVector | None:
        return constant(arguments[name]) if name in arguments else None

    def scalar(name: str, default: Any) -> Any:
        values = option(name)
        return default if values is None else _scalar(values, function, name)

    def flag(name: str, default: bool) -> bool:
        values = option(name)
        return default if values is None else _flag(values, function, name)

    if function in _MATRIX_CALLS:
        raise ValueError(f"{function}() is a matrix term and cannot be used inside an expression")
    if function == "c":
        return _recycled(constant(node), n, function)
    if function in {"pmin", "pmax"}:
        return _extreme(function, [evaluate(child) for child in dots], flag("na.rm", False), n)
    if function == "interaction":
        return _interaction(
            [evaluate(child) for child in dots],
            drop=flag("drop", False),
            sep=str(scalar("sep", ".")),
            lex_order=flag("lex.order", False),
        )
    if function == "ifelse":
        return _ifelse(argument("test"), argument("yes"), argument("no"))
    x = argument("x")
    label = _deparse(arguments["x"])
    if function in {"I", "identity"}:
        return x
    if function == "factor":
        ordered = option("ordered")
        return _factor(
            x,
            option("levels"),
            option("labels"),
            option("exclude"),
            None if ordered is None else _flag(ordered, function, "ordered"),
        )
    if function == "as.factor":
        return x if x.kind == "factor" else _factor(x)
    if function == "relevel":
        if "ref" not in arguments:
            raise ValueError('relevel(): argument "ref" is missing, with no default')
        return _relevel(x, constant(arguments["ref"]))
    if function == "as.character":
        return _ExpressionVector(_labels(x), "character")
    if function == "as.integer":
        return _as_integer(x)
    if function == "as.logical":
        return _as_logical(x)
    if function == "is.na":
        return _ExpressionVector((_is_missing_value(value) for value in x), "logical")
    if function == "scale":
        return _scale(x, option("center"), option("scale"))
    if function in {"round", "signif"}:
        digits = scalar("digits", 0 if function == "round" else 6)
        if _is_missing_value(digits) or isinstance(digits, str):
            raise ValueError(f"{function}(): digits must be a number")
        return _rounded(function, x, float(digits))
    if function == "log" and "base" in arguments:
        base = scalar("base", math.e)
        return _math("log", x, label, None if _is_missing_value(base) else float(base))
    if function in _MATH:
        return _math(function, x, label)
    numeric = _numeric_values(x, cast=function == "as.numeric")
    return _ExpressionVector(transform(numeric, function, label), "numeric")


def _scale(
    values: _ExpressionVector,
    center: _ExpressionVector | None,
    scale: _ExpressionVector | None,
) -> _ExpressionVector:
    """R's ``scale`` of one column: centred by its mean and divided by its standard
    deviation (or by its root mean square when it is not centred), ignoring NA."""

    if values.kind in {"factor", "character"}:
        raise ValueError("scale() requires numeric values")
    numeric = np.asarray(_floats_or_nan(values), dtype=float)
    observed = numeric[~np.isnan(numeric)]

    def setting(option: _ExpressionVector | None, name: str) -> bool | float:
        if option is None:
            return True
        value = _scalar(option, "scale", name)
        if option.kind == "logical":
            return bool(value)
        return float(value)

    centre = setting(center, "center")
    if centre is True:
        array = observed.astype(np.longdouble)
        mean = np.sum(array) / len(array) if len(array) else np.nan
        mean += np.sum(array - mean) / len(array) if len(array) else 0.0
        numeric = numeric - float(mean)
        observed = observed - float(mean)
    elif centre is not False:
        numeric = numeric - float(centre)
        observed = observed - float(centre)
    divisor = setting(scale, "scale")
    if divisor is True:
        divisor = (
            math.sqrt(float(np.sum(observed**2)) / (len(observed) - 1))
            if len(observed) > 1
            else math.nan
        )
    if divisor is not False:
        numeric = numeric / float(divisor)
    return _ExpressionVector(numeric.tolist(), "numeric")
