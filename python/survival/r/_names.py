"""R's ``make.names`` and ``make.unique``: the names R gives the columns of a data frame."""

from __future__ import annotations

from collections.abc import Sequence

# the words isValidName (R's gram.c) rejects
_RESERVED_WORDS = frozenset(
    {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "next",
        "break",
        "in",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "NA_complex_",
    }
)


def _make_names(value: str) -> str:
    """R's ``make.names`` for one name in a UTF-8 locale (``do_makenames``): an ``X`` before a
    name that does not start with a letter or with a dot not followed by a digit, a dot for
    every character that is not alphanumeric, a dot or an underscore, and a dot after a
    reserved word."""

    first, second = value[:1], value[1:2]
    if not (first.isalpha() or (first == "." and not "0" <= second <= "9")):
        value = f"X{value}"
    name = "".join(ch if ch.isalnum() or ch in "._" else "." for ch in value)
    return f"{name}." if name in _RESERVED_WORDS else name


def _make_unique(names: Sequence[str]) -> list[str]:
    """R's ``make.unique``: a repeated name becomes ``name.1``, ``name.2``, ... skipping any
    name already in *names* or given out earlier."""

    taken = set(names)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
            continue
        count = counts.get(name, 1)
        while f"{name}.{count}" in taken:
            count += 1
        unique = f"{name}.{count}"
        taken.add(unique)
        counts[name] = count + 1
        out.append(unique)
    return out
