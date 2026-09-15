"""Input coercion, option normalisation, and small shared numeric helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from operator import index
from typing import Any

from .. import _survival as _core
from ._types import _SurvfitComputation

_EXP_CLAMP_MIN = -745.0
_EXP_CLAMP_MAX = 709.0
_SURVFIT_TIME_EPSILON = 1e-9
_VARIANCE_SCALE_FLOOR = 1e-12
_COX_DFBETAS_SCALE_FLOOR = 1e-10
_SURV_TYPES = ("right", "left", "interval", "counting", "interval2", "mstate")
_SURV_RESPONSE_TYPES = (*_SURV_TYPES[:-1], "mright", "mcounting")


def _coerce_mapping_rows(values: Mapping[Any, Any], name: str) -> list[list[Any]]:
    keys = tuple(values)
    if not keys:
        return []

    columns: list[list[Any]] = []
    row_count: int | None = None
    for key in keys:
        column = _coerce_array_like(values[key], f"{name}[{key!r}]")
        if column and isinstance(column[0], list | tuple):
            raise ValueError(f"{name} columns must be one-dimensional")
        if row_count is None:
            row_count = len(column)
        elif len(column) != row_count:
            raise ValueError(f"{name} columns must have the same length")
        columns.append(column)

    n_rows = row_count or 0
    return [[column[row_idx] for column in columns] for row_idx in range(n_rows)]


def _coerce_array_like(values: Any, name: str) -> list[Any]:
    if values is None:
        raise ValueError(f"{name} is required")
    if isinstance(values, Mapping):
        return _coerce_mapping_rows(values, name)
    if hasattr(values, "to_list"):
        values = values.to_list()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()

    if isinstance(values, str | bytes):
        raise TypeError(f"{name} must be array-like, not a string")

    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be array-like") from exc

    return result


class _RFactorVector:
    """Iterable factor values with level metadata preserved across reticulate."""

    def __init__(self, values: Any, levels: Any):
        self._values = tuple(values)
        self.categories = tuple(levels)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, item: int) -> Any:
        return self._values[item]


def _r_factor(values: Any, levels: Any) -> _RFactorVector:
    return _RFactorVector(values, levels)


def _materialize_1d(values: Any, name: str) -> list[Any]:
    result = _coerce_array_like(values, name)
    if result and isinstance(result[0], list | tuple):
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _materialize_labels(values: Any, name: str) -> list[Any]:
    result = _coerce_array_like(values, name)
    if any(isinstance(value, list) for value in result):
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _float_vector(values: Any, name: str) -> list[float]:
    return [float(value) for value in _materialize_1d(values, name)]


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _int_vector(values: Any, name: str) -> list[int]:
    return [int(value) for value in _materialize_1d(values, name)]


def _integer_scalar(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _integer_code_vector(values: Any, name: str, description: str) -> list[int]:
    result: list[int] = []
    for value in _materialize_1d(values, name):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must use {description}") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must use {description}")
        result.append(int(numeric))
    return result


def _bool_vector(values: Any, name: str) -> list[bool]:
    result = []
    for value in _materialize_1d(values, name):
        if not _is_bool_like(value):
            raise TypeError(f"{name} must contain only True or False values")
        result.append(bool(value))
    return result


def _safe_exp(value: float) -> float:
    return math.exp(min(max(value, _EXP_CLAMP_MIN), _EXP_CLAMP_MAX))


def _event_vector(values: Any, name: str) -> list[int]:
    events = _integer_code_vector(values, name, "0/1 or 1/2 event coding")
    observed = set(events)
    if observed <= {0, 1}:
        return events
    if observed <= {1, 2}:
        return [int(value == 2) for value in events]
    raise ValueError(f"{name} must use 0/1 or 1/2 event coding")


def _mstate_categories(values: Any) -> Any | None:
    return _categories(values)


def _mstate_event_vector(values: Any, name: str) -> tuple[list[int | None], tuple[str, ...]]:
    raw = _materialize_1d(values, name)
    categories = _mstate_categories(values)
    if categories is None:
        levels = _mstate_inferred_levels(raw)
    else:
        levels = [
            _mstate_event_label(value)
            for value in _materialize_1d(categories, f"{name} categories")
            if not _is_missing_value(value)
        ]
    level_index = {level: idx for idx, level in enumerate(levels)}
    events = [
        None if _is_missing_value(value) else level_index[_mstate_event_label(value)]
        for value in raw
    ]
    return events, tuple(levels[1:])


def _mstate_levels(values: Any, name: str) -> tuple[list[Any], list[str]]:
    raw = _materialize_1d(values, name)
    categories = _mstate_categories(values)
    levels = (
        _mstate_inferred_levels(raw)
        if categories is None
        else [
            _mstate_event_label(value)
            for value in _materialize_1d(categories, f"{name} categories")
            if not _is_missing_value(value)
        ]
    )
    return raw, levels


def _interval_status_vector(values: Any, name: str) -> list[int]:
    status = _integer_code_vector(values, name, "0/1/2/3 interval censoring codes")
    if not set(status) <= {0, 1, 2, 3}:
        raise ValueError(f"{name} must use 0/1/2/3 interval censoring codes")
    return status


def _interval_endpoint_vector(values: Any, name: str, missing_value: float) -> list[float]:
    endpoints = _coerce_array_like(values, name)
    if endpoints and isinstance(endpoints[0], list | tuple):
        raise ValueError(f"{name} must be one-dimensional")
    return [missing_value if _is_missing_value(value) else float(value) for value in endpoints]


def _optional_float_vector(values: Any | None, name: str, n: int) -> list[float] | None:
    if values is None:
        return None
    result = _float_vector(values, name)
    if len(result) != n:
        raise ValueError(f"{name} must have length {n}")
    return result


def _quantile_vector(values: Any, name: str) -> list[float]:
    try:
        return [float(values)]
    except (TypeError, ValueError):
        return _float_vector(values, name)


def _normalize_na_action(na_action: str | None) -> str:
    if na_action is None:
        return "pass"
    if not isinstance(na_action, str):
        raise TypeError("na_action must be a string or None")
    action = na_action.strip().lower().replace(".", "_")
    aliases = {
        "fail": "fail",
        "na_fail": "fail",
        "omit": "omit",
        "na_omit": "omit",
        "exclude": "omit",
        "na_exclude": "omit",
        "pass": "pass",
        "na_pass": "pass",
    }
    try:
        return aliases[action]
    except KeyError as exc:
        raise ValueError(
            "na_action must be 'fail', 'omit', 'pass', 'na.fail', "
            "'na.omit', 'na.exclude', or 'na.pass'"
        ) from exc


def _is_missing_value(value: Any) -> bool:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _row_has_missing(value: Any) -> bool:
    value_type = type(value)
    if value is None:
        return True
    if value_type is float:
        return math.isnan(value)
    if value_type is int or value_type is bool or value_type is str:
        return False
    if isinstance(value, list | tuple):
        return any(_row_has_missing(item) for item in value)
    return _is_missing_value(value)


def _missing_row_indices(columns: list[tuple[str, Any]], n: int) -> set[int]:
    missing: set[int] = set()
    for name, values in columns:
        materialized = _coerce_array_like(values, name)
        if len(materialized) != n:
            raise ValueError(f"{name} must have length {n}")
        missing.update(idx for idx, value in enumerate(materialized) if _row_has_missing(value))
    return missing


def _keep_rows_after_na_action(
    missing: set[int],
    n: int,
    na_action: str | None,
    context: str,
) -> list[int] | None:
    action = _normalize_na_action(na_action)
    if action == "pass" or not missing:
        return None
    if action == "fail":
        raise ValueError(f"missing values in {context}")
    return [idx for idx in range(n) if idx not in missing]


def _is_bool_like(value: Any) -> bool:
    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__ == "numpy" and value_type.__name__ in {"bool", "bool_"}
    )


def _subset_indices(subset: Any, n: int) -> list[int]:
    values = _materialize_1d(subset, "subset")
    if values and all(_is_bool_like(value) for value in values):
        if len(values) != n:
            raise ValueError("subset mask must have the same length as the Surv response")
        indices = [idx for idx, value in enumerate(values) if bool(value)]
    else:
        indices = []
        for value in values:
            try:
                idx = index(value)
            except TypeError as exc:
                raise TypeError("subset must contain booleans or integer row indices") from exc
            if idx < 0 or idx >= n:
                raise ValueError("subset row indices must be between 0 and n - 1")
            indices.append(idx)

    if not indices:
        raise ValueError("subset selects no rows")
    return indices


def _subset_sequence(values: Any, indices: list[int], name: str) -> Any:
    materialized = _coerce_array_like(values, name)
    if indices and max(indices) >= len(materialized):
        raise ValueError(f"{name} must have enough rows for subset")
    subsetted = [materialized[idx] for idx in indices]
    categories = _mstate_categories(values)
    if categories is not None:
        return _RFactorVector(subsetted, categories)
    return subsetted


def _subset_optional_sequence(
    values: Any | None,
    indices: list[int],
    name: str,
) -> list[Any] | None:
    if values is None:
        return None
    return _subset_sequence(values, indices, name)


def _subset_data(data: Any, indices: list[int]) -> Any:
    if isinstance(data, Mapping):
        return {key: _subset_sequence(value, indices, str(key)) for key, value in data.items()}
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    if hasattr(data, "take"):
        try:
            return data.take(indices)
        except TypeError:
            pass
    raise TypeError("subset with formula data requires a mapping or tabular object")


def _as_rows(values: Any, name: str) -> list[list[float]]:
    return _as_matrix_rows(values, name, allow_empty_columns=False)


def _matrix_input_column_names(values: Any) -> tuple[str, ...] | None:
    if isinstance(values, Mapping):
        return tuple(str(key) for key in values) or None
    columns = getattr(values, "columns", None)
    if columns is None:
        return None
    try:
        names = tuple(str(column) for column in columns)
    except TypeError:
        return None
    return names or None


def _validated_matrix_column_names(
    names: tuple[str, ...] | None,
    rows: list[list[float]],
) -> tuple[str, ...] | None:
    if names is None:
        return None
    width = len(rows[0]) if rows else 0
    return names if len(names) == width else None


def _as_matrix_rows(
    values: Any,
    name: str,
    *,
    allow_empty_columns: bool,
) -> list[list[float]]:
    rows = _coerce_array_like(values, name)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    if not isinstance(rows[0], list | tuple):
        return [[float(value)] for value in rows]

    width = len(rows[0])
    if width == 0 and not allow_empty_columns:
        raise ValueError(f"{name} must have at least one column")
    matrix = [[float(value) for value in row] for row in rows]
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return matrix


def _label_levels(values: list[Any], name: str) -> tuple[Any, ...]:
    labels: dict[Any, None] = {}
    for value in values:
        try:
            labels.setdefault(value, None)
        except TypeError as exc:
            raise TypeError(f"{name} contains unhashable labels") from exc
    return tuple(labels)


def _encode_groups(
    group: Any,
    n: int,
    *,
    levels: Sequence[Any] | None = None,
) -> list[int]:
    values = _materialize_labels(group, "group")
    if len(values) != n:
        raise ValueError("group must have the same length as the Surv response")
    if levels is not None:
        return _encode_labels_with_levels(values, levels, "group")
    return _encode_labels(values, "group")


def _encode_labels(values: list[Any], name: str) -> list[int]:
    labels = {value: idx for idx, value in enumerate(_label_levels(values, name))}
    return [labels[value] for value in values]


def _encode_labels_with_levels(
    values: list[Any],
    levels: Sequence[Any],
    name: str,
) -> list[int]:
    try:
        labels = {value: idx for idx, value in enumerate(levels)}
    except TypeError as exc:
        raise TypeError(f"{name} contains unhashable labels") from exc
    try:
        return [labels[value] for value in values]
    except KeyError as exc:
        raise ValueError(f"{name} contains a value outside the supplied levels") from exc


def _group_indices(
    group: Any,
    n: int,
    *,
    levels: Sequence[Any] | None = None,
) -> dict[Any, list[int]]:
    values = _materialize_labels(group, "group")
    if len(values) != n:
        raise ValueError("group must have the same length as the Surv response")

    indices: dict[Any, list[int]] = {}
    for idx, value in enumerate(values):
        try:
            indices.setdefault(value, []).append(idx)
        except TypeError as exc:
            raise TypeError("group contains unhashable labels") from exc
    if levels is None:
        return indices
    ordered: dict[Any, list[int]] = {}
    for level in levels:
        if level in indices:
            ordered[level] = indices[level]
    if len(ordered) != len(indices):
        raise ValueError("group contains a value outside the supplied levels")
    return ordered


def _cox_tie_method(method: str | None, ties: str | None) -> str:
    choices = ("efron", "breslow", "exact")
    message = "coxph ties must be 'efron', 'breslow', or 'exact'"
    if method is not None and ties is not None:
        method_name = _match_string_arg(method, "method", choices, message)
        ties_name = _match_string_arg(ties, "ties", choices, message)
        if method_name != ties_name:
            raise ValueError("use only one of method or ties")
        return method_name

    return _match_string_arg(
        ties if ties is not None else method or "efron", "ties", choices, message
    )


def _match_string_arg(
    value: Any,
    name: str,
    choices: Sequence[str],
    message: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if normalized in choices:
        return normalized
    matches = [choice for choice in choices if choice.startswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"{name} is ambiguous; use a full value")
    raise ValueError(message)


# ---------------------------------------------------------------------------
# R's factor() / as.character() / format(): the single level and label path.
# ---------------------------------------------------------------------------


def _categories(values: Any) -> list[Any] | None:
    """The declared levels of a factor-like column, or ``None``.

    A pandas ``Categorical`` (``dtype.categories``), the test suite's ``RFactor`` and the
    reticulate bridge's ``_RFactorVector`` all expose ``categories``; R keeps the
    level order they declare, so every level helper starts here.
    """

    categories = getattr(values, "categories", None)
    if categories is None:
        categories = getattr(getattr(values, "dtype", None), "categories", None)
    if categories is None:
        return None
    return list(_coerce_array_like(categories, "categories"))


def _r_scientific(value: float, digits: int) -> tuple[bool, int, int]:
    """R's ``scientific()`` (format.c): sign, decimal exponent and significant digits."""

    mantissa, exponent = f"{abs(value):.{digits - 1}e}".split("e")
    significant = mantissa.replace(".", "").rstrip("0") or "0"
    return value < 0.0, int(exponent), len(significant)


def _r_format_numbers(values: Sequence[Any], digits: int = 7) -> list[str]:
    """R's ``formatReal``: a numeric vector in a common fixed or scientific layout.

    Every finite element shares the decimals (``format(c(1.5, 10))`` is
    ``" 1.5"``, ``"10.0"``) and the layout is fixed unless scientific notation is
    narrower; missing values print as ``NA`` and the infinities as ``Inf``.  ``as.character``
    formats each value on its own with 15 significant digits, ``format`` and
    ``print`` a whole vector with 7.
    """

    numbers = [None if _is_missing_value(value) else float(value) for value in values]
    finite = [number for number in numbers if number is not None and math.isfinite(number)]
    negative = any(number < 0.0 for number in finite)
    rgt = mxsl = mxns = -(10**9)
    mxl = -(10**9)
    mnl = 10**9
    for number in finite:
        neg, kpower, nsig = _r_scientific(number, digits) if number != 0.0 else (False, 0, 1)
        left = kpower + 1
        sleft = int(neg) + (left if left > 0 else 1)
        rgt = max(rgt, nsig - left)
        mxl, mnl = max(mxl, left), min(mnl, left)
        mxsl = max(mxsl, sleft)
        mxns = max(mxns, nsig)
    fixed, decimals = True, 0
    if finite:
        if mxl < 0:
            mxsl = 1 + int(negative)
        rgt = max(rgt, 0)
        fixed_width = mxsl + rgt + (1 if rgt else 0)
        exponent_width = 2 if mxl > 100 or mnl <= -99 else 1
        sci_decimals = mxns - 1
        scientific_width = (
            int(negative) + (1 if sci_decimals else 0) + sci_decimals + 4 + exponent_width
        )
        fixed = fixed_width <= scientific_width
        decimals = rgt if fixed else sci_decimals
    out: list[str] = []
    for number in numbers:
        if number is None or math.isnan(number):
            out.append("NA")  # NaN is the package's numeric NA
        elif math.isinf(number):
            out.append("Inf" if number > 0.0 else "-Inf")
        elif fixed:
            out.append(f"{number:.{decimals}f}")
        else:
            mantissa, exponent = f"{number:.{decimals}e}".split("e")
            out.append(f"{mantissa}e{'-' if int(exponent) < 0 else '+'}{abs(int(exponent)):02d}")
    width = max((len(label) for label in out), default=0)
    return [label.rjust(width) for label in out]


def _r_format_number(value: Any, digits: int = 7) -> str:
    """R's ``formatReal`` for one number (``as.character`` with ``digits=15``)."""

    if _is_missing_value(value):
        return "NA"
    return _r_format_numbers([value], digits)[0]


def _as_character(value: Any) -> str:
    """R's ``as.character`` of one atomic value (``"TRUE"``, ``"1"`` not ``"1.0"``, ``"NA"``)."""

    if _is_bool_like(value):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _r_format_number(value, 15)
    if _is_missing_value(value):
        return "NA"
    return str(value)


def _r_sort_key(value: Any) -> tuple[Any, ...]:
    """R's ``sort`` order for factor levels: numbers (and logicals) before strings."""

    if isinstance(value, tuple):
        return tuple(_r_sort_key(part) for part in value)
    if _is_bool_like(value):
        return (0, int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, str(value))


def _factor_levels(values: Any, name: str = "values") -> list[Any]:
    """R's ``levels(factor(x))``: the declared categories, else the sorted unique values.

    Missing values never form a level; the returned levels are the raw values so
    that codes can be looked up, ``_as_character`` renders them as R labels them.
    """

    declared = _categories(values)
    if declared is not None:
        return [level for level in declared if not _is_missing_value(level)]
    unique: dict[Any, None] = {}
    for value in _materialize_labels(values, name):
        if _is_missing_value(value):
            continue
        try:
            unique.setdefault(value, None)
        except TypeError as exc:
            raise TypeError(f"{name} contains unhashable labels") from exc
    return sorted(unique, key=_r_sort_key)


def _factor(values: Any, name: str = "values") -> tuple[list[int | None], list[str]]:
    """R's ``factor(x)`` as zero-based codes (``None`` for ``NA``) and level labels."""

    levels = _factor_levels(values, name)
    index = {level: code for code, level in enumerate(levels)}
    codes: list[int | None] = []
    for value in _materialize_labels(values, name):
        if _is_missing_value(value):
            codes.append(None)
            continue
        try:
            codes.append(index[value])
        except KeyError as exc:
            raise ValueError(f"{name} contains a value outside the declared categories") from exc
    return codes, [_as_character(level) for level in levels]


def _surv_format_number(value: float) -> str:
    """R's ``format()`` of one time value (7 significant digits)."""

    return _r_format_number(value, 7)


# Older spellings kept for the modules that still import them; each is an alias of
# the single implementation above.
_strata_value_label = _as_character
_strata_level_sort_key = _r_sort_key
_mstate_event_label = _as_character
_survdiff_r_level_sort_key = _r_sort_key


def _r_formula_ordered_levels(values: list[Any], name: str) -> tuple[Any, ...]:
    return tuple(_factor_levels(values, name))


def _mstate_inferred_levels(values: Sequence[Any]) -> list[str]:
    return [_as_character(level) for level in _factor_levels(values, "event")]


def _normalize_positive_scale(value: Any) -> float:
    scale = _finite_float(value, "scale")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return scale


def _scalar_or_vector_with_flag(values: Any, name: str) -> tuple[list[Any], bool]:
    try:
        return _materialize_1d(values, name), False
    except TypeError:
        if isinstance(values, str | bytes):
            raise
        return [values], True


def _scalar_or_vector(values: Any, name: str) -> list[Any]:
    values, _is_scalar = _scalar_or_vector_with_flag(values, name)
    return values


def _recycle_r_vector(values: list[Any], n: int, name: str) -> list[Any]:
    if not values:
        return []
    if len(values) == n:
        return values
    if n == 1:
        return [values[0]]
    return [values[idx % len(values)] for idx in range(n)]


def _survcheck_integer_labels(values: Any, name: str) -> list[int]:
    labels = _materialize_labels(values, name)
    if not labels:
        return []
    result: list[int] = []
    for value in labels:
        if isinstance(value, bool):
            break
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            break
        if not math.isfinite(numeric) or not numeric.is_integer():
            break
        result.append(int(numeric))
    else:
        return result
    return [code + 1 for code in _encode_labels(labels, name)]


def _hashable_group_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (_hashable_group_value(key), _hashable_group_value(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return tuple(_hashable_group_value(item) for item in value)
        return repr(value)
    return value


def _normalize_survfit_style(value: int | None, name: str) -> int:
    if value is None:
        return 1
    try:
        style = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be 1 or 2") from exc
    if style not in {1, 2}:
        raise ValueError(f"{name} must be 1 or 2")
    return style


def _normalize_survfit_conf_type(conf_type: str | None) -> str:
    if conf_type is None:
        return "log"
    if not isinstance(conf_type, str):
        raise TypeError("conf_type must be a string")
    value = conf_type.strip().lower().replace("_", "-")
    aliases = {
        "loglog": "log-log",
    }
    if value in aliases:
        return aliases[value]
    return _match_string_arg(
        value,
        "conf_type",
        ("plain", "log", "log-log", "logit", "arcsin", "none"),
        "conf_type must be 'plain', 'log', 'log-log', 'logit', 'arcsin', or 'none'",
    )


def _normalize_conf_level(conf_level: Any, name: str = "conf_level") -> float:
    try:
        value = float(conf_level)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _normalize_survfit_conf_level(conf_level: Any, conf_int: Any | None) -> float:
    if conf_int is None:
        return _normalize_conf_level(conf_level)
    if conf_level != 0.95:
        raise ValueError("use only one of conf_level or conf_int")
    return _normalize_conf_level(conf_int, "conf_int")


def _pop_dotted_keyword(
    kwargs: dict[str, Any],
    dotted: str,
    canonical: str,
    current: Any,
    default: Any,
) -> Any:
    if dotted not in kwargs:
        return current
    value = kwargs.pop(dotted)
    if current != default:
        raise ValueError(f"use only one of {canonical} or {dotted}")
    return value


def _normalize_start_time(start_time: Any | None) -> float | None:
    if start_time is None:
        return None
    try:
        value = float(start_time)
    except (TypeError, ValueError) as exc:
        raise TypeError("start_time must be a single numeric value") from exc
    if not math.isfinite(value):
        raise ValueError("start_time must be finite")
    return value


def _normalize_bool_option(value: Any | None, name: str) -> bool:
    if value is None:
        return False
    if not _is_bool_like(value):
        raise TypeError(f"{name} must be True or False")
    return bool(value)


def _normalize_bool_option_with_default(value: Any | None, name: str, default: bool) -> bool:
    if value is None:
        return default
    return _normalize_bool_option(value, name)


def _normalize_numeric_sequence_or_none(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must be numeric or array-like")
    try:
        return [_finite_float(value, name)]
    except TypeError:
        pass
    return [_finite_float(item, name) for item in _materialize_1d(value, name)]


def _normalize_optional_bool_option(value: Any | None, name: str) -> bool | None:
    if value is None:
        return None
    return _normalize_bool_option(value, name)


def _control_mapping(control: Any | None, name: str) -> dict[str, Any]:
    if control is None:
        return {}
    if isinstance(control, Mapping):
        items = control.items()
    else:
        items_method = getattr(control, "items", None)
        if callable(items_method):
            items = items_method()
        elif hasattr(control, "__dict__"):
            items = vars(control).items()
        else:
            raise TypeError(f"{name} must be a mapping")
    try:
        return {str(key): value for key, value in items}
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a mapping") from exc


def _pop_control_alias(
    control: dict[str, Any],
    aliases: tuple[str, ...],
    canonical: str,
    current: Any,
    default: Any,
) -> tuple[Any, str | None]:
    present = [alias for alias in aliases if alias in control]
    if not present:
        return current, None
    first = present[0]
    value = control.pop(first)
    for alias in present[1:]:
        other = control.pop(alias)
        if other != value:
            raise ValueError(f"use only one of control.{first} or control.{alias}")
    if current != default:
        raise ValueError(f"use only one of {canonical} or control.{first}")
    return value, first


def _pop_finite_control_value(
    control: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    positive: bool,
) -> float | None:
    present = [alias for alias in aliases if alias in control]
    if not present:
        return None
    first = present[0]
    value = control.pop(first)
    for alias in present[1:]:
        other = control.pop(alias)
        if other != value:
            raise ValueError(f"use only one of control.{first} or control.{alias}")
    numeric = _finite_float(value, f"control.{first}")
    if positive and numeric <= 0.0:
        raise ValueError(f"control.{first} must be positive")
    return numeric


def _reject_unknown_control_options(control: dict[str, Any], function_name: str) -> None:
    if control:
        unexpected = ", ".join(sorted(control))
        raise ValueError(f"{function_name} control has unsupported option(s): {unexpected}")


def _apply_coxph_control(
    control: Any | None,
    max_iter: int,
    eps: float | None,
    toler: float | None,
) -> tuple[int, float | None, float | None, bool]:
    values = _control_mapping(control, "coxph control")
    if not values:
        return max_iter, eps, toler, True

    max_iter_value, name = _pop_control_alias(
        values,
        ("iter.max", "iter_max", "max_iter"),
        "max_iter",
        max_iter,
        20,
    )
    if name is not None:
        max_iter = _integer_scalar(max_iter_value, f"control.{name}")

    eps_value, name = _pop_control_alias(values, ("eps",), "eps", eps, None)
    if name is not None:
        eps = _finite_float(eps_value, f"control.{name}")

    toler_value, name = _pop_control_alias(
        values,
        ("toler.chol", "toler_chol", "tol_chol", "toler"),
        "toler",
        toler,
        None,
    )
    if name is not None:
        toler = _finite_float(toler_value, f"control.{name}")

    timefix_value, name = _pop_control_alias(
        values,
        ("timefix", "time.fix", "time_fix"),
        "timefix",
        True,
        True,
    )
    fix_time = _normalize_bool_option(timefix_value, f"control.{name}") if name else True

    _pop_finite_control_value(values, ("toler.inf", "toler_inf"), positive=True)
    _pop_finite_control_value(values, ("outer.max", "outer_max"), positive=True)
    _reject_unknown_control_options(values, "coxph")
    return max_iter, eps, toler, fix_time


def _apply_survreg_control(
    control: Any | None,
    max_iter: int | None,
    eps: float | None,
    tol_chol: float | None,
) -> tuple[int | None, float | None, float | None]:
    values = _control_mapping(control, "survreg control")
    if not values:
        return max_iter, eps, tol_chol

    max_iter_value, name = _pop_control_alias(
        values,
        ("maxiter", "iter.max", "iter_max", "max_iter"),
        "max_iter",
        max_iter,
        None,
    )
    if name is not None:
        max_iter = _integer_scalar(max_iter_value, f"control.{name}")

    eps_value, name = _pop_control_alias(
        values,
        ("rel.tolerance", "rel_tolerance", "eps"),
        "eps",
        eps,
        None,
    )
    if name is not None:
        eps = _finite_float(eps_value, f"control.{name}")

    tol_chol_value, name = _pop_control_alias(
        values,
        ("toler.chol", "toler_chol", "tol_chol"),
        "tol_chol",
        tol_chol,
        None,
    )
    if name is not None:
        tol_chol = _finite_float(tol_chol_value, f"control.{name}")

    _pop_finite_control_value(values, ("debug",), positive=False)
    _pop_finite_control_value(values, ("outer.max", "outer_max"), positive=True)
    _reject_unknown_control_options(values, "survreg")
    return max_iter, eps, tol_chol


def _normalize_survfit_type(
    survfit_type: str | None,
    stype: int | None,
    ctype: int | None,
) -> _SurvfitComputation:
    if survfit_type is None:
        return _SurvfitComputation(
            stype=_normalize_survfit_style(stype, "stype"),
            ctype=_normalize_survfit_style(ctype, "ctype"),
        )
    if not isinstance(survfit_type, str):
        raise TypeError("survfit type must be a string")

    value = survfit_type.strip().lower().replace("_", "-")
    aliases = {
        "k": "kaplan-meier",
        "km": "kaplan-meier",
        "f": "fleming-harrington",
        "nelson-aalen": "fleming-harrington",
        "na": "fleming-harrington",
        "fh": "fh2",
    }
    normalized = aliases.get(value) or _match_string_arg(
        value,
        "survfit type",
        ("kaplan-meier", "fleming-harrington", "fh2"),
        "survfit type must be 'kaplan-meier', 'fleming-harrington', or 'fh2'",
    )
    if normalized == "kaplan-meier":
        return _SurvfitComputation(1, 1)
    if normalized == "fleming-harrington":
        return _SurvfitComputation(2, 1)
    return _SurvfitComputation(2, 2)


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _survfit_confidence_interval(
    survival: float,
    std_err: float,
    z: float,
    conf_type: str,
) -> tuple[float, float]:
    if std_err <= 0.0 or survival <= 0.0 or survival >= 1.0:
        bounded = _clamp_probability(survival)
        return bounded, bounded
    if conf_type == "plain":
        return (
            _clamp_probability(survival - z * std_err),
            _clamp_probability(survival + z * std_err),
        )
    if conf_type == "log":
        log_survival = math.log(survival)
        log_std_err = std_err / survival
        return (
            _clamp_probability(_safe_exp(log_survival - z * log_std_err)),
            _clamp_probability(_safe_exp(log_survival + z * log_std_err)),
        )
    if conf_type == "log-log":
        log_survival = math.log(survival)
        transformed_std_err = z * (std_err / survival) / log_survival
        log_neg_log_survival = math.log(-log_survival)
        return (
            _clamp_probability(_safe_exp(-_safe_exp(log_neg_log_survival - transformed_std_err))),
            _clamp_probability(_safe_exp(-_safe_exp(log_neg_log_survival + transformed_std_err))),
        )
    if conf_type == "logit":
        logit_survival = math.log(survival / (1.0 - survival))
        transformed_std_err = z * std_err / (survival * (1.0 - survival))
        return (
            _clamp_probability(1.0 - 1.0 / (1.0 + _safe_exp(logit_survival - transformed_std_err))),
            _clamp_probability(1.0 - 1.0 / (1.0 + _safe_exp(logit_survival + transformed_std_err))),
        )
    if conf_type == "arcsin":
        angle = math.asin(math.sqrt(survival))
        transformed_std_err = 0.5 * z * std_err / math.sqrt(survival * (1.0 - survival))
        return (
            _clamp_probability(math.sin(max(angle - transformed_std_err, 0.0)) ** 2),
            _clamp_probability(math.sin(min(angle + transformed_std_err, math.pi / 2.0)) ** 2),
        )
    raise AssertionError("conf_type is validated before confidence intervals are computed")


def _r_numeric_vector(values: Any, name: str) -> list[float]:
    result: list[float] = []
    for value in _scalar_or_vector(values, name):
        if _is_missing_value(value):
            result.append(math.nan)
        else:
            try:
                result.append(float(value))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be numeric") from exc
    return result


def _aeq_times(
    *columns: Sequence[float], tolerance: float | None = None
) -> tuple[list[float], ...]:
    """R's ``aeqSurv`` on one or two time columns (``time``, or ``start``/``stop``).

    This is the package's only timefix path: the Rust ``aeq_surv`` kernel snaps
    near-tied times exactly as R does, and raises R's "an interval has effective
    length 0" error when a ``(start, stop]`` interval collapses.
    """

    if len(columns) not in {1, 2}:
        raise ValueError("_aeq_times takes one or two time columns")
    first = [float(value) for value in columns[0]]
    if len(columns) == 1:
        return (list(_core.aeq_surv(first, None, tolerance).time),)
    second = [float(value) for value in columns[1]]
    result = _core.aeq_surv(first, second, tolerance)
    return list(result.time), list(result.time2 or [])


def _survdiff_timefix_values(times: list[float], timefix: bool) -> list[float]:
    """Alias of :func:`_aeq_times` for one column (kept for the modules that import it)."""

    if not timefix:
        return times
    return _aeq_times(times)[0]


def _timefix_vectors(*vectors: list[float]) -> tuple[list[float], ...]:
    """Alias of :func:`_aeq_times` (kept for the modules that import it)."""

    return _aeq_times(*vectors)


def _concordance_core_time_values(
    times: list[float],
    timefix: bool,
) -> tuple[list[float], dict[float, float] | None]:
    if timefix or len(times) < 2:
        return times, None

    unique_times = sorted(set(times))
    if len(unique_times) < 2:
        return times, None

    step = _SURVFIT_TIME_EPSILON * 2.0
    display_by_core_time = {index * step: value for index, value in enumerate(unique_times)}
    core_by_display_time = {value: index * step for index, value in enumerate(unique_times)}
    return [core_by_display_time[value] for value in times], display_by_core_time


def _normalize_predict_type(predict_type: Any, *, survreg: bool) -> str:
    if not isinstance(predict_type, str):
        raise TypeError("predict type must be a string")
    value = predict_type.strip().lower().replace("-", "_")
    aliases = {
        "lp": "lp",
        "link": "lp",
        "linear": "lp",
        "linear_predictor": "lp",
        "linear_predictors": "lp",
        "response": "response",
        "risk": "risk",
        "relative_risk": "risk",
        "terms": "terms",
        "term": "terms",
        "surv": "survival",
        "survival": "survival",
        "survival_curve": "survival",
        "expected": "expected",
        "quantile": "quantile",
        "quantiles": "quantile",
        "uquantile": "uquantile",
        "uquantiles": "uquantile",
    }
    if value in aliases:
        return aliases[value]

    choices = (
        ("response", "lp", "terms", "quantile", "uquantile")
        if survreg
        else ("lp", "risk", "expected", "terms", "survival")
    )
    matches = [choice for choice in choices if choice.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("predict type is ambiguous; use a full type name")
    raise ValueError(
        "predict type must be 'lp', 'response', 'risk', 'terms', "
        "'survival', 'expected', 'quantile', or 'uquantile'"
    )


def _normalize_residual_type(residual_type: Any) -> str:
    if not isinstance(residual_type, str):
        raise TypeError("residuals type must be a string")
    value = residual_type.strip().lower().replace("-", "_")
    aliases = {
        "martingale": "martingale",
        "mart": "martingale",
        "deviance": "deviance",
        "dev": "deviance",
        "score": "score",
        "dfbeta": "dfbeta",
        "dfbetas": "dfbetas",
        "schoenfeld": "schoenfeld",
        "sch": "schoenfeld",
        "scaledsch": "scaledsch",
        "scaled_sch": "scaledsch",
        "scaledschoenfeld": "scaledsch",
        "scaled_schoenfeld": "scaledsch",
        "partial": "partial",
        "partials": "partial",
    }
    if value in aliases:
        return aliases[value]

    choices = (
        "martingale",
        "deviance",
        "score",
        "dfbeta",
        "dfbetas",
        "schoenfeld",
        "scaledsch",
        "partial",
    )
    matches = [choice for choice in choices if choice.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("residuals type is ambiguous; use a full type name")
    raise ValueError(
        "residuals type must be 'martingale', 'deviance', 'score', "
        "'dfbeta', 'dfbetas', 'schoenfeld', 'scaledsch', or 'partial'"
    )


def _normalize_survreg_residual_type(residual_type: Any) -> str:
    if not isinstance(residual_type, str):
        raise TypeError("residuals type must be a string")
    value = residual_type.strip().lower().replace("-", "_")
    aliases = {"dfb": "dfbeta"}
    if value in aliases:
        return aliases[value]

    choices = (
        "response",
        "deviance",
        "working",
        "ldcase",
        "ldresp",
        "ldshape",
        "dfbeta",
        "dfbetas",
        "matrix",
    )
    if value in choices:
        return value
    matches = [choice for choice in choices if choice.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("residuals type is ambiguous; use a full type name")
    raise ValueError(
        "residuals type must be 'response', 'deviance', 'working', "
        "'ldcase', 'ldresp', 'ldshape', 'dfbeta', 'dfbetas', or 'matrix'"
    )


def _normalize_survreg_distribution(distribution: Any | None) -> str | None:
    if distribution is None:
        return None
    if not isinstance(distribution, str):
        raise TypeError("distribution must be a string")
    value = distribution.strip().lower().replace("_", "-")
    aliases = {
        "normal": "gaussian",
        "log-normal": "lognormal",
        "loggaussian": "lognormal",
        "log-gaussian": "lognormal",
        "log-logistic": "loglogistic",
        "extreme": "extreme_value",
        "extreme value": "extreme_value",
        "extreme-value": "extreme_value",
        "extremevalue": "extreme_value",
        "student": "t",
        "student-t": "t",
    }
    if value in aliases:
        return aliases[value]
    message = (
        "distribution must be one of weibull, exponential, rayleigh, extreme, "
        "gaussian, logistic, loggaussian, lognormal, loglogistic, or t"
    )
    matched = _match_string_arg(
        value,
        "distribution",
        (
            "weibull",
            "exponential",
            "rayleigh",
            "extreme_value",
            "gaussian",
            "logistic",
            "loggaussian",
            "lognormal",
            "loglogistic",
            "t",
        ),
        message,
    )
    return "lognormal" if matched == "loggaussian" else matched


def _collapse_is_false(collapse: Any) -> bool:
    return collapse is False or collapse is None


def _result_is_matrix(values: list[Any]) -> bool:
    return bool(values) and isinstance(values[0], list | tuple)


def _model_residual_weights(fit: Any, n: int) -> list[float]:
    weights = getattr(fit, "weights", None)
    if weights is None:
        return [1.0] * n
    result = [float(value) for value in weights]
    if len(result) != n:
        raise ValueError("fitted model weights do not match residual length")
    return result


def _weight_residual_result(values: Any, weights: list[float]) -> Any:
    rows = list(values)
    if len(rows) != len(weights):
        raise ValueError("weights must have the same length as residuals")
    if _result_is_matrix(rows):
        return [
            [float(value) * weights[row_idx] for value in row] for row_idx, row in enumerate(rows)
        ]
    return [float(value) * weights[row_idx] for row_idx, value in enumerate(rows)]


def _collapse_result(
    values: Any,
    collapse: Any,
    n: int,
    *,
    value_name: str,
    matrix_name: str,
) -> Any:
    labels = _materialize_labels(collapse, "collapse")
    if len(labels) != n:
        raise ValueError(f"collapse must have the same length as {value_name}")
    rows = list(values)
    if len(rows) != n:
        raise ValueError(f"collapse must have the same length as {value_name}")

    groups: dict[Any, int] = {}
    if _result_is_matrix(rows):
        collapsed: list[list[float]] = []
        for row, label in zip(rows, labels, strict=True):
            try:
                group_idx = groups.setdefault(label, len(groups))
            except TypeError as exc:
                raise TypeError("collapse contains unhashable labels") from exc
            if group_idx == len(collapsed):
                collapsed.append([0.0] * len(row))
            if len(row) != len(collapsed[group_idx]):
                raise ValueError(f"{matrix_name} rows must be rectangular")
            for col_idx, value in enumerate(row):
                collapsed[group_idx][col_idx] += float(value)
        return collapsed

    collapsed_vector: list[float] = []
    for value, label in zip(rows, labels, strict=True):
        try:
            group_idx = groups.setdefault(label, len(groups))
        except TypeError as exc:
            raise TypeError("collapse contains unhashable labels") from exc
        if group_idx == len(collapsed_vector):
            collapsed_vector.append(0.0)
        collapsed_vector[group_idx] += float(value)
    return collapsed_vector


def _collapse_residual_result(values: Any, collapse: Any, n: int) -> Any:
    if _collapse_is_false(collapse):
        return values
    return _collapse_result(
        values,
        collapse,
        n,
        value_name="residuals",
        matrix_name="residual matrix",
    )


def _collapse_prediction_result(values: Any, collapse: Any) -> Any:
    if _collapse_is_false(collapse):
        return values
    rows = list(values)
    return _collapse_result(
        rows,
        collapse,
        len(rows),
        value_name="predictions",
        matrix_name="prediction matrix",
    )


def _collapse_prediction_se(values: Any, collapse: Any) -> Any:
    if _collapse_is_false(collapse):
        return values
    rows = list(values)
    if _result_is_matrix(rows):
        squared_rows = [[float(value) * float(value) for value in row] for row in rows]
        collapsed = _collapse_prediction_result(squared_rows, collapse)
        return [[math.sqrt(max(float(value), 0.0)) for value in row] for row in collapsed]
    squared = [float(value) * float(value) for value in rows]
    return [
        math.sqrt(max(float(value), 0.0))
        for value in _collapse_prediction_result(squared, collapse)
    ]


def _quadratic_form(values: list[float], variance: list[list[float]]) -> float:
    return sum(
        values[row_idx] * variance[row_idx][col_idx] * values[col_idx]
        for row_idx in range(len(values))
        for col_idx in range(len(values))
    )
