"""Input coercion, option normalisation, and small shared numeric helpers."""

from __future__ import annotations

import math
import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from itertools import compress
from operator import index
from typing import Any, cast

import numpy as np

from .. import _survival as _core

_SURV_TYPES = ("right", "left", "interval", "counting", "interval2", "mstate")
_SURV_RESPONSE_TYPES = (*_SURV_TYPES[:-1], "mright", "mcounting")
_PACKAGE_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _numeric_design_matrix(
    x: Any, n: int, *, vector: bool = False, empty: bool = False, invalid: str = "Invalid X matrix"
) -> np.ndarray:
    """Prepared numeric columns without per-row lists at the native boundary."""
    if isinstance(x, Mapping):
        values = np.column_stack(list(x.values())) if x else np.empty((n, 0))
    else:
        values = x.to_numpy() if hasattr(x, "to_numpy") else x
    matrix = np.asarray(values)
    if matrix.ndim == 1 and matrix.size == 0 and empty:
        matrix = np.empty((n, 0))
    elif matrix.ndim == 1 and vector:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(invalid)
    if matrix.shape[0] != n:
        raise ValueError("x and y have different numbers of rows")
    if matrix.dtype.kind not in "biuf":
        raise TypeError("x must be a numeric matrix")
    return matrix.astype(np.float64, copy=False)


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
    if isinstance(values, _core.TcutResult):
        return list(values.values)
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


class _RFactorVector(Sequence[Any]):
    """Iterable factor values with level metadata preserved across reticulate."""

    def __init__(self, values: Any, levels: Any):
        self._values = tuple(values)
        self.categories = tuple(levels)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, item: int | slice) -> Any:
        return self._values[item]


def _curve_matrix(values: list[float] | list[list[float]]) -> list[list[float]]:
    """View homogeneous curve columns as a matrix, wrapping a vector as one column.

    Result containers declare either a vector or a matrix. The first row chooses
    that shape; no matrix copy is needed at the native read-only boundary.
    """
    if values and isinstance(values[0], list):
        return cast(list[list[float]], values)
    return [[value] for value in cast(list[float], values)]


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


def _start_time_value(start_time: Any | None) -> float | None:
    """The ``start.time`` argument of the survfit methods: ``None`` or one finite number."""

    if start_time is None:
        return None
    if isinstance(start_time, bool | str):
        raise ValueError("start.time must be a single numeric value")
    try:
        value = float(start_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("start.time must be a single numeric value") from exc
    if not math.isfinite(value):
        raise ValueError("start.time must be a single numeric value")
    return value


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


def _mstate_categories(values: Any) -> Any | None:
    return _categories(values)


def _r_factor_levels(values: Sequence[Any]) -> list[Any]:
    """The levels ``as.factor`` gives ``values``: R factor levels when present, else sorted."""

    categories = _mstate_categories(values)
    present = {value for value in values if not _is_missing_value(value)}
    if categories is not None:
        return [level for level in _materialize_1d(categories, "levels") if level in present]
    try:
        return sorted(present)
    except TypeError:
        return sorted(present, key=str)


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


# R's getOption("na.action"), the na.action of every model function that does not
# name its own
_DEFAULT_NA_ACTION = "na.omit"


def _normalize_na_action(na_action: str | None) -> str:
    """``"fail"``, ``"omit"``, ``"exclude"`` or ``"pass"`` for an R na.action name;
    ``None`` is R's ``na.action = NULL``, which applies none.  ``exclude`` drops rows
    exactly as ``omit`` does; ``naresid``/``napredict`` pad them back as ``NA``."""

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
        "exclude": "exclude",
        "na_exclude": "exclude",
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


def _float_or_nan(value: Any) -> float:
    """``float(value)``, with NaN for a missing value (R's ``NA``)."""

    return math.nan if _is_missing_value(value) else float(value)


def _floats_or_nan(values: Sequence[Any]) -> list[float]:
    """:func:`_float_or_nan` of each value, as fast as ``float`` when none is missing."""

    try:
        return list(map(float, values))
    except TypeError:
        return list(map(_float_or_nan, values))


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


def _numeric_ndarray(values: Any, *, ndim: int = 1) -> np.ndarray | None:
    """*values* as an array of *ndim* dimensions (one by default), from a plain ndarray or column
    whose dtype kind is logical, integer or double, else ``None``.

    Masked arrays, object, string, datetime and nullable extension columns (which
    ``to_numpy`` turns into object arrays) and factor-like columns get ``None``: they keep
    the per-element paths that know masks, ``None``, ``pd.NA``, ``NaT`` and declared levels.
    """

    if isinstance(values, np.ndarray):
        if isinstance(values, np.ma.MaskedArray):
            return None
        array = values
    elif hasattr(values, "to_numpy") and hasattr(values, "dtype"):
        if _categories(values) is not None:
            return None
        array = values.to_numpy()
        if not isinstance(array, np.ndarray):
            return None
    else:
        return None
    if array.ndim != ndim or array.dtype.kind not in "biuf":
        return None
    return array


def _missing_row_indices(columns: list[tuple[str, Any]], n: int) -> set[int]:
    missing: set[int] = set()
    for name, values in columns:
        array = _numeric_ndarray(values)
        if array is None:
            array = _numeric_ndarray(values, ndim=2)
        if array is not None:
            if len(array) != n:
                raise ValueError(f"{name} must have length {n}")
            if array.dtype.kind == "f":
                mask = np.isnan(array)
                if array.ndim == 2:
                    mask = mask.any(axis=1)
                missing.update(np.flatnonzero(mask).tolist())
            continue
        materialized = _coerce_array_like(values, name)
        if len(materialized) != n:
            raise ValueError(f"{name} must have length {n}")
        try:
            # a numeric column is missing only where it is NaN
            missing.update(compress(range(n), map(math.isnan, materialized)))
        except (TypeError, OverflowError):
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


def _warn_outside_package(message: str, category: type[Warning] = UserWarning) -> None:
    """``warnings.warn(message, category)`` reported at the first caller outside this
    package.

    Shared helpers run at a different depth under each public function, so no fixed
    ``stacklevel`` fits them all (``skip_file_prefixes`` needs Python 3.12).
    """

    frame = sys._getframe(1)
    level = 2
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_PACKAGE_PREFIX):
        frame = frame.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)


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


def _rows_of(source: Any, kept: list[Any]) -> Any:
    """``source[rows]`` from the values at the kept rows, with what R's ``[`` methods
    keep: a factor's levels, and a ``tcut``'s cutpoints and labels (``[.tcut``)."""

    if isinstance(source, _core.TcutResult):
        # scale 1 keeps the already scaled values and cutpoints
        return _core.tcut(kept, list(source.cutpoints), list(source.labels), 1.0)
    categories = _mstate_categories(source)
    return kept if categories is None else _RFactorVector(kept, categories)


def _subset_sequence(values: Any, indices: list[int], name: str) -> Any:
    materialized = _coerce_array_like(values, name)
    if indices and max(indices) >= len(materialized):
        raise ValueError(f"{name} must have enough rows for subset")
    return _rows_of(values, [materialized[idx] for idx in indices])


def _subset_optional_sequence(
    values: Any | None,
    indices: list[int],
    name: str,
) -> list[Any] | None:
    if values is None:
        return None
    return _subset_sequence(values, indices, name)


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


def _as_matrix_rows(
    values: Any,
    name: str,
    *,
    allow_empty_columns: bool,
    convert: Callable[[Any], float] = float,
) -> list[list[float]]:
    rows = _coerce_array_like(values, name)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    if not isinstance(rows[0], list | tuple):
        return [[convert(value)] for value in rows]

    width = len(rows[0])
    if width == 0 and not allow_empty_columns:
        raise ValueError(f"{name} must have at least one column")
    matrix = [[convert(value) for value in row] for row in rows]
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return matrix


def _label_levels(values: Sequence[Any], name: str) -> tuple[Any, ...]:
    labels: dict[Any, None] = {}
    for value in values:
        try:
            labels.setdefault(value, None)
        except TypeError as exc:
            raise TypeError(f"{name} contains unhashable labels") from exc
    return tuple(labels)


def _encode_labels(values: Sequence[Any], name: str) -> list[int]:
    labels = {value: idx for idx, value in enumerate(_label_levels(values, name))}
    return [labels[value] for value in values]


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
    for finite_number in finite:
        neg, kpower, nsig = (
            _r_scientific(finite_number, digits) if finite_number != 0.0 else (False, 0, 1)
        )
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
    if math.isnan(numeric):
        return (1, str(value))
    return (0, numeric)


def _factor_levels(values: Any, name: str = "values") -> list[Any]:
    """R's ``levels(factor(x))``: the declared categories, else the sorted unique values.

    Missing values never form a level; the returned levels are the raw values so
    that codes can be looked up, ``_as_character`` renders them as R labels them.
    """

    declared = _categories(values)
    if declared is not None:
        return [level for level in declared if not _is_missing_value(level)]
    array = _numeric_ndarray(values)
    if array is not None:
        return _numeric_factor(array)[1].tolist()
    try:
        unique = dict.fromkeys(_materialize_labels(values, name))
    except TypeError as exc:
        raise TypeError(f"{name} contains unhashable labels") from exc
    return sorted((value for value in unique if not _is_missing_value(value)), key=_r_sort_key)


def _numeric_factor(array: np.ndarray) -> tuple[list[int | None], np.ndarray]:
    """``factor(x)`` of a numeric or logical array: codes (``None`` where NaN) and the
    sorted distinct values, which is R's ``sort(unique(x))`` level order."""

    present = ~np.isnan(array) if array.dtype.kind == "f" else None
    observed = array if present is None else array[present]
    levels, inverse = np.unique(observed, return_inverse=True)
    if levels.dtype.kind == "f":
        levels = levels + 0.0  # -0 and 0 are one level, labelled "0"
    if present is None or present.all():
        return inverse.tolist(), levels
    full = np.zeros(len(array), dtype=np.int64)
    full[present] = inverse
    codes = cast(list[int | None], full.tolist())
    for row in np.flatnonzero(~present).tolist():
        codes[row] = None
    return codes, levels


def _factor(values: Any, name: str = "values") -> tuple[list[int | None], list[str]]:
    """R's ``factor(x)`` as zero-based codes (``None`` for ``NA``) and level labels."""

    array = _numeric_ndarray(values)
    if array is not None:
        numeric_codes, numeric_levels = _numeric_factor(array)
        return numeric_codes, [_as_character(level) for level in numeric_levels.tolist()]
    levels = _factor_levels(values, name)
    index = {level: code for code, level in enumerate(levels)}
    materialized = _materialize_labels(values, name)
    codes: list[int | None] = list(map(index.get, materialized))
    if None in codes:
        for value, code in zip(materialized, codes, strict=True):
            if code is None and not _is_missing_value(value):
                raise ValueError(f"{name} contains a value outside the declared categories")
    return codes, [_as_character(level) for level in levels]


# Older spellings kept for the modules that still import them; each is an alias of
# the single implementation above.
_strata_value_label = _as_character
_strata_level_sort_key = _r_sort_key
_mstate_event_label = _as_character


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


def _normalize_conf_level(conf_level: Any, name: str = "conf_level") -> float:
    try:
        value = float(conf_level)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _coefficient_selection(parm: Any, names: list[str]) -> list[int]:
    """``confint``'s ``parm``: coefficient names or 1-based positions, as 0-based indices
    (every coefficient when ``None``)."""

    if parm is None:
        return list(range(len(names)))
    values = [parm] if isinstance(parm, str | int) else list(_materialize_1d(parm, "parm"))
    indices: list[int] = []
    for value in values:
        if isinstance(value, str):
            if value not in names:
                raise ValueError(f"unknown coefficient name {value!r}")
            indices.append(names.index(value))
        else:
            idx = _integer_scalar(value, "parm") - 1
            if idx < 0 or idx >= len(names):
                raise IndexError("parm index out of range")
            indices.append(idx)
    return indices


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
