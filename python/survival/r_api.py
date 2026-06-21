from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, product
from operator import index
from statistics import NormalDist
from typing import Any

from . import _survival as _core

__all__ = [
    "Surv",
    "CoxSurvfitResult",
    "CoxBaseHazardResult",
    "CoxPHDetailResult",
    "CoxZPHResult",
    "PredictResult",
    "SurvfitResult",
    "aic",
    "as_data_frame",
    "basehaz",
    "anova",
    "coef",
    "coef_names",
    "confint",
    "concordance",
    "coxph",
    "coxph_detail",
    "cox_zph",
    "df_residual",
    "degrees_freedom",
    "bic",
    "extract_aic",
    "fitted",
    "is_surv",
    "loglik",
    "model_formula",
    "model_summary",
    "model_frame",
    "model_matrix",
    "model_weights",
    "nobs",
    "predict",
    "residuals",
    "survdiff",
    "survfit",
    "survreg",
    "vcov",
]

_EXP_CLAMP_MIN = -745.0
_EXP_CLAMP_MAX = 709.0
_SURVFIT_TIME_EPSILON = 1e-9
_VARIANCE_SCALE_FLOOR = 1e-12
_COX_DFBETAS_SCALE_FLOOR = 1e-10
_CONCORDANCE_RISK_TIE_FLOOR = 1e-10
_SURV_TYPES = ("right", "left", "interval", "counting", "interval2")


@dataclass(frozen=True)
class _CovariateTerm:
    column: str
    categorical: bool = False
    transform: str | None = None
    arithmetic: str | None = None


@dataclass(frozen=True)
class _InteractionTerm:
    factors: tuple[_CovariateTerm, ...]


_CovariateSpec = _CovariateTerm | _InteractionTerm


@dataclass(frozen=True)
class _FormulaTerms:
    covariates: list[_CovariateSpec]
    strata: list[str]
    offsets: list[_CovariateTerm]
    clusters: list[str]
    intercept: bool = True


@dataclass(frozen=True)
class _NumericDesignTerm:
    term: _CovariateTerm


@dataclass(frozen=True)
class _CategoricalDesignTerm:
    term: _CovariateTerm
    levels: tuple[Any, ...]


_SingleDesignTerm = _NumericDesignTerm | _CategoricalDesignTerm


@dataclass(frozen=True)
class _InteractionDesignTerm:
    factors: tuple[_SingleDesignTerm, ...]


_DesignTerm = _SingleDesignTerm | _InteractionDesignTerm


@dataclass(frozen=True)
class _FormulaDesign:
    response: _SurvResponseSpec
    covariates: tuple[_DesignTerm, ...]
    offsets: tuple[_CovariateTerm, ...]
    strata: tuple[str, ...] = ()
    strata_levels: tuple[Any, ...] = ()
    intercept: bool = False


@dataclass(frozen=True)
class _FormulaFit:
    fit: Any
    design: _FormulaDesign | None
    formula: str | None = None
    case_weights: list[float] | None = None
    robust_variance: list[list[float]] | None = None
    naive_variance: list[list[float]] | None = None
    cluster: list[Any] | None = None
    id_values: list[Any] | None = None
    x_matrix: list[list[float]] | None = None
    y_response: Surv | None = None
    model_frame: dict[str, Any] | None = None
    score_values: list[float] | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "id" and self.id_values is not None:
            return self.id_values
        if name == "x" and self.x_matrix is not None:
            return self.x_matrix
        if name == "y" and self.y_response is not None:
            return self.y_response
        if name == "model" and self.model_frame is not None:
            return self.model_frame
        if name == "weights" and self.case_weights is not None:
            return self.case_weights
        if name == "score" and self.score_values is not None:
            return self.score_values
        if name == "scaled_schoenfeld_residuals" and hasattr(self.fit, "schoenfeld_residuals"):
            return self._cox_scaled_schoenfeld_residuals
        if name == "dfbeta" and hasattr(self.fit, "score_residuals"):
            return self._cox_dfbeta
        if name == "dfbetas" and hasattr(self.fit, "score_residuals"):
            return self._cox_dfbetas
        return getattr(self.fit, name)

    def _cox_scaled_schoenfeld_residuals(self) -> list[list[float]]:
        raw = [[float(value) for value in row] for row in self.fit.schoenfeld_residuals()]
        return _cox_scaled_schoenfeld_from_raw(self, raw)

    def _cox_dfbeta(self) -> list[list[float]]:
        return _cox_dfbeta_from_score_residuals(self, scaled=False)

    def _cox_dfbetas(self) -> list[list[float]]:
        return _cox_dfbeta_from_score_residuals(self, scaled=True)

    @property
    def information_matrix(self) -> list[list[float]]:
        if self.robust_variance is not None:
            return self.robust_variance
        matrix = getattr(self.fit, "information_matrix", None)
        if matrix is not None:
            return matrix
        matrix = getattr(self.fit, "variance_matrix", None)
        if matrix is not None:
            return matrix
        raise AttributeError("wrapped fit does not expose a variance matrix")

    @property
    def variance_matrix(self) -> list[list[float]]:
        return self.information_matrix

    @property
    def naive_information_matrix(self) -> list[list[float]] | None:
        return self.naive_variance

    @property
    def naive_var(self) -> list[list[float]] | None:
        return self.naive_variance

    @property
    def robust(self) -> bool:
        return self.robust_variance is not None


@dataclass(frozen=True)
class _SurvResponseSpec:
    arguments: tuple[str, ...]
    columns: tuple[str, ...]
    type: str | None
    origin: float = 0.0


@dataclass(frozen=True)
class _ResponseOperand:
    column: str | None = None
    value: Any = None


@dataclass(frozen=True)
class ConcordanceResult:
    concordance: float | list[float]
    n: int
    n_event: int
    reverse: bool = False
    concordant: float | list[float] = 0.0
    comparable: float | list[float] = 0.0
    ranks: list[dict[str, float]] | list[list[dict[str, float]] | None] | None = None
    dfbeta: list[float] | list[list[float] | None] | None = None
    influence: list[list[float]] | list[list[list[float]] | None] | None = None
    variance: float | list[float | None] | None = None
    score_names: list[str] | None = None

    @property
    def c_index(self) -> float | list[float]:
        return self.concordance

    @property
    def var(self) -> float | list[float | None] | None:
        return self.variance


@dataclass(frozen=True)
class PredictResult:
    fit: Any
    se_fit: Any

    def __iter__(self):
        yield self.fit
        yield self.se_fit

    @property
    def predictions(self) -> Any:
        return self.fit

    @property
    def se(self) -> Any:
        return self.se_fit


@dataclass(frozen=True)
class CoxZPHResult:
    variable_names: list[str]
    chi2_values: list[float]
    df: list[int]
    p_values: list[float]
    x: list[float]
    time: list[float]
    y: list[list[float]]
    var: list[list[float]]
    transform: str
    global_chi2: float | None
    global_df: int | None
    global_p_value: float | None

    @property
    def table(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = [
            {
                "name": name,
                "chisq": self.chi2_values[idx],
                "df": self.df[idx],
                "p": self.p_values[idx],
            }
            for idx, name in enumerate(self.variable_names)
        ]
        if self.global_chi2 is not None:
            rows.append(
                {
                    "name": "GLOBAL",
                    "chisq": self.global_chi2,
                    "df": self.global_df if self.global_df is not None else 0,
                    "p": self.global_p_value if self.global_p_value is not None else 1.0,
                }
            )
        return rows


@dataclass(frozen=True)
class CoxPHDetailResult:
    time: list[float]
    nevent: list[int]
    nrisk: list[int]
    means: list[list[float]]
    score: list[list[float]]
    imat: list[list[list[float]]]
    hazard: list[float]
    varhaz: list[float]
    wtrisk: list[float]
    x: list[list[float]]
    y: list[list[float]]
    strata: dict[int, int] | None = None
    riskmat: list[list[int]] | None = None
    weights: list[float] | None = None
    nevent_wt: list[float] | None = None
    nrisk_wt: list[float] | None = None
    sortorder: list[int] | None = None

    @property
    def n_event(self) -> list[int]:
        return self.nevent

    @property
    def n_risk(self) -> list[int]:
        return self.nrisk

    @property
    def var_hazard(self) -> list[float]:
        return self.varhaz

    @property
    def cumulative_hazard(self) -> list[float]:
        total = 0.0
        values: list[float] = []
        for increment in self.hazard:
            total += increment
            values.append(total)
        return values

    def times(self) -> list[float]:
        return self.time

    def hazards(self) -> list[float]:
        return self.hazard

    def cumulative_hazards(self) -> list[float]:
        return self.cumulative_hazard

    def n_risk_at_times(self) -> list[int]:
        return self.nrisk

    def schoenfeld_residuals(self) -> list[list[float]]:
        return self.score


@dataclass(frozen=True)
class CoxBaseHazardResult:
    time: list[float]
    cumhaz: list[float] | list[list[float]]
    strata: list[int] | None = None
    centered: bool = True
    curve_strata: list[int] | None = None

    def __iter__(self):
        yield self.time
        yield self.cumhaz

    @property
    def hazard(self) -> list[float] | list[list[float]]:
        return self.cumhaz

    @property
    def cumulative_hazard(self) -> list[float] | list[list[float]]:
        return self.cumhaz


@dataclass(frozen=True)
class CoxSurvfitResult:
    time: list[float]
    surv: list[list[float]]
    cumhaz: list[list[float]]
    linear_predictors: list[float]
    centered: bool = True
    strata: list[int] | None = None
    start_time: float | None = None
    std_err: list[list[float]] = field(default_factory=list)
    std_chaz: list[list[float]] = field(default_factory=list)
    conf_lower: list[list[float]] = field(default_factory=list)
    conf_upper: list[list[float]] = field(default_factory=list)
    model: dict[str, Any] | None = None

    def __iter__(self):
        yield self.time
        yield self.surv

    @property
    def curves(self) -> list[list[float]]:
        return self.surv

    @property
    def estimate(self) -> list[list[float]]:
        return self.surv

    @property
    def cumulative_hazard(self) -> list[list[float]]:
        return self.cumhaz

    @property
    def cumulative_hazard_std_err(self) -> list[list[float]]:
        return self.std_chaz


@dataclass(frozen=True)
class SurvfitResult:
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    estimate: list[float]
    std_err: list[float]
    conf_lower: list[float]
    conf_upper: list[float]
    cumhaz: list[float]
    std_chaz: list[float]
    n_enter: list[float] | None = None
    model: dict[str, Any] | None = None

    @property
    def surv(self) -> list[float]:
        return self.estimate

    @property
    def cumulative_hazard(self) -> list[float]:
        return self.cumhaz

    @property
    def cumulative_hazard_std_err(self) -> list[float]:
        return self.std_chaz


@dataclass(frozen=True)
class TurnbullSurvfitResult:
    time_points: list[float]
    survival: list[float]
    survival_lower: list[float]
    survival_upper: list[float]
    n_iter: int
    converged: bool
    model: dict[str, Any] | None = None


@dataclass(frozen=True)
class _SurvfitComputation:
    stype: int
    ctype: int

    @property
    def is_kaplan_meier(self) -> bool:
        return self.stype == 1 and self.ctype == 1


def _coerce_array_like(values: Any, name: str) -> list[Any]:
    if values is None:
        raise ValueError(f"{name} is required")
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


def _subset_sequence(values: Any, indices: list[int], name: str) -> list[Any]:
    materialized = _coerce_array_like(values, name)
    if indices and max(indices) >= len(materialized):
        raise ValueError(f"{name} must have enough rows for subset")
    return [materialized[idx] for idx in indices]


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


def _as_matrix_rows(
    values: Any,
    name: str,
    *,
    allow_empty_columns: bool,
) -> list[list[float]]:
    if values is None:
        raise ValueError(f"{name} is required")
    if hasattr(values, "to_numpy"):
        values = values.to_numpy().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()

    rows = list(values)
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


def _cox_centered_design_rank(
    rows: list[list[float]],
    weights: list[float] | None,
    tolerance: float,
) -> int:
    width = len(rows[0]) if rows else 0
    if width == 0:
        return 0

    case_weights = [1.0] * len(rows) if weights is None else weights
    active = [
        idx for idx, weight in enumerate(case_weights) if math.isfinite(weight) and weight > 0.0
    ]
    if not active:
        return width

    total_weight = math.fsum(case_weights[idx] for idx in active)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return width

    means = [
        math.fsum(case_weights[idx] * rows[idx][col_idx] for idx in active) / total_weight
        for col_idx in range(width)
    ]
    columns: list[list[float]] = []
    for col_idx in range(width):
        column = [
            (rows[idx][col_idx] - means[col_idx]) * math.sqrt(case_weights[idx]) for idx in active
        ]
        scale = max((abs(value) for value in column), default=0.0)
        columns.append([value / scale for value in column] if scale > 0.0 else column)

    rank = 0
    basis: list[list[float]] = []
    pivot_tolerance = max(abs(tolerance), 1e-12) * max(len(active), width, 1)
    for column in columns:
        residual = list(column)
        for basis_column in basis:
            projection = math.fsum(
                value * basis_value
                for value, basis_value in zip(residual, basis_column, strict=True)
            )
            residual = [
                value - projection * basis_value
                for value, basis_value in zip(residual, basis_column, strict=True)
            ]
        norm = math.sqrt(math.fsum(value * value for value in residual))
        if norm > pivot_tolerance:
            basis.append([value / norm for value in residual])
            rank += 1
    return rank


def _check_cox_design_full_rank(
    rows: list[list[float]],
    weights: list[float] | None,
    tolerance: float,
) -> None:
    width = len(rows[0]) if rows else 0
    if width == 0:
        return
    if _cox_centered_design_rank(rows, weights, tolerance) < width:
        raise ValueError(
            "coxph design matrix is singular; use singular_ok=True to allow dependent covariates"
        )


def _column(data: Any, name: str) -> list[Any]:
    if data is None:
        raise ValueError("data is required when using a formula")
    if isinstance(data, Mapping):
        try:
            return _materialize_1d(data[name], name)
        except KeyError as exc:
            raise KeyError(f"column {name!r} not found in data") from exc
    try:
        return _materialize_1d(data[name], name)
    except Exception as exc:
        raise KeyError(f"column {name!r} not found in data") from exc


def _formula_name(name: str) -> tuple[str, bool]:
    name = name.strip()
    if name.startswith("`") and name.endswith("`") and len(name) >= 2:
        inner = name[1:-1]
        if not inner:
            raise ValueError("backtick formula names must not be empty")
        return inner, True
    return name, False


def _formula_name_items(segment: str) -> list[tuple[str, bool]]:
    names: list[tuple[str, bool]] = []
    start = 0
    in_backtick = False

    for idx, char in enumerate(segment):
        if char == "`":
            in_backtick = not in_backtick
        elif char == "," and not in_backtick:
            name = segment[start:idx].strip()
            if name:
                names.append(_formula_name(name))
            start = idx + 1

    name = segment[start:].strip()
    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    if name:
        names.append(_formula_name(name))
    return names


def _formula_names(segment: str) -> list[str]:
    return [name for name, _quoted in _formula_name_items(segment)]


def _formula_response_parts(segment: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_backtick = False
    quote: str | None = None

    for idx, char in enumerate(segment):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {"'", '"'} and not in_backtick:
            quote = char
        elif char == "`":
            in_backtick = not in_backtick
        elif not in_backtick and char == "(":
            depth += 1
        elif not in_backtick and char == ")":
            depth = max(0, depth - 1)
        elif char == "," and not in_backtick and depth == 0:
            part = segment[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    if quote is not None:
        raise ValueError("unterminated quote in formula")

    part = segment[start:].strip()
    if part:
        parts.append(part)
    return parts


def _formula_named_option(part: str) -> tuple[str, str] | None:
    depth = 0
    in_backtick = False
    quote: str | None = None

    for idx, char in enumerate(part):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'} and not in_backtick:
            quote = char
            continue
        if char == "`":
            in_backtick = not in_backtick
            continue
        if in_backtick:
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if char != "=" or depth != 0:
            continue

        previous = part[idx - 1] if idx > 0 else ""
        next_char = part[idx + 1] if idx + 1 < len(part) else ""
        if previous in {"=", "!", "<", ">"} or next_char == "=":
            continue
        return part[:idx].strip(), part[idx + 1 :].strip()

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    if quote is not None:
        raise ValueError("unterminated quote in formula")
    return None


def _top_level_comparison(part: str) -> tuple[str, str, str] | None:
    depth = 0
    in_backtick = False
    quote: str | None = None
    operators = ("==", "!=", "<=", ">=", "<", ">")

    idx = 0
    while idx < len(part):
        char = part[idx]
        if quote is not None:
            if char == quote:
                quote = None
            idx += 1
            continue
        if char in {"'", '"'} and not in_backtick:
            quote = char
            idx += 1
            continue
        if char == "`":
            in_backtick = not in_backtick
            idx += 1
            continue
        if in_backtick:
            idx += 1
            continue
        if char == "(":
            depth += 1
            idx += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            idx += 1
            continue
        if depth == 0:
            for operator in operators:
                if part.startswith(operator, idx):
                    left = part[:idx].strip()
                    right = part[idx + len(operator) :].strip()
                    if not left or not right:
                        raise ValueError("formula response comparisons require both operands")
                    return left, operator, right
        idx += 1

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    if quote is not None:
        raise ValueError("unterminated quote in formula")
    return None


def _parse_formula_literal(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "t"}:
        return True
    if lowered in {"false", "f"}:
        return False

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
    for wrapper in ("I", "identity"):
        prefix = f"{wrapper}("
        if expression.startswith(prefix) and expression.endswith(")"):
            return expression[len(prefix) : -1].strip()
    return expression


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


def _response_arg_values(data: Any, part: str) -> list[Any]:
    part = _unwrap_response_identity(part)
    comparison = _top_level_comparison(part)
    if comparison is None:
        operand = _response_operand(part, allow_literal=False)
        if operand.column is None:
            raise ValueError("Surv(...) formula response arguments must be data columns")
        return _column(data, operand.column)

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


def _formula_response_values(data: Any, spec: _SurvResponseSpec) -> list[list[Any]]:
    return [_response_arg_values(data, argument) for argument in spec.arguments]


def _parse_formula_type_option(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return _normalize_surv_type(value)


def _normalize_surv_type(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Surv type must be a string")
    normalized = value.strip().lower()
    if normalized in _SURV_TYPES:
        return normalized
    matches = [choice for choice in _SURV_TYPES if choice.startswith(normalized)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Surv type is ambiguous; use a full type name")
    raise ValueError("Surv type must be 'right', 'left', 'counting', 'interval', or 'interval2'")


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
    if not lhs.startswith("Surv(") or not lhs.endswith(")"):
        raise ValueError("formula response must be Surv(...)")

    columns: list[str] = []
    surv_type: str | None = None
    origin = 0.0
    has_origin = False
    arguments: list[str] = []
    for part in _formula_response_parts(lhs[5:-1]):
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
            else:
                raise ValueError(
                    "formula Surv(...) supports only named type= and origin= arguments"
                )
            continue
        arguments.append(part)
        _append_unique(columns, _response_arg_columns(part))

    if len(arguments) not in {1, 2, 3}:
        raise ValueError("Surv(...) formula response must have 1, 2, or 3 column arguments")
    return _SurvResponseSpec(
        arguments=tuple(arguments),
        columns=tuple(columns),
        type=surv_type,
        origin=origin,
    )


def _formula_response_args(formula: str) -> list[str]:
    return list(_formula_response_spec(formula).columns)


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
    if term.arithmetic is not None:
        return _arithmetic_expression_columns(term.arithmetic)
    return [term.column]


def _formula_columns(formula: str, data: Any) -> list[str]:
    args = _formula_response_args(formula)
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
    n = len(_column(data, _formula_response_args(formula)[0]))
    indices = _subset_indices(subset, n)
    filtered = {
        name: _subset_optional_sequence(values, indices, name)
        for name, values in row_aligned.items()
    }
    return _subset_data(data, indices), filtered


def _apply_formula_na_action(
    formula: str,
    data: Any,
    na_action: str | None,
    **row_aligned: Any,
) -> tuple[Any, dict[str, Any]]:
    action = _normalize_na_action(na_action)
    if action == "pass":
        return data, row_aligned

    columns = _formula_columns(formula, data)
    n = len(_column(data, columns[0]))
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


def _split_top_level(segment: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_backtick = False

    for idx, char in enumerate(segment):
        if char == "`":
            in_backtick = not in_backtick
        elif not in_backtick and char == "(":
            depth += 1
        elif not in_backtick and char == ")":
            depth = max(0, depth - 1)
        elif char == separator and not in_backtick and depth == 0:
            parts.append(segment[start:idx].strip())
            start = idx + 1

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    parts.append(segment[start:].strip())
    return parts


def _split_top_level_token(segment: str, token: str) -> list[str]:
    parts: list[str] = []
    start = 0
    idx = 0
    depth = 0
    in_backtick = False

    while idx < len(segment):
        char = segment[idx]
        if char == "`":
            in_backtick = not in_backtick
            idx += 1
            continue
        if not in_backtick and char == "(":
            depth += 1
        elif not in_backtick and char == ")":
            depth = max(0, depth - 1)
        elif not in_backtick and depth == 0 and segment.startswith(token, idx):
            parts.append(segment[start:idx].strip())
            idx += len(token)
            start = idx
            continue
        idx += 1

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    parts.append(segment[start:].strip())
    return parts


def _previous_non_space(text: str, idx: int) -> str | None:
    cursor = idx - 1
    while cursor >= 0:
        if not text[cursor].isspace():
            return text[cursor]
        cursor -= 1
    return None


def _find_top_level_arithmetic_operator(
    expression: str,
    operators: set[str],
) -> tuple[str, str, str] | None:
    depth = 0
    in_backtick = False
    for idx in range(len(expression) - 1, -1, -1):
        char = expression[idx]
        if char == "`":
            in_backtick = not in_backtick
            continue
        if in_backtick:
            continue
        if char == ")":
            depth += 1
            continue
        if char == "(":
            depth -= 1
            continue
        if depth == 0 and char in operators:
            previous = _previous_non_space(expression, idx)
            if previous is None or previous in "+-*/(^":
                continue
            left = expression[:idx].strip()
            right = expression[idx + 1 :].strip()
            if not left or not right:
                raise ValueError("formula arithmetic terms require both operands")
            return left, char, right

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    return None


def _find_top_level_power_operator(expression: str) -> tuple[str, str, str] | None:
    depth = 0
    in_backtick = False

    for idx, char in enumerate(expression):
        if char == "`":
            in_backtick = not in_backtick
            continue
        if in_backtick:
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if char == "^" and depth == 0:
            left = expression[:idx].strip()
            right = expression[idx + 1 :].strip()
            if not left or not right:
                raise ValueError("formula arithmetic terms require both operands")
            return left, char, right

    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    return None


def _formula_tokens(rhs: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    op = "+"
    start = 0
    depth = 0
    in_backtick = False

    for idx, char in enumerate(rhs):
        if char == "`":
            in_backtick = not in_backtick
        elif not in_backtick and char == "(":
            depth += 1
        elif not in_backtick and char == ")":
            depth = max(0, depth - 1)
        elif not in_backtick and depth == 0 and char in "+-":
            term = rhs[start:idx].strip()
            if term:
                tokens.append((op, term))
            op = char
            start = idx + 1

    term = rhs[start:].strip()
    if in_backtick:
        raise ValueError("unterminated backtick in formula")
    if term:
        tokens.append((op, term))
    return tokens


def _unsupported_formula_name(name: str, quoted: bool) -> bool:
    return not quoted and any(token in name for token in "():*/+-^%")


def _factor_column_items(term: str) -> tuple[str, list[tuple[str, bool]]] | None:
    for wrapper in ("factor", "as.factor"):
        prefix = f"{wrapper}("
        if term.startswith(prefix) and term.endswith(")"):
            return wrapper, _formula_name_items(term[len(prefix) : -1])
    return None


def _transform_column_items(term: str) -> tuple[str, list[tuple[str, bool]]] | None:
    for wrapper in ("log", "sqrt", "exp", "I", "identity", "as.numeric"):
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


def _dot_covariate_terms(dot_terms: list[str] | None) -> list[_CovariateSpec]:
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
        return _CovariateTerm(columns[0], categorical=True)

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

    term_name, quoted = _formula_name(term)
    if _unsupported_formula_name(term_name, quoted):
        raise ValueError(f"unsupported formula term(s): {term_name}")
    return _CovariateTerm(term_name)


def _parse_interaction_term(
    term: str,
    dot_terms: list[str] | None,
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


def _strip_outer_formula_parentheses(term: str) -> str:
    value = term.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        in_backtick = False
        wraps = True
        for idx, char in enumerate(value):
            if char == "`":
                in_backtick = not in_backtick
            elif not in_backtick and char == "(":
                depth += 1
            elif not in_backtick and char == ")":
                depth -= 1
                if depth == 0 and idx != len(value) - 1:
                    wraps = False
                    break
                if depth < 0:
                    wraps = False
                    break
        if in_backtick:
            raise ValueError("unterminated backtick in formula")
        if not wraps or depth != 0:
            break
        value = value[1:-1].strip()
    return value


def _parse_formula_power_degree(value: str) -> int:
    text = value.strip()
    if not text.isdigit():
        raise ValueError("formula ^ degree must be a nonnegative integer")
    return int(text)


def _parse_formula_power_base_terms(
    term: str,
    dot_terms: list[str] | None,
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
    dot_terms: list[str] | None,
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
    dot_terms: list[str] | None,
) -> list[_CovariateSpec] | None:
    stripped = _strip_outer_formula_parentheses(term)
    if stripped == term.strip():
        return None
    return _parse_formula_power_base_terms(stripped, dot_terms)


def _parse_covariate_expression(
    term: str,
    dot_terms: list[str] | None,
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


def _split_terms(rhs: str, dot_terms: list[str] | None = None) -> _FormulaTerms:
    covariates: list[_CovariateSpec] = []
    strata: list[str] = []
    offsets: list[_CovariateTerm] = []
    clusters: list[str] = []
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
            if op == "-":
                _remove_values(covariates, terms)
            else:
                _append_unique(covariates, terms)
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
            else:
                _append_unique(strata, columns)
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
            else:
                _append_unique(clusters, columns)
            continue
        if term.startswith("offset(") and term.endswith(")"):
            offset_term = _parse_offset_term(term[7:-1])
            if op == "-":
                _remove_values(offsets, [offset_term])
            else:
                _append_unique(offsets, [offset_term])
            continue
        covariate_terms = _parse_covariate_expression(term, dot_terms)
        if op == "-":
            _remove_values(covariates, covariate_terms)
        else:
            _append_unique(covariates, covariate_terms)

    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"unsupported formula term(s): {joined}")
    return _FormulaTerms(
        covariates=covariates,
        strata=strata,
        offsets=offsets,
        clusters=clusters,
        intercept=intercept,
    )


def _formula_has_time_transform_term(formula: str) -> bool:
    _lhs, sep, rhs = formula.partition("~")
    if not sep:
        return False

    in_backtick = False
    for idx, char in enumerate(rhs):
        if char == "`":
            in_backtick = not in_backtick
            continue
        if in_backtick or char != "t":
            continue
        if rhs[idx : idx + 2] != "tt":
            continue
        previous = rhs[idx - 1] if idx > 0 else ""
        next_char = rhs[idx + 2] if idx + 2 < len(rhs) else ""
        if (not previous or not (previous.isalnum() or previous in "._")) and next_char == "(":
            return True
    return False


def _parse_formula(formula: str, data: Any) -> tuple[Surv, _FormulaTerms]:
    _lhs, sep, rhs = formula.partition("~")
    if not sep:
        raise ValueError("formula must contain '~'")

    response_spec = _formula_response_spec(formula)
    args = _formula_response_values(data, response_spec)
    if len(args) == 1:
        surv = Surv(args[0], type=response_spec.type, origin=response_spec.origin)
    elif len(args) == 2:
        surv = Surv(
            args[0],
            args[1],
            type=response_spec.type,
            origin=response_spec.origin,
        )
    elif len(args) == 3:
        surv = Surv(
            args[0],
            args[1],
            args[2],
            type=response_spec.type,
            origin=response_spec.origin,
        )
    else:
        raise ValueError("Surv(...) formula response must have 1, 2, or 3 column arguments")

    terms = _split_terms(rhs, _dot_terms(data, response_spec.columns))
    return surv, terms


def _reject_formula_clusters(function_name: str, terms: _FormulaTerms) -> None:
    if terms.clusters:
        raise ValueError(f"{function_name} formula does not support cluster() terms")


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
    if transform in {"I", "identity", "as.numeric"}:
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
    if term.arithmetic is not None:
        return _arithmetic_expression_values(data, term.arithmetic, n)
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
    values = _term_raw_values(data, term, n)
    if not term.categorical:
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


def _fit_design_term(data: Any, term: _CovariateSpec, n: int) -> _DesignTerm:
    if isinstance(term, _InteractionTerm):
        return _InteractionDesignTerm(
            tuple(_fit_single_design_term(data, factor, n) for factor in term.factors)
        )
    return _fit_single_design_term(data, term, n)


def _fit_formula_design(
    data: Any,
    response_spec: _SurvResponseSpec,
    terms: _FormulaTerms,
    n: int,
    *,
    include_intercept: bool = False,
) -> _FormulaDesign:
    strata_values = _combined_columns(data, terms.strata, n) if terms.strata else []
    return _FormulaDesign(
        response=response_spec,
        covariates=tuple(_fit_design_term(data, term, n) for term in terms.covariates),
        offsets=tuple(terms.offsets),
        strata=tuple(terms.strata),
        strata_levels=_label_levels(strata_values, "strata") if terms.strata else (),
        intercept=include_intercept and terms.intercept,
    )


def _single_design_columns(
    data: Any,
    spec: _SingleDesignTerm,
    n: int,
) -> list[list[float]]:
    values = _term_raw_values(data, spec.term, n)
    if isinstance(spec, _NumericDesignTerm):
        return [_numeric_term_values(values, spec.term)]

    levels = spec.levels
    for value in values:
        if all(value != level for level in levels):
            raise ValueError(
                f"newdata column {spec.term.column!r} contains unknown level {value!r}"
            )
    return [[1.0 if value == level else 0.0 for value in values] for level in levels[1:]]


def _design_term_columns(data: Any, spec: _DesignTerm, n: int) -> list[list[float]]:
    if isinstance(spec, _InteractionDesignTerm):
        factor_columns = [_single_design_columns(data, factor, n) for factor in spec.factors]
        interaction_columns: list[list[float]] = []
        for column_combo in product(*factor_columns):
            interaction_columns.append(
                [math.prod(column[idx] for column in column_combo) for idx in range(n)]
            )
        return interaction_columns
    return _single_design_columns(data, spec, n)


def _design_rows_from_spec(data: Any, design: _FormulaDesign, n: int) -> list[list[float]]:
    columns = [
        column for term in design.covariates for column in _design_term_columns(data, term, n)
    ]
    if design.intercept:
        columns.insert(0, [1.0] * n)
    return [[column[i] for column in columns] for i in range(n)]


def _display_single_design_term(spec: _SingleDesignTerm) -> str:
    term = spec.term
    if term.transform is not None:
        return f"{term.transform}({term.column})"
    if isinstance(spec, _CategoricalDesignTerm):
        return f"factor({term.column})"
    return term.column


def _design_term_name(spec: _DesignTerm) -> str:
    if isinstance(spec, _InteractionDesignTerm):
        return ":".join(_display_single_design_term(factor) for factor in spec.factors)
    return _display_single_design_term(spec)


def _single_design_term_output_names(spec: _SingleDesignTerm) -> list[str]:
    term = spec.term
    if isinstance(spec, _CategoricalDesignTerm):
        return [f"{term.column}{level}" for level in spec.levels[1:]]
    if term.transform is not None:
        return [f"{term.transform}({term.column})"]
    return [term.column]


def _design_term_output_names(spec: _DesignTerm) -> list[str]:
    if isinstance(spec, _InteractionDesignTerm):
        factor_names = [_single_design_term_output_names(factor) for factor in spec.factors]
        return [":".join(combo) for combo in product(*factor_names)]
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
    id: Any | None = None,  # noqa: A002
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
    ):
        if values is not None:
            frame[name] = _materialize_1d(values, name)
    return frame


def _matrix_model_frame(
    response: Surv,
    rows: list[list[float]],
    *,
    weights: Any | None = None,
    offset: Any | None = None,
    offsets: Any | None = None,
    strata: Any | None = None,
    cluster: Any | None = None,
    id: Any | None = None,  # noqa: A002
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "response": response,
        "x": [list(row) for row in rows],
    }
    for name, values in (
        ("(weights)", weights),
        ("(offset)", offsets if offsets is not None else offset),
        ("(strata)", strata),
        ("(cluster)", cluster),
        ("(id)", id),
    ):
        if values is not None:
            frame[name] = _materialize_1d(values, name)
    return frame


def _survreg_matrix_model_frame(
    time: list[float],
    status: list[float],
    time2: list[float] | None,
    rows: list[list[float]],
    *,
    weights: Any | None = None,
    offset: Any | None = None,
    offsets: Any | None = None,
    strata: Any | None = None,
    cluster: Any | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "time": list(time),
        "status": list(status),
        "x": [list(row) for row in rows],
    }
    if time2 is not None:
        frame["time2"] = list(time2)
    for name, values in (
        ("(weights)", weights),
        ("(offset)", offsets if offsets is not None else offset),
        ("(strata)", strata),
        ("(cluster)", cluster),
    ):
        if values is not None:
            frame[name] = _materialize_1d(values, name)
    return frame


def _survfit_formula_model_frame(
    formula: str,
    data: Any,
    response: Surv,
    weights: Any | None,
    id: Any | None = None,  # noqa: A002
    id_column: str | None = None,
) -> dict[str, Any]:
    response_spec = _formula_response_spec(formula)
    frame: dict[str, Any] = {_surv_response_model_name(response_spec): response}
    for column in _formula_columns(formula, data):
        frame[column] = _column(data, column)
    if id_column is not None and id_column not in frame:
        frame[id_column] = _column(data, id_column)
    if weights is not None:
        frame["(weights)"] = _materialize_1d(weights, "(weights)")
    if id is not None:
        frame["(id)"] = _materialize_1d(id, "(id)")
    return frame


def _survfit_model_frame(
    response: Surv,
    group: Any | None,
    weights: Any | None,
    id: Any | None = None,  # noqa: A002
) -> dict[str, Any]:
    frame: dict[str, Any] = {"response": response}
    if group is not None:
        frame["group"] = _materialize_1d(group, "group")
    if weights is not None:
        frame["(weights)"] = _materialize_1d(weights, "(weights)")
    if id is not None:
        frame["(id)"] = _materialize_1d(id, "(id)")
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


def _design_rows(data: Any, terms: list[_CovariateSpec], n: int) -> list[list[float]]:
    columns = [column for term in terms for column in _term_columns(data, term, n)]
    return [[column[i] for column in columns] for i in range(n)]


def _combine_aligned_columns(columns: list[list[Any]], n: int) -> list[Any]:
    if any(len(column) != n for column in columns):
        raise ValueError("formula columns must have the same length as the Surv response")
    if len(columns) == 1:
        return columns[0]
    return [tuple(column[i] for column in columns) for i in range(n)]


def _combined_columns(data: Any, terms: list[str], n: int) -> list[Any]:
    return _combine_aligned_columns([_column(data, term) for term in terms], n)


def _combined_formula_groups(
    data: Any,
    strata_terms: list[str],
    covariate_terms: list[_CovariateSpec],
    n: int,
) -> list[Any]:
    columns = [
        *[_column(data, term) for term in strata_terms],
        *[_term_values(data, term, n) for term in covariate_terms],
    ]
    return _combine_aligned_columns(columns, n)


def _offset_vector(data: Any, terms: Sequence[_CovariateTerm], n: int) -> list[float] | None:
    if not terms:
        return None
    columns = [_numeric_term_values(_term_raw_values(data, term, n), term) for term in terms]
    return [sum(column[i] for column in columns) for i in range(n)]


def _label_levels(values: list[Any], name: str) -> tuple[Any, ...]:
    labels: dict[Any, None] = {}
    for value in values:
        try:
            labels.setdefault(value, None)
        except TypeError as exc:
            raise TypeError(f"{name} contains unhashable labels") from exc
    return tuple(labels)


def _encode_groups(group: Any, n: int) -> list[int]:
    values = _materialize_labels(group, "group")
    if len(values) != n:
        raise ValueError("group must have the same length as the Surv response")
    labels = {value: idx for idx, value in enumerate(_label_levels(values, "group"))}
    encoded = []
    for value in values:
        encoded.append(labels[value])
    return encoded


def _group_indices(group: Any, n: int) -> dict[Any, list[int]]:
    values = _materialize_labels(group, "group")
    if len(values) != n:
        raise ValueError("group must have the same length as the Surv response")

    indices: dict[Any, list[int]] = {}
    for idx, value in enumerate(values):
        try:
            indices.setdefault(value, []).append(idx)
        except TypeError as exc:
            raise TypeError("group contains unhashable labels") from exc
    return indices


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


def _derive_interval2_status(left: list[float], right: list[float]) -> list[int]:
    if len(left) != len(right):
        raise ValueError("Surv inputs must have the same length")

    status: list[int] = []
    for idx, (lo, hi) in enumerate(zip(left, right, strict=True)):
        lo_missing = math.isinf(lo) and lo < 0.0
        hi_missing = math.isinf(hi) and hi > 0.0
        if lo_missing and hi_missing:
            raise ValueError("interval2 observations cannot have both endpoints missing")
        if lo_missing:
            status.append(2)
        elif hi_missing:
            status.append(0)
        elif hi < lo:
            raise ValueError(f"interval2 right endpoint is less than left endpoint at index {idx}")
        elif hi == lo:
            status.append(1)
        else:
            status.append(3)
    return status


def _validate_surv_intervals(
    time: list[float],
    time2: list[float] | None,
    event: list[int],
    surv_type: str,
) -> None:
    if surv_type in {"right", "left"}:
        if any(value not in {0, 1} for value in event):
            raise ValueError(f"{surv_type} Surv status must contain only 0/1 values")
        return
    if surv_type == "interval":
        if time2 is None:
            raise ValueError("interval Surv requires time2")
        for idx, status in enumerate(event):
            if status == 3 and time2[idx] < time[idx]:
                raise ValueError(
                    f"interval right endpoint is less than left endpoint at index {idx}"
                )
        return
    if surv_type == "interval2" and time2 is None:
        raise ValueError("interval2 Surv requires time2")


def _validate_surv_time_values(name: str, values: list[float]) -> None:
    for idx, value in enumerate(values):
        if math.isnan(value):
            continue
        if not math.isfinite(value):
            raise ValueError(f"{name} contains non-finite value at index {idx}")


def _validate_surv_time_structure(
    time: list[float],
    time2: list[float] | None,
    event: list[int],
    start: list[float] | None,
    surv_type: str,
) -> None:
    if surv_type == "interval2":
        return

    _validate_surv_time_values("stop" if start is not None else "time", time)
    if time2 is not None:
        if surv_type == "interval":
            for idx, (status, value) in enumerate(zip(event, time2, strict=True)):
                if status == 3 and not (math.isnan(value) or math.isfinite(value)):
                    raise ValueError(f"time2 contains non-finite value at index {idx}")
        else:
            _validate_surv_time_values("time2", time2)
    if start is None:
        return

    _validate_surv_time_values("start", start)
    for idx, (start_value, stop_value) in enumerate(zip(start, time, strict=True)):
        if math.isnan(start_value) or math.isnan(stop_value):
            continue
        if start_value >= stop_value:
            raise ValueError(f"start[{idx}] must be less than stop[{idx}]")


def _turnbull_intervals(response: Surv) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    if response.type == "left":
        for time, event in zip(response.time, response.event, strict=True):
            if event == 1:
                left.append(time)
                right.append(time)
            else:
                left.append(0.0)
                right.append(time)
        return left, right

    if response.type == "interval":
        if response.time2 is None:
            raise ValueError("interval Surv response is missing time2")
        for time, time2, status in zip(
            response.time,
            response.time2,
            response.event,
            strict=True,
        ):
            if status == 0:
                left.append(time)
                right.append(float("inf"))
            elif status == 1:
                left.append(time)
                right.append(time)
            elif status == 2:
                left.append(0.0)
                right.append(time)
            elif status == 3:
                left.append(time)
                right.append(time2)
        return left, right

    if response.type == "interval2":
        if response.time2 is None:
            raise ValueError("interval2 Surv response is missing time2")
        for time, time2, status in zip(
            response.time,
            response.time2,
            response.event,
            strict=True,
        ):
            if status == 2:
                left.append(0.0)
                right.append(time2)
            else:
                left.append(time)
                right.append(time2)
        return left, right

    raise TypeError("Turnbull intervals require left or interval-censored Surv responses")


def _survreg_response_arrays(response: Surv) -> tuple[list[float], list[float], list[float] | None]:
    if response.type == "right":
        return list(response.time), [float(value) for value in response.event], None

    if response.type == "left":
        return (
            list(response.time),
            [1.0 if value == 1 else 2.0 for value in response.event],
            None,
        )

    if response.type == "interval":
        if response.time2 is None:
            raise ValueError("interval Surv response is missing time2")
        return (
            list(response.time),
            [float(value) for value in response.event],
            list(response.time2),
        )

    if response.type == "interval2":
        if response.time2 is None:
            raise ValueError("interval2 Surv response is missing time2")
        time: list[float] = []
        time2: list[float] = []
        for left, right, status in zip(
            response.time,
            response.time2,
            response.event,
            strict=True,
        ):
            if status == 2:
                time.append(right)
                time2.append(right)
            elif status == 0:
                time.append(left)
                time2.append(left)
            else:
                time.append(left)
                time2.append(right)
        return time, [float(value) for value in response.event], time2

    raise NotImplementedError(
        "survreg currently supports right, left, interval, and interval2 Surv responses"
    )


@dataclass(frozen=True, init=False)
class Surv:
    """Survival response container, like R's Surv."""

    time: tuple[float, ...]
    event: tuple[int, ...]
    start: tuple[float, ...] | None
    time2: tuple[float, ...] | None
    type: str

    def __init__(self, *args: Any, type: str | None = None, origin: Any = 0.0) -> None:  # noqa: A002
        surv_type = _normalize_surv_type(type) if type is not None else None
        origin_value = _finite_float(origin, "origin")
        if len(args) == 1:
            if surv_type is not None:
                raise ValueError("one-argument Surv does not accept an explicit type")
            start = None
            time = [value - origin_value for value in _float_vector(args[0], "time")]
            time2 = None
            event = [1] * len(time)
            surv_type = "right"
        elif len(args) == 2:
            start = None
            if surv_type == "interval2":
                time = [
                    value - origin_value
                    for value in _interval_endpoint_vector(args[0], "time", float("-inf"))
                ]
                time2 = [
                    value - origin_value
                    for value in _interval_endpoint_vector(args[1], "time2", float("inf"))
                ]
                event = _derive_interval2_status(time, time2)
            else:
                time = [value - origin_value for value in _float_vector(args[0], "time")]
                time2 = None
                event = _event_vector(args[1], "event")
                if surv_type not in {None, "right", "left"}:
                    raise ValueError(
                        "two-argument Surv supports type='right', 'left', or 'interval2'"
                    )
                surv_type = surv_type or "right"
        elif len(args) == 3:
            if surv_type == "interval":
                start = None
                time = [value - origin_value for value in _float_vector(args[0], "time")]
                time2 = [value - origin_value for value in _float_vector(args[1], "time2")]
                event = _interval_status_vector(args[2], "event")
            else:
                start = [value - origin_value for value in _float_vector(args[0], "start")]
                time = [value - origin_value for value in _float_vector(args[1], "stop")]
                time2 = None
                event = _event_vector(args[2], "event")
                if surv_type not in {None, "counting"}:
                    raise ValueError("three-argument Surv supports type='counting' or 'interval'")
                surv_type = surv_type or "counting"
        else:
            raise TypeError("Surv expects (time), (time, event), or (start, stop, event)")

        if (
            len(time) != len(event)
            or (start is not None and len(start) != len(time))
            or (time2 is not None and len(time2) != len(time))
        ):
            raise ValueError("Surv inputs must have the same length")
        if not time:
            raise ValueError("Surv inputs must not be empty")
        if surv_type not in _SURV_TYPES:
            raise ValueError(
                "Surv type must be 'right', 'left', 'counting', 'interval', or 'interval2'"
            )
        _validate_surv_intervals(time, time2, event, surv_type)
        _validate_surv_time_structure(time, time2, event, start, surv_type)

        object.__setattr__(self, "time", tuple(time))
        object.__setattr__(self, "event", tuple(event))
        object.__setattr__(self, "start", tuple(start) if start is not None else None)
        object.__setattr__(self, "time2", tuple(time2) if time2 is not None else None)
        object.__setattr__(self, "type", surv_type)

    def __len__(self) -> int:
        return len(self.time)

    @property
    def status(self) -> tuple[int, ...]:
        return self.event


def is_surv(value: Any) -> bool:
    """Return whether *value* is a survival response object, like R's is.Surv."""

    return isinstance(value, Surv)


def _subset_surv(response: Surv, indices: list[int]) -> Surv:
    times = [response.time[idx] for idx in indices]
    events = [response.event[idx] for idx in indices]
    if response.type in {"right", "left"}:
        return Surv(times, events, type=response.type)
    if response.type == "interval":
        if response.time2 is None:
            raise ValueError("interval Surv response is missing time2")
        return Surv(
            times,
            [response.time2[idx] for idx in indices],
            events,
            type="interval",
        )
    if response.type == "interval2":
        if response.time2 is None:
            raise ValueError("interval2 Surv response is missing time2")
        return Surv(times, [response.time2[idx] for idx in indices], type="interval2")
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    return Surv([response.start[idx] for idx in indices], times, events)


def _apply_surv_na_action(
    response: Surv,
    na_action: str | None,
    context: str,
    **row_aligned: Any,
) -> tuple[Surv, dict[str, Any]]:
    action = _normalize_na_action(na_action)
    if action == "pass":
        return response, row_aligned

    columns: list[tuple[str, Any]] = [("time", response.time), ("event", response.event)]
    if response.start is not None:
        columns.append(("start", response.start))
    if response.time2 is not None:
        columns.append(("time2", response.time2))
    columns.extend((name, values) for name, values in row_aligned.items() if values is not None)

    keep = _keep_rows_after_na_action(
        _missing_row_indices(columns, len(response)),
        len(response),
        action,
        context,
    )
    if keep is None:
        return response, row_aligned

    filtered = {
        name: _subset_sequence(values, keep, name) if values is not None else None
        for name, values in row_aligned.items()
    }
    return _subset_surv(response, keep), filtered


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


def _unweighted_survfit_event_counts(
    output_times: list[float],
    time: list[float],
    status: list[int],
    *,
    reverse: bool,
    timefix: bool,
) -> list[float]:
    if not timefix:
        event_counts: dict[float, float] = {}
        for event_time, event in zip(time, status, strict=True):
            if event <= 0 if reverse else event > 0:
                event_counts[event_time] = event_counts.get(event_time, 0.0) + 1.0
        return [event_counts.get(output_time, 0.0) for output_time in output_times]

    event_times = sorted(
        float(event_time)
        for event_time, event in zip(time, status, strict=True)
        if (event <= 0 if reverse else event > 0)
    )
    counts: list[float] = []
    cursor = 0
    for output_time in output_times:
        while (
            cursor < len(event_times) and event_times[cursor] < output_time - _SURVFIT_TIME_EPSILON
        ):
            cursor += 1
        matched = cursor
        while (
            matched < len(event_times)
            and abs(event_times[matched] - output_time) < _SURVFIT_TIME_EPSILON
        ):
            matched += 1
        counts.append(float(matched - cursor))
    return counts


def _exact_survfitkm(
    time: list[float],
    status: list[int],
    weights: list[float] | None,
    entry_times: list[float] | None,
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
) -> SurvfitResult:
    n = len(time)
    case_weights = [1.0] * n if weights is None else weights
    order = sorted(range(n), key=lambda idx: (time[idx], idx))
    entry_order = (
        sorted(range(n), key=lambda idx: (entry_times[idx], idx)) if entry_times is not None else []
    )
    entry_cursor = 0
    current_risk = 0.0 if entry_times is not None else float(sum(case_weights))
    current_estimate = 1.0
    cumulative_variance = 0.0
    cumulative_hazard = 0.0
    cumulative_hazard_variance = 0.0
    alpha = 1.0 - conf_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    output_time: list[float] = []
    n_risk: list[float] = []
    n_event: list[float] = []
    n_censor: list[float] = []
    estimate: list[float] = []
    std_err: list[float] = []
    cumhaz: list[float] = []
    std_chaz: list[float] = []
    conf_lower: list[float] = []
    conf_upper: list[float] = []

    cursor = 0
    while cursor < n:
        current_time = time[order[cursor]]
        if entry_times is not None:
            while entry_cursor < n and entry_times[entry_order[entry_cursor]] < current_time:
                current_risk += case_weights[entry_order[entry_cursor]]
                entry_cursor += 1

        weighted_events = 0.0
        weighted_censor = 0.0
        scan = cursor
        while scan < n and time[order[scan]] == current_time:
            idx = order[scan]
            is_event = status[idx] <= 0 if reverse else status[idx] > 0
            if is_event:
                weighted_events += case_weights[idx]
            else:
                weighted_censor += case_weights[idx]
            scan += 1

        if weighted_events > 0.0 or weighted_censor > 0.0:
            risk_at_time = current_risk
            if weighted_events > 0.0 and risk_at_time > 0.0:
                hazard = weighted_events / risk_at_time
                cumulative_hazard += hazard
                cumulative_hazard_variance += weighted_events / (risk_at_time * risk_at_time)
                current_estimate *= 1.0 - hazard
                if risk_at_time > weighted_events:
                    cumulative_variance += weighted_events / (
                        risk_at_time * (risk_at_time - weighted_events)
                    )

            se = current_estimate * math.sqrt(max(cumulative_variance, 0.0))
            output_time.append(current_time)
            n_risk.append(risk_at_time)
            n_event.append(weighted_events)
            n_censor.append(weighted_censor)
            estimate.append(current_estimate)
            std_err.append(se)
            cumhaz.append(cumulative_hazard)
            std_chaz.append(math.sqrt(max(cumulative_hazard_variance, 0.0)))
            if conf_type != "none":
                lower, upper = _survfit_confidence_interval(
                    current_estimate,
                    se,
                    z,
                    conf_type,
                )
                conf_lower.append(lower)
                conf_upper.append(upper)

        current_risk -= weighted_events + weighted_censor
        cursor = scan

    return SurvfitResult(
        time=output_time,
        n_risk=n_risk,
        n_event=n_event,
        n_censor=n_censor,
        estimate=estimate,
        std_err=std_err,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        cumhaz=cumhaz,
        std_chaz=std_chaz,
    )


def _survfitkm(
    time: list[float],
    status: list[int],
    *,
    weights: list[float] | None,
    entry_times: list[float] | None,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    timefix: bool,
) -> Any:
    if not timefix:
        return _exact_survfitkm(
            time,
            status,
            weights,
            entry_times,
            reverse=reverse,
            conf_level=conf_level,
            conf_type=conf_type,
        )
    return _core.survfitkm(
        time,
        status,
        weights=weights,
        entry_times=entry_times,
        reverse=reverse,
        computation_type=0,
        conf_level=conf_level,
        conf_type=conf_type,
    )


def _survfit_from_km_counts(
    km: Any,
    conf_level: float,
    computation: _SurvfitComputation,
    event_counts: list[float],
    conf_type: str,
) -> SurvfitResult:
    n_risk = [float(value) for value in km.n_risk]
    n_event = [float(value) for value in km.n_event]
    n_censor = [float(value) for value in km.n_censor]
    cumhaz: list[float] = []
    estimate: list[float] = []
    std_err: list[float] = []
    std_chaz: list[float] = []
    conf_lower: list[float] = []
    conf_upper: list[float] = []
    hazard = 0.0
    variance = 0.0
    alpha = 1.0 - conf_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    for risk, events, unweighted_events in zip(
        n_risk,
        n_event,
        event_counts,
        strict=True,
    ):
        if risk > 0.0 and events > 0.0 and unweighted_events > 0.0:
            if computation.ctype == 1:
                hazard += events / risk
                variance += events / (risk * risk)
            else:
                for step in range(int(unweighted_events)):
                    denominator = risk - step * events / unweighted_events
                    if denominator > 0.0:
                        hazard += events / (unweighted_events * denominator)
                        variance += events / (unweighted_events * denominator * denominator)
        se_hazard = math.sqrt(max(variance, 0.0))
        survival = (
            _safe_exp(-hazard) if computation.stype == 2 else float(km.estimate[len(estimate)])
        )
        cumhaz.append(hazard)
        std_chaz.append(se_hazard)
        estimate.append(survival)
        if computation.stype == 2:
            std_err.append(survival * se_hazard)
            if conf_type != "none":
                lower, upper = _survfit_confidence_interval(
                    survival,
                    survival * se_hazard,
                    z,
                    conf_type,
                )
                conf_lower.append(lower)
                conf_upper.append(upper)
        else:
            std_err.append(float(km.std_err[len(std_err)]))
            if conf_type != "none":
                conf_lower.append(float(km.conf_lower[len(conf_lower)]))
                conf_upper.append(float(km.conf_upper[len(conf_upper)]))

    return SurvfitResult(
        time=[float(value) for value in km.time],
        n_risk=n_risk,
        n_event=n_event,
        n_censor=n_censor,
        estimate=estimate,
        std_err=std_err,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        cumhaz=cumhaz,
        std_chaz=std_chaz,
        n_enter=(
            [float(value) for value in km.n_enter]
            if getattr(km, "n_enter", None) is not None
            else None
        ),
    )


def _survfit_time_equal(left: float, right: float, timefix: bool) -> bool:
    return abs(left - right) < _SURVFIT_TIME_EPSILON if timefix else left == right


def _survfit_time_less(left: float, right: float, timefix: bool) -> bool:
    return left < right - _SURVFIT_TIME_EPSILON if timefix else left < right


def _survfit_positions(
    response: Surv,
    id_values: list[Any],
    timefix: bool,
) -> list[int]:
    if response.start is None:
        return [3] * len(response)

    id_labels = _label_levels(id_values, "id")
    id_codes = {value: idx for idx, value in enumerate(id_labels)}
    starts = [float(value) for value in response.start]
    stops = [float(value) for value in response.time]
    order = sorted(
        range(len(response)),
        key=lambda idx: (id_codes[id_values[idx]], stops[idx], idx),
    )
    positions = [0] * len(response)
    for sorted_idx, row_idx in enumerate(order):
        current_id = id_codes[id_values[row_idx]]
        previous_row = order[sorted_idx - 1] if sorted_idx > 0 else None
        next_row = order[sorted_idx + 1] if sorted_idx + 1 < len(order) else None

        first = previous_row is None or id_codes[id_values[previous_row]] != current_id
        if previous_row is not None and not first:
            first = _survfit_time_less(stops[previous_row], starts[row_idx], timefix)

        last = next_row is None or id_codes[id_values[next_row]] != current_id
        if next_row is not None and not last:
            last = _survfit_time_less(stops[row_idx], starts[next_row], timefix)

        positions[row_idx] = (1 if first else 0) + (2 if last else 0)
    return positions


def _survfit_counting_times(
    starts: list[float],
    stops: list[float],
    status: list[int],
    positions: list[int],
    include_entry: bool,
    timefix: bool,
) -> list[float]:
    n = len(stops)
    sort_stop = sorted(range(n), key=lambda idx: (stops[idx], idx))
    if not include_entry:
        times: list[float] = []
        for idx in sort_stop:
            if (positions[idx] > 1 or status[idx] > 0 or not times) and (
                not times or not _survfit_time_equal(stops[idx], times[-1], timefix)
            ):
                times.append(stops[idx])
        return times

    sort_start = sorted(range(n), key=lambda idx: (starts[idx], idx))
    times = [starts[sort_start[0]]]
    current = times[0]
    entry_cursor = 1
    for stop_idx in sort_stop:
        while entry_cursor < n and _survfit_time_less(
            starts[sort_start[entry_cursor]],
            stops[stop_idx],
            timefix,
        ):
            start_idx = sort_start[entry_cursor]
            if positions[start_idx] & 1 and not _survfit_time_equal(
                starts[start_idx],
                current,
                timefix,
            ):
                current = starts[start_idx]
                times.append(current)
            entry_cursor += 1

        if (positions[stop_idx] > 1 or status[stop_idx] > 0) and not _survfit_time_equal(
            stops[stop_idx],
            current,
            timefix,
        ):
            current = stops[stop_idx]
            times.append(current)
    return times


def _survfit_from_count_tables(
    times: list[float],
    n_risk: list[float],
    n_event: list[float],
    n_event_count: list[float],
    n_censor: list[float],
    n_censor_count: list[float],
    n_enter: list[float] | None,
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
) -> SurvfitResult:
    current_survival = 1.0
    greenwood_variance = 0.0
    cumulative_hazard = 0.0
    cumulative_hazard_variance = 0.0
    alpha = 1.0 - conf_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    estimate: list[float] = []
    std_err: list[float] = []
    cumhaz: list[float] = []
    std_chaz: list[float] = []
    conf_lower: list[float] = []
    conf_upper: list[float] = []

    for idx, risk_count in enumerate(n_risk):
        event_weight = n_censor[idx] if reverse else n_event[idx]
        event_count = n_censor_count[idx] if reverse else n_event_count[idx]
        risk_for_curve = risk_count - n_event[idx] if reverse else risk_count

        if event_weight > 0.0 and event_count > 0.0 and risk_for_curve > 0.0:
            if computation.ctype == 1:
                cumulative_hazard += event_weight / risk_for_curve
                cumulative_hazard_variance += event_weight / (risk_for_curve * risk_for_curve)
            else:
                for step in range(int(event_count)):
                    denominator = risk_for_curve - step * event_weight / event_count
                    if denominator > 0.0:
                        cumulative_hazard += event_weight / (event_count * denominator)
                        cumulative_hazard_variance += event_weight / (
                            event_count * denominator * denominator
                        )

        if computation.stype == 1:
            if event_weight > 0.0 and event_count > 0.0 and risk_for_curve > 0.0:
                current_survival *= max((risk_for_curve - event_weight) / risk_for_curve, 0.0)
                if risk_for_curve > event_weight:
                    greenwood_variance += event_weight / (
                        risk_for_curve * (risk_for_curve - event_weight)
                    )
            survival = current_survival
            survival_se = survival * math.sqrt(max(greenwood_variance, 0.0))
        else:
            survival = _safe_exp(-cumulative_hazard)
            survival_se = survival * math.sqrt(max(cumulative_hazard_variance, 0.0))

        estimate.append(survival)
        std_err.append(survival_se)
        cumhaz.append(cumulative_hazard)
        std_chaz.append(math.sqrt(max(cumulative_hazard_variance, 0.0)))
        if conf_type != "none":
            lower, upper = _survfit_confidence_interval(survival, survival_se, z, conf_type)
            conf_lower.append(lower)
            conf_upper.append(upper)

    return SurvfitResult(
        time=times,
        n_risk=n_risk,
        n_event=n_event,
        n_censor=n_censor,
        estimate=estimate,
        std_err=std_err,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        cumhaz=cumhaz,
        std_chaz=std_chaz,
        n_enter=n_enter,
    )


def _survfit_counting_with_id(
    response: Surv,
    weights: list[float] | None,
    id_values: list[Any],
    *,
    include_entry: bool,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
    timefix: bool,
) -> SurvfitResult:
    if response.start is None:
        raise ValueError("survfit id-aware entry counts require counting-process Surv input")

    n = len(response)
    starts = [float(value) for value in response.start]
    stops = [float(value) for value in response.time]
    status = [int(value) for value in response.event]
    case_weights = [1.0] * n if weights is None else [float(value) for value in weights]
    positions = _survfit_positions(response, id_values, timefix)
    times = _survfit_counting_times(starts, stops, status, positions, include_entry, timefix)
    sort_start = sorted(range(n), key=lambda idx: (starts[idx], idx))
    sort_stop = sorted(range(n), key=lambda idx: (stops[idx], idx))

    n_risk = [0.0] * len(times)
    n_event = [0.0] * len(times)
    n_event_count = [0.0] * len(times)
    n_censor = [0.0] * len(times)
    n_censor_count = [0.0] * len(times)
    n_enter = [0.0] * len(times) if include_entry else None

    stop_cursor = n - 1
    start_cursor = n - 1
    weighted_risk = 0.0
    for time_idx in range(len(times) - 1, -1, -1):
        current_time = times[time_idx]
        event_weight = 0.0
        event_count = 0.0
        censor_weight = 0.0
        censor_count = 0.0
        while stop_cursor >= 0 and not _survfit_time_less(
            stops[sort_stop[stop_cursor]],
            current_time,
            timefix,
        ):
            row_idx = sort_stop[stop_cursor]
            weighted_risk += case_weights[row_idx]
            if status[row_idx] > 0:
                event_count += 1.0
                event_weight += case_weights[row_idx]
            elif positions[row_idx] & 2:
                censor_count += 1.0
                censor_weight += case_weights[row_idx]
            stop_cursor -= 1

        enter_weight = 0.0
        while start_cursor >= 0 and not _survfit_time_less(
            starts[sort_start[start_cursor]],
            current_time,
            timefix,
        ):
            row_idx = sort_start[start_cursor]
            weighted_risk -= case_weights[row_idx]
            if (
                include_entry
                and positions[row_idx] & 1
                and _survfit_time_equal(starts[row_idx], current_time, timefix)
            ):
                enter_weight += case_weights[row_idx]
            start_cursor -= 1

        n_risk[time_idx] = max(weighted_risk, 0.0)
        n_event[time_idx] = event_weight
        n_event_count[time_idx] = event_count
        n_censor[time_idx] = censor_weight
        n_censor_count[time_idx] = censor_count
        if n_enter is not None:
            n_enter[time_idx] = enter_weight

    return _survfit_from_count_tables(
        times,
        n_risk,
        n_event,
        n_event_count,
        n_censor,
        n_censor_count,
        n_enter,
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        computation=computation,
    )


def _survfit_start_time_indices(
    response: Surv,
    start_time: float,
    timefix: bool,
) -> list[int]:
    if timefix:
        indices = [
            idx
            for idx, stop_time in enumerate(response.time)
            if stop_time >= start_time - _SURVFIT_TIME_EPSILON
        ]
    else:
        indices = [idx for idx, stop_time in enumerate(response.time) if stop_time >= start_time]
    if not indices:
        raise ValueError("all observations removed by start_time")
    return indices


def _survfit_default_time0(response: Surv) -> float:
    values = [0.0, *response.time]
    if response.start is not None:
        values.extend(response.start)
    return float(min(values))


def _initial_survfit_risk(
    response: Surv,
    weights: list[float] | None,
    t0: float,
    timefix: bool,
) -> float:
    case_weights = [1.0] * len(response) if weights is None else weights
    if response.start is None:
        return float(sum(case_weights))
    if not timefix:
        return float(
            sum(
                weight
                for start, stop, weight in zip(
                    response.start,
                    response.time,
                    case_weights,
                    strict=True,
                )
                if start <= t0 <= stop
            )
        )
    return float(
        sum(
            weight
            for start, stop, weight in zip(response.start, response.time, case_weights, strict=True)
            if start <= t0 + _SURVFIT_TIME_EPSILON and stop >= t0 - _SURVFIT_TIME_EPSILON
        )
    )


def _cumhaz_from_survfit_counts(n_risk: list[float], n_event: list[float]) -> list[float]:
    hazard = 0.0
    cumhaz = []
    for risk, events in zip(n_risk, n_event, strict=True):
        if risk > 0.0:
            hazard += events / risk
        cumhaz.append(hazard)
    return cumhaz


def _std_chaz_from_survfit_counts(n_risk: list[float], n_event: list[float]) -> list[float]:
    variance = 0.0
    std_chaz = []
    for risk, events in zip(n_risk, n_event, strict=True):
        if risk > 0.0:
            variance += events / (risk * risk)
        std_chaz.append(math.sqrt(max(variance, 0.0)))
    return std_chaz


def _survfit_with_time0(
    result: Any,
    t0: float,
    conf_type: str,
    initial_n_risk: float,
    timefix: bool,
) -> Any:
    times = [float(value) for value in result.time]
    if times and (abs(times[0] - t0) < _SURVFIT_TIME_EPSILON if timefix else times[0] == t0):
        return result

    n_risk = [float(value) for value in result.n_risk]
    n_event = [float(value) for value in result.n_event]
    n_censor = [float(value) for value in result.n_censor]
    estimate = [float(value) for value in result.estimate]
    std_err = [float(value) for value in result.std_err]
    conf_lower = [float(value) for value in result.conf_lower]
    conf_upper = [float(value) for value in result.conf_upper]
    cumhaz = (
        [float(value) for value in result.cumhaz]
        if hasattr(result, "cumhaz")
        else _cumhaz_from_survfit_counts(n_risk, n_event)
    )
    std_chaz = (
        [float(value) for value in result.std_chaz]
        if hasattr(result, "std_chaz")
        else _std_chaz_from_survfit_counts(n_risk, n_event)
    )
    n_risk0 = n_risk[0] if n_risk else initial_n_risk

    return SurvfitResult(
        time=[t0, *times],
        n_risk=[n_risk0, *n_risk],
        n_event=[0.0, *n_event],
        n_censor=[0.0, *n_censor],
        estimate=[1.0, *estimate],
        std_err=[0.0, *std_err],
        conf_lower=([1.0, *conf_lower] if conf_type != "none" else []),
        conf_upper=([1.0, *conf_upper] if conf_type != "none" else []),
        cumhaz=[0.0, *cumhaz],
        std_chaz=[0.0, *std_chaz],
        n_enter=([0.0, *result.n_enter] if getattr(result, "n_enter", None) is not None else None),
    )


def _survfit_without_standard_errors(result: Any) -> Any:
    if isinstance(result, SurvfitResult):
        return SurvfitResult(
            time=result.time,
            n_risk=result.n_risk,
            n_event=result.n_event,
            n_censor=result.n_censor,
            estimate=result.estimate,
            std_err=[],
            conf_lower=[],
            conf_upper=[],
            cumhaz=result.cumhaz,
            std_chaz=[],
            n_enter=result.n_enter,
            model=result.model,
        )
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[],
            conf_lower=[],
            conf_upper=[],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            model=getattr(result, "model", None),
        )
    if isinstance(result, CoxSurvfitResult):
        return CoxSurvfitResult(
            time=result.time,
            surv=result.surv,
            cumhaz=result.cumhaz,
            linear_predictors=result.linear_predictors,
            centered=result.centered,
            strata=result.strata,
            start_time=result.start_time,
            std_err=[],
            std_chaz=[],
            conf_lower=[],
            conf_upper=[],
            model=result.model,
        )
    if isinstance(result, Mapping):
        return {label: _survfit_without_standard_errors(curve) for label, curve in result.items()}
    return result


def _survfit_with_model_frame(result: Any, model_frame: dict[str, Any]) -> Any:
    if isinstance(result, SurvfitResult):
        return SurvfitResult(
            time=result.time,
            n_risk=result.n_risk,
            n_event=result.n_event,
            n_censor=result.n_censor,
            estimate=result.estimate,
            std_err=result.std_err,
            conf_lower=result.conf_lower,
            conf_upper=result.conf_upper,
            cumhaz=result.cumhaz,
            std_chaz=result.std_chaz,
            n_enter=result.n_enter,
            model=model_frame,
        )
    if all(
        hasattr(result, name)
        for name in (
            "time_points",
            "survival",
            "survival_lower",
            "survival_upper",
            "n_iter",
            "converged",
        )
    ):
        return TurnbullSurvfitResult(
            time_points=[float(value) for value in result.time_points],
            survival=[float(value) for value in result.survival],
            survival_lower=[float(value) for value in result.survival_lower],
            survival_upper=[float(value) for value in result.survival_upper],
            n_iter=int(result.n_iter),
            converged=bool(result.converged),
            model=model_frame,
        )
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[float(value) for value in result.std_err],
            conf_lower=[float(value) for value in result.conf_lower],
            conf_upper=[float(value) for value in result.conf_upper],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[float(value) for value in result.std_chaz],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            model=model_frame,
        )
    if isinstance(result, CoxSurvfitResult):
        return CoxSurvfitResult(
            time=result.time,
            surv=result.surv,
            cumhaz=result.cumhaz,
            linear_predictors=result.linear_predictors,
            centered=result.centered,
            strata=result.strata,
            start_time=result.start_time,
            std_err=result.std_err,
            std_chaz=result.std_chaz,
            conf_lower=result.conf_lower,
            conf_upper=result.conf_upper,
            model=model_frame,
        )
    if isinstance(result, Mapping):
        return {
            label: _survfit_with_model_frame(curve, model_frame) for label, curve in result.items()
        }
    return result


def _cox_flat_basehaz_with_training_times(
    fit: Any,
    centered: bool,
) -> CoxBaseHazardResult:
    with_strata = getattr(fit, "basehaz_with_strata", None)
    if with_strata is None:
        base_times, base_hazards = fit.basehaz(centered)
        event_times = getattr(fit, "event_times", None)
        training_times = (
            sorted({float(value) for value in event_times})
            if event_times is not None
            else [float(value) for value in base_times]
        )
        hazards = [float(value) for value in base_hazards]
        times = [float(value) for value in base_times]
        return CoxBaseHazardResult(
            time=training_times,
            cumhaz=[_step_hazard_at(times, hazards, time) for time in training_times],
            centered=centered,
        )

    base_times, base_hazards, base_strata = with_strata(centered)
    event_times = getattr(fit, "event_times", None)
    if event_times is None:
        strata_values = [int(value) for value in base_strata]
        return CoxBaseHazardResult(
            time=[float(value) for value in base_times],
            cumhaz=[float(value) for value in base_hazards],
            strata=strata_values if len(set(strata_values)) > 1 else None,
            centered=centered,
        )

    stop_times = [float(value) for value in event_times]
    row_strata = _cox_training_strata(fit, len(stop_times))
    expanded_times: list[float] = []
    expanded_hazards: list[float] = []
    expanded_strata: list[int] = []
    baselines = _cox_baselines_by_stratum(base_times, base_hazards, base_strata)
    stop_times_by_stratum: dict[int, set[float]] = {}
    for stop, row_stratum in zip(stop_times, row_strata, strict=True):
        stop_times_by_stratum.setdefault(row_stratum, set()).add(stop)

    for stratum, stratum_stop_times in sorted(stop_times_by_stratum.items()):
        stratum_times, stratum_hazards = baselines.get(stratum, ([], []))
        for time in sorted(stratum_stop_times):
            expanded_times.append(time)
            expanded_hazards.append(_step_hazard_at(stratum_times, stratum_hazards, time))
            expanded_strata.append(stratum)

    return CoxBaseHazardResult(
        time=expanded_times,
        cumhaz=expanded_hazards,
        strata=expanded_strata if len(set(row_strata)) > 1 else None,
        centered=centered,
    )


def survfit(
    response: Any,
    data: Any | None = None,
    *,
    group: Any | None = None,
    newdata: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    conf_level: float = 0.95,
    conf_int: Any | None = None,
    conf_type: str | None = "log",
    se_fit: Any = True,
    start_time: Any | None = None,
    time0: bool = False,
    reverse: bool = False,
    censor: bool = True,
    type: str | None = None,  # noqa: A002
    stype: int | None = None,
    ctype: int | None = None,
    id: Any | None = None,  # noqa: A002
    cluster: Any | None = None,
    robust: Any | None = None,
    istate: Any | None = None,
    etype: Any | None = None,
    model: Any = False,
    error: Any | None = None,
    entry: Any = False,
    timefix: bool = True,
    **kwargs: Any,
):
    """Fit Kaplan-Meier curves or Cox-model survival curves."""

    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, None)
    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, "log")
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, True)
    start_time = _pop_dotted_keyword(kwargs, "start.time", "start_time", start_time, None)
    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", timefix, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit got unexpected keyword argument(s): {unexpected}")

    keep_model = _normalize_bool_option_with_default(model, "model", False)
    include_entry = _normalize_bool_option_with_default(entry, "entry", False)
    include_se = _normalize_bool_option_with_default(se_fit, "se_fit", True)
    robust_value = _normalize_optional_bool_option(robust, "robust")
    id_arg = id
    if cluster is not None:
        raise NotImplementedError("survfit cluster is not supported")
    if robust_value is True:
        raise NotImplementedError("survfit robust variance is not supported")
    if istate is not None or etype is not None:
        raise NotImplementedError("survfit multi-state istate/etype inputs are not supported")
    # R's survival keeps `error` for backward compatibility but no longer uses it.

    computation = _normalize_survfit_type(type, stype, ctype)
    normalized_conf_level = _normalize_survfit_conf_level(conf_level, conf_int)
    normalized_conf_type = _normalize_survfit_conf_type(conf_type)
    normalized_start_time = _normalize_start_time(start_time)
    include_time0 = _normalize_bool_option(time0, "time0")
    reverse_curve = _normalize_bool_option(reverse, "reverse")
    include_censor = _normalize_bool_option(censor, "censor")
    fix_time = _normalize_bool_option(timefix, "timefix")
    model_frame = None
    if isinstance(response, str):
        formula = response
        id_column = id_arg if isinstance(id_arg, str) else None
        if id_column is not None:
            id_arg = _column(data, id_column)
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                formula,
                data,
                subset,
                weights=weights,
                id=id_arg,
            )
            weights = aligned["weights"]
            id_arg = aligned["id"]
            subset = None
        data, aligned = _apply_formula_na_action(
            formula,
            data,
            na_action,
            weights=weights,
            id=id_arg,
        )
        weights = aligned["weights"]
        id_arg = aligned["id"]
        na_action = "pass"
        response, terms = _parse_formula(formula, data)
        _reject_formula_clusters("survfit", terms)
        if keep_model:
            model_frame = _survfit_formula_model_frame(
                formula,
                data,
                response,
                weights,
                id_arg,
                id_column,
            )
        if terms.strata or terms.covariates:
            group = _combined_formula_groups(data, terms.strata, terms.covariates, len(response))

    if not isinstance(response, Surv) and hasattr(response, "survival_curve"):
        if not computation.is_kaplan_meier:
            raise ValueError(
                "non-Kaplan-Meier survfit styles are only supported for Surv or formula inputs"
            )
        if reverse_curve:
            raise ValueError("reverse survfit is only supported for Surv or formula inputs")
        if subset is not None:
            raise ValueError("subset is only supported for Surv or formula inputs")
        rows, offsets = _prediction_inputs(response, newdata)
        if hasattr(response, "means"):
            result = _cox_survfit_result(
                response,
                rows,
                offsets,
                True,
                newdata,
                normalized_start_time,
                include_time0,
                include_censor,
                normalized_conf_level,
                normalized_conf_type,
                compute_confidence=include_se,
            )
            return (
                _survfit_with_model_frame(result, _cox_survfit_model_frame(response, newdata))
                if keep_model
                else result
            )
        if normalized_start_time is not None:
            raise ValueError("start_time is only supported for Surv, formula, or fitted Cox inputs")
        if include_time0:
            raise ValueError("time0 is only supported for Surv, formula, or fitted Cox inputs")
        if keep_model:
            raise NotImplementedError(
                "survfit model=TRUE is only supported for Surv, formula, or fitted Cox inputs"
            )
        if rows is None:
            coefficients = getattr(response, "coefficients", [])
            width = len(coefficients[0]) if coefficients else 0
            if width == 0:
                raise ValueError("newdata is required for an unfitted Cox model")
            rows = [[0.0] * width]
        return response.survival_curve(rows, None)

    if not isinstance(response, Surv):
        raise TypeError("survfit response must be a Surv object, formula, or fitted Cox model")
    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
        weights = _subset_optional_sequence(weights, indices, "weights")
        id_arg = _subset_optional_sequence(id_arg, indices, "id")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "survfit inputs",
        group=group,
        weights=weights,
        id=id_arg,
    )
    group = aligned["group"]
    weights = aligned["weights"]
    id_arg = aligned["id"]
    id_values = _materialize_labels(id_arg, "id") if id_arg is not None else None
    if id_values is not None and len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if keep_model and model_frame is None:
        model_frame = _survfit_model_frame(response, group, weights, id_values)
    if newdata is not None:
        raise ValueError("newdata is only supported for fitted Cox models")
    if not include_censor:
        raise ValueError("censor is only supported for fitted Cox models")
    if include_entry and (response.start is None or id_values is None):
        raise ValueError("survfit entry=TRUE requires counting-process Surv input and id")
    if response.type in {"left", "interval", "interval2"}:
        if not computation.is_kaplan_meier:
            raise ValueError(
                "non-Kaplan-Meier survfit styles are only supported for right-censored data"
            )
        if normalized_conf_type != "log":
            raise ValueError("conf_type is only supported for right-censored data")
        if normalized_start_time is not None:
            raise ValueError("start_time is only supported for right-censored data")
        if include_time0:
            raise ValueError("time0 is only supported for right-censored data")
        if reverse_curve:
            raise ValueError("reverse survfit is only supported for right-censored data")
        wt = _float_vector(weights, "weights") if weights is not None else None
        if wt is not None and len(wt) != len(response):
            raise ValueError("weights must have the same length as the Surv response")
        if response.start is not None:
            raise ValueError("interval-censored survfit does not support entry times")
        if group is None:
            left, right = _turnbull_intervals(response)
            result = _core.turnbull_estimator(left, right, weights=wt)
            return (
                _survfit_with_model_frame(result, model_frame)
                if keep_model and model_frame is not None
                else result
            )

        turnbull_results: dict[Any, Any] = {}
        for label, indices in _group_indices(group, len(response)).items():
            group_response = _subset_surv(response, indices)
            left, right = _turnbull_intervals(group_response)
            group_weights = [wt[idx] for idx in indices] if wt is not None else None
            turnbull_results[label] = _core.turnbull_estimator(left, right, weights=group_weights)
        return (
            _survfit_with_model_frame(turnbull_results, model_frame)
            if keep_model and model_frame is not None
            else turnbull_results
        )

    wt = _float_vector(weights, "weights") if weights is not None else None
    if wt is not None and len(wt) != len(response):
        raise ValueError("weights must have the same length as the Surv response")
    t0 = (
        normalized_start_time
        if normalized_start_time is not None
        else _survfit_default_time0(response)
    )
    if normalized_start_time is not None:
        indices = _survfit_start_time_indices(response, normalized_start_time, fix_time)
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
        wt = _subset_optional_sequence(wt, indices, "weights")
        id_values = _subset_optional_sequence(id_values, indices, "id")
    entry_times = list(response.start) if response.start is not None else None

    if group is None:
        km = (
            _survfit_counting_with_id(
                response,
                wt,
                id_values,
                include_entry=include_entry,
                reverse=reverse_curve,
                conf_level=normalized_conf_level,
                conf_type=normalized_conf_type,
                computation=computation,
                timefix=fix_time,
            )
            if response.start is not None and id_values is not None
            else _survfitkm(
                list(response.time),
                list(response.event),
                weights=wt,
                entry_times=entry_times,
                reverse=reverse_curve,
                conf_level=normalized_conf_level,
                conf_type=normalized_conf_type,
                timefix=fix_time,
            )
        )
        if computation.is_kaplan_meier:
            result = (
                _survfit_with_time0(
                    km,
                    t0,
                    normalized_conf_type,
                    _initial_survfit_risk(response, wt, t0, fix_time),
                    fix_time,
                )
                if include_time0
                else km
            )
            result = _survfit_without_standard_errors(result) if not include_se else result
            return (
                _survfit_with_model_frame(result, model_frame)
                if model_frame is not None
                else result
            )
        event_counts = _unweighted_survfit_event_counts(
            [float(value) for value in km.time],
            list(response.time),
            list(response.event),
            reverse=reverse_curve,
            timefix=fix_time,
        )
        result = _survfit_from_km_counts(
            km,
            normalized_conf_level,
            computation,
            event_counts,
            normalized_conf_type,
        )
        result = (
            _survfit_with_time0(
                result,
                t0,
                normalized_conf_type,
                _initial_survfit_risk(response, wt, t0, fix_time),
                fix_time,
            )
            if include_time0
            else result
        )
        result = _survfit_without_standard_errors(result) if not include_se else result
        return _survfit_with_model_frame(result, model_frame) if model_frame is not None else result

    results: dict[Any, Any] = {}
    for label, indices in _group_indices(group, len(response)).items():
        group_response = _subset_surv(response, indices)
        group_weights = [wt[idx] for idx in indices] if wt is not None else None
        group_ids = [id_values[idx] for idx in indices] if id_values is not None else None
        km = (
            _survfit_counting_with_id(
                group_response,
                group_weights,
                group_ids,
                include_entry=include_entry,
                reverse=reverse_curve,
                conf_level=normalized_conf_level,
                conf_type=normalized_conf_type,
                computation=computation,
                timefix=fix_time,
            )
            if group_response.start is not None and group_ids is not None
            else _survfitkm(
                list(group_response.time),
                list(group_response.event),
                weights=group_weights,
                entry_times=(
                    list(group_response.start) if group_response.start is not None else None
                ),
                reverse=reverse_curve,
                conf_level=normalized_conf_level,
                conf_type=normalized_conf_type,
                timefix=fix_time,
            )
        )
        if computation.is_kaplan_meier:
            results[label] = (
                _survfit_with_time0(
                    km,
                    t0,
                    normalized_conf_type,
                    _initial_survfit_risk(group_response, group_weights, t0, fix_time),
                    fix_time,
                )
                if include_time0
                else km
            )
        else:
            event_counts = _unweighted_survfit_event_counts(
                [float(value) for value in km.time],
                list(group_response.time),
                list(group_response.event),
                reverse=reverse_curve,
                timefix=fix_time,
            )
            result = _survfit_from_km_counts(
                km,
                normalized_conf_level,
                computation,
                event_counts,
                normalized_conf_type,
            )
            results[label] = (
                _survfit_with_time0(
                    result,
                    t0,
                    normalized_conf_type,
                    _initial_survfit_risk(group_response, group_weights, t0, fix_time),
                    fix_time,
                )
                if include_time0
                else result
            )
    result = _survfit_without_standard_errors(results) if not include_se else results
    return _survfit_with_model_frame(result, model_frame) if model_frame is not None else result


def _survdiff_weight_type(rho: float) -> str:
    return "LogRank" if rho == 0.0 else f"FlemingHarrington(p={rho}, q=0)"


def _survdiff_formula_groups(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[Any], list[Any] | None]:
    if not terms.covariates:
        if terms.strata:
            raise ValueError("survdiff formula has no groups to test")
        raise ValueError("survdiff formula requires at least one grouping term")
    group = _combined_formula_groups(data, [], terms.covariates, n)
    strata = _combined_columns(data, terms.strata, n) if terms.strata else None
    return group, strata


def _survdiff_strata_markers(strata: list[int]) -> list[int]:
    markers = [0] * len(strata)
    for idx, value in enumerate(strata):
        if idx + 1 == len(strata) or strata[idx + 1] != value:
            markers[idx] = 1
    return markers


def _survdiff_timefix_values(times: list[float], timefix: bool) -> list[float]:
    if not timefix:
        return times

    fixed = list(times)
    order = sorted(range(len(times)), key=lambda idx: (times[idx], idx))
    cursor = 0
    while cursor < len(order):
        base = fixed[order[cursor]]
        scan = cursor + 1
        while scan < len(order) and fixed[order[scan]] - base < _SURVFIT_TIME_EPSILON:
            fixed[order[scan]] = base
            scan += 1
        cursor = scan
    return fixed


def _timefix_vectors(*vectors: list[float]) -> tuple[list[float], ...]:
    fixed = [list(vector) for vector in vectors]
    points = [
        (value, vector_idx, row_idx)
        for vector_idx, vector in enumerate(fixed)
        for row_idx, value in enumerate(vector)
    ]
    points.sort(key=lambda item: (item[0], item[1], item[2]))
    cursor = 0
    while cursor < len(points):
        base = points[cursor][0]
        scan = cursor + 1
        while scan < len(points) and points[scan][0] - base < _SURVFIT_TIME_EPSILON:
            _value, vector_idx, row_idx = points[scan]
            fixed[vector_idx][row_idx] = base
            scan += 1
        cursor = scan
    return tuple(fixed)


def _survdiff_result_from_components(components: Any, rho: float) -> Any:
    statistic = float(components.chi_squared)
    df = int(components.degrees_of_freedom)
    p_value = 1.0 if df == 0 else float(_core.lrt_test(statistic / 2.0, 0.0, df).p_value)
    variance = (
        float(components.variance[0][0]) if components.variance and components.variance[0] else 0.0
    )
    return _core.LogRankResult(
        statistic,
        p_value,
        df,
        [float(value) for value in components.observed],
        [float(value) for value in components.expected],
        variance,
        _survdiff_weight_type(rho),
    )


def _survdiff_result_from_summary(
    observed: list[float],
    expected: list[float],
    variance: float,
    rho: float,
) -> Any:
    df = max(len(observed) - 1, 0)
    statistic = (observed[0] - expected[0]) ** 2 / variance if df > 0 and variance > 0.0 else 0.0
    p_value = 1.0 if df == 0 else float(_core.lrt_test(statistic / 2.0, 0.0, df).p_value)
    return _core.LogRankResult(
        statistic,
        p_value,
        df,
        observed,
        expected,
        variance,
        _survdiff_weight_type(rho),
    )


def _exact_counting_survdiff_result(
    stop_times: list[float],
    status: list[int],
    entry_times: list[float],
    group_codes: list[int],
    n_groups: int,
    rho: float,
) -> Any:
    observed = [0.0] * n_groups
    expected = [0.0] * n_groups
    variance = 0.0
    if n_groups < 2:
        return _survdiff_result_from_summary(observed, expected, variance, rho)

    event_times = sorted(
        {time for time, event in zip(stop_times, status, strict=True) if event == 1}
    )
    km_survival = 1.0
    for event_time in event_times:
        at_risk = [0.0] * n_groups
        events = [0.0] * n_groups
        total_events = 0.0
        for idx, stop_time in enumerate(stop_times):
            group_idx = group_codes[idx]
            if entry_times[idx] < event_time <= stop_time:
                at_risk[group_idx] += 1.0
            if status[idx] == 1 and stop_time == event_time:
                events[group_idx] += 1.0
                total_events += 1.0

        total_at_risk = sum(at_risk)
        if total_events <= 0.0 or total_at_risk <= 0.0:
            continue

        weight = km_survival**rho
        for group_idx in range(n_groups):
            observed[group_idx] += weight * events[group_idx]
            expected[group_idx] += weight * total_events * at_risk[group_idx] / total_at_risk

        if total_at_risk > 1.0:
            var_factor = (
                total_events
                * (total_at_risk - total_events)
                / (total_at_risk * total_at_risk * (total_at_risk - 1.0))
            )
            for n_group in at_risk[: n_groups - 1]:
                variance += weight * weight * var_factor * n_group * (total_at_risk - n_group)

        km_survival *= 1.0 - total_events / total_at_risk

    return _survdiff_result_from_summary(observed, expected, variance, rho)


def _exact_survdiff(response: Surv, group: Any, rho: float, timefix: bool) -> Any:
    group_codes = _encode_groups(group, len(response))
    if response.start is not None:
        return _exact_counting_survdiff_result(
            list(response.time),
            list(response.event),
            list(response.start),
            group_codes,
            len(set(group_codes)),
            rho,
        )

    components = _core.survdiff2(
        _survdiff_timefix_values(list(response.time), timefix),
        list(response.event),
        [code + 1 for code in group_codes],
        None,
        rho,
    )
    return _survdiff_result_from_components(components, rho)


def _stratified_survdiff(
    response: Surv,
    group: Any,
    strata: Any,
    rho: float,
    timefix: bool,
) -> Any:
    n = len(response)
    group_codes = _encode_groups(group, n)
    strata_codes = _encode_groups(strata, n)
    times = _survdiff_timefix_values(list(response.time), timefix)
    if response.start is not None:
        n_groups = len(set(group_codes))
        observed = [0.0] * n_groups
        expected = [0.0] * n_groups
        variance = 0.0
        starts = list(response.start)
        for indices in _group_indices(strata_codes, n).values():
            local_groups = [group_codes[idx] for idx in indices]
            if not timefix:
                local_result = _exact_counting_survdiff_result(
                    [times[idx] for idx in indices],
                    [response.event[idx] for idx in indices],
                    [starts[idx] for idx in indices],
                    local_groups,
                    n_groups,
                    rho,
                )
            else:
                local_result = (
                    _core.logrank_test(
                        [times[idx] for idx in indices],
                        [response.event[idx] for idx in indices],
                        local_groups,
                        entry_times=[starts[idx] for idx in indices],
                    )
                    if rho == 0.0
                    else _core.fleming_harrington_test(
                        [times[idx] for idx in indices],
                        [response.event[idx] for idx in indices],
                        local_groups,
                        rho,
                        0.0,
                        entry_times=[starts[idx] for idx in indices],
                    )
                )
            if not timefix:
                for group_idx in range(n_groups):
                    observed[group_idx] += float(local_result.observed[group_idx])
                    expected[group_idx] += float(local_result.expected[group_idx])
            else:
                for local_idx, group_code in enumerate(sorted(set(local_groups))):
                    observed[group_code] += float(local_result.observed[local_idx])
                    expected[group_code] += float(local_result.expected[local_idx])
            variance += float(local_result.variance)

        return _survdiff_result_from_summary(observed, expected, variance, rho)

    order = sorted(range(n), key=lambda idx: (strata_codes[idx], times[idx], idx))
    sorted_strata = [strata_codes[idx] for idx in order]
    components = _core.survdiff2(
        [times[idx] for idx in order],
        [response.event[idx] for idx in order],
        [group_codes[idx] + 1 for idx in order],
        _survdiff_strata_markers(sorted_strata),
        rho,
    )
    return _survdiff_result_from_components(components, rho)


def survdiff(
    response: Surv | str,
    data: Any | None = None,
    *,
    group: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    rho: float = 0.0,
    timefix: bool = True,
    **kwargs: Any,
):
    """Compare survival curves with R's G-rho family for common survdiff use."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", timefix, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survdiff got unexpected keyword argument(s): {unexpected}")

    formula_strata: Any | None = None
    if isinstance(response, str):
        if subset is not None:
            data, _aligned = _subset_formula_inputs(response, data, subset)
            subset = None
        data, _aligned = _apply_formula_na_action(response, data, na_action)
        na_action = "pass"
        response, terms = _parse_formula(response, data)
        _reject_formula_clusters("survdiff", terms)
        group, formula_strata = _survdiff_formula_groups(data, terms, len(response))

    if not isinstance(response, Surv):
        raise TypeError("survdiff response must be a Surv object or formula")
    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        group = _subset_optional_sequence(group, indices, "group")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "survdiff inputs",
        group=group,
        strata=formula_strata,
    )
    group = aligned["group"]
    formula_strata = aligned["strata"]
    if group is None:
        raise ValueError("group is required")
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survdiff currently supports right-censored and counting Surv responses"
        )
    rho_value = _finite_float(rho, "rho")
    fix_time = _normalize_bool_option(timefix, "timefix")
    if formula_strata is not None:
        return _stratified_survdiff(response, group, formula_strata, rho_value, fix_time)
    if not fix_time:
        return _exact_survdiff(response, group, rho_value, fix_time)

    groups = _encode_groups(group, len(response))
    entry_times = list(response.start) if response.start is not None else None
    if rho_value == 0.0:
        return _core.logrank_test(
            list(response.time),
            list(response.event),
            groups,
            entry_times=entry_times,
        )
    return _core.fleming_harrington_test(
        list(response.time),
        list(response.event),
        groups,
        rho_value,
        0.0,
        entry_times=entry_times,
    )


def basehaz(
    fit: Any | None = None,
    status: Any | None = None,
    linear_predictors: Any | None = None,
    centered: bool = True,
    *,
    newdata: Any | None = None,
    time: Any | None = None,
    entry_times: Any | None = None,
    weights: Any | None = None,
):
    """Return Cox baseline cumulative hazard, like R's basehaz."""

    centered_value = _normalize_bool_option(centered, "centered")
    if time is not None:
        if newdata is not None:
            raise ValueError("newdata is only supported with fitted Cox models")
        if fit is not None:
            raise ValueError("use either a fitted Cox model or time=, not both")
        if status is None or linear_predictors is None:
            raise ValueError("status and linear_predictors are required with time=")
        time_values = _float_vector(time, "time")
        return _core.basehaz(
            time_values,
            _event_vector(status, "status"),
            _float_vector(linear_predictors, "linear_predictors"),
            centered_value,
            _optional_float_vector(entry_times, "entry_times", len(time_values)),
            _optional_float_vector(weights, "weights", len(time_values)),
        )

    if fit is None:
        raise TypeError("basehaz requires a fitted Cox model or time/status inputs")
    if hasattr(fit, "basehaz") and status is not None and linear_predictors is None:
        if newdata is not None:
            raise ValueError("use either positional newdata or newdata=, not both")
        newdata = status
        status = None
    if hasattr(fit, "basehaz") and status is None and linear_predictors is None:
        if entry_times is not None:
            raise ValueError("entry_times is already stored on fitted Cox models")
        if weights is not None:
            raise ValueError("weights are already stored on fitted Cox models")
        if newdata is not None:
            rows, offsets = _prediction_inputs(fit, newdata)
            result = _cox_survfit_result(
                fit,
                rows,
                offsets,
                True,
                newdata,
                compute_confidence=False,
            )
            if len(result.cumhaz) == 1:
                curve_strata = result.strata
                strata = [curve_strata[0]] * len(result.time) if curve_strata is not None else None
                return CoxBaseHazardResult(
                    time=result.time,
                    cumhaz=result.cumhaz[0],
                    strata=strata,
                    centered=True,
                    curve_strata=curve_strata,
                )
            return CoxBaseHazardResult(
                time=result.time,
                cumhaz=result.cumhaz,
                centered=True,
                curve_strata=result.strata,
            )
        return _cox_flat_basehaz_with_training_times(fit, centered_value)
    if newdata is not None:
        raise ValueError("newdata is only supported with fitted Cox models")
    if status is None or linear_predictors is None:
        raise ValueError("status and linear_predictors are required with raw time input")
    time_values = _float_vector(fit, "time")
    return _core.basehaz(
        time_values,
        _event_vector(status, "status"),
        _float_vector(linear_predictors, "linear_predictors"),
        centered_value,
        _optional_float_vector(entry_times, "entry_times", len(time_values)),
        _optional_float_vector(weights, "weights", len(time_values)),
    )


def cox_zph(
    fit: Any,
    transform: Any = "km",
    *,
    terms: bool = True,
    singledf: bool = False,
    global_test: bool = True,
    **kwargs: Any,
) -> CoxZPHResult:
    """R-style proportional hazards diagnostic for fitted Cox models."""

    global_test = _pop_dotted_keyword(kwargs, "global", "global_test", global_test, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected cox_zph argument(s): {unexpected}")
    if _is_survreg_fit(fit) or not hasattr(fit, "schoenfeld_residuals"):
        raise TypeError("cox_zph requires a fitted Cox model")
    group_terms = _normalize_bool_option(terms, "terms")
    single_df = _normalize_bool_option(singledf, "singledf")
    include_global = _normalize_bool_option(global_test, "global")

    raw = [[float(value) for value in row] for row in fit.schoenfeld_residuals()]
    scaled = _cox_scaled_schoenfeld_from_raw(fit, raw)
    if len(raw) != len(scaled):
        raise ValueError("Schoenfeld residual arrays have inconsistent lengths")
    if not raw:
        return CoxZPHResult(
            variable_names=[],
            chi2_values=[],
            df=[],
            p_values=[],
            x=[],
            time=[],
            y=[],
            var=[],
            transform=str(transform),
            global_chi2=0.0 if include_global else None,
            global_df=0 if include_global else None,
            global_p_value=1.0 if include_global else None,
        )

    nvar = len(raw[0])
    if any(len(row) != nvar for row in raw) or any(len(row) != nvar for row in scaled):
        raise ValueError("Schoenfeld residual arrays must be rectangular")
    beta = _cox_beta(fit)
    if len(beta) != nvar:
        raise ValueError("fitted Cox model coefficients do not match residual width")

    event_indices = _cox_event_indices(fit)
    if len(event_indices) != len(raw):
        raise ValueError("fitted Cox model event times do not match Schoenfeld residuals")
    event_times = [float(fit.event_times[idx]) for idx in event_indices]
    transform_name, transformed_time = _cox_zph_transform(fit, event_times, transform)
    groups = _cox_zph_column_groups(fit, nvar, group_terms)
    test_residuals = scaled

    variable_names: list[str] = []
    chi2_values: list[float] = []
    df_values: list[int] = []
    p_values: list[float] = []
    for name, columns in groups:
        variable_names.append(name)
        if single_df and len(columns) > 1:
            residual_matrix = [
                [sum(row[col_idx] * beta[col_idx] for col_idx in columns)] for row in test_residuals
            ]
            test = _core.ph_test(residual_matrix, transformed_time, None)
            chi2_values.append(float(test.chi2_values[0]) if test.chi2_values else 0.0)
            df_values.append(1)
            p_values.append(float(test.p_values[0]) if test.p_values else 1.0)
            continue

        test = _core.ph_test(_matrix_columns(test_residuals, columns), transformed_time, None)
        chi2_values.append(float(test.global_chi2))
        df_values.append(int(test.global_df))
        p_values.append(float(test.global_p_value))

    global_result = (
        _core.ph_test(test_residuals, transformed_time, None) if include_global else None
    )
    grouped_y = (
        _cox_zph_term_matrix(scaled, groups, beta)
        if group_terms
        else _matrix_columns(scaled, [idx for _name, columns in groups for idx in columns])
    )
    return CoxZPHResult(
        variable_names=variable_names,
        chi2_values=chi2_values,
        df=df_values,
        p_values=p_values,
        x=transformed_time,
        time=event_times,
        y=grouped_y,
        var=_cox_zph_group_variance(fit, groups, beta),
        transform=transform_name,
        global_chi2=float(global_result.global_chi2) if global_result is not None else None,
        global_df=int(global_result.global_df) if global_result is not None else None,
        global_p_value=(float(global_result.global_p_value) if global_result is not None else None),
    )


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
        "log-gaussian": "lognormal",
        "log-logistic": "loglogistic",
        "extreme": "extreme_value",
        "extreme value": "extreme_value",
        "extreme-value": "extreme_value",
        "extremevalue": "extreme_value",
    }
    if value in aliases:
        return aliases[value]
    message = (
        "distribution must be one of weibull, exponential, extreme, gaussian, "
        "logistic, lognormal, or loglogistic"
    )
    return _match_string_arg(
        value,
        "distribution",
        (
            "weibull",
            "exponential",
            "extreme_value",
            "gaussian",
            "logistic",
            "lognormal",
            "loglogistic",
        ),
        message,
    )


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


def _cox_deviance_from_martingale(martingale: list[float], status: list[float]) -> list[float]:
    if len(martingale) != len(status):
        raise ValueError("status must have the same length as martingale residuals")
    residuals: list[float] = []
    for residual, event_count in zip(martingale, status, strict=True):
        log_term = 0.0
        if event_count > 0.0:
            expected = max(event_count - residual, 1e-12)
            log_term = event_count * math.log(expected)
        magnitude = math.sqrt(max(-2.0 * (residual + log_term), 0.0))
        residuals.append(magnitude if residual >= 0.0 else -magnitude)
    return residuals


def _cox_term_contributions(fit: Any, n: int) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    covariates = getattr(fit, "covariates", None)
    if covariates is None:
        raise TypeError("model does not expose fitted covariates")
    rows = [list(row) for row in covariates]
    if len(rows) != n:
        raise ValueError("fitted Cox model covariates do not match residual length")
    if any(len(row) != nvar for row in rows):
        raise ValueError("fitted Cox model covariates do not match coefficient width")
    return [[float(row[col_idx]) * beta[col_idx] for col_idx in range(nvar)] for row in rows]


def _cox_predict_term_groups(fit: Any, nvar: int) -> list[tuple[str, list[int]]]:
    design = _formula_design_for_fit(fit)
    if design is None:
        return [(f"x{idx + 1}", [idx]) for idx in range(nvar)]

    groups: list[tuple[str, list[int]]] = []
    cursor = 1 if design.intercept else 0
    for term in design.covariates:
        output_names = _design_term_output_names(term)
        indices = list(range(cursor, cursor + len(output_names)))
        groups.append((_design_term_name(term), indices))
        cursor += len(output_names)

    if cursor != nvar:
        return [(f"x{idx + 1}", [idx]) for idx in range(nvar)]
    return groups


def _predict_terms_selection(terms: Any | None, names: list[str]) -> list[int]:
    if terms is None:
        return list(range(len(names)))
    requested = [terms] if isinstance(terms, str) else _coerce_array_like(terms, "terms")

    selected: list[int] = []
    for value in requested:
        if isinstance(value, str):
            try:
                term_idx = names.index(value)
            except ValueError as exc:
                raise ValueError(f"terms contains unknown model term {value!r}") from exc
        else:
            try:
                term_idx = index(value) - 1
            except TypeError as exc:
                raise TypeError("terms must contain term names or 1-based term indices") from exc
            if term_idx < 0 or term_idx >= len(names):
                raise ValueError("terms indices must be between 1 and the number of model terms")
        if term_idx not in selected:
            selected.append(term_idx)
    return selected


def _cox_predict_terms(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
    reference: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    if rows is None:
        covariates = getattr(fit, "covariates", None)
        if covariates is None:
            raise ValueError("newdata is required for predict type='terms'")
        rows = [[float(value) for value in row] for row in covariates]
    if any(len(row) != len(beta) for row in rows):
        raise ValueError(f"newdata must have {len(beta)} columns")

    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    means_by_row = _cox_reference_means_for_rows(fit, reference, rows, newdata)
    return [
        [
            sum(
                (float(row[col_idx]) - means_by_row[row_idx][col_idx]) * beta[col_idx]
                for col_idx in groups[group_idx][1]
            )
            for group_idx in selected
        ]
        for row_idx, row in enumerate(rows)
    ]


def _cox_partial_residuals(
    fit: Any,
    terms: Any | None,
    martingale_weights: list[float] | None = None,
) -> list[list[float]]:
    martingale_method = getattr(fit, "martingale_residuals", None)
    if martingale_method is None:
        raise TypeError("model does not support partial residuals")
    martingale = [float(value) for value in martingale_method()]
    if martingale_weights is not None:
        if len(martingale_weights) != len(martingale):
            raise ValueError("weights must have the same length as martingale residuals")
        martingale = [
            residual * float(weight)
            for residual, weight in zip(martingale, martingale_weights, strict=True)
        ]
    contributions = _cox_term_contributions(fit, len(martingale))
    if terms is None:
        return [
            [martingale[row_idx] + term for term in row_terms]
            for row_idx, row_terms in enumerate(contributions)
        ]

    groups = _cox_predict_term_groups(fit, len(_cox_beta(fit)))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    return [
        [
            martingale[row_idx]
            + sum(contributions[row_idx][col_idx] for col_idx in groups[group_idx][1])
            for group_idx in selected
        ]
        for row_idx in range(len(martingale))
    ]


def _cox_event_indices(fit: Any) -> list[int]:
    status = [int(value) for value in fit.status]
    times = [float(value) for value in fit.event_times]
    strata_values = fit.strata if hasattr(fit, "strata") else [0] * len(status)
    strata = [int(value) for value in strata_values]
    if len(times) != len(status) or len(strata) != len(status):
        raise ValueError("fitted Cox model event arrays have inconsistent lengths")

    order = sorted(range(len(status)), key=lambda idx: (strata[idx], times[idx], idx))
    event_indices: list[int] = []
    stratum_start = 0
    while stratum_start < len(order):
        stratum = strata[order[stratum_start]]
        stratum_end = stratum_start
        while stratum_end + 1 < len(order) and strata[order[stratum_end + 1]] == stratum:
            stratum_end += 1

        time_start = stratum_start
        while time_start <= stratum_end:
            event_time = times[order[time_start]]
            time_end = time_start
            while time_end < stratum_end and times[order[time_end + 1]] == event_time:
                time_end += 1
            event_indices.extend(
                order[pos] for pos in range(time_start, time_end + 1) if status[order[pos]] == 1
            )
            time_start = time_end + 1

        stratum_start = stratum_end + 1

    return event_indices


def _cox_scaled_schoenfeld_from_raw(fit: Any, raw: list[list[float]]) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    if nvar == 0 or not raw:
        return raw
    variance = getattr(fit, "information_matrix", None)
    if variance is None:
        raise TypeError("model does not expose coefficient variance")
    matrix = [list(row) for row in variance]
    if len(matrix) != nvar or any(len(row) != nvar for row in matrix):
        raise ValueError("fitted Cox model information matrix does not match coefficient width")
    if any(len(row) != nvar for row in raw):
        raise ValueError("Schoenfeld residuals do not match coefficient width")
    event_count = len(raw)
    return [
        [
            beta[col_idx]
            + event_count
            * sum(row[inner_idx] * matrix[inner_idx][col_idx] for inner_idx in range(nvar))
            for col_idx in range(nvar)
        ]
        for row in raw
    ]


def _cox_dfbeta_from_score_residuals(fit: Any, *, scaled: bool) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    score_method = getattr(fit, "score_residuals", None)
    if score_method is None:
        raise TypeError("model does not expose score residuals")
    score = [[float(value) for value in row] for row in score_method()]
    if nvar == 0:
        return score
    variance = getattr(fit, "information_matrix", None)
    if variance is None:
        raise TypeError("model does not expose coefficient variance")
    matrix = [list(row) for row in variance]
    if len(matrix) != nvar or any(len(row) != nvar for row in matrix):
        raise ValueError("fitted Cox model information matrix does not match coefficient width")
    if any(len(row) != nvar for row in score):
        raise ValueError("score residuals do not match coefficient width")

    scales = (
        [
            max(math.sqrt(abs(matrix[col_idx][col_idx])), _COX_DFBETAS_SCALE_FLOOR)
            for col_idx in range(nvar)
        ]
        if scaled
        else [1.0] * nvar
    )
    return [
        [
            sum(matrix[col_idx][inner_idx] * row[inner_idx] for inner_idx in range(nvar))
            / scales[col_idx]
            for col_idx in range(nvar)
        ]
        for row in score
    ]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2.0
        for pos in range(start, end + 1):
            ranks[order[pos]] = average
        start = end + 1
    return ranks


def _cox_zph_km_transform(fit: Any, event_times: list[float]) -> list[float]:
    all_times = [float(value) for value in fit.event_times]
    status = [int(value) for value in fit.status]
    entry_times = getattr(fit, "entry_times", None)
    km = _core.survfitkm(
        all_times,
        status,
        entry_times=[float(value) for value in entry_times] if entry_times is not None else None,
        conf_type="none",
    )
    curve_times = [float(value) for value in km.time]
    estimates = [float(value) for value in km.estimate]
    transformed: list[float] = []
    cursor = 0
    for event_time in event_times:
        while (
            cursor < len(curve_times) and curve_times[cursor] < event_time - _SURVFIT_TIME_EPSILON
        ):
            cursor += 1
        previous_survival = estimates[cursor - 1] if cursor > 0 else 1.0
        transformed.append(1.0 - previous_survival)
    return transformed


def _cox_zph_transform(
    fit: Any,
    event_times: list[float],
    transform: Any,
) -> tuple[str, list[float]]:
    if callable(transform):
        transformed = _float_vector(transform(event_times), "transform result")
        if len(transformed) != len(event_times):
            raise ValueError("transform result must have the same length as event times")
        return getattr(transform, "__name__", "user"), transformed

    message = "transform must be 'km', 'rank', 'identity', 'log', or a callable"
    transform_name = "km" if transform is None else str(transform).strip().lower()
    transform_name = transform_name.replace("_", "-")
    normalized = (
        "km"
        if transform_name in {"kaplan", "kaplan-meier"}
        else _match_string_arg(
            transform_name,
            "transform",
            ("km", "rank", "identity", "log"),
            message,
        )
    )

    if normalized == "km":
        return "km", _cox_zph_km_transform(fit, event_times)
    if normalized == "rank":
        return "rank", _average_ranks(event_times)
    if normalized == "log":
        if any(value <= 0.0 for value in event_times):
            raise ValueError("log transform requires positive event times")
        return "log", [math.log(value) for value in event_times]
    return "identity", event_times


def _matrix_columns(rows: list[list[float]], columns: list[int]) -> list[list[float]]:
    return [[float(row[col_idx]) for col_idx in columns] for row in rows]


def _cox_zph_column_groups(
    fit: Any,
    nvar: int,
    terms: bool,
) -> list[tuple[str, list[int]]]:
    design = _formula_design_for_fit(fit)
    if design is None:
        return [(f"var{idx}", [idx]) for idx in range(nvar)]

    groups: list[tuple[str, list[int]]] = []
    cursor = 0
    for term in design.covariates:
        output_names = _design_term_output_names(term)
        indices = list(range(cursor, cursor + len(output_names)))
        if terms:
            groups.append((_design_term_name(term), indices))
        else:
            groups.extend((name, [idx]) for name, idx in zip(output_names, indices, strict=True))
        cursor += len(output_names)

    if cursor != nvar:
        return [(f"var{idx}", [idx]) for idx in range(nvar)]
    return groups


def _cox_zph_term_matrix(
    scaled: list[list[float]],
    groups: list[tuple[str, list[int]]],
    beta: list[float],
) -> list[list[float]]:
    result: list[list[float]] = []
    for row in scaled:
        grouped_row: list[float] = []
        for _name, columns in groups:
            if len(columns) == 1:
                grouped_row.append(float(row[columns[0]]))
            else:
                grouped_row.append(sum(float(row[col_idx]) * beta[col_idx] for col_idx in columns))
        result.append(grouped_row)
    return result


def _cox_zph_group_variance(
    fit: Any,
    groups: list[tuple[str, list[int]]],
    beta: list[float],
) -> list[list[float]]:
    raw_variance = getattr(fit, "information_matrix", None)
    if raw_variance is None:
        return []
    variance = [[float(value) for value in row] for row in raw_variance]
    nvar = len(beta)
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        return []

    loadings: list[list[float]] = []
    for _name, columns in groups:
        loading = [0.0] * nvar
        for col_idx in columns:
            loading[col_idx] = beta[col_idx] if len(columns) > 1 else 1.0
        loadings.append(loading)
    return [
        [
            sum(left[i] * variance[i][j] * right[j] for i in range(nvar) for j in range(nvar))
            for right in loadings
        ]
        for left in loadings
    ]


def _cox_beta(fit: Any) -> list[float]:
    coefficients = getattr(fit, "coefficients", None)
    if coefficients is None:
        raise TypeError("model does not expose fitted coefficients")
    values = list(coefficients)
    if values and isinstance(values[0], list | tuple):
        return [float(value) for value in values[0]]
    return [float(value) for value in values]


def _unwrap_formula_fit(fit: Any) -> Any:
    return fit.fit if isinstance(fit, _FormulaFit) else fit


def _is_coxph_fit(fit: Any) -> bool:
    model = _unwrap_formula_fit(fit)
    return all(
        hasattr(model, name)
        for name in (
            "coefficients",
            "covariates",
            "event_times",
            "status",
            "log_likelihood",
        )
    ) and not _is_survreg_fit(model)


def _require_coxph_fit(fit: Any) -> Any:
    if not _is_coxph_fit(fit):
        raise TypeError("anova requires fitted Cox model objects")
    return _unwrap_formula_fit(fit)


def _cox_loglik_values(fit: Any) -> list[float]:
    values = [float(value) for value in getattr(fit, "log_likelihood", [])]
    if len(values) < 2:
        raise ValueError("fitted Cox model must expose null and fitted log likelihoods")
    return values


def _cox_full_loglik(fit: Any) -> float:
    return _cox_loglik_values(fit)[-1]


def _cox_degrees_of_freedom(fit: Any) -> int:
    return len(_cox_beta(fit))


def _formula_design_output_names(design: _FormulaDesign) -> list[str]:
    names = [name for term in design.covariates for name in _design_term_output_names(term)]
    if design.intercept:
        names.insert(0, "(Intercept)")
    return names


def _fallback_coef_names(width: int) -> list[str]:
    return [f"x{idx + 1}" for idx in range(width)]


def _fit_location_coef_names(fit: Any, width: int) -> list[str]:
    design = _formula_design_for_fit(fit)
    if design is not None:
        names = _formula_design_output_names(design)
        if len(names) == width:
            return names
    return _fallback_coef_names(width)


def _survreg_scale_coef_names(fit: Any, width: int) -> list[str]:
    if width <= 0:
        return []
    if width == 1:
        return ["Log(scale)"]
    design = _formula_design_for_fit(fit)
    if design is not None and len(design.strata_levels) == width:
        return [f"Log(scale:{level})" for level in design.strata_levels]
    return [f"Log(scale{idx + 1})" for idx in range(width)]


def _is_model_fit(fit: Any) -> bool:
    return _is_coxph_fit(fit) or _is_survreg_fit(fit)


def _require_model_fit(fit: Any, generic: str) -> Any:
    if not _is_model_fit(fit):
        raise TypeError(f"{generic} requires a fitted coxph or survreg model")
    return fit


def coef(fit: Any) -> list[float]:
    """Return fitted model coefficients, like R's coef generic."""

    _require_model_fit(fit, "coef")
    if _is_survreg_fit(fit):
        return _location_beta(fit)
    return _cox_beta(fit)


def coef_names(fit: Any, *, complete: Any = False) -> list[str]:
    """Return fitted coefficient names for R-style model helpers."""

    _require_model_fit(fit, "coef_names")
    include_complete = _normalize_bool_option_with_default(complete, "complete", False)
    if _is_survreg_fit(fit):
        location_width = len(_location_beta(fit))
        names = _fit_location_coef_names(fit, location_width)
        if include_complete:
            total_width = len(list(fit.coefficients))
            names.extend(_survreg_scale_coef_names(fit, total_width - location_width))
        return names

    beta = _cox_beta(fit)
    return _fit_location_coef_names(fit, len(beta))


def vcov(fit: Any, *, complete: Any = True) -> list[list[float]]:
    """Return a fitted model variance-covariance matrix, like R's vcov generic."""

    _require_model_fit(fit, "vcov")
    include_complete = _normalize_bool_option_with_default(complete, "complete", True)
    if _is_survreg_fit(fit):
        width = len(list(fit.coefficients)) if include_complete else len(_location_beta(fit))
        return _survreg_variance_matrix(fit, width)
    return _cox_variance_matrix(fit, len(_cox_beta(fit)))


def loglik(fit: Any) -> float:
    """Return a fitted model log likelihood."""

    _require_model_fit(fit, "loglik")
    if _is_survreg_fit(fit):
        return _survreg_original_scale_loglik(fit)
    return _cox_full_loglik(_unwrap_formula_fit(fit))


def nobs(fit: Any) -> int:
    """Return the number of observations used by a fitted model."""

    _require_model_fit(fit, "nobs")
    values = getattr(fit, "status", None)
    if values is None:
        values = getattr(fit, "event_times", None)
    if values is None:
        raise TypeError("nobs requires a fitted model with stored observations")
    return len(list(values))


def degrees_freedom(fit: Any) -> int:
    """Return the number of fitted parameters counted by model log likelihoods."""

    _require_model_fit(fit, "degrees_freedom")
    if _is_survreg_fit(fit):
        return len(list(fit.coefficients))
    return _cox_degrees_of_freedom(_unwrap_formula_fit(fit))


def df_residual(fit: Any) -> int:
    """Return residual degrees of freedom for fitted ``survreg`` models."""

    _require_model_fit(fit, "df_residual")
    if not _is_survreg_fit(fit):
        raise TypeError("df_residual is only defined for fitted survreg models")
    return nobs(fit) - degrees_freedom(fit)


def _finite_numeric_option(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def aic(fit: Any, *, k: Any = 2.0) -> float:
    """Return Akaike-style information criterion for a fitted model."""

    penalty = _finite_numeric_option(k, "k")
    return -2.0 * loglik(fit) + penalty * degrees_freedom(fit)


def bic(fit: Any) -> float:
    """Return Bayesian information criterion for a fitted model."""

    return aic(fit, k=math.log(nobs(fit)))


def extract_aic(fit: Any, *, scale: Any = 0.0, k: Any = 2.0) -> list[float]:
    """Return ``[df, AIC]`` like R's ``extractAIC`` generic."""

    _finite_numeric_option(scale, "scale")
    return [float(degrees_freedom(fit)), aic(fit, k=k)]


def model_formula(fit: Any) -> str:
    """Return the formula string used to create a formula-based model fit."""

    _require_model_fit(fit, "model_formula")
    if isinstance(fit, _FormulaFit) and fit.formula is not None:
        return fit.formula
    raise TypeError("model_formula requires a formula-based fitted model")


def model_weights(fit: Any) -> list[float] | None:
    """Return explicit case weights for a fitted model, or ``None`` when absent."""

    _require_model_fit(fit, "model_weights")
    if isinstance(fit, _FormulaFit) and fit.case_weights is not None:
        return list(fit.case_weights)
    values = getattr(_unwrap_formula_fit(fit), "weights", None)
    if values is None:
        return None
    weights = [float(value) for value in _materialize_1d(values, "weights")]
    if all(abs(value - 1.0) <= 1e-12 for value in weights):
        return None
    return weights


def _model_matrix_column_names(fit: Any, width: int) -> list[str]:
    design = _formula_design_for_fit(fit)
    if design is not None:
        names = _formula_design_output_names(design)
        if len(names) == width:
            return names
    if _is_model_fit(fit):
        names = coef_names(fit)
        if len(names) == width:
            return names
    return _fallback_coef_names(width)


def model_matrix(fit: Any) -> dict[str, Any]:
    """Return the training design matrix and column names for a fitted model."""

    _require_model_fit(fit, "model_matrix")
    rows = getattr(fit, "covariates", None)
    if rows is None:
        rows = getattr(fit, "x", None)
    if rows is None:
        raise TypeError("model_matrix requires a fitted model with stored covariates")
    matrix = [[float(value) for value in row] for row in rows]
    width = len(matrix[0]) if matrix else 0
    if any(len(row) != width for row in matrix):
        raise ValueError("stored model matrix must be rectangular")
    return {
        "data": matrix,
        "columns": _model_matrix_column_names(fit, width),
    }


def _model_frame_surv_columns(response: Surv, existing: set[str]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    if response.start is not None:
        if "start" not in existing:
            columns["start"] = list(response.start)
        if "stop" not in existing:
            columns["stop"] = list(response.time)
    elif "time" not in existing:
        columns["time"] = list(response.time)
    if response.time2 is not None and "time2" not in existing:
        columns["time2"] = list(response.time2)
    if "status" not in existing:
        columns["status"] = list(response.event)
    return columns


def model_frame(fit: Any) -> dict[str, list[Any]]:
    """Return a plain stored model frame for fits created with ``model=True``."""

    _require_model_fit(fit, "model_frame")
    frame = getattr(fit, "model", None)
    if frame is None:
        raise TypeError("model_frame requires fitting with model=True")
    if not isinstance(frame, Mapping):
        raise TypeError("stored model frame must be mapping-like")

    columns: dict[str, list[Any]] = {}
    for name, values in frame.items():
        if isinstance(values, Surv):
            columns.update(_model_frame_surv_columns(values, set(columns)))
            continue
        if isinstance(values, Mapping):
            continue
        materialized = _materialize_1d(values, str(name))
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns[str(name)] = list(materialized)
    return columns


def fitted(
    fit: Any,
    *,
    type: str | None = None,  # noqa: A002
    centered: bool | None = None,
    terms: Any | None = None,
    collapse: Any = False,
    reference: str | None = None,
    se_fit: bool = False,
    times: Any | None = None,
    p: Any | None = None,
    quantiles: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return fitted values for the training observations of a model."""

    _require_model_fit(fit, "fitted")
    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"fitted got unexpected keyword argument(s): {unexpected}")

    return predict(
        fit,
        type=type,
        centered=centered,
        terms=terms,
        collapse=collapse,
        reference=reference,
        se_fit=se_fit,
        times=times,
        p=p,
        quantiles=quantiles,
    )


def _normal_two_sided_p_value(statistic: float) -> float:
    if math.isnan(statistic):
        return math.nan
    if math.isinf(statistic):
        return 0.0
    return 2.0 * NormalDist().cdf(-abs(statistic))


def _coefficient_summary_rows(
    names: list[str],
    coefficients: list[float],
    variance: list[list[float]],
) -> list[dict[str, float | str]]:
    if len(names) != len(coefficients):
        raise ValueError("coefficient names do not match coefficient width")
    if len(variance) != len(coefficients) or any(len(row) != len(coefficients) for row in variance):
        raise ValueError("variance matrix does not match coefficient width")

    rows = []
    for idx, value in enumerate(coefficients):
        standard_error = math.sqrt(max(float(variance[idx][idx]), 0.0))
        if standard_error > 0.0:
            statistic = value / standard_error
        elif value == 0.0:
            statistic = 0.0
        else:
            statistic = math.copysign(math.inf, value)
        rows.append(
            {
                "name": names[idx],
                "coef": value,
                "se": standard_error,
                "statistic": statistic,
                "p": _normal_two_sided_p_value(statistic),
            }
        )
    return rows


def _coefficient_selection_indices(parm: Any, names: list[str]) -> list[int]:
    if parm is None:
        return list(range(len(names)))

    if isinstance(parm, str):
        values: list[Any] = [parm]
    elif isinstance(parm, bool):
        raise TypeError("parm must be coefficient names or 1-based indices")
    else:
        if isinstance(parm, Sequence) and not isinstance(parm, bytes):
            values = list(_materialize_1d(parm, "parm"))
        else:
            values = [parm]

    indices: list[int] = []
    for value in values:
        if isinstance(value, str):
            try:
                idx = names.index(value)
            except ValueError as exc:
                raise ValueError(f"unknown coefficient name {value!r}") from exc
        else:
            if isinstance(value, bool):
                raise TypeError("parm must be coefficient names or 1-based indices")
            try:
                raw_idx = index(value)
            except TypeError:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError("parm must be coefficient names or 1-based indices") from exc
                if not numeric.is_integer():
                    raise TypeError("parm must be coefficient names or 1-based indices") from None
                raw_idx = int(numeric)
            except ValueError as exc:
                raise TypeError("parm must be coefficient names or 1-based indices") from exc
            idx = raw_idx - 1
            if idx < 0 or idx >= len(names):
                raise IndexError("parm index out of range")
        indices.append(idx)
    return indices


def confint(
    fit: Any,
    parm: Any | None = None,
    *,
    level: Any = 0.95,
) -> list[dict[str, float | str]]:
    """Return normal-approximation confidence intervals for model coefficients."""

    _require_model_fit(fit, "confint")
    confidence_level = _normalize_conf_level(level, "level")
    alpha = 1.0 - confidence_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    names = coef_names(fit)
    coefficients = coef(fit)
    variance = vcov(fit, complete=False)
    indices = _coefficient_selection_indices(parm, names)

    intervals = []
    for idx in indices:
        standard_error = math.sqrt(max(float(variance[idx][idx]), 0.0))
        margin = z * standard_error
        intervals.append(
            {
                "name": names[idx],
                "lower": coefficients[idx] - margin,
                "upper": coefficients[idx] + margin,
            }
        )
    return intervals


def model_summary(fit: Any) -> dict[str, Any]:
    """Return a compact R-style model summary as plain Python data."""

    _require_model_fit(fit, "model_summary")
    names = coef_names(fit)
    coefficients = coef(fit)
    variance = vcov(fit, complete=False)
    result: dict[str, Any] = {
        "model_type": "survreg" if _is_survreg_fit(fit) else "coxph",
        "coefficients": _coefficient_summary_rows(names, coefficients, variance),
        "coefficient_names": names,
        "loglik": loglik(fit),
        "df": degrees_freedom(fit),
        "n": nobs(fit),
        "robust": bool(getattr(fit, "robust", False)),
    }
    if _is_survreg_fit(fit):
        result["scale"] = float(fit.scale)
        result["scales"] = _survreg_scales(fit)
        distribution = getattr(fit, "distribution", None)
        if distribution is not None:
            result["distribution"] = str(distribution)
    else:
        model = _unwrap_formula_fit(fit)
        logliks = _cox_loglik_values(model)
        result["null_loglik"] = logliks[0]
        result["n_event"] = sum(1 for event in model.status if int(event) == 1)
        result["method"] = str(getattr(model, "method", "breslow"))
    return result


def _empty_columns(names: tuple[str, ...]) -> dict[str, list[Any]]:
    return {name: [] for name in names}


def _survfit_frame(result: SurvfitResult) -> dict[str, list[Any]]:
    std_err = list(result.std_err)
    conf_lower = list(result.conf_lower)
    conf_upper = list(result.conf_upper)
    for idx, survival in enumerate(result.estimate):
        if survival <= 0.0:
            if idx < len(std_err):
                std_err[idx] = math.nan
            if idx < len(conf_lower):
                conf_lower[idx] = math.nan
            if idx < len(conf_upper):
                conf_upper[idx] = math.nan
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "surv": result.estimate,
        "std.err": std_err,
        "lower": conf_lower,
        "upper": conf_upper,
        "cumhaz": result.cumhaz,
        "std.chaz": result.std_chaz,
    }
    if result.n_enter is not None:
        frame["n.enter"] = result.n_enter
    return frame


def _turnbull_survfit_frame(result: TurnbullSurvfitResult) -> dict[str, list[Any]]:
    return {
        "time": result.time_points,
        "surv": result.survival,
        "lower": result.survival_lower,
        "upper": result.survival_upper,
    }


def _grouped_survfit_frame(result: Mapping[Any, Any]) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {}
    for label, curve in result.items():
        curve_frame = as_data_frame(curve)
        if not curve_frame:
            continue
        n_rows = len(next(iter(curve_frame.values())))
        if not frame:
            frame = {"strata": []}
            for name in curve_frame:
                frame[name] = []
        elif set(curve_frame) != set(frame) - {"strata"}:
            raise ValueError("grouped survfit results must share tabular columns")
        frame["strata"].extend([str(label)] * n_rows)
        for name, values in curve_frame.items():
            frame[name].extend(values)
    return frame


def _raw_survfit_frame(result: Any) -> dict[str, list[Any]]:
    return _survfit_frame(
        SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[float(value) for value in result.std_err],
            conf_lower=[float(value) for value in result.conf_lower],
            conf_upper=[float(value) for value in result.conf_upper],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[float(value) for value in result.std_chaz],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
        )
    )


def _raw_turnbull_survfit_frame(result: Any) -> dict[str, list[Any]]:
    return _turnbull_survfit_frame(
        TurnbullSurvfitResult(
            time_points=[float(value) for value in result.time_points],
            survival=[float(value) for value in result.survival],
            survival_lower=[float(value) for value in result.survival_lower],
            survival_upper=[float(value) for value in result.survival_upper],
            n_iter=int(result.n_iter),
            converged=bool(result.converged),
        )
    )


def _cox_basehaz_frame(result: CoxBaseHazardResult) -> dict[str, list[Any]]:
    if result.cumhaz and isinstance(result.cumhaz[0], Sequence):
        frame: dict[str, list[Any]] = {
            "curve": [],
            "time": [],
            "cumhaz": [],
        }
        if result.curve_strata is not None:
            frame["strata"] = []
        for curve_idx, curve in enumerate(result.cumhaz):
            if isinstance(curve, (str, bytes)):
                raise TypeError("basehaz cumulative hazards must be numeric")
            if len(curve) != len(result.time):
                raise ValueError("basehaz curve length must match time length")
            frame["curve"].extend([curve_idx + 1] * len(result.time))
            frame["time"].extend(result.time)
            frame["cumhaz"].extend([float(value) for value in curve])
            if result.curve_strata is not None:
                frame["strata"].extend([result.curve_strata[curve_idx]] * len(result.time))
        return frame

    frame = {
        "time": result.time,
        "cumhaz": [float(value) for value in result.cumhaz],
    }
    if result.strata is not None:
        frame["strata"] = result.strata
    return frame


def _cox_survfit_optional_curve_column(
    values: list[list[float]],
    curve_idx: int,
    time_count: int,
) -> list[float] | None:
    if len(values) <= curve_idx:
        return None
    column = values[curve_idx]
    if len(column) != time_count:
        raise ValueError("Cox survfit curve columns must match time length")
    return column


def _cox_survfit_frame(result: CoxSurvfitResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "curve": [],
        "time": [],
        "surv": [],
        "cumhaz": [],
        "linear.predictor": [],
    }
    if result.strata is not None:
        frame["strata"] = []
    if result.start_time is not None:
        frame["start.time"] = []

    optional_columns = {
        "std.err": result.std_err,
        "std.chaz": result.std_chaz,
        "lower": result.conf_lower,
        "upper": result.conf_upper,
    }
    active_optional = {name: values for name, values in optional_columns.items() if values}
    for name in active_optional:
        frame[name] = []

    for curve_idx, (surv_curve, cumhaz_curve, linear_predictor) in enumerate(
        zip(result.surv, result.cumhaz, result.linear_predictors, strict=True)
    ):
        if len(surv_curve) != len(result.time) or len(cumhaz_curve) != len(result.time):
            raise ValueError("Cox survfit curves must match time length")
        n_times = len(result.time)
        frame["curve"].extend([curve_idx + 1] * n_times)
        frame["time"].extend(result.time)
        frame["surv"].extend(surv_curve)
        frame["cumhaz"].extend(cumhaz_curve)
        frame["linear.predictor"].extend([linear_predictor] * n_times)
        if result.strata is not None:
            frame["strata"].extend([result.strata[curve_idx]] * n_times)
        if result.start_time is not None:
            frame["start.time"].extend([result.start_time] * n_times)
        for name, values in active_optional.items():
            optional_curve = _cox_survfit_optional_curve_column(values, curve_idx, n_times)
            if optional_curve is not None:
                frame[name].extend(optional_curve)
    return frame


def _survdiff_frame(result: Any) -> dict[str, list[Any]]:
    observed = [float(value) for value in result.observed]
    expected = [float(value) for value in result.expected]
    if len(observed) != len(expected):
        raise ValueError("survdiff observed and expected lengths differ")
    variance = getattr(result, "variance", None)
    if isinstance(variance, int | float):
        variance_diag = [float(variance)] * len(observed)
    elif variance is not None:
        variance_diag = [float(row[idx]) for idx, row in enumerate(variance)]
    else:
        variance_diag = [math.nan] * len(observed)
    return {
        "group": [idx + 1 for idx in range(len(observed))],
        "observed": observed,
        "expected": expected,
        "variance": variance_diag,
    }


def _cox_zph_frame(result: CoxZPHResult) -> dict[str, list[Any]]:
    rows = result.table
    return {
        "name": [str(row["name"]) for row in rows],
        "chisq": [float(row["chisq"]) for row in rows],
        "df": [int(row["df"]) for row in rows],
        "p": [float(row["p"]) for row in rows],
    }


def _coxph_detail_frame(result: CoxPHDetailResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "n.event": result.nevent,
        "n.risk": result.nrisk,
        "hazard": result.hazard,
        "varhaz": result.varhaz,
        "cumhaz": result.cumulative_hazard,
        "wtrisk": result.wtrisk,
    }
    if result.nevent_wt is not None:
        frame["n.event.weight"] = result.nevent_wt
    if result.nrisk_wt is not None:
        frame["n.risk.weight"] = result.nrisk_wt
    if result.strata is not None:
        frame["strata"] = []
        for stratum, count in result.strata.items():
            frame["strata"].extend([stratum] * int(count))
    return frame


def _anova_frame(result: Any) -> dict[str, list[Any]]:
    rows = list(result.rows)
    return {
        "model": [str(row.model_name) for row in rows],
        "loglik": [float(row.loglik) for row in rows],
        "df": [int(row.df) for row in rows],
        "chisq": [math.nan if row.chisq is None else float(row.chisq) for row in rows],
        "p": [math.nan if row.p_value is None else float(row.p_value) for row in rows],
    }


def _concordance_frame(result: ConcordanceResult) -> dict[str, list[Any]]:
    if isinstance(result.concordance, list):
        n_scores = len(result.concordance)
        score_names = result.score_names or [f"score{idx + 1}" for idx in range(n_scores)]
        variance = result.variance if isinstance(result.variance, list) else [math.nan] * n_scores
        return {
            "score": score_names,
            "concordance": [float(value) for value in result.concordance],
            "concordant": [float(value) for value in result.concordant],
            "comparable": [float(value) for value in result.comparable],
            "n": [result.n] * n_scores,
            "n.event": [result.n_event] * n_scores,
            "variance": [math.nan if value is None else float(value) for value in variance],
        }

    variance_value = result.variance if isinstance(result.variance, int | float) else math.nan
    return {
        "score": [result.score_names[0] if result.score_names else "score"],
        "concordance": [float(result.concordance)],
        "concordant": [float(result.concordant)],
        "comparable": [float(result.comparable)],
        "n": [result.n],
        "n.event": [result.n_event],
        "variance": [float(variance_value)],
    }


def _surv_response_frame(response: Surv) -> dict[str, list[Any]]:
    if response.start is not None:
        frame: dict[str, list[Any]] = {
            "start": list(response.start),
            "stop": list(response.time),
            "status": list(response.event),
        }
    else:
        frame = {
            "time": list(response.time),
            "status": list(response.event),
        }
        if response.time2 is not None:
            frame["time2"] = list(response.time2)
    frame["type"] = [response.type] * len(response)
    return frame


def as_data_frame(result: Any) -> dict[str, list[Any]]:
    """Return a plain column-oriented table for common R-style result objects."""

    if isinstance(result, Surv):
        return _surv_response_frame(result)
    if isinstance(result, CoxSurvfitResult):
        return _cox_survfit_frame(result)
    if isinstance(result, CoxBaseHazardResult):
        return _cox_basehaz_frame(result)
    if isinstance(result, SurvfitResult):
        return _survfit_frame(result)
    if isinstance(result, TurnbullSurvfitResult):
        return _turnbull_survfit_frame(result)
    if all(
        hasattr(result, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    ):
        return _raw_survfit_frame(result)
    if all(
        hasattr(result, name)
        for name in ("time_points", "survival", "survival_lower", "survival_upper")
    ):
        return _raw_turnbull_survfit_frame(result)
    if isinstance(result, CoxZPHResult):
        return _cox_zph_frame(result)
    if isinstance(result, CoxPHDetailResult):
        return _coxph_detail_frame(result)
    if isinstance(result, ConcordanceResult):
        return _concordance_frame(result)
    if isinstance(result, Mapping):
        return _grouped_survfit_frame(result)
    if hasattr(result, "observed") and hasattr(result, "expected") and hasattr(result, "variance"):
        return _survdiff_frame(result)
    if hasattr(result, "rows") and hasattr(result, "test_type"):
        return _anova_frame(result)
    raise TypeError("as_data_frame requires a survival result object")


def _cox_anova_test(test: str | None) -> tuple[str, bool]:
    if test is None:
        return "none", False
    if not isinstance(test, str):
        raise TypeError("anova test must be a string or None")
    value = test.strip().lower().replace("_", "-")
    if not value:
        return "none", False
    aliases = {
        "chisquare": "chisq",
        "chi-square": "chisq",
        "chi-squared": "chisq",
        "likelihood": "lrt",
        "likelihood-ratio": "lrt",
        "likelihood-ratio-test": "lrt",
    }
    normalized = aliases.get(value) or _match_string_arg(
        value,
        "anova test",
        ("chisq", "lrt", "none"),
        "anova test must be 'Chisq', 'LRT', or 'none'",
    )
    if normalized == "none":
        return "none", False
    return ("Chisq" if normalized == "chisq" else "LRT"), True


def _anova_result(
    logliks: list[float],
    dfs: list[int],
    names: list[str],
    test_name: str,
    with_tests: bool,
) -> Any:
    if with_tests and len(logliks) >= 2:
        return _core.anova_coxph(logliks, dfs, names, test_name)

    rows = []
    for name, loglik, df in zip(names, logliks, dfs, strict=True):
        rows.append(_core.AnovaRow(name, loglik, df, None, None))
    return _core.AnovaCoxphResult(rows, test_name)


def _cox_design_groups(fit: Any, n_columns: int) -> list[tuple[str, int]]:
    design = _formula_design_for_fit(fit)
    if design is None:
        return [(f"x{idx + 1}", 1) for idx in range(n_columns)]

    groups = [
        (_design_term_name(term), len(_design_term_output_names(term)))
        for term in design.covariates
    ]
    if sum(width for _, width in groups) != n_columns:
        return [(f"x{idx + 1}", 1) for idx in range(n_columns)]
    return groups


def _cox_fit_offset(fit: Any, beta: list[float]) -> list[float] | None:
    rows = getattr(fit, "covariates", None)
    linear_predictors = getattr(fit, "linear_predictors", None)
    if rows is None or linear_predictors is None:
        return None

    offsets = []
    for row, linear_predictor in zip(rows, linear_predictors, strict=True):
        row_values = [float(value) for value in row]
        fitted = sum(
            value * coefficient for value, coefficient in zip(row_values, beta, strict=True)
        )
        offsets.append(float(linear_predictor) - fitted)
    if all(abs(value) <= 1e-12 for value in offsets):
        return None
    return offsets


def _cox_refit_loglik(fit: Any, width: int, offset: list[float] | None) -> float:
    rows = [[float(value) for value in row[:width]] for row in fit.covariates]
    nocenter = getattr(fit, "nocenter", None)
    refit = _core.coxph_fit(
        [float(value) for value in fit.event_times],
        [int(value) for value in fit.status],
        rows,
        strata=[int(value) for value in fit.strata] if hasattr(fit, "strata") else None,
        weights=[float(value) for value in fit.weights] if hasattr(fit, "weights") else None,
        offset=offset,
        initial_beta=None,
        max_iter=None,
        eps=None,
        toler=None,
        method=getattr(fit, "method", None),
        entry_times=(
            [float(value) for value in fit.entry_times]
            if getattr(fit, "entry_times", None) is not None
            else None
        ),
        nocenter=[float(value) for value in nocenter] if nocenter is not None else None,
    )
    return _cox_full_loglik(refit)


def _anova_single_coxph(fit: Any, test_name: str, with_tests: bool) -> Any:
    model = _require_coxph_fit(fit)
    beta = _cox_beta(model)
    n_columns = len(beta)
    names = ["NULL"]
    dfs = [0]
    logliks = [_cox_loglik_values(model)[0]]
    if n_columns == 0:
        return _anova_result(logliks, dfs, names, test_name, with_tests)

    groups = _cox_design_groups(fit, n_columns)
    offset = _cox_fit_offset(model, beta)
    width = 0
    for idx, (name, group_width) in enumerate(groups):
        width += group_width
        names.append(name)
        dfs.append(width)
        if idx == len(groups) - 1:
            logliks.append(_cox_full_loglik(model))
        else:
            logliks.append(_cox_refit_loglik(model, width, offset))
    return _anova_result(logliks, dfs, names, test_name, with_tests)


def _anova_multiple_coxph(fits: tuple[Any, ...], test_name: str, with_tests: bool) -> Any:
    models = [_require_coxph_fit(fit) for fit in fits]
    logliks = [_cox_full_loglik(model) for model in models]
    dfs = [_cox_degrees_of_freedom(model) for model in models]
    names = [f"Model {idx + 1}" for idx in range(len(models))]
    return _anova_result(logliks, dfs, names, test_name, with_tests)


def _cox_detail_method(fit: Any) -> str:
    method = str(getattr(fit, "method", "breslow")).lower().replace("_", "-")
    if method in {"breslow", "efron"}:
        return method
    raise ValueError(f"detailed output is not available for the {method} method")


def _cox_detail_rorder(rorder: Any) -> str:
    return _match_string_arg(
        rorder,
        "rorder",
        ("data", "time"),
        "rorder must be 'data' or 'time'",
    )


def _cox_detail_event_times(
    time: list[float],
    status: list[int],
    strata: list[int],
) -> list[tuple[int, float]]:
    groups: list[tuple[int, float]] = []
    for stratum in sorted(set(strata)):
        values = sorted(
            {time[idx] for idx, event in enumerate(status) if event == 1 and strata[idx] == stratum}
        )
        groups.extend((stratum, value) for value in values)
    return groups


def _cox_detail_at_risk(
    time: list[float],
    entry: list[float] | None,
    strata: list[int],
    stratum: int,
    event_time: float,
) -> list[int]:
    return [
        idx
        for idx, stop in enumerate(time)
        if strata[idx] == stratum
        and stop >= event_time
        and (entry is None or entry[idx] < event_time)
    ]


def _cox_detail_y(
    time: list[float],
    status: list[int],
    entry: list[float] | None,
) -> list[list[float]]:
    if entry is None:
        return [[stop, float(event)] for stop, event in zip(time, status, strict=True)]
    return [
        [start, stop, float(event)] for start, stop, event in zip(entry, time, status, strict=True)
    ]


def _cox_detail_row_order(
    time: list[float],
    status: list[int],
    strata: list[int],
    rorder: str,
) -> list[int]:
    if rorder == "data":
        return list(range(len(time)))
    if rorder == "time":
        return sorted(range(len(time)), key=lambda idx: (strata[idx], time[idx], -status[idx], idx))
    raise ValueError("rorder must be 'data' or 'time'")


def _cox_detail_strata_table(
    strata: list[int],
    event_groups: list[tuple[int, float]],
) -> dict[int, int] | None:
    if len(set(strata)) <= 1:
        return None
    table: dict[int, int] = {}
    for stratum, _event_time in event_groups:
        table[stratum] = table.get(stratum, 0) + 1
    return table


def _is_survreg_fit(fit: Any) -> bool:
    return hasattr(fit, "n_covariates") and hasattr(fit, "location_coefficients")


def _location_beta(fit: Any) -> list[float]:
    values = getattr(fit, "location_coefficients", None)
    if values is not None:
        return [float(value) for value in values]
    return _cox_beta(fit)


def _training_linear_predictor_center(fit: Any) -> float:
    values = getattr(fit, "linear_predictors", None)
    if values is None:
        return 0.0
    linear_predictors = [float(value) for value in values]
    if not linear_predictors:
        return 0.0
    return sum(linear_predictors) / len(linear_predictors)


def _normalize_predict_reference(
    reference: str | None,
    centered: bool | None,
    predict_type: str,
) -> str:
    if reference is None:
        if centered is False:
            return "zero"
        if centered is True or predict_type == "terms":
            return "sample"
        return "strata"
    return _match_string_arg(
        reference,
        "reference",
        ("sample", "zero", "strata"),
        "reference must be 'sample', 'zero', or 'strata'",
    )


def _cox_reference_means(fit: Any, reference: str) -> list[float]:
    beta = _cox_beta(fit)
    if reference == "zero" or not beta:
        return [0.0] * len(beta)

    means = getattr(fit, "means", None)
    if means is None:
        covariates = getattr(fit, "covariates", None)
        if covariates is None:
            return [0.0] * len(beta)
        rows = [[float(value) for value in row] for row in covariates]
        means = (
            [sum(row[col_idx] for row in rows) / len(rows) for col_idx in range(len(beta))]
            if rows
            else [0.0] * len(beta)
        )
    else:
        means = [float(value) for value in means]

    if len(means) != len(beta):
        return [0.0] * len(beta)

    nocenter = _cox_nocenter_columns(fit, len(beta))
    for col_idx in nocenter:
        means[col_idx] = 0.0
    return means


def _cox_nocenter_columns(fit: Any, nvar: int) -> set[int]:
    covariates = getattr(fit, "covariates", None)
    if covariates is None:
        return set()
    rows = [[float(value) for value in row] for row in covariates]
    if not rows or any(len(row) != nvar for row in rows):
        return set()
    values = getattr(fit, "nocenter", (-1.0, 0.0, 1.0))
    if values is None:
        return set()
    nocenter_values = [float(value) for value in values]
    if not nocenter_values:
        return set()
    nocenter: set[int] = set()
    for col_idx in range(nvar):
        if all(any(row[col_idx] == value for value in nocenter_values) for row in rows):
            nocenter.add(col_idx)
    return nocenter


def _cox_training_rows(fit: Any, nvar: int) -> list[list[float]]:
    covariates = getattr(fit, "covariates", None)
    if covariates is None:
        return []
    rows = [[float(value) for value in row] for row in covariates]
    if any(len(row) != nvar for row in rows):
        return []
    return rows


def _cox_training_strata(fit: Any, n: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata = [int(value) for value in values]
    if len(strata) != n:
        raise ValueError("fitted Cox model strata do not match training rows")
    return strata


def _cox_strata_reference_means(fit: Any, nvar: int) -> dict[int, list[float]]:
    rows = _cox_training_rows(fit, nvar)
    if not rows:
        return {0: [0.0] * nvar}
    strata = _cox_training_strata(fit, len(rows))
    weights = _model_residual_weights(fit, len(rows))
    totals: dict[int, list[float]] = {}
    weight_totals: dict[int, float] = {}
    for row, stratum, weight in zip(rows, strata, weights, strict=True):
        totals.setdefault(stratum, [0.0] * nvar)
        weight_totals[stratum] = weight_totals.get(stratum, 0.0) + weight
        for col_idx, value in enumerate(row):
            totals[stratum][col_idx] += value * weight

    nocenter = _cox_nocenter_columns(fit, nvar)
    means: dict[int, list[float]] = {}
    for stratum, values in totals.items():
        denom = weight_totals[stratum]
        row_means = [value / denom if denom > 0.0 else 0.0 for value in values]
        for col_idx in nocenter:
            row_means[col_idx] = 0.0
        means[stratum] = row_means
    return means


def _cox_prediction_strata(fit: Any, newdata: Any | None, n: int) -> list[int]:
    training_rows = _cox_training_rows(fit, len(_cox_beta(fit)))
    training_strata = _cox_training_strata(fit, len(training_rows)) if training_rows else [0]
    if newdata is None:
        if n == len(training_strata):
            return training_strata
        if len(set(training_strata)) == 1:
            return [training_strata[0]] * n
        raise ValueError("newdata strata are required for reference='strata'")

    if len(set(training_strata)) <= 1:
        return [training_strata[0]] * n

    design = _formula_design_for_fit(fit)
    if (
        design is not None
        and design.strata
        and (isinstance(newdata, Mapping) or hasattr(newdata, "columns"))
    ):
        labels = _combined_columns(newdata, list(design.strata), n)
        level_map = {value: idx for idx, value in enumerate(design.strata_levels)}
        strata: list[int] = []
        for value in labels:
            try:
                strata.append(level_map[value])
            except KeyError as exc:
                raise ValueError(f"newdata contains unknown strata level {value!r}") from exc
        return strata

    raise ValueError("newdata strata are required for reference='strata'")


def _cox_reference_means_for_rows(
    fit: Any,
    reference: str,
    rows: list[list[float]],
    newdata: Any | None,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    if reference != "strata":
        means = _cox_reference_means(fit, reference)
        return [means for _row in rows]

    strata = _cox_prediction_strata(fit, newdata, len(rows))
    means_by_stratum = _cox_strata_reference_means(fit, len(beta))
    sample_means = _cox_reference_means(fit, "sample")
    return [means_by_stratum.get(stratum, sample_means) for stratum in strata]


def _cox_prediction_design_rows(
    fit: Any,
    rows: list[list[float]] | None,
    reference: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    if rows is None:
        rows = _cox_training_rows(fit, nvar)
        if not rows and nvar:
            raise ValueError("stored training covariates are required for prediction SEs")
    if any(len(row) != nvar for row in rows):
        raise ValueError(f"newdata must have {nvar} columns")

    means_by_row = _cox_reference_means_for_rows(fit, reference, rows, newdata)
    return [
        [float(value) - float(means_by_row[row_idx][col_idx]) for col_idx, value in enumerate(row)]
        for row_idx, row in enumerate(rows)
    ]


def _cox_variance_matrix(fit: Any, nvar: int) -> list[list[float]]:
    raw_variance = getattr(fit, "information_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row] for row in raw_variance]
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        raise ValueError("fitted Cox model information matrix does not match coefficient width")
    return variance


def _location_variance_matrix(fit: Any, nvar: int) -> list[list[float]]:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row[:nvar]] for row in list(raw_variance)[:nvar]]
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        raise ValueError("fitted survreg variance matrix does not match coefficient width")
    return variance


def _survreg_has_variance_width(fit: Any, width: int) -> bool:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        return False
    matrix = list(raw_variance)
    return len(matrix) >= width and all(len(row) >= width for row in matrix[:width])


def _survreg_variance_matrix(fit: Any, width: int) -> list[list[float]]:
    if width == 0:
        return []
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    variance = [[float(value) for value in row[:width]] for row in list(raw_variance)[:width]]
    if len(variance) != width or any(len(row) != width for row in variance):
        raise ValueError("fitted survreg variance matrix does not match residual width")
    return variance


def _survreg_scales(fit: Any) -> list[float]:
    values = getattr(fit, "scales", None)
    if values is None:
        values = [getattr(fit, "scale", 1.0)]
    scales = [float(value) for value in values]
    if not scales:
        raise ValueError("fitted survreg model does not expose scale values")
    return scales


def _survreg_strata(fit: Any, n: int, nstrata: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata = [int(value) for value in values]
    if len(strata) != n:
        raise ValueError("fitted survreg strata do not match training rows")
    if any(value < 0 or value >= nstrata for value in strata):
        raise ValueError("fitted survreg strata reference missing scale values")
    return strata


def _survreg_derivative_context(
    fit: Any,
    *,
    rsigma: bool | None,
) -> tuple[list[list[float]], list[list[float]], list[float], list[int], list[list[float]], bool]:
    nvar = len(_location_beta(fit))
    rows = _cox_training_rows(fit, nvar)
    if not rows and nvar:
        raise ValueError("stored training covariates are required for survreg residuals")
    matrix = _core.survreg_residual_matrix(
        fit.time,
        fit.status,
        fit.linear_predictors,
        fit.scale,
        fit.distribution,
        time2=getattr(fit, "time2", None),
    )
    scales = _survreg_scales(fit)
    strata = _survreg_strata(fit, len(matrix), len(scales))
    rsigma_requested = True if rsigma is None else rsigma
    include_scale = rsigma_requested and _survreg_has_variance_width(
        fit,
        nvar + len(scales),
    )
    width = nvar + (len(scales) if include_scale else 0)
    variance = _survreg_variance_matrix(fit, width)
    return matrix, rows, scales, strata, variance, include_scale


def _survreg_dfbeta_residuals(
    fit: Any,
    residual_type: str,
    *,
    rsigma: bool | None,
) -> list[list[float]]:
    matrix, rows, scales, strata, variance, include_scale = _survreg_derivative_context(
        fit,
        rsigma=rsigma,
    )
    return _core.survreg_dfbeta_residuals(
        matrix,
        rows,
        scales,
        strata,
        variance,
        include_scale,
        residual_type == "dfbetas",
    )


def _survreg_influence_residuals(
    fit: Any,
    residual_type: str,
    *,
    rsigma: bool | None,
) -> list[float]:
    matrix, rows, scales, strata, variance, include_scale = _survreg_derivative_context(
        fit,
        rsigma=rsigma,
    )
    return _core.survreg_influence_residuals(
        matrix,
        rows,
        scales,
        strata,
        variance,
        residual_type,
        include_scale,
    )


def _survreg_prediction_rows(
    fit: Any,
    rows: list[list[float]] | None,
    purpose: str,
) -> list[list[float]]:
    nvar = len(_location_beta(fit))
    if rows is None:
        training_rows = _cox_training_rows(fit, nvar)
        if not training_rows and nvar:
            raise ValueError(f"stored training covariates are required for {purpose}")
        return training_rows

    prediction_rows = [[float(value) for value in row] for row in rows]
    if any(len(row) != nvar for row in prediction_rows):
        raise ValueError(f"newdata must have {nvar} columns")
    return prediction_rows


def _survreg_training_means(fit: Any, nvar: int) -> list[float]:
    rows = _cox_training_rows(fit, nvar)
    if not rows:
        return [0.0] * nvar
    return [sum(row[col_idx] for row in rows) / len(rows) for col_idx in range(nvar)]


def _survreg_term_design_rows(
    fit: Any,
    rows: list[list[float]] | None,
) -> list[list[float]]:
    prediction_rows = _survreg_prediction_rows(fit, rows, "predict type='terms'")
    design = _formula_design_for_fit(fit)
    if design is None or not design.intercept:
        return prediction_rows
    means = _survreg_training_means(fit, len(_location_beta(fit)))
    return [
        [float(value) - means[col_idx] for col_idx, value in enumerate(row)]
        for row in prediction_rows
    ]


def _quadratic_form(values: list[float], variance: list[list[float]]) -> float:
    return sum(
        values[row_idx] * variance[row_idx][col_idx] * values[col_idx]
        for row_idx in range(len(values))
        for col_idx in range(len(values))
    )


def _matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    if not left or not right:
        return []
    width = len(right[0])
    inner = len(right)
    return [
        [
            sum(left[row_idx][inner_idx] * right[inner_idx][col_idx] for inner_idx in range(inner))
            for col_idx in range(width)
        ]
        for row_idx in range(len(left))
    ]


def _cox_robust_variance_matrix(
    fit: Any,
    cluster: Any,
) -> tuple[list[list[float]], list[list[float]], list[Any]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    naive = _cox_variance_matrix(fit, nvar)
    cluster_values = _materialize_labels(cluster, "cluster")
    score = fit.score_residuals()
    n = len(score)
    if len(cluster_values) != n:
        raise ValueError("cluster must have the same length as the Surv response")
    if any(len(row) != nvar for row in score):
        raise ValueError("fitted Cox model score residuals do not match coefficient width")

    weights = [float(value) for value in getattr(fit, "weights", [1.0] * n)]
    if len(weights) != n:
        raise ValueError("fitted Cox model weights do not match residual length")

    _label_levels(cluster_values, "cluster")
    cluster_scores: dict[Any, list[float]] = {}
    for row_idx, label in enumerate(cluster_values):
        score_row = cluster_scores.setdefault(label, [0.0] * nvar)
        for col_idx in range(nvar):
            score_row[col_idx] += weights[row_idx] * float(score[row_idx][col_idx])

    meat = [[0.0 for _ in range(nvar)] for _ in range(nvar)]
    for score_row in cluster_scores.values():
        for row_idx in range(nvar):
            for col_idx in range(nvar):
                meat[row_idx][col_idx] += score_row[row_idx] * score_row[col_idx]

    robust = _matrix_multiply(_matrix_multiply(naive, meat), naive)
    return robust, naive, cluster_values


def _crossprod_rows(rows: list[list[float]], width: int, name: str) -> list[list[float]]:
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} rows must be rectangular")
    result = [[0.0 for _ in range(width)] for _ in range(width)]
    for row in rows:
        for row_idx in range(width):
            for col_idx in range(width):
                result[row_idx][col_idx] += row[row_idx] * row[col_idx]
    return result


def _survreg_robust_variance_matrix(
    fit: Any,
    cluster: Any,
) -> tuple[list[list[float]], list[list[float]], list[Any]]:
    cluster_values = _materialize_labels(cluster, "cluster")
    _label_levels(cluster_values, "cluster")
    dfbeta_rows = _survreg_dfbeta_residuals(fit, "dfbeta", rsigma=True)
    n = len(dfbeta_rows)
    if len(cluster_values) != n:
        raise ValueError("cluster must have the same length as the Surv response")

    width = len(dfbeta_rows[0]) if dfbeta_rows else len(list(getattr(fit, "variance_matrix", [])))
    naive = _survreg_variance_matrix(fit, width)
    weights = _model_residual_weights(fit, n)
    weighted_rows = _weight_residual_result(dfbeta_rows, weights)
    collapsed_rows = _collapse_residual_result(weighted_rows, cluster_values, n)
    robust = _crossprod_rows(collapsed_rows, width, "survreg dfbeta")
    return robust, naive, cluster_values


def _cox_linear_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    reference: str,
    newdata: Any | None,
) -> list[float]:
    design_rows = _cox_prediction_design_rows(fit, rows, reference, newdata)
    variance = _cox_variance_matrix(fit, len(_cox_beta(fit)))
    return [math.sqrt(max(_quadratic_form(row, variance), 0.0)) for row in design_rows]


def _submatrix(matrix: list[list[float]], columns: list[int]) -> list[list[float]]:
    return [[matrix[row_idx][col_idx] for col_idx in columns] for row_idx in columns]


def _cox_term_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
    reference: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    design_rows = _cox_prediction_design_rows(fit, rows, reference, newdata)
    variance = _cox_variance_matrix(fit, len(beta))
    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    result: list[list[float]] = []
    for row in design_rows:
        result_row: list[float] = []
        for group_idx in selected:
            columns = groups[group_idx][1]
            values = [row[col_idx] for col_idx in columns]
            result_row.append(
                math.sqrt(max(_quadratic_form(values, _submatrix(variance, columns)), 0.0))
            )
        result.append(result_row)
    return result


def _survreg_predict_terms(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_term_design_rows(fit, rows)
    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    return [
        [
            sum(float(row[col_idx]) * beta[col_idx] for col_idx in groups[group_idx][1])
            for group_idx in selected
        ]
        for row in prediction_rows
    ]


def _survreg_term_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    terms: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_term_design_rows(fit, rows)
    variance = _location_variance_matrix(fit, len(beta))
    groups = _cox_predict_term_groups(fit, len(beta))
    selected = _predict_terms_selection(terms, [name for name, _columns in groups])
    result: list[list[float]] = []
    for row in prediction_rows:
        result_row: list[float] = []
        for group_idx in selected:
            columns = groups[group_idx][1]
            values = [float(row[col_idx]) for col_idx in columns]
            result_row.append(
                math.sqrt(max(_quadratic_form(values, _submatrix(variance, columns)), 0.0))
            )
        result.append(result_row)
    return result


def _survreg_linear_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
) -> list[float]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_prediction_rows(fit, rows, "prediction SEs")
    variance = _location_variance_matrix(fit, len(beta))
    return [math.sqrt(max(_quadratic_form(row, variance), 0.0)) for row in prediction_rows]


def _survreg_response_uses_log_transform(fit: Any) -> bool:
    distribution = str(getattr(fit, "distribution", "")).lower().replace("-", "_")
    return distribution in {
        "weibull",
        "exponential",
        "lognormal",
        "log_normal",
        "loggaussian",
        "loglogistic",
        "log_logistic",
    }


def _survreg_original_scale_loglik(fit: Any) -> float:
    model = _unwrap_formula_fit(fit)
    log_likelihood = float(model.log_likelihood)
    if not _survreg_response_uses_log_transform(model):
        return log_likelihood

    times = [float(value) for value in model.time]
    status = [int(value) for value in model.status]
    weights = [float(value) for value in getattr(model, "weights", [1.0] * len(times))]
    if len(status) != len(times) or len(weights) != len(times):
        raise ValueError("fitted survreg model has inconsistent likelihood arrays")
    jacobian = math.fsum(
        weight * math.log(time)
        for time, event, weight in zip(times, status, weights, strict=True)
        if event == 1
    )
    return log_likelihood - jacobian


def _survreg_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    predict_type: str,
    predictions: list[float],
) -> list[float]:
    se = _survreg_linear_prediction_se(fit, rows)
    if predict_type != "response" or not _survreg_response_uses_log_transform(fit):
        return se
    return [
        value * abs(float(prediction)) for value, prediction in zip(se, predictions, strict=True)
    ]


def _survreg_distribution_family(fit: Any) -> str:
    distribution = str(getattr(fit, "distribution", "")).lower().replace("-", "_")
    if distribution in {"logistic", "loglogistic", "log_logistic"}:
        return "logistic"
    if distribution in {"gaussian", "normal", "lognormal", "log_normal", "loggaussian"}:
        return "gaussian"
    return "extreme"


def _survreg_quantile_probabilities(values: Any | None) -> list[float]:
    probabilities = _quantile_vector(values, "p") if values is not None else [0.1, 0.9]
    if any(value <= 0.0 or value >= 1.0 for value in probabilities):
        raise ValueError("p must be between 0 and 1")
    return probabilities


def _survreg_quantile_scores(fit: Any, probabilities: list[float]) -> list[float]:
    family = _survreg_distribution_family(fit)
    if family == "logistic":
        return [math.log(value / (1.0 - value)) for value in probabilities]
    if family == "gaussian":
        normal = NormalDist()
        return [normal.inv_cdf(value) for value in probabilities]
    return [math.log(-math.log1p(-value)) for value in probabilities]


def _survreg_training_strata(fit: Any, n: int) -> list[int]:
    values = getattr(fit, "strata", None)
    if values is None:
        return [0] * n
    strata = [int(value) for value in values]
    if len(strata) != n:
        raise ValueError("fitted survreg strata do not match prediction rows")
    scales = _survreg_scales(fit)
    if any(value < 0 or value >= len(scales) for value in strata):
        raise ValueError("fitted survreg strata do not match scale estimates")
    return strata


def _survreg_prediction_strata(
    fit: Any,
    newdata: Any | None,
    n: int,
) -> list[int]:
    scales = _survreg_scales(fit)
    if len(scales) <= 1:
        return [0] * n
    if newdata is None:
        return _survreg_training_strata(fit, n)

    design = _formula_design_for_fit(fit)
    if (
        design is not None
        and design.strata
        and (isinstance(newdata, Mapping) or hasattr(newdata, "columns"))
    ):
        labels = _combined_columns(newdata, list(design.strata), n)
        level_map = {value: idx for idx, value in enumerate(design.strata_levels)}
        strata: list[int] = []
        for value in labels:
            try:
                strata.append(level_map[value])
            except KeyError as exc:
                raise ValueError(f"newdata contains unknown strata level {value!r}") from exc
        if any(value >= len(scales) for value in strata):
            raise ValueError("newdata strata do not match scale estimates")
        return strata

    raise ValueError("newdata strata are required for survreg quantile predictions")


def _survreg_quantile_variance_matrix(
    fit: Any,
    nvar: int,
    nscale: int,
) -> list[list[float]]:
    raw_variance = getattr(fit, "variance_matrix", None)
    if raw_variance is None:
        raise TypeError("model does not expose coefficient variance")
    matrix = [[float(value) for value in row] for row in raw_variance]
    full_width = nvar + nscale
    if len(matrix) >= full_width and all(len(row) >= full_width for row in matrix[:full_width]):
        return [row[:full_width] for row in matrix[:full_width]]
    if len(matrix) >= nvar and all(len(row) >= nvar for row in matrix[:nvar]):
        return [row[:nvar] for row in matrix[:nvar]]
    raise ValueError("fitted survreg variance matrix does not match quantile width")


def _survreg_quantile_linear_values(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    quantile_scores: list[float],
    newdata: Any | None,
) -> list[list[float]]:
    result = fit.predict(rows, "lp", offsets, False)
    linear_predictors = [float(value) for value in result.predictions]
    strata = _survreg_prediction_strata(fit, newdata, len(linear_predictors))
    scales = _survreg_scales(fit)
    return [
        [linear_predictor + score * scales[strata[row_idx]] for score in quantile_scores]
        for row_idx, linear_predictor in enumerate(linear_predictors)
    ]


def _survreg_quantile_prediction_matrix(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    quantile_scores: list[float],
    predict_type: str,
    newdata: Any | None,
) -> list[list[float]]:
    linear_values = _survreg_quantile_linear_values(
        fit,
        rows,
        offsets,
        quantile_scores,
        newdata,
    )
    if predict_type != "quantile" or not _survreg_response_uses_log_transform(fit):
        return linear_values
    return [[_safe_exp(value) for value in row] for row in linear_values]


def _survreg_quantile_prediction_se_matrix(
    fit: Any,
    rows: list[list[float]] | None,
    quantile_scores: list[float],
    predictions: list[list[float]],
    predict_type: str,
    newdata: Any | None,
) -> list[list[float]]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_prediction_rows(fit, rows, "survreg quantile prediction SEs")
    scales = _survreg_scales(fit)
    strata = _survreg_prediction_strata(fit, newdata, len(prediction_rows))
    variance = _survreg_quantile_variance_matrix(fit, len(beta), len(scales))
    has_scale_variance = len(variance) >= len(beta) + len(scales)
    transform_se = predict_type == "quantile" and _survreg_response_uses_log_transform(fit)

    se_matrix: list[list[float]] = []
    for row_idx, row in enumerate(prediction_rows):
        se_row: list[float] = []
        for score_idx, score in enumerate(quantile_scores):
            values = list(row)
            if has_scale_variance:
                scale_values = [0.0] * len(scales)
                stratum = strata[row_idx]
                scale_values[stratum] = score * scales[stratum]
                values.extend(scale_values)
            linear_se = math.sqrt(max(_quadratic_form(values, variance), 0.0))
            if transform_se:
                linear_se *= abs(float(predictions[row_idx][score_idx]))
            se_row.append(linear_se)
        se_matrix.append(se_row)
    return se_matrix


def _drop_single_quantile(values: list[list[float]], probabilities: list[float]) -> Any:
    if len(probabilities) == 1:
        return [row[0] for row in values]
    return values


def _cox_reference_centers(
    fit: Any,
    reference: str,
    n: int,
    newdata: Any | None,
) -> list[float]:
    if reference == "zero":
        return [0.0] * n
    beta = _cox_beta(fit)
    offset_center = _cox_training_offset_center(fit, beta)
    if reference != "strata":
        return [_cox_reference_center(fit, reference)] * n

    strata = _cox_prediction_strata(fit, newdata, n)
    means_by_stratum = _cox_strata_reference_means(fit, len(beta))
    sample_means = _cox_reference_means(fit, "sample")
    centers: list[float] = []
    for stratum in strata:
        means = means_by_stratum.get(stratum, sample_means)
        centers.append(
            sum(value * coefficient for value, coefficient in zip(means, beta, strict=True))
            + offset_center
        )
    return centers


def _cox_training_offset_center(fit: Any, beta: list[float]) -> float:
    offsets = _cox_fit_offset(fit, beta)
    if offsets is None:
        return 0.0
    return sum(offsets) / len(offsets) if offsets else 0.0


def _cox_reference_center(fit: Any, reference: str) -> float:
    if reference == "zero":
        return 0.0
    beta = _cox_beta(fit)
    means = _cox_reference_means(fit, reference)
    return sum(
        value * coefficient for value, coefficient in zip(means, beta, strict=True)
    ) + _cox_training_offset_center(fit, beta)


def _formula_design_for_fit(fit: Any) -> _FormulaDesign | None:
    return fit.design if isinstance(fit, _FormulaFit) else None


def _prediction_inputs(
    fit: Any,
    newdata: Any | None,
) -> tuple[list[list[float]] | None, list[float] | None]:
    if newdata is None:
        return None, None
    design = _formula_design_for_fit(fit)
    if design is not None and (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
        n = _formula_design_row_count(newdata, design)
        offsets = _offset_vector(newdata, list(design.offsets), n) if design.offsets else None
        return _design_rows_from_spec(newdata, design, n), offsets
    if isinstance(newdata, Mapping):
        raise TypeError("newdata must be a design matrix unless the fit was created from a formula")
    rows = _as_rows(newdata, "newdata")
    if (
        design is not None
        and design.intercept
        and _is_survreg_fit(fit)
        and rows
        and len(rows[0]) + 1 == len(_location_beta(fit))
    ):
        return [[1.0, *row] for row in rows], None
    return rows, None


def _linear_predictors_for_fit(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None = None,
) -> list[float]:
    if rows is None:
        values = getattr(fit, "linear_predictors", None)
        if values is not None:
            return [float(value) for value in values]
        raise ValueError(
            "newdata is required when the fitted model does not store training predictors"
        )
    if not hasattr(fit, "predict"):
        raise TypeError("model does not support prediction")
    linear_predictors = [float(value) for value in fit.predict(rows)]
    if offsets is not None:
        if len(offsets) != len(linear_predictors):
            raise ValueError("newdata offset columns must match prediction rows")
        linear_predictors = [
            value + offset for value, offset in zip(linear_predictors, offsets, strict=True)
        ]
    return linear_predictors


def _cox_prediction_offset_vector(fit: Any, n: int) -> list[float]:
    beta = _cox_beta(fit)
    offsets = _cox_fit_offset(_unwrap_formula_fit(fit), beta)
    if offsets is None:
        return [0.0] * n
    if len(offsets) != n:
        raise ValueError("fitted Cox model offsets do not match training rows")
    return [float(value) for value in offsets]


def _surv_from_formula_design(data: Any, design: _FormulaDesign) -> Surv:
    try:
        args = _formula_response_values(data, design.response)
        if len(args) == 1:
            return Surv(
                args[0],
                type=design.response.type,
                origin=design.response.origin,
            )
        if len(args) == 2:
            return Surv(
                args[0],
                args[1],
                type=design.response.type,
                origin=design.response.origin,
            )
        if len(args) == 3:
            return Surv(
                args[0],
                args[1],
                args[2],
                type=design.response.type,
                origin=design.response.origin,
            )
    except KeyError as exc:
        raise ValueError(
            "predict type='expected' with newdata requires formula response columns"
        ) from exc
    raise ValueError("formula response must have 1, 2, or 3 columns")


def _newdata_has_formula_response(fit: Any, newdata: Any | None) -> bool:
    design = _formula_design_for_fit(fit)
    if design is None or newdata is None:
        return False
    names = _data_column_names(newdata)
    if names is None:
        return False
    available = set(names)
    response_columns = set(design.response.columns)
    if response_columns <= available:
        return True
    if response_columns & available:
        raise ValueError(
            "predict type='survival' with partial formula response columns is ambiguous"
        )
    return False


@dataclass(frozen=True)
class _CoxExpectedBaseline:
    times: list[float]
    cumhaz: list[float]
    varhaz: list[float]
    xbar: list[list[float]]


def _cox_expected_baseline_by_stratum(fit: Any) -> dict[int, _CoxExpectedBaseline]:
    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    nvar = len(beta)
    rows = _cox_training_rows(model, nvar)
    times = [float(value) for value in model.event_times]
    status = [int(value) for value in model.status]
    n = len(times)
    if len(rows) != n or len(status) != n:
        raise ValueError("fitted Cox model event arrays have inconsistent lengths")

    entry_values = getattr(model, "entry_times", None)
    entry = [float(value) for value in entry_values] if entry_values is not None else None
    if entry is not None and len(entry) != n:
        raise ValueError("fitted Cox model entry times do not match event rows")
    weights = _model_residual_weights(model, n)
    strata = _cox_training_strata(model, n)
    offsets = _cox_prediction_offset_vector(model, n)
    means = _cox_reference_means(model, "sample")
    centered_rows = [
        [float(value) - means[col_idx] for col_idx, value in enumerate(row)] for row in rows
    ]
    risk_weights = [
        weights[idx]
        * _safe_exp(
            offsets[idx]
            + sum(value * coefficient for value, coefficient in zip(row, beta, strict=True))
        )
        for idx, row in enumerate(rows)
    ]
    method = _cox_detail_method(model)
    baselines: dict[int, _CoxExpectedBaseline] = {}
    strata_values = sorted(set(strata))

    for stratum in strata_values:
        event_times = sorted(
            {
                times[idx]
                for idx, event in enumerate(status)
                if event == 1 and strata[idx] == stratum
            }
        )
        out_times: list[float] = []
        out_hazard: list[float] = []
        out_varhaz: list[float] = []
        out_xbar: list[list[float]] = []
        cumulative_hazard = 0.0
        cumulative_varhaz = 0.0
        cumulative_xbar = [0.0] * nvar

        for event_time in event_times:
            at_risk = [
                idx
                for idx in range(n)
                if strata[idx] == stratum
                and times[idx] >= event_time
                and (entry is None or entry[idx] < event_time)
            ]
            deaths = [
                idx
                for idx in at_risk
                if status[idx] == 1 and abs(times[idx] - event_time) < _SURVFIT_TIME_EPSILON
            ]
            if not deaths:
                continue
            event_weight = sum(weights[idx] for idx in deaths)
            denom = sum(risk_weights[idx] for idx in at_risk)
            if denom <= 0.0:
                hazard = 0.0
                varhaz = 0.0
                xbar_increment = [0.0] * nvar
            else:
                risk_xsum = [
                    sum(risk_weights[idx] * centered_rows[idx][col_idx] for idx in at_risk)
                    for col_idx in range(nvar)
                ]
                if method == "efron" and len(deaths) > 1:
                    death_risk = sum(risk_weights[idx] for idx in deaths)
                    death_xsum = [
                        sum(risk_weights[idx] * centered_rows[idx][col_idx] for idx in deaths)
                        for col_idx in range(nvar)
                    ]
                    step_weight = event_weight / len(deaths)
                    hazard = 0.0
                    varhaz = 0.0
                    xbar_increment = [0.0] * nvar
                    for step in range(len(deaths)):
                        fraction = step / len(deaths)
                        step_denom = denom - fraction * death_risk
                        if step_denom <= 0.0:
                            continue
                        hazard += step_weight / step_denom
                        varhaz += step_weight / (step_denom * step_denom)
                        for col_idx in range(nvar):
                            step_xsum = risk_xsum[col_idx] - fraction * death_xsum[col_idx]
                            xbar_increment[col_idx] += (
                                step_weight * step_xsum / (step_denom * step_denom)
                            )
                else:
                    hazard = event_weight / denom
                    varhaz = event_weight / (denom * denom)
                    xbar_increment = [
                        hazard * risk_xsum[col_idx] / denom for col_idx in range(nvar)
                    ]

            cumulative_hazard += hazard
            cumulative_varhaz += varhaz
            cumulative_xbar = [
                cumulative_xbar[col_idx] + xbar_increment[col_idx] for col_idx in range(nvar)
            ]
            out_times.append(event_time)
            out_hazard.append(cumulative_hazard)
            out_varhaz.append(cumulative_varhaz)
            out_xbar.append(list(cumulative_xbar))

        baselines[stratum] = _CoxExpectedBaseline(
            times=out_times,
            cumhaz=out_hazard,
            varhaz=out_varhaz,
            xbar=out_xbar,
        )

    return baselines


def _cox_expected_baseline_at(
    baseline: _CoxExpectedBaseline,
    time: float,
    nvar: int,
) -> tuple[float, float, list[float]]:
    pos = bisect_right(baseline.times, time)
    if pos == 0:
        return 0.0, 0.0, [0.0] * nvar
    idx = pos - 1
    return baseline.cumhaz[idx], baseline.varhaz[idx], list(baseline.xbar[idx])


def _cox_training_response(fit: Any) -> Surv:
    model = _unwrap_formula_fit(fit)
    entry_values = getattr(model, "entry_times", None)
    if entry_values is None:
        return Surv(model.event_times, model.status)
    return Surv(entry_values, model.event_times, model.status)


def _cox_expected_events_with_se(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    newdata: Any | None,
) -> PredictResult:
    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    nvar = len(beta)
    if rows is None:
        rows = _cox_training_rows(model, nvar)
        if len(rows) != len(model.event_times):
            raise ValueError("stored training covariates are required for expected prediction SEs")
        response = _cox_training_response(model)
        prediction_strata = _cox_training_strata(model, len(rows))
        linear_predictors = _linear_predictors_for_fit(model, None)
    else:
        design = _formula_design_for_fit(fit)
        if design is None or not (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
            raise ValueError(
                "predict type='expected' with newdata requires formula response columns"
            )
        response = _surv_from_formula_design(newdata, design)
        if len(response) != len(rows):
            raise ValueError("newdata response and covariates must have the same row count")
        model_is_counting = getattr(model, "entry_times", None) is not None
        if model_is_counting != (response.start is not None):
            raise ValueError("newdata survival type differs from the fitted Cox model")
        prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)

    if any(len(row) != nvar for row in rows):
        raise ValueError(f"newdata must have {nvar} columns")
    means = _cox_reference_means(model, "sample")
    variance = _cox_variance_matrix(model, nvar)
    baselines = _cox_expected_baseline_by_stratum(model)

    predictions: list[float] = []
    se: list[float] = []
    for row_idx, (row, stop, stratum, linear_predictor) in enumerate(
        zip(rows, response.time, prediction_strata, linear_predictors, strict=True)
    ):
        baseline = baselines.get(stratum)
        if baseline is None:
            raise ValueError(f"newdata contains unknown strata level {stratum!r}")
        start = response.start[row_idx] if response.start is not None else None
        start_hazard, start_varhaz, start_xbar = (
            _cox_expected_baseline_at(baseline, float(start), nvar)
            if start is not None
            else (0.0, 0.0, [0.0] * nvar)
        )
        stop_hazard, stop_varhaz, stop_xbar = _cox_expected_baseline_at(
            baseline,
            float(stop),
            nvar,
        )
        centered_row = [float(value) - means[col_idx] for col_idx, value in enumerate(row)]
        start_delta = [
            start_hazard * centered_row[col_idx] - start_xbar[col_idx] for col_idx in range(nvar)
        ]
        stop_delta = [
            stop_hazard * centered_row[col_idx] - stop_xbar[col_idx] for col_idx in range(nvar)
        ]
        interval_delta = [stop_delta[col_idx] - start_delta[col_idx] for col_idx in range(nvar)]
        variance_value = stop_varhaz - start_varhaz + _quadratic_form(interval_delta, variance)
        risk = _safe_exp(float(linear_predictor))
        predictions.append(max(stop_hazard - start_hazard, 0.0) * risk)
        se.append(math.sqrt(max(variance_value, 0.0)) * risk)
    return PredictResult(predictions, se)


def _cox_expected_events_for_newdata(
    fit: Any,
    rows: list[list[float]],
    offsets: list[float] | None,
    newdata: Any,
) -> list[float]:
    design = _formula_design_for_fit(fit)
    if design is None or not (isinstance(newdata, Mapping) or hasattr(newdata, "columns")):
        raise ValueError("predict type='expected' with newdata requires formula response columns")

    response = _surv_from_formula_design(newdata, design)
    if len(response) != len(rows):
        raise ValueError("newdata response and covariates must have the same row count")

    model = _unwrap_formula_fit(fit)
    model_is_counting = getattr(model, "entry_times", None) is not None
    if model_is_counting != (response.start is not None):
        raise ValueError("newdata survival type differs from the fitted Cox model")

    basehaz_with_strata = getattr(model, "basehaz_with_strata", None)
    if basehaz_with_strata is None:
        base_times, base_hazards = model.basehaz(False)
        base_strata = [0] * len(base_times)
    else:
        base_times, base_hazards, base_strata = basehaz_with_strata(False)

    prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
    linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    baselines = _cox_baselines_by_stratum(base_times, base_hazards, base_strata)
    expected: list[float] = []
    for idx, (stop, stratum, linear_predictor) in enumerate(
        zip(response.time, prediction_strata, linear_predictors, strict=True)
    ):
        stratum_times, stratum_hazards = baselines.get(stratum, ([], []))
        start_hazard = (
            _step_hazard_at(stratum_times, stratum_hazards, float(response.start[idx]))
            if response.start is not None
            else 0.0
        )
        stop_hazard = _step_hazard_at(stratum_times, stratum_hazards, float(stop))
        expected.append(max(stop_hazard - start_hazard, 0.0) * _safe_exp(linear_predictor))
    return expected


def _step_curve_at(
    times: list[float],
    curve: list[float],
    requested_times: list[float],
) -> list[float]:
    values: list[float] = []
    for time in requested_times:
        pos = bisect_right(times, time)
        values.append(1.0 if pos == 0 else curve[pos - 1])
    return values


def _step_std_err_at(
    times: list[float],
    curve: list[float],
    requested_times: list[float],
) -> list[float]:
    values: list[float] = []
    for time in requested_times:
        pos = bisect_right(times, time)
        values.append(0.0 if pos == 0 else curve[pos - 1])
    return values


def _step_hazard_at(times: list[float], hazards: list[float], time: float) -> float:
    pos = bisect_right(times, time)
    return 0.0 if pos == 0 else hazards[pos - 1]


def _cox_baselines_by_stratum(
    base_times: list[float],
    base_hazards: list[float],
    base_strata: list[int],
) -> dict[int, tuple[list[float], list[float]]]:
    baselines: dict[int, tuple[list[float], list[float]]] = {}
    for time, hazard, stratum_value in zip(base_times, base_hazards, base_strata, strict=True):
        times, hazards = baselines.setdefault(int(stratum_value), ([], []))
        times.append(float(time))
        hazards.append(float(hazard))
    return baselines


def _cox_survival_curve(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    centered: bool,
    newdata: Any | None,
) -> tuple[list[float], list[list[float]]]:
    with_strata = getattr(fit, "survival_curve_with_strata", None)
    if rows is not None and with_strata is not None:
        prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
        if offsets is None:
            times, curves = with_strata(rows, prediction_strata, centered)
            return [float(value) for value in times], curves

        basehaz_with_strata = getattr(fit, "basehaz_with_strata", None)
        if basehaz_with_strata is None:
            raise TypeError("model does not support stratified baseline hazard prediction")
        base_times, base_hazards, base_strata = basehaz_with_strata(centered)
        requested_strata = set(prediction_strata)
        baselines = _cox_baselines_by_stratum(base_times, base_hazards, base_strata)
        curve_times = sorted(
            {time for stratum in requested_strata for time in baselines.get(stratum, ([], []))[0]}
        )
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
        center = _training_linear_predictor_center(fit) if centered else 0.0
        curves = []
        for linear_predictor, stratum in zip(linear_predictors, prediction_strata, strict=True):
            risk_multiplier = _safe_exp(linear_predictor - center)
            stratum_times, stratum_hazards = baselines.get(stratum, ([], []))
            curves.append(
                [
                    min(
                        max(
                            _safe_exp(
                                -_step_hazard_at(stratum_times, stratum_hazards, time)
                                * risk_multiplier,
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                    for time in curve_times
                ]
            )
        return curve_times, curves

    if offsets is None:
        try:
            times, curves = fit.survival_curve(rows, centered)
        except TypeError:
            if rows is None:
                raise ValueError("newdata is required for survival prediction") from None
            times, curves = fit.survival_curve(rows, None)
        return [float(value) for value in times], curves

    if rows is None:
        raise ValueError("newdata is required for survival prediction")
    if not hasattr(fit, "basehaz"):
        raise TypeError("model does not support baseline hazard prediction")

    curve_times, hazards = fit.basehaz(centered)
    linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    center = _training_linear_predictor_center(fit) if centered else 0.0
    curves = []
    for linear_predictor in linear_predictors:
        risk_multiplier = _safe_exp(linear_predictor - center)
        curves.append(
            [min(max(_safe_exp(-float(hazard) * risk_multiplier), 0.0), 1.0) for hazard in hazards]
        )
    return [float(value) for value in curve_times], curves


def _cox_default_survfit_linear_predictor(fit: Any) -> float:
    means = getattr(fit, "means", None)
    if means is None:
        return 0.0
    beta = _cox_beta(fit)
    mean_values = [float(value) for value in means]
    if len(mean_values) != len(beta):
        return 0.0
    return sum(value * coefficient for value, coefficient in zip(mean_values, beta, strict=True))


def _cox_survfit_curve_strata(
    fit: Any,
    rows: list[list[float]] | None,
    newdata: Any | None,
    n_curves: int,
) -> list[int] | None:
    if getattr(fit, "basehaz_with_strata", None) is None:
        return None
    beta = _cox_beta(fit)
    training_rows = _cox_training_rows(fit, len(beta))
    training_strata = _cox_training_strata(fit, len(training_rows)) if training_rows else [0]
    unique_strata = sorted(set(training_strata))
    if rows is None:
        if len(unique_strata) > 1 and n_curves == len(unique_strata):
            return unique_strata
        return None
    prediction_strata = _cox_prediction_strata(fit, newdata, len(rows))
    return prediction_strata if len(prediction_strata) == n_curves else None


def _cox_survfit_default_time0(fit: Any) -> float:
    values = [0.0]
    event_times = getattr(fit, "event_times", None)
    if event_times is not None:
        values.extend(float(value) for value in event_times)
    entry_times = getattr(fit, "entry_times", None)
    if entry_times is not None:
        values.extend(float(value) for value in entry_times)
    return min(values)


def _cox_survfit_training_times(
    fit: Any,
    curve_strata: list[int] | None,
) -> list[float]:
    event_times = getattr(fit, "event_times", None)
    if event_times is None:
        return []
    times = [float(value) for value in event_times]
    if not times:
        return []
    strata = _cox_training_strata(fit, len(times))
    selected_strata = set(curve_strata) if curve_strata is not None else set(strata)
    return sorted(
        {time for time, stratum in zip(times, strata, strict=True) if stratum in selected_strata}
    )


def _cox_survfit_with_censor_times(
    fit: Any,
    result: CoxSurvfitResult,
) -> CoxSurvfitResult:
    times = _cox_survfit_training_times(fit, result.strata)
    if not times or times == result.time:
        return result

    expanded_cumhaz: list[list[float]] = []
    expanded_surv: list[list[float]] = []
    for hazards in result.cumhaz:
        hazard_values = [float(value) for value in hazards]
        curve_hazards = [_step_hazard_at(result.time, hazard_values, time) for time in times]
        expanded_cumhaz.append(curve_hazards)
        expanded_surv.append([_clamp_probability(_safe_exp(-hazard)) for hazard in curve_hazards])

    return CoxSurvfitResult(
        time=times,
        surv=expanded_surv,
        cumhaz=expanded_cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        start_time=result.start_time,
        std_err=result.std_err,
        std_chaz=result.std_chaz,
        conf_lower=result.conf_lower,
        conf_upper=result.conf_upper,
    )


def _cox_survfit_conditioned(
    fit: Any,
    result: CoxSurvfitResult,
    start_time: float | None,
    include_time0: bool,
) -> CoxSurvfitResult:
    if start_time is None and not include_time0:
        return result

    t0 = start_time if start_time is not None else _cox_survfit_default_time0(fit)
    times = [float(value) for value in result.time]
    if start_time is None:
        keep_indices = list(range(len(times)))
    else:
        keep_indices = [idx for idx, time in enumerate(times) if time >= t0 - _SURVFIT_TIME_EPSILON]
        if not keep_indices:
            raise ValueError("start_time argument has removed all endpoints")

    start_pos = bisect_left(times, t0)
    kept_times = [times[idx] for idx in keep_indices]
    conditioned_surv: list[list[float]] = []
    conditioned_cumhaz: list[list[float]] = []
    for _curve, hazards in zip(result.surv, result.cumhaz, strict=True):
        hazard_values = [float(value) for value in hazards]
        start_hazard = hazard_values[start_pos - 1] if start_pos > 0 else 0.0
        curve_hazards = [max(hazard_values[idx] - start_hazard, 0.0) for idx in keep_indices]
        conditioned_cumhaz.append(curve_hazards)
        conditioned_surv.append(
            [_clamp_probability(_safe_exp(-hazard)) for hazard in curve_hazards]
        )

    if include_time0 and not (kept_times and abs(kept_times[0] - t0) < _SURVFIT_TIME_EPSILON):
        kept_times = [t0, *kept_times]
        conditioned_surv = [[1.0, *curve] for curve in conditioned_surv]
        conditioned_cumhaz = [[0.0, *curve] for curve in conditioned_cumhaz]

    return CoxSurvfitResult(
        time=kept_times,
        surv=conditioned_surv,
        cumhaz=conditioned_cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        start_time=t0 if start_time is not None else None,
        std_err=result.std_err,
        std_chaz=result.std_chaz,
        conf_lower=result.conf_lower,
        conf_upper=result.conf_upper,
    )


def _cox_survfit_curve_rows(
    fit: Any,
    rows: list[list[float]] | None,
    n_curves: int,
) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    if rows is not None:
        curve_rows = [[float(value) for value in row] for row in rows]
        if len(curve_rows) != n_curves:
            raise ValueError("newdata rows do not match fitted Cox survival curves")
        if any(len(row) != nvar for row in curve_rows):
            raise ValueError(f"newdata must have {nvar} columns")
        return curve_rows

    means = getattr(_unwrap_formula_fit(fit), "means", None)
    if means is None:
        row = [0.0] * nvar
    else:
        row = [float(value) for value in means]
        if len(row) != nvar:
            row = _cox_reference_means(fit, "sample")
    return [list(row) for _ in range(n_curves)]


def _cox_survfit_with_confidence(
    fit: Any,
    result: CoxSurvfitResult,
    rows: list[list[float]],
    conf_level: float,
    conf_type: str,
) -> CoxSurvfitResult:
    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    nvar = len(beta)
    variance = _cox_variance_matrix(model, nvar)
    baselines = _cox_expected_baseline_by_stratum(model)
    means = _cox_reference_means(model, "sample")
    z = NormalDist().inv_cdf(1.0 - (1.0 - conf_level) / 2.0)

    std_err: list[list[float]] = []
    std_chaz: list[list[float]] = []
    conf_lower: list[list[float]] = []
    conf_upper: list[list[float]] = []

    for curve_idx, (survival_curve, row, linear_predictor) in enumerate(
        zip(result.surv, rows, result.linear_predictors, strict=True)
    ):
        stratum = result.strata[curve_idx] if result.strata is not None else 0
        baseline = baselines.get(
            stratum,
            _CoxExpectedBaseline([], [], [], []),
        )
        centered_row = [float(value) - means[col_idx] for col_idx, value in enumerate(row)]
        start_hazard, start_varhaz, start_xbar = (
            _cox_expected_baseline_at(baseline, result.start_time, nvar)
            if result.start_time is not None
            else (0.0, 0.0, [0.0] * nvar)
        )
        curve_std_err: list[float] = []
        curve_std_chaz: list[float] = []
        curve_lower: list[float] = []
        curve_upper: list[float] = []
        risk = _safe_exp(float(linear_predictor))

        for time, survival in zip(result.time, survival_curve, strict=True):
            stop_hazard, stop_varhaz, stop_xbar = _cox_expected_baseline_at(
                baseline,
                float(time),
                nvar,
            )
            start_delta = [
                start_hazard * centered_row[col_idx] - start_xbar[col_idx]
                for col_idx in range(nvar)
            ]
            stop_delta = [
                stop_hazard * centered_row[col_idx] - stop_xbar[col_idx] for col_idx in range(nvar)
            ]
            interval_delta = [stop_delta[col_idx] - start_delta[col_idx] for col_idx in range(nvar)]
            variance_value = stop_varhaz - start_varhaz + _quadratic_form(interval_delta, variance)
            chaz_se = math.sqrt(max(variance_value, 0.0)) * risk
            surv_se = float(survival) * chaz_se
            curve_std_chaz.append(chaz_se)
            curve_std_err.append(surv_se)
            if conf_type != "none":
                lower, upper = _survfit_confidence_interval(
                    float(survival),
                    surv_se,
                    z,
                    conf_type,
                )
                curve_lower.append(lower)
                curve_upper.append(upper)

        std_chaz.append(curve_std_chaz)
        std_err.append(curve_std_err)
        if conf_type != "none":
            conf_lower.append(curve_lower)
            conf_upper.append(curve_upper)

    return CoxSurvfitResult(
        time=result.time,
        surv=result.surv,
        cumhaz=result.cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        start_time=result.start_time,
        std_err=std_err,
        std_chaz=std_chaz,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
    )


def _cox_survfit_result(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    centered: bool,
    newdata: Any | None,
    start_time: float | None = None,
    include_time0: bool = False,
    include_censor: bool = True,
    conf_level: float = 0.95,
    conf_type: str = "log",
    compute_confidence: bool = True,
) -> CoxSurvfitResult:
    times, curves = _cox_survival_curve(fit, rows, offsets, centered, newdata)
    center = _training_linear_predictor_center(fit) if centered else 0.0
    if rows is None:
        linear_predictors = [_cox_default_survfit_linear_predictor(fit)] * len(curves)
    else:
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    curve_strata = _cox_survfit_curve_strata(fit, rows, newdata, len(curves))
    basehaz_with_strata = getattr(fit, "basehaz_with_strata", None)
    if curve_strata is not None and basehaz_with_strata is not None:
        base_times, base_hazards, base_strata = basehaz_with_strata(centered)
        baselines = _cox_baselines_by_stratum(base_times, base_hazards, base_strata)
        cumhaz = []
        for linear_predictor, stratum in zip(linear_predictors, curve_strata, strict=True):
            stratum_times, stratum_hazards = baselines.get(stratum, ([], []))
            risk_multiplier = _safe_exp(linear_predictor - center)
            cumhaz.append(
                [
                    _step_hazard_at(stratum_times, stratum_hazards, time) * risk_multiplier
                    for time in times
                ]
            )
    else:
        baseline_times, baseline_hazards = fit.basehaz(centered)
        baseline_times = [float(value) for value in baseline_times]
        hazards = [float(value) for value in baseline_hazards]
        if len(baseline_times) != len(times) or any(
            abs(left - right) > _SURVFIT_TIME_EPSILON
            for left, right in zip(baseline_times, times, strict=False)
        ):
            hazards = [
                0.0 if (pos := bisect_right(baseline_times, time)) == 0 else hazards[pos - 1]
                for time in times
            ]
        risk_multipliers = [_safe_exp(value - center) for value in linear_predictors]
        cumhaz = [[hazard * risk for hazard in hazards] for risk in risk_multipliers]
    result = CoxSurvfitResult(
        time=times,
        surv=[[float(value) for value in curve] for curve in curves],
        cumhaz=cumhaz,
        linear_predictors=linear_predictors,
        centered=centered,
        strata=curve_strata,
    )
    if include_censor:
        result = _cox_survfit_with_censor_times(fit, result)
    result = _cox_survfit_conditioned(fit, result, start_time, include_time0)
    if not compute_confidence:
        return result
    curve_rows = _cox_survfit_curve_rows(fit, rows, len(result.surv))
    return _cox_survfit_with_confidence(fit, result, curve_rows, conf_level, conf_type)


def _cox_survival_curve_with_se(
    fit: Any,
    rows: list[list[float]] | None,
    offsets: list[float] | None,
    centered: bool,
    newdata: Any | None,
    times: Any | None,
    collapse: Any,
) -> PredictResult:
    result = _cox_survfit_result(
        fit,
        rows,
        offsets,
        centered,
        newdata,
        include_censor=False,
        conf_type="none",
    )
    curve_times = [float(value) for value in result.time]
    curves = [[float(value) for value in curve] for curve in result.surv]
    std_err = [[float(value) for value in curve] for curve in result.std_err]

    if times is not None:
        requested_times = _float_vector(times, "times")
        curves = [_step_curve_at(curve_times, curve, requested_times) for curve in curves]
        std_err = [_step_std_err_at(curve_times, curve, requested_times) for curve in std_err]
        curve_times = requested_times

    return PredictResult(
        (curve_times, _collapse_prediction_result(curves, collapse)),
        (curve_times, _collapse_prediction_se(std_err, collapse)),
    )


def _matrix_score_columns(rows: list[list[float]]) -> tuple[list[list[float]], list[str]]:
    width = len(rows[0]) if rows else 0
    columns = [[row[col_idx] for row in rows] for col_idx in range(width)]
    names = [f"score{col_idx + 1}" for col_idx in range(width)]
    return columns, names


def _external_concordance_score_columns(
    scores: Any,
    n: int,
) -> tuple[list[list[float]], list[str]]:
    rows = _as_matrix_rows(scores, "scores", allow_empty_columns=False)
    if len(rows) != n:
        raise ValueError("scores must have the same number of rows as the Surv response")
    return _matrix_score_columns(rows)


def _concordance_score_columns(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[list[float]], list[str]]:
    if not terms.covariates and not terms.offsets:
        raise ValueError("concordance formula requires a risk score or offset term")

    columns: list[list[float]] = []
    names: list[str] = []
    for term in terms.covariates:
        design_term = _fit_design_term(data, term, n)
        term_columns = _design_term_columns(data, design_term, n)
        columns.extend(term_columns)
        names.extend(_design_term_output_names(design_term))

    offsets = _offset_vector(data, terms.offsets, n) if terms.offsets else None
    if not columns:
        if offsets is None:
            raise ValueError("concordance formula requires a risk score or offset term")
        return [offsets], ["offset"]

    if offsets is not None:
        columns = [[value + offsets[idx] for idx, value in enumerate(column)] for column in columns]
    return columns, names


_CONCORDANCE_TIMEWT_CHOICES = ("n", "S", "S/G", "n/G2", "I")


def _normalize_concordance_timewt(timewt: Any) -> str:
    if timewt is None:
        return "n"
    if isinstance(timewt, str):
        value = timewt
    else:
        try:
            values = list(timewt)
        except TypeError as exc:
            raise TypeError("timewt must be a string") from exc
        if tuple(values) == _CONCORDANCE_TIMEWT_CHOICES:
            return "n"
        if len(values) != 1:
            raise ValueError("timewt must be a single value")
        value = str(values[0])
    if value not in _CONCORDANCE_TIMEWT_CHOICES:
        choices = "', '".join(_CONCORDANCE_TIMEWT_CHOICES)
        raise ValueError(f"timewt must be one of '{choices}'")
    return value


def _normalize_concordance_influence(influence: Any) -> int:
    if influence is None:
        return 0
    if _is_bool_like(influence):
        value = int(bool(influence))
    else:
        value = _integer_scalar(influence, "influence")
    if value not in {0, 1, 2, 3}:
        raise ValueError("influence must be 0, 1, 2, or 3")
    return value


def _validate_concordance_keepstrata(keepstrata: Any) -> None:
    if keepstrata is None:
        return
    if _is_bool_like(keepstrata):
        return
    _finite_float(keepstrata, "keepstrata")


def _normalize_concordance_time_bound(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _formula_weight_values(data: Any, weights: Any | None) -> Any | None:
    if isinstance(weights, str):
        return _column(data, weights)
    return weights


def _formula_cluster_values(data: Any, cluster: Any | None) -> Any | None:
    if isinstance(cluster, str):
        return _column(data, cluster)
    return cluster


def _concordance_weight_values(weights: Any | None, n: int) -> list[float] | None:
    values = _optional_float_vector(weights, "weights", n)
    if values is None:
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("weights must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("weights must be non-negative")
    return values


def _exact_counting_concordance_summary(
    start: list[float],
    stop: list[float],
    status: list[int],
    risk_values: list[float],
    weights: list[float] | None,
    timewt: str,
) -> dict[str, float]:
    event_time_multipliers = _counting_time_weight_multipliers(
        start,
        stop,
        status,
        weights,
        timewt,
    )
    concordant = 0.0
    comparable = 0.0
    for event_idx, event in enumerate(status):
        if event != 1:
            continue
        event_time = stop[event_idx]
        event_time_multiplier = event_time_multipliers.get(event_time, 0.0)
        if event_time_multiplier <= 0.0:
            continue
        for risk_idx in range(len(stop)):
            if risk_idx == event_idx:
                continue
            if start[risk_idx] < event_time < stop[risk_idx]:
                pair_weight = (
                    1.0 if weights is None else float(weights[event_idx]) * float(weights[risk_idx])
                ) * event_time_multiplier
                comparable += pair_weight
                diff = risk_values[event_idx] - risk_values[risk_idx]
                if diff > 0.0:
                    concordant += pair_weight
                elif abs(diff) < _CONCORDANCE_RISK_TIE_FLOOR:
                    concordant += 0.5 * pair_weight
    return {
        "concordance": concordant / comparable if comparable > 0.0 else 0.5,
        "concordant": concordant,
        "comparable": comparable,
    }


def _concordance_time_weight_multiplier(
    timewt: str,
    total_weight: float,
    survival: float,
    censoring_survival: float,
    nrisk: float,
) -> float:
    if nrisk <= 0.0:
        return 0.0
    if timewt == "S":
        return total_weight * survival / nrisk
    if timewt == "S/G":
        return (
            total_weight * survival / (censoring_survival * nrisk)
            if censoring_survival > 0.0
            else 0.0
        )
    if timewt == "n/G2":
        return 1.0 / (censoring_survival * censoring_survival) if censoring_survival > 0.0 else 0.0
    if timewt == "I":
        return 1.0 / nrisk
    return 1.0


def _counting_time_weight_multipliers(
    start: list[float],
    stop: list[float],
    status: list[int],
    weights: list[float] | None,
    timewt: str,
) -> dict[float, float]:
    if timewt == "n":
        return dict.fromkeys(
            {stop[idx] for idx, event in enumerate(status) if event == 1},
            1.0,
        )
    total_weight = float(len(stop)) if weights is None else sum(float(value) for value in weights)
    survival = 1.0
    multipliers: dict[float, float] = {}
    for event_time in sorted({stop[idx] for idx, event in enumerate(status) if event == 1}):
        nrisk = sum(
            (1.0 if weights is None else float(weights[idx]))
            for idx, (entry, exit_time) in enumerate(zip(start, stop, strict=True))
            if entry < event_time <= exit_time
        )
        multipliers[event_time] = _concordance_time_weight_multiplier(
            timewt,
            total_weight,
            survival,
            1.0,
            nrisk,
        )
        death_weight = sum(
            (1.0 if weights is None else float(weights[idx]))
            for idx, event in enumerate(status)
            if event == 1 and stop[idx] == event_time
        )
        if nrisk > 0.0:
            survival *= max((nrisk - death_weight) / nrisk, 0.0)
    return multipliers


def _concordance_bounded_times_and_status(
    times: list[float],
    status: list[int],
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[float], list[int]]:
    bounded_times = [max(value, ymin) for value in times] if ymin is not None else list(times)
    bounded_status = list(status)
    if ymax is not None:
        bounded_status = [
            0 if event == 1 and bounded_times[idx] > ymax else event
            for idx, event in enumerate(bounded_status)
        ]
    return bounded_times, bounded_status


def _right_concordance_time_weight_multipliers(
    time: list[float],
    status: list[int],
    weights: list[float],
    timewt: str,
) -> dict[float, float]:
    if timewt == "n":
        return dict.fromkeys(
            {time[idx] for idx, event in enumerate(status) if event == 1},
            1.0,
        )
    total_weight = math.fsum(weights)
    survival = 1.0
    censoring_survival = 1.0
    multipliers: dict[float, float] = {}
    for event_time in sorted(set(time)):
        indices = [idx for idx, value in enumerate(time) if value == event_time]
        nrisk = math.fsum(weights[idx] for idx, value in enumerate(time) if value >= event_time)
        death_weight = math.fsum(weights[idx] for idx in indices if status[idx] == 1)
        censor_weight = math.fsum(weights[idx] for idx in indices if status[idx] != 1)
        if death_weight > 0.0:
            multipliers[event_time] = _concordance_time_weight_multiplier(
                timewt,
                total_weight,
                survival,
                censoring_survival,
                nrisk,
            )
            if nrisk > 0.0:
                survival *= max((nrisk - death_weight) / nrisk, 0.0)
        if censor_weight > 0.0 and nrisk > 0.0:
            censoring_survival *= max((nrisk - censor_weight) / nrisk, 0.0)
    return multipliers


def _concordance_rank_at_event(
    risk_values: list[float],
    weights: list[float],
    event_idx: int,
    at_risk: list[int],
) -> tuple[float, float] | None:
    risk_weight = math.fsum(weights[idx] for idx in at_risk)
    if risk_weight <= 0.0:
        return None
    event_risk = risk_values[event_idx]
    greater = 0.0
    lower = 0.0
    for idx in at_risk:
        diff = risk_values[idx] - event_risk
        if diff > _CONCORDANCE_RISK_TIE_FLOOR:
            greater += weights[idx]
        elif diff < -_CONCORDANCE_RISK_TIE_FLOOR:
            lower += weights[idx]
    return (lower - greater) / risk_weight, risk_weight


def _single_concordance_ranks(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> list[dict[str, float]]:
    case_weights = [1.0] * len(response) if weights is None else list(weights)
    rows: list[dict[str, float]] = []
    if response.type == "right":
        times = _survdiff_timefix_values(list(response.time), timefix)
        status = list(response.event)
        times, status = _concordance_bounded_times_and_status(times, status, ymin, ymax)
        multipliers = _right_concordance_time_weight_multipliers(
            times,
            status,
            case_weights,
            timewt,
        )
        event_indices = sorted(
            (idx for idx, event in enumerate(status) if event == 1),
            key=lambda idx: (times[idx], idx),
        )
        for event_idx in event_indices:
            event_time = times[event_idx]
            multiplier = multipliers.get(event_time, 0.0)
            if multiplier <= 0.0:
                continue
            at_risk = [idx for idx, value in enumerate(times) if value >= event_time]
            rank_result = _concordance_rank_at_event(
                risk_values,
                case_weights,
                event_idx,
                at_risk,
            )
            if rank_result is None:
                continue
            rank, risk_weight = rank_result
            rows.append(
                {
                    "time": event_time,
                    "rank": rank,
                    "timewt": risk_weight * multiplier,
                    "casewt": case_weights[event_idx],
                }
            )
        return rows
    if response.type == "counting":
        if timewt in {"S/G", "n/G2"}:
            raise ValueError(
                "S/G and n/G2 timewt options are not supported for counting-process data"
            )
        if response.start is None:
            raise ValueError("counting-process concordance requires start times")
        start = list(response.start)
        stop = list(response.time)
        if timefix:
            start, stop = _timefix_vectors(start, stop)
        status = list(response.event)
        stop, status = _concordance_bounded_times_and_status(stop, status, ymin, ymax)
        multipliers = _counting_time_weight_multipliers(
            start,
            stop,
            status,
            case_weights,
            timewt,
        )
        event_indices = sorted(
            (idx for idx, event in enumerate(status) if event == 1),
            key=lambda idx: (stop[idx], idx),
        )
        for event_idx in event_indices:
            event_time = stop[event_idx]
            multiplier = multipliers.get(event_time, 0.0)
            if multiplier <= 0.0:
                continue
            at_risk = [
                idx
                for idx, (entry, exit_time) in enumerate(zip(start, stop, strict=True))
                if entry < event_time <= exit_time
            ]
            rank_result = _concordance_rank_at_event(
                risk_values,
                case_weights,
                event_idx,
                at_risk,
            )
            if rank_result is None:
                continue
            rank, risk_weight = rank_result
            rows.append(
                {
                    "time": event_time,
                    "rank": rank,
                    "timewt": risk_weight * multiplier,
                    "casewt": case_weights[event_idx],
                }
            )
        return rows
    raise NotImplementedError(
        "concordance currently supports right-censored and counting Surv responses"
    )


def _concordance_ranks(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> list[dict[str, float]]:
    if strata is None:
        return _single_concordance_ranks(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
    rows: list[dict[str, float]] = []
    for indices in _group_indices(strata, len(response)).values():
        group_response = _subset_surv(response, indices)
        group_risk = [risk_values[idx] for idx in indices]
        group_weights = [weights[idx] for idx in indices] if weights is not None else None
        rows.extend(
            _single_concordance_ranks(
                group_response,
                group_risk,
                group_weights,
                timefix,
                timewt,
                ymin,
                ymax,
            )
        )
    return sorted(rows, key=lambda row: row["time"])


def _add_concordance_pair_influence(
    influence_rows: list[list[float]],
    left: int,
    right: int,
    column: int,
    value: float,
) -> None:
    share = 0.5 * value
    influence_rows[left][column] += share
    influence_rows[right][column] += share


def _concordance_influence_from_rows(
    influence_rows: list[list[float]],
    concordant: float,
    comparable: float,
) -> tuple[list[list[float]], list[float], float | None]:
    if comparable <= 0.0:
        return influence_rows, [0.0 for _ in influence_rows], 0.0
    somer = (2.0 * concordant - comparable) / comparable
    dfbeta = []
    for row in influence_rows:
        comparable_row = row[0] + row[1] + row[2]
        dfbeta.append(((row[0] - row[1]) - comparable_row * somer) / (2.0 * comparable))
    return influence_rows, dfbeta, math.fsum(value * value for value in dfbeta)


def _single_concordance_influence(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[list[float]], list[float], float | None]:
    case_weights = [1.0] * len(response) if weights is None else list(weights)
    influence_rows = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(len(response))]
    concordant = 0.0
    comparable = 0.0
    if response.type == "right":
        times = _survdiff_timefix_values(list(response.time), timefix)
        status = list(response.event)
        times, status = _concordance_bounded_times_and_status(times, status, ymin, ymax)
        multipliers = _right_concordance_time_weight_multipliers(
            times,
            status,
            case_weights,
            timewt,
        )
        for left in range(len(times)):
            for right in range(left + 1, len(times)):
                if status[left] == 1 and status[right] == 1 and times[left] == times[right]:
                    multiplier = multipliers.get(times[left], 0.0)
                    pair_weight = case_weights[left] * case_weights[right] * multiplier
                    if pair_weight <= 0.0:
                        continue
                    column = (
                        4
                        if abs(risk_values[left] - risk_values[right]) < _CONCORDANCE_RISK_TIE_FLOOR
                        else 3
                    )
                    _add_concordance_pair_influence(
                        influence_rows,
                        left,
                        right,
                        column,
                        pair_weight,
                    )
                    continue
                if status[left] == 1 and times[left] < times[right]:
                    event_idx, risk_idx = left, right
                elif status[right] == 1 and times[right] < times[left]:
                    event_idx, risk_idx = right, left
                else:
                    continue
                multiplier = multipliers.get(times[event_idx], 0.0)
                pair_weight = case_weights[event_idx] * case_weights[risk_idx] * multiplier
                if pair_weight <= 0.0:
                    continue
                comparable += pair_weight
                diff = risk_values[event_idx] - risk_values[risk_idx]
                if diff > _CONCORDANCE_RISK_TIE_FLOOR:
                    concordant += pair_weight
                    column = 0
                elif diff < -_CONCORDANCE_RISK_TIE_FLOOR:
                    column = 1
                else:
                    concordant += 0.5 * pair_weight
                    column = 2
                _add_concordance_pair_influence(
                    influence_rows,
                    event_idx,
                    risk_idx,
                    column,
                    pair_weight,
                )
        return _concordance_influence_from_rows(influence_rows, concordant, comparable)
    if response.type == "counting":
        if timewt in {"S/G", "n/G2"}:
            raise ValueError(
                "S/G and n/G2 timewt options are not supported for counting-process data"
            )
        if response.start is None:
            raise ValueError("counting-process concordance requires start times")
        start = list(response.start)
        stop = list(response.time)
        if timefix:
            start, stop = _timefix_vectors(start, stop)
        status = list(response.event)
        stop, status = _concordance_bounded_times_and_status(stop, status, ymin, ymax)
        multipliers = _counting_time_weight_multipliers(
            start,
            stop,
            status,
            case_weights,
            timewt,
        )
        for event_idx, event in enumerate(status):
            if event != 1:
                continue
            event_time = stop[event_idx]
            multiplier = multipliers.get(event_time, 0.0)
            if multiplier <= 0.0:
                continue
            for risk_idx in range(len(stop)):
                if risk_idx == event_idx:
                    continue
                pair_weight = case_weights[event_idx] * case_weights[risk_idx] * multiplier
                if pair_weight <= 0.0:
                    continue
                if status[risk_idx] == 1 and stop[risk_idx] == event_time:
                    if event_idx < risk_idx:
                        column = (
                            4
                            if abs(risk_values[event_idx] - risk_values[risk_idx])
                            < _CONCORDANCE_RISK_TIE_FLOOR
                            else 3
                        )
                        _add_concordance_pair_influence(
                            influence_rows,
                            event_idx,
                            risk_idx,
                            column,
                            pair_weight,
                        )
                    continue
                if not (start[risk_idx] < event_time < stop[risk_idx]):
                    continue
                comparable += pair_weight
                diff = risk_values[event_idx] - risk_values[risk_idx]
                if diff > _CONCORDANCE_RISK_TIE_FLOOR:
                    concordant += pair_weight
                    column = 0
                elif diff < -_CONCORDANCE_RISK_TIE_FLOOR:
                    column = 1
                else:
                    concordant += 0.5 * pair_weight
                    column = 2
                _add_concordance_pair_influence(
                    influence_rows,
                    event_idx,
                    risk_idx,
                    column,
                    pair_weight,
                )
        return _concordance_influence_from_rows(influence_rows, concordant, comparable)
    raise NotImplementedError(
        "concordance currently supports right-censored and counting Surv responses"
    )


def _concordance_influence(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[list[float]], list[float], float | None]:
    if strata is None:
        return _single_concordance_influence(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
    influence_rows = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(len(response))]
    dfbeta = [0.0 for _ in range(len(response))]
    variance = 0.0
    for indices in _group_indices(strata, len(response)).values():
        group_response = _subset_surv(response, indices)
        group_risk = [risk_values[idx] for idx in indices]
        group_weights = [weights[idx] for idx in indices] if weights is not None else None
        group_influence, group_dfbeta, group_variance = _single_concordance_influence(
            group_response,
            group_risk,
            group_weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
        for local_idx, original_idx in enumerate(indices):
            influence_rows[original_idx] = group_influence[local_idx]
            dfbeta[original_idx] = group_dfbeta[local_idx]
        if group_variance is not None:
            variance += group_variance
    return influence_rows, dfbeta, variance


def _concordance_cluster_values(cluster: Any, n: int) -> list[Any]:
    values = _materialize_labels(cluster, "cluster")
    if len(values) != n:
        raise ValueError("cluster must have the same length as the Surv response")
    _label_levels(values, "cluster")
    return values


def _clustered_concordance_dfbeta(
    dfbeta: list[float],
    cluster: list[Any],
) -> tuple[list[float], float]:
    collapsed: dict[Any, float] = {}
    order: list[Any] = []
    for label, value in zip(cluster, dfbeta, strict=True):
        if label not in collapsed:
            collapsed[label] = 0.0
            order.append(label)
        collapsed[label] += value
    cluster_dfbeta = [collapsed[label] for label in order]
    return cluster_dfbeta, math.fsum(value * value for value in cluster_dfbeta)


def _single_concordance_summary(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> dict[str, float]:
    if response.type == "right":
        times = _survdiff_timefix_values(list(response.time), timefix)
        status = list(response.event)
        times, status = _concordance_bounded_times_and_status(times, status, ymin, ymax)
        summary = _core.concordance_summary(
            times,
            status,
            risk_values,
            weights,
            timewt,
        )
        summary["n_event"] = float(sum(1 for event in status if event == 1))
        return summary
    if response.type == "counting":
        if timewt in {"S/G", "n/G2"}:
            raise ValueError(
                "S/G and n/G2 timewt options are not supported for counting-process data"
            )
        if response.start is None:
            raise ValueError("counting-process concordance requires start times")
        start = list(response.start)
        stop = list(response.time)
        if timefix:
            start, stop = _timefix_vectors(start, stop)
        status = list(response.event)
        stop, status = _concordance_bounded_times_and_status(stop, status, ymin, ymax)
        if not timefix:
            summary = _exact_counting_concordance_summary(
                start,
                stop,
                status,
                risk_values,
                weights,
                timewt,
            )
            summary["n_event"] = float(sum(1 for event in status if event == 1))
            return summary
        summary = _core.counting_concordance_summary(
            start,
            stop,
            status,
            risk_values,
            weights,
            timewt,
        )
        summary["n_event"] = float(sum(1 for event in status if event == 1))
        return summary
    raise NotImplementedError(
        "concordance currently supports right-censored and counting Surv responses"
    )


def _concordance_summary(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> dict[str, float]:
    if strata is None:
        return _single_concordance_summary(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )

    concordant = 0.0
    comparable = 0.0
    n_event = 0.0
    for indices in _group_indices(strata, len(response)).values():
        group_response = _subset_surv(response, indices)
        group_risk = [risk_values[idx] for idx in indices]
        group_weights = [weights[idx] for idx in indices] if weights is not None else None
        summary = _single_concordance_summary(
            group_response,
            group_risk,
            group_weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
        concordant += float(summary["concordant"])
        comparable += float(summary["comparable"])
        n_event += float(summary["n_event"])

    return {
        "concordance": concordant / comparable if comparable > 0.0 else 0.5,
        "concordant": concordant,
        "comparable": comparable,
        "n_event": n_event,
    }


def _single_score_concordance_result(
    response: Surv,
    score_values: list[float],
    weight_values: list[float] | None,
    strata_values: Any | None,
    cluster_values: list[Any] | None,
    reverse_scores: bool,
    fix_time: bool,
    time_weight: str,
    lower_bound: float | None,
    upper_bound: float | None,
    influence_value: int,
    include_ranks: bool,
) -> ConcordanceResult:
    if len(score_values) != len(response):
        raise ValueError("scores must have the same length as the Surv response")

    risk_values = [-value for value in score_values] if reverse_scores else score_values
    summary = _concordance_summary(
        response,
        risk_values,
        weight_values,
        strata_values,
        fix_time,
        time_weight,
        lower_bound,
        upper_bound,
    )
    rank_rows = (
        _concordance_ranks(
            response,
            risk_values,
            weight_values,
            strata_values,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
        )
        if include_ranks
        else None
    )
    influence_rows = None
    dfbeta = None
    variance = None
    if influence_value or cluster_values is not None:
        influence_rows, dfbeta, variance = _concordance_influence(
            response,
            risk_values,
            weight_values,
            strata_values,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
        )
        if cluster_values is not None and dfbeta is not None:
            dfbeta, variance = _clustered_concordance_dfbeta(dfbeta, cluster_values)
    return ConcordanceResult(
        concordance=float(summary["concordance"]),
        n=len(response),
        n_event=int(summary["n_event"]),
        reverse=reverse_scores,
        concordant=float(summary["concordant"]),
        comparable=float(summary["comparable"]),
        ranks=rank_rows,
        dfbeta=dfbeta if influence_value in {1, 3} else None,
        influence=influence_rows if influence_value in {2, 3} else None,
        variance=variance if influence_value or cluster_values is not None else None,
    )


def _multi_score_concordance_result(
    response: Surv,
    score_columns: list[list[float]],
    score_names: list[str],
    weight_values: list[float] | None,
    strata_values: Any | None,
    cluster_values: list[Any] | None,
    reverse_scores: bool,
    fix_time: bool,
    time_weight: str,
    lower_bound: float | None,
    upper_bound: float | None,
    influence_value: int,
    include_ranks: bool,
) -> ConcordanceResult:
    results = [
        _single_score_concordance_result(
            response,
            score_values,
            weight_values,
            strata_values,
            cluster_values,
            reverse_scores,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
            influence_value,
            include_ranks,
        )
        for score_values in score_columns
    ]
    return ConcordanceResult(
        concordance=[float(result.concordance) for result in results],
        n=len(response),
        n_event=results[0].n_event if results else 0,
        reverse=reverse_scores,
        concordant=[float(result.concordant) for result in results],
        comparable=[float(result.comparable) for result in results],
        ranks=[result.ranks for result in results] if include_ranks else None,
        dfbeta=[result.dfbeta for result in results] if influence_value in {1, 3} else None,
        influence=[result.influence for result in results] if influence_value in {2, 3} else None,
        variance=(
            [result.variance for result in results]
            if influence_value or cluster_values is not None
            else None
        ),
        score_names=score_names,
    )


def concordance(
    response: Surv | str,
    data: Any | None = None,
    *,
    scores: Any | None = None,
    risk_scores: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    cluster: Any | None = None,
    ymin: Any | None = None,
    ymax: Any | None = None,
    timewt: Any = "n",
    influence: Any = 0,
    ranks: bool = False,
    reverse: bool = False,
    timefix: bool = True,
    keepstrata: Any = 10,
    **kwargs: Any,
) -> ConcordanceResult:
    """R-style concordance wrapper backed by Rust Harrell C-index."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", timefix, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"concordance got unexpected keyword argument(s): {unexpected}")

    reverse_scores = _normalize_bool_option(reverse, "reverse")
    fix_time = _normalize_bool_option(timefix, "timefix")
    time_weight = _normalize_concordance_timewt(timewt)
    lower_bound = _normalize_concordance_time_bound(ymin, "ymin")
    upper_bound = _normalize_concordance_time_bound(ymax, "ymax")
    influence_value = _normalize_concordance_influence(influence)
    include_ranks = _normalize_bool_option(ranks, "ranks")
    _validate_concordance_keepstrata(keepstrata)
    if scores is not None and risk_scores is not None:
        raise ValueError("use only one of scores or risk_scores")
    external_scores = risk_scores if risk_scores is not None else scores
    strata_values = None
    cluster_values = None
    score_names: list[str] | None = None

    if isinstance(response, str):
        if external_scores is not None:
            raise ValueError("concordance formula input cannot be combined with scores")
        weights = _formula_weight_values(data, weights)
        cluster = _formula_cluster_values(data, cluster)
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                response,
                data,
                subset,
                weights=weights,
                cluster=cluster,
            )
            weights = aligned["weights"]
            cluster = aligned["cluster"]
            subset = None
        data, aligned = _apply_formula_na_action(
            response,
            data,
            na_action,
            weights=weights,
            cluster=cluster,
        )
        weights = aligned["weights"]
        cluster = aligned["cluster"]
        na_action = "pass"
        response, terms = _parse_formula(response, data)
        if terms.clusters:
            if cluster is not None:
                raise ValueError("use only one of formula cluster(...) or cluster")
            cluster = _combined_columns(data, terms.clusters, len(response))
        score_columns, score_names = _concordance_score_columns(data, terms, len(response))
        if terms.strata:
            strata_values = _combined_columns(data, terms.strata, len(response))
        weight_values = _concordance_weight_values(weights, len(response))
    else:
        if not isinstance(response, Surv):
            raise TypeError("concordance response must be a Surv object or formula")
        if external_scores is None:
            raise ValueError("scores are required when response is not a formula")
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            external_scores = _subset_sequence(external_scores, indices, "scores")
            weights = _subset_optional_sequence(weights, indices, "weights")
            cluster = _subset_optional_sequence(cluster, indices, "cluster")
        response, aligned = _apply_surv_na_action(
            response,
            na_action,
            "concordance inputs",
            scores=external_scores,
            weights=weights,
            cluster=cluster,
        )
        external_scores = aligned["scores"]
        weights = aligned["weights"]
        cluster = aligned["cluster"]
        score_columns, score_names = _external_concordance_score_columns(
            external_scores,
            len(response),
        )
        weight_values = _concordance_weight_values(weights, len(response))

    if cluster is not None:
        cluster_values = _concordance_cluster_values(cluster, len(response))

    if len(score_columns) == 1:
        result = _single_score_concordance_result(
            response,
            score_columns[0],
            weight_values,
            strata_values,
            cluster_values,
            reverse_scores,
            fix_time,
            time_weight,
            lower_bound,
            upper_bound,
            influence_value,
            include_ranks,
        )
        return (
            ConcordanceResult(
                concordance=result.concordance,
                n=result.n,
                n_event=result.n_event,
                reverse=result.reverse,
                concordant=result.concordant,
                comparable=result.comparable,
                ranks=result.ranks,
                dfbeta=result.dfbeta,
                influence=result.influence,
                variance=result.variance,
                score_names=score_names,
            )
            if score_names is not None
            else result
        )

    return _multi_score_concordance_result(
        response,
        score_columns,
        score_names or [f"score{idx + 1}" for idx in range(len(score_columns))],
        weight_values,
        strata_values,
        cluster_values,
        reverse_scores,
        fix_time,
        time_weight,
        lower_bound,
        upper_bound,
        influence_value,
        include_ranks,
    )


def predict(
    fit: Any,
    newdata: Any | None = None,
    *,
    type: str | None = None,  # noqa: A002
    centered: bool | None = None,
    terms: Any | None = None,
    collapse: Any = False,
    reference: str | None = None,
    se_fit: bool = False,
    times: Any | None = None,
    p: Any | None = None,
    quantiles: Any | None = None,
    **kwargs: Any,
) -> Any:
    """R-style prediction generic for fitted survival models."""

    se_fit = _pop_dotted_keyword(kwargs, "se.fit", "se_fit", se_fit, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"predict got unexpected keyword argument(s): {unexpected}")

    is_survreg = _is_survreg_fit(fit)
    predict_type = _normalize_predict_type(
        type if type is not None else ("response" if is_survreg else "lp"),
        survreg=is_survreg,
    )
    centered_value = _normalize_optional_bool_option(centered, "centered")
    include_se = _normalize_bool_option(se_fit, "se_fit")
    rows, offsets = _prediction_inputs(fit, newdata)

    if is_survreg:
        if not _collapse_is_false(collapse):
            raise ValueError("collapse is only supported for Cox model predictions")
        if reference is not None:
            raise ValueError("reference is only supported for Cox model predictions")
        if centered_value:
            raise ValueError("centered predictions are only supported for Cox models")
        if predict_type in {"survival", "expected", "risk"}:
            raise ValueError(f"predict type={predict_type!r} is not supported for survreg fits")
        if predict_type == "terms":
            term_predictions = _survreg_predict_terms(fit, rows, terms)
            if include_se:
                return PredictResult(
                    term_predictions,
                    _survreg_term_prediction_se(fit, rows, terms),
                )
            return term_predictions
        if predict_type in {"quantile", "uquantile"}:
            if p is not None and quantiles is not None:
                raise ValueError("use only one of p or quantiles")
            q_values = quantiles if quantiles is not None else p
            q = _survreg_quantile_probabilities(q_values)
            scores = _survreg_quantile_scores(fit, q)
            predictions = _survreg_quantile_prediction_matrix(
                fit,
                rows,
                offsets,
                scores,
                predict_type,
                newdata,
            )
            if include_se:
                se = _survreg_quantile_prediction_se_matrix(
                    fit,
                    rows,
                    scores,
                    predictions,
                    predict_type,
                    newdata,
                )
                return PredictResult(
                    _drop_single_quantile(predictions, q),
                    _drop_single_quantile(se, q),
                )
            return _drop_single_quantile(predictions, q)
        result = fit.predict(rows, predict_type, offsets, False)
        if include_se:
            return PredictResult(
                result.predictions,
                _survreg_prediction_se(fit, rows, predict_type, result.predictions),
            )
        return result.predictions

    reference_name = _normalize_predict_reference(reference, centered_value, predict_type)

    if predict_type == "survival":
        if not hasattr(fit, "survival_curve"):
            raise TypeError("model does not support survival curve prediction")
        has_response = (
            rows is not None and times is None and _newdata_has_formula_response(fit, newdata)
        )
        if include_se:
            if times is not None or (rows is not None and not has_response):
                return _cox_survival_curve_with_se(
                    fit,
                    rows,
                    offsets,
                    True if centered_value is None else centered_value,
                    newdata,
                    times,
                    collapse,
                )
            expected = _cox_expected_events_with_se(fit, rows, offsets, newdata)
            probabilities = [_clamp_probability(_safe_exp(-value)) for value in expected.fit]
            probability_se = [
                float(se) * probability
                for se, probability in zip(expected.se_fit, probabilities, strict=True)
            ]
            return PredictResult(
                _collapse_prediction_result(probabilities, collapse),
                _collapse_prediction_se(probability_se, collapse),
            )
        if has_response:
            if rows is None:
                raise AssertionError("has_response implies prediction rows are available")
            probabilities = [
                _clamp_probability(_safe_exp(-expected))
                for expected in _cox_expected_events_for_newdata(fit, rows, offsets, newdata)
            ]
            return _collapse_prediction_result(probabilities, collapse)
        curve_times, curves = _cox_survival_curve(
            fit,
            rows,
            offsets,
            True if centered_value is None else centered_value,
            newdata,
        )
        if times is None:
            return curve_times, _collapse_prediction_result(curves, collapse)
        requested_times = _float_vector(times, "times")
        stepped_curves = [
            _step_curve_at(curve_times, [float(value) for value in curve], requested_times)
            for curve in curves
        ]
        return requested_times, _collapse_prediction_result(stepped_curves, collapse)

    if predict_type == "expected":
        if include_se:
            expected = _cox_expected_events_with_se(fit, rows, offsets, newdata)
            return PredictResult(
                _collapse_prediction_result(expected.fit, collapse),
                _collapse_prediction_se(expected.se_fit, collapse),
            )
        if rows is not None:
            expected_values = _cox_expected_events_for_newdata(fit, rows, offsets, newdata)
            return _collapse_prediction_result(expected_values, collapse)
        if not hasattr(fit, "expected_events"):
            raise TypeError("model does not support expected event prediction")
        return _collapse_prediction_result(fit.expected_events(), collapse)

    if predict_type == "terms":
        term_predictions = _cox_predict_terms(fit, rows, terms, reference_name, newdata)
        if include_se:
            term_se = _cox_term_prediction_se(fit, rows, terms, reference_name, newdata)
            return PredictResult(
                _collapse_prediction_result(term_predictions, collapse),
                _collapse_prediction_se(term_se, collapse),
            )
        return _collapse_prediction_result(term_predictions, collapse)

    linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
    linear_se = (
        _cox_linear_prediction_se(fit, rows, reference_name, newdata)
        if include_se and predict_type in {"lp", "risk"}
        else None
    )
    if reference_name != "zero":
        centers = _cox_reference_centers(fit, reference_name, len(linear_predictors), newdata)
        linear_predictors = [
            value - center for value, center in zip(linear_predictors, centers, strict=True)
        ]
    if predict_type == "lp":
        if include_se:
            return PredictResult(
                _collapse_prediction_result(linear_predictors, collapse),
                _collapse_prediction_se(linear_se, collapse),
            )
        return _collapse_prediction_result(linear_predictors, collapse)
    if predict_type == "risk":
        risks = [_safe_exp(value) for value in linear_predictors]
        if include_se:
            if linear_se is None:
                raise AssertionError("se_fit risk predictions require linear SEs")
            risk_se = [float(se) * risk for se, risk in zip(linear_se, risks, strict=True)]
            return PredictResult(
                _collapse_prediction_result(risks, collapse),
                _collapse_prediction_se(risk_se, collapse),
            )
        return _collapse_prediction_result(risks, collapse)
    if predict_type == "response":
        raise ValueError("predict type='response' is only supported for survreg fits")
    if predict_type == "quantile":
        raise ValueError("predict type='quantile' is only supported for survreg fits")
    raise AssertionError(f"unhandled predict type {predict_type!r}")


def residuals(
    fit: Any,
    *,
    type: str = "martingale",  # noqa: A002
    terms: Any | None = None,
    collapse: Any = False,
    weighted: bool | None = None,
    rsigma: bool | None = None,
) -> Any:
    """R-style residual generic for fitted survival models."""

    weighted_value = _normalize_optional_bool_option(weighted, "weighted")
    rsigma_value = _normalize_optional_bool_option(rsigma, "rsigma")
    if _is_survreg_fit(fit):
        if terms is not None:
            raise ValueError("terms is only supported for Cox partial residuals")
        residual_type = _normalize_survreg_residual_type(type)
        if residual_type in {"dfbeta", "dfbetas"}:
            dfbeta_values = _survreg_dfbeta_residuals(fit, residual_type, rsigma=rsigma_value)
            if weighted_value:
                dfbeta_values = _weight_residual_result(
                    dfbeta_values,
                    _model_residual_weights(fit, len(dfbeta_values)),
                )
            return _collapse_residual_result(dfbeta_values, collapse, len(dfbeta_values))
        if residual_type == "matrix":
            matrix_values = _core.survreg_residual_matrix(
                fit.time,
                fit.status,
                fit.linear_predictors,
                fit.scale,
                fit.distribution,
                time2=getattr(fit, "time2", None),
            )
            if weighted_value:
                matrix_values = _weight_residual_result(
                    matrix_values,
                    _model_residual_weights(fit, len(matrix_values)),
                )
            return _collapse_residual_result(matrix_values, collapse, len(matrix_values))
        if residual_type in {"ldcase", "ldresp", "ldshape"}:
            influence_values = _survreg_influence_residuals(
                fit,
                residual_type,
                rsigma=rsigma_value,
            )
            if weighted_value:
                influence_values = _weight_residual_result(
                    influence_values,
                    _model_residual_weights(fit, len(influence_values)),
                )
            return _collapse_residual_result(influence_values, collapse, len(influence_values))
        if residual_type in {"response", "deviance", "working"}:
            scalar_values = fit.residuals(residual_type).residuals
            if weighted_value:
                scalar_values = _weight_residual_result(
                    scalar_values,
                    _model_residual_weights(fit, len(scalar_values)),
                )
            return _collapse_residual_result(scalar_values, collapse, len(scalar_values))
        raise AssertionError(f"unhandled survreg residual type {residual_type!r}")

    residual_type = _normalize_residual_type(type)
    if terms is not None and residual_type != "partial":
        raise ValueError("terms is only supported for Cox partial residuals")
    method_names = {
        "martingale": "martingale_residuals",
        "deviance": "deviance_residuals",
        "score": "score_residuals",
        "dfbeta": "dfbeta",
        "dfbetas": "dfbetas",
        "schoenfeld": "schoenfeld_residuals",
        "scaledsch": "scaled_schoenfeld_residuals",
        "partial": "partial_residuals",
    }
    use_weights = (
        residual_type in {"dfbeta", "dfbetas"} if weighted_value is None else weighted_value
    )

    if residual_type in {"schoenfeld", "scaledsch"}:
        method = getattr(fit, "schoenfeld_residuals", None)
        if method is None:
            raise TypeError(f"model does not support {residual_type} residuals")
        raw = method()
        if use_weights:
            weights = _model_residual_weights(fit, len(fit.status))
            event_weights = [weights[idx] for idx in _cox_event_indices(fit)]
            raw = _weight_residual_result(raw, event_weights)
        return raw if residual_type == "schoenfeld" else _cox_scaled_schoenfeld_from_raw(fit, raw)

    if residual_type == "deviance" and (use_weights or not _collapse_is_false(collapse)):
        martingale_method = getattr(fit, "martingale_residuals", None)
        if martingale_method is None:
            raise TypeError("model does not support deviance residuals")
        martingale = [float(value) for value in martingale_method()]
        status = [float(value) for value in fit.status]
        if use_weights:
            martingale = _weight_residual_result(
                martingale,
                _model_residual_weights(fit, len(martingale)),
            )
        if not _collapse_is_false(collapse):
            martingale = _collapse_residual_result(martingale, collapse, len(martingale))
            status = _collapse_residual_result(status, collapse, len(status))
        return _cox_deviance_from_martingale(martingale, status)

    if residual_type == "partial" and (
        terms is not None or use_weights or not _collapse_is_false(collapse)
    ):
        result = _cox_partial_residuals(
            fit,
            terms,
            _model_residual_weights(fit, len(fit.status)) if use_weights else None,
        )
        if not _collapse_is_false(collapse):
            result = _collapse_residual_result(result, collapse, len(result))
        return result

    method_name = method_names[residual_type]
    method = getattr(fit, method_name, None)
    if method is None:
        raise TypeError(f"model does not support {residual_type} residuals")
    result = method()
    if use_weights:
        result = _weight_residual_result(result, _model_residual_weights(fit, len(result)))
    if not _collapse_is_false(collapse):
        result = _collapse_residual_result(result, collapse, len(result))
    return result


def anova(*fits: Any, test: str | None = "Chisq") -> Any:
    """Analysis of deviance for one or more fitted Cox models."""

    if not fits:
        raise TypeError("anova requires at least one fitted model")
    if len(fits) == 1 and isinstance(fits[0], list | tuple):
        fits = tuple(fits[0])
        if not fits:
            raise TypeError("anova requires at least one fitted model")

    test_name, with_tests = _cox_anova_test(test)
    if len(fits) == 1:
        return _anova_single_coxph(fits[0], test_name, with_tests)
    return _anova_multiple_coxph(fits, test_name, with_tests)


def coxph_detail(
    fit: Any | None = None,
    riskmat: bool = False,
    rorder: str = "data",
    *,
    time: Any | None = None,
    status: Any | None = None,
    covariates: Any | None = None,
    coefficients: Any | None = None,
    weights: Any | None = None,
) -> Any:
    """Return event-time Cox model details, like R's coxph.detail."""

    include_riskmat = _normalize_bool_option(riskmat, "riskmat")
    raw_args = (time, status, covariates, coefficients)
    if any(value is not None for value in raw_args):
        if fit is not None:
            raise ValueError("use either a fitted Cox model or raw Cox detail arrays")
        if not all(value is not None for value in raw_args):
            raise ValueError("time, status, covariates, and coefficients are required")
        rows = _as_matrix_rows(covariates, "covariates", allow_empty_columns=True)
        return _core.coxph_detail(
            _float_vector(time, "time"),
            _event_vector(status, "status"),
            rows,
            _float_vector(coefficients, "coefficients"),
            _optional_float_vector(weights, "weights", len(rows)) if weights is not None else None,
        )

    if fit is None:
        raise TypeError("coxph_detail requires a fitted Cox model")
    if not _is_coxph_fit(fit):
        raise TypeError("coxph_detail requires a fitted Cox model")
    rorder_name = _cox_detail_rorder(rorder)
    model = _unwrap_formula_fit(fit)
    method = _cox_detail_method(model)
    beta = _cox_beta(model)
    nvar = len(beta)
    rows = _cox_training_rows(model, nvar)
    time = [float(value) for value in model.event_times]
    status = [int(value) for value in model.status]
    n = len(time)
    if len(status) != n or len(rows) != n:
        raise ValueError("fitted Cox model detail arrays have inconsistent lengths")

    entry_values = getattr(model, "entry_times", None)
    entry = [float(value) for value in entry_values] if entry_values is not None else None
    if entry is not None and len(entry) != n:
        raise ValueError("fitted Cox model entry times do not match event rows")
    weights = _model_residual_weights(model, n)
    strata = _cox_training_strata(model, n)
    linear_predictors = [float(value) for value in model.linear_predictors]
    if len(linear_predictors) != n:
        raise ValueError("fitted Cox model linear predictors do not match event rows")

    center = _cox_reference_center(model, "sample")
    offset = _cox_fit_offset(model, beta)
    event_groups = _cox_detail_event_times(time, status, strata)
    detail = _core.coxph_detail(
        time,
        status,
        rows,
        beta,
        weights,
        entry_times=entry,
        strata=strata,
        offset=offset,
        method=method,
        center=center,
    )
    detail_rows = list(detail.rows)
    risk_matrix_columns: list[list[int]] = []

    for stratum, event_time in event_groups:
        at_risk = _cox_detail_at_risk(time, entry, strata, stratum, event_time)
        if include_riskmat:
            at_risk_set = set(at_risk)
            risk_matrix_columns.append([1 if idx in at_risk_set else 0 for idx in range(n)])

    row_order = _cox_detail_row_order(time, status, strata, rorder_name)
    x_rows = [rows[idx] for idx in row_order]
    y_rows = _cox_detail_y(time, status, entry)
    y_rows = [y_rows[idx] for idx in row_order]
    ordered_weights = [weights[idx] for idx in row_order]
    risk_matrix = None
    sortorder = row_order if rorder_name == "time" else None
    if include_riskmat:
        risk_matrix = [[column[idx] for column in risk_matrix_columns] for idx in row_order]

    has_case_weights = any(abs(weight - 1.0) > 1e-12 for weight in weights)
    return CoxPHDetailResult(
        time=[float(row.time) for row in detail_rows],
        nevent=[int(row.n_event) for row in detail_rows],
        nrisk=[int(row.n_risk) for row in detail_rows],
        means=[[float(value) for value in row.means] for row in detail_rows],
        score=[[float(value) for value in row.score] for row in detail_rows],
        imat=[
            [[float(value) for value in matrix_row] for matrix_row in row.imat]
            for row in detail_rows
        ],
        hazard=[float(row.hazard) for row in detail_rows],
        varhaz=[float(row.varhaz) for row in detail_rows],
        wtrisk=[float(row.wtrisk) for row in detail_rows],
        x=x_rows,
        y=y_rows,
        strata=_cox_detail_strata_table(strata, event_groups),
        riskmat=risk_matrix,
        weights=ordered_weights if has_case_weights else None,
        nevent_wt=[float(row.n_event_weight) for row in detail_rows] if has_case_weights else None,
        nrisk_wt=[float(row.wtrisk) for row in detail_rows] if has_case_weights else None,
        sortorder=sortorder,
    )


def coxph(
    response: Surv | str,
    data: Any | None = None,
    *,
    x: Any | None = None,
    weights: Any | None = None,
    offset: Any | None = None,
    strata: Any | None = None,
    cluster: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    init: Any | None = None,
    initial_beta: Any | None = None,
    max_iter: int = 20,
    eps: float | None = None,
    toler: float | None = None,
    method: str | None = None,
    ties: str | None = None,
    robust: Any | None = None,
    model: Any = False,
    y: Any = True,
    tt: Any | None = None,
    id: Any | None = None,  # noqa: A002
    istate: Any | None = None,
    statedata: Any | None = None,
    singular_ok: Any = True,
    nocenter: Any = (-1, 0, 1),
    control: Any | None = None,
    **kwargs: Any,
):
    """Fit a Cox proportional hazards model from Surv plus covariates."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    singular_ok = _pop_dotted_keyword(
        kwargs,
        "singular.ok",
        "singular_ok",
        singular_ok,
        True,
    )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"coxph got unexpected keyword argument(s): {unexpected}")

    method_name = _cox_tie_method(method, ties)
    robust_value = _normalize_optional_bool_option(robust, "robust")
    explicit_weights = weights is not None
    keep_model = _normalize_bool_option_with_default(model, "model", False)
    keep_y = _normalize_bool_option_with_default(y, "y", True)
    singular_ok_value = _normalize_bool_option_with_default(singular_ok, "singular_ok", True)
    nocenter_values = _normalize_numeric_sequence_or_none(nocenter, "nocenter")
    id_arg = id
    if istate is not None or statedata is not None:
        raise NotImplementedError("coxph multi-state istate/statedata inputs are not supported")
    if init is not None and initial_beta is not None:
        raise ValueError("use only one of init or initial_beta")
    max_iter = _integer_scalar(max_iter, "max_iter")
    max_iter, eps, toler, fix_time = _apply_coxph_control(control, max_iter, eps, toler)

    formula_design: _FormulaDesign | None = None
    formula_string: str | None = None
    formula_x_matrix: list[list[float]] | None = None
    formula_model_data: Any | None = None
    formula_cluster_columns: tuple[str, ...] = ()
    if isinstance(response, str):
        formula_string = response
        response_spec = _formula_response_spec(response)
        if _formula_has_time_transform_term(response):
            raise NotImplementedError("coxph tt time-transform terms are not supported")
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                response,
                data,
                subset,
                weights=weights,
                offset=offset,
                strata=strata,
                cluster=cluster,
                id=id_arg,
            )
            weights = aligned["weights"]
            offset = aligned["offset"]
            strata = aligned["strata"]
            cluster = aligned["cluster"]
            id_arg = aligned["id"]
            subset = None
        data, aligned = _apply_formula_na_action(
            response,
            data,
            na_action,
            weights=weights,
            offset=offset,
            strata=strata,
            cluster=cluster,
            id=id_arg,
        )
        weights = aligned["weights"]
        offset = aligned["offset"]
        strata = aligned["strata"]
        cluster = aligned["cluster"]
        id_arg = aligned["id"]
        na_action = "pass"
        formula_x = False
        if x is not None:
            if not _is_bool_like(x):
                raise TypeError("x must be True or False for coxph formula input")
            formula_x = _normalize_bool_option(x, "x")
        response, terms = _parse_formula(response, data)
        if terms.strata:
            if strata is not None:
                raise ValueError("use only one of formula strata(...) or strata")
            strata = _combined_columns(data, terms.strata, len(response))
        if terms.offsets:
            if offset is not None:
                raise ValueError("use only one of formula offset(...) or offset")
            offset = _offset_vector(data, terms.offsets, len(response))
        if terms.clusters:
            if cluster is not None:
                raise ValueError("use only one of formula cluster(...) or cluster")
            cluster = _combined_columns(data, terms.clusters, len(response))
            formula_cluster_columns = tuple(terms.clusters)
        formula_design = _fit_formula_design(data, response_spec, terms, len(response))
        x = _design_rows_from_spec(data, formula_design, len(response))
        formula_x_matrix = [list(row) for row in x] if formula_x else None
        formula_model_data = data

    if not isinstance(response, Surv):
        raise TypeError("coxph response must be a Surv object or formula")
    if subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        x = _subset_optional_sequence(x, indices, "x")
        weights = _subset_optional_sequence(weights, indices, "weights")
        offset = _subset_optional_sequence(offset, indices, "offset")
        strata = _subset_optional_sequence(strata, indices, "strata")
        cluster = _subset_optional_sequence(cluster, indices, "cluster")
        id_arg = _subset_optional_sequence(id_arg, indices, "id")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "coxph inputs",
        x=x,
        weights=weights,
        offset=offset,
        strata=strata,
        cluster=cluster,
        id=id_arg,
    )
    x = aligned["x"]
    weights = aligned["weights"]
    offset = aligned["offset"]
    strata = aligned["strata"]
    cluster = aligned["cluster"]
    id_arg = aligned["id"]
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "coxph currently supports right-censored and counting Surv responses"
        )

    rows = _as_matrix_rows(x, "x", allow_empty_columns=True)
    if len(rows) != len(response):
        raise ValueError("x must have the same number of rows as the Surv response")

    n = len(response)
    id_values = _materialize_labels(id_arg, "id") if id_arg is not None else None
    if id_values is not None and len(id_values) != n:
        raise ValueError("id must have the same length as the Surv response")
    fit_strata = _encode_groups(strata, n) if strata is not None else None
    fit_weights = _optional_float_vector(weights, "weights", n)
    case_weights = fit_weights if explicit_weights else None
    fit_offset = _optional_float_vector(offset, "offset", n)
    if not singular_ok_value:
        _check_cox_design_full_rank(rows, fit_weights, toler if toler is not None else 1e-9)
    model_frame = None
    if keep_model:
        model_frame = (
            _formula_model_frame(
                formula_model_data,
                response,
                formula_design,
                extra_columns=formula_cluster_columns,
                weights=weights,
                offset=offset,
                strata=strata,
                cluster=cluster,
                id=id_values,
            )
            if formula_design is not None
            else _matrix_model_frame(
                response,
                rows,
                weights=weights,
                offset=offset,
                strata=strata,
                cluster=cluster,
                id=id_values,
            )
        )

    fit_times = list(response.time)
    entry_times = list(response.start) if response.start is not None else None
    if fix_time:
        if entry_times is None:
            fit_times = _survdiff_timefix_values(fit_times, True)
        else:
            entry_times, fit_times = _timefix_vectors(entry_times, fit_times)
    fit = _core.coxph_fit(
        fit_times,
        list(response.event),
        rows,
        strata=fit_strata,
        weights=fit_weights,
        offset=fit_offset,
        initial_beta=(
            _float_vector(initial_beta if initial_beta is not None else init, "init")
            if init is not None or initial_beta is not None
            else None
        ),
        max_iter=max_iter,
        eps=eps,
        toler=toler,
        method=method_name,
        entry_times=entry_times,
        nocenter=nocenter_values,
    )
    robust_cluster = cluster if cluster is not None else id_values
    if robust_value is True and robust_cluster is None:
        robust_cluster = list(range(n))
    if robust_value is False and robust_cluster is not None:
        raise ValueError("cluster or id cannot be combined with robust=False")
    robust_variance = None
    naive_variance = None
    cluster_values = None
    if robust_cluster is not None:
        robust_variance, naive_variance, cluster_values = _cox_robust_variance_matrix(
            fit,
            robust_cluster,
        )
    if (
        formula_design is not None
        or case_weights is not None
        or robust_variance is not None
        or model_frame is not None
    ):
        return _FormulaFit(
            fit,
            formula_design,
            formula=formula_string,
            case_weights=case_weights,
            robust_variance=robust_variance,
            naive_variance=naive_variance,
            cluster=cluster_values,
            id_values=id_values,
            x_matrix=formula_x_matrix,
            y_response=response if formula_design is not None and keep_y else None,
            model_frame=model_frame,
        )
    return fit


def survreg(
    response: Surv | str | None = None,
    data: Any | None = None,
    *,
    x: Any | None = None,
    time: Any | None = None,
    time2: Any | None = None,
    status: Any | None = None,
    covariates: Any | None = None,
    weights: Any | None = None,
    offset: Any | None = None,
    offsets: Any | None = None,
    init: Any | None = None,
    initial: Any | None = None,
    initial_beta: Any | None = None,
    strata: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    dist: str | None = None,
    distribution: str | None = None,
    scale: Any = 0.0,
    parms: Any | None = None,
    model: Any = False,
    y: Any = True,
    robust: Any | None = None,
    cluster: Any | None = None,
    score: Any = False,
    max_iter: int | None = None,
    eps: float | None = None,
    tol_chol: float | None = None,
    control: Any | None = None,
    **kwargs: Any,
):
    """Fit an accelerated failure-time model using R-style or matrix inputs."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survreg got unexpected keyword argument(s): {unexpected}")

    if offset is not None and offsets is not None:
        raise ValueError("use only one of offset or offsets")
    initial_options = {
        "init": init,
        "initial": initial,
        "initial_beta": initial_beta,
    }
    if sum(value is not None for value in initial_options.values()) > 1:
        raise ValueError("use only one of init, initial, or initial_beta")
    if x is not None and covariates is not None:
        raise ValueError("use only one of x or covariates")
    scale_value = _finite_float(scale, "scale")
    if scale_value < 0.0:
        raise ValueError("scale must be non-negative")
    if parms is not None:
        raise NotImplementedError("survreg distribution parameters via parms are not supported")
    keep_model = _normalize_bool_option_with_default(model, "model", False)
    keep_y = _normalize_bool_option_with_default(y, "y", True)
    keep_score = _normalize_bool_option_with_default(score, "score", False)
    explicit_weights = weights is not None
    robust_requested = None if robust is None else _normalize_bool_option(robust, "robust")
    if max_iter is not None:
        max_iter = _integer_scalar(max_iter, "max_iter")
    max_iter, eps, tol_chol = _apply_survreg_control(control, max_iter, eps, tol_chol)

    formula_rows: list[list[float]] | None = None
    formula_design: _FormulaDesign | None = None
    formula_string: str | None = None
    formula_x_matrix: list[list[float]] | None = None
    formula_model_data: Any | None = None
    formula_cluster_columns: tuple[str, ...] = ()
    matrix_input = response is None and time is not None and status is not None
    if matrix_input:
        response_time = _float_vector(time, "time")
        response_time2 = _float_vector(time2, "time2") if time2 is not None else None
        response_status = _materialize_1d(status, "status")
        rows = _as_rows(covariates if covariates is not None else x, "covariates")
        if subset is not None:
            indices = _subset_indices(subset, len(response_time))
            response_time = [response_time[idx] for idx in indices]
            response_time2 = (
                [response_time2[idx] for idx in indices] if response_time2 is not None else None
            )
            response_status = [response_status[idx] for idx in indices]
            rows = [rows[idx] for idx in indices]
            weights = _subset_optional_sequence(weights, indices, "weights")
            offset = _subset_optional_sequence(offset, indices, "offset")
            offsets = _subset_optional_sequence(offsets, indices, "offsets")
            strata = _subset_optional_sequence(strata, indices, "strata")
            cluster = _subset_optional_sequence(cluster, indices, "cluster")
            subset = None
        keep = _keep_rows_after_na_action(
            _missing_row_indices(
                [
                    ("time", response_time),
                    *([("time2", response_time2)] if response_time2 is not None else []),
                    ("status", response_status),
                    ("covariates", rows),
                    *(
                        (name, values)
                        for name, values in (
                            ("weights", weights),
                            ("offset", offset),
                            ("offsets", offsets),
                            ("strata", strata),
                            ("cluster", cluster),
                        )
                        if values is not None
                    ),
                ],
                len(response_time),
            ),
            len(response_time),
            na_action,
            "survreg inputs",
        )
        if keep is not None:
            response_time = [response_time[idx] for idx in keep]
            response_time2 = (
                [response_time2[idx] for idx in keep] if response_time2 is not None else None
            )
            response_status = [response_status[idx] for idx in keep]
            rows = [rows[idx] for idx in keep]
            weights = _subset_optional_sequence(weights, keep, "weights")
            offset = _subset_optional_sequence(offset, keep, "offset")
            offsets = _subset_optional_sequence(offsets, keep, "offsets")
            strata = _subset_optional_sequence(strata, keep, "strata")
            cluster = _subset_optional_sequence(cluster, keep, "cluster")
        response_status = [
            float(value)
            for value in _integer_code_vector(
                response_status,
                "status",
                "0/1/2/3 censoring codes",
            )
        ]
        distribution_name = distribution or dist or "weibull"
    else:
        if time2 is not None:
            raise ValueError("time2 is only supported with matrix time/status input")
        if isinstance(response, str):
            formula_string = response
            response_spec = _formula_response_spec(response)
            if subset is not None:
                data, aligned = _subset_formula_inputs(
                    response,
                    data,
                    subset,
                    weights=weights,
                    offset=offset,
                    offsets=offsets,
                    strata=strata,
                    cluster=cluster,
                )
                weights = aligned["weights"]
                offset = aligned["offset"]
                offsets = aligned["offsets"]
                strata = aligned["strata"]
                cluster = aligned["cluster"]
                subset = None
            data, aligned = _apply_formula_na_action(
                response,
                data,
                na_action,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
            weights = aligned["weights"]
            offset = aligned["offset"]
            offsets = aligned["offsets"]
            strata = aligned["strata"]
            cluster = aligned["cluster"]
            na_action = "pass"
            formula_x = False
            if x is not None:
                if not _is_bool_like(x):
                    raise TypeError("x must be True or False for survreg formula input")
                formula_x = _normalize_bool_option(x, "x")
            if covariates is not None:
                raise ValueError("survreg formula input cannot be combined with x or covariates")
            response, terms = _parse_formula(response, data)
            formula_design = _fit_formula_design(
                data,
                response_spec,
                terms,
                len(response),
                include_intercept=True,
            )
            formula_rows = (
                _design_rows_from_spec(data, formula_design, len(response))
                if terms.covariates or formula_design.intercept
                else [[] for _ in range(len(response))]
            )
            formula_x_matrix = [list(row) for row in formula_rows] if formula_x else None
            if terms.strata:
                if strata is not None:
                    raise ValueError("use only one of formula strata(...) or strata")
                strata = _combined_columns(data, terms.strata, len(response))
            if terms.offsets:
                if offset is not None or offsets is not None:
                    raise ValueError("use only one of formula offset(...) or offset")
                offsets = _offset_vector(data, terms.offsets, len(response))
            if terms.clusters:
                if cluster is not None:
                    raise ValueError("use only one of formula cluster(...) or cluster")
                cluster = _combined_columns(data, terms.clusters, len(response))
                formula_cluster_columns = tuple(terms.clusters)
            formula_model_data = data

        if not isinstance(response, Surv):
            raise TypeError("survreg response must be a Surv object, formula, or time/status input")
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            x = _subset_optional_sequence(x, indices, "x")
            covariates = _subset_optional_sequence(covariates, indices, "covariates")
            weights = _subset_optional_sequence(weights, indices, "weights")
            offset = _subset_optional_sequence(offset, indices, "offset")
            offsets = _subset_optional_sequence(offsets, indices, "offsets")
            strata = _subset_optional_sequence(strata, indices, "strata")
            cluster = _subset_optional_sequence(cluster, indices, "cluster")
        response, aligned = _apply_surv_na_action(
            response,
            na_action,
            "survreg inputs",
            x=x,
            covariates=covariates,
            weights=weights,
            offset=offset,
            offsets=offsets,
            strata=strata,
            cluster=cluster,
        )
        x = aligned["x"]
        covariates = aligned["covariates"]
        weights = aligned["weights"]
        offset = aligned["offset"]
        offsets = aligned["offsets"]
        strata = aligned["strata"]
        cluster = aligned["cluster"]

        response_time, response_status, response_time2 = _survreg_response_arrays(response)
        rows = (
            formula_rows
            if formula_rows is not None
            else _as_rows(covariates if covariates is not None else x, "x")
        )
        distribution_name = distribution or dist or "weibull"

    n = len(response_time)
    if len(response_status) != n:
        raise ValueError("status must have the same length as time")
    if rows and len(rows) != n:
        raise ValueError("covariates must have the same number of rows as the Surv response")

    normalized_distribution = _normalize_survreg_distribution(distribution_name)
    if normalized_distribution is None:
        normalized_distribution = "weibull"
    distribution_name = normalized_distribution
    offset_values = _optional_float_vector(offsets if offsets is not None else offset, "offsets", n)
    weight_values = _optional_float_vector(weights, "weights", n)
    case_weights = weight_values if explicit_weights else None
    strata_values = _encode_groups(strata, n) if strata is not None else None
    if scale_value > 0.0 and strata_values is not None and len(set(strata_values)) > 1:
        raise ValueError("cannot have both a fixed scale and strata")
    cluster_values_for_validation = (
        _materialize_labels(cluster, "cluster") if cluster is not None else None
    )
    if cluster_values_for_validation is not None:
        if len(cluster_values_for_validation) != n:
            raise ValueError("cluster must have the same length as the Surv response")
        _label_levels(cluster_values_for_validation, "cluster")
    robust_value = (
        cluster_values_for_validation is not None if robust_requested is None else robust_requested
    )
    model_frame = None
    if keep_model:
        if formula_design is not None:
            model_frame = _formula_model_frame(
                formula_model_data,
                response,
                formula_design,
                extra_columns=formula_cluster_columns,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
        elif isinstance(response, Surv):
            model_frame = _matrix_model_frame(
                response,
                rows,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
        else:
            model_frame = _survreg_matrix_model_frame(
                response_time,
                response_status,
                response_time2,
                rows,
                weights=weights,
                offset=offset,
                offsets=offsets,
                strata=strata,
                cluster=cluster,
            )
    initial_name, initial_source = next(
        ((name, value) for name, value in initial_options.items() if value is not None),
        ("initial", None),
    )
    initial_values = (
        _float_vector(initial_source, initial_name) if initial_source is not None else None
    )

    fit = _core.survreg(
        response_time,
        response_status,
        rows,
        weights=weight_values,
        offsets=offset_values,
        initial_beta=initial_values,
        strata=strata_values,
        distribution=distribution_name,
        max_iter=max_iter,
        eps=eps,
        tol_chol=tol_chol,
        time2=response_time2,
        fixed_scale=scale_value if scale_value > 0.0 else None,
    )
    robust_cluster = cluster_values_for_validation
    if robust_value and robust_cluster is None:
        robust_cluster = list(range(n))
    robust_variance = None
    naive_variance = None
    cluster_values = None
    if robust_cluster is not None and robust_value:
        robust_variance, naive_variance, cluster_values = _survreg_robust_variance_matrix(
            fit,
            robust_cluster,
        )
    score_values = list(fit.score_vector) if keep_score else None
    return (
        _FormulaFit(
            fit,
            formula_design,
            formula=formula_string,
            case_weights=case_weights,
            robust_variance=robust_variance,
            naive_variance=naive_variance,
            cluster=cluster_values,
            x_matrix=formula_x_matrix,
            y_response=response if keep_y else None,
            model_frame=model_frame,
            score_values=score_values,
        )
        if (
            formula_design is not None
            or case_weights is not None
            or robust_variance is not None
            or model_frame is not None
            or score_values is not None
        )
        else fit
    )
