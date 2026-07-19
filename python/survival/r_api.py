from __future__ import annotations

import math
import random
import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as _Date  # noqa: N812
from datetime import datetime as _DateTime  # noqa: N812
from functools import lru_cache
from itertools import combinations, product
from operator import index
from statistics import NormalDist
from typing import Any, NoReturn

from . import _survival as _core

_COX_NONCONVERGENCE_FLAG = 1000

__all__ = [
    "Surv",
    "Surv2",
    "Surv2data",
    "CoxSurvfitResult",
    "CoxBaseHazardResult",
    "CoxPHDetailResult",
    "CoxPHWTestResult",
    "CoxZPHResult",
    "FineGrayOutput",
    "PredictResult",
    "PyearsResult",
    "RateTable",
    "StrataFactor",
    "SurvObrienResult",
    "SurvExpResult",
    "SurvfitResult",
    "SurvfitConfidenceIntervalResult",
    "TcutResult",
    "aic",
    "aeqSurv",
    "as_data_frame",
    "basehaz",
    "anova",
    "aggregate_survfit_result",
    "brier",
    "bcloglog",
    "coef",
    "coef_names",
    "confint",
    "concordance",
    "coxph",
    "coxph_detail",
    "coxph_wtest",
    "cox_zph",
    "dsurvreg",
    "df_residual",
    "degrees_freedom",
    "bic",
    "blog",
    "blogit",
    "bprobit",
    "cipoisson",
    "extract_aic",
    "fitted",
    "finegray",
    "fromtimeline",
    "format_surv",
    "is_surv",
    "is_na_surv",
    "is_ratetable",
    "loglik",
    "lvcf",
    "model_formula",
    "model_summary",
    "model_frame",
    "model_matrix",
    "model_weights",
    "neardate",
    "nobs",
    "nostutter",
    "nsk",
    "pyears",
    "pspline",
    "pseudo",
    "predict",
    "psurvreg",
    "qsurvreg",
    "ratetableDate",
    "residuals",
    "royston",
    "rsurvreg",
    "rttright",
    "statefig",
    "strata",
    "survdiff",
    "survConcordance",
    "survConcordance_fit",
    "survcondense",
    "survcheck",
    "survexp",
    "survexp_individual",
    "survexp_mn",
    "survobrien",
    "survexp_us",
    "survexp_usr",
    "survfit",
    "survfit0",
    "survfit_confint",
    "survfit_residuals",
    "survfitkm_counting_influence",
    "survfitkm_influence",
    "survSplit",
    "survreg",
    "tcut",
    "totimeline",
    "yates",
    "yates_contrast",
    "yates_pairwise",
    "vcov",
    "YatesPairwiseResult",
    "YatesResult",
]

_EXP_CLAMP_MIN = -745.0
_EXP_CLAMP_MAX = 709.0
_SURVFIT_TIME_EPSILON = 1e-9
_VARIANCE_SCALE_FLOOR = 1e-12
_COX_DFBETAS_SCALE_FLOOR = 1e-10
_SURV_TYPES = ("right", "left", "interval", "counting", "interval2")

FineGrayOutput = _core.FineGrayOutput
RateTable = _core.RateTable
SurvObrienResult = _core.SurvObrienResult
TcutResult = _core.TcutResult
YatesPairwiseResult = _core.YatesPairwiseResult
YatesResult = _core.YatesResult


class _MissingArgument:
    __slots__ = ()

    def __repr__(self) -> str:
        return "..."


_MISSING = _MissingArgument()


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
class _ModelCovariateTerm:
    term: _CovariateSpec


@dataclass(frozen=True)
class _ModelStrataTerm:
    columns: tuple[str, ...]


@dataclass(frozen=True)
class _ModelOffsetTerm:
    term: _CovariateTerm


@dataclass(frozen=True)
class _ModelClusterTerm:
    column: str


_FormulaModelTerm = _ModelCovariateTerm | _ModelStrataTerm | _ModelOffsetTerm | _ModelClusterTerm


@dataclass(frozen=True)
class _FormulaTerms:
    covariates: list[_CovariateSpec]
    strata: list[str]
    offsets: list[_CovariateTerm]
    clusters: list[str]
    model_terms: list[_FormulaModelTerm] = field(default_factory=list)
    intercept: bool = True


@dataclass(frozen=True)
class _CachedFormulaTerms:
    covariates: tuple[_CovariateSpec, ...]
    strata: tuple[str, ...]
    offsets: tuple[_CovariateTerm, ...]
    clusters: tuple[str, ...]
    model_terms: tuple[_FormulaModelTerm, ...] = ()
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
    coefficient_names: tuple[str, ...] | None = None
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
    tied_x: float | list[float] = 0.0
    tied_y: float | list[float] = 0.0
    tied_xy: float | list[float] = 0.0
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
class StrataFactor:
    codes: list[int | None]
    levels: list[str]
    labels: list[str | None]
    counts: list[int]

    def __iter__(self):
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.codes)


@dataclass(frozen=True)
class SurvExpResult:
    time: list[float]
    surv: list[float]
    n_risk: list[float]
    cumhaz: list[float]
    method: str
    n: int


@dataclass(frozen=True)
class PyearsResult:
    pyears: list[float]
    n: list[float]
    offtable: float
    group: list[str]
    observations: int
    event: list[float] | None = None
    expected: list[float] | None = None
    tcut: bool = False


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
class CoxPHWTestResult:
    test: list[float]
    df: int
    solve: list[float] | list[list[float]] | float


@dataclass(frozen=True)
class CoxBaseHazardResult:
    time: list[float]
    cumhaz: list[float] | list[list[float]]
    strata: list[int] | None = None
    centered: bool = True
    curve_strata: list[int] | None = None
    strata_labels: list[Any] | None = None
    curve_strata_labels: list[Any] | None = None

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
    strata_labels: list[Any] | None = None
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
    n_risk_count: list[float] | None = None
    n_event_count: list[float] | None = None
    n_censor_count: list[float] | None = None
    n_enter_count: list[float] | None = None
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
class SurvfitConfidenceIntervalResult:
    lower: list[float]
    upper: list[float]

    def __iter__(self):
        yield self.lower
        yield self.upper


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
class _PseudoMatrixResult:
    pseudo: list[list[float]]
    time: list[float]


@dataclass(frozen=True)
class _SurvfitComputation:
    stype: int
    ctype: int

    @property
    def is_kaplan_meier(self) -> bool:
        return self.stype == 1 and self.ctype == 1


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


def _statefig_layout_matrix(layout: Any) -> tuple[list[list[float]], bool]:
    rows = _coerce_array_like(layout, "layout")
    if not rows:
        raise ValueError("layout must not be empty")
    if isinstance(rows[0], list | tuple):
        width = len(rows[0])
        matrix = [[_finite_float(value, "layout") for value in row] for row in rows]
        if any(len(row) != width for row in matrix):
            raise ValueError("layout must be rectangular")
        return matrix, True
    return [[_finite_float(value, "layout") for value in rows]], False


def _statefig_space(n: int) -> list[float]:
    return [(idx + 0.5) / n for idx in range(n)]


def _statefig_positions_from_layout(
    layout: Any,
    n_states: int,
) -> tuple[list[list[float]], list[int]]:
    matrix, is_matrix = _statefig_layout_matrix(layout)
    n_row = len(matrix)
    n_col = len(matrix[0]) if matrix else 0

    if is_matrix and n_col == 2 and n_row > 1:
        if n_row != n_states:
            raise ValueError("layout matrix should have one row per state")
        positions = []
        for row in matrix:
            x, y = row
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                raise ValueError("layout coordinates must be between 0 and 1")
            positions.append([x, y])
        return positions, [n_states]

    values = [value for row in matrix for value in row]
    layout_counts = []
    for value in values:
        if value <= 0.0 or not float(value).is_integer():
            raise ValueError("non-integer number of states in layout argument")
        layout_counts.append(int(value))
    if sum(layout_counts) != n_states:
        raise ValueError("number of boxes != number of states")

    positions = [[0.0, 0.0] for _ in range(n_states)]
    group_space = _statefig_space(len(layout_counts))
    state_idx = 0
    column_layout = (not is_matrix) or n_col > 1
    for group_idx, count in enumerate(layout_counts):
        within = _statefig_space(count)
        for offset in range(count):
            if column_layout:
                positions[state_idx] = [group_space[group_idx], 1.0 - within[offset]]
            else:
                positions[state_idx] = [within[offset], 1.0 - group_space[group_idx]]
            state_idx += 1
    return positions, layout_counts


def statefig(
    layout: Any,
    connect: Any,
    states: Any | None = None,
    *,
    margin: Any = 0.03,
    box: Any = True,
    cex: Any = 1,
    col: Any = 1,
    lwd: Any = 1,
    lty: Any = 1,
    bcol: Any | None = None,
    acol: Any | None = None,
    alwd: Any | None = None,
    alty: Any | None = None,
    offset: Any = 0,
) -> dict[str, Any]:
    """Return R ``survival::statefig`` state coordinates from layout/connect inputs."""

    del cex, col, lwd, lty, bcol, acol, alwd, alty
    _finite_float(margin, "margin")
    _finite_float(offset, "offset")
    _normalize_bool_option(box, "box")

    connect_rows = _as_rows(connect, "connect")
    n_states = len(connect_rows)
    if n_states == 0 or any(len(row) != n_states for row in connect_rows):
        raise ValueError("connect must be a square matrix")
    state_names = (
        [str(value) for value in _materialize_1d(states, "states")]
        if states is not None
        else [str(idx + 1) for idx in range(n_states)]
    )
    if len(state_names) != n_states:
        raise ValueError("states must have one entry per connect row")

    positions, layout_counts = _statefig_positions_from_layout(layout, n_states)
    edges = [
        [row_idx, col_idx, int(value)]
        for row_idx, row in enumerate(connect_rows)
        for col_idx, value in enumerate(row)
        if row_idx != col_idx and value != 0.0
    ]
    return {
        "states": state_names,
        "positions": positions,
        "layout": layout_counts,
        "edges": edges,
    }


def _brier_response_from_model_frame(frame: Mapping[str, Any]) -> Surv | None:
    for value in frame.values():
        if isinstance(value, Surv):
            return value
    return None


def _brier_fit_response_and_data(fit: Any, newdata: Any | None) -> tuple[Surv, Any | None]:
    if newdata is not None:
        if not isinstance(fit, _FormulaFit) or fit.formula is None:
            raise ValueError("newdata brier calculations require a formula Cox model")
        response, _terms = _parse_formula(fit.formula, newdata)
        return response, newdata

    if isinstance(fit, _FormulaFit):
        if fit.y_response is not None:
            return fit.y_response, fit.model_frame
        if fit.model_frame is not None:
            response = _brier_response_from_model_frame(fit.model_frame)
            if response is not None:
                return response, fit.model_frame
        raise ValueError("fitted Cox model does not retain its response; refit with y/model data")

    model = _unwrap_formula_fit(fit)
    if not hasattr(model, "event_times") or not hasattr(model, "status"):
        raise ValueError("fitted Cox model does not expose response data")
    return Surv(list(model.event_times), list(model.status)), None


def _brier_case_weights(fit: Any, n: int) -> list[float]:
    weights = _model_residual_weights(fit, n)
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return weights


def _brier_id_column(data: Any | None, name: str) -> list[Any] | None:
    if data is None:
        return None
    if isinstance(data, Mapping) and name not in data:
        return None
    try:
        return _materialize_labels(_column(data, name), name)
    except KeyError:
        return None


def _brier_id_values(fit: Any, model_data: Any | None, n: int) -> list[Any] | None:
    for name in ("(id)", "id"):
        values = _brier_id_column(model_data, name)
        if values is not None:
            if len(values) != n:
                raise ValueError("id must have the same length as the Surv response")
            return values

    if isinstance(fit, _FormulaFit) and fit.id_values is not None:
        values = _materialize_labels(fit.id_values, "id")
        if len(values) == n:
            return values
    return None


def _brier_counting_has_gaps_or_overlaps(
    starts: Sequence[float],
    stops: Sequence[float],
    id_values: Sequence[Any],
) -> bool:
    intervals_by_id: dict[Any, list[tuple[float, float]]] = {}
    for start, stop, id_value in zip(starts, stops, id_values, strict=True):
        intervals_by_id.setdefault(_hashable_group_value(id_value), []).append(
            (float(stop), float(start))
        )

    for intervals in intervals_by_id.values():
        previous_stop: float | None = None
        for stop, start in sorted(intervals):
            if start > stop:
                return True
            if previous_stop is not None and start != previous_stop:
                return True
            previous_stop = stop
    return False


def _brier_validate_counting_response(
    starts: Sequence[float],
    stops: Sequence[float],
    status: Sequence[int],
    id_values: Sequence[Any] | None,
) -> None:
    if id_values is None:
        raise ValueError("id is required for start-stop data")
    if len(id_values) != len(stops):
        raise ValueError("id must have the same length as the Surv response")
    if any(value not in (0, 1) for value in status):
        raise ValueError("response must be right censored")
    if _brier_counting_has_gaps_or_overlaps(starts, stops, id_values):
        raise ValueError("one or more flags are >0 in survcheck")
    if not _rttright_counting_common_start(starts, id_values):
        raise NotImplementedError("delayed entry is not yet implemented")


def _brier_event_times(
    response: Surv,
    timefix: bool,
    id_values: Sequence[Any] | None = None,
) -> tuple[list[float], list[int]]:
    if response.type not in {"right", "counting"}:
        raise ValueError("response must be right censored")
    times = [float(value) for value in response.time]
    status = [int(value) for value in response.event]
    if response.start is not None:
        starts = [float(value) for value in response.start]
        if any(not math.isfinite(value) for value in starts):
            raise ValueError("start times must be finite")
        if timefix:
            starts, times = _timefix_vectors(starts, times)
        _brier_validate_counting_response(starts, times, status, id_values)
    elif timefix:
        times = [float(value) for value in _core.aeq_surv(times, None).time]
    return times, status


def _brier_prediction_curves(
    fit: Any,
    prediction_data: Any | None,
) -> tuple[list[float], list[list[float]]]:
    if prediction_data is not None:
        cox_survfit = survfit(fit, newdata=prediction_data, se_fit=False)
        if not isinstance(cox_survfit, CoxSurvfitResult):
            raise TypeError("brier requires Cox survival curves")
        return cox_survfit.time, cox_survfit.surv

    model = _unwrap_formula_fit(fit)
    beta = _cox_beta(model)
    rows = _cox_training_rows(model, len(beta))
    return _cox_survival_curve(model, rows, None, True, None)


def _brier_default_times(response: Surv, weights: list[float], efron: bool) -> list[float]:
    baseline = survfit(
        response,
        weights=weights,
        se_fit=False,
        stype=2 if efron else 1,
        ctype=2 if efron else 1,
    )
    if not isinstance(baseline, SurvfitResult):
        raise TypeError("brier baseline curve must be a Kaplan-Meier survfit result")
    return [
        float(time)
        for time, event_count in zip(baseline.time, baseline.n_event, strict=True)
        if float(event_count) > 0.0
    ]


def _brier_censoring_survival(
    dtime: list[float],
    dstat: list[int],
    weights: list[float],
) -> SurvfitResult:
    censor_response = Surv(dtime, [1 - int(value) for value in dstat])
    censor_fit = survfit(censor_response, weights=weights, se_fit=False)
    censor_fit0 = survfit0(censor_fit)
    if not isinstance(censor_fit0, SurvfitResult):
        raise TypeError("brier censoring curve must be a Kaplan-Meier survfit result")
    return censor_fit0


def _brier_apply_ties(dtime: list[float], dstat: list[int], ties: bool) -> list[float]:
    if not ties:
        return list(dtime)
    unique_times = sorted(set(dtime))
    if len(unique_times) < 2:
        return list(dtime)
    mindiff = min(b - a for a, b in zip(unique_times[:-1], unique_times[1:], strict=True))
    return [
        time + mindiff / 2.0 if status == 0 else time
        for time, status in zip(dtime, dstat, strict=True)
    ]


def _brier_rsquared(model_brier: float, null_brier: float) -> float:
    if null_brier == 0.0:
        if model_brier == 0.0:
            return math.nan
        return math.inf if model_brier < 0.0 else -math.inf
    return 1.0 - model_brier / null_brier


def brier(
    fit: Any,
    times: Any | None = None,
    newdata: Any | None = None,
    ties: Any = True,
    detail: Any = False,
    timefix: Any = True,
    efron: Any = False,
) -> dict[str, Any]:
    """Compute R ``survival::brier`` IPCW Brier scores for Cox model fits."""

    if not _is_coxph_fit(fit):
        raise TypeError("fit must be a coxph object")
    ties_value = _normalize_bool_option(ties, "ties")
    detail_value = _normalize_bool_option(detail, "detail")
    timefix_value = _normalize_bool_option(timefix, "timefix")
    efron_value = _normalize_bool_option(efron, "efron")
    response, prediction_data = _brier_fit_response_and_data(fit, newdata)
    id_values = _brier_id_values(fit, prediction_data, len(response))
    dtime, dstat = _brier_event_times(response, timefix_value, id_values)
    n = len(dtime)
    weights = _brier_case_weights(fit, n)
    eval_times = (
        _float_vector(times, "times")
        if times is not None
        else _brier_default_times(
            response,
            weights,
            efron_value and getattr(_unwrap_formula_fit(fit), "method", None) == "efron",
        )
    )

    baseline = survfit(response, weights=weights, se_fit=False, stype=1)
    if not isinstance(baseline, SurvfitResult):
        raise TypeError("brier baseline curve must be a Kaplan-Meier survfit result")
    p0 = [1.0 - value for value in _step_curve_at(baseline.time, baseline.estimate, eval_times)]

    curve_times, curves = _brier_prediction_curves(fit, prediction_data)
    if len(curves) != n:
        raise ValueError("Cox survival predictions do not match response length")
    phat = [[0.0] * n for _ in eval_times]
    for row_idx, curve in enumerate(curves):
        survival = _step_curve_at(curve_times, [float(value) for value in curve], eval_times)
        for time_idx, value in enumerate(survival):
            phat[time_idx][row_idx] = 1.0 - value

    adjusted_time = _brier_apply_ties(dtime, dstat, ties_value)
    censor_fit = _brier_censoring_survival(adjusted_time, dstat, weights)
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]
    brier_values: list[float] = []
    rsquared_values: list[float] = []
    eff_n: list[float] = []

    for time_idx, eval_time in enumerate(eval_times):
        censor_survival = _step_curve_at(
            censor_fit.time,
            censor_fit.estimate,
            [min(time_value, eval_time) for time_value in adjusted_time],
        )
        null_numerator = 0.0
        model_numerator = 0.0
        denominator = 0.0
        weight_square_sum = 0.0
        null_prediction = p0[time_idx]
        model_predictions = phat[time_idx]

        for row_idx, (time_value, status, censor_value) in enumerate(
            zip(adjusted_time, dstat, censor_survival, strict=True)
        ):
            if time_value < eval_time and status == 0:
                weight = 0.0
            elif censor_value > 0.0:
                weight = normalized_weights[row_idx] / censor_value
            else:
                weight = math.inf

            if time_value > eval_time:
                null_loss = null_prediction * null_prediction
                model_prediction = model_predictions[row_idx]
                model_loss = model_prediction * model_prediction
            else:
                null_residual = status - null_prediction
                model_residual = status - model_predictions[row_idx]
                null_loss = null_residual * null_residual
                model_loss = model_residual * model_residual

            denominator += weight
            weight_square_sum += weight * weight
            null_numerator += weight * null_loss
            model_numerator += weight * model_loss

        eff_n.append(1.0 / weight_square_sum)
        null_brier = math.nan if denominator == 0.0 else null_numerator / denominator
        model_brier = math.nan if denominator == 0.0 else model_numerator / denominator
        brier_values.append(model_brier)
        rsquared_values.append(_brier_rsquared(model_brier, null_brier))

    result: dict[str, Any] = {
        "rsquared": rsquared_values,
        "brier": brier_values,
        "times": eval_times,
    }
    if detail_value:
        result["p0"] = p0
        result["phat"] = phat
        result["eff.n"] = eff_n
    return result


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
    for wrapper in ("I", "identity"):
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


def _response_arg_values(data: Any, part: str, inferred_length: int | None = None) -> list[Any]:
    part = _unwrap_response_identity(part)
    rep_call = _response_rep_call(part)
    if rep_call is not None:
        repeated_value, count_expression = rep_call
        return [repeated_value] * _response_rep_count(count_expression, inferred_length)

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
    inferred_length: int | None = None
    if spec.columns:
        inferred_length = len(_column(data, spec.columns[0]))
    return [_response_arg_values(data, argument, inferred_length) for argument in spec.arguments]


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


_FORMULA_RESPONSE_ARGUMENT_ALIASES = {
    "event": "event",
    "start": "time",
    "status": "event",
    "stop": "time2",
    "time": "time",
    "time1": "time",
    "time2": "time2",
}


def _formula_response_argument_name(name: str) -> str | None:
    return _FORMULA_RESPONSE_ARGUMENT_ALIASES.get(name.strip().lower())


def _ordered_named_response_arguments(named_arguments: dict[str, str]) -> list[str]:
    if "time" not in named_arguments:
        raise ValueError("named Surv(...) formula response requires time=")
    arguments = [named_arguments["time"]]
    if "time2" in named_arguments:
        arguments.append(named_arguments["time2"])
    if "event" in named_arguments:
        arguments.append(named_arguments["event"])
    return arguments


def _ordered_named_surv_arguments(named_arguments: Mapping[str, Any]) -> tuple[Any, ...]:
    if "time" not in named_arguments:
        raise ValueError("named Surv(...) requires time=, time1=, or start=")
    arguments = [named_arguments["time"]]
    if "time2" in named_arguments:
        arguments.append(named_arguments["time2"])
    if "event" in named_arguments:
        arguments.append(named_arguments["event"])
    return tuple(arguments)


def _collect_named_surv_arguments(arguments: Mapping[str, Any]) -> tuple[Any, ...] | None:
    named_arguments: dict[str, Any] = {}
    for option, value in arguments.items():
        if value is _MISSING:
            continue
        argument_name = _formula_response_argument_name(option)
        if argument_name is None:
            raise TypeError(f"Surv got an unexpected keyword argument {option!r}")
        if argument_name in named_arguments:
            raise ValueError(f"Surv(...) contains multiple {argument_name}= arguments")
        named_arguments[argument_name] = value
    if not named_arguments:
        return None
    return _ordered_named_surv_arguments(named_arguments)


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
    cluster: Any | None = None,
    cluster_column: str | None = None,
) -> dict[str, Any]:
    response_spec = _formula_response_spec(formula)
    frame: dict[str, Any] = {_surv_response_model_name(response_spec): response}
    for column in _formula_columns(formula, data):
        frame[column] = _column(data, column)
    if id_column is not None and id_column not in frame:
        frame[id_column] = _column(data, id_column)
    if cluster_column is not None and cluster_column not in frame:
        frame[cluster_column] = _column(data, cluster_column)
    if weights is not None:
        frame["(weights)"] = _materialize_1d(weights, "(weights)")
    if id is not None:
        frame["(id)"] = _materialize_1d(id, "(id)")
    if cluster is not None:
        frame["(cluster)"] = _materialize_1d(cluster, "(cluster)")
    return frame


def _survfit_model_frame(
    response: Surv,
    group: Any | None,
    weights: Any | None,
    id: Any | None = None,  # noqa: A002
    cluster: Any | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {"response": response}
    if group is not None:
        frame["group"] = _materialize_labels(group, "group")
    if weights is not None:
        frame["(weights)"] = _materialize_1d(weights, "(weights)")
    if id is not None:
        frame["(id)"] = _materialize_labels(id, "id")
    if cluster is not None:
        frame["(cluster)"] = _materialize_labels(cluster, "cluster")
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


def _strata_is_vector_sequence(value: Any) -> bool:
    if isinstance(value, str | bytes | Mapping):
        return False
    try:
        items = list(value)
    except TypeError:
        return False
    if not items or isinstance(items[0], str | bytes | Mapping):
        return False
    try:
        list(items[0])
    except TypeError:
        return False
    return True


def _strata_legacy_core_call(
    variables: tuple[Any, ...],
    na_group: bool,
    shortlabel: Any,
    sep: str,
    labels: Any,
) -> bool:
    if na_group or shortlabel is not None or sep != ", " or labels is not None:
        return False
    if len(variables) != 1 or not _strata_is_vector_sequence(variables[0]):
        return False
    try:
        columns = [list(column) for column in variables[0]]
    except TypeError:
        return False
    if not columns:
        return False
    for column in columns:
        for value in column:
            if isinstance(value, bool) or _is_missing_value(value):
                return False
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(numeric) or not numeric.is_integer():
                return False
    return True


def _strata_variables_from_args(variables: tuple[Any, ...]) -> list[list[Any]]:
    if not variables:
        raise ValueError("strata requires at least one variable")
    if len(variables) == 1 and _strata_is_vector_sequence(variables[0]):
        variables = tuple(variables[0])
    columns = [
        _materialize_1d(variable, f"variable {idx + 1}") for idx, variable in enumerate(variables)
    ]
    n = len(columns[0])
    if any(len(column) != n for column in columns):
        raise ValueError("all arguments must be the same length")
    return columns


def _strata_value_label(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _surv_format_number(value)
    return str(value)


def _strata_level_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, str(value))


def _strata_column_levels(column: Sequence[Any], na_group: bool) -> tuple[list[Any], list[str]]:
    values: dict[Any, None] = {}
    saw_missing = False
    for value in column:
        if _is_missing_value(value):
            saw_missing = True
            continue
        try:
            values.setdefault(value, None)
        except TypeError as exc:
            raise TypeError("strata variables must contain hashable values") from exc
    levels = sorted(values, key=_strata_level_sort_key)
    labels = [_strata_value_label(value) for value in levels]
    if na_group and saw_missing:
        levels.append(None)
        labels.append("NA")
    return levels, labels


def _strata_default_labels(n_terms: int) -> list[str]:
    return [f"v{idx + 1}" for idx in range(n_terms)]


def _strata_normalize_labels(labels: Any, n_terms: int) -> list[str]:
    if labels is None:
        return _strata_default_labels(n_terms)
    result = [str(label) for label in _materialize_1d(labels, "labels")]
    if len(result) != n_terms:
        raise ValueError("labels must have one entry per strata variable")
    return result


def _strata_default_shortlabel(columns: Sequence[Sequence[Any]], labels: Any) -> bool:
    if labels is not None:
        return False
    return all(
        all(_is_missing_value(value) or isinstance(value, str) for value in column)
        for column in columns
    )


def strata(
    *variables: Any,
    na_group: bool = False,
    shortlabel: bool | None = None,
    sep: str = ", ",
    labels: Any | None = None,
) -> Any:
    """Create R-style strata factor codes and labels."""

    if _strata_legacy_core_call(variables, na_group, shortlabel, sep, labels):
        return _core.strata([[int(value) for value in column] for column in variables[0]])
    if not isinstance(na_group, bool):
        raise TypeError("na_group must be True or False")
    if shortlabel is not None and not isinstance(shortlabel, bool):
        raise TypeError("shortlabel must be True, False, or None")
    if not isinstance(sep, str):
        raise TypeError("sep must be a string")

    columns = _strata_variables_from_args(variables)
    n = len(columns[0])
    term_labels = _strata_normalize_labels(labels, len(columns))
    short = _strata_default_shortlabel(columns, labels) if shortlabel is None else shortlabel

    column_levels: list[list[Any]] = []
    column_level_labels: list[list[str]] = []
    column_maps: list[dict[Any, int]] = []
    for column in columns:
        levels, level_labels = _strata_column_levels(column, na_group)
        column_levels.append(levels)
        column_level_labels.append(level_labels)
        column_maps.append({value: idx for idx, value in enumerate(levels)})

    raw_codes: list[int | None] = []
    raw_to_parts: dict[int, list[int]] = {}
    for row_idx in range(n):
        raw_code = 0
        parts: list[int] = []
        missing = False
        for column_idx, column in enumerate(columns):
            value = column[row_idx]
            if _is_missing_value(value):
                if not na_group:
                    missing = True
                    break
                value = None
            try:
                part = column_maps[column_idx][value]
            except KeyError as exc:
                raise ValueError("missing strata level could not be encoded") from exc
            raw_code = part if column_idx == 0 else part + raw_code * len(column_levels[column_idx])
            parts.append(part)
        if missing:
            raw_codes.append(None)
            continue
        raw_codes.append(raw_code)
        raw_to_parts.setdefault(raw_code, parts)

    observed_raw = sorted(raw_to_parts)
    compact = {raw_code: idx + 1 for idx, raw_code in enumerate(observed_raw)}
    codes = [None if raw_code is None else compact[raw_code] for raw_code in raw_codes]
    levels: list[str] = []
    for raw_code in observed_raw:
        pieces: list[str] = []
        for term_idx, part_idx in enumerate(raw_to_parts[raw_code]):
            level_label = column_level_labels[term_idx][part_idx]
            pieces.append(level_label if short else f"{term_labels[term_idx]}={level_label}")
        levels.append(sep.join(pieces))
    row_labels = [None if code is None else levels[code - 1] for code in codes]
    counts = [sum(1 for code in codes if code == idx + 1) for idx in range(len(levels))]
    return StrataFactor(codes=codes, levels=levels, labels=row_labels, counts=counts)


def _lvcf_order_key(value: Any) -> tuple[int, Any]:
    if _is_missing_value(value):
        raise ValueError("id must not contain missing values")
    if isinstance(value, bool):
        return (0, int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, str(value))


def _integerish_vector_or_none(values: Any, name: str) -> list[int] | None:
    try:
        return _integer_code_vector(values, name, "integer id values")
    except (TypeError, ValueError):
        return None


def neardate(
    id1: Any,
    id2: Any,
    y1: Any,
    y2: Any,
    best: Any = "after",
    nomatch: Any | None = None,
) -> list[int | None]:
    """Find nearest matching dates by id, returning R-style 1-based indices."""

    best_value = _match_string_arg(
        best,
        "best",
        ("after", "prior", "closest"),
        "best must be 'after', 'prior', or 'closest'",
    )
    y1_values = _float_vector(y1, "y1")
    y2_values = _float_vector(y2, "y2")
    id1_integer = _integerish_vector_or_none(id1, "id1")
    id2_integer = _integerish_vector_or_none(id2, "id2")
    nomatch_value = None if nomatch is None else _integer_scalar(nomatch, "nomatch")

    if id1_integer is not None and id2_integer is not None:
        result = _core.neardate(
            id1_integer,
            y1_values,
            id2_integer,
            y2_values,
            best_value,
            None,
        )
    else:
        result = _core.neardate_str(
            [str(value) for value in _materialize_1d(id1, "id1")],
            y1_values,
            [str(value) for value in _materialize_1d(id2, "id2")],
            y2_values,
            best_value,
            None,
        )

    return [nomatch_value if idx is None else int(idx) + 1 for idx in result.indices]


def _tcut_default_labels(breaks: list[float]) -> list[str]:
    return [f"{breaks[idx]:g}+ thru {breaks[idx + 1]:g}" for idx in range(len(breaks) - 1)]


def tcut(
    x: Any,
    breaks: Any,
    labels: Any | None = None,
    scale: Any = 1,
) -> TcutResult:
    """Create a Rust-backed R-style ``tcut`` interval result."""

    x_values = _float_vector(x, "x")
    break_values = _float_vector(breaks, "breaks")
    if len(break_values) < 2:
        raise ValueError("breaks must have at least 2 elements")
    if any(
        later <= earlier for earlier, later in zip(break_values[:-1], break_values[1:], strict=True)
    ):
        raise ValueError("breaks must be strictly increasing")
    scale_value = _normalize_positive_scale(scale)
    label_values = (
        _tcut_default_labels(break_values)
        if labels is None
        else [str(value) for value in _materialize_1d(labels, "labels")]
    )
    if len(label_values) != len(break_values) - 1:
        raise ValueError("labels length must equal length(breaks) - 1")
    return _core.tcut(
        [value * scale_value for value in x_values],
        [value * scale_value for value in break_values],
        label_values,
    )


def _quantile_type7(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("x must contain at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    p = min(max(probability, 0.0), 1.0)
    position = p * (len(sorted_values) - 1)
    lower_idx = int(math.floor(position))
    upper_idx = int(math.ceil(position))
    weight = position - lower_idx
    return sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight


def _unique_sorted_floats(values: Sequence[float]) -> list[float]:
    return sorted(set(values))


def _validate_nsk_boundary_pair(boundary_knots: tuple[float, float]) -> tuple[float, float]:
    low, high = boundary_knots
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("Boundary.knots must be finite and strictly increasing")
    return boundary_knots


def _normalize_nsk_knots(knots: Any | None) -> list[float] | None:
    if knots is None:
        return None
    knot_values = _normalize_numeric_sequence_or_none(knots, "knots") or []
    return _unique_sorted_floats(knot_values)


def _default_nsk_boundary_knots(x: list[float], b: Any) -> tuple[float, float]:
    b_value = _finite_float(b, "b")
    if b_value < 0.0 or b_value > 1.0:
        raise ValueError("b must be between 0 and 1")
    sorted_x = sorted(x)
    return _validate_nsk_boundary_pair(
        tuple(
            sorted(
                (
                    _quantile_type7(sorted_x, b_value),
                    _quantile_type7(sorted_x, 1.0 - b_value),
                )
            )
        )
    )


def _nsk_boundary_from_knots(knots: list[float] | None) -> tuple[tuple[float, float], list[float]]:
    if knots is None or len(knots) < 2:
        raise ValueError("wrong length for Boundary.knots")
    return _validate_nsk_boundary_pair((knots[0], knots[-1])), knots[1:-1]


def _adjust_nsk_boundary_for_knots(
    boundary_knots: tuple[float, float],
    knots: list[float] | None,
) -> tuple[tuple[float, float], list[float] | None]:
    if not knots:
        return boundary_knots, None

    kept_boundary = [boundary_knots[0], boundary_knots[1]]
    if kept_boundary[1] <= max(knots):
        kept_boundary = kept_boundary[:1]
    if kept_boundary and kept_boundary[0] >= min(knots):
        kept_boundary = kept_boundary[1:]

    all_knots = _unique_sorted_floats([*knots, *kept_boundary])
    if len(all_knots) < 2:
        raise ValueError("at least two distinct finite knots are required")
    return _validate_nsk_boundary_pair((all_knots[0], all_knots[-1])), all_knots[1:-1]


def _pop_nsk_boundary_alias(kwargs: dict[str, Any], current: Any, alias: str) -> Any:
    if alias not in kwargs:
        return current
    value = kwargs.pop(alias)
    if current is not _MISSING:
        raise ValueError(f"use only one of Boundary_knots or {alias}")
    return value


def _normalize_nsk_boundary_knots(
    x: list[float],
    knots: list[float] | None,
    b: Any,
    boundary_arg: Any,
) -> tuple[tuple[float, float], list[float] | None]:
    if boundary_arg is _MISSING:
        boundary_knots = _default_nsk_boundary_knots(x, b)
        return _adjust_nsk_boundary_for_knots(boundary_knots, knots)

    if _is_bool_like(boundary_arg):
        if bool(boundary_arg):
            boundary_knots = _validate_nsk_boundary_pair((min(x), max(x)))
            return _adjust_nsk_boundary_for_knots(boundary_knots, knots)
        return _nsk_boundary_from_knots(knots)

    if boundary_arg is None:
        return _nsk_boundary_from_knots(knots)

    boundary_values = _normalize_numeric_sequence_or_none(boundary_arg, "Boundary.knots") or []
    if len(boundary_values) == 0:
        return _nsk_boundary_from_knots(knots)
    if len(boundary_values) != 2:
        raise ValueError("wrong length for Boundary.knots")

    boundary_knots = _validate_nsk_boundary_pair(tuple(sorted(boundary_values)))
    return _adjust_nsk_boundary_for_knots(boundary_knots, knots)


def _computed_nsk_knots(
    x: list[float],
    boundary_knots: tuple[float, float],
    df: int | None,
    intercept: bool,
) -> list[float] | None:
    minimum_df = 2 if intercept else 1
    effective_df = minimum_df if df is None else df
    if effective_df < minimum_df:
        return None

    n_interior = effective_df - minimum_df
    if n_interior == 0:
        return None

    low, high = boundary_knots
    inside = sorted(value for value in x if low <= value <= high)
    if not inside:
        raise ValueError(
            f"not enough x values inside Boundary.knots to compute {n_interior} interior knots"
        )
    return [_quantile_type7(inside, idx / (n_interior + 1)) for idx in range(1, n_interior + 1)]


def _pspline_basis_row(knots: Sequence[float], x: float, order: int) -> list[float]:
    n_basis = len(knots) - order
    values = [0.0] * (len(knots) - 1)
    for idx in range(len(knots) - 1):
        if knots[idx] <= x < knots[idx + 1] or (
            x == knots[-1] and knots[idx] <= x <= knots[idx + 1]
        ):
            values[idx] = 1.0

    for current_order in range(2, order + 1):
        next_values = [0.0] * (len(knots) - current_order)
        for idx in range(len(next_values)):
            left_denominator = knots[idx + current_order - 1] - knots[idx]
            right_denominator = knots[idx + current_order] - knots[idx + 1]
            left = (
                0.0
                if left_denominator == 0.0
                else (x - knots[idx]) / left_denominator * values[idx]
            )
            right = (
                0.0
                if right_denominator == 0.0
                else (knots[idx + current_order] - x) / right_denominator * values[idx + 1]
            )
            next_values[idx] = left + right
        values = next_values
    return values[:n_basis]


def _pspline_basis_derivative_row(
    knots: Sequence[float],
    x: float,
    order: int,
) -> list[float]:
    lower_order = _pspline_basis_row(knots, x, order - 1)
    n_basis = len(knots) - order
    result: list[float] = []
    for idx in range(n_basis):
        left_denominator = knots[idx + order - 1] - knots[idx]
        right_denominator = knots[idx + order] - knots[idx + 1]
        left = 0.0 if left_denominator == 0.0 else (order - 1) / left_denominator * lower_order[idx]
        right = (
            0.0
            if right_denominator == 0.0
            else (order - 1) / right_denominator * lower_order[idx + 1]
        )
        result.append(left - right)
    return result


def _pspline_difference_penalty(n_cols: int) -> list[list[float]]:
    if n_cols == 0:
        return []
    diff_rows = []
    for row_idx in range(n_cols - 2):
        row = [0.0] * n_cols
        row[row_idx] = 1.0
        row[row_idx + 1] = -2.0
        row[row_idx + 2] = 1.0
        diff_rows.append(row)

    penalty = [[0.0] * n_cols for _ in range(n_cols)]
    for row in diff_rows:
        for i, left in enumerate(row):
            if left == 0.0:
                continue
            for j, right in enumerate(row):
                if right != 0.0:
                    penalty[i][j] += left * right
    return penalty


def _frailty_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _frailty_encoding(
    x: Any,
    *,
    levels: Any | None = None,
    sparse: Any | None = None,
) -> dict[str, Any]:
    values = _materialize_labels(x, "x")
    if levels is None:
        level_values = sorted({str(value) for value in values if not _frailty_missing(value)})
    else:
        level_values = [str(value) for value in _materialize_1d(levels, "levels")]
    level_index = {level: idx + 1 for idx, level in enumerate(level_values)}

    codes: list[int | None] = []
    for value in values:
        if _frailty_missing(value):
            codes.append(None)
        else:
            key = str(value)
            if key not in level_index:
                raise ValueError(f"x contains value {key!r} outside supplied levels")
            codes.append(level_index[key])

    sparse_value = (
        len(level_values) > 5 if sparse is None else _normalize_bool_option(sparse, "sparse")
    )
    return {
        "codes": codes,
        "levels": level_values,
        "nclass": len(level_values),
        "sparse": sparse_value,
    }


def _normalize_pspline_method(
    df: Any,
    theta: Any | None,
    nterm: Any | None,
    method: Any | None,
    eps: Any,
) -> tuple[int | float, float | None, int, float, str]:
    df_value = float(df)
    if not math.isfinite(df_value):
        raise ValueError("df must be finite")
    if df_value.is_integer():
        df_value = int(df_value)

    eps_value = 0.1 if eps is None else _finite_float(eps, "eps")
    nterm_value = None if nterm is None else int(round(_finite_float(nterm, "nterm")))
    if theta is not None:
        theta_value = _finite_float(theta, "theta")
        if theta_value <= 0.0 or theta_value >= 1.0:
            raise ValueError("Invalid value for theta")
        if nterm_value is None:
            nterm_value = int(round(2.5 * float(df_value)))
        return df_value, theta_value, nterm_value, eps_value, "fixed"

    method_value = None if method is None else str(method).lower()
    if float(df_value) == 0.0 or method_value == "aic":
        return df_value, None, 15, 1e-5, "aic"

    if float(df_value) <= 1.0:
        raise ValueError("Too few degrees of freedom")
    if nterm_value is None:
        nterm_value = int(round(2.5 * float(df_value)))
    if float(df_value) > nterm_value:
        raise ValueError(f"`nterm' too small for df={df_value:g}")
    return df_value, None, nterm_value, eps_value, "df"


def _pspline_combine_matrix(
    matrix: list[list[float]],
    combine: Any | None,
    intercept: bool,
) -> tuple[list[list[float]], list[int] | None]:
    if combine is None:
        return matrix, None

    raw_values = [float(value) for value in _materialize_1d(combine, "combine")]
    combine_values = [int(value) for value in raw_values]
    if any(value != math.floor(value) or value < 0.0 for value in raw_values):
        raise ValueError("combine must be an increasing vector of positive integers")
    if any(
        later < earlier for earlier, later in zip(combine_values, combine_values[1:], strict=False)
    ):
        raise ValueError("combine must be an increasing vector of positive integers")

    n_cols = len(matrix[0]) if matrix else 0
    column_groups = combine_values if intercept else [0, *combine_values]
    if len(column_groups) != n_cols:
        raise ValueError("wrong length for combine")

    unique_groups = sorted(set(column_groups))
    group_index = {group: idx for idx, group in enumerate(unique_groups)}
    combined = [[0.0] * len(unique_groups) for _ in matrix]
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            combined[row_idx][group_index[column_groups[col_idx]]] += value
    return combined, combine_values


def pspline(
    x: Any,
    df: Any = 4,
    theta: Any | None = None,
    nterm: Any | None = None,
    degree: Any = 3,
    eps: Any = 0.1,
    method: Any | None = None,
    Boundary_knots: Any | None = None,  # noqa: N803
    *,
    boundary_knots: Any | None = None,
    intercept: Any = False,
    penalty: Any = True,
    combine: Any | None = None,
) -> dict[str, Any]:
    """Create R-compatible ``survival::pspline`` basis data."""

    if Boundary_knots is not None and boundary_knots is not None:
        raise ValueError("use only one of Boundary_knots or boundary_knots")
    boundary_arg = boundary_knots if boundary_knots is not None else Boundary_knots

    x_values = _float_vector(x, "x")
    finite_x = [value for value in x_values if not math.isnan(value)]
    if not finite_x:
        raise ValueError("x must contain at least one non-missing value")
    if any(math.isinf(value) for value in finite_x):
        raise ValueError("x must contain only finite values")

    degree_value = _integer_scalar(degree, "degree")
    if degree_value < 1:
        raise ValueError("degree must be positive")
    order = degree_value + 1
    intercept_value = _normalize_bool_option(intercept, "intercept")
    penalty_value = _normalize_bool_option(penalty, "penalty")

    df_value, theta_value, nterm_value, eps_value, method_value = _normalize_pspline_method(
        df,
        theta,
        nterm,
        method,
        eps,
    )
    if nterm_value < 3:
        raise ValueError("Too few basis functions")

    if boundary_arg is None:
        boundary = (min(finite_x), max(finite_x))
    else:
        boundary_values = _float_vector(boundary_arg, "Boundary.knots")
        if len(boundary_values) != 2:
            raise ValueError("Invalid values for Boundary.knots")
        boundary = (boundary_values[0], boundary_values[1])
    if (
        not math.isfinite(boundary[0])
        or not math.isfinite(boundary[1])
        or boundary[0] >= boundary[1]
    ):
        raise ValueError("Invalid values for Boundary.knots")

    dx = (boundary[1] - boundary[0]) / nterm_value
    knots = [boundary[0] + dx * idx for idx in range(-degree_value, nterm_value)] + [
        boundary[1] + dx * idx for idx in range(0, degree_value + 1)
    ]

    left_basis = _pspline_basis_row(knots, boundary[0], order)
    left_derivative = _pspline_basis_derivative_row(knots, boundary[0], order)
    right_basis = _pspline_basis_row(knots, boundary[1], order)
    right_derivative = _pspline_basis_derivative_row(knots, boundary[1], order)

    full_matrix: list[list[float]] = []
    for value in x_values:
        if math.isnan(value):
            full_matrix.append([math.nan] * (nterm_value + degree_value))
        elif value < boundary[0]:
            full_matrix.append(
                [
                    basis + (value - boundary[0]) * derivative
                    for basis, derivative in zip(left_basis, left_derivative, strict=True)
                ]
            )
        elif value > boundary[1]:
            full_matrix.append(
                [
                    basis + (value - boundary[1]) * derivative
                    for basis, derivative in zip(right_basis, right_derivative, strict=True)
                ]
            )
        else:
            full_matrix.append(_pspline_basis_row(knots, value, order))

    full_matrix, combine_values = _pspline_combine_matrix(
        full_matrix,
        combine,
        intercept_value,
    )
    dmat = _pspline_difference_penalty(len(full_matrix[0]) if full_matrix else 0)
    if not intercept_value:
        full_matrix = [row[1:] for row in full_matrix]
        dmat = [row[1:] for row in dmat[1:]]

    n_cols = len(full_matrix[0]) if full_matrix else 0
    cbase_length = max(0, n_cols - 1) if intercept_value else n_cols
    return {
        "basis": full_matrix,
        "n_cols": n_cols,
        "nterm": nterm_value,
        "degree": degree_value,
        "df": df_value,
        "theta": theta_value,
        "eps": eps_value,
        "method": method_value,
        "boundary_knots": [boundary[0], boundary[1]],
        "dmat": dmat,
        "combine": combine_values,
        "penalty": penalty_value,
        "intercept": intercept_value,
        "cbase": [knots[idx] + (boundary[0] - knots[0]) for idx in range(1, cbase_length + 1)],
    }


def nsk(
    x: Any,
    df: Any | None = None,
    knots: Any | None = None,
    intercept: Any = False,
    b: Any = 0.05,
    Boundary_knots: Any = _MISSING,  # noqa: N803
    **kwargs: Any,
) -> Any:
    """Create a Rust-backed natural spline basis with R ``survival::nsk`` arguments."""

    boundary_arg = _pop_nsk_boundary_alias(kwargs, Boundary_knots, "Boundary.knots")
    boundary_arg = _pop_nsk_boundary_alias(kwargs, boundary_arg, "boundary_knots")
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"nsk got an unexpected keyword argument {unexpected!r}")

    x_values = _float_vector(x, "x")
    if not x_values:
        raise ValueError("x must contain at least one value")
    if any(not math.isfinite(value) for value in x_values):
        raise ValueError("x must contain only finite values")

    intercept_value = _normalize_bool_option(intercept, "intercept")
    df_value: int | None = None
    if df is not None:
        df_value = _integer_scalar(df, "df")
        if df_value <= 0:
            raise ValueError("df must be positive")

    normalized_knots = _normalize_nsk_knots(knots)
    boundary_knots, core_knots = _normalize_nsk_boundary_knots(
        x_values,
        normalized_knots,
        b,
        boundary_arg,
    )
    if not normalized_knots and core_knots is None:
        core_knots = _computed_nsk_knots(x_values, boundary_knots, df_value, intercept_value)
    spline = _core.NaturalSplineKnot(core_knots, boundary_knots, df_value, intercept_value)
    return spline.basis(x_values)


def lvcf(id: Any, x: Any, time: Any | None = None) -> list[Any]:  # noqa: A002
    """Carry the last non-missing value forward within each id, like R's ``lvcf``."""

    id_values = _materialize_labels(id, "id")
    result = _materialize_1d(x, "x")
    if len(result) != len(id_values):
        raise ValueError("x must have the same length as id")
    if time is None:
        order = sorted(
            range(len(id_values)),
            key=lambda idx: (_lvcf_order_key(id_values[idx]), idx),
        )
    else:
        time_values = _float_vector(time, "time")
        if len(time_values) != len(id_values):
            raise ValueError("time must have the same length as id")
        if any(not math.isfinite(value) for value in time_values):
            raise ValueError("time must contain only finite values")
        order = sorted(
            range(len(id_values)),
            key=lambda idx: (_lvcf_order_key(id_values[idx]), time_values[idx], idx),
        )

    current: Any = None
    previous_id: Any = None
    for position, row_idx in enumerate(order):
        id_key = _hashable_group_value(id_values[row_idx])
        value = result[row_idx]
        if position == 0 or not _is_missing_value(value) or id_key != previous_id:
            current = value
        else:
            result[row_idx] = current
        previous_id = id_key
    return result


def nostutter(
    id: Any,  # noqa: A002
    x: Any,
    censor: Any = 0,
    single: bool = False,
) -> list[Any]:
    """Replace repeated adjacent states within each id by the censor value."""

    id_values = _materialize_labels(id, "id")
    result = _materialize_1d(x, "x")
    if len(result) != len(id_values):
        raise ValueError("x must have the same length as id")
    if any(_is_missing_value(value) for value in id_values):
        raise ValueError("id must not contain missing values")

    current: Any = None
    previous_id: Any = None
    censor_key = _hashable_group_value(censor)
    used_for_id: set[Any] = set()
    for row_idx, (id_value, value) in enumerate(zip(id_values, result, strict=True)):
        id_key = _hashable_group_value(id_value)
        if row_idx == 0 or id_key != previous_id:
            used_for_id = set()
            current = censor if _is_missing_value(value) else value
            current_key = _hashable_group_value(current)
            if single and current_key != censor_key and not _is_missing_value(value):
                used_for_id.add(current_key)
        elif not _is_missing_value(value):
            value_key = _hashable_group_value(value)
            current_key = _hashable_group_value(current)
            if value_key == current_key or (single and value_key in used_for_id):
                result[row_idx] = censor
            elif value_key != censor_key:
                current = value
                if single:
                    used_for_id.add(value_key)
        previous_id = id_key
    return result


@dataclass(frozen=True, init=False)
class Surv:
    """Survival response container, like R's Surv."""

    time: tuple[float, ...]
    event: tuple[int, ...]
    start: tuple[float, ...] | None
    time2: tuple[float, ...] | None
    type: str

    def __init__(  # noqa: A002
        self,
        *args: Any,
        type: str | None = None,  # noqa: A002
        origin: Any = 0.0,
        time: Any = _MISSING,
        time1: Any = _MISSING,
        time2: Any = _MISSING,
        event: Any = _MISSING,
        status: Any = _MISSING,
        start: Any = _MISSING,
        stop: Any = _MISSING,
    ) -> None:
        named_options = {
            "time": time,
            "time1": time1,
            "time2": time2,
            "event": event,
            "status": status,
            "start": start,
            "stop": stop,
        }
        if args and any(value is not _MISSING for value in named_options.values()):
            raise TypeError(
                "Surv(...) must not mix positional and named time/time2/event arguments"
            )
        named_args = _collect_named_surv_arguments(named_options)
        if named_args is not None:
            args = named_args

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


@dataclass(frozen=True, init=False)
class Surv2:
    """Multi-state response container, like R's ``Surv2``."""

    time: tuple[float, ...]
    status: tuple[int | None, ...]
    states: tuple[str, ...]
    repeated: bool

    def __init__(self, time: Any, event: Any, repeated: Any = False) -> None:
        time_values = [
            math.nan if _is_missing_value(value) else float(value)
            for value in _materialize_1d(time, "time")
        ]
        event_values = _materialize_1d(event, "event")
        if len(event_values) != len(time_values):
            raise ValueError("Time and event are different lengths")
        if not _is_bool_like(repeated):
            raise ValueError("invalid value for repeated option")
        repeated_value = bool(repeated)

        levels = _surv2_levels(event_values)
        states = levels[1:]
        if any(state == "" for state in states):
            raise ValueError("each state must have a non-blank name")
        level_index = {level: idx for idx, level in enumerate(levels)}
        status = [
            None if _is_missing_value(value) else level_index[_surv2_event_label(value)]
            for value in event_values
        ]

        object.__setattr__(self, "time", tuple(time_values))
        object.__setattr__(self, "status", tuple(status))
        object.__setattr__(self, "states", tuple(states))
        object.__setattr__(self, "repeated", repeated_value)

    def __len__(self) -> int:
        return len(self.time)


def _surv2_event_label(value: Any) -> str:
    if isinstance(value, bool):
        return "FALSE" if not value else "TRUE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _surv_format_number(value)
    return str(value)


def _surv2_level_sort_key(label: str) -> tuple[int, Any]:
    try:
        numeric = float(label)
    except ValueError:
        return (1, label)
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, label)


def _surv2_levels(events: Sequence[Any]) -> list[str]:
    levels: dict[str, None] = {}
    for value in events:
        if not _is_missing_value(value):
            levels.setdefault(_surv2_event_label(value), None)
    return sorted(levels, key=_surv2_level_sort_key)


def _surv2data_status_values(status: Any) -> list[int | None]:
    values: list[int | None] = []
    for value in _materialize_1d(status, "status"):
        if _is_missing_value(value):
            values.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("Surv2 status values must be integer codes")
        values.append(int(numeric))
    return values


def _surv2data_sort_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, str(value))


def Surv2data(  # noqa: N802
    time: Any,
    status: Any,
    *,
    states: Any | None = None,
    repeated: Any = False,
    id: Any,  # noqa: A002
) -> dict[str, Any]:
    """Convert R ``Surv2`` timeline rows into start-stop transition rows."""

    time_values = _float_vector(time, "time")
    status_values = _surv2data_status_values(status)
    id_values = _materialize_labels(id, "id")
    if len(status_values) != len(time_values) or len(id_values) != len(time_values):
        raise ValueError("time, status, and id must have the same length")
    if any(_is_missing_value(value) for value in id_values) or any(
        math.isnan(value) for value in time_values
    ):
        raise ValueError("id and time cannot be missing")
    repeated_value = _normalize_bool_option(repeated, "repeated")
    state_values = (
        [str(value) for value in _materialize_1d(states, "states")] if states is not None else []
    )

    order = sorted(
        range(len(time_values)),
        key=lambda idx: (_surv2data_sort_value(id_values[idx]), time_values[idx], idx),
    )
    intervals: list[dict[str, Any]] = []
    for pos, row_idx in enumerate(order):
        if pos + 1 >= len(order):
            continue
        next_idx = order[pos + 1]
        if _hashable_group_value(id_values[row_idx]) != _hashable_group_value(id_values[next_idx]):
            continue
        if time_values[row_idx] == time_values[next_idx]:
            raise ValueError("duplicated time values for a single id")
        event = status_values[next_idx]
        current_state = status_values[row_idx]
        event_code = 0 if event is None else int(event)
        istate_code = None if current_state is None else int(current_state)
        if not repeated_value and istate_code is not None and event_code == istate_code:
            event_code = 0
        intervals.append(
            {
                "row": row_idx,
                "start": time_values[row_idx],
                "stop": time_values[next_idx],
                "status": event_code,
                "id": id_values[row_idx],
                "istate": istate_code,
            }
        )

    intervals.sort(key=lambda item: item["row"])
    starts = [float(item["start"]) for item in intervals]
    response_type = (
        "mright"
        if state_values and starts and all(value == 0.0 for value in starts)
        else "mcounting"
        if state_values
        else "right"
        if starts and all(value == 0.0 for value in starts)
        else "counting"
    )
    return {
        "row": [int(item["row"]) for item in intervals],
        "start": starts,
        "stop": [float(item["stop"]) for item in intervals],
        "status": [int(item["status"]) for item in intervals],
        "id": [item["id"] for item in intervals],
        "istate": [item["istate"] for item in intervals],
        "states": state_values,
        "type": response_type,
    }


def _totimeline_state_values(states: Any | None) -> list[str]:
    return [str(value) for value in _materialize_1d(states, "states")] if states is not None else []


def _totimeline_check_states(
    event_states: Sequence[str],
    istate_levels: Any | None,
) -> list[str]:
    if istate_levels is None:
        return ["(s0)", *event_states]
    levels = [str(value) for value in _materialize_1d(istate_levels, "istate_levels")]
    return [level for level in levels if level not in event_states] + list(event_states)


def _totimeline_istate_codes(
    istate: Any | None,
    check_states: Sequence[str],
    n: int,
) -> list[int]:
    if istate is None:
        return [1] * n
    labels = [
        None if _is_missing_value(value) else str(value)
        for value in _materialize_1d(istate, "istate")
    ]
    if len(labels) != n:
        raise ValueError("istate must have the same length as the Surv response")
    code_by_state = {state: idx + 1 for idx, state in enumerate(check_states)}
    result: list[int] = []
    for label in labels:
        if label is None:
            raise ValueError("istate contains missing values")
        try:
            result.append(code_by_state[label])
        except KeyError as exc:
            raise ValueError(f"istate level {label!r} is not a recognized state") from exc
    return result


def totimeline(
    start: Any,
    stop: Any,
    status: Any,
    *,
    states: Any,
    id: Any,  # noqa: A002
    istate: Any | None = None,
    istate_levels: Any | None = None,
) -> dict[str, Any]:
    """Convert start-stop multi-state rows into R ``totimeline`` rows."""

    start_values = _float_vector(start, "start")
    stop_values = _float_vector(stop, "stop")
    status_values = [int(value) for value in _int_vector(status, "status")]
    id_values = _materialize_labels(id, "id")
    n = len(start_values)
    if len(stop_values) != n or len(status_values) != n or len(id_values) != n:
        raise ValueError("start, stop, status, and id must have the same length")
    if any(not math.isfinite(value) for value in [*start_values, *stop_values]):
        raise ValueError("start and stop times must be finite")
    event_states = _totimeline_state_values(states)
    if not event_states:
        raise ValueError("states must contain at least one event state")
    check_states = _totimeline_check_states(event_states, istate_levels)
    istate_codes = _totimeline_istate_codes(istate, check_states, n)
    event_code_by_status = {
        status_idx + 1: check_states.index(state) + 1
        for status_idx, state in enumerate(event_states)
    }

    first = []
    seen: set[Any] = set()
    for id_value in id_values:
        key = _hashable_group_value(id_value)
        first.append(key not in seen)
        seen.add(key)

    last = [False] * n
    seen.clear()
    for row_idx in range(n - 1, -1, -1):
        key = _hashable_group_value(id_values[row_idx])
        last[row_idx] = key not in seen
        seen.add(key)

    times: list[float] = []
    state_codes: list[int] = []
    data_rows: list[int] = []
    for row_idx in range(n):
        if first[row_idx]:
            times.append(start_values[row_idx])
            state_codes.append(istate_codes[row_idx])
            data_rows.append(row_idx)

        times.append(stop_values[row_idx])
        status_value = status_values[row_idx]
        if status_value < 0 or status_value > len(event_states):
            raise ValueError("status code is outside the event state range")
        state_codes.append(0 if status_value == 0 else event_code_by_status[status_value])
        data_rows.append(row_idx if last[row_idx] else row_idx + 1)

    state_levels = (
        ["(censor)", *check_states]
        if any(state == "censor" for state in check_states)
        else ["censor", *check_states]
    )
    return {
        "time": times,
        "status": state_codes,
        "data_row": data_rows,
        "state_levels": state_levels,
    }


def _fromtimeline_data_columns(data: Any | None, n: int) -> tuple[list[str], list[list[Any]]]:
    if data is None:
        return [], []
    if not isinstance(data, Mapping):
        raise TypeError("data must be mapping-like")
    names = [str(name) for name in data]
    columns = [_materialize_1d(data[name], str(name)) for name in data]
    for name, column in zip(names, columns, strict=True):
        if len(column) != n:
            raise ValueError(f"{name} must have the same length as the Surv response")
    return names, columns


def _fromtimeline_static_columns(
    columns: Sequence[Sequence[Any]],
    id_values: Sequence[Any],
    column_names: Sequence[str],
    id_name: str,
) -> list[bool]:
    result: list[bool] = []
    for name, column in zip(column_names, columns, strict=True):
        if name == id_name:
            result.append(True)
            continue
        if any(_is_missing_value(value) for value in column):
            result.append(False)
            continue
        first_by_id: dict[Any, Any] = {}
        static = True
        for value, id_value in zip(column, id_values, strict=True):
            key = _hashable_group_value(id_value)
            if key not in first_by_id:
                first_by_id[key] = value
            elif value != first_by_id[key]:
                static = False
                break
        result.append(static)
    return result


def fromtimeline(
    time: Any,
    status: Any,
    *,
    id: Any,  # noqa: A002
    states: Any | None = None,
    data: Any | None = None,
    id_name: Any = "id",
) -> dict[str, Any]:
    """Convert right-censored timeline rows into R ``fromtimeline`` intervals."""

    time_values = _float_vector(time, "time")
    status_values = [int(value) for value in _int_vector(status, "status")]
    id_values = _materialize_labels(id, "id")
    n = len(time_values)
    if len(status_values) != n or len(id_values) != n:
        raise ValueError("time, status, and id must have the same length")
    if any(not math.isfinite(value) for value in time_values):
        raise ValueError("time values must be finite")
    id_name_value = str(id_name)
    column_names, columns = _fromtimeline_data_columns(data, n)
    static_columns = _fromtimeline_static_columns(columns, id_values, column_names, id_name_value)

    groups: dict[Any, list[int]] = {}
    ordered_keys: list[Any] = []
    for row_idx, id_value in enumerate(id_values):
        key = _hashable_group_value(id_value)
        if key not in groups:
            ordered_keys.append(key)
            groups[key] = []
        groups[key].append(row_idx)

    output_start: list[float] = []
    output_stop: list[float] = []
    output_status: list[int] = []
    output_istate: list[int] = []
    static_rows: list[int] = []
    dynamic_rows: list[int] = []
    removed_ids: list[Any] = []

    for key in ordered_keys:
        rows = sorted(groups[key], key=lambda idx: (time_values[idx], idx))
        if len(rows) < 2 or time_values[rows[0]] == time_values[rows[-1]]:
            removed_ids.append(id_values[rows[0]])
            continue
        if status_values[rows[0]] == 0:
            raise ValueError("no observation should start in a censored state")
        first_row = rows[0]
        for position, row_idx in enumerate(rows[:-1]):
            next_idx = rows[position + 1]
            output_start.append(time_values[row_idx])
            output_stop.append(time_values[next_idx])
            output_status.append(status_values[next_idx])
            output_istate.append(status_values[row_idx])
            static_rows.append(first_row)
            dynamic_rows.append(row_idx)

    state_values = _totimeline_state_values(states) if states is not None else []
    if state_values:
        state_levels = ["censor", *state_values]
        istate_levels = state_values
    else:
        state_levels = []
        istate_levels = []

    return {
        "start": output_start,
        "stop": output_stop,
        "status": output_status,
        "istate": output_istate,
        "static": static_columns,
        "static_row": static_rows,
        "dynamic_row": dynamic_rows,
        "state_levels": state_levels,
        "istate_levels": istate_levels,
        "removed_id": removed_ids,
    }


def is_surv(value: Any) -> bool:
    """Return whether *value* is a survival response object, like R's is.Surv."""

    return isinstance(value, Surv)


def _surv_missing_row(response: Surv, idx: int) -> bool:
    if response.start is not None and math.isnan(response.start[idx]):
        return True
    if math.isnan(response.time[idx]):
        return True
    return response.time2 is not None and math.isnan(response.time2[idx])


def is_na_surv(x: Any) -> list[bool]:
    """Return row-wise missingness for a ``Surv`` response, like R's ``is.na.Surv``."""

    if isinstance(x, Surv2):
        return [
            math.isnan(time) or status is None
            for time, status in zip(x.time, x.status, strict=True)
        ]
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    return [_surv_missing_row(x, idx) for idx in range(len(x))]


def _surv_format_number(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "Inf" if value > 0.0 else "-Inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_surv_right_or_left(response: Surv) -> list[str]:
    suffix = "+" if response.type == "right" else "-"
    times = [_surv_format_number(value) for value in response.time]
    width = max(len(value) for value in times)
    return [
        f"{time.rjust(width)}{' ' if event else suffix}"
        for time, event in zip(times, response.event, strict=True)
    ]


def _format_surv_counting(response: Surv) -> list[str]:
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    starts = [_surv_format_number(value) for value in response.start]
    stops = [_surv_format_number(value) for value in response.time]
    start_width = max(len(value) for value in starts)
    stop_width = max(len(value) for value in stops)
    labels = [
        f"({start.rjust(start_width)}, {stop.rjust(stop_width)}{'' if event else '+'}]"
        for start, stop, event in zip(starts, stops, response.event, strict=True)
    ]
    width = max(len(value) for value in labels)
    return [value.ljust(width) for value in labels]


def _format_surv_interval(response: Surv) -> list[str]:
    if response.time2 is None:
        raise ValueError(f"{response.type} Surv response is missing time2")
    labels: list[str] = []
    for left, right, status in zip(response.time, response.time2, response.event, strict=True):
        left_label = _surv_format_number(left)
        right_label = _surv_format_number(right)
        if status == 0:
            labels.append(f"{left_label}+")
        elif status == 1:
            labels.append(left_label)
        elif status == 2:
            labels.append(f"{right_label}-")
        else:
            labels.append(f"[{left_label}, {right_label}]")
    width = max(len(value) for value in labels)
    return [value.ljust(width) for value in labels]


def _format_surv2(response: Surv2) -> list[str]:
    labels: list[str] = []
    suffixes = ["+", *(f":{state}" for state in response.states)]
    for time, status in zip(response.time, response.status, strict=True):
        suffix = "?" if status is None else suffixes[status]
        labels.append(f"{_surv_format_number(time)}{suffix}")
    width = max(len(value) for value in labels) if labels else 0
    return [value.ljust(width) for value in labels]


def format_surv(x: Any) -> list[str]:
    """Return R-style display strings for a ``Surv`` response."""

    if isinstance(x, Surv2):
        return _format_surv2(x)
    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    if x.type in {"right", "left"}:
        return _format_surv_right_or_left(x)
    if x.type == "counting":
        return _format_surv_counting(x)
    if x.type in {"interval", "interval2"}:
        return _format_surv_interval(x)
    raise ValueError(f"unsupported Surv type {x.type!r}")


def is_ratetable(
    x: Any,
    has_rates: Any | None = None,
    has_dims: Any | None = None,
    verbose: Any = False,
) -> bool:
    """Return whether *x* is a population rate table, like R's ``is.ratetable``."""

    _normalize_bool_option(verbose, "verbose")
    if has_rates is not None or has_dims is not None:
        if has_rates is None or has_dims is None:
            raise TypeError("has_rates and has_dims must be supplied together")
        return _core.is_ratetable(
            _integer_scalar(x, "ndim"),
            _normalize_bool_option(has_rates, "has_rates"),
            _normalize_bool_option(has_dims, "has_dims"),
        )
    return isinstance(x, RateTable)


def _ratetable_date_from_components(
    year: Any,
    month: Any,
    day: Any,
    origin_year: Any,
) -> float:
    result = _core.ratetable_date(
        _integer_scalar(year, "year"),
        _integer_scalar(month, "month"),
        _integer_scalar(day, "day"),
        _integer_scalar(origin_year, "origin_year"),
    )
    return float(result.days)


def _ratetable_date_value(value: Any, origin_year: Any) -> float:
    if value is None or _is_missing_value(value):
        return math.nan
    if isinstance(value, _DateTime):
        return _ratetable_date_from_components(
            value.year,
            value.month,
            value.day,
            origin_year,
        )
    if isinstance(value, _Date):
        return _ratetable_date_from_components(
            value.year,
            value.month,
            value.day,
            origin_year,
        )
    if isinstance(value, str):
        parsed = _Date.fromisoformat(value[:10])
        return _ratetable_date_from_components(
            parsed.year,
            parsed.month,
            parsed.day,
            origin_year,
        )
    return float(value)


def ratetableDate(  # noqa: N802
    x: Any,
    month: Any | None = None,
    day: Any | None = None,
    *,
    origin_year: Any = 1970,
) -> Any:
    """Convert dates to rate-table day counts, matching R's ``ratetableDate``."""

    if month is not None or day is not None:
        if month is None or day is None:
            raise TypeError("month and day must be supplied together")
        return _ratetable_date_from_components(x, month, day, origin_year)
    if isinstance(x, Sequence) and not isinstance(x, str | bytes | bytearray):
        return [_ratetable_date_value(value, origin_year) for value in x]
    return _ratetable_date_value(x, origin_year)


def _survexp_ratetable(ratetable: Any | None) -> RateTable:
    if ratetable is None:
        return _core.survexp_us()
    if not isinstance(ratetable, RateTable):
        raise TypeError("ratetable must be a RateTable")
    return ratetable


def _normalize_survexp_method(
    method: Any | None,
    cohort: Any,
    conditional: Any,
) -> str:
    cohort_value = _normalize_bool_option_with_default(cohort, "cohort", True)
    conditional_value = _normalize_bool_option(conditional, "conditional")
    if method is None:
        if conditional_value:
            return "conditional"
        if not cohort_value:
            return "individual.s"
        return "hakulinen"
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    value = method.strip().lower().replace("_", ".")
    aliases = {
        "ederer": "hakulinen",
        "hakulinen": "hakulinen",
        "conditional": "conditional",
        "individual": "individual",
        "individual.h": "individual.h",
        "individual.s": "individual.s",
    }
    if value not in aliases:
        raise ValueError(
            "method must be 'ederer', 'hakulinen', 'conditional', "
            "'individual.h', 'individual.s', or 'individual'"
        )
    return aliases[value]


def _normalize_positive_scale(value: Any) -> float:
    scale = _finite_float(value, "scale")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return scale


def _survexp_result_from_core(result: Any, scale: float) -> SurvExpResult:
    return SurvExpResult(
        time=[float(value) / scale for value in result.time],
        surv=[float(value) for value in result.surv],
        n_risk=[float(value) for value in result.n_risk],
        cumhaz=[float(value) for value in result.cumhaz],
        method=str(result.method),
        n=int(result.n),
    )


def survexp(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
    times: Any | None = None,
    method: Any | None = None,
    *,
    cohort: Any = True,
    conditional: Any = False,
    scale: Any = 1.0,
    se_fit: Any | None = None,
) -> SurvExpResult | list[float]:
    """Compute expected survival from direct vectors and a population rate table."""

    if se_fit is not None and _normalize_bool_option(se_fit, "se_fit"):
        warnings.warn("se_fit value ignored", RuntimeWarning, stacklevel=2)
    method_value = _normalize_survexp_method(method, cohort, conditional)
    scale_value = _normalize_positive_scale(scale)
    table = _survexp_ratetable(ratetable)
    time_values = _float_vector(time, "time")
    age_values = _float_vector(age, "age")
    year_values = _float_vector(year, "year")
    sex_values = None if sex is None else _int_vector(sex, "sex")

    if method_value in {"individual.h", "individual.s"}:
        individual = _core.survexp_individual(
            time_values,
            age_values,
            year_values,
            table,
            sex_values,
        )
        values = [float(value) for value in individual]
        if method_value == "individual.s":
            return values
        return [-math.log(value) if value > 0.0 else math.inf for value in values]

    result = _core.survexp(
        time_values,
        age_values,
        year_values,
        table,
        sex_values,
        None if times is None else _float_vector(times, "times"),
        method_value,
    )
    return _survexp_result_from_core(result, scale_value)


def survexp_individual(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
) -> list[float]:
    """Return per-subject expected survival from direct vectors."""

    return [
        float(value)
        for value in _core.survexp_individual(
            _float_vector(time, "time"),
            _float_vector(age, "age"),
            _float_vector(year, "year"),
            _survexp_ratetable(ratetable),
            None if sex is None else _int_vector(sex, "sex"),
        )
    ]


def _pyears_response_from_direct(
    response: Any,
    *,
    time: Any,
    start: Any,
    stop: Any,
    event: Any,
) -> tuple[list[float], list[float], list[float] | None, int, bool]:
    if response is not None:
        if isinstance(response, Surv):
            if response.type == "right":
                return [], list(response.time), list(response.event), 2, True
            if response.type == "counting":
                if response.start is None:
                    raise ValueError("counting Surv response is missing start times")
                return list(response.start), list(response.time), list(response.event), 3, True
            raise ValueError("pyears supports only right-censored and counting Surv responses")
        if time is not _MISSING or stop is not _MISSING:
            raise ValueError("use either response or explicit time/start/stop inputs")
        stop_values = _float_vector(response, "time")
        event_values = None if event is _MISSING or event is None else _event_vector(event, "event")
        return [], stop_values, event_values, 2, True

    if stop is not _MISSING:
        if start is _MISSING:
            raise TypeError("start must be supplied with stop")
        start_values = _float_vector(start, "start")
        stop_values = _float_vector(stop, "stop")
        event_values = None if event is _MISSING or event is None else _event_vector(event, "event")
        return (
            start_values,
            stop_values,
            event_values,
            3 if event_values is not None else 2,
            (event_values is not None),
        )
    if start is not _MISSING:
        raise TypeError("stop must be supplied with start")
    if time is _MISSING:
        raise TypeError("pyears requires a response or time vector")
    stop_values = _float_vector(time, "time")
    event_values = None if event is _MISSING or event is None else _event_vector(event, "event")
    return [], stop_values, event_values, 2, True


def _pyears_validate_time_columns(
    start: list[float],
    stop: list[float],
    event: list[float] | None,
) -> None:
    n = len(stop)
    if start and len(start) != n:
        raise ValueError("start and stop must have the same length")
    if event is not None and len(event) != n:
        raise ValueError("event must have the same length as time")
    if n == 0:
        raise ValueError("pyears requires at least one observation")
    for idx, value in enumerate(stop):
        if not math.isfinite(value):
            raise ValueError(f"time contains non-finite value at index {idx}")
        if value < 0.0:
            raise ValueError(f"time contains negative value at index {idx}")
    for idx, value in enumerate(start):
        if not math.isfinite(value):
            raise ValueError(f"start contains non-finite value at index {idx}")
        if value < 0.0:
            raise ValueError(f"start contains negative value at index {idx}")
        if stop[idx] < value:
            raise ValueError(f"stop must be greater than or equal to start at index {idx}")


def _pyears_weights(weights: Any | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    values = [_finite_float(value, "weights") for value in _materialize_1d(weights, "weights")]
    if len(values) != n:
        raise ValueError("weights must have the same length as the response")
    for idx, value in enumerate(values):
        if value < 0.0:
            raise ValueError(f"weights contains negative value at index {idx}")
    return values


def _pyears_group_codes(
    group: Any | None,
    n: int,
    *,
    levels: Sequence[Any] | None = None,
) -> tuple[list[float], list[str]]:
    if group is None:
        return [1.0] * n, ["(all)"]
    values = _materialize_labels(group, "group")
    if len(values) != n:
        raise ValueError("group must have the same length as the response")
    group_levels = tuple(levels) if levels is not None else _label_levels(values, "group")
    labels = {value: idx + 1 for idx, value in enumerate(group_levels)}
    try:
        codes = [float(labels[value]) for value in values]
    except KeyError as exc:
        raise ValueError("group contains a value outside the supplied levels") from exc
    return codes, [str(value) for value in group_levels]


def _pyears_formula_group_and_levels(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[Any] | None, tuple[Any, ...] | None]:
    columns: list[list[Any]] = []
    if terms.model_terms:
        for model_term in terms.model_terms:
            if isinstance(model_term, _ModelCovariateTerm):
                columns.append(_term_values(data, model_term.term, n))
            elif isinstance(model_term, _ModelStrataTerm):
                columns.append(_survcondense_strata_values(data, model_term.columns, n))
            elif isinstance(model_term, _ModelClusterTerm):
                columns.append(_column(data, model_term.column))
    else:
        columns = [
            *[_column(data, term) for term in terms.strata],
            *[_term_values(data, term, n) for term in terms.covariates],
            *[_column(data, term) for term in terms.clusters],
        ]
    if not columns:
        return None, None
    group = _combine_aligned_columns(columns, n)
    column_levels = [
        _r_formula_ordered_levels(column, "pyears formula groups") for column in columns
    ]
    if len(column_levels) == 1:
        return group, column_levels[0]
    levels = tuple(tuple(reversed(parts)) for parts in product(*reversed(column_levels)))
    return group, levels


def _pyears_formula_inputs(
    formula: str,
    data: Any,
    weights: Any | None,
    subset: Any | None,
    na_action: str | None,
) -> tuple[Surv, list[Any] | None, tuple[Any, ...] | None, list[float] | None]:
    if data is None:
        raise ValueError("data is required when pyears response is a formula")
    aligned_weights = None if weights is None else _materialize_1d(weights, "weights")
    if subset is not None:
        data, aligned = _subset_formula_inputs(formula, data, subset, weights=aligned_weights)
        aligned_weights = aligned["weights"]
    data, aligned = _apply_formula_na_action(formula, data, na_action, weights=aligned_weights)
    aligned_weights = aligned["weights"]
    response, terms = _parse_formula(formula, data)
    if any(isinstance(term, _InteractionTerm) for term in terms.covariates):
        raise ValueError("pyears formula does not support interaction terms")
    group, group_levels = _pyears_formula_group_and_levels(data, terms, len(response))
    return (
        response,
        group,
        group_levels,
        None if aligned_weights is None else [float(value) for value in aligned_weights],
    )


def _pyears_result_frame(result: PyearsResult) -> dict[str, list[Any]]:
    frame: dict[str, list[Any]] = {
        "group": result.group,
        "pyears": result.pyears,
        "n": result.n,
    }
    if result.expected is not None:
        frame["expected"] = result.expected
    if result.event is not None:
        frame["event"] = result.event
    return frame


def _finegray_frame(result: Any) -> dict[str, list[Any]]:
    return {
        "row": [int(value) for value in result.row],
        "start": [float(value) for value in result.start],
        "end": [float(value) for value in result.end],
        "wt": [float(value) for value in result.wt],
        "add": [int(value) for value in result.add],
    }


def pyears(
    response: Any = None,
    data: Any | None = None,
    *,
    time: Any = _MISSING,
    start: Any = _MISSING,
    stop: Any = _MISSING,
    event: Any = _MISSING,
    group: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    scale: Any = 365.25,
    data_frame: Any = False,
) -> PyearsResult | dict[str, list[Any]]:
    """Tabulate person-years for direct ``Surv``/formula inputs, like R's ``pyears``."""

    if isinstance(response, str) and "~" in response:
        response, formula_group, formula_group_levels, formula_weights = _pyears_formula_inputs(
            response,
            data,
            weights,
            subset,
            na_action,
        )
        if group is not None:
            raise ValueError("group must not be supplied separately when response is a formula")
        group = formula_group
        weights = formula_weights
    else:
        formula_group_levels = None
        start_values, stop_values, event_values, _ny, _do_event = _pyears_response_from_direct(
            response,
            time=time,
            start=start,
            stop=stop,
            event=event,
        )
        n_direct = len(stop_values)
        if subset is not None:
            keep = _subset_indices(subset, n_direct)
            response = (
                Surv([stop_values[idx] for idx in keep], [event_values[idx] for idx in keep])
                if not start_values and event_values is not None
                else None
            )
            start = [start_values[idx] for idx in keep] if start_values else _MISSING
            stop = [stop_values[idx] for idx in keep]
            event = [event_values[idx] for idx in keep] if event_values is not None else _MISSING
            group = _subset_optional_sequence(group, keep, "group")
            weights = _subset_optional_sequence(weights, keep, "weights")
    start_values, stop_values, event_values, ny, do_event = _pyears_response_from_direct(
        response,
        time=time,
        start=start,
        stop=stop,
        event=event,
    )
    n = len(stop_values)
    event_for_core = [0.0] * n if event_values is None else [float(value) for value in event_values]
    _pyears_validate_time_columns(start_values, stop_values, event_values)
    weight_values = _pyears_weights(weights, n)
    group_codes, group_labels = _pyears_group_codes(group, n, levels=formula_group_levels)
    scale_value = _normalize_positive_scale(scale)
    _normalize_bool_option(data_frame, "data_frame")

    time_data = (
        [*start_values, *stop_values, *event_for_core]
        if start_values and event_values is not None
        else [*start_values, *stop_values]
        if start_values
        else [*stop_values, *event_for_core]
    )
    raw = _core.perform_pyears_calculation(
        time_data,
        weight_values,
        1,
        [1],
        [1],
        [],
        [0.0],
        [1.0] * n,
        1,
        [1],
        [len(group_labels)],
        [],
        1,
        group_codes,
        1 if do_event else 0,
        ny,
    )
    result = PyearsResult(
        pyears=[float(value) / scale_value for value in raw["pyears"]],
        n=[float(value) for value in raw["pn"]],
        offtable=float(raw["offtable"]) / scale_value,
        group=group_labels,
        observations=n,
        event=[float(value) for value in raw["pcount"]] if event_values is not None else None,
        expected=None,
        tcut=False,
    )
    return _pyears_result_frame(result) if data_frame else result


def finegray(
    tstart: Any,
    tstop: Any,
    ctime: Any,
    cprob: Any,
    extend: Any,
    keep: Any,
) -> FineGrayOutput:
    """Expand intervals for Fine-Gray competing-risk data, like R's internal kernel."""

    return _core.finegray(
        _float_vector(tstart, "tstart"),
        _float_vector(tstop, "tstop"),
        _float_vector(ctime, "ctime"),
        _float_vector(cprob, "cprob"),
        _bool_vector(extend, "extend"),
        _bool_vector(keep, "keep"),
    )


def _survobrien_default_transform(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: (values[idx], idx))
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start + 1
        while end < n and values[order[end]] == values[order[start]]:
            end += 1
        rank_sum = sum(range(start + 1, end + 1))
        rank = rank_sum / (end - start)
        for pos in range(start, end):
            ranks[order[pos]] = float(rank)
        start = end
    transformed = []
    for rank in ranks:
        probability = (rank - 0.5) / n
        transformed.append(math.log(probability / (1.0 - probability)))
    return transformed


def _survobrien_transform_values(
    values: Sequence[float],
    transform: Any | None,
) -> list[float]:
    if transform is None:
        return _survobrien_default_transform(values)
    if not callable(transform):
        raise TypeError("transform must be callable")
    raw_result = transform(list(values))
    try:
        result = _materialize_1d(raw_result, "transform")
    except TypeError:
        if len(values) != 1:
            raise
        result = [raw_result]
    if len(result) != len(values):
        raise ValueError("Transform function must be 1 to 1")
    try:
        transformed = [float(value) for value in result]
    except (TypeError, ValueError) as exc:
        raise ValueError("transform must return numeric values") from exc
    if any(not math.isfinite(value) for value in transformed):
        raise ValueError("transform must return finite values")
    return transformed


def _survobrien_term_name(term: _CovariateTerm) -> str:
    if term.transform is not None:
        return f"{term.transform}({term.column})"
    return term.column


def _survobrien_formula_terms(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[tuple[str, list[Any]]], list[tuple[str, list[float]]]]:
    keepers: list[tuple[str, list[Any]]] = []
    continuous: list[tuple[str, list[float]]] = []
    for term in terms.covariates:
        if isinstance(term, _InteractionTerm):
            raise ValueError("This function cannot deal with interaction terms")
        values = _term_values(data, term, n)
        if term.categorical:
            keepers.append((_survobrien_term_name(term), values))
            continue
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            keepers.append((_survobrien_term_name(term), values))
            continue
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"formula term {term.column!r} must be finite")
        continuous.append((_survobrien_term_name(term), numeric))
    if not continuous:
        raise ValueError("No continuous variables to modify")
    return keepers, continuous


def _survobrien_event_sets(
    response: Surv,
    strata_values: list[Any] | None,
) -> list[tuple[float, list[int]]]:
    if response.type == "right":
        if strata_values is None:
            event_times = sorted(
                {
                    float(time)
                    for time, event in zip(response.time, response.event, strict=True)
                    if event == 1
                }
            )
            return [
                (
                    event_time,
                    [idx for idx, time in enumerate(response.time) if float(time) >= event_time],
                )
                for event_time in event_times
            ]

        seen: set[tuple[float, Any]] = set()
        result: list[tuple[float, list[int]]] = []
        for event_time, event, stratum in zip(
            response.time,
            response.event,
            strata_values,
            strict=True,
        ):
            if event != 1:
                continue
            key = (float(event_time), _hashable_group_value(stratum))
            if key in seen:
                continue
            seen.add(key)
            # Match survival::survobrien's right-censored strata branch, including
            # its status-column risk-set test.
            result.append(
                (
                    float(event_time),
                    [
                        idx
                        for idx, (row_event, row_stratum) in enumerate(
                            zip(response.event, strata_values, strict=True)
                        )
                        if float(row_event) >= float(event_time) and row_stratum == stratum
                    ],
                )
            )
        return result

    if response.type == "counting":
        if response.start is None:
            raise ValueError("counting Surv response is missing start times")
        if strata_values is None:
            event_times = sorted(
                {
                    float(stop)
                    for stop, event in zip(response.time, response.event, strict=True)
                    if event == 1
                }
            )
            return [
                (
                    event_time,
                    [
                        idx
                        for idx, (start, stop) in enumerate(
                            zip(response.start, response.time, strict=True)
                        )
                        if float(start) < event_time <= float(stop)
                    ],
                )
                for event_time in event_times
            ]

        seen: set[tuple[float, Any]] = set()
        result: list[tuple[float, list[int]]] = []
        for event_time, event, stratum in zip(
            response.time,
            response.event,
            strata_values,
            strict=True,
        ):
            if event != 1:
                continue
            event_time_float = float(event_time)
            stratum_key = _hashable_group_value(stratum)
            key = (event_time_float, stratum_key)
            if key in seen:
                continue
            seen.add(key)
            # Match survival::survobrien's counting-process strata branch,
            # whose risk-set predicate uses rows outside the event stratum.
            result.append(
                (
                    event_time_float,
                    [
                        idx
                        for idx, (start, stop, row_stratum) in enumerate(
                            zip(
                                response.start,
                                response.time,
                                strata_values,
                                strict=True,
                            )
                        )
                        if float(start) < event_time_float <= float(stop)
                        and _hashable_group_value(row_stratum) != stratum_key
                    ],
                )
            )
        return result

    raise ValueError("Response must be right censored or (start, stop] data")


def _survobrien_formula_frame(
    formula: str,
    data: Any,
    *,
    subset: Any | None,
    na_action: Any | None,
    transform: Any | None,
) -> dict[str, list[Any]]:
    if data is None:
        raise ValueError("survobrien formula requires data")
    if subset is not None:
        data, _aligned = _subset_formula_inputs(formula, data, subset)
    data, _aligned = _apply_formula_na_action(formula, data, na_action)
    response, terms = _parse_formula(formula, data)
    if len(terms.clusters) > 1:
        raise ValueError("Can have only 1 cluster term")
    n = len(response)
    keepers, continuous = _survobrien_formula_terms(data, terms, n)
    strata_values = _combined_columns(data, terms.strata, n) if terms.strata else None
    event_sets = _survobrien_event_sets(response, strata_values)

    row_indices: list[int] = []
    set_numbers: list[int] = []
    event_times: list[float] = []
    for set_idx, (event_time, indices) in enumerate(event_sets, start=1):
        row_indices.extend(indices)
        set_numbers.extend([set_idx] * len(indices))
        event_times.extend([event_time] * len(indices))

    frame: dict[str, list[Any]] = {}
    if response.type == "counting":
        if response.start is None:
            raise ValueError("counting Surv response is missing start times")
        frame["start"] = [float(response.start[idx]) for idx in row_indices]
        frame["stop"] = [float(response.time[idx]) for idx in row_indices]
    else:
        frame["time"] = [float(response.time[idx]) for idx in row_indices]
    frame["status"] = [
        1 if response.event[idx] == 1 and float(response.time[idx]) == event_time else 0
        for idx, event_time in zip(row_indices, event_times, strict=True)
    ]
    for name, values in keepers:
        frame[name] = [values[idx] for idx in row_indices]
    if not terms.clusters:
        frame[".id."] = [idx + 1 for idx in row_indices]

    grouped_indices: list[list[int]] = []
    start = 0
    for _event_time, indices in event_sets:
        grouped_indices.append(list(range(start, start + len(indices))))
        start += len(indices)

    for name, values in continuous:
        output = [0.0] * len(row_indices)
        for positions in grouped_indices:
            transformed = _survobrien_transform_values(
                [values[row_indices[pos]] for pos in positions],
                transform,
            )
            for pos, value in zip(positions, transformed, strict=True):
                output[pos] = value
        frame[name] = output
    frame[".strata."] = set_numbers
    return frame


def survobrien(
    time: Any,
    status: Any | None = None,
    covariate: Any | None = None,
    strata: Any | None = None,
    *,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    transform: Any | None = None,
) -> SurvObrienResult | dict[str, list[Any]]:
    """Run O'Brien's direct statistic or build R-style formula transformed rows."""

    if isinstance(time, str) and "~" in time:
        formula_data = data if data is not None else status
        return _survobrien_formula_frame(
            time,
            formula_data,
            subset=subset,
            na_action=na_action,
            transform=transform,
        )
    if status is None or covariate is None:
        raise TypeError("direct survobrien calls require time, status, and covariate")

    time_values = _float_vector(time, "time")
    strata_groups = None
    if strata is not None:
        strata_values = _materialize_labels(strata, "strata")
        if len(strata_values) != len(time_values):
            raise ValueError("strata length mismatch")
        strata_groups = _encode_groups(strata_values, len(time_values))
    return _core.survobrien(
        time_values,
        _integer_code_vector(status, "status", "0/1 event coding"),
        _float_vector(covariate, "covariate"),
        strata_groups,
    )


def yates(
    predictions: Any,
    factor: Any,
    weights: Any | None = None,
    conf_level: Any | None = None,
) -> YatesResult:
    """Compute direct Yates-style adjusted means from predictions and a factor."""

    prediction_values = _float_vector(predictions, "predictions")
    factor_values = [str(value) for value in _materialize_labels(factor, "factor")]
    weight_values = None if weights is None else _float_vector(weights, "weights")
    confidence = None if conf_level is None else _normalize_conf_level(conf_level)
    return _core.yates(prediction_values, factor_values, weight_values, confidence)


def yates_contrast(
    x: Any,
    coef: Any,
    n_obs: Any,
    n_vars: Any,
    factor_col: Any,
    factor_levels: Any,
    predict_type: str | None = None,
) -> YatesResult:
    """Compute model-based direct Yates contrasts from a flattened design matrix."""

    return _core.yates_contrast(
        _float_vector(x, "x"),
        _float_vector(coef, "coef"),
        _integer_scalar(n_obs, "n_obs"),
        _integer_scalar(n_vars, "n_vars"),
        _integer_scalar(factor_col, "factor_col"),
        _float_vector(factor_levels, "factor_levels"),
        predict_type,
    )


def yates_pairwise(result: YatesResult) -> YatesPairwiseResult:
    """Compute pairwise differences from a direct Yates result."""

    return _core.yates_pairwise(result)


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


def _cipoisson_count(value: Any) -> int | None:
    if _is_missing_value(value):
        return None
    count = _integer_scalar(value, "k")
    if count < 0:
        raise ValueError("k must be non-negative")
    return count


def _cipoisson_float(value: Any, name: str) -> float | None:
    if _is_missing_value(value):
        return None
    return float(value)


def cipoisson(
    k: Any,
    time: Any = 1.0,
    p: Any = 0.95,
    method: Any = "exact",
) -> tuple[float, float] | list[tuple[float, float]]:
    """Return Poisson rate confidence intervals, like R's ``cipoisson``."""

    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method_value = method.strip().lower()
    k_values = _scalar_or_vector(k, "k")
    time_values = _scalar_or_vector(time, "time")
    p_values = _scalar_or_vector(p, "p")
    n = max(len(k_values), len(time_values), len(p_values))
    if n == 0:
        return []

    k_values = _recycle_r_vector(k_values, n, "k")
    time_values = _recycle_r_vector(time_values, n, "time")
    p_values = _recycle_r_vector(p_values, n, "p")
    if not k_values or not time_values or not p_values:
        return []

    intervals: list[tuple[float, float]] = []
    for raw_k, raw_time, raw_p in zip(k_values, time_values, p_values, strict=True):
        count = _cipoisson_count(raw_k)
        exposure = _cipoisson_float(raw_time, "time")
        confidence = _cipoisson_float(raw_p, "p")
        if count is None or exposure is None or confidence is None or exposure <= 0.0:
            intervals.append((math.nan, math.nan))
            continue
        lower, upper = _core.cipoisson(count, exposure, confidence, method_value)
        intervals.append((float(lower), float(upper)))

    return intervals[0] if n == 1 else intervals


def _bounded_link_transform(x: Any, edge: Any, method_name: str) -> float | list[float]:
    edge_value = _finite_float(edge, "edge")
    values, is_scalar = _scalar_or_vector_with_flag(x, "x")
    link = _core.LinkFunctionParams(edge_value)
    transform = getattr(link, method_name)
    result = [
        math.nan if _is_missing_value(value) else float(transform(float(value))) for value in values
    ]
    return result[0] if is_scalar else result


def blogit(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded logit link transform."""

    return _bounded_link_transform(x, edge, "blogit")


def bprobit(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded probit link transform."""

    return _bounded_link_transform(x, edge, "bprobit")


def bcloglog(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded complementary log-log link transform."""

    return _bounded_link_transform(x, edge, "bcloglog")


def blog(x: Any, edge: Any = 0.05) -> float | list[float]:
    """Return R survival's bounded log link transform."""

    return _bounded_link_transform(x, edge, "blog")


def survexp_us() -> RateTable:
    """Return the bundled US population mortality rate table."""

    return _core.survexp_us()


def survexp_mn() -> RateTable:
    """Return the bundled Minnesota population mortality rate table."""

    return _core.survexp_mn()


def survexp_usr() -> RateTable:
    """Return the rural US population mortality rate table alias."""

    return _core.survexp_usr()


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


def _survcheck_old_style_call(
    id_values: Any,
    time1: Any,
    time2: Any,
    status: Any,
    istate: Any | None,
):
    return _core.survcheck(
        _survcheck_integer_labels(id_values, "id"),
        _float_vector(time1, "time1"),
        _float_vector(time2, "time2"),
        _int_vector(status, "status"),
        None if istate is None else _survcheck_integer_labels(istate, "istate"),
    )


def _column_or_values(data: Any, values: Any, name: str) -> Any:
    if isinstance(values, str):
        if data is None:
            raise ValueError(f"{name} column lookup requires data")
        return _column(data, values)
    return values


def _survcheck_response_from_formula(
    formula: str,
    data: Any,
    subset: Any | None,
    na_action: str | None,
    id_values: Any | None,
    istate: Any | None,
) -> tuple[Surv, Any | None, Any | None]:
    if data is None:
        raise ValueError("survcheck formula requires data")
    id_values = _column_or_values(data, id_values, "id") if id_values is not None else None
    istate = _column_or_values(data, istate, "istate") if istate is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            id=id_values,
            istate=istate,
        )
        id_values = aligned["id"]
        istate = aligned["istate"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        id=id_values,
        istate=istate,
    )
    response, _terms = _parse_formula(formula, data)
    return response, aligned["id"], aligned["istate"]


def _survcondense_legacy_call(
    id_values: Any,
    time1: Any,
    time2: Any,
    status: Any,
) -> Any:
    return _core.survcondense(
        _int_vector(id_values, "id"),
        _float_vector(time1, "time1"),
        _float_vector(time2, "time2"),
        _int_vector(status, "status"),
    )


def _survcondense_term_name(term: _CovariateSpec) -> str:
    if isinstance(term, _InteractionTerm):
        return ":".join(_survcondense_term_name(factor) for factor in term.factors)
    if term.arithmetic is not None:
        return term.arithmetic
    if term.transform is not None:
        return f"{term.transform}({term.column})"
    if term.categorical:
        return f"factor({term.column})"
    return term.column


def _survcondense_strata_name(columns: Sequence[str]) -> str:
    return f"strata({', '.join(columns)})"


def _survcondense_strata_values(data: Any, columns: Sequence[str], n: int) -> list[Any]:
    values = [_column(data, column) for column in columns]
    if any(len(column) != n for column in values):
        raise ValueError("formula columns must have the same length as the Surv response")
    if len(values) == 1:
        return list(values[0])
    return [
        ", ".join(_strata_value_label(column[row_idx]) for column in values) for row_idx in range(n)
    ]


def _survcondense_model_columns(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> list[tuple[str, list[Any]]]:
    columns: list[tuple[str, list[Any]]] = []
    model_terms: Sequence[_FormulaModelTerm]
    model_terms = terms.model_terms or [_ModelCovariateTerm(term) for term in terms.covariates]
    for model_term in model_terms:
        if isinstance(model_term, _ModelCovariateTerm):
            columns.append(
                (
                    _survcondense_term_name(model_term.term),
                    _term_values(data, model_term.term, n),
                )
            )
        elif isinstance(model_term, _ModelStrataTerm):
            columns.append(
                (
                    _survcondense_strata_name(model_term.columns),
                    _survcondense_strata_values(data, model_term.columns, n),
                )
            )
        elif isinstance(model_term, _ModelOffsetTerm):
            name = f"offset({_survcondense_term_name(model_term.term)})"
            values = _numeric_term_values(
                _term_raw_values(data, model_term.term, n),
                model_term.term,
            )
            columns.append((name, values))
        else:
            continue
    return columns


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


def _survcondense_same_or_missing(left: Any, right: Any) -> bool:
    left_missing = _is_missing_value(left)
    right_missing = _is_missing_value(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return left == right


def _survcondense_order_key(value: Any) -> tuple[int, int, Any]:
    if _is_missing_value(value):
        return (1, 0, "")
    if isinstance(value, bool):
        return (0, 0, int(value))
    if isinstance(value, int | float):
        numeric = float(value)
        if math.isfinite(numeric):
            return (0, 0, numeric)
    return (0, 1, str(value))


def _survcondense_rle(values: Sequence[bool]) -> tuple[list[bool], list[int], list[int]]:
    if not values:
        return [], [], []

    run_values = [bool(values[0])]
    run_lengths = [1]
    for value in values[1:]:
        value = bool(value)
        if value == run_values[-1]:
            run_lengths[-1] += 1
        else:
            run_values.append(value)
            run_lengths.append(1)

    cumulative: list[int] = []
    total = 0
    for length in run_lengths:
        total += length
        cumulative.append(total)
    return run_values, run_lengths, cumulative


def _survcondense_adjust_starts_like_r(
    starts: list[float],
    order: Sequence[int],
    droprow: Sequence[bool],
) -> None:
    if not any(droprow):
        return

    run_values, run_lengths, cumulative = _survcondense_rle(droprow)
    del run_lengths
    if len(cumulative) == 2:
        starts[cumulative[0]] = starts[0]
        return
    if len(cumulative) == 3:
        starts[cumulative[1]] = starts[cumulative[0]]
        return

    run_start = 0
    for value, run_end_exclusive in zip(run_values, cumulative, strict=True):
        run_end = run_end_exclusive - 1
        if value and run_end + 1 < len(order):
            starts[order[run_end + 1]] = starts[order[run_start]]
        run_start = run_end_exclusive


def _survcondense_unique_columns(
    columns: Sequence[tuple[str, list[Any]]],
) -> list[tuple[str, list[Any]]]:
    seen: set[str] = set()
    unique: list[tuple[str, list[Any]]] = []
    for name, values in columns:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, values))
    return unique


def _survcondense_from_formula(
    formula: str,
    data: Any,
    subset: Any | None,
    na_action: str | None,
    id_values: Any,
    weights: Any | None,
    start: str,
    end: str,
    event: str,
    id_name: str | None = None,
    weights_name: str | None = None,
) -> dict[str, list[Any]]:
    if data is None:
        raise ValueError("survcondense formula requires data")
    if id_values is None:
        raise ValueError("survcondense requires an id argument")
    if not isinstance(start, str) or not start:
        raise ValueError("start must be a non-empty string")
    if not isinstance(end, str) or not end:
        raise ValueError("end must be a non-empty string")
    if not isinstance(event, str) or not event:
        raise ValueError("event must be a non-empty string")

    id_output_name = id_name or (id_values if isinstance(id_values, str) else "id")
    weights_output_name = weights_name or (weights if isinstance(weights, str) else "(weights)")
    id_values = _column_or_values(data, id_values, "id")
    weights = _column_or_values(data, weights, "weights") if weights is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            id=id_values,
            weights=weights,
        )
        id_values = aligned["id"]
        weights = aligned["weights"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        id=id_values,
        weights=weights,
    )
    id_values = _materialize_1d(aligned["id"], "id")
    weights = aligned["weights"]
    response, terms = _parse_formula(formula, data)
    _reject_formula_clusters("survcondense", terms)
    if response.type != "counting":
        raise ValueError("survcondense requires a counting-process Surv response")
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    if len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if weights is not None:
        weights = _materialize_1d(weights, "weights")
        if len(weights) != len(response):
            raise ValueError("weights must have the same length as the Surv response")

    model_columns = _survcondense_model_columns(data, terms, len(response))

    comparison_columns = [values for _name, values in model_columns]
    if weights is not None:
        comparison_columns.append(weights)
    comparison_columns.append(id_values)

    order = sorted(
        range(len(response)),
        key=lambda idx: (_survcondense_order_key(id_values[idx]), response.time[idx]),
    )
    droprow = [False] * len(response)
    if len(response) > 1:
        for sorted_idx, current_idx in enumerate(order[:-1]):
            next_idx = order[sorted_idx + 1]
            xdup = all(
                _survcondense_same_or_missing(values[next_idx], values[current_idx])
                for values in comparison_columns
            )
            ydup = response.start[next_idx] == response.time[current_idx]
            droprow[sorted_idx] = bool(xdup and ydup)

    starts = list(response.start)
    _survcondense_adjust_starts_like_r(starts, order, droprow)
    drop_indices = {order[pos] for pos, drop in enumerate(droprow) if drop}
    keep_indices = (
        [idx for idx in range(len(response)) if idx not in drop_indices] if any(droprow) else []
    )

    if weights is not None:
        model_columns.append((str(weights_output_name), weights))
    model_columns.append((str(id_output_name), id_values))

    output: dict[str, list[Any]] = {
        name: [values[idx] for idx in keep_indices]
        for name, values in _survcondense_unique_columns(model_columns)
    }
    output[start] = [starts[idx] for idx in keep_indices]
    output[end] = [response.time[idx] for idx in keep_indices]
    output[event] = [response.event[idx] for idx in keep_indices]
    return output


def survcondense(
    formula: Any,
    data: Any | None = None,
    subset: Any | None = None,
    weights: Any | None = None,
    na_action: Any | None = "pass",
    *,
    id: Any | None = None,  # noqa: A002
    start: str = "tstart",
    end: str = "tstop",
    event: str = "event",
    **kwargs: Any,
) -> Any:
    """Condense counting-process survival data, preserving the legacy vector API."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    id_name = kwargs.pop("_id_name", None)
    weights_name = kwargs.pop("_weights_name", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survcondense got unexpected keyword argument(s): {unexpected}")

    if not isinstance(formula, str):
        if data is None or subset is None or weights is None or id is not None:
            raise TypeError(
                "survcondense requires a formula/data/id call or legacy "
                "(id, time1, time2, status) vectors"
            )
        return _survcondense_legacy_call(formula, data, subset, weights)

    return _survcondense_from_formula(
        formula,
        data,
        subset,
        _normalize_na_action(na_action),
        id,
        weights,
        start,
        end,
        event,
        id_name,
        weights_name,
    )


def survcheck(
    response: Any = _MISSING,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "pass",
    id: Any | None = None,  # noqa: A002
    istate: Any | None = None,
    istate0: str = "(s0)",
    timefix: bool = True,
    *,
    time1: Any = _MISSING,
    time2: Any = _MISSING,
    status: Any = _MISSING,
    **kwargs: Any,
):
    """Check survival response consistency, like R's ``survcheck`` for common inputs."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survcheck got unexpected keyword argument(s): {unexpected}")

    if response is _MISSING:
        if time1 is _MISSING or time2 is _MISSING or status is _MISSING or id is None:
            raise TypeError(
                "survcheck requires a Surv response/formula or low-level "
                "id=, time1=, time2=, and status= vectors"
            )
        return _survcheck_old_style_call(id, time1, time2, status, istate)
    if time1 is not _MISSING or time2 is not _MISSING or status is not _MISSING:
        raise TypeError("time1, time2, and status are only valid for low-level survcheck calls")

    # Backward compatibility for the legacy low-level root call:
    # survcheck(id, time1, time2, status, istate=None).
    if not isinstance(response, Surv | str):
        if data is None or subset is None or na_action is None:
            raise TypeError(
                "survcheck requires a Surv response/formula or low-level "
                "(id, time1, time2, status) vectors"
            )
        return _survcheck_old_style_call(response, data, subset, na_action, id)

    if not isinstance(timefix, bool):
        raise TypeError("timefix must be True or False")
    if not isinstance(istate0, str):
        raise TypeError("istate0 must be a string")

    id_values = id
    if isinstance(response, str):
        response, id_values, istate = _survcheck_response_from_formula(
            response,
            data,
            subset,
            _normalize_na_action(na_action),
            id_values,
            istate,
        )
        subset = None
        na_action = "pass"
    elif subset is not None:
        indices = _subset_indices(subset, len(response))
        response = _subset_surv(response, indices)
        id_values = _subset_optional_sequence(id_values, indices, "id")
        istate = _subset_optional_sequence(istate, indices, "istate")
        subset = None

    response, aligned = _apply_surv_na_action(
        response,
        _normalize_na_action(na_action),
        "survcheck inputs",
        id=id_values,
        istate=istate,
    )
    id_values = aligned["id"]
    istate = aligned["istate"]

    if response.type == "right":
        times = list(response.time)
        if timefix:
            times = _survdiff_timefix_values(times, True)
        return _core.survcheck_simple(times, list(response.event))
    if response.type != "counting":
        raise ValueError(f"survcheck is not valid for {response.type} censored survival data")
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    if id_values is None:
        raise ValueError("an id argument is required")
    if len(_materialize_labels(id_values, "id")) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if istate is not None and len(_materialize_labels(istate, "istate")) != len(response):
        raise ValueError("istate must have the same length as the Surv response")

    start = list(response.start)
    stop = list(response.time)
    if timefix:
        start, stop = _timefix_vectors(start, stop)
    return _core.survcheck(
        _survcheck_integer_labels(id_values, "id"),
        start,
        stop,
        list(response.event),
        None if istate is None else _survcheck_integer_labels(istate, "istate"),
    )


def _rttright_original_order_weights(result: Any, n: int) -> list[float]:
    weights = [0.0] * n
    for sorted_pos, original_idx in enumerate(result.order):
        weights[int(original_idx)] = float(result.weights[sorted_pos])
    return weights


def _rttright_formula_groups(data: Any, terms: _FormulaTerms, n: int) -> list[int] | None:
    if not terms.strata and not terms.covariates:
        return None
    labels = _combined_formula_groups(data, terms.strata, terms.covariates, n)
    return _encode_labels(labels, "rttright strata")


def _rttright_response_from_formula(
    formula: str,
    data: Any,
    subset: Any | None,
    na_action: str | None,
    weights: Any | None,
    id: Any | None,  # noqa: A002
    *,
    warn_offset: bool = True,
) -> tuple[Surv, Any | None, Any | None, list[int] | None]:
    if data is None:
        raise ValueError("rttright formula requires data")
    weights = _column_or_values(data, weights, "weights") if weights is not None else None
    id_values = _column_or_values(data, id, "id") if id is not None else None
    if subset is not None:
        data, aligned = _subset_formula_inputs(
            formula,
            data,
            subset,
            weights=weights,
            id=id_values,
        )
        weights = aligned["weights"]
        id_values = aligned["id"]
    data, aligned = _apply_formula_na_action(
        formula,
        data,
        na_action,
        weights=weights,
        id=id_values,
    )
    response, terms = _parse_formula(formula, data)
    if terms.offsets and warn_offset:
        warnings.warn("Offset term ignored", RuntimeWarning, stacklevel=2)
    return (
        response,
        aligned["weights"],
        aligned["id"],
        _rttright_formula_groups(data, terms, len(response)),
    )


def _rttright_initial_weights(weights: Any | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    result = _float_vector(weights, "weights")
    if len(result) != n:
        raise ValueError("weights must have same length as time")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("weights must be finite")
    if any(value < 0.0 for value in result):
        raise ValueError("weights must be non-negative")
    return result


def _rttright_times_vector(times: Any) -> list[float]:
    if isinstance(times, str | bytes):
        raise TypeError("times must be numeric")
    try:
        result = [float(times)]
    except (TypeError, ValueError):
        result = _float_vector(times, "times")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("times must be finite")
    return result


def _rttright_validate_id(response: Surv, id: Any | None) -> list[Any] | None:  # noqa: A002
    if id is None:
        return None
    id_values = _materialize_labels(id, "id")
    if len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if response.type not in {"right", "counting"}:
        raise NotImplementedError("rttright id handling is currently supported only for right data")

    seen: set[Any] = set()
    for value in id_values:
        try:
            key = _hashable_group_value(value)
        except TypeError as exc:
            raise TypeError("id values must be hashable") from exc
        if response.type == "counting":
            continue
        if key in seen:
            raise ValueError("one or more flags are >0 in survcheck")
        seen.add(key)
    return id_values


def _rttright_divide(numerator: float, denominator: float) -> float:
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return math.nan
    return math.inf


def _rttright_apply_timefix(time: list[float], timefix: bool) -> list[float]:
    if not timefix or not time:
        return time
    return [float(value) for value in _core.aeq_surv(time, None).time]


def _rttright_counting_common_start(
    start: Sequence[float],
    id_values: Sequence[Any],
) -> bool:
    first_by_id: dict[Any, float] = {}
    for row_idx, id_value in enumerate(id_values):
        key = _hashable_group_value(id_value)
        first_by_id[key] = min(first_by_id.get(key, start[row_idx]), start[row_idx])
    if not first_by_id:
        return False
    first_start = next(iter(first_by_id.values()))
    return all(value == first_start for value in first_by_id.values())


def _rttright_counting_last_rows(
    stop: Sequence[float],
    id_values: Sequence[Any],
) -> list[bool]:
    last_by_id: dict[Any, tuple[float, int]] = {}
    for row_idx, id_value in enumerate(id_values):
        key = _hashable_group_value(id_value)
        candidate = (float(stop[row_idx]), row_idx)
        if key not in last_by_id or candidate > last_by_id[key]:
            last_by_id[key] = candidate
    return [
        row_idx == last_by_id[_hashable_group_value(id_value)][1]
        for row_idx, id_value in enumerate(id_values)
    ]


def _rttright_counting_validate_subject_weights(
    weights: Sequence[float],
    id_values: Sequence[Any],
) -> None:
    ranges: dict[Any, list[float]] = {}
    for weight, id_value in zip(weights, id_values, strict=True):
        key = _hashable_group_value(id_value)
        if key not in ranges:
            ranges[key] = [float(weight), float(weight)]
        else:
            ranges[key][0] = min(ranges[key][0], float(weight))
            ranges[key][1] = max(ranges[key][1], float(weight))
    if any(high > low for low, high in ranges.values()):
        raise ValueError("there are subjects with multiple weights")


def _rttright_counting_group_values(group: Sequence[int] | None, n: int) -> list[int]:
    if group is None:
        return [0] * n
    group_values = [int(value) for value in group]
    if len(group_values) != n:
        raise ValueError("rttright strata must have the same length as the Surv response")
    return group_values


def _rttright_counting_case_weights(
    weights: Any | None,
    id_values: Sequence[Any],
    group_values: Sequence[int],
    n: int,
    renorm: bool,
) -> list[float]:
    case_weights = _rttright_initial_weights(weights, n)
    _rttright_counting_validate_subject_weights(case_weights, id_values)
    if not renorm:
        return case_weights

    normalized = list(case_weights)
    group_indices: dict[int, list[int]] = {}
    for row_idx, group_value in enumerate(group_values):
        group_indices.setdefault(group_value, []).append(row_idx)

    for indices in group_indices.values():
        seen_ids: set[Any] = set()
        denominator = 0.0
        for row_idx in indices:
            key = _hashable_group_value(id_values[row_idx])
            if key in seen_ids:
                continue
            seen_ids.add(key)
            denominator += case_weights[row_idx]
        if denominator <= 0.0:
            raise ValueError("weights must have positive sum when renorm is true")
        for row_idx in indices:
            normalized[row_idx] = case_weights[row_idx] / denominator
    return normalized


def _rttright_counting_delta(
    start: Sequence[float],
    stop: Sequence[float],
    query_times: Sequence[float] | None,
) -> float:
    values = [*map(float, start), *map(float, stop)]
    if query_times is not None:
        values.extend(float(value) for value in query_times)
    unique = sorted(set(values))
    diffs = [right - left for left, right in zip(unique, unique[1:], strict=False) if right > left]
    if not diffs:
        raise NotImplementedError("function not defined for delayed entry or multistate data")
    return min(diffs) / 2.0


def _rttright_counting_km(
    start: Sequence[float],
    stop: Sequence[float],
    censor: Sequence[int],
    weights: Sequence[float],
) -> tuple[list[float], list[float]]:
    event_times = sorted({float(stop[idx]) for idx, value in enumerate(censor) if value == 1})
    survival_times: list[float] = []
    survival_values: list[float] = []
    current = 1.0
    for event_time in event_times:
        risk = sum(
            float(weight)
            for left, right, weight in zip(start, stop, weights, strict=True)
            if float(left) < event_time <= float(right)
        )
        events = sum(
            float(weights[idx])
            for idx, value in enumerate(censor)
            if value == 1 and float(stop[idx]) == event_time
        )
        if risk > 0.0:
            current *= 1.0 - events / risk
        survival_times.append(event_time)
        survival_values.append(current)
    return survival_times, survival_values


def _rttright_km_survival_at(
    survival_times: Sequence[float],
    survival_values: Sequence[float],
    time: float,
) -> float:
    index = bisect_left(survival_times, float(time))
    return 1.0 if index == 0 else float(survival_values[index - 1])


def _rttright_group_case_weights(
    weights: Sequence[float],
    indices: Sequence[int],
    renorm: bool,
) -> list[float]:
    group_weights = [float(weights[idx]) for idx in indices]
    if not renorm:
        return group_weights
    total = sum(group_weights)
    if total <= 0.0:
        raise ValueError("weights must have positive sum when renorm is true")
    return [weight / total for weight in group_weights]


def _rttright_group_time_matrix(
    time: Sequence[float],
    status: Sequence[int],
    weights: Sequence[float],
    query_times: Sequence[float],
) -> list[list[float]]:
    n = len(time)
    n_times = len(query_times)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda idx: (time[idx], idx))
    sorted_time = [float(time[idx]) for idx in order]
    sorted_status = [int(status[idx]) for idx in order]
    sorted_weights = [float(weights[idx]) for idx in order]

    event_g = [1.0] * n
    block_times: list[float] = []
    post_block_g: list[float] = []
    current_g = 1.0
    n_at_risk = sum(sorted_weights)

    start = 0
    while start < n:
        block_time = sorted_time[start]
        end = start + 1
        while end < n and sorted_time[end] == block_time:
            end += 1

        event_weight = 0.0
        censor_weight = 0.0
        for sorted_pos in range(start, end):
            local_idx = order[sorted_pos]
            event_g[local_idx] = current_g
            if sorted_status[sorted_pos] == 1:
                event_weight += sorted_weights[sorted_pos]
            else:
                censor_weight += sorted_weights[sorted_pos]

        risk_after_events = n_at_risk - event_weight
        if risk_after_events > 0.0 and censor_weight > 0.0:
            current_g *= 1.0 - censor_weight / risk_after_events
        n_at_risk = risk_after_events - censor_weight
        block_times.append(block_time)
        post_block_g.append(current_g)
        start = end

    query_g: list[float] = []
    for query_time in query_times:
        block_idx = bisect_left(block_times, query_time)
        query_g.append(1.0 if block_idx == 0 else post_block_g[block_idx - 1])

    matrix = [[0.0] * n_times for _ in range(n)]
    for row_idx, (row_time, row_status, row_weight) in enumerate(
        zip(time, status, weights, strict=True)
    ):
        for col_idx, g_at_time in enumerate(query_g):
            if row_status == 1:
                matrix[row_idx][col_idx] = _rttright_divide(
                    float(row_weight),
                    max(event_g[row_idx], g_at_time),
                )
            elif float(row_time) >= float(query_times[col_idx]):
                matrix[row_idx][col_idx] = _rttright_divide(float(row_weight), g_at_time)
    return matrix


def _rttright_time_matrix(
    time: Sequence[float],
    status: Sequence[int],
    weights: Any | None,
    times: Any,
    group: Sequence[int] | None,
    timefix: bool,
    renorm: bool,
) -> list[float] | list[list[float]]:
    time_values = _rttright_apply_timefix([float(value) for value in time], timefix)
    status_values = [int(value) for value in status]
    n = len(time_values)
    if len(status_values) != n:
        raise ValueError("time and status must have same length")
    if any(value not in (0, 1) for value in status_values):
        raise ValueError("status must contain only 0/1 values")
    if any(not math.isfinite(value) for value in time_values):
        raise ValueError("time must be finite")

    query_times = _rttright_times_vector(times)
    case_weights = _rttright_initial_weights(weights, n)
    matrix = [[0.0] * len(query_times) for _ in range(n)]

    if group is None:
        group_values = [0] * n
    else:
        group_values = [int(value) for value in group]
        if len(group_values) != n:
            raise ValueError("rttright strata must have the same length as the Surv response")

    group_indices: dict[int, list[int]] = {}
    for idx, group_value in enumerate(group_values):
        group_indices.setdefault(group_value, []).append(idx)

    for indices in group_indices.values():
        group_weights = _rttright_group_case_weights(case_weights, indices, renorm)
        group_matrix = _rttright_group_time_matrix(
            [time_values[idx] for idx in indices],
            [status_values[idx] for idx in indices],
            group_weights,
            query_times,
        )
        for local_idx, row_idx in enumerate(indices):
            matrix[row_idx] = group_matrix[local_idx]

    if len(query_times) == 1:
        return [row[0] for row in matrix]
    return matrix


def _rttright_counting_group_result(
    start: Sequence[float],
    stop: Sequence[float],
    status: Sequence[int],
    weights: Sequence[float],
    last: Sequence[bool],
    query_times: Sequence[float] | None,
    delta: float,
) -> list[float] | list[list[float]]:
    km_stop = [float(value) for value in stop]
    censor = [
        1 if is_last and int(event) == 0 else 0 for is_last, event in zip(last, status, strict=True)
    ]
    for row_idx, value in enumerate(censor):
        if value == 1:
            km_stop[row_idx] += delta
    survival_times, survival_values = _rttright_counting_km(start, km_stop, censor, weights)

    if query_times is None:
        result: list[float] = []
        for row_stop, row_status, row_weight, is_last in zip(
            stop,
            status,
            weights,
            last,
            strict=True,
        ):
            if is_last and int(row_status) > 0:
                gwt = _rttright_km_survival_at(survival_times, survival_values, float(row_stop))
                result.append(_rttright_divide(float(row_weight), gwt))
            else:
                result.append(0.0)
        return result

    matrix = [[0.0] * len(query_times) for _ in range(len(start))]
    gwt = [_rttright_km_survival_at(survival_times, survival_values, time) for time in query_times]
    gwt2 = [
        _rttright_km_survival_at(survival_times, survival_values, row_stop) for row_stop in stop
    ]
    for row_idx, (row_start, row_stop, _row_status, row_weight, is_last) in enumerate(
        zip(start, stop, status, weights, last, strict=True)
    ):
        for col_idx, query_time in enumerate(query_times):
            if float(row_start) < float(query_time) <= float(row_stop):
                matrix[row_idx][col_idx] = _rttright_divide(float(row_weight), gwt[col_idx])
        # R's timed counting branch tests the stop column, so final censored rows
        # also receive redistributed weights across all requested times.
        if is_last and float(row_stop) > 0.0:
            for col_idx, query_gwt in enumerate(gwt):
                matrix[row_idx][col_idx] = _rttright_divide(
                    float(row_weight),
                    max(gwt2[row_idx], query_gwt),
                )
    return matrix


def _rttright_counting_result(
    response: Surv,
    weights: Any | None,
    times: Any | None,
    group: Sequence[int] | None,
    id_values: Sequence[Any] | None,
    timefix: bool,
    renorm: bool,
) -> list[float] | list[list[float]]:
    if response.start is None:
        raise ValueError("counting Surv response is missing start times")
    if id_values is None:
        raise ValueError("id is required for start-stop data")

    id_labels = _materialize_labels(id_values, "id")
    n = len(response)
    if len(id_labels) != n:
        raise ValueError("id must have the same length as the Surv response")
    start = [float(value) for value in response.start]
    stop = [float(value) for value in response.time]
    status_values = [int(value) for value in response.event]
    if any(value not in (0, 1) for value in status_values):
        raise ValueError("rttright counting response must contain only 0/1 status values")
    if timefix:
        start, stop = _timefix_vectors(start, stop)

    check = _core.survcheck(
        _survcheck_integer_labels(id_labels, "id"),
        start,
        stop,
        status_values,
        None,
    )
    if any(flag > 0 for flag in check.flags):
        raise ValueError("one or more flags are >0 in survcheck")
    if not _rttright_counting_common_start(start, id_labels):
        raise NotImplementedError("function not defined for delayed entry or multistate data")

    last = _rttright_counting_last_rows(stop, id_labels)
    if (
        sum(1 for is_last, event in zip(last, status_values, strict=True) if is_last and event > 0)
        <= 1
    ):
        raise NotImplementedError("function not defined for delayed entry or multistate data")

    query_times = None if times is None else _rttright_times_vector(times)
    group_values = _rttright_counting_group_values(group, n)
    case_weights = _rttright_counting_case_weights(weights, id_labels, group_values, n, renorm)
    delta = _rttright_counting_delta(start, stop, query_times)

    matrix: list[list[float]] | None = None
    vector = [0.0] * n
    if query_times is not None:
        matrix = [[0.0] * len(query_times) for _ in range(n)]

    group_indices: dict[int, list[int]] = {}
    for row_idx, group_value in enumerate(group_values):
        group_indices.setdefault(group_value, []).append(row_idx)

    for indices in group_indices.values():
        group_result = _rttright_counting_group_result(
            [start[idx] for idx in indices],
            [stop[idx] for idx in indices],
            [status_values[idx] for idx in indices],
            [case_weights[idx] for idx in indices],
            [last[idx] for idx in indices],
            query_times,
            delta,
        )
        if query_times is None:
            for local_idx, row_idx in enumerate(indices):
                vector[row_idx] = float(group_result[local_idx])
        else:
            if matrix is None:
                raise RuntimeError("rttright counting matrix was not initialized")
            for local_idx, row_idx in enumerate(indices):
                matrix[row_idx] = [float(value) for value in group_result[local_idx]]

    if query_times is None:
        return vector
    if matrix is None:
        raise RuntimeError("rttright counting matrix was not initialized")
    if len(query_times) == 1:
        return [row[0] for row in matrix]
    return matrix


def _rttright_core_weights(
    response: Surv,
    weights: Any | None,
    group: Sequence[int] | None,
    timefix: bool,
    renorm: bool,
) -> list[float]:
    if group is not None:
        result = _core.rttright_stratified(
            list(response.time),
            list(response.event),
            list(group),
            None if weights is None else _float_vector(weights, "weights"),
            timefix,
            renorm,
        )
        return [float(weight) for weight in result.weights]
    result = _core.rttright(
        list(response.time),
        list(response.event),
        None if weights is None else _float_vector(weights, "weights"),
        timefix,
        renorm,
    )
    return _rttright_original_order_weights(result, len(response))


def _normalize_pseudo_type(value: Any | None) -> str:
    if value is None:
        return "survival"
    if not isinstance(value, str):
        raise TypeError("type must be a string or None")
    normalized = value.strip().lower()
    aliases = {
        "pstate": "survival",
        "survival": "survival",
        "cumhaz": "cumhaz",
        "chaz": "cumhaz",
        "rmst": "rmst",
        "rmts": "rmst",
        "auc": "rmst",
        "sojourn": "rmst",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "type must be 'pstate', 'survival', 'cumhaz', 'chaz', 'rmst', 'rmts', 'auc', "
            "or 'sojourn'"
        ) from exc


def _pseudo_eval_times(times: Any | None, eval_times: Any | None) -> list[float] | None:
    if times is not None and eval_times is not None:
        raise TypeError("use at most one of times or eval_times")
    values = times if times is not None else eval_times
    if values is None:
        return None
    if isinstance(values, str | bytes):
        raise TypeError("times must be numeric")
    try:
        result = [float(values)]
    except (TypeError, ValueError):
        result = _float_vector(values, "times")
    if any(not math.isfinite(value) for value in result):
        raise ValueError("times must be finite")
    return result


def _pseudo_model_response(fit: Any) -> Surv:
    model = getattr(fit, "model", None)
    if model is None:
        raise TypeError("pseudo requires a survfit result with a stored model frame")
    if not isinstance(model, Mapping):
        raise TypeError("stored survfit model frame must be mapping-like")
    responses = [value for value in model.values() if isinstance(value, Surv)]
    if len(responses) != 1:
        raise TypeError("stored survfit model frame must contain exactly one Surv response")
    return responses[0]


def _pseudo_group_values_from_model(model: Mapping[Any, Any], response: Surv) -> list[Any]:
    n = len(response)
    if "group" in model:
        group_values = _materialize_1d(model["group"], "group")
        if len(group_values) == n:
            return group_values

    response_columns: set[str] = {"time", "time1", "time2", "start", "stop", "event", "status"}
    for name, value in model.items():
        if value is response and isinstance(name, str) and "~" in name:
            response_columns.update(_formula_response_args(name))

    candidate_columns: list[list[Any]] = []
    for name, values in model.items():
        if values is response or (isinstance(name, str) and name.startswith("(")):
            continue
        if isinstance(name, str) and name in response_columns:
            continue
        try:
            column = _materialize_1d(values, str(name))
        except TypeError:
            continue
        if len(column) == n:
            candidate_columns.append(column)

    if not candidate_columns:
        raise TypeError("stored grouped survfit model frame does not contain grouping columns")
    return _combine_aligned_columns(candidate_columns, n)


def _pseudo_model_id_values(model: Mapping[Any, Any], response: Surv) -> list[Any] | None:
    values = model.get("(id)")
    if values is None:
        return None
    id_values = _materialize_labels(values, "id")
    if len(id_values) != len(response):
        raise TypeError("stored survfit model frame id does not match response length")
    return id_values


def _pseudo_model_weights(model: Mapping[Any, Any], response: Surv) -> list[float] | None:
    values = model.get("(weights)")
    if values is None:
        return None
    weights = _float_vector(values, "weights")
    if len(weights) != len(response):
        raise TypeError("stored survfit model frame weights do not match response length")
    return weights


def _pseudo_subset_model_frame(
    model: Mapping[Any, Any],
    response: Surv,
    indices: Sequence[int],
) -> dict[str, Any]:
    subset: dict[str, Any] = {"response": response}
    for name in ("(weights)", "(id)", "(cluster)"):
        values = model.get(name)
        if values is not None:
            materialized = _materialize_1d(values, name)
            subset[name] = [materialized[idx] for idx in indices]
    return subset


def _pseudo_rmst_values(
    curve_time: Sequence[float],
    curve_survival: Sequence[float],
    eval_times: Sequence[float],
) -> list[float]:
    result: list[float] = []
    times = [float(value) for value in curve_time]
    survival = [float(value) for value in curve_survival]
    for eval_time in eval_times:
        target = float(eval_time)
        area = 0.0
        previous_time = 0.0
        previous_survival = 1.0
        for time, estimate in zip(times, survival, strict=True):
            if target <= previous_time:
                break
            upper = min(target, time)
            if upper > previous_time:
                area += previous_survival * (upper - previous_time)
                previous_time = upper
            if time > target:
                break
            previous_survival = estimate
        if target > previous_time:
            area += previous_survival * (target - previous_time)
        result.append(area)
    return result


def _pseudo_curve_values(
    curve: SurvfitResult,
    eval_times: Sequence[float],
    pseudo_type: str,
) -> list[float]:
    times = [float(value) for value in curve.time]
    if pseudo_type == "survival":
        return _step_curve_at(times, [float(value) for value in curve.estimate], list(eval_times))
    if pseudo_type == "cumhaz":
        return _core.step_values_at(
            times,
            [float(value) for value in curve.cumhaz],
            [float(value) for value in eval_times],
            0.0,
        )
    return _pseudo_rmst_values(times, [float(value) for value in curve.estimate], eval_times)


def _pseudo_integrated_step_values(
    curve_time: Sequence[float],
    step_values: Sequence[float],
    eval_times: Sequence[float],
) -> list[float]:
    result: list[float] = []
    times = [float(value) for value in curve_time]
    values = [float(value) for value in step_values]
    for eval_time in eval_times:
        target = float(eval_time)
        area = 0.0
        previous_time = 0.0
        previous_value = 0.0
        for time, value in zip(times, values, strict=True):
            if target <= previous_time:
                break
            upper = min(target, time)
            if upper > previous_time:
                area += previous_value * (upper - previous_time)
                previous_time = upper
            if time > target:
                break
            previous_value = value
        if target > previous_time:
            area += previous_value * (target - previous_time)
        result.append(area)
    return result


def _pseudo_counting_candidate_cumhaz(fit: SurvfitResult, ctype: int) -> list[float]:
    hazard = 0.0
    cumhaz: list[float] = []
    event_counts = fit.n_event_count if fit.n_event_count is not None else fit.n_event
    for risk, events, event_count in zip(
        fit.n_risk,
        fit.n_event,
        event_counts,
        strict=True,
    ):
        risk_value = float(risk)
        event_value = float(events)
        event_count_value = float(event_count)
        if risk_value > 0.0 and event_value > 0.0 and event_count_value > 0.0:
            if ctype == 1:
                hazard += event_value / risk_value
            else:
                unweighted_events = int(round(event_count_value))
                if unweighted_events > 0:
                    event_step = event_value / unweighted_events
                    for step in range(unweighted_events):
                        denominator = risk_value - step * event_step
                        if denominator > 0.0:
                            hazard += event_step / denominator
        cumhaz.append(hazard)
    return cumhaz


def _pseudo_values_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-10,
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(left_value), float(right_value), rel_tol=rel_tol, abs_tol=abs_tol)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _pseudo_counting_computation(fit: SurvfitResult) -> _SurvfitComputation:
    ctype1_hazard = _pseudo_counting_candidate_cumhaz(fit, 1)
    ctype2_hazard = _pseudo_counting_candidate_cumhaz(fit, 2)
    ctype = 2 if _pseudo_values_close(fit.cumhaz, ctype2_hazard) else 1
    hazard = ctype2_hazard if ctype == 2 else ctype1_hazard
    fh_survival = [math.exp(-value) for value in hazard]
    stype = 2 if _pseudo_values_close(fit.estimate, fh_survival) else 1
    return _SurvfitComputation(stype, ctype)


def _pseudo_counting_residual_rows(
    influence: Any,
    fit: SurvfitResult,
    eval_times: Sequence[float],
    pseudo_type: str,
) -> list[list[float]]:
    times = [float(value) for value in fit.time]
    requested_times = [float(value) for value in eval_times]
    if pseudo_type == "survival":
        return [
            _core.step_values_at(times, [float(value) for value in row], requested_times, 0.0)
            for row in influence.influence_surv
        ]
    if pseudo_type == "cumhaz":
        return [
            _core.step_values_at(times, [float(value) for value in row], requested_times, 0.0)
            for row in influence.influence_chaz
        ]
    return [
        _pseudo_integrated_step_values(times, [float(value) for value in row], requested_times)
        for row in influence.influence_surv
    ]


def _pseudo_counting_survfit(
    fit: SurvfitResult,
    response: Surv,
    model: Mapping[Any, Any],
    eval_times: list[float] | None,
    pseudo_type: str,
    collapse: bool,
    data_frame: bool,
) -> Any:
    if eval_times is None:
        raise TypeError("times are required for counting-process pseudo-values")
    id_values = _pseudo_model_id_values(model, response)
    weights = _pseudo_model_weights(model, response)
    n_rows = len(response)
    if id_values is None:
        id_values = list(range(n_rows))
        subject_count = n_rows
    else:
        subject_count = len(_label_levels(id_values, "id"))
    if subject_count == 0:
        result = _PseudoMatrixResult([], [float(value) for value in eval_times])
        return _pseudo_matrix_or_frame(result, data_frame)

    full_values = _pseudo_curve_values(fit, eval_times, pseudo_type)
    computation = _pseudo_counting_computation(fit)
    start_values = [] if response.start is None else list(response.start)
    influence = survfitkm_counting_influence(
        start_values,
        list(response.time),
        [int(value) for value in response.event],
        [float(value) for value in fit.time],
        [float(value) for value in fit.estimate],
        cluster=id_values if collapse else list(range(n_rows)),
        weights=weights,
        stype=computation.stype,
        ctype=computation.ctype,
    )
    residual_rows = _pseudo_counting_residual_rows(influence, fit, eval_times, pseudo_type)
    scale = float(subject_count)
    pseudo_matrix: list[list[float]] = []
    for residuals in residual_rows:
        pseudo_matrix.append(
            [
                full_value + scale * residual
                for full_value, residual in zip(full_values, residuals, strict=True)
            ]
        )

    result = _PseudoMatrixResult(pseudo_matrix, [float(value) for value in eval_times])
    return _pseudo_matrix_or_frame(result, data_frame)


def _pseudo_for_grouped_survfit(
    fit: Mapping[Any, Any],
    eval_times: list[float] | None,
    pseudo_type: str,
    collapse: bool,
    data_frame: bool,
) -> Any:
    grouped_result: dict[Any, Any] = {}
    for label, curve in fit.items():
        response = _pseudo_model_response(curve)
        if response.type not in {"right", "counting"}:
            raise NotImplementedError(
                "pseudo currently supports right-censored or counting survfit results"
            )
        group_values = _pseudo_group_values_from_model(curve.model, response)
        indices = [idx for idx, value in enumerate(group_values) if value == label]
        if not indices:
            raise TypeError("stored grouped survfit model frame does not match curve labels")
        group_response = _subset_surv(response, indices)
        if response.type == "counting":
            group_model = _pseudo_subset_model_frame(curve.model, group_response, indices)
            grouped_result[label] = _pseudo_counting_survfit(
                curve,
                group_response,
                group_model,
                eval_times,
                pseudo_type,
                collapse,
                data_frame,
            )
            continue
        result = _core.pseudo(
            list(group_response.time),
            list(group_response.event),
            eval_times,
            pseudo_type,
        )
        grouped_result[label] = _pseudo_matrix_or_frame(result, data_frame)

    if not data_frame:
        return grouped_result

    frame: dict[str, list[Any]] = {"strata": [], "id": [], "time": [], "pseudo": []}
    for label, group_frame in grouped_result.items():
        row_count = len(group_frame["pseudo"])
        frame["strata"].extend([str(label)] * row_count)
        frame["id"].extend(group_frame["id"])
        frame["time"].extend(group_frame["time"])
        frame["pseudo"].extend(group_frame["pseudo"])
    return frame


def _pseudo_matrix_or_frame(result: Any, data_frame: bool) -> Any:
    matrix = [[float(value) for value in row] for row in result.pseudo]
    if not data_frame:
        return matrix
    frame: dict[str, list[float | int]] = {"id": [], "time": [], "pseudo": []}
    for row_idx, row in enumerate(matrix, start=1):
        for time, value in zip(result.time, row, strict=True):
            frame["id"].append(row_idx)
            frame["time"].append(float(time))
            frame["pseudo"].append(float(value))
    return frame


def pseudo(
    fit: Any = None,
    status: Any | None = None,
    eval_times: Any | None = None,
    type_: Any | None = None,
    *,
    times: Any | None = None,
    type: Any | None = None,  # noqa: A002
    collapse: bool = True,
    data_frame: bool = False,
    time: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Compute pseudo-values, preserving direct vector and R-style ``survfit`` calls."""

    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"pseudo got unexpected keyword argument(s): {unexpected}")
    if type is not None and type_ is not None:
        raise TypeError("use at most one of type or type_")
    pseudo_type = _normalize_pseudo_type(type if type is not None else type_)
    eval_time_values = _pseudo_eval_times(times, eval_times)

    if time is not None:
        if fit is not None:
            raise TypeError("time= cannot be combined with fit")
        fit = time
    if status is not None:
        return _core.pseudo(
            _float_vector(fit, "time"),
            _int_vector(status, "status"),
            eval_time_values,
            pseudo_type,
        )
    if fit is None:
        raise TypeError("pseudo requires a survfit result or time/status vectors")
    _normalize_bool_option(collapse, "collapse")
    data_frame = _normalize_bool_option(data_frame, "data_frame")
    if isinstance(fit, Mapping):
        return _pseudo_for_grouped_survfit(
            fit,
            eval_time_values,
            pseudo_type,
            _normalize_bool_option(collapse, "collapse"),
            data_frame,
        )

    response = _pseudo_model_response(fit)
    model = getattr(fit, "model", None)
    if response.type == "counting":
        if not isinstance(fit, SurvfitResult) or not isinstance(model, Mapping):
            raise TypeError(
                "counting-process pseudo-values require a survfit result with a stored model frame"
            )
        return _pseudo_counting_survfit(
            fit,
            response,
            model,
            eval_time_values,
            pseudo_type,
            _normalize_bool_option(collapse, "collapse"),
            data_frame,
        )
    if response.type != "right":
        raise NotImplementedError("pseudo currently supports right-censored survfit results")
    result = _core.pseudo(
        list(response.time),
        list(response.event),
        eval_time_values,
        pseudo_type,
    )
    return _pseudo_matrix_or_frame(result, data_frame)


def rttright(
    response: Any,
    status: Any | None = None,
    weights: Any | None = None,
    *,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "pass",
    times: Any | None = None,
    id: Any | None = None,  # noqa: A002
    timefix: bool = True,
    renorm: bool = True,
    **kwargs: Any,
) -> Any:
    """Redistribute censored mass to the right, like R's ``rttright``."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "pass")
    warn_offset = kwargs.pop("_warn_offset", True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"rttright got unexpected keyword argument(s): {unexpected}")
    if not isinstance(warn_offset, bool):
        raise TypeError("_warn_offset must be True or False")
    if not isinstance(timefix, bool):
        raise TypeError("timefix must be True or False")
    if not isinstance(renorm, bool):
        raise TypeError("renorm must be True or False")

    group: list[int] | None = None
    id_values = id
    if isinstance(response, str):
        surv_response, weights, id_values, group = _rttright_response_from_formula(
            response,
            data,
            subset,
            _normalize_na_action(na_action),
            weights,
            id,
            warn_offset=warn_offset,
        )
        response = surv_response
        subset = None
        na_action = "pass"
    elif isinstance(response, Surv):
        if subset is not None:
            indices = _subset_indices(subset, len(response))
            response = _subset_surv(response, indices)
            weights = _subset_optional_sequence(weights, indices, "weights")
            id_values = _subset_optional_sequence(id_values, indices, "id")
            subset = None
        response, aligned = _apply_surv_na_action(
            response,
            _normalize_na_action(na_action),
            "rttright inputs",
            weights=weights,
            id=id_values,
        )
        weights = aligned["weights"]
        id_values = aligned["id"]
    else:
        if status is None:
            raise TypeError("rttright direct-vector calls require status")
        time_values = _float_vector(response, "time")
        status_values = _int_vector(status, "status")
        if len(time_values) != len(status_values):
            raise ValueError("time and status must have same length")
        if any(value not in (0, 1) for value in status_values):
            raise ValueError("status must contain only 0/1 values")
        response_values = Surv(time_values, status_values)
        _rttright_validate_id(response_values, id_values)
        if times is not None:
            return _rttright_time_matrix(
                list(response_values.time),
                list(response_values.event),
                weights,
                times,
                None,
                timefix,
                renorm,
            )
        return _core.rttright(
            list(response_values.time),
            list(response_values.event),
            None if weights is None else _float_vector(weights, "weights"),
            timefix,
            renorm,
        )

    if not isinstance(response, Surv):
        raise TypeError("rttright response must be a Surv object, formula, or time vector")
    _rttright_validate_id(response, id_values)
    if response.type == "counting":
        return _rttright_counting_result(
            response,
            weights,
            times,
            group,
            id_values,
            timefix,
            renorm,
        )
    if response.type != "right":
        raise ValueError(f"rttright is not valid for {response.type} censored survival data")
    if times is not None:
        return _rttright_time_matrix(
            list(response.time),
            list(response.event),
            weights,
            times,
            group,
            timefix,
            renorm,
        )
    return _rttright_core_weights(
        response,
        weights,
        group,
        timefix,
        renorm,
    )


def _aeq_adjust_time_columns(
    columns: Sequence[Sequence[float]],
    tolerance: float | None,
) -> list[list[float]]:
    adjusted = [[float(value) for value in column] for column in columns]
    finite_values: list[float] = []
    finite_positions: list[tuple[int, int]] = []

    for col_idx, column in enumerate(adjusted):
        for row_idx, value in enumerate(column):
            if math.isfinite(value):
                finite_values.append(value)
                finite_positions.append((col_idx, row_idx))

    if not finite_values:
        return adjusted

    result = _core.aeq_surv(finite_values, tolerance)
    for (col_idx, row_idx), value in zip(finite_positions, result.time, strict=True):
        adjusted[col_idx][row_idx] = float(value)
    return adjusted


def _raise_if_aeq_zero_interval(
    original_left: Sequence[float],
    original_right: Sequence[float],
    adjusted_left: Sequence[float],
    adjusted_right: Sequence[float],
) -> None:
    for left, right, new_left, new_right in zip(
        original_left,
        original_right,
        adjusted_left,
        adjusted_right,
        strict=True,
    ):
        if left != right and new_left == new_right:
            raise ValueError("aeqSurv exception, an interval has effective length 0")


def aeqSurv(x: Any, tolerance: Any | None = None) -> Surv:  # noqa: N802
    """Adjudicate near-tied times in a ``Surv`` response, like R's ``aeqSurv``."""

    if not isinstance(x, Surv):
        raise TypeError("argument is not a Surv object")
    tolerance_value = None if tolerance is None else _finite_float(tolerance, "tolerance")
    if tolerance_value is not None and tolerance_value <= 0.0:
        return x

    if x.start is not None:
        start, stop = _aeq_adjust_time_columns((x.start, x.time), tolerance_value)
        _raise_if_aeq_zero_interval(x.start, x.time, start, stop)
        return Surv(start, stop, list(x.event), type="counting")

    if x.time2 is not None:
        left, right = _aeq_adjust_time_columns((x.time, x.time2), tolerance_value)
        _raise_if_aeq_zero_interval(x.time, x.time2, left, right)
        if x.type == "interval2":
            return Surv(left, right, type="interval2")
        return Surv(left, right, list(x.event), type=x.type)

    (time,) = _aeq_adjust_time_columns((x.time,), tolerance_value)
    return Surv(time, list(x.event), type=x.type)


def _survsplit_output_name(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a variable name")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a variable name")
    return stripped


def _survsplit_data_columns(data: Any | None, n: int) -> dict[str, list[Any]]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping of column names to values")
    columns: dict[str, list[Any]] = {}
    for key, values in data.items():
        column_name = str(key)
        column = _materialize_1d(values, column_name)
        if len(column) != n:
            raise ValueError(f"data column {column_name!r} must have length {n}")
        columns[column_name] = column
    return columns


def survSplit(  # noqa: N802
    response: Surv,
    data: Any | None = None,
    *,
    cut: Any,
    start: str = "tstart",
    end: str = "tstop",
    event: str = "event",
    episode: str | None = None,
    id: str | None = None,  # noqa: A002
    zero: Any = 0,
) -> dict[str, list[Any]]:
    """Split right or counting-process survival data at fixed cut points."""

    if not isinstance(response, Surv):
        raise TypeError("survSplit response must be a Surv object")
    if response.type not in {"right", "counting"}:
        raise ValueError(f"not valid for {response.type} censored survival data")

    cut_values = _float_vector(cut, "cut")
    if any(not math.isfinite(value) for value in cut_values):
        raise ValueError("cut must be a vector of finite numbers")
    start_name = _survsplit_output_name(start, "start")
    end_name = _survsplit_output_name(end, "end")
    event_name = _survsplit_output_name(event, "event")
    episode_name = _survsplit_output_name(episode, "episode") if episode is not None else None
    id_name = _survsplit_output_name(id, "id") if id is not None else None

    n = len(response)
    frame = _survsplit_data_columns(data, n)
    if id_name is not None and id_name in frame:
        raise ValueError("the suggested id name is already present")

    if response.start is None:
        zero_value = _finite_float(zero, "zero")
        stop_values = list(response.time)
        observed_times = [value for value in stop_values if not math.isnan(value)]
        if any(value <= zero_value for value in observed_times):
            raise ValueError("'zero' parameter must be less than any observed times")
        start_values = [zero_value] * n
    else:
        start_values = list(response.start)
        stop_values = list(response.time)

    split = _core.survsplit(start_values, stop_values, cut_values)
    row_indices = [int(row) - 1 for row in split.row]

    result: dict[str, list[Any]] = {
        name: [column[row_idx] for row_idx in row_indices] for name, column in frame.items()
    }
    if id_name is not None:
        result[id_name] = [row_idx + 1 for row_idx in row_indices]

    original_status = list(response.event)
    result[start_name] = [float(value) for value in split.start]
    result[end_name] = [float(value) for value in split.end]
    result[event_name] = [
        0 if censor else int(original_status[row_idx])
        for censor, row_idx in zip(split.censor, row_indices, strict=True)
    ]
    if episode_name is not None:
        result[episode_name] = [int(value) for value in split.interval]
    return result


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


def _r_output_length(*vectors: Sequence[float]) -> int:
    if any(len(vector) == 0 for vector in vectors):
        return 0
    return max((len(vector) for vector in vectors), default=0)


def _r_value(vector: Sequence[float], idx: int) -> float:
    return float(vector[idx % len(vector)])


def _r_unary(vector: Sequence[float], op: Any) -> list[float]:
    return [op(value) for value in vector]


def _r_binary(left: Sequence[float], right: Sequence[float], op: Any) -> list[float]:
    n = _r_output_length(left, right)
    return [op(_r_value(left, idx), _r_value(right, idx)) for idx in range(n)]


def _r_add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return _r_binary(left, right, lambda x, y: x + y)


def _r_sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return _r_binary(left, right, lambda x, y: x - y)


def _r_mul(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return _r_binary(left, right, lambda x, y: x * y)


def _r_div(left: Sequence[float], right: Sequence[float]) -> list[float]:
    def divide(x: float, y: float) -> float:
        try:
            return x / y
        except ZeroDivisionError:
            if x == 0.0:
                return math.nan
            return math.copysign(math.inf, x * y if y else x)

    return _r_binary(left, right, divide)


def _r_scalar_mul(vector: Sequence[float], scalar: float) -> list[float]:
    return [value * scalar for value in vector]


def _r_eq_zero(vector: Sequence[float]) -> list[bool | float]:
    return [math.nan if math.isnan(value) else value == 0.0 for value in vector]


def _r_eq_one(vector: Sequence[float]) -> list[bool | float]:
    return [math.nan if math.isnan(value) else value == 1.0 for value in vector]


def _r_or(left: Sequence[bool | float], right: Sequence[bool | float]) -> list[bool | float]:
    n = _r_output_length(left, right)
    result: list[bool | float] = []
    for idx in range(n):
        lval = left[idx % len(left)]
        rval = right[idx % len(right)]
        if lval is True or rval is True:
            result.append(True)
        elif lval is False and rval is False:
            result.append(False)
        else:
            result.append(math.nan)
    return result


def _r_ifelse(
    condition: Sequence[bool | float],
    yes: Sequence[float],
    no: Sequence[float],
) -> list[float]:
    if len(condition) == 0 or len(yes) == 0 or len(no) == 0:
        return []
    result: list[float] = []
    for idx, cond in enumerate(condition):
        if cond is True:
            result.append(_r_value(yes, idx))
        elif cond is False:
            result.append(_r_value(no, idx))
        else:
            result.append(math.nan)
    return result


def _r_log(value: float) -> float:
    if math.isnan(value) or value <= 0.0:
        return math.nan
    return math.log(value)


def _r_sqrt(value: float) -> float:
    if math.isnan(value) or value < 0.0:
        return math.nan
    return math.sqrt(value)


def _r_asin(value: float) -> float:
    if math.isnan(value) or value < -1.0 or value > 1.0:
        return math.nan
    return math.asin(value)


def _r_exp(value: float) -> float:
    return math.nan if math.isnan(value) else _safe_exp(value)


def _r_pmax_zero(values: Sequence[float]) -> list[float]:
    return [math.nan if math.isnan(value) else max(value, 0.0) for value in values]


def _r_pmin(values: Sequence[float], limit: float) -> list[float]:
    return [math.nan if math.isnan(value) else min(value, limit) for value in values]


def _survfit_confint_scale(se: list[float], selow: Any | None) -> list[float]:
    if selow is None:
        return [1.0]
    selow_values = _r_numeric_vector(selow, "selow")
    return _r_ifelse(
        _r_eq_zero(selow_values),
        [1.0],
        _r_div(selow_values, se),
    )


def _survfit_confint_prepared_se(p: list[float], se: list[float], logse: bool) -> list[float]:
    if logse:
        return se
    return _r_ifelse(_r_eq_zero(se), [0.0], _r_div(se, p))


def survfit_confint(
    p: Any,
    se: Any,
    logse: Any = True,
    conf_type: str | None = None,
    conf_int: Any = 0.95,
    selow: Any | None = None,
    ulimit: Any = True,
    **kwargs: Any,
) -> SurvfitConfidenceIntervalResult:
    """Return R ``survival::survfit_confint`` confidence bounds."""

    conf_type = _pop_dotted_keyword(kwargs, "conf.type", "conf_type", conf_type, None)
    conf_int = _pop_dotted_keyword(kwargs, "conf.int", "conf_int", conf_int, 0.95)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected survfit_confint argument(s): {unexpected}")
    if conf_type is None:
        raise TypeError("conf_type is required")
    if not isinstance(conf_type, str):
        raise TypeError("conf_type must be a string")
    if conf_type not in {"plain", "log", "log-log", "logit", "arcsin"}:
        raise ValueError("invalid conf.int type")
    if not _is_bool_like(logse):
        raise TypeError("logse must be True or False")
    if not _is_bool_like(ulimit):
        raise TypeError("ulimit must be True or False")

    p_values = _r_numeric_vector(p, "p")
    se_values = _r_numeric_vector(se, "se")
    confidence = _normalize_conf_level(conf_int, "conf_int")
    zval = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    scale = _survfit_confint_scale(se_values, selow)
    se_values = _survfit_confint_prepared_se(p_values, se_values, bool(logse))

    if conf_type == "plain":
        se2 = _r_scalar_mul(_r_mul(se_values, p_values), zval)
        lower = _r_pmax_zero(_r_sub(p_values, _r_mul(se2, scale)))
        upper_raw = _r_add(p_values, se2)
        upper = _r_pmin(upper_raw, 1.0) if bool(ulimit) else upper_raw
        return SurvfitConfidenceIntervalResult(lower=lower, upper=upper)

    if conf_type == "log":
        xx = _r_ifelse(_r_eq_zero(p_values), [math.nan], p_values)
        se2 = _r_scalar_mul(se_values, zval)
        log_xx = _r_unary(xx, _r_log)
        temp1 = _r_ifelse(
            _r_eq_zero(se_values),
            p_values,
            _r_unary(_r_sub(log_xx, _r_mul(se2, scale)), _r_exp),
        )
        temp2 = _r_ifelse(
            _r_eq_zero(se_values),
            p_values,
            _r_unary(_r_add(log_xx, se2), _r_exp),
        )
        upper = _r_pmin(temp2, 1.0) if bool(ulimit) else temp2
        return SurvfitConfidenceIntervalResult(lower=temp1, upper=upper)

    if conf_type == "log-log":
        xx = _r_ifelse(_r_or(_r_eq_zero(p_values), _r_eq_one(p_values)), [math.nan], p_values)
        log_xx = _r_unary(xx, _r_log)
        se2 = _r_scalar_mul(_r_div(se_values, log_xx), zval)
        log_neg_log_xx = _r_unary(_r_scalar_mul(log_xx, -1.0), _r_log)
        temp1 = _r_ifelse(
            _r_eq_zero(se_values),
            p_values,
            _r_unary(
                _r_scalar_mul(
                    _r_unary(_r_sub(log_neg_log_xx, _r_mul(se2, scale)), _r_exp),
                    -1.0,
                ),
                _r_exp,
            ),
        )
        temp2 = _r_ifelse(
            _r_eq_zero(se_values),
            p_values,
            _r_unary(
                _r_scalar_mul(_r_unary(_r_add(log_neg_log_xx, se2), _r_exp), -1.0),
                _r_exp,
            ),
        )
        return SurvfitConfidenceIntervalResult(lower=temp1, upper=temp2)

    if conf_type == "logit":
        xx = _r_ifelse(_r_eq_zero(p_values), [math.nan], p_values)
        one_minus_xx = _r_sub([1.0], xx)
        se2 = _r_scalar_mul(_r_mul(se_values, _r_add([1.0], _r_div(xx, one_minus_xx))), zval)
        logit_p = _r_unary(_r_div(p_values, _r_sub([1.0], p_values)), _r_log)
        temp1 = _r_ifelse(
            _r_eq_zero(se_values),
            p_values,
            _r_sub(
                [1.0],
                _r_div(
                    [1.0],
                    _r_add(
                        [1.0],
                        _r_unary(_r_sub(logit_p, _r_mul(se2, scale)), _r_exp),
                    ),
                ),
            ),
        )
        temp2 = _r_ifelse(
            _r_eq_zero(se_values),
            p_values,
            _r_sub(
                [1.0],
                _r_div([1.0], _r_add([1.0], _r_unary(_r_add(logit_p, se2), _r_exp))),
            ),
        )
        return SurvfitConfidenceIntervalResult(lower=temp1, upper=temp2)

    xx = _r_ifelse(_r_eq_zero(p_values), [math.nan], p_values)
    sqrt_xx = _r_unary(xx, _r_sqrt)
    se2 = _r_scalar_mul(
        _r_mul(se_values, _r_unary(_r_div(xx, _r_sub([1.0], xx)), _r_sqrt)),
        0.5 * zval,
    )
    asin_sqrt_xx = _r_unary(sqrt_xx, _r_asin)
    lower_angle = _r_pmax_zero(_r_sub(asin_sqrt_xx, _r_mul(se2, scale)))
    upper_angle = _r_pmin(_r_add(asin_sqrt_xx, se2), math.pi / 2.0)
    lower = [math.sin(value) ** 2 if not math.isnan(value) else math.nan for value in lower_angle]
    upper = [math.sin(value) ** 2 if not math.isnan(value) else math.nan for value in upper_angle]
    return SurvfitConfidenceIntervalResult(lower=lower, upper=upper)


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
    return _core.survfitkm(
        time,
        status,
        weights=weights,
        entry_times=entry_times,
        reverse=reverse,
        computation_type=0,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
    )


def survfitkm_influence(
    time: Any,
    status: Any,
    cluster: Any | None = None,
    *,
    weights: Any | None = None,
    reverse: Any = False,
    stype: Any = 1,
    ctype: Any = 1,
    conf_level: Any = 0.95,
    conf_type: Any = "log",
    timefix: Any = True,
) -> Any:
    """Return right-censored ``survfitKM`` influence matrices."""

    time_values = _float_vector(time, "time")
    status_values = _float_vector(status, "status")
    if len(status_values) != len(time_values):
        raise ValueError("status must have the same length as time")
    if cluster is None:
        cluster_values = list(range(len(time_values)))
    else:
        labels = _materialize_labels(cluster, "cluster")
        if len(labels) != len(time_values):
            raise ValueError("cluster must have the same length as time")
        cluster_values = _encode_labels(labels, "cluster")
    weight_values = None if weights is None else _float_vector(weights, "weights")
    if weight_values is not None and len(weight_values) != len(time_values):
        raise ValueError("weights must have the same length as time")
    return _core.survfitkm_influence(
        time_values,
        status_values,
        cluster_values,
        weights=weight_values,
        reverse=_normalize_bool_option(reverse, "reverse"),
        stype=_normalize_survfit_style(stype, "stype"),
        ctype=_normalize_survfit_style(ctype, "ctype"),
        conf_level=_normalize_conf_level(conf_level),
        conf_type=_normalize_survfit_conf_type(conf_type),
        timefix=_normalize_bool_option(timefix, "timefix"),
    )


def survfitkm_counting_influence(
    start: Any,
    stop: Any,
    status: Any,
    curve_time: Any,
    curve_estimate: Any,
    cluster: Any | None = None,
    *,
    weights: Any | None = None,
    reverse: Any = False,
    stype: Any = 1,
    ctype: Any = 1,
    conf_level: Any = 0.95,
    conf_type: Any = "log",
    timefix: Any = True,
) -> Any:
    """Return counting-process ``survfitKM`` influence matrices."""

    start_values = _float_vector(start, "start")
    stop_values = _float_vector(stop, "stop")
    status_values = _integer_code_vector(status, "status", "status")
    if len(stop_values) != len(start_values):
        raise ValueError("stop must have the same length as start")
    if len(status_values) != len(start_values):
        raise ValueError("status must have the same length as start")
    curve_time_values = _float_vector(curve_time, "curve_time")
    curve_estimate_values = _float_vector(curve_estimate, "curve_estimate")
    if len(curve_estimate_values) != len(curve_time_values):
        raise ValueError("curve_estimate must have the same length as curve_time")
    if cluster is None:
        cluster_values = list(range(len(start_values)))
    else:
        labels = _materialize_labels(cluster, "cluster")
        if len(labels) != len(start_values):
            raise ValueError("cluster must have the same length as start")
        cluster_values = _encode_labels(labels, "cluster")
    weight_values = None if weights is None else _float_vector(weights, "weights")
    if weight_values is not None and len(weight_values) != len(start_values):
        raise ValueError("weights must have the same length as start")
    return _core.survfitkm_counting_influence(
        start_values,
        stop_values,
        status_values,
        curve_time_values,
        curve_estimate_values,
        cluster_values,
        weights=weight_values,
        reverse=_normalize_bool_option(reverse, "reverse"),
        stype=_normalize_survfit_style(stype, "stype"),
        ctype=_normalize_survfit_style(ctype, "ctype"),
        conf_level=_normalize_conf_level(conf_level),
        conf_type=_normalize_survfit_conf_type(conf_type),
        timefix=_normalize_bool_option(timefix, "timefix"),
    )


def _survfit_residual_type(value: str) -> str:
    choices = ("pstate", "cumhaz", "sojourn", "survival", "chaz", "rmst", "rmts", "auc")
    normalized = value.casefold().replace("-", "_")
    if normalized in choices:
        matched = normalized
    else:
        matches = [choice for choice in choices if choice.startswith(normalized)]
        if len(matches) != 1:
            raise ValueError(
                "type must be one of pstate, cumhaz, sojourn, survival, chaz, rmst, rmts, or auc"
            )
        matched = matches[0]
    if matched in {"pstate", "survival"}:
        return "pstate"
    if matched in {"cumhaz", "chaz"}:
        return "cumhaz"
    return "auc"


def _survfit_residual_times(times: Any | None) -> list[float]:
    if times is None:
        raise TypeError("the times argument is required")
    if isinstance(times, int | float) and not isinstance(times, bool):
        return [float(times)]
    return sorted(set(_float_vector(times, "times")))


def _survfit_residual_model_frame(fit: Any) -> Mapping[Any, Any]:
    if isinstance(fit, Mapping):
        if not fit:
            raise TypeError("residuals.survfit requires a non-empty survfit result")
        return _survfit_residual_model_frame(next(iter(fit.values())))
    frame = getattr(fit, "model", None)
    if frame is None:
        raise TypeError("residuals.survfit requires a survfit result with a stored model frame")
    if not isinstance(frame, Mapping):
        raise TypeError("stored survfit model frame must be mapping-like")
    return frame


def _survfit_residual_response(frame: Mapping[Any, Any]) -> Surv:
    for value in frame.values():
        if isinstance(value, Surv):
            if value.type not in {"right", "counting"}:
                raise NotImplementedError(
                    "residuals.survfit currently supports right-censored or counting "
                    "Kaplan-Meier fits"
                )
            return value
    raise TypeError("stored survfit model frame does not contain a Surv response")


def _survfit_residual_weights(
    frame: Mapping[Any, Any],
    n: int,
) -> tuple[list[float], bool]:
    if "(weights)" not in frame:
        return [1.0] * n, False
    weights = _float_vector(frame["(weights)"], "weights")
    if len(weights) != n:
        raise ValueError("weights must have the same length as the Surv response")
    return weights, True


def _survfit_residual_ids(frame: Mapping[Any, Any], n: int) -> tuple[list[Any], str | None, bool]:
    if "(id)" not in frame:
        return list(range(1, n + 1)), None, False
    values = _materialize_labels(frame["(id)"], "id")
    if len(values) != n:
        raise ValueError("id must have the same length as the Surv response")
    return values, "(id)", True


def _survfit_residual_group_values(frame: Mapping[Any, Any], n: int) -> list[Any] | None:
    if "group" in frame:
        values = _materialize_labels(frame["group"], "group")
        if len(values) != n:
            raise ValueError("group must have the same length as the Surv response")
        return values

    columns: list[list[Any]] = []
    for name, values in frame.items():
        text_name = str(name)
        if isinstance(values, (Surv, Mapping)) or text_name.startswith("("):
            continue
        if text_name in {"response", "time", "status", "start", "stop", "time2"}:
            continue
        materialized = _materialize_1d(values, text_name)
        if len(materialized) != n:
            continue
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns.append(materialized)

    if not columns:
        return None
    return _combine_aligned_columns(columns, n)


def _survfit_residual_grouped_indices(
    fit: Mapping[Any, Any],
    group_values: list[Any] | None,
    n: int,
) -> list[tuple[Any, Any, list[int], int]]:
    if group_values is None:
        raise TypeError("grouped survfit residuals require stored grouping columns")
    grouped = _group_indices(group_values, n)
    ordered_groups = list(grouped.items())
    curves: list[tuple[Any, Any, list[int], int]] = []
    for curve_idx, (label, curve) in enumerate(fit.items(), start=1):
        indices = grouped.get(label)
        if indices is None:
            matching = [values for key, values in grouped.items() if str(key) == str(label)]
            if len(matching) == 1:
                indices = matching[0]
        if indices is None and curve_idx <= len(ordered_groups):
            indices = ordered_groups[curve_idx - 1][1]
        if indices is None:
            raise ValueError(f"could not match grouped survfit curve {label!r} to model rows")
        curves.append((label, curve, indices, curve_idx))
    return curves


def _survfit_residual_curve_specs(
    fit: Any,
    frame: Mapping[Any, Any],
    n: int,
) -> list[tuple[Any, Any, list[int], int]]:
    if isinstance(fit, Mapping):
        return _survfit_residual_grouped_indices(
            fit,
            _survfit_residual_group_values(frame, n),
            n,
        )
    return [(None, fit, list(range(n)), 1)]


def _survfit_residual_rows_at_times(
    influence: Any,
    times: Sequence[float],
    residual_type: str,
) -> list[list[float]]:
    curve_times = [float(value) for value in influence.time]
    eval_times = [float(value) for value in times]
    if residual_type == "cumhaz":
        return [
            _core.step_values_at(curve_times, [float(value) for value in row], eval_times, 0.0)
            for row in influence.influence_chaz
        ]
    if residual_type == "auc":
        return [
            _pseudo_integrated_step_values(curve_times, [float(value) for value in row], eval_times)
            for row in influence.influence_surv
        ]
    return [
        _core.step_values_at(curve_times, [float(value) for value in row], eval_times, 0.0)
        for row in influence.influence_surv
    ]


def _survfit_residual_matrix(
    response: Surv,
    weights: list[float],
    curve_specs: list[tuple[Any, Any, list[int], int]],
    times: Sequence[float],
    residual_type: str,
) -> tuple[list[list[float]], list[int] | None]:
    matrix = [[0.0 for _ in times] for _ in range(len(response))]
    curve_numbers = [0 for _ in range(len(response))] if len(curve_specs) > 1 else None
    for _label, curve, indices, curve_idx in curve_specs:
        if not isinstance(curve, SurvfitResult):
            raise TypeError("residuals.survfit currently supports Kaplan-Meier survfit results")
        group_response = _subset_surv(response, indices)
        if group_response.type == "counting":
            if group_response.start is None:
                raise ValueError("counting-process Surv response is missing start times")
            group_weights = [weights[idx] for idx in indices]
            computation = _pseudo_counting_computation(curve)
            influence = survfitkm_counting_influence(
                list(group_response.start),
                list(group_response.time),
                [int(value) for value in group_response.event],
                [float(value) for value in curve.time],
                [float(value) for value in curve.estimate],
                list(range(len(indices))),
                weights=group_weights,
                stype=computation.stype,
                ctype=computation.ctype,
            )
        elif group_response.type == "right" and group_response.start is None:
            group_weights = [weights[idx] for idx in indices]
            computation = _pseudo_counting_computation(curve)
            influence = survfitkm_influence(
                list(group_response.time),
                [int(value) for value in group_response.event],
                list(range(len(indices))),
                weights=group_weights,
                stype=computation.stype,
                ctype=computation.ctype,
            )
        else:
            raise NotImplementedError(
                "residuals.survfit currently supports right-censored or counting Kaplan-Meier fits"
            )
        rows = _survfit_residual_rows_at_times(influence, times, residual_type)
        for local_idx, source_idx in enumerate(indices):
            matrix[source_idx] = rows[local_idx]
            if curve_numbers is not None:
                curve_numbers[source_idx] = curve_idx
    return matrix, curve_numbers


def _collapse_survfit_residual_matrix(
    matrix: list[list[float]],
    ids: list[Any],
    weights: list[float],
    curve_numbers: list[int] | None,
) -> tuple[list[list[float]], list[Any], list[int] | None]:
    levels = _label_levels(ids, "id")
    index_by_id = {value: idx for idx, value in enumerate(levels)}
    collapsed = [[0.0 for _ in matrix[0]] for _ in levels] if matrix else [[] for _ in levels]
    collapsed_curve = [0 for _ in levels] if curve_numbers is not None else None
    for row_idx, id_value in enumerate(ids):
        target = index_by_id[id_value]
        collapsed[target] = [
            current + float(weights[row_idx]) * float(value)
            for current, value in zip(collapsed[target], matrix[row_idx], strict=True)
        ]
        if collapsed_curve is not None and collapsed_curve[target] == 0:
            collapsed_curve[target] = curve_numbers[row_idx]
    return collapsed, list(levels), collapsed_curve


def _weight_survfit_residual_matrix(
    matrix: list[list[float]],
    weights: list[float],
) -> list[list[float]]:
    return [
        [float(weight) * float(value) for value in row]
        for row, weight in zip(matrix, weights, strict=True)
    ]


def survfit_residuals(
    fit: Any,
    times: Any | None = None,
    *,
    type: str = "pstate",  # noqa: A002
    collapse: Any = False,
    weighted: Any = False,
    data_frame: Any = False,
    extra: Any = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return R-style ``residuals.survfit`` influence residuals for KM curves."""

    data_frame = _pop_dotted_keyword(kwargs, "data.frame", "data_frame", data_frame, False)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survfit_residuals got unexpected keyword argument(s): {unexpected}")
    if isinstance(fit, CoxSurvfitResult):
        raise TypeError("residuals method for coxph survival curve not found")
    if isinstance(fit, TurnbullSurvfitResult):
        raise NotImplementedError("residuals for interval-censored data are not available")

    eval_times = _survfit_residual_times(times)
    residual_type = _survfit_residual_type(type)
    collapse_value = _normalize_bool_option(collapse, "collapse")
    weighted_value = _normalize_bool_option(weighted, "weighted")
    data_frame_value = _normalize_bool_option(data_frame, "data_frame")
    extra_value = _normalize_bool_option(extra, "extra")
    if collapse_value and not weighted_value:
        raise ValueError("invalid combination of options: collapse=True and weighted=False")

    frame = _survfit_residual_model_frame(fit)
    response = _survfit_residual_response(frame)
    n = len(response)
    if n == 0:
        raise ValueError("data set has no non-missing observations")
    weights_values, has_case_weights = _survfit_residual_weights(frame, n)
    id_values, id_name, has_id = _survfit_residual_ids(frame, n)
    if not has_case_weights:
        weighted_value = False
    if not has_id or len(_label_levels(id_values, "id")) == n:
        collapse_value = False

    matrix, curve_numbers = _survfit_residual_matrix(
        response,
        weights_values,
        _survfit_residual_curve_specs(fit, frame, n),
        eval_times,
        residual_type,
    )

    if collapse_value:
        matrix, id_values, curve_numbers = _collapse_survfit_residual_matrix(
            matrix,
            id_values,
            weights_values,
            curve_numbers,
        )
    elif weighted_value and any(weight != 1.0 for weight in weights_values):
        matrix = _weight_survfit_residual_matrix(matrix, weights_values)

    return {
        "resid": matrix,
        "id": id_values,
        "id_name": id_name,
        "time": eval_times,
        "curve": curve_numbers,
        "data_frame": data_frame_value,
        "extra": extra_value,
    }


def _survfit_from_km_counts(
    km: Any,
    conf_level: float,
    computation: _SurvfitComputation,
    conf_type: str,
) -> SurvfitResult:
    curve = _core.survfit_curve_from_tables(
        [float(value) for value in km.time],
        [float(value) for value in km.n_risk],
        [float(value) for value in km.n_event],
        [float(value) for value in km.n_event_count],
        [float(value) for value in km.n_censor],
        [float(value) for value in km.n_censor_count],
        None if getattr(km, "n_enter", None) is None else [float(value) for value in km.n_enter],
        False,
        computation.stype,
        computation.ctype,
        conf_level,
        conf_type,
    )

    return SurvfitResult(
        time=[float(value) for value in curve.time],
        n_risk=[float(value) for value in curve.n_risk],
        n_event=[float(value) for value in curve.n_event],
        n_censor=[float(value) for value in curve.n_censor],
        estimate=[float(value) for value in curve.estimate],
        std_err=[float(value) for value in curve.std_err],
        conf_lower=[float(value) for value in curve.conf_lower],
        conf_upper=[float(value) for value in curve.conf_upper],
        cumhaz=[float(value) for value in curve.cumhaz],
        std_chaz=[float(value) for value in curve.std_chaz],
        n_enter=(
            [float(value) for value in curve.n_enter]
            if getattr(curve, "n_enter", None) is not None
            else None
        ),
        n_risk_count=(
            [float(value) for value in km.n_risk_count]
            if getattr(km, "n_risk_count", None) is not None
            else None
        ),
        n_event_count=(
            [float(value) for value in km.n_event_count]
            if getattr(km, "n_event_count", None) is not None
            else None
        ),
        n_censor_count=(
            [float(value) for value in km.n_censor_count]
            if getattr(km, "n_censor_count", None) is not None
            else None
        ),
        n_enter_count=(
            [float(value) for value in km.n_enter_count]
            if getattr(km, "n_enter_count", None) is not None
            else None
        ),
    )


def _survfit_from_count_tables(
    times: list[float],
    n_risk: list[float],
    n_event: list[float],
    n_event_count: list[float],
    n_censor: list[float],
    n_censor_count: list[float],
    n_enter: list[float] | None,
    n_risk_count: list[float] | None = None,
    n_enter_count: list[float] | None = None,
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
) -> SurvfitResult:
    curve = _core.survfit_curve_from_tables(
        times,
        n_risk,
        n_event,
        n_event_count,
        n_censor,
        n_censor_count,
        n_enter,
        reverse,
        computation.stype,
        computation.ctype,
        conf_level,
        conf_type,
    )

    return SurvfitResult(
        time=[float(value) for value in curve.time],
        n_risk=[float(value) for value in curve.n_risk],
        n_event=[float(value) for value in curve.n_event],
        n_censor=[float(value) for value in curve.n_censor],
        estimate=[float(value) for value in curve.estimate],
        std_err=[float(value) for value in curve.std_err],
        conf_lower=[float(value) for value in curve.conf_lower],
        conf_upper=[float(value) for value in curve.conf_upper],
        cumhaz=[float(value) for value in curve.cumhaz],
        std_chaz=[float(value) for value in curve.std_chaz],
        n_enter=(
            [float(value) for value in curve.n_enter]
            if getattr(curve, "n_enter", None) is not None
            else None
        ),
        n_risk_count=(None if n_risk_count is None else [float(value) for value in n_risk_count]),
        n_event_count=[float(value) for value in n_event_count],
        n_censor_count=[float(value) for value in n_censor_count],
        n_enter_count=(
            None if n_enter_count is None else [float(value) for value in n_enter_count]
        ),
    )


def _survfit_cluster_values(cluster: Any, n: int) -> list[Any]:
    values = _materialize_labels(cluster, "cluster")
    if len(values) != n:
        raise ValueError("cluster must have the same length as the Surv response")
    _label_levels(values, "cluster")
    return values


def _survfit_robust_cluster_values(
    response: Surv,
    cluster: Any | None,
    id_values: list[Any] | None,
    weights: list[float] | None,
    robust: bool | None,
) -> list[Any] | None:
    if robust is False:
        if cluster is not None:
            warnings.warn(
                "cluster specified with robust=False; cluster will be ignored",
                RuntimeWarning,
                stacklevel=3,
            )
        return None
    if cluster is not None:
        return _survfit_cluster_values(cluster, len(response))
    if robust is not True:
        if weights is not None and any(not float(weight).is_integer() for weight in weights):
            return list(range(len(response)))
        return None
    if id_values is not None:
        return _survfit_cluster_values(id_values, len(response))
    if response.start is not None:
        raise NotImplementedError(
            "survfit robust variance for counting-process data requires cluster or id"
        )
    return list(range(len(response)))


def _survfit_robust_km_result(
    result: Any,
    response: Surv,
    weights: list[float] | None,
    cluster_values: list[Any],
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    timefix: bool,
) -> SurvfitResult:
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survfit robust variance is currently supported only for right-censored or "
            "counting-process Kaplan-Meier curves"
        )
    if len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")

    if response.start is not None:
        std_err, std_chaz, conf_lower, conf_upper = _core.robust_counting_survfit_variance(
            list(response.start),
            list(response.time),
            [int(value) for value in response.event],
            [float(value) for value in result.time],
            [float(value) for value in result.estimate],
            _encode_labels(cluster_values, "cluster"),
            weights=weights,
            reverse=reverse,
            conf_level=conf_level,
            conf_type=conf_type,
            timefix=timefix,
        )
        return SurvfitResult(
            time=[float(value) for value in result.time],
            n_risk=[float(value) for value in result.n_risk],
            n_event=[float(value) for value in result.n_event],
            n_censor=[float(value) for value in result.n_censor],
            estimate=[float(value) for value in result.estimate],
            std_err=[float(value) for value in std_err],
            conf_lower=[float(value) for value in conf_lower],
            conf_upper=[float(value) for value in conf_upper],
            cumhaz=[float(value) for value in result.cumhaz],
            std_chaz=[float(value) for value in std_chaz],
            n_enter=(
                [float(value) for value in result.n_enter]
                if getattr(result, "n_enter", None) is not None
                else None
            ),
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
            model=getattr(result, "model", None),
        )

    robust = _core.robust_survfitkm(
        list(response.time),
        list(response.event),
        _encode_labels(cluster_values, "cluster"),
        weights=weights,
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
    )
    return SurvfitResult(
        time=[float(value) for value in robust.time],
        n_risk=[float(value) for value in robust.n_risk],
        n_event=[float(value) for value in robust.n_event],
        n_censor=[float(value) for value in robust.n_censor],
        estimate=[float(value) for value in robust.estimate],
        std_err=[float(value) for value in robust.std_err],
        conf_lower=[float(value) for value in robust.conf_lower],
        conf_upper=[float(value) for value in robust.conf_upper],
        cumhaz=[float(value) for value in robust.cumhaz],
        std_chaz=[float(value) for value in robust.std_chaz],
        n_enter=None,
        n_risk_count=_optional_float_list(robust, "n_risk_count"),
        n_event_count=_optional_float_list(robust, "n_event_count"),
        n_censor_count=_optional_float_list(robust, "n_censor_count"),
        n_enter_count=_optional_float_list(robust, "n_enter_count"),
        model=getattr(result, "model", None),
    )


def _matrix_column_norms(matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return []
    width = len(matrix[0])
    return [
        math.sqrt(sum(float(row[col]) * float(row[col]) for row in matrix)) for col in range(width)
    ]


def _survfit_robust_right_result(
    result: SurvfitResult,
    response: Surv,
    weights: list[float] | None,
    cluster_values: list[Any],
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
    timefix: bool,
) -> SurvfitResult:
    if response.start is not None or response.type != "right":
        raise NotImplementedError(
            "survfit robust variance for non-Kaplan-Meier curves is currently supported only "
            "for right-censored data"
        )
    if len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")

    influence = survfitkm_influence(
        list(response.time),
        list(response.event),
        cluster_values,
        weights=weights,
        reverse=reverse,
        stype=computation.stype,
        ctype=computation.ctype,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
    )
    std_err = _matrix_column_norms(influence.influence_surv)
    std_chaz = _matrix_column_norms(influence.influence_chaz)
    if conf_type == "none":
        conf_lower: list[float] = []
        conf_upper: list[float] = []
    else:
        alpha = 1.0 - conf_level
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        intervals = [
            _survfit_confidence_interval(estimate, se, z, conf_type)
            for estimate, se in zip(result.estimate, std_err, strict=True)
        ]
        conf_lower = [lower for lower, _upper in intervals]
        conf_upper = [upper for _lower, upper in intervals]

    return SurvfitResult(
        time=result.time,
        n_risk=result.n_risk,
        n_event=result.n_event,
        n_censor=result.n_censor,
        estimate=result.estimate,
        std_err=std_err,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        cumhaz=result.cumhaz,
        std_chaz=std_chaz,
        n_enter=result.n_enter,
        n_risk_count=result.n_risk_count,
        n_event_count=result.n_event_count,
        n_censor_count=result.n_censor_count,
        n_enter_count=result.n_enter_count,
        model=result.model,
    )


def _survfit_robust_counting_result(
    result: SurvfitResult,
    response: Surv,
    weights: list[float] | None,
    cluster_values: list[Any],
    *,
    reverse: bool,
    conf_level: float,
    conf_type: str,
    computation: _SurvfitComputation,
    timefix: bool,
) -> SurvfitResult:
    if response.start is None or response.type != "counting":
        raise NotImplementedError(
            "survfit robust variance for counting-process curves requires counting-process data"
        )
    if len(cluster_values) != len(response):
        raise ValueError("cluster must have the same length as the Surv response")

    std_err, std_chaz, conf_lower, conf_upper = _core.robust_counting_survfit_variance(
        list(response.start),
        list(response.time),
        [int(value) for value in response.event],
        [float(value) for value in result.time],
        [float(value) for value in result.estimate],
        _encode_labels(cluster_values, "cluster"),
        weights=weights,
        reverse=reverse,
        conf_level=conf_level,
        conf_type=conf_type,
        timefix=timefix,
        stype=computation.stype,
        ctype=computation.ctype,
    )

    return SurvfitResult(
        time=result.time,
        n_risk=result.n_risk,
        n_event=result.n_event,
        n_censor=result.n_censor,
        estimate=result.estimate,
        std_err=[float(value) for value in std_err],
        conf_lower=[float(value) for value in conf_lower],
        conf_upper=[float(value) for value in conf_upper],
        cumhaz=result.cumhaz,
        std_chaz=[float(value) for value in std_chaz],
        n_enter=result.n_enter,
        n_risk_count=result.n_risk_count,
        n_event_count=result.n_event_count,
        n_censor_count=result.n_censor_count,
        n_enter_count=result.n_enter_count,
        model=result.model,
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
    id_codes = _encode_labels(id_values, "id")
    tables = _core.counting_survfit_tables(
        starts,
        stops,
        status,
        id_codes,
        case_weights,
        include_entry,
        timefix,
    )

    return _survfit_from_count_tables(
        [float(value) for value in tables.time],
        [float(value) for value in tables.n_risk],
        [float(value) for value in tables.n_event],
        [float(value) for value in tables.n_event_count],
        [float(value) for value in tables.n_censor],
        [float(value) for value in tables.n_censor_count],
        None if tables.n_enter is None else [float(value) for value in tables.n_enter],
        n_risk_count=[float(value) for value in tables.n_risk_count],
        n_enter_count=(
            [float(value) for value in tables.n_enter_count]
            if getattr(tables, "n_enter_count", None) is not None
            else None
        ),
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
        n_risk_count=(
            [result.n_risk_count[0] if result.n_risk_count else 0.0, *result.n_risk_count]
            if getattr(result, "n_risk_count", None) is not None
            else None
        ),
        n_event_count=(
            [0.0, *result.n_event_count]
            if getattr(result, "n_event_count", None) is not None
            else None
        ),
        n_censor_count=(
            [0.0, *result.n_censor_count]
            if getattr(result, "n_censor_count", None) is not None
            else None
        ),
        n_enter_count=(
            [0.0, *result.n_enter_count]
            if getattr(result, "n_enter_count", None) is not None
            else None
        ),
    )


def _needs_time0_insert(times: Sequence[float], t0: float) -> bool:
    return not times or abs(float(times[0]) - t0) >= _SURVFIT_TIME_EPSILON


def _prepend_curve_time0(values: list[float], initial: float) -> list[float]:
    return [initial, *[float(value) for value in values]]


def _prepend_curve_time0_optional(values: list[float], initial: float) -> list[float]:
    return _prepend_curve_time0(values, initial) if values else []


def _optional_float_list(value: Any, name: str) -> list[float] | None:
    items = getattr(value, name, None)
    if items is None:
        return None
    return [float(item) for item in items]


def _prepend_matrix_time0(values: list[list[float]], initial: float) -> list[list[float]]:
    return [[initial, *[float(value) for value in row]] for row in values] if values else []


def _survfit0_default_time(result: Any) -> float:
    if isinstance(result, CoxSurvfitResult) and result.start_time is not None:
        return float(result.start_time)
    times = getattr(result, "time", None)
    if times is None:
        times = getattr(result, "time_points", None)
    values = [0.0]
    if times is not None:
        values.extend(float(value) for value in times)
    return min(values)


def _survfit0_result(result: SurvfitResult, t0: float | None = None) -> SurvfitResult:
    initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time, initial_time):
        return result
    n_risk0 = float(result.n_risk[0]) if result.n_risk else 0.0
    return SurvfitResult(
        time=_prepend_curve_time0(result.time, initial_time),
        n_risk=_prepend_curve_time0(result.n_risk, n_risk0),
        n_event=_prepend_curve_time0(result.n_event, 0.0),
        n_censor=_prepend_curve_time0(result.n_censor, 0.0),
        estimate=_prepend_curve_time0(result.estimate, 1.0),
        std_err=_prepend_curve_time0_optional(result.std_err, 0.0),
        conf_lower=_prepend_curve_time0_optional(result.conf_lower, 1.0),
        conf_upper=_prepend_curve_time0_optional(result.conf_upper, 1.0),
        cumhaz=_prepend_curve_time0(result.cumhaz, 0.0),
        std_chaz=_prepend_curve_time0_optional(result.std_chaz, 0.0),
        n_enter=(_prepend_curve_time0(result.n_enter, 0.0) if result.n_enter is not None else None),
        n_risk_count=(
            _prepend_curve_time0(
                result.n_risk_count,
                result.n_risk_count[0] if result.n_risk_count else 0.0,
            )
            if result.n_risk_count is not None
            else None
        ),
        n_event_count=(
            _prepend_curve_time0(result.n_event_count, 0.0)
            if result.n_event_count is not None
            else None
        ),
        n_censor_count=(
            _prepend_curve_time0(result.n_censor_count, 0.0)
            if result.n_censor_count is not None
            else None
        ),
        n_enter_count=(
            _prepend_curve_time0(result.n_enter_count, 0.0)
            if result.n_enter_count is not None
            else None
        ),
        model=result.model,
    )


def _survfit0_cox_result(
    result: CoxSurvfitResult,
    t0: float | None = None,
) -> CoxSurvfitResult:
    initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time, initial_time):
        return result
    return CoxSurvfitResult(
        time=_prepend_curve_time0(result.time, initial_time),
        surv=_prepend_matrix_time0(result.surv, 1.0),
        cumhaz=_prepend_matrix_time0(result.cumhaz, 0.0),
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
        start_time=result.start_time,
        std_err=_prepend_matrix_time0(result.std_err, 0.0),
        std_chaz=_prepend_matrix_time0(result.std_chaz, 0.0),
        conf_lower=_prepend_matrix_time0(result.conf_lower, 1.0),
        conf_upper=_prepend_matrix_time0(result.conf_upper, 1.0),
        model=result.model,
    )


def _survfit0_turnbull_result(
    result: TurnbullSurvfitResult,
    t0: float | None = None,
) -> TurnbullSurvfitResult:
    initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
    if not _needs_time0_insert(result.time_points, initial_time):
        return result
    return TurnbullSurvfitResult(
        time_points=_prepend_curve_time0(result.time_points, initial_time),
        survival=_prepend_curve_time0(result.survival, 1.0),
        survival_lower=_prepend_curve_time0_optional(result.survival_lower, 1.0),
        survival_upper=_prepend_curve_time0_optional(result.survival_upper, 1.0),
        n_iter=result.n_iter,
        converged=result.converged,
        model=result.model,
    )


def _is_survfit_result_like(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("time", "n_risk", "n_event", "n_censor", "estimate", "cumhaz")
    )


def _coerce_survfit_result_like(value: Any) -> SurvfitResult:
    if isinstance(value, SurvfitResult):
        return value
    return SurvfitResult(
        time=[float(item) for item in value.time],
        n_risk=[float(item) for item in value.n_risk],
        n_event=[float(item) for item in value.n_event],
        n_censor=[float(item) for item in value.n_censor],
        estimate=[float(item) for item in value.estimate],
        std_err=[float(item) for item in getattr(value, "std_err", [])],
        conf_lower=[float(item) for item in getattr(value, "conf_lower", [])],
        conf_upper=[float(item) for item in getattr(value, "conf_upper", [])],
        cumhaz=[float(item) for item in value.cumhaz],
        std_chaz=[float(item) for item in getattr(value, "std_chaz", [])],
        n_enter=(
            [float(item) for item in value.n_enter]
            if getattr(value, "n_enter", None) is not None
            else None
        ),
        n_risk_count=_optional_float_list(value, "n_risk_count"),
        n_event_count=_optional_float_list(value, "n_event_count"),
        n_censor_count=_optional_float_list(value, "n_censor_count"),
        n_enter_count=_optional_float_list(value, "n_enter_count"),
        model=getattr(value, "model", None),
    )


def _is_turnbull_result_like(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("time_points", "survival", "survival_lower", "survival_upper")
    )


def _coerce_turnbull_result_like(value: Any) -> TurnbullSurvfitResult:
    if isinstance(value, TurnbullSurvfitResult):
        return value
    return TurnbullSurvfitResult(
        time_points=[float(item) for item in value.time_points],
        survival=[float(item) for item in value.survival],
        survival_lower=[float(item) for item in value.survival_lower],
        survival_upper=[float(item) for item in value.survival_upper],
        n_iter=int(getattr(value, "n_iter", 0)),
        converged=bool(getattr(value, "converged", True)),
        model=getattr(value, "model", None),
    )


def _survfit0_any_result(value: Any, t0: float | None = None) -> Any:
    if isinstance(value, SurvfitResult) or _is_survfit_result_like(value):
        result = _coerce_survfit_result_like(value)
        initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
        return (
            value
            if not _needs_time0_insert(result.time, initial_time)
            else _survfit0_result(
                result,
                initial_time,
            )
        )
    if isinstance(value, CoxSurvfitResult):
        return _survfit0_cox_result(value, t0)
    if isinstance(value, TurnbullSurvfitResult) or _is_turnbull_result_like(value):
        result = _coerce_turnbull_result_like(value)
        initial_time = _survfit0_default_time(result) if t0 is None else float(t0)
        return (
            value
            if not _needs_time0_insert(
                result.time_points,
                initial_time,
            )
            else _survfit0_turnbull_result(result, initial_time)
        )
    return value


def _mapping_survfit0_time(results: Mapping[Any, Any]) -> float:
    values = [0.0]
    for result in results.values():
        times = getattr(result, "time", None)
        if times is None:
            times = getattr(result, "time_points", None)
        if times is not None:
            values.extend(float(value) for value in times)
    return min(values)


def survfit0(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Insert an initial survival row into an existing survfit result."""

    if args or kwargs:
        raise TypeError("survfit0 got unexpected arguments")
    if isinstance(x, Mapping):
        t0 = _mapping_survfit0_time(x)
        return {label: _survfit0_any_result(result, t0) for label, result in x.items()}
    result = _survfit0_any_result(x)
    if result is not x or isinstance(x, SurvfitResult | CoxSurvfitResult | TurnbullSurvfitResult):
        return result
    if _is_survfit_result_like(x) or _is_turnbull_result_like(x):
        return result
    raise TypeError("survfit0 requires a survfit result")


def _cox_survfit_result_cumhaz(surv: list[float]) -> list[float]:
    return [math.inf if value <= 0.0 else -math.log(value) for value in surv]


def _cox_survfit_result_std_chaz(
    surv: list[float],
    std_err: list[float],
) -> list[float]:
    result: list[float] = []
    for survival, se in zip(surv, std_err, strict=True):
        if survival > 0.0:
            result.append(float(se) / float(survival))
        else:
            result.append(math.inf if se > 0.0 else 0.0)
    return result


def _weighted_average(values: Sequence[float], weights: Sequence[float] | None) -> float:
    if not values:
        return math.nan
    if weights is None:
        return sum(float(value) for value in values) / len(values)
    total = sum(float(weight) for weight in weights)
    if total <= 0.0:
        return math.nan
    return (
        sum(float(value) * float(weight) for value, weight in zip(values, weights, strict=True))
        / total
    )


def _cox_survfit_from_aggregates(
    source: CoxSurvfitResult,
    aggregates: Sequence[Any],
    linear_predictors: Sequence[float],
) -> CoxSurvfitResult:
    surv = [[float(value) for value in aggregate.surv] for aggregate in aggregates]
    std_err = (
        [[float(value) for value in aggregate.std_err] for aggregate in aggregates]
        if source.std_err
        else []
    )
    std_chaz = (
        [
            _cox_survfit_result_std_chaz(surv_curve, se_curve)
            for surv_curve, se_curve in zip(surv, std_err, strict=True)
        ]
        if source.std_chaz and std_err
        else []
    )
    conf_lower = (
        [[float(value) for value in aggregate.lower] for aggregate in aggregates]
        if source.conf_lower and source.conf_upper and std_err
        else []
    )
    conf_upper = (
        [[float(value) for value in aggregate.upper] for aggregate in aggregates]
        if source.conf_lower and source.conf_upper and std_err
        else []
    )

    return CoxSurvfitResult(
        time=[float(value) for value in aggregates[0].time] if aggregates else [],
        surv=surv,
        cumhaz=[_cox_survfit_result_cumhaz(curve) for curve in surv],
        linear_predictors=[float(value) for value in linear_predictors],
        centered=source.centered,
        start_time=source.start_time,
        std_err=std_err,
        std_chaz=std_chaz,
        conf_lower=conf_lower,
        conf_upper=conf_upper,
        model=source.model,
    )


def aggregate_survfit_result(
    result: CoxSurvfitResult,
    groups: Any | None = None,
    weights: Any | None = None,
) -> CoxSurvfitResult:
    """Average Cox survfit prediction curves, optionally by group code."""

    if not isinstance(result, CoxSurvfitResult):
        raise TypeError("survfit object does not have a 'data' margin")

    n_curves = len(result.surv)
    if len(result.cumhaz) != n_curves or len(result.linear_predictors) != n_curves:
        raise ValueError("Cox survfit result has inconsistent curve counts")
    if n_curves == 0:
        return CoxSurvfitResult(
            time=[float(value) for value in result.time],
            surv=[],
            cumhaz=[],
            linear_predictors=[],
            centered=result.centered,
            start_time=result.start_time,
            model=result.model,
        )

    curve_times = [[float(value) for value in result.time] for _ in range(n_curves)]
    curve_survs = [[float(value) for value in curve] for curve in result.surv]
    curve_std_errs = (
        [[float(value) for value in curve] for curve in result.std_err] if result.std_err else None
    )
    curve_weights = _float_vector(weights, "weights") if weights is not None else None

    if groups is None:
        aggregate = _core.aggregate_survfit(
            curve_times,
            curve_survs,
            curve_std_errs,
            curve_weights,
            None,
        )
        linear_predictor = _weighted_average(result.linear_predictors, curve_weights)
        return _cox_survfit_from_aggregates(result, [aggregate], [linear_predictor])

    group_codes = _integer_code_vector(groups, "groups", "integer group codes")
    if len(group_codes) != n_curves:
        raise ValueError("groups must have same length as number of curves")
    if any(code < 1 for code in group_codes):
        raise ValueError("groups must use positive integer group codes")

    if len(set(group_codes)) == 1:
        return aggregate_survfit_result(result, weights=curve_weights)

    aggregates = []
    linear_predictors = []
    for code in sorted(set(group_codes)):
        indices = [idx for idx, group_code in enumerate(group_codes) if group_code == code]
        group_weights = (
            [curve_weights[idx] for idx in indices] if curve_weights is not None else None
        )
        aggregate = _core.aggregate_survfit(
            [curve_times[idx] for idx in indices],
            [curve_survs[idx] for idx in indices],
            [curve_std_errs[idx] for idx in indices] if curve_std_errs is not None else None,
            group_weights,
            None,
        )
        aggregates.append(aggregate)
        linear_predictors.append(
            _weighted_average(
                [float(result.linear_predictors[idx]) for idx in indices],
                group_weights,
            )
        )

    return _cox_survfit_from_aggregates(result, aggregates, linear_predictors)


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
            n_risk_count=result.n_risk_count,
            n_event_count=result.n_event_count,
            n_censor_count=result.n_censor_count,
            n_enter_count=result.n_enter_count,
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
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
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
            n_risk_count=result.n_risk_count,
            n_event_count=result.n_event_count,
            n_censor_count=result.n_censor_count,
            n_enter_count=result.n_enter_count,
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
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
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
            cumhaz=_core.step_values_at(times, hazards, training_times, 0.0),
            centered=centered,
        )

    base_times, base_hazards, base_strata = with_strata(centered)
    event_times = getattr(fit, "event_times", None)
    if event_times is None:
        strata_values = [int(value) for value in base_strata]
        strata = strata_values if len(set(strata_values)) > 1 else None
        return CoxBaseHazardResult(
            time=[float(value) for value in base_times],
            cumhaz=[float(value) for value in base_hazards],
            strata=strata,
            centered=centered,
            strata_labels=_cox_strata_labels_for_fit(fit, strata),
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
        requested_times = sorted(stratum_stop_times)
        requested_hazards = _core.step_values_at(
            stratum_times,
            stratum_hazards,
            requested_times,
            0.0,
        )
        for time, hazard in zip(requested_times, requested_hazards, strict=True):
            expanded_times.append(time)
            expanded_hazards.append(hazard)
            expanded_strata.append(stratum)

    strata = expanded_strata if len(set(row_strata)) > 1 else None
    return CoxBaseHazardResult(
        time=expanded_times,
        cumhaz=expanded_hazards,
        strata=strata,
        centered=centered,
        strata_labels=_cox_strata_labels_for_fit(fit, strata),
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
    formula_group_levels: tuple[Any, ...] | None = None
    if isinstance(response, str):
        formula = response
        id_column = id_arg if isinstance(id_arg, str) else None
        cluster_column = cluster if isinstance(cluster, str) else None
        if id_column is not None:
            id_arg = _column(data, id_column)
        if cluster_column is not None:
            cluster = _column(data, cluster_column)
        if subset is not None:
            data, aligned = _subset_formula_inputs(
                formula,
                data,
                subset,
                weights=weights,
                id=id_arg,
                cluster=cluster,
            )
            weights = aligned["weights"]
            id_arg = aligned["id"]
            cluster = aligned["cluster"]
            subset = None
        data, aligned = _apply_formula_na_action(
            formula,
            data,
            na_action,
            weights=weights,
            id=id_arg,
            cluster=cluster,
        )
        weights = aligned["weights"]
        id_arg = aligned["id"]
        cluster = aligned["cluster"]
        na_action = "pass"
        response, terms = _parse_formula(formula, data)
        if terms.clusters:
            if cluster is not None:
                raise ValueError("survfit formula cluster() cannot be combined with cluster")
            cluster = _combined_columns(data, terms.clusters, len(response))
        model_frame = _survfit_formula_model_frame(
            formula,
            data,
            response,
            weights,
            id_arg,
            id_column,
            cluster,
            cluster_column,
        )
        if terms.strata or terms.covariates:
            group = _combined_formula_groups(data, terms.strata, terms.covariates, len(response))
            formula_group_levels = _r_formula_ordered_levels(group, "survfit formula groups")

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
        cluster = _subset_optional_sequence(cluster, indices, "cluster")
    response, aligned = _apply_surv_na_action(
        response,
        na_action,
        "survfit inputs",
        group=group,
        weights=weights,
        id=id_arg,
        cluster=cluster,
    )
    group = aligned["group"]
    weights = aligned["weights"]
    id_arg = aligned["id"]
    cluster = aligned["cluster"]
    id_values = _materialize_labels(id_arg, "id") if id_arg is not None else None
    if id_values is not None and len(id_values) != len(response):
        raise ValueError("id must have the same length as the Surv response")
    if model_frame is None:
        model_frame = _survfit_model_frame(response, group, weights, id_values, cluster)
    if newdata is not None:
        raise ValueError("newdata is only supported for fitted Cox models")
    if not include_censor:
        raise ValueError("censor is only supported for fitted Cox models")
    if include_entry and (response.start is None or id_values is None):
        raise ValueError("survfit entry=TRUE requires counting-process Surv input and id")
    if response.type in {"left", "interval", "interval2"}:
        if (
            include_se
            and _survfit_robust_cluster_values(response, cluster, id_values, None, robust_value)
            is not None
        ):
            raise NotImplementedError(
                "survfit robust variance is currently supported only for right-censored or "
                "counting-process Kaplan-Meier curves"
            )
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
        for label, indices in _group_indices(
            group, len(response), levels=formula_group_levels
        ).items():
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
        cluster = _subset_optional_sequence(cluster, indices, "cluster")
    entry_times = list(response.start) if response.start is not None else None
    robust_clusters = (
        _survfit_robust_cluster_values(response, cluster, id_values, wt, robust_value)
        if include_se
        else None
    )
    if robust_clusters is not None and response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survfit robust variance is currently supported only for right-censored or "
            "counting-process curves"
        )
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
            if robust_clusters is not None:
                km = _survfit_robust_km_result(
                    km,
                    response,
                    wt,
                    robust_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    timefix=fix_time,
                )
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
        result = _survfit_from_km_counts(
            km,
            normalized_conf_level,
            computation,
            normalized_conf_type,
        )
        if robust_clusters is not None:
            if response.start is not None:
                result = _survfit_robust_counting_result(
                    result,
                    response,
                    wt,
                    robust_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    computation=computation,
                    timefix=fix_time,
                )
            else:
                result = _survfit_robust_right_result(
                    result,
                    response,
                    wt,
                    robust_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    computation=computation,
                    timefix=fix_time,
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
    for label, indices in _group_indices(group, len(response), levels=formula_group_levels).items():
        group_response = _subset_surv(response, indices)
        group_weights = [wt[idx] for idx in indices] if wt is not None else None
        group_ids = [id_values[idx] for idx in indices] if id_values is not None else None
        group_clusters = (
            [robust_clusters[idx] for idx in indices] if robust_clusters is not None else None
        )
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
            if group_clusters is not None:
                km = _survfit_robust_km_result(
                    km,
                    group_response,
                    group_weights,
                    group_clusters,
                    reverse=reverse_curve,
                    conf_level=normalized_conf_level,
                    conf_type=normalized_conf_type,
                    timefix=fix_time,
                )
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
            result = _survfit_from_km_counts(
                km,
                normalized_conf_level,
                computation,
                normalized_conf_type,
            )
            if group_clusters is not None:
                if group_response.start is not None:
                    result = _survfit_robust_counting_result(
                        result,
                        group_response,
                        group_weights,
                        group_clusters,
                        reverse=reverse_curve,
                        conf_level=normalized_conf_level,
                        conf_type=normalized_conf_type,
                        computation=computation,
                        timefix=fix_time,
                    )
                else:
                    result = _survfit_robust_right_result(
                        result,
                        group_response,
                        group_weights,
                        group_clusters,
                        reverse=reverse_curve,
                        conf_level=normalized_conf_level,
                        conf_type=normalized_conf_type,
                        computation=computation,
                        timefix=fix_time,
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


def _survdiff_r_level_sort_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_survdiff_r_level_sort_key(part) for part in value)
    return _strata_level_sort_key(value)


def _r_formula_ordered_levels(values: list[Any], name: str) -> tuple[Any, ...]:
    levels: dict[Any, None] = {}
    for value in values:
        try:
            levels.setdefault(value, None)
        except TypeError as exc:
            raise TypeError(f"{name} contain unhashable labels") from exc
    return tuple(sorted(levels, key=_survdiff_r_level_sort_key))


def _survdiff_r_ordered_levels(values: list[Any]) -> tuple[Any, ...]:
    return _r_formula_ordered_levels(values, "survdiff formula groups")


def _survdiff_formula_groups(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> tuple[list[Any], tuple[Any, ...], list[Any] | None]:
    if not terms.covariates:
        if terms.strata:
            raise ValueError("survdiff formula has no groups to test")
        raise ValueError("survdiff formula requires at least one grouping term")
    group = _combined_formula_groups(data, [], terms.covariates, n)
    group_levels = _survdiff_r_ordered_levels(group)
    strata = _combined_columns(data, terms.strata, n) if terms.strata else None
    return group, group_levels, strata


def _survdiff_offset_formula_values(
    data: Any,
    terms: _FormulaTerms,
    n: int,
) -> list[float] | None:
    if not terms.offsets:
        return None
    if terms.covariates or terms.strata:
        raise ValueError("Cannot have both an offset and groups")
    values = _offset_vector(data, terms.offsets, n)
    if values is None:
        raise ValueError("offset formula did not produce values")
    return values


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


def _survdiff_offset_expected(offsets: Sequence[float]) -> float:
    total = 0.0
    for value in offsets:
        if value == 0.0:
            return math.inf
        total += -math.log(value)
    return total


def _survdiff_divide_statistic(numerator: float, variance: float) -> float:
    squared = numerator * numerator
    if variance == 0.0:
        return math.nan if squared == 0.0 else math.inf
    return squared / variance


def _survdiff_chisq_p_value(statistic: float) -> float:
    return math.erfc(math.sqrt(statistic / 2.0))


def _survdiff_offset_result(response: Surv, offsets: Sequence[float], rho: float) -> Any:
    if response.type != "right":
        raise NotImplementedError("survdiff offset formulas require right-censored Surv responses")
    if len(offsets) != len(response):
        raise ValueError("offset must have the same length as the Surv response")
    if any(value < 0.0 or value > 1.0 for value in offsets):
        raise ValueError("The offset must be a survival probability")

    observed = float(sum(response.event))
    expected = _survdiff_offset_expected(offsets)
    if rho == 0.0:
        variance = expected
        numerator = observed - variance
    else:
        inverse_rho = 1.0 / rho
        numerator = sum(
            inverse_rho - ((inverse_rho + float(event)) * (offset**rho))
            for offset, event in zip(offsets, response.event, strict=True)
        )
        variance = sum((1.0 - (offset ** (2.0 * rho))) / (2.0 * rho) for offset in offsets)
    statistic = _survdiff_divide_statistic(numerator, variance)
    return _core.LogRankResult(
        statistic,
        _survdiff_chisq_p_value(statistic),
        1,
        [observed],
        [expected],
        variance,
        _survdiff_weight_type(rho),
    )


def _stratified_survdiff(
    response: Surv,
    group: Any,
    strata: Any,
    rho: float,
    timefix: bool,
    group_levels: Sequence[Any] | None = None,
) -> Any:
    n = len(response)
    group_codes = _encode_groups(group, n, levels=group_levels)
    strata_codes = _encode_groups(strata, n)
    times = list(response.time)
    if response.start is not None:
        components = _core.stratified_counting_logrank_components(
            times,
            list(response.event),
            [code + 1 for code in group_codes],
            list(response.start),
            strata_codes,
            rho,
            timefix,
        )
        return _survdiff_result_from_components(components, rho)

    components = _core.stratified_logrank_components(
        times,
        list(response.event),
        [code + 1 for code in group_codes],
        strata_codes,
        rho,
        timefix,
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
    formula_group_levels: Sequence[Any] | None = None
    formula_offsets: list[float] | None = None
    if isinstance(response, str):
        if subset is not None:
            data, _aligned = _subset_formula_inputs(response, data, subset)
            subset = None
        data, _aligned = _apply_formula_na_action(response, data, na_action)
        na_action = "pass"
        response, terms = _parse_formula(response, data)
        _reject_formula_clusters("survdiff", terms)
        formula_offsets = _survdiff_offset_formula_values(data, terms, len(response))
        if formula_offsets is None:
            group, formula_group_levels, formula_strata = _survdiff_formula_groups(
                data,
                terms,
                len(response),
            )

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
    if response.type not in {"right", "counting"}:
        raise NotImplementedError(
            "survdiff currently supports right-censored and counting Surv responses"
        )
    rho_value = _finite_float(rho, "rho")
    fix_time = _normalize_bool_option(timefix, "timefix")
    if formula_offsets is not None:
        return _survdiff_offset_result(response, formula_offsets, rho_value)
    if group is None:
        raise ValueError("group is required")
    if formula_strata is not None:
        return _stratified_survdiff(
            response,
            group,
            formula_strata,
            rho_value,
            fix_time,
            formula_group_levels,
        )

    groups = _encode_groups(group, len(response), levels=formula_group_levels)
    group_codes = [code + 1 for code in groups]
    if response.start is not None:
        components = _core.compute_counting_logrank_components(
            list(response.time),
            list(response.event),
            group_codes,
            list(response.start),
            None,
            rho_value,
            fix_time,
        )
    else:
        components = _core.survdiff2(
            list(response.time),
            list(response.event),
            group_codes,
            None,
            rho_value,
            fix_time,
        )
    return _survdiff_result_from_components(components, rho_value)


def _coxph_wtest_b_matrix(b: Any) -> tuple[list[list[float | None]], bool]:
    raw = _coerce_array_like(b, "b")
    if raw and isinstance(raw[0], list | tuple):
        rows: list[list[float | None]] = []
        width = len(raw[0])
        for row in raw:
            if not isinstance(row, list | tuple) or len(row) != width:
                raise ValueError("b matrix rows must be rectangular")
            rows.append([None if _is_missing_value(value) else float(value) for value in row])
        return rows, True
    return [[None if _is_missing_value(value) else float(value)] for value in raw], False


def _coxph_wtest_var_matrix(var: Any) -> tuple[list[list[float]], int]:
    raw = _coerce_array_like(var, "var")
    if raw and isinstance(raw[0], list | tuple):
        rows = _as_matrix_rows(raw, "var", allow_empty_columns=False)
        return rows, len(raw) * (len(raw[0]) if raw else 0)
    values = [float(value) for value in raw]
    if len(values) == 1:
        return [[values[0]]], 1
    return [], len(values)


def _coxph_solve_linear_system(matrix: list[list[float]], rhs: Sequence[float]) -> list[float]:
    n = len(matrix)
    if n == 0:
        return []
    augmented = [list(row) + [float(rhs[idx])] for idx, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) <= 1e-14:
            raise ValueError("First argument must be a square matrix")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for row in range(col + 1, n):
            factor = augmented[row][col] / pivot_value
            if factor == 0.0:
                continue
            for idx in range(col, n + 1):
                augmented[row][idx] -= factor * augmented[col][idx]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n] - sum(
            augmented[row][col] * solution[col] for col in range(row + 1, n)
        )
        solution[row] = total / augmented[row][row]
    return solution


def _coxph_wtest_active_indices(matrix: list[list[float]], toler_chol: float) -> list[int]:
    n = len(matrix)
    max_diag = max((abs(matrix[idx][idx]) for idx in range(n)), default=0.0)
    threshold = toler_chol * max_diag
    active: list[int] = []
    for idx in range(n):
        variance = matrix[idx][idx]
        if active:
            submatrix = [[matrix[row][col] for col in active] for row in active]
            covariance = [matrix[idx][col] for col in active]
            projected = _coxph_solve_linear_system(submatrix, covariance)
            variance -= sum(value * coef for value, coef in zip(covariance, projected, strict=True))
        if variance > threshold and variance > 0.0:
            active.append(idx)
    return active


def _coxph_wtest_solve(
    matrix: list[list[float]],
    b_columns: list[list[float]],
    toler_chol: float,
) -> tuple[list[float], int, list[list[float]]]:
    n = len(matrix)
    active = _coxph_wtest_active_indices(matrix, toler_chol)
    if not active:
        return [0.0 for _ in b_columns], 0, [[0.0 for _ in b_columns] for _ in range(n)]

    submatrix = [[matrix[row][col] for col in active] for row in active]
    solve_columns: list[list[float]] = []
    tests: list[float] = []
    for column in b_columns:
        rhs = [column[idx] for idx in active]
        active_solution = _coxph_solve_linear_system(submatrix, rhs)
        solution = [0.0] * n
        for idx, value in zip(active, active_solution, strict=True):
            solution[idx] = value
        solve_columns.append(solution)
        tests.append(sum(value * coef for value, coef in zip(column, solution, strict=True)))

    solve_rows = [
        [solve_columns[col_idx][row_idx] for col_idx in range(len(solve_columns))]
        for row_idx in range(n)
    ]
    return tests, len(active), solve_rows


def coxph_wtest(var: Any, b: Any, toler_chol: Any = 1e-9) -> CoxPHWTestResult:
    """Compute the Wald test helper exported as R's ``coxph.wtest``."""

    toler_value = _finite_float(toler_chol, "toler_chol")
    if toler_value < 0.0:
        raise ValueError("toler_chol must be non-negative")
    b_rows, b_is_matrix = _coxph_wtest_b_matrix(b)
    if any(value is None for row in b_rows for value in row):
        return CoxPHWTestResult(test=[], df=0, solve=0.0)
    b_numeric = [[float(value) for value in row] for row in b_rows]
    if any(not math.isfinite(value) for row in b_numeric for value in row):
        raise ValueError("infinite argument in coxph.wtest")

    nvar = len(b_rows)
    ntest = len(b_rows[0]) if b_rows and b_is_matrix else 1
    matrix, raw_var_length = _coxph_wtest_var_matrix(var)
    if raw_var_length == 0:
        if nvar == 0:
            return CoxPHWTestResult(test=[], df=0, solve=0.0)
        raise ValueError("Argument lengths do not match")
    if raw_var_length == 1:
        if nvar != 1:
            raise ValueError("Argument lengths do not match")
        if b_is_matrix and ntest != 1:
            raise ValueError("non-conformable arrays")
        variance = matrix[0][0]
        if not math.isfinite(variance):
            raise ValueError("infinite argument in coxph.wtest")
        if variance == 0.0:
            raise ZeroDivisionError("division by zero")
        values = [row[0] for row in b_numeric]
        return CoxPHWTestResult(
            test=[value * value / variance for value in values],
            df=1,
            solve=[value / variance for value in values],
        )

    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("First argument must be a square matrix")
    if len(matrix) != nvar:
        raise ValueError("Argument lengths do not match")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("infinite argument in coxph.wtest")

    b_columns = [
        [b_numeric[row_idx][col_idx] for row_idx in range(nvar)] for col_idx in range(ntest)
    ]
    tests, df, solve_rows = _coxph_wtest_solve(matrix, b_columns, toler_value)
    solve: list[float] | list[list[float]] = (
        solve_rows if b_is_matrix and ntest > 1 else [row[0] for row in solve_rows]
    )
    return CoxPHWTestResult(test=tests, df=df, solve=solve)


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
                curve_strata_labels = _cox_strata_labels_for_fit(fit, curve_strata)
                return CoxBaseHazardResult(
                    time=result.time,
                    cumhaz=result.cumhaz[0],
                    strata=strata,
                    centered=True,
                    curve_strata=curve_strata,
                    strata_labels=_cox_strata_labels_for_fit(fit, strata),
                    curve_strata_labels=curve_strata_labels,
                )
            curve_strata_labels = _cox_strata_labels_for_fit(fit, result.strata)
            return CoxBaseHazardResult(
                time=result.time,
                cumhaz=result.cumhaz,
                centered=True,
                curve_strata=result.strata,
                curve_strata_labels=curve_strata_labels,
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
    beta = _cox_beta(fit)
    aliases = _cox_alias_mask(fit)
    if len(aliases) != len(beta):
        raise ValueError("fitted Cox model alias metadata does not match coefficient width")
    active_columns = [idx for idx, aliased in enumerate(aliases) if not aliased]
    if not active_columns:
        raise ValueError("cox_zph requires at least one estimable coefficient")
    if not raw:
        raise ValueError("cox_zph requires at least one event")

    full_nvar = len(raw[0])
    if any(len(row) != full_nvar for row in raw) or any(len(row) != full_nvar for row in scaled):
        raise ValueError("Schoenfeld residual arrays must be rectangular")
    if len(beta) != full_nvar:
        raise ValueError("fitted Cox model coefficients do not match residual width")
    groups = _cox_zph_active_groups(
        _cox_zph_column_groups(fit, full_nvar, group_terms),
        active_columns,
    )
    scaled = _matrix_columns(scaled, active_columns)
    beta = [beta[idx] for idx in active_columns]

    event_indices = _cox_event_indices(fit)
    if len(event_indices) != len(raw):
        raise ValueError("fitted Cox model event times do not match Schoenfeld residuals")
    event_times = [float(fit.event_times[idx]) for idx in event_indices]
    transform_name, transformed_time = _cox_zph_transform(fit, event_times, transform)
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
        if group_terms and groups
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
        var=_cox_zph_group_variance(fit, groups, beta, active_columns),
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
        coefficient_names = fit.coefficient_names if isinstance(fit, _FormulaFit) else None
        names = (
            list(coefficient_names)
            if coefficient_names is not None and len(coefficient_names) == nvar
            else _fallback_coef_names(nvar)
        )
        return [(name, [idx]) for idx, name in enumerate(names)]

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
    return [int(idx) for idx in _core.cox_event_indices(times, status, strata)]


def _cox_scaled_schoenfeld_from_raw(fit: Any, raw: list[list[float]]) -> list[list[float]]:
    beta = _cox_beta(fit)
    nvar = len(beta)
    if nvar == 0 or not raw:
        return raw
    variance = getattr(fit, "information_matrix", None)
    if variance is None:
        raise TypeError("model does not expose coefficient variance")
    matrix = [list(row) for row in variance]
    return _core.scale_schoenfeld_residuals(raw, beta, matrix)


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
    return _core.cox_dfbeta_from_score_residuals(score, matrix, scaled)


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


def _cox_zph_active_groups(
    groups: list[tuple[str, list[int]]],
    active_columns: list[int],
) -> list[tuple[str, list[int]]]:
    active_index = {
        original_index: dense_index for dense_index, original_index in enumerate(active_columns)
    }
    active_groups: list[tuple[str, list[int]]] = []
    for name, columns in groups:
        remapped = [active_index[column] for column in columns if column in active_index]
        if remapped:
            active_groups.append((name, remapped))
    return active_groups


def _cox_zph_term_matrix(
    scaled: list[list[float]],
    groups: list[tuple[str, list[int]]],
    beta: list[float],
) -> list[list[float]]:
    return _core.cox_zph_term_matrix(scaled, [columns for _name, columns in groups], beta)


def _cox_zph_group_variance(
    fit: Any,
    groups: list[tuple[str, list[int]]],
    beta: list[float],
    active_columns: list[int],
) -> list[list[float]]:
    raw_variance = getattr(fit, "information_matrix", None)
    if raw_variance is None:
        return []
    full_variance = [[float(value) for value in row] for row in raw_variance]
    full_nvar = len(full_variance)
    if any(len(row) != full_nvar for row in full_variance):
        return []
    if any(column < 0 or column >= full_nvar for column in active_columns):
        return []
    variance = [[full_variance[row][column] for column in active_columns] for row in active_columns]
    nvar = len(beta)
    if len(variance) != nvar or any(len(row) != nvar for row in variance):
        return []
    return _core.cox_zph_group_variance(variance, [columns for _name, columns in groups], beta)


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


def _cox_alias_mask(fit: Any) -> list[bool]:
    """Identify coefficients removed by the fitted Cox information rank."""

    model = _unwrap_formula_fit(fit)
    width = len(_cox_beta(model))
    aliases = [False] * width
    if width == 0 or int(getattr(model, "iterations", 0)) <= 0:
        return aliases

    try:
        rank = int(model.convergence_flag)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return aliases
    # A nonconverged or zero-iteration fit keeps its raw coefficients. Successful
    # step-halving fits use a negative flag, so their exact zero diagonals still
    # identify aliases even though the flag no longer records the fitted rank.
    if rank == _COX_NONCONVERGENCE_FLAG or rank >= width:
        return aliases

    raw_variance = getattr(model, "information_matrix", None)
    if raw_variance is None:
        return aliases
    variance = list(raw_variance)
    if len(variance) != width or any(len(row) != width for row in variance):
        return aliases

    aliases = [float(variance[idx][idx]) == 0.0 for idx in range(width)]
    if rank >= 0 and sum(aliases) != width - rank:
        return [False] * width
    return aliases


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
    return sum(not aliased for aliased in _cox_alias_mask(fit))


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
    coefficient_names = fit.coefficient_names if isinstance(fit, _FormulaFit) else None
    if coefficient_names is not None and len(coefficient_names) == width:
        return list(coefficient_names)
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
    beta = _cox_beta(fit)
    return [
        math.nan if aliased else value
        for value, aliased in zip(beta, _cox_alias_mask(fit), strict=True)
    ]


def coef_names(fit: Any, *, complete: Any | None = None) -> list[str]:
    """Return fitted coefficient names for R-style model helpers."""

    _require_model_fit(fit, "coef_names")
    include_complete = (
        _is_coxph_fit(fit) if complete is None else _normalize_bool_option(complete, "complete")
    )
    if _is_survreg_fit(fit):
        location_width = len(_location_beta(fit))
        names = _fit_location_coef_names(fit, location_width)
        if include_complete:
            total_width = len(list(fit.coefficients))
            names.extend(_survreg_scale_coef_names(fit, total_width - location_width))
        return names

    beta = _cox_beta(fit)
    names = _fit_location_coef_names(fit, len(beta))
    if include_complete:
        return names
    return [name for name, aliased in zip(names, _cox_alias_mask(fit), strict=True) if not aliased]


def vcov(fit: Any, *, complete: Any = True) -> list[list[float]]:
    """Return a fitted model variance-covariance matrix, like R's vcov generic."""

    _require_model_fit(fit, "vcov")
    include_complete = _normalize_bool_option_with_default(complete, "complete", True)
    if _is_survreg_fit(fit):
        width = len(list(fit.coefficients)) if include_complete else len(_location_beta(fit))
        return _survreg_variance_matrix(fit, width)
    variance = _cox_variance_matrix(fit, len(_cox_beta(fit)))
    if include_complete:
        return variance
    active = [idx for idx, aliased in enumerate(_cox_alias_mask(fit)) if not aliased]
    return [[variance[row_idx][col_idx] for col_idx in active] for row_idx in active]


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


def _sample_variance(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    return math.fsum((value - mean) ** 2 for value in values) / (n - 1)


def _rank_average(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[indexed[pos][0]] = rank
        idx = end
    return ranks


def _rank_first(values: Sequence[float]) -> list[int]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0] * len(values)
    for rank, (original_idx, _value) in enumerate(indexed, start=1):
        ranks[original_idx] = rank
    return ranks


def _royston_normal_scores(eta: Sequence[float], ties: bool) -> list[float]:
    n = len(eta)
    normal = NormalDist()
    if ties and len(set(eta)) != n:
        z = [normal.inv_cdf((rank - 0.375) / (n + 0.25)) for rank in range(1, n + 1)]
        rank_first = _rank_first(eta)
        grouped: dict[float, list[float]] = {value: [] for value in sorted(set(eta))}
        for value, rank in zip(eta, rank_first, strict=True):
            grouped[value].append(z[rank - 1])
        means = {value: math.fsum(scores) / len(scores) for value, scores in grouped.items()}
        return [means[value] for value in eta]

    return [normal.inv_cdf((rank - 0.375) / (n + 0.25)) for rank in _rank_average(eta)]


def _royston_gonen_heller(eta: Sequence[float]) -> float:
    if len(eta) < 2:
        return math.nan
    ordered = sorted(eta)
    total = 0.0
    for idx, value in enumerate(ordered[:-1]):
        total += math.fsum(1.0 / (1.0 + math.exp(value - later)) for later in ordered[idx + 1 :])
    return total * 2.0 / (len(ordered) * (len(ordered) - 1))


def _royston_response_for_fit(fit: Any, newdata: Any | None) -> Surv:
    if newdata is None:
        response = getattr(fit, "y", None)
        return response if isinstance(response, Surv) else _cox_training_response(fit)
    design = _formula_design_for_fit(fit)
    if design is None:
        raise ValueError("newdata royston predictions require a formula Cox model")
    return _surv_from_formula_design(newdata, design)


def royston(
    fit: Any,
    newdata: Any | None = None,
    ties: Any = True,
    adjust: Any = False,
) -> dict[str, float]:
    """R-compatible ``survival::royston`` statistics for fitted Cox models."""

    if not _is_coxph_fit(fit):
        raise TypeError("function defined only for coxph models")
    ties_value = _normalize_bool_option(ties, "ties")
    adjust_value = _normalize_bool_option(adjust, "adjust")
    response = _royston_response_for_fit(fit, newdata)
    if response.type not in {"right", "counting"}:
        raise ValueError("royston requires a right-censored or counting-process response")

    eta = [float(value) for value in predict(fit, newdata, type="lp")]
    if newdata is not None:
        preliminary = coxph(response, x=[[value] for value in eta])
        eta = [float(value) for value in predict(preliminary, type="lp")]

    n = len(eta)
    if n != len(response):
        raise ValueError("linear predictor length must match response length")
    if n < 2:
        raise ValueError("at least two observations are required")

    qhat = _royston_normal_scores(eta, ties_value)
    rfit = coxph(response, x=[[value] for value in qhat])
    beta_values = _cox_beta(rfit)
    if len(beta_values) != 1:
        raise ValueError("internal royston Cox fit did not return one coefficient")
    beta = beta_values[0]
    variance = _cox_variance_matrix(_unwrap_formula_fit(rfit), 1)[0][0]

    pi = math.pi
    d_value = beta * math.sqrt(8.0 / pi)
    se_d = math.sqrt(max(variance, 0.0) * 8.0 / pi)
    r_d = beta * beta / (pi * pi / 6.0 + beta * beta)
    r_i = beta * beta / (1.0 + beta * beta)

    if adjust_value:
        n_events = sum(1 for value in response.event if value == 1)
        n_coef = len(_cox_beta(fit))
        if n_events <= n_coef:
            raise ValueError("adjusted royston statistic requires events > model coefficients")
        ratio = n_events / (n_events - n_coef)
        temp = (1.0 + beta * beta - ratio) / ratio
        d_value = math.copysign(math.sqrt(abs(temp) * 8.0 / pi), beta * temp)
        se_d = se_d * abs(beta) / (ratio * math.sqrt(abs(temp))) if temp != 0.0 else math.inf
        r_d = 1.0 - ratio * (1.0 - r_i)

    eta_variance = _sample_variance(eta)
    result = {
        "D": d_value,
        "se(D)": se_d,
        "R.D": r_d,
        "R.KO": eta_variance / (pi * pi / 6.0 + eta_variance),
        "C.GH": _royston_gonen_heller(eta),
    }
    if newdata is None:
        loglik_values = _cox_loglik_values(_unwrap_formula_fit(fit))
        logtest = -2.0 * (loglik_values[0] - loglik_values[1])
        denominator = 1.0 - math.exp(2.0 * loglik_values[0] / n)
        result["R.N"] = (
            (1.0 - math.exp(-logtest / n)) / denominator if denominator != 0.0 else math.nan
        )
        return {
            "D": result["D"],
            "se(D)": result["se(D)"],
            "R.D": result["R.D"],
            "R.KO": result["R.KO"],
            "R.N": result["R.N"],
            "C.GH": result["C.GH"],
        }
    return result


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
    """Return a plain stored model frame for compatible fitted objects."""

    if isinstance(fit, Mapping):
        if not fit:
            raise TypeError("model_frame requires a non-empty grouped survfit result")
        return model_frame(next(iter(fit.values())))

    frame = getattr(fit, "model", None)
    if frame is None:
        raise TypeError("model_frame requires a stored model frame")
    if not isinstance(frame, Mapping):
        raise TypeError("stored model frame must be mapping-like")

    columns: dict[str, list[Any]] = {}
    for name, values in frame.items():
        if isinstance(values, Surv):
            columns.update(_model_frame_surv_columns(values, set(columns)))
            continue
        if isinstance(values, Mapping):
            continue
        text_name = str(name)
        if text_name in {"group", "(id)", "(cluster)", "(strata)"}:
            columns[text_name] = _materialize_labels(values, text_name)
            continue
        materialized = _materialize_1d(values, text_name)
        if materialized and isinstance(materialized[0], list | tuple):
            continue
        columns[text_name] = list(materialized)
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
        if math.isnan(value):
            statistic = math.nan
        elif standard_error > 0.0:
            statistic = value / standard_error
        elif value == 0.0:
            statistic = math.nan
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
    variance = vcov(fit, complete=not _is_survreg_fit(fit))
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
    variance = vcov(fit, complete=not _is_survreg_fit(fit))
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
        distribution_parameters = getattr(fit, "distribution_parameters", None)
        if distribution_parameters is not None:
            parameter_values = [float(value) for value in distribution_parameters]
            if parameter_values:
                result["distribution_parameters"] = parameter_values
    else:
        model = _unwrap_formula_fit(fit)
        logliks = _cox_loglik_values(model)
        result["null_loglik"] = logliks[0]
        result["n_event"] = sum(1 for event in model.status if int(event) == 1)
        result["method"] = str(getattr(model, "method", "breslow"))
    return result


def _empty_columns(names: tuple[str, ...]) -> dict[str, list[Any]]:
    return {name: [] for name in names}


def _add_optional_survfit_column(
    frame: dict[str, list[Any]],
    name: str,
    values: Sequence[Any],
    row_count: int,
) -> None:
    column = list(values)
    if not column:
        return
    if len(column) != row_count:
        raise ValueError(f"survfit column {name!r} must match time length")
    frame[name] = column


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
    row_count = len(result.time)
    frame: dict[str, list[Any]] = {
        "time": result.time,
        "n.risk": result.n_risk,
        "n.event": result.n_event,
        "n.censor": result.n_censor,
        "surv": result.estimate,
        "cumhaz": result.cumhaz,
    }
    _add_optional_survfit_column(frame, "std.err", std_err, row_count)
    _add_optional_survfit_column(frame, "lower", conf_lower, row_count)
    _add_optional_survfit_column(frame, "upper", conf_upper, row_count)
    _add_optional_survfit_column(frame, "std.chaz", result.std_chaz, row_count)
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
            n_risk_count=_optional_float_list(result, "n_risk_count"),
            n_event_count=_optional_float_list(result, "n_event_count"),
            n_censor_count=_optional_float_list(result, "n_censor_count"),
            n_enter_count=_optional_float_list(result, "n_enter_count"),
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
        curve_strata = result.curve_strata_labels or result.curve_strata
        if curve_strata is not None:
            frame["strata"] = []
        for curve_idx, curve in enumerate(result.cumhaz):
            if isinstance(curve, (str, bytes)):
                raise TypeError("basehaz cumulative hazards must be numeric")
            if len(curve) != len(result.time):
                raise ValueError("basehaz curve length must match time length")
            frame["curve"].extend([curve_idx + 1] * len(result.time))
            frame["time"].extend(result.time)
            frame["cumhaz"].extend([float(value) for value in curve])
            if curve_strata is not None:
                frame["strata"].extend([curve_strata[curve_idx]] * len(result.time))
        return frame

    frame = {
        "time": result.time,
        "cumhaz": [float(value) for value in result.cumhaz],
    }
    strata = result.strata_labels or result.strata
    if strata is not None:
        frame["strata"] = strata
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
    strata = result.strata_labels or result.strata
    if strata is not None:
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
        if strata is not None:
            frame["strata"].extend([strata[curve_idx]] * n_times)
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
        tied_x = result.tied_x if isinstance(result.tied_x, list) else [result.tied_x] * n_scores
        tied_y = result.tied_y if isinstance(result.tied_y, list) else [result.tied_y] * n_scores
        tied_xy = (
            result.tied_xy if isinstance(result.tied_xy, list) else [result.tied_xy] * n_scores
        )
        return {
            "score": score_names,
            "concordance": [float(value) for value in result.concordance],
            "concordant": [float(value) for value in result.concordant],
            "comparable": [float(value) for value in result.comparable],
            "tied.x": [float(value) for value in tied_x],
            "tied.y": [float(value) for value in tied_y],
            "tied.xy": [float(value) for value in tied_xy],
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
        "tied.x": [float(result.tied_x)],
        "tied.y": [float(result.tied_y)],
        "tied.xy": [float(result.tied_xy)],
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
    if isinstance(result, PyearsResult):
        return _pyears_result_frame(result)
    if isinstance(result, FineGrayOutput):
        return _finegray_frame(result)
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


def _cox_refit_loglik_and_df(
    fit: Any,
    width: int,
    offset: list[float] | None,
) -> tuple[float, int]:
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
    return _cox_full_loglik(refit), _cox_degrees_of_freedom(refit)


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
        if idx == len(groups) - 1:
            logliks.append(_cox_full_loglik(model))
            dfs.append(_cox_degrees_of_freedom(model))
        else:
            refit_loglik, refit_df = _cox_refit_loglik_and_df(model, width, offset)
            logliks.append(refit_loglik)
            dfs.append(refit_df)
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
        distribution_parameter=(
            _survreg_t_fit_degrees_of_freedom(getattr(fit, "distribution_parameters", None))
            if _survreg_distribution_family(fit) == "t"
            else None
        ),
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

    cluster_codes = _encode_labels(cluster_values, "cluster")
    robust = _core.clustered_sandwich_variance(score, weights, cluster_codes, naive)
    return robust, naive, cluster_values


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
    cluster_codes = _encode_labels(cluster_values, "cluster")
    robust = _core.clustered_crossprod(dfbeta_rows, weights, cluster_codes, width)
    return robust, naive, cluster_values


def _cox_linear_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
    reference: str,
    newdata: Any | None,
) -> list[float]:
    design_rows = _cox_prediction_design_rows(fit, rows, reference, newdata)
    variance = _cox_variance_matrix(fit, len(_cox_beta(fit)))
    return _core.prediction_se_from_variance(design_rows, variance)


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
    return _core.term_prediction_se_from_variance(
        design_rows,
        variance,
        [groups[group_idx][1] for group_idx in selected],
    )


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
    return _core.term_prediction_se_from_variance(
        prediction_rows,
        variance,
        [groups[group_idx][1] for group_idx in selected],
    )


def _survreg_linear_prediction_se(
    fit: Any,
    rows: list[list[float]] | None,
) -> list[float]:
    beta = _location_beta(fit)
    prediction_rows = _survreg_prediction_rows(fit, rows, "prediction SEs")
    variance = _location_variance_matrix(fit, len(beta))
    return _core.prediction_se_from_variance(prediction_rows, variance)


def _survreg_response_uses_log_transform(fit: Any) -> bool:
    distribution = str(getattr(fit, "distribution", "")).lower().replace("-", "_")
    return distribution in {
        "weibull",
        "exponential",
        "rayleigh",
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
    if distribution in {"t", "student", "student_t", "studentt"}:
        return "t"
    return "extreme"


def _survreg_quantile_probabilities(values: Any | None) -> list[float]:
    probabilities = _quantile_vector(values, "p") if values is not None else [0.1, 0.9]
    if any(not math.isfinite(value) or value <= 0.0 or value >= 1.0 for value in probabilities):
        raise ValueError("p must be between 0 and 1")
    return probabilities


def _survreg_quantile_scores(fit: Any, probabilities: list[float]) -> list[float]:
    family = _survreg_distribution_family(fit)
    if family == "logistic":
        return [math.log(value / (1.0 - value)) for value in probabilities]
    if family == "gaussian":
        normal = NormalDist()
        return [normal.inv_cdf(value) for value in probabilities]
    if family == "t":
        df = _survreg_t_fit_degrees_of_freedom(
            getattr(fit, "distribution_parameters", None),
        )
        return [_student_t_ppf(value, df) for value in probabilities]
    return [math.log(-math.log1p(-value)) for value in probabilities]


def _normalize_survreg_distribution_helper(distribution: Any | None) -> str:
    if distribution is None:
        return "weibull"
    if not isinstance(distribution, str):
        raise TypeError("distribution must be a string")
    value = distribution.strip().lower().replace("_", "-")
    if value in {"t", "student", "student-t"}:
        return "t"
    normalized = _normalize_survreg_distribution(distribution)
    return normalized or "weibull"


def _survreg_numeric_vector(values: Any, name: str) -> list[float]:
    return _quantile_vector(values, name)


def _survreg_t_degrees_of_freedom(parms: Any | None) -> float:
    if parms is None:
        raise TypeError("parms is required for distribution='t'")
    values = _survreg_numeric_vector(parms, "parms")
    if len(values) != 1:
        raise ValueError("parms for distribution='t' must be a single degrees-of-freedom value")
    df = values[0]
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("parms for distribution='t' must be a positive finite value")
    return df


def _survreg_t_fit_degrees_of_freedom(parms: Any | None) -> float:
    df = 4.0 if parms is None else _survreg_t_degrees_of_freedom(parms)
    if df <= 2.0:
        raise ValueError("Degrees of freedom must be >=3")
    return df


def _regularized_beta_continued_fraction(a: float, b: float, x: float) -> float:
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            break
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _regularized_beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _regularized_beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_pdf(value: float, df: float) -> float:
    if math.isinf(value):
        return 0.0
    coefficient = math.exp(
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * (math.log(df) + math.log(math.pi))
    )
    return coefficient * (1.0 + value * value / df) ** (-(df + 1.0) / 2.0)


def _student_t_cdf(value: float, df: float) -> float:
    if math.isinf(value):
        return 0.0 if value < 0.0 else 1.0
    if value == 0.0:
        return 0.5
    x = df / (df + value * value)
    ibeta = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return 1.0 - 0.5 * ibeta if value > 0.0 else 0.5 * ibeta


def _student_t_ppf(probability: float, df: float) -> float:
    if probability < 0.0 or probability > 1.0 or math.isnan(probability):
        raise ValueError("p must be between 0 and 1")
    if probability == 0.0:
        return float("-inf")
    if probability == 1.0:
        return float("inf")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_ppf(1.0 - probability, df)
    low = 0.0
    high = 1.0
    while _student_t_cdf(high, df) < probability:
        high *= 2.0
    for _ in range(120):
        mid = (low + high) / 2.0
        if _student_t_cdf(mid, df) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _survreg_t_distribution_values(
    values: list[float],
    means: list[float],
    scales: list[float],
    df: float,
    kind: str,
) -> list[float]:
    result: list[float] = []
    for value, mean, scale in zip(values, means, scales, strict=True):
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must contain positive finite values")
        if kind == "density":
            result.append(_student_t_pdf((value - mean) / scale, df) / scale)
        elif kind == "distribution":
            result.append(_student_t_cdf((value - mean) / scale, df))
        elif kind == "quantile":
            result.append(mean + scale * _student_t_ppf(value, df))
        else:
            raise ValueError("kind must be density, distribution, or quantile")
    return result


def _expand_survreg_distribution_inputs(
    values: Any,
    value_name: str,
    mean: Any,
    scale: Any,
    *,
    target_length: int | None = None,
) -> tuple[list[float], list[float], list[float]]:
    vectors = {
        value_name: _survreg_numeric_vector(values, value_name),
        "mean": _survreg_numeric_vector(mean, "mean"),
        "scale": _survreg_numeric_vector(scale, "scale"),
    }
    main_length = len(vectors[value_name])
    n = (
        target_length
        if target_length is not None
        else main_length
        if main_length > 1
        else max(len(vector) for vector in vectors.values())
    )
    if n < 0:
        raise ValueError("n must be non-negative")

    expanded: dict[str, list[float]] = {}
    for name, vector in vectors.items():
        if len(vector) == n:
            expanded[name] = vector
        elif len(vector) == 1:
            expanded[name] = vector * n
        else:
            raise ValueError(f"{name} must have length 1 or {n}")
    return expanded[value_name], expanded["mean"], expanded["scale"]


def _survreg_distribution_values(
    values: Any,
    value_name: str,
    mean: Any,
    scale: Any,
    distribution: Any | None,
    parms: Any | None,
    kind: str,
) -> list[float]:
    distribution_name = _normalize_survreg_distribution_helper(distribution)
    value_values, mean_values, scale_values = _expand_survreg_distribution_inputs(
        values,
        value_name,
        mean,
        scale,
    )
    if distribution_name == "t":
        return _survreg_t_distribution_values(
            value_values,
            mean_values,
            scale_values,
            _survreg_t_degrees_of_freedom(parms),
            kind,
        )
    return _core.survreg_distribution(
        value_values,
        mean_values,
        scale_values,
        distribution_name,
        kind,
    )


def dsurvreg(
    x: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Density for R ``survreg`` location-scale distributions."""

    return _survreg_distribution_values(x, "x", mean, scale, distribution, parms, "density")


def psurvreg(
    q: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Distribution function for R ``survreg`` location-scale distributions."""

    return _survreg_distribution_values(q, "q", mean, scale, distribution, parms, "distribution")


def qsurvreg(
    p: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Quantiles for R ``survreg`` location-scale distributions."""

    return _survreg_distribution_values(p, "p", mean, scale, distribution, parms, "quantile")


def rsurvreg(
    n: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]:
    """Random draws from R ``survreg`` location-scale distributions."""

    count = _integer_scalar(n, "n")
    if count < 0:
        raise ValueError("n must be non-negative")
    distribution_name = _normalize_survreg_distribution_helper(distribution)
    if count == 0:
        return []
    probabilities = [random.random() for _ in range(count)]  # noqa: S311
    probability_values, mean_values, scale_values = _expand_survreg_distribution_inputs(
        probabilities,
        "p",
        mean,
        scale,
        target_length=count,
    )
    if distribution_name == "t":
        return _survreg_t_distribution_values(
            probability_values,
            mean_values,
            scale_values,
            _survreg_t_degrees_of_freedom(parms),
            "quantile",
        )
    return _core.survreg_distribution(
        probability_values,
        mean_values,
        scale_values,
        distribution_name,
        "quantile",
    )


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
    transform_se = predict_type == "quantile" and _survreg_response_uses_log_transform(fit)
    return _core.survreg_quantile_prediction_se_matrix(
        prediction_rows,
        scales,
        strata,
        variance,
        quantile_scores,
        predictions,
        transform_se,
    )


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


def _cox_strata_labels_for_fit(
    fit: Any,
    strata: Sequence[int] | None,
) -> list[Any] | None:
    if strata is None:
        return None
    design = _formula_design_for_fit(fit)
    if design is None or not design.strata_levels:
        return None
    labels = list(design.strata_levels)
    result: list[Any] = []
    for value in strata:
        idx = int(value)
        if idx < 0 or idx >= len(labels):
            return None
        result.append(labels[idx])
    return result


def _direct_named_prediction_rows(
    newdata: Any,
    coefficient_names: tuple[str, ...],
) -> list[list[float]]:
    columns = [_column(newdata, name) for name in coefficient_names]
    row_count = len(columns[0]) if columns else 0
    if any(len(column) != row_count for column in columns):
        raise ValueError("newdata columns must have the same length")
    return [[float(column[row_idx]) for column in columns] for row_idx in range(row_count)]


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
    coefficient_names = fit.coefficient_names if isinstance(fit, _FormulaFit) else None
    if coefficient_names is not None and (
        isinstance(newdata, Mapping) or hasattr(newdata, "columns")
    ):
        return _direct_named_prediction_rows(newdata, coefficient_names), None
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
    method = _cox_detail_method(model)
    strata_values, baseline_times, cumhaz, varhaz, xbar = _core.cox_expected_baseline_by_stratum(
        times,
        status,
        rows,
        beta,
        weights,
        strata,
        offsets,
        means,
        entry,
        method,
    )
    return {
        int(stratum): _CoxExpectedBaseline(
            times=[float(value) for value in stratum_times],
            cumhaz=[float(value) for value in stratum_cumhaz],
            varhaz=[float(value) for value in stratum_varhaz],
            xbar=[[float(value) for value in row] for row in stratum_xbar],
        )
        for stratum, stratum_times, stratum_cumhaz, stratum_varhaz, stratum_xbar in zip(
            strata_values,
            baseline_times,
            cumhaz,
            varhaz,
            xbar,
            strict=True,
        )
    }


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
    centered_rows: list[list[float]] = []
    start_hazards: list[float] = []
    start_varhazes: list[float] = []
    start_xbars: list[list[float]] = []
    stop_hazards: list[float] = []
    stop_varhazes: list[float] = []
    stop_xbars: list[list[float]] = []
    risks: list[float] = []
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
        risk = _safe_exp(float(linear_predictor))
        predictions.append(max(stop_hazard - start_hazard, 0.0) * risk)
        centered_rows.append(centered_row)
        start_hazards.append(start_hazard)
        start_varhazes.append(start_varhaz)
        start_xbars.append(start_xbar)
        stop_hazards.append(stop_hazard)
        stop_varhazes.append(stop_varhaz)
        stop_xbars.append(stop_xbar)
        risks.append(risk)
    se = _core.cox_interval_cumulative_hazard_se(
        centered_rows,
        start_hazards,
        start_varhazes,
        start_xbars,
        stop_hazards,
        stop_varhazes,
        stop_xbars,
        risks,
        variance,
    )
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
    return _core.step_values_at(times, curve, requested_times, 1.0)


def _step_std_err_at(
    times: list[float],
    curve: list[float],
    requested_times: list[float],
) -> list[float]:
    return _core.step_values_at(times, curve, requested_times, 0.0)


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


def _cox_baseline_survival_curves(
    base_times: list[float],
    base_hazards: list[float],
    linear_predictors: list[float],
    center: float,
    base_strata: list[int] | None = None,
    curve_strata: list[int] | None = None,
    requested_times: list[float] | None = None,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    times, curves, cumhaz = _core.cox_survfit_from_baseline(
        [float(value) for value in base_times],
        [float(value) for value in base_hazards],
        [float(value) for value in linear_predictors],
        float(center),
        None if base_strata is None else [int(value) for value in base_strata],
        None if curve_strata is None else [int(value) for value in curve_strata],
        None if requested_times is None else [float(value) for value in requested_times],
    )
    return (
        [float(value) for value in times],
        [[float(value) for value in curve] for curve in curves],
        [[float(value) for value in curve] for curve in cumhaz],
    )


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
        linear_predictors = _linear_predictors_for_fit(fit, rows, offsets)
        center = _training_linear_predictor_center(fit) if centered else 0.0
        curve_times, curves, _ = _cox_baseline_survival_curves(
            [float(value) for value in base_times],
            [float(value) for value in base_hazards],
            linear_predictors,
            center,
            [int(value) for value in base_strata],
            prediction_strata,
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
    curve_times, curves, _ = _cox_baseline_survival_curves(
        [float(value) for value in curve_times],
        [float(value) for value in hazards],
        linear_predictors,
        center,
    )
    return curve_times, curves


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
    if len(unique_strata) <= 1:
        return None
    if rows is None:
        if n_curves == len(unique_strata):
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
        curve_hazards = _core.step_values_at(result.time, hazard_values, times, 0.0)
        expanded_cumhaz.append(curve_hazards)
        expanded_surv.append([_clamp_probability(_safe_exp(-hazard)) for hazard in curve_hazards])

    return CoxSurvfitResult(
        time=times,
        surv=expanded_surv,
        cumhaz=expanded_cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
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
    kept_times, conditioned_surv, conditioned_cumhaz = _core.condition_cox_survfit_curves(
        times,
        [[float(value) for value in curve] for curve in result.cumhaz],
        float(t0),
        include_time0,
        start_time is not None,
        _SURVFIT_TIME_EPSILON,
    )

    return CoxSurvfitResult(
        time=kept_times,
        surv=conditioned_surv,
        cumhaz=conditioned_cumhaz,
        linear_predictors=result.linear_predictors,
        centered=result.centered,
        strata=result.strata,
        strata_labels=result.strata_labels,
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
        strata_labels=result.strata_labels,
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
    curve_strata_labels = None
    if curve_strata is not None:
        curve_strata_labels = (
            _cox_strata_labels_for_fit(fit, curve_strata)
            if rows is None
            else list(range(1, len(curve_strata) + 1))
        )
    basehaz_with_strata = getattr(fit, "basehaz_with_strata", None)
    if curve_strata is not None and basehaz_with_strata is not None:
        base_times, base_hazards, base_strata = basehaz_with_strata(centered)
        _, _, cumhaz = _cox_baseline_survival_curves(
            [float(value) for value in base_times],
            [float(value) for value in base_hazards],
            linear_predictors,
            center,
            [int(value) for value in base_strata],
            curve_strata,
            times,
        )
    else:
        baseline_times, baseline_hazards = fit.basehaz(centered)
        _, _, cumhaz = _cox_baseline_survival_curves(
            [float(value) for value in baseline_times],
            [float(value) for value in baseline_hazards],
            linear_predictors,
            center,
            requested_times=times,
        )
    result = CoxSurvfitResult(
        time=times,
        surv=[[float(value) for value in curve] for curve in curves],
        cumhaz=cumhaz,
        linear_predictors=linear_predictors,
        centered=centered,
        strata=curve_strata,
        strata_labels=curve_strata_labels,
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
    if terms.offsets:
        raise ValueError("Offset terms not allowed")
    if not terms.covariates:
        raise ValueError("concordance formula requires a risk score")

    columns: list[list[float]] = []
    names: list[str] = []
    for term in terms.covariates:
        design_term = _fit_design_term(data, term, n)
        term_columns = _design_term_columns(data, design_term, n)
        columns.extend(term_columns)
        names.extend(_design_term_output_names(design_term))

    if not columns:
        raise ValueError("concordance formula requires a risk score")
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


def _concordance_rank_row_dicts(
    rows: list[tuple[float, float, float, float]],
    display_by_core_time: dict[float, float] | None = None,
) -> list[dict[str, float]]:
    return [
        {
            "time": float(display_by_core_time.get(time, time) if display_by_core_time else time),
            "rank": float(rank),
            "timewt": float(time_weight),
            "casewt": float(case_weight),
        }
        for time, rank, time_weight, case_weight in rows
    ]


@dataclass(frozen=True)
class _RightConcordanceData:
    times: list[float]
    status: list[int]
    display_by_core_time: dict[float, float] | None


@dataclass(frozen=True)
class _CountingConcordanceData:
    start: list[float]
    stop: list[float]
    status: list[int]


def _unsupported_concordance_response() -> NoReturn:
    raise NotImplementedError(
        "concordance currently supports right-censored and counting Surv responses"
    )


def _right_concordance_data(
    response: Surv,
    timefix: bool,
    ymin: float | None,
    ymax: float | None,
) -> _RightConcordanceData:
    times = _survdiff_timefix_values(list(response.time), timefix)
    status = list(response.event)
    times, status = _concordance_bounded_times_and_status(times, status, ymin, ymax)
    core_times, display_by_core_time = _concordance_core_time_values(times, timefix)
    return _RightConcordanceData(core_times, status, display_by_core_time)


def _counting_concordance_data(
    response: Surv,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
    *,
    preapply_timefix: bool,
) -> _CountingConcordanceData:
    if timewt in {"S/G", "n/G2"}:
        raise ValueError("S/G and n/G2 timewt options are not supported for counting-process data")
    if response.start is None:
        raise ValueError("counting-process concordance requires start times")

    start = list(response.start)
    stop = list(response.time)
    if preapply_timefix and timefix:
        start, stop = _timefix_vectors(start, stop)
    status = list(response.event)
    stop, status = _concordance_bounded_times_and_status(stop, status, ymin, ymax)
    return _CountingConcordanceData(start, stop, status)


def _single_concordance_ranks(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> list[dict[str, float]]:
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_rank_row_dicts(
            _core.concordance_rank_rows(
                data.times,
                data.status,
                risk_values,
                case_weights,
                timewt,
            ),
            data.display_by_core_time,
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_rank_row_dicts(
            _core.counting_concordance_rank_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


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
    strata_codes = _encode_groups(strata, len(response))
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_rank_row_dicts(
            _core.stratified_concordance_rank_rows(
                data.times,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
            ),
            data.display_by_core_time,
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_rank_row_dicts(
            _core.stratified_counting_concordance_rank_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


def _concordance_influence_result(
    result: tuple[list[list[float]], list[float], float],
) -> tuple[list[list[float]], list[float], float]:
    influence_rows, dfbeta, variance = result
    return (
        [[float(value) for value in row] for row in influence_rows],
        [float(value) for value in dfbeta],
        float(variance),
    )


def _single_concordance_influence(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[list[list[float]], list[float], float | None]:
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_influence_result(
            _core.concordance_influence_rows(
                data.times,
                data.status,
                risk_values,
                case_weights,
                timewt,
            )
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_influence_result(
            _core.counting_concordance_influence_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


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
    strata_codes = _encode_groups(strata, len(response))
    case_weights = None if weights is None else list(weights)
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return _concordance_influence_result(
            _core.stratified_concordance_influence_rows(
                data.times,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
            )
        )
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=True,
        )
        return _concordance_influence_result(
            _core.stratified_counting_concordance_influence_rows(
                data.start,
                data.stop,
                data.status,
                risk_values,
                strata_codes,
                case_weights,
                timewt,
                False,
            )
        )
    return _unsupported_concordance_response()


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
        data = _right_concordance_data(response, timefix, ymin, ymax)
        summary = _core.concordance_summary(
            data.times,
            data.status,
            risk_values,
            weights,
            timewt,
        )
        summary["n_event"] = float(sum(1 for event in data.status if event == 1))
        return summary
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=False,
        )
        summary = _core.counting_concordance_summary(
            data.start,
            data.stop,
            data.status,
            risk_values,
            weights,
            timewt,
            timefix,
        )
        summary["n_event"] = float(sum(1 for event in data.status if event == 1))
        return summary
    return _unsupported_concordance_response()


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

    strata_codes = _encode_groups(strata, len(response))
    if response.type == "right":
        data = _right_concordance_data(response, timefix, ymin, ymax)
        return {
            key: float(value)
            for key, value in _core.stratified_concordance_summary(
                data.times,
                data.status,
                risk_values,
                strata_codes,
                weights,
                timewt,
            ).items()
        }
    if response.type == "counting":
        data = _counting_concordance_data(
            response,
            timefix,
            timewt,
            ymin,
            ymax,
            preapply_timefix=False,
        )
        return {
            key: float(value)
            for key, value in _core.stratified_counting_concordance_summary(
                data.start,
                data.stop,
                data.status,
                risk_values,
                strata_codes,
                weights,
                timewt,
                timefix,
            ).items()
        }
    return _unsupported_concordance_response()


def _concordance_time_multiplier(
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
        if censoring_survival > 0.0:
            return total_weight * survival / (censoring_survival * nrisk)
        return 0.0
    if timewt == "n/G2":
        if censoring_survival > 0.0:
            return 1.0 / (censoring_survival * censoring_survival)
        return 0.0
    if timewt == "I":
        return 1.0 / nrisk
    return 1.0


def _right_concordance_time_multipliers(
    time: list[float],
    status: list[int],
    weights: list[float] | None,
    timewt: str,
) -> dict[float, float]:
    if timewt == "n":
        return {time[idx]: 1.0 for idx, event in enumerate(status) if event == 1}

    case_weights = [1.0] * len(time) if weights is None else weights
    total_weight = math.fsum(case_weights)
    survival = 1.0
    censoring_survival = 1.0
    multipliers: dict[float, float] = {}
    for event_time in sorted(set(time)):
        indices = [idx for idx, value in enumerate(time) if value == event_time]
        nrisk = math.fsum(
            weight for weight, value in zip(case_weights, time, strict=True) if value >= event_time
        )
        death_weight = math.fsum(case_weights[idx] for idx in indices if status[idx] == 1)
        censor_weight = math.fsum(case_weights[idx] for idx in indices if status[idx] != 1)
        if death_weight > 0.0:
            multipliers[event_time] = _concordance_time_multiplier(
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


def _right_concordance_tie_counts(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[float, float, float]:
    data = _right_concordance_data(response, timefix, ymin, ymax)
    case_weights = [1.0] * len(data.times) if weights is None else weights
    multipliers = _right_concordance_time_multipliers(data.times, data.status, weights, timewt)
    tied_x = 0.0
    tied_y = 0.0
    tied_xy = 0.0
    for left in range(len(data.times)):
        for right in range(left + 1, len(data.times)):
            pair_weight = case_weights[left] * case_weights[right]
            same_event_time = (
                data.status[left] == 1
                and data.status[right] == 1
                and data.times[left] == data.times[right]
            )
            if same_event_time:
                pair_weight *= multipliers.get(data.times[left], 0.0)
                if pair_weight <= 0.0:
                    continue
                if abs(risk_values[left] - risk_values[right]) < 1e-12:
                    tied_xy += pair_weight
                else:
                    tied_y += pair_weight
                continue

            if data.status[left] == 1 and data.times[left] < data.times[right]:
                event_idx, risk_idx = left, right
            elif data.status[right] == 1 and data.times[right] < data.times[left]:
                event_idx, risk_idx = right, left
            else:
                continue
            pair_weight *= multipliers.get(data.times[event_idx], 0.0)
            if pair_weight > 0.0 and abs(risk_values[event_idx] - risk_values[risk_idx]) < 1e-12:
                tied_x += pair_weight
    return tied_x, tied_y, tied_xy


def _concordance_tie_counts(
    response: Surv,
    risk_values: list[float],
    weights: list[float] | None,
    strata: Any | None,
    timefix: bool,
    timewt: str,
    ymin: float | None,
    ymax: float | None,
) -> tuple[float, float, float]:
    if response.type != "right":
        return 0.0, 0.0, 0.0
    if strata is None:
        return _right_concordance_tie_counts(
            response,
            risk_values,
            weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
    strata_codes = _encode_groups(strata, len(response))
    totals = [0.0, 0.0, 0.0]
    for indices in _group_indices(strata_codes, len(response)).values():
        group_response = _subset_surv(response, indices)
        group_risk = [risk_values[idx] for idx in indices]
        group_weights = [weights[idx] for idx in indices] if weights is not None else None
        counts = _right_concordance_tie_counts(
            group_response,
            group_risk,
            group_weights,
            timefix,
            timewt,
            ymin,
            ymax,
        )
        for idx, value in enumerate(counts):
            totals[idx] += value
    return tuple(totals)


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
    tied_x, tied_y, tied_xy = _concordance_tie_counts(
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
        tied_x=tied_x,
        tied_y=tied_y,
        tied_xy=tied_xy,
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
        tied_x=[float(result.tied_x) for result in results],
        tied_y=[float(result.tied_y) for result in results],
        tied_xy=[float(result.tied_xy) for result in results],
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
    formula_input = isinstance(response, str)
    effective_reverse_scores = reverse_scores

    if formula_input:
        effective_reverse_scores = not reverse_scores
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
            effective_reverse_scores,
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
                tied_x=result.tied_x,
                tied_y=result.tied_y,
                tied_xy=result.tied_xy,
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
        effective_reverse_scores,
        fix_time,
        time_weight,
        lower_bound,
        upper_bound,
        influence_value,
        include_ranks,
    )


def survConcordance(  # noqa: N802
    formula: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    **kwargs: Any,
) -> ConcordanceResult:
    """Deprecated R-compatible alias for ``concordance``."""

    na_action = _pop_dotted_keyword(kwargs, "na.action", "na_action", na_action, "fail")
    warnings.warn(
        "survConcordance is deprecated; use concordance instead",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.setdefault("reverse", True)
    return concordance(
        formula,
        data=data,
        weights=weights,
        subset=subset,
        na_action=na_action,
        **kwargs,
    )


def _survConcordance_legacy_stats(  # noqa: N802
    result: ConcordanceResult,
) -> dict[str, float]:
    if isinstance(result.concordance, list):
        raise ValueError("survConcordance.fit expects a single score vector")
    concordant = float(result.concordant)
    comparable = float(result.comparable)
    discordant = max(comparable - concordant, 0.0)
    variance = result.variance
    if isinstance(variance, list):
        variance = variance[0] if variance else None
    std_cd = math.nan
    if variance is not None and math.isfinite(float(variance)) and variance >= 0.0:
        std_cd = math.sqrt(float(variance)) * 2.0 * comparable
    return {
        "concordant": concordant,
        "discordant": discordant,
        "tied.risk": 0.0,
        "tied.time": 0.0,
        "std(c-d)": std_cd,
    }


def survConcordance_fit(  # noqa: N802
    y: Any,
    x: Any,
    strata: Any | None = None,
    weight: Any | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Deprecated R-compatible ``survConcordance.fit`` statistics helper."""

    timefix = _pop_dotted_keyword(kwargs, "time.fix", "timefix", True, True)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"survConcordance.fit got unexpected keyword argument(s): {unexpected}")
    if not isinstance(y, Surv):
        raise TypeError("y must be a Surv object")
    result = _single_score_concordance_result(
        y,
        _float_vector(x, "x"),
        _concordance_weight_values(weight, len(y)),
        None if strata is None else _materialize_labels(strata, "strata"),
        None,
        False,
        _normalize_bool_option(timefix, "timefix"),
        "n",
        None,
        None,
        1,
        False,
    )
    warnings.warn(
        "survConcordance.fit is deprecated; use concordance instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return _survConcordance_legacy_stats(result)


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
                distribution_parameter=(
                    _survreg_t_fit_degrees_of_freedom(
                        getattr(fit, "distribution_parameters", None),
                    )
                    if _survreg_distribution_family(fit) == "t"
                    else None
                ),
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
    direct_coefficient_names: tuple[str, ...] | None = None
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
    if formula_design is None:
        direct_coefficient_names = _matrix_input_column_names(x)
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
    direct_coefficient_names = _validated_matrix_column_names(direct_coefficient_names, rows)
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
    if not singular_ok_value and any(_cox_alias_mask(fit)):
        raise ValueError(
            "coxph design matrix is singular; use singular_ok=True to allow dependent covariates"
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
        or direct_coefficient_names is not None
        or case_weights is not None
        or robust_variance is not None
        or model_frame is not None
    ):
        return _FormulaFit(
            fit,
            formula_design,
            formula=formula_string,
            coefficient_names=direct_coefficient_names,
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
    direct_coefficient_names: tuple[str, ...] | None = None
    matrix_input = response is None and time is not None and status is not None
    if matrix_input:
        response_time = _float_vector(time, "time")
        response_time2 = _float_vector(time2, "time2") if time2 is not None else None
        response_status = _materialize_1d(status, "status")
        matrix_values = covariates if covariates is not None else x
        direct_coefficient_names = _matrix_input_column_names(matrix_values)
        rows = _as_rows(matrix_values, "covariates")
        direct_coefficient_names = _validated_matrix_column_names(
            direct_coefficient_names,
            rows,
        )
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
        if formula_design is None:
            direct_coefficient_names = _matrix_input_column_names(
                covariates if covariates is not None else x,
            )
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
        direct_coefficient_names = _validated_matrix_column_names(
            direct_coefficient_names,
            rows,
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
    distribution_parameter = None
    if distribution_name == "t":
        distribution_parameter = _survreg_t_fit_degrees_of_freedom(parms)
    elif parms is not None:
        raise ValueError("parms is only supported for distribution='t'")
    exponential_fixed_scale = distribution_name == "exponential"
    rayleigh_fixed_scale = distribution_name == "rayleigh"
    if exponential_fixed_scale and scale_value > 0.0:
        warnings.warn(
            "Exponential has a fixed scale; user specified value ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    if rayleigh_fixed_scale and scale_value > 0.0:
        warnings.warn(
            "Rayleigh has a fixed scale; user specified value ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    offset_values = _optional_float_vector(offsets if offsets is not None else offset, "offsets", n)
    weight_values = _optional_float_vector(weights, "weights", n)
    case_weights = weight_values if explicit_weights else None
    strata_values = _encode_groups(strata, n) if strata is not None else None
    if (
        (scale_value > 0.0 or exponential_fixed_scale or rayleigh_fixed_scale)
        and strata_values is not None
        and len(set(strata_values)) > 1
    ):
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

    fixed_scale = (
        0.5
        if rayleigh_fixed_scale
        else 1.0
        if exponential_fixed_scale
        else (scale_value if scale_value > 0.0 else None)
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
        fixed_scale=fixed_scale,
        distribution_parameter=distribution_parameter,
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
            coefficient_names=direct_coefficient_names,
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
            or direct_coefficient_names is not None
            or case_weights is not None
            or robust_variance is not None
            or model_frame is not None
            or score_values is not None
        )
        else fit
    )
