"""Result containers and formula/design dataclasses shared by the R-style API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from operator import index
from typing import TYPE_CHECKING, Any

from .. import _survival as _core

if TYPE_CHECKING:
    from ._surv import Surv


# _core class re-exports.  ``SurvObrienResult`` and ``YatesPairwiseResult`` were removed from
# the Rust core (``survobrien`` returns a ``SurvObrienExpansion``; ``yates`` returns one
# ``YatesResult`` carrying both the global and the pairwise contrasts).  They are aliased here
# only so that ``survival.r`` keeps importing until its survobrien/yates wrappers are ported.
FineGrayOutput = _core.FineGrayOutput
RateTable = _core.RateTable
SurvObrienResult = _core.SurvObrienExpansion
TcutResult = _core.TcutResult
YatesPairwiseResult = _core.YatesResult
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
    categorical_wrapper: str | None = None
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
    full: bool = False


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
    term_assignments: tuple[int, ...] = ()
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
    case_weight_column: str | None = None
    robust_variance: list[list[float]] | None = None
    naive_variance: list[list[float]] | None = None
    cluster: list[Any] | None = None
    id_values: list[Any] | None = None
    id_column: str | None = None
    x_matrix: list[list[float]] | None = None
    y_response: Surv | None = None
    model_frame: dict[str, Any] | None = None
    score_values: list[float] | None = None
    conditional_logistic: bool = False
    n_observations: int | None = None

    def __getattr__(self, name: str) -> Any:
        if self.conditional_logistic and name in {
            "basehaz",
            "basehaz_with_strata",
            "survival_curve",
            "survival_curve_with_strata",
        }:
            raise ValueError("predicted survival curves are not defined for a clogit model")
        if self.conditional_logistic and getattr(self.fit, "method", None) == "exact":
            unavailable = {
                "score_residuals": "score",
                "schoenfeld_residuals": "schoenfeld",
                "scaled_schoenfeld_residuals": "scaledsch",
                "dfbeta": "dfbeta",
                "dfbetas": "dfbetas",
            }
            if name in unavailable:
                raise ValueError(
                    f"{unavailable[name]} residuals are not available for the exact method"
                )
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
        if name == "n" and self.n_observations is not None:
            return self.n_observations
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
class CchModelResult:
    """Formula metadata and R-style aliases for a native case-cohort fit."""

    fit: Any
    design: _FormulaDesign
    formula: str
    coefficient_names: tuple[str, ...]
    response: Surv
    id_values: list[Any]
    subcohort: list[int]
    stratum_values: list[Any] | None
    cohort_sizes: list[int]

    def __getattr__(self, name: str) -> Any:
        aliases = {
            "var": "information_matrix",
            "naive_var": "naive_information_matrix",
            "phase2var": "phase2_variance",
        }
        if name == "coef":
            return list(self.fit.coefficients[0])
        if name == "x":
            return self.fit.covariates
        if name == "y":
            return self.response
        if name == "id":
            return self.id_values
        if name == "stratum":
            return self.stratum_values
        if name == "stratified":
            return self.fit.stratified
        return getattr(self.fit, aliases.get(name, name))

    @property
    def coefficients(self) -> list[list[float]]:
        return self.fit.coefficients

    @property
    def information_matrix(self) -> list[list[float]]:
        return self.fit.information_matrix

    @property
    def variance_matrix(self) -> list[list[float]]:
        return self.fit.information_matrix

    @property
    def naive_information_matrix(self) -> list[list[float]]:
        return self.fit.naive_information_matrix


@dataclass(frozen=True)
class AaregModelResult:
    n: list[int]
    times: list[float]
    n_risk: list[float]
    coefficient: list[list[float]]
    coefficient_names: list[str]
    test_statistic: list[float]
    test_statistic_names: list[str]
    test_variance: list[list[float]]
    test: str
    time_weights: list[list[float]]
    dfbeta: list[list[list[float]]] | None = None
    robust_test_variance: list[list[float]] | None = None
    formula: str | None = None
    weights: list[float] | None = None
    cluster: list[Any] | None = None
    cluster_levels: list[Any] | None = None
    model: dict[str, Any] | None = None
    x: list[list[float]] | None = None
    y: Surv | None = None

    @property
    def nrisk(self) -> list[float]:
        return self.n_risk

    @property
    def coefficients(self) -> list[list[float]]:
        return self.coefficient

    @property
    def tweight(self) -> list[list[float]]:
        return self.time_weights

    @property
    def test_var(self) -> list[list[float]]:
        return self.test_variance

    @property
    def test_var2(self) -> list[list[float]] | None:
        return self.robust_test_variance


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
    conditional_variance: float | list[float] | None = None
    score_names: list[str] | None = None

    @property
    def c_index(self) -> float | list[float]:
        return self.concordance

    @property
    def var(self) -> float | list[float | None] | None:
        return self.variance

    @property
    def cvar(self) -> float | list[float] | None:
        return self.conditional_variance


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


class FineGrayFrame(dict[str, list[Any]]):
    """Column-oriented Fine-Gray expansion with the selected endpoint attached."""

    event: Any

    def __init__(
        self,
        columns: Mapping[str, Sequence[Any]] | None = None,
        *,
        event: Any = None,
    ) -> None:
        super().__init__()
        if columns is not None:
            for name, values in columns.items():
                self[str(name)] = values if isinstance(values, list) else list(values)
        self.event = event

    def copy(self) -> FineGrayFrame:
        """Return a shallow copy that retains the selected endpoint."""

        return FineGrayFrame(self, event=self.event)


@dataclass(frozen=True)
class TMergeOperation:
    """A time-dependent update consumed by :func:`tmerge`."""

    kind: str
    time: Any
    value: Any | None = None
    default: Any | None = None
    censor: Any | None = None


@dataclass(frozen=True)
class TMergeFrame(Mapping[str, list[Any]]):
    """Column-oriented start/stop data with retained ``tmerge`` metadata."""

    columns: dict[str, list[Any]]
    tname: dict[str, str]
    tevent: dict[str, Any]
    tdcvar: tuple[str, ...]
    tcount: dict[str, dict[str, int]]

    def __getitem__(self, key: str) -> list[Any]:
        return self.columns[key]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def copy(self) -> TMergeFrame:
        return TMergeFrame(
            columns={name: list(values) for name, values in self.columns.items()},
            tname=dict(self.tname),
            tevent=dict(self.tevent),
            tdcvar=tuple(self.tdcvar),
            tcount={name: dict(counts) for name, counts in self.tcount.items()},
        )


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
    strata: list[Any] | None = None

    def subset(
        self,
        indices: Sequence[int],
        *,
        include_global: bool = False,
    ) -> CoxZPHResult:
        """Return diagnostics for the selected variables."""

        selected = [index(value) for value in indices]
        width = len(self.variable_names)
        if any(value < 0 or value >= width for value in selected):
            raise IndexError("cox_zph variable index out of range")

        y = [[row[col_idx] for col_idx in selected] for row in self.y]
        keep = list(range(len(y)))
        if self.strata is not None:
            keep = [
                row_idx
                for row_idx, row in enumerate(y)
                if not all(math.isnan(value) for value in row)
            ]

        return CoxZPHResult(
            variable_names=[self.variable_names[col_idx] for col_idx in selected],
            chi2_values=[self.chi2_values[col_idx] for col_idx in selected],
            df=[self.df[col_idx] for col_idx in selected],
            p_values=[self.p_values[col_idx] for col_idx in selected],
            x=[self.x[row_idx] for row_idx in keep],
            time=[self.time[row_idx] for row_idx in keep],
            y=[y[row_idx] for row_idx in keep],
            var=[[self.var[row][col] for col in selected] for row in selected],
            transform=self.transform,
            global_chi2=self.global_chi2 if include_global else None,
            global_df=self.global_df if include_global else None,
            global_p_value=self.global_p_value if include_global else None,
            strata=(
                [self.strata[row_idx] for row_idx in keep] if self.strata is not None else None
            ),
        )

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
class SurvfitMultiStateResult:
    """Aalen--Johansen state-probability curves from a multi-state response."""

    time: list[float]
    n_risk: list[list[float]]
    n_event: list[list[float]]
    n_censor: list[list[float]]
    pstate: list[list[float]]
    cumhaz: list[list[float]]
    states: tuple[str, ...]
    transitions: tuple[tuple[int, int], ...]
    p0: list[float]
    t0: float
    n: int
    n_id: int
    std_err: list[list[float]] | None = None
    std_err0: list[float] | None = None
    std_chaz: list[list[float]] | None = None
    std_auc: list[list[float]] | None = None
    conf_lower: list[list[float]] | None = None
    conf_upper: list[list[float]] | None = None
    n_risk_count: list[list[float]] | None = None
    n_event_count: list[list[float]] | None = None
    n_censor_count: list[list[float]] | None = None
    n_enter: list[list[float]] | None = None
    n_enter_count: list[list[float]] | None = None
    n_transition: list[list[float]] = field(default_factory=list)
    n_transition_count: list[list[float]] | None = None
    model: dict[str, Any] | None = None
    surv_type: str = "mright"
    conf_type: str = "log"
    conf_level: float = 0.95
    oldstate: tuple[str, ...] | None = None
    p0_fixed: bool = False
    timefix: bool = True
    influence_state: list[list[float]] | None = None
    influence_state0: list[float] | None = None
    influence_chaz: list[list[float]] | None = None
    influence_auc: list[list[float]] | None = None

    def __iter__(self):
        yield self.time
        yield self.pstate

    @property
    def surv(self) -> list[list[float]]:
        return self.pstate

    @property
    def estimate(self) -> list[list[float]]:
        return self.pstate

    @property
    def state_probabilities(self) -> list[list[float]]:
        return self.pstate

    @property
    def cumulative_hazard(self) -> list[list[float]]:
        return self.cumhaz

    @property
    def cumulative_hazard_std_err(self) -> list[list[float]] | None:
        return self.std_chaz

    @property
    def transition_labels(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.states[source], self.states[target]) for source, target in self.transitions
        )


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
    variance = getattr(fit, "naive_variance", None)
    if variance is None:
        variance = getattr(fit, "information_matrix", None)
    if variance is None:
        raise TypeError("model does not expose coefficient variance")
    matrix = [list(row) for row in variance]
    return _core.cox_dfbeta_from_score_residuals(score, matrix, scaled)


def _cox_beta(fit: Any) -> list[float]:
    coefficients = getattr(fit, "coefficients", None)
    if coefficients is None:
        raise TypeError("model does not expose fitted coefficients")
    values = list(coefficients)
    if values and isinstance(values[0], list | tuple):
        return [float(value) for value in values[0]]
    return [float(value) for value in values]
