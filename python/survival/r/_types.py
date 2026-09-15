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
    """Transitional survreg wrapper (still built by ``_survreg.survreg``).

    Cox fits no longer use it: they are ``survival.r._coxph.CoxphModel`` objects.  This
    class goes away once ``_survreg``/``_misc`` move to their typed wrapper.
    """

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
        return getattr(self.fit, name)

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
    """R's ``cch`` object: the engine fit plus the formula metadata ``cch()`` keeps."""

    fit: _core.CchFitResult
    formula: str
    design: _FormulaDesign
    coef_names: tuple[str, ...]
    y: Surv
    id: tuple[Any, ...]
    subcoh: tuple[int, ...]
    stratum: tuple[Any, ...] | None
    cohort_size: tuple[int, ...]
    subcohort_size: tuple[int, ...]

    @property
    def coefficients(self) -> list[float]:
        return list(self.fit.coefficients)

    @property
    def var(self) -> list[list[float]]:
        return [list(row) for row in self.fit.var]

    @property
    def naive_var(self) -> list[list[float]]:
        return [list(row) for row in self.fit.naive_var]

    @property
    def phase2var(self) -> list[list[float]]:
        return [list(row) for row in self.fit.phase2var]

    @property
    def method(self) -> str:
        return str(self.fit.method)

    @property
    def stratified(self) -> bool:
        return bool(self.fit.stratified)


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
    """R's ``concordance`` object.

    ``concordance``, ``var``, ``cvar`` and ``dfbeta`` are scalars/vectors for one
    predictor and vectors/matrices for several, exactly as ``concordancefit`` returns
    them; ``count`` is one named row of ``concordant``/``discordant``/``tied.x``/
    ``tied.y``/``tied.xy`` (a list of rows when several predictors or kept strata),
    and ``names`` labels those rows (predictor names or stratum levels).
    """

    concordance: float | list[float]
    count: dict[str, float] | list[dict[str, float]]
    n: int
    names: list[str] | None = None
    var: float | list[list[float]] | None = None
    cvar: float | list[float] | None = None
    dfbeta: list[float] | list[list[float]] | None = None
    influence: list[list[float]] | list[list[list[float]]] | None = None
    ranks: list[dict[str, float]] | list[list[dict[str, float]]] | None = None
    formula: str | None = None

    @property
    def std(self) -> float | list[float] | None:
        """``sqrt(var)`` (``sqrt(diag(var))`` for several predictors), as ``print`` reports."""

        if self.var is None:
            return None
        if isinstance(self.var, list):
            return [math.sqrt(self.var[idx][idx]) for idx in range(len(self.var))]
        return math.sqrt(self.var)


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
    """R's ``cox.zph`` object: ``table`` rows (``name``, ``chisq``, ``df``, ``p``), the
    transformed times ``x``, the death times ``time``, the scaled Schoenfeld residual
    matrix ``y`` (one column per term, named by ``names``) and its variance ``var``."""

    table: list[dict[str, float | int | str]]
    x: list[float]
    time: list[float]
    y: list[list[float]]
    var: list[list[float]]
    transform: str
    names: tuple[str, ...]
    strata: list[Any] | None = None

    def subset(self, indices: Sequence[int]) -> CoxZPHResult:
        """``[.cox.zph``: keep the selected terms (0-based), dropping deaths that
        played no role in them (strata by covariate interactions)."""

        selected = [index(value) for value in indices]
        if any(value < 0 or value >= len(self.names) for value in selected):
            raise IndexError("invalid variable requested")
        y = [[row[col] for col in selected] for row in self.y]
        keep = list(range(len(y)))
        if self.strata is not None:
            keep = [idx for idx, row in enumerate(y) if not all(math.isnan(v) for v in row)]
        return CoxZPHResult(
            table=[self.table[col] for col in selected],
            x=[self.x[idx] for idx in keep],
            time=[self.time[idx] for idx in keep],
            y=[y[idx] for idx in keep],
            var=[[self.var[row][col] for col in selected] for row in selected],
            transform=self.transform,
            names=tuple(self.names[col] for col in selected),
            strata=None if self.strata is None else [self.strata[idx] for idx in keep],
        )


@dataclass(frozen=True)
class CoxPHDetailResult:
    """R's ``coxph.detail`` list: one entry per unique event time (``time``,
    ``nevent``, ``nrisk``, ``hazard``, ``varhaz``, ``wtrisk``, ``means``, ``score``,
    ``imat``) plus the ``x``/``y`` data in ``rorder`` order."""

    time: list[float]
    nevent: list[int]
    nrisk: list[int]
    hazard: list[float]
    varhaz: list[float]
    wtrisk: list[float]
    means: list[list[float]]
    score: list[list[float]]
    imat: list[list[list[float]]]
    x: list[list[float]]
    y: list[list[float]]
    strata: dict[str, int] | None = None
    riskmat: list[list[int]] | None = None
    sortorder: list[int] | None = None
    weights: list[float] | None = None
    nevent_wt: list[float] | None = None
    nrisk_wt: list[float] | None = None

    @property
    def cumhaz(self) -> list[float]:
        total = 0.0
        values: list[float] = []
        for increment in self.hazard:
            total += increment
            values.append(total)
        return values


@dataclass(frozen=True)
class CoxPHWTestResult:
    """R's ``coxph.wtest`` list."""

    test: list[float]
    df: int
    solve: list[float] | list[list[float]] | float


@dataclass(frozen=True)
class CoxBaseHazardResult:
    """R's ``basehaz`` data frame: ``hazard`` (one column per newdata row when
    ``newdata`` had several), ``time`` and the ``strata`` labels."""

    hazard: list[float] | list[list[float]]
    time: list[float]
    strata: list[str] | None = None


@dataclass(frozen=True)
class CoxSurvfitResult:
    """R's ``survfit.coxph`` object (class ``survfitcox``).

    ``time``/``n_risk``/``n_event``/``n_censor`` are the strata blocks laid end to end
    (``strata`` maps each block's name to its length, ``n`` is the size of each);
    ``surv``, ``cumhaz``, ``std_err``, ``std_chaz``, ``lower`` and ``upper`` are vectors
    for one curve, or ``ntime x ncurve`` matrices (one column per ``newdata`` row).
    """

    n: list[int]
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    surv: list[float] | list[list[float]]
    cumhaz: list[float] | list[list[float]]
    type: str
    strata: dict[str, int] | None = None
    std_err: list[float] | list[list[float]] | None = None
    std_chaz: list[float] | list[list[float]] | None = None
    lower: list[float] | list[list[float]] | None = None
    upper: list[float] | list[list[float]] | None = None
    logse: bool = True
    conf_type: str = "none"
    conf_int: float | None = None
    start_time: float | None = None
    newdata: Any | None = None

    @property
    def ncurve(self) -> int:
        return len(self.surv[0]) if self.surv and isinstance(self.surv[0], list) else 1


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


def _cox_beta(fit: Any) -> list[float]:
    coefficients = getattr(fit, "coefficients", None)
    if coefficients is None:
        raise TypeError("model does not expose fitted coefficients")
    values = list(coefficients)
    if values and isinstance(values[0], list | tuple):
        return [float(value) for value in values[0]]
    return [float(value) for value in values]
