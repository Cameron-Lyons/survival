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


# _core class re-exports.
FineGrayOutput = _core.FineGrayOutput
RateTable = _core.RateTable
TcutResult = _core.TcutResult


class _MissingArgument:
    __slots__ = ()

    def __repr__(self) -> str:
        return "..."


_MISSING = _MissingArgument()


@dataclass(frozen=True)
class _CovariateTerm:
    """One factor of a formula term.

    ``call`` carries the text of an opaque categorising call (``tcut(...)``,
    ``cut(...)``) that only ``pyears`` evaluates; ``column`` is then the data
    column it reads (or the first column of its ``arithmetic`` argument).
    """

    column: str
    categorical: bool = False
    categorical_wrapper: str | None = None
    transform: str | None = None
    arithmetic: str | None = None
    call: str | None = None


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


@dataclass(frozen=True)
class _PenaltyDesignTerm:
    """A fitted penalty basis, including the state needed to transform new data."""

    term: _CovariateTerm
    columns: tuple[str, ...]
    names: tuple[str, ...]
    penalty: Any
    degree: int = 3
    boundary: tuple[float, float] | None = None
    levels: tuple[Any, ...] = ()
    intercept: bool = False


_SingleDesignTerm = _NumericDesignTerm | _CategoricalDesignTerm | _PenaltyDesignTerm


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
    """The left-hand side of a formula: a ``Surv(...)`` call, or a plain numeric response.

    ``surv`` is ``False`` for a formula such as ``time ~ 1`` (``pyears``, ``survexp``),
    whose single argument is the follow-up expression.
    """

    arguments: tuple[str, ...]
    columns: tuple[str, ...]
    type: str | None
    origin: float = 0.0
    surv: bool = True

    @property
    def name(self) -> str:
        """R's name of the response column in the model frame."""

        return f"Surv({', '.join(self.arguments)})" if self.surv else self.arguments[0]


@dataclass(frozen=True)
class ModelFrame:
    """R's ``model.frame`` for a survival formula.

    ``data`` is the caller's data after ``subset`` and the ``na.action``; ``response``
    is the ``Surv`` response (``y`` a plain numeric response such as ``time ~ 1``, or
    both ``None`` for ``~ x``); the R-style extra arguments (``weights``, ``offset``,
    ``id``, ``cluster``, ``istate``) are row aligned with it.
    """

    formula: str
    data: Any
    n: int
    spec: _SurvResponseSpec | None
    response: Surv | None
    y: list[float] | None
    terms: _FormulaTerms
    weights: list[Any] | None = None
    offset: list[float] | None = None
    id: list[Any] | None = None
    cluster: list[Any] | None = None
    istate: list[Any] | None = None
    na_action: str = "pass"
    extra: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def response_name(self) -> str | None:
        return None if self.spec is None else self.spec.name

    @property
    def response_columns(self) -> tuple[str, ...]:
        return () if self.spec is None else self.spec.columns


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
class Surv2Data:
    """Counting-process rows built from a ``Surv2`` timeline (R's ``surv2counting``).

    ``row`` is the zero-based input row each interval starts from; ``type`` is the
    ``Surv`` type of the ``(start, stop, status)`` response; ``istate`` holds the
    state codes each interval starts in when the timeline records initial states.
    """

    row: list[int]
    start: list[float]
    stop: list[float]
    status: list[int | None]
    istate: list[int] | None
    states: list[str]
    type: str


@dataclass(frozen=True)
class Timeline:
    """Timeline rows built from counting-process data (R's ``totimeline``).

    ``status`` codes index ``state_levels`` (0 is the censoring level) and
    ``data_row`` is the zero-based input row that supplies the covariates.
    """

    time: list[float]
    status: list[int]
    data_row: list[int]
    state_levels: list[str]


@dataclass(frozen=True)
class SurvExpResult:
    """R's ``survexp`` object: expected survival at ``time`` for each group.

    ``surv`` and ``n_risk`` are vectors for one curve and row-major
    ``ntime x ngroup`` matrices (``strata`` naming the columns) otherwise; the
    individual methods return a plain list instead.
    """

    time: list[float]
    surv: list[float] | list[list[float]]
    n_risk: list[float] | list[list[float]]
    method: str
    n: int
    strata: list[str] | None = None

    @property
    def cumhaz(self) -> list[float] | list[list[float]]:
        """``-log(surv)``, the expected cumulative hazard."""

        def negative_log(value: float) -> float:
            return -math.log(value) if value > 0.0 else math.inf

        if self.surv and isinstance(self.surv[0], list):
            return [[negative_log(value) for value in row] for row in self.surv]
        return [negative_log(value) for value in self.surv]


@dataclass(frozen=True)
class PyearsResult:
    """R's ``pyears`` object.

    ``pyears``, ``n``, ``event`` and ``expected`` are the R arrays as row-major
    nested lists over ``dim`` (a flat list for one dimension, a scalar list for no
    grouping); ``dimnames`` maps each term label to its level labels, in formula
    order.  ``data`` is the ``data.frame = TRUE`` layout instead.
    """

    pyears: Any
    n: Any
    offtable: float
    observations: int
    tcut: bool
    dim: list[int]
    dimnames: dict[str, list[str]]
    event: Any = None
    expected: Any = None
    data: dict[str, list[Any]] | None = None

    @property
    def group(self) -> list[str]:
        """The cell labels in column-major order (the reticulate bridge's view)."""

        if not self.dimnames:
            return ["(all)"]
        labels = [""] * math.prod(self.dim)
        for cell in range(len(labels)):
            parts = []
            remainder = cell
            for extent, levels in zip(self.dim, self.dimnames.values(), strict=True):
                parts.append(levels[remainder % extent])
                remainder //= extent
            labels[cell] = ", ".join(parts)
        return labels


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
    names: list[str]
    strata: list[Any] | None = None

    def subset(
        self, indices: Sequence[int], table_indices: Sequence[int] | None = None
    ) -> CoxZPHResult:
        """``[.cox.zph``: keep the selected terms (0-based), dropping deaths that
        played no role in them (strata by covariate interactions).  ``table_indices``
        are the rows of the table the same subscript selects (R applies it to the
        table too, so a negative subscript keeps the GLOBAL row); the selected terms
        by default."""

        selected = [index(value) for value in indices]
        if any(value < 0 or value >= len(self.names) for value in selected):
            raise IndexError("invalid variable requested")
        rows = selected if table_indices is None else [index(value) for value in table_indices]
        if any(value < 0 or value >= len(self.table) for value in rows):
            raise IndexError("invalid variable requested")
        y = [[row[col] for col in selected] for row in self.y]
        keep = list(range(len(y)))
        if self.strata is not None:
            keep = [idx for idx, row in enumerate(y) if not all(math.isnan(v) for v in row)]
        return CoxZPHResult(
            table=[self.table[row] for row in rows],
            x=[self.x[idx] for idx in keep],
            time=[self.time[idx] for idx in keep],
            y=[y[idx] for idx in keep],
            var=[[self.var[row][col] for col in selected] for row in selected],
            transform=self.transform,
            names=[self.names[col] for col in selected],
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
class SurvfitCall:
    """The parts of R's ``fit$call`` that ``residuals.survfit`` and ``pseudo`` re-read.

    ``terms`` are the right-hand side term labels; ``strata(mf[terms])`` is the curve factor.
    """

    terms: tuple[str, ...] = ()
    stype: int = 1
    ctype: int = 1
    timefix: bool = True
    start_time: float | None = None
    p0: list[float] | None = None
    id: str | None = None


@dataclass(frozen=True)
class SurvfitResult:
    """R's ``survfit`` object for a single-endpoint curve (``survfitKM`` / ``survfitTurnbull``).

    Curves are stacked as in R: ``strata`` maps each curve's label to its number of rows.
    ``std_err`` is the standard error of ``log(surv)`` when ``logse`` is true and of ``surv``
    otherwise (the robust variance); ``std_chaz`` is always that of ``cumhaz``.  ``model`` is
    the model frame of the call (R re-evaluates it through ``model.frame``) and ``engine`` the
    Rust result the summary methods work from (absent for Turnbull curves, whose ``cumhaz``
    and ``t0`` are the values R's ``survfit0`` derives).
    """

    n: list[int]
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    surv: list[float]
    cumhaz: list[float]
    type: str
    t0: float
    n_enter: list[float] | None = None
    counts: _core.SurvfitCounts | None = None
    std_err: list[float] | None = None
    std_chaz: list[float] | None = None
    lower: list[float] | None = None
    upper: list[float] | None = None
    strata: dict[str, int] | None = None
    n_id: list[int] | None = None
    logse: bool | None = None
    conf_int: float | None = None
    conf_type: str | None = None
    conf_lower: str | None = None
    influence_surv: list[_core.SurvfitInfluence] | None = None
    influence_chaz: list[_core.SurvfitInfluence] | None = None
    start_time: float | None = None
    time0: bool = False
    call: SurvfitCall = field(default_factory=SurvfitCall)
    model: dict[str, Any] | None = None
    engine: _core.SurvfitKMResult | None = field(default=None, repr=False, compare=False)

    @property
    def strata_names(self) -> list[str]:
        """The curve labels (``names(fit$strata)``), ``[]`` for a single unnamed curve."""

        return list(self.strata) if self.strata else []


@dataclass(frozen=True)
class SurvfitMultiStateResult:
    """R's ``survfitms`` object: Aalen-Johansen probability-in-state curves.

    Row-major matrices have one row per time; the columns of ``n_risk``, ``n_event``,
    ``n_censor``, ``pstate``, ``std_err``, ``lower`` and ``upper`` are ``states``, those of
    ``n_transition``, ``cumhaz`` and ``std_chaz`` the observed transitions ``hazard_names``
    (R's ``"from:to"`` column names).  ``p0`` has one row per curve and ``transitions`` is
    ``survcheck``'s table of observed transitions (from state x to state or censored).
    """

    n: list[int]
    time: list[float]
    n_risk: list[list[float]]
    n_event: list[list[float]]
    n_censor: list[list[float]]
    n_transition: list[list[float]]
    pstate: list[list[float]]
    cumhaz: list[list[float]]
    p0: list[list[float]]
    states: list[str]
    hazard_names: list[str]
    transitions: NamedMatrix
    n_id: list[int]
    type: str
    t0: float
    n_enter: list[list[float]] | None = None
    counts: _core.SurvfitAJCounts | None = None
    std_err: list[list[float]] | None = None
    std_chaz: list[list[float]] | None = None
    std_auc: list[list[float]] | None = None
    se0: list[list[float]] | None = None
    lower: list[list[float]] | None = None
    upper: list[list[float]] | None = None
    strata: dict[str, int] | None = None
    logse: bool | None = None
    conf_int: float | None = None
    conf_type: str | None = None
    influence_pstate: list[_core.SurvfitAJInfluence] | None = None
    start_time: float | None = None
    time0: bool = False
    call: SurvfitCall = field(default_factory=SurvfitCall)
    model: dict[str, Any] | None = None
    engine: _core.SurvfitAJResult | None = field(default=None, repr=False, compare=False)
    # the states before `fit[, states]` selected some (R's oldstate)
    oldstate: tuple[str, ...] | None = None

    @property
    def strata_names(self) -> list[str]:
        return list(self.strata) if self.strata else []


@dataclass(frozen=True)
class SummarySurvfitResult:
    """R's ``summary.survfit``: the fit at its event times or at ``times``, plus the table."""

    time: list[float]
    n_risk: list[float] | list[list[float]]
    n_event: list[float] | list[list[float]]
    n_censor: list[float] | list[list[float]]
    surv: list[float] | None
    cumhaz: list[float] | list[list[float]]
    strata: list[str] | None
    table: NamedMatrix
    n: list[int]
    n_enter: list[float] | list[list[float]] | None = None
    std_err: list[float] | list[list[float]] | None = None
    std_chaz: list[float] | list[list[float]] | None = None
    lower: list[float] | list[list[float]] | None = None
    upper: list[float] | list[list[float]] | None = None
    rmean_endtime: list[float] | None = None
    conf_int: float | None = None
    conf_type: str | None = None
    pstate: list[list[float]] | None = None
    states: list[str] | None = None
    n_transition: list[list[float]] | None = None


@dataclass(frozen=True)
class NamedMatrix:
    """An R matrix with dimnames: ``summary(fit)$table``, ``fit$transitions``."""

    rownames: list[str] | None
    colnames: list[str]
    values: list[list[float]]


@dataclass(frozen=True)
class SurvfitQuantileResult:
    """``quantile.survfit``: rows are curves, columns ``probs``; the limits when requested."""

    probs: list[float]
    quantile: list[list[float]]
    strata: list[str] | None = None
    lower: list[list[float]] | None = None
    upper: list[list[float]] | None = None


@dataclass(frozen=True)
class SurvfitResidualsResult:
    """``residuals.survfit``: ``resid[row][time]``, or ``resid[row][column][time]`` for a
    multi-state curve where ``columns`` are the states (or the transitions of ``cumhaz``).

    ``id`` labels the rows (subjects when collapsed, observations otherwise) and ``curve`` is
    the 1-based curve each row belongs to (``None`` for a single curve).
    """

    resid: list[Any]
    time: list[float]
    id: list[Any]
    curve: list[int] | None = None
    columns: list[str] | None = None
    column_name: str | None = None
    id_name: str | None = None


@dataclass(frozen=True)
class SurvDiffResult:
    """R's ``survdiff`` object.

    ``obs`` and ``exp`` are per group, or ``groups x strata`` matrices when the formula has a
    ``strata()`` term; ``var`` is the ``groups x groups`` variance of ``obs - exp``.  The
    one-sample test (an ``offset()`` of expected survival) has a single group.
    """

    n: list[int]
    obs: list[Any]
    exp: list[Any]
    var: list[list[float]]
    chisq: float
    pvalue: float
    df: int
    groups: list[str]
    strata: dict[str, int] | None = None


SurvfitConfidenceIntervalResult = _core.ConfidenceBands
# R has one ``survfit`` class for Kaplan-Meier and Turnbull curves; the old name stays for callers.
TurnbullSurvfitResult = SurvfitResult


def _cox_beta(fit: Any) -> list[float]:
    coefficients = getattr(fit, "coefficients", None)
    if coefficients is None:
        raise TypeError("model does not expose fitted coefficients")
    values = list(coefficients)
    if values and isinstance(values[0], list | tuple):
        return [float(value) for value in values[0]]
    return [float(value) for value in values]


# ---------------------------------------------------------------------------
# Results of the R functions in ``_misc``: survcheck, brier, yates, pspline, statefig.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvCheckProblem:
    """Rows and subjects of one kind of data problem (R's ``overlap``, ``gap``, ``jump``,
    ``teleport``): 1-based row numbers of the data given to ``survcheck`` (after ``subset``,
    before ``na.action``) as R reports them, and the distinct subject identifiers involved.
    """

    row: list[int]
    id: list[Any]


@dataclass(frozen=True)
class SurvCheckResult:
    """R's ``survcheck`` object."""

    states: list[str]
    transitions: _core.SurvCheckTransitions
    events: _core.SurvCheckEvents | None
    flag: _core.SurvCheckFlags
    istate: list[str]
    n_id: int
    n_observations: int
    n_transitions: int
    overlap: SurvCheckProblem | None
    gap: SurvCheckProblem | None
    jump: SurvCheckProblem | None
    teleport: SurvCheckProblem | None
    y: Surv
    id: list[Any]
    na_action: list[int] | None

    @property
    def n(self) -> dict[str, int]:
        """R's ``fit$n``: ``c(id=, observations=, transitions=)``."""

        return {
            "id": self.n_id,
            "observations": self.n_observations,
            "transitions": self.n_transitions,
        }


@dataclass(frozen=True)
class SurvCheckCodes:
    """The kernel's answer for a response given as integer codes: the code of the current
    state of every row, the 0-based rows of each kind of problem and the number of
    transitions.  This is the form the R bridge uses, which evaluates the model frame and
    assembles R's ``survcheck`` object itself.
    """

    current_states: list[int]
    overlap_rows: list[int]
    gap_rows: list[int]
    jump_rows: list[int]
    teleport_rows: list[int]
    n_transitions: int


@dataclass(frozen=True)
class BrierResult:
    """R's ``brier`` list; ``p0``, ``phat`` and ``eff_n`` are filled in with ``detail=True``.

    ``phat[i][j]`` is the model's predicted probability of an event by ``times[i]`` for subject
    ``j`` (R's ``ntime`` by ``n`` matrix).  Components can also be read by their R names,
    ``result["eff.n"]``, as R's ``fit[["eff.n"]]`` does.
    """

    rsquared: list[float]
    brier: list[float]
    times: list[float]
    p0: list[float] | None = None
    phat: list[list[float]] | None = None
    eff_n: list[float] | None = None

    def __getitem__(self, name: str) -> Any:
        attribute = name.replace(".", "_")
        if attribute not in {"rsquared", "brier", "times", "p0", "phat", "eff_n"}:
            raise KeyError(name)
        return getattr(self, attribute)


@dataclass(frozen=True)
class StateFigResult:
    """R's ``statefig`` value: the box centres it returns invisibly (``positions``, one
    ``(x, y)`` pair per state, ``states`` being R's row names) and the arrows it draws.
    """

    states: list[str]
    positions: list[tuple[float, float]]
    arrows: list[_core.StateFigArrow]


@dataclass(frozen=True)
class YatesResult:
    """R's ``yates`` object (population marginal means on the linear predictor scale).

    ``estimate`` is R's data frame: one column per tested variable listing its levels, then
    ``pmm`` and ``std``.  ``test`` rows carry R's row names (``global``, ``1 vs 2``, ...).
    ``cmat`` is the population-averaged design over the coefficient columns ``cmat_names``.
    """

    estimate: dict[str, list[Any]]
    test: list[_core.YatesContrast]
    mvar: list[list[float]]
    cmat: list[list[float]]
    cmat_names: list[str]


@dataclass(frozen=True)
class PsplineResult:
    """R's ``pspline`` term: the basis matrix and the attributes ``coxph`` reads from it.

    ``basis`` drops the first column unless ``intercept`` (as R does); ``dmat`` is the
    second-difference penalty matrix (R's ``pparm``) and ``cbase`` the basis centres used by
    R's ``printfun``.  ``theta`` is set for ``method="fixed"``, ``combine`` when it was given.
    """

    basis: list[list[float]]
    knots: list[float]
    nterm: int
    degree: int
    boundary_knots: tuple[float, float]
    intercept: bool
    penalty: bool
    df: int | float
    eps: float
    method: str
    dmat: list[list[float]]
    cbase: list[float]
    theta: float | None = None
    combine: list[int] | None = None

    @property
    def n_cols(self) -> int:
        return len(self.basis[0]) if self.basis else 0
