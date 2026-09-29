"""Result containers and formula/design dataclasses shared by the R-style API."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from operator import index
from typing import TYPE_CHECKING, Any, NamedTuple

from .. import _survival as _core

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._surv import Surv, Surv2


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
    """A fitted penalty basis, including the state needed to transform new data.

    ``nterm``, ``degree``, ``boundary``, ``intercept`` and ``combine`` (the group of every
    basis column when pspline's ``combine`` sums them) describe a pspline basis, whose
    ``penalty`` is ``None`` for ``pspline(penalty=FALSE)``; ``levels`` are a frailty's groups.
    """

    term: _CovariateTerm
    columns: tuple[str, ...]
    names: tuple[str, ...]
    penalty: Any
    degree: int = 3
    boundary: tuple[float, float] | None = None
    levels: tuple[Any, ...] = ()
    intercept: bool = False
    nterm: int = 0
    combine: tuple[int, ...] | None = None

    @property
    def penalized(self) -> bool:
        """False for ``pspline(penalty=FALSE)``, whose basis is an ordinary matrix term."""
        return self.penalty is not None

    @property
    def kind(self) -> str:
        """The penalty function: ``"ridge"``, ``"pspline"`` or ``"frailty"``."""
        return self.penalty.kind if self.penalized else "pspline"


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
    intercept: bool = False


@dataclass(frozen=True)
class CchModelResult:
    """R's ``cch`` object: the engine fit plus the formula metadata ``cch()`` keeps.

    ``sc_ids`` are the ids of the rows of the Borgan estimators' ``sc`` (R's rownames), in
    R's ``rowsum`` order: numbers ascending, factor ids in level order, other labels sorted.
    """

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
    sc_ids: tuple[Any, ...] | None

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
    def sc(self) -> list[list[float]] | None:
        """The Borgan estimators' weighted score residuals collapsed by id, one row per id."""

        sc = self.fit.sc
        return None if sc is None else [list(row) for row in sc]

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
    # attr(terms, "term.labels"): the formula's terms, which labels.aareg returns
    term_labels: tuple[str, ...] = ()

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
    whose single argument is the follow-up expression.  ``timeline`` marks a
    ``Surv2(time, event)`` response, whose ``repeated`` option it keeps.
    """

    arguments: tuple[str, ...]
    columns: tuple[str, ...]
    type: str | None
    origin: float = 0.0
    surv: bool = True
    timeline: bool = False
    repeated: bool | str = False

    @property
    def name(self) -> str:
        """R's name of the response column in the model frame."""

        if not self.surv:
            return self.arguments[0]
        return f"{'Surv2' if self.timeline else 'Surv'}({', '.join(self.arguments)})"


@dataclass(frozen=True)
class NaAction:
    """R's ``na.action`` attribute of a model frame (a fit's ``fit$na.action``): the
    1-based rows ``na.omit`` or ``na.exclude`` removed, and which of the two did (R's
    class, ``"omit"`` or ``"exclude"``).  The ``residuals`` and ``predict`` methods pad
    an ``"exclude"`` fit's values back to every row, with NaN at these (``naresid``)."""

    rows: tuple[int, ...]
    kind: str

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class ModelFrame:
    """R's ``model.frame`` for a survival formula.

    ``data`` is the caller's data or, when ``subset`` or the ``na.action`` removed rows,
    a mapping of the formula's variables at the kept rows; ``response`` is the ``Surv``
    (or ``Surv2``) response (``y`` a plain numeric response such as ``time ~ 1``, or both
    ``None`` for ``~ x``); the R-style extra arguments (``weights``, ``offset``, ``id``,
    ``cluster``, ``istate``) are row aligned with it.  ``na_action`` records the rows the
    ``na.action`` removed (``None`` when it removed none).
    """

    formula: str
    data: Any
    n: int
    spec: _SurvResponseSpec | None
    response: Surv | Surv2 | None
    y: list[float] | None
    terms: _FormulaTerms
    weights: list[Any] | None = None
    offset: list[float] | None = None
    id: list[Any] | None = None
    cluster: list[Any] | None = None
    istate: list[Any] | None = None
    na_action: NaAction | None = None
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
    and ``names`` labels those rows (predictor names or stratum levels).  With a
    cluster, ``dfbeta`` has one row per cluster in sorted cluster order.  ``ranks`` is
    R's data frame as the columns ``time``/``rank``/``timewt``/``casewt`` (a list of
    such tables for several predictors; for several fits one table led by a ``fit``
    column, as ``cord.work`` stacks them).
    """

    concordance: float | list[float]
    count: dict[str, float] | list[dict[str, float]]
    n: int
    names: list[str] | None = None
    var: float | list[list[float]] | None = None
    cvar: float | list[float] | None = None
    dfbeta: list[float] | list[list[float]] | None = None
    influence: list[list[float]] | list[list[list[float]]] | None = None
    ranks: dict[str, list[Any]] | list[dict[str, list[float]]] | None = None
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
class SurvConcordanceResult:
    """R's deprecated ``survConcordance`` object.

    ``stats`` is ``survConcordance.fit``'s row of ``concordant``/``discordant``/
    ``tied.risk``/``tied.time``/``std(c-d)``, or one such row per stratum keyed by its
    level; ``std_err`` is R's ``std.err``, the summed ``std(c-d)`` over twice the number
    of comparable pairs.
    """

    concordance: float
    stats: dict[str, float] | dict[str, dict[str, float]]
    n: int
    std_err: float


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

    ``model=True`` retains the evaluated model frame in ``model``. Otherwise,
    ``x=True`` retains a ``StrataFactor`` (a vector of ones without groups), and
    ``y=True`` retains the numeric follow-up times. ``formula`` and ``term_labels``
    identify the original formula regardless of the retention flags.
    """

    time: list[float]
    surv: list[float] | list[list[float]]
    n_risk: list[float] | list[list[float]]
    method: str
    n: int
    strata: list[str] | None = None
    formula: str | None = None
    term_labels: list[str] = field(default_factory=list)
    model: dict[str, Any] | None = None
    x: StrataFactor | list[float] | None = None
    y: list[float] | None = None

    @property
    def cumhaz(self) -> list[float] | list[list[float]]:
        """``-log(surv)``, the expected cumulative hazard."""

        def negative_log(value: float) -> float:
            return -math.log(value) if value > 0.0 else math.inf

        if self.surv and isinstance(self.surv[0], list):
            return [[negative_log(value) for value in row] for row in self.surv]
        return [negative_log(value) for value in self.surv]


@dataclass(frozen=True)
class SurvExpSummary:
    """Selected expected-survival rows, with one vector or a time-by-curve matrix.

    ``strata`` names the curve columns, as in :class:`SurvExpResult`.
    """

    time: list[float]
    surv: list[float] | list[list[float]]
    n_risk: list[float] | list[list[float]]
    method: str
    strata: list[str] | None = None


@dataclass(frozen=True)
class PyearsResult:
    """R's ``pyears`` object.

    ``pyears``, ``n``, ``event`` and ``expected`` are the R arrays as row-major
    nested lists over ``dim`` (a flat list for one dimension, a scalar list for no
    grouping); ``dimnames`` maps each term label to its level labels, in formula
    order.  ``data`` is the ``data.frame = TRUE`` layout instead.  ``na_action``
    records the rows the ``na.action`` removed (``None`` when it removed none).

    ``model=True`` retains the evaluated model frame in ``model``. Otherwise,
    ``x=True`` retains the one-based grouping codes and raw ``tcut`` times as a
    row-major matrix (a vector of ones without groups), and ``y=True`` retains
    the ``Surv`` response or a one-column numeric matrix. ``formula`` and
    ``term_labels`` identify the formula regardless of the retention flags.
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
    na_action: NaAction | None = None
    formula: str | None = None
    term_labels: list[str] = field(default_factory=list)
    model: dict[str, Any] | None = None
    x: list[float] | list[list[float]] | None = None
    y: Surv | list[list[float]] | None = None

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
    matrix ``y`` (one column per term, named by ``names``) and its variance ``var``.
    ``df`` is a float: a penalized fit's terms have fractional degrees of freedom."""

    table: list[dict[str, float | str]]
    x: list[float] = field(repr=False)
    time: list[float] = field(repr=False)
    y: list[list[float]] = field(repr=False)
    var: list[list[float]] = field(repr=False)
    transform: str
    names: list[str]
    strata: list[Any] | None = field(default=None, repr=False)

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
    for one curve, or ``ntime x ncurve`` matrices (one column per ``newdata`` row, named
    by ``colnames``, R's ``colnames(fit$surv)``).
    """

    n: list[int]
    time: list[float] = field(repr=False)
    n_risk: list[float] = field(repr=False)
    n_event: list[float] = field(repr=False)
    n_censor: list[float] = field(repr=False)
    surv: list[float] | list[list[float]] = field(repr=False)
    cumhaz: list[float] | list[list[float]] = field(repr=False)
    type: str
    strata: dict[str, int] | None = None
    std_err: list[float] | list[list[float]] | None = field(default=None, repr=False)
    std_chaz: list[float] | list[list[float]] | None = field(default=None, repr=False)
    lower: list[float] | list[list[float]] | None = field(default=None, repr=False)
    upper: list[float] | list[list[float]] | None = field(default=None, repr=False)
    logse: bool = True
    conf_type: str = "none"
    conf_int: float | None = None
    start_time: float | None = None
    newdata: Any | None = field(default=None, repr=False)
    colnames: list[str] | None = None

    @property
    def ncurve(self) -> int:
        return len(self.surv[0]) if self.surv and isinstance(self.surv[0], list) else 1


@dataclass(frozen=True)
class SurvfitCall:
    """The parts of R's ``fit$call`` that ``residuals.survfit`` and ``pseudo`` re-read.

    ``terms`` are the right-hand side term labels; ``strata(mf[terms])`` is the curve factor.
    ``stype`` and ``ctype`` give the curve that was fitted; ``type`` is the old-style ``type``
    argument as given, in which case R's call carries no ``stype`` or ``ctype``.
    """

    terms: tuple[str, ...] = ()
    stype: int = 1
    ctype: int = 1
    timefix: bool = True
    start_time: float | None = None
    p0: list[float] | None = None
    id: str | None = None
    type: str | None = None


class _EngineInfluence(NamedTuple):
    """Pickled in place of influence matrices that are views of the ``engine``'s own.

    ``pickle`` and ``copy`` would otherwise store each matrix twice, once in the fit's
    list and once in the encoded engine, and a restored fit would hold two copies;
    ``__setstate__`` takes the matrices from the restored engine instead (and names the
    rows of a Kaplan-Meier curve's matrices by ``clname``, as ``_km_result`` does).
    """

    clname: Sequence[Any] | None


def _views_of_engine(mine: Sequence[Any] | None, engines: Sequence[Any] | None) -> bool:
    """Whether each influence matrix of ``mine`` is a view of the engine curve's matrix."""

    if not mine or engines is None or len(mine) != len(engines):
        return False
    return all(
        curve.values.__array_interface__ == engine.values.__array_interface__
        for curve, engine in zip(mine, engines, strict=True)
    )


@dataclass(frozen=True)
class SurvfitInfluenceMatrix:
    """One curve's ``influence.surv`` or ``influence.chaz`` matrix of a ``survfit`` object.

    ``values[k]`` is the influence of cluster ``cluster[k]`` at each time of the curve;
    ``cluster`` holds R's row names: the ``cluster`` (else ``id``) values, in order of first
    appearance, or the observation numbers ``1..n`` when the observations are the clusters.
    Like ``survfitKM``, which names the rows ``clname[clusterid]``, it holds the engine's
    ``survival.surv_analysis.SurvfitInfluence`` (0-based cluster codes) and ``clname``, the
    levels every curve of the fit shares, or ``None`` when the engine's labels are already
    the observation numbers.  ``values`` is a read-only NumPy view of the engine's matrix.
    """

    influence: _core.SurvfitInfluence
    clname: Sequence[Any] | None = field(default=None, repr=False)

    @property
    def cluster(self) -> list[Any]:
        codes = self.influence.cluster
        return codes if self.clname is None else [self.clname[code] for code in codes]

    @property
    def values(self) -> NDArray[np.float64]:
        return self.influence.values


@dataclass(frozen=True)
class SurvfitResult:
    """R's ``survfit`` object for a single-endpoint curve (``survfitKM`` / ``survfitTurnbull``).

    Curves are stacked as in R: ``strata`` maps each curve's label to its number of rows.
    ``std_err`` is the standard error of ``log(surv)`` when ``logse`` is true and of ``surv``
    otherwise (the robust variance); ``std_chaz`` is always that of ``cumhaz``.  ``model`` is
    the model frame of the call (R re-evaluates it through ``model.frame``) and ``engine`` the
    Rust result the summary methods work from (for Turnbull curves, whose ``cumhaz``,
    ``std_chaz`` and ``t0`` are the values R's ``survfit0`` derives, it holds the curves as
    fitted).  As in R, a ``start.time`` of a Kaplan-Meier fit shows only as ``t0`` and
    ``time0`` marks a curve that already starts with its ``t0`` row, the result of
    ``survfit0``.
    """

    n: list[int]
    time: list[float] = field(repr=False)
    n_risk: list[float] = field(repr=False)
    n_event: list[float] = field(repr=False)
    n_censor: list[float] = field(repr=False)
    surv: list[float] = field(repr=False)
    cumhaz: list[float] = field(repr=False)
    type: str
    t0: float
    n_enter: list[float] | None = field(default=None, repr=False)
    counts: _core.SurvfitCounts | None = field(default=None, repr=False)
    std_err: list[float] | None = field(default=None, repr=False)
    std_chaz: list[float] | None = field(default=None, repr=False)
    lower: list[float] | None = field(default=None, repr=False)
    upper: list[float] | None = field(default=None, repr=False)
    strata: dict[str, int] | None = None
    n_id: list[int] | None = None
    logse: bool | None = None
    conf_int: float | None = None
    conf_type: str | None = None
    conf_lower: str | None = None
    influence_surv: list[SurvfitInfluenceMatrix] | None = field(default=None, repr=False)
    influence_chaz: list[SurvfitInfluenceMatrix] | None = field(default=None, repr=False)
    time0: bool = False
    call: SurvfitCall = field(default_factory=SurvfitCall)
    model: dict[str, Any] | None = field(default=None, repr=False)
    engine: _core.SurvfitKMResult | None = field(default=None, repr=False, compare=False)

    @property
    def strata_names(self) -> list[str]:
        """The curve labels (``names(fit$strata)``), ``[]`` for a single unnamed curve."""

        return list(self.strata) if self.strata else []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for name in ("influence_surv", "influence_chaz"):
            matrices = state[name]
            if _views_of_engine(
                [matrix.influence for matrix in matrices or ()], getattr(self.engine, name, None)
            ) and all(matrix.clname is matrices[0].clname for matrix in matrices):
                state[name] = _EngineInfluence(matrices[0].clname)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name in ("influence_surv", "influence_chaz"):
            if isinstance(state[name], _EngineInfluence):
                clname = state[name].clname
                state[name] = [
                    SurvfitInfluenceMatrix(curve, clname)
                    for curve in getattr(state["engine"], name)
                ]
        self.__dict__.update(state)


@dataclass(frozen=True)
class SurvfitMultiStateResult:
    """R's ``survfitms`` object: Aalen-Johansen probability-in-state curves.

    Row-major matrices have one row per time; the columns of ``n_risk``, ``n_event``,
    ``n_censor``, ``pstate``, ``std_err``, ``lower`` and ``upper`` are ``states``, those of
    ``n_transition``, ``cumhaz`` and ``std_chaz`` the observed transitions ``hazard_names``
    (R's ``"from:to"`` column names).  ``p0`` has one row per curve and ``transitions`` is
    ``survcheck``'s table of observed transitions (from state x to state or censored), which
    ``fit[, states]`` drops, as it drops ``n_id`` from a fit without strata.  The rows of
    each ``influence_pstate`` array are named, as in R, by the clusters' numbers ``1, 2, ...``
    in order of first appearance.
    """

    n: list[int]
    time: list[float] = field(repr=False)
    n_risk: list[list[float]] = field(repr=False)
    n_event: list[list[float]] = field(repr=False)
    n_censor: list[list[float]] = field(repr=False)
    n_transition: list[list[float]] = field(repr=False)
    pstate: list[list[float]] = field(repr=False)
    cumhaz: list[list[float]] = field(repr=False)
    p0: list[list[float]] = field(repr=False)
    states: list[str]
    hazard_names: list[str]
    transitions: NamedMatrix | None
    n_id: list[int] | None
    type: str
    t0: float
    n_enter: list[list[float]] | None = field(default=None, repr=False)
    counts: _core.SurvfitAJCounts | None = field(default=None, repr=False)
    std_err: list[list[float]] | None = field(default=None, repr=False)
    std_chaz: list[list[float]] | None = field(default=None, repr=False)
    std_auc: list[list[float]] | None = field(default=None, repr=False)
    se0: list[list[float]] | None = field(default=None, repr=False)
    lower: list[list[float]] | None = field(default=None, repr=False)
    upper: list[list[float]] | None = field(default=None, repr=False)
    strata: dict[str, int] | None = None
    logse: bool | None = None
    conf_int: float | None = None
    conf_type: str | None = None
    influence_pstate: list[_core.SurvfitAJInfluence] | None = field(default=None, repr=False)
    start_time: float | None = None
    time0: bool = False
    call: SurvfitCall = field(default_factory=SurvfitCall)
    model: dict[str, Any] | None = field(default=None, repr=False)
    engine: _core.SurvfitAJResult | None = field(default=None, repr=False, compare=False)
    # the states before `fit[, states]` selected some (R's oldstate)
    oldstate: tuple[str, ...] | None = None

    @property
    def strata_names(self) -> list[str]:
        return list(self.strata) if self.strata else []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if _views_of_engine(self.influence_pstate, getattr(self.engine, "influence_pstate", None)):
            state["influence_pstate"] = _EngineInfluence(None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if isinstance(state["influence_pstate"], _EngineInfluence):
            state["influence_pstate"] = state["engine"].influence_pstate
        self.__dict__.update(state)


@dataclass(frozen=True)
class CoxSurvfitMultiStateResult:
    """R's ``survfit.coxphms`` object (class ``c("survfitcoxms", "survfitms",
    "survfit")``): probability-in-state curves predicted from a multi-state Cox model.

    ``pstate[t, i, s]`` is the probability of state ``states[s]`` at ``time[t]`` for
    newdata row ``i``, and ``cumhaz[t, i, k]`` the cumulative hazard of transition
    ``cumhaz_names[k]`` (every transition of the model).  The counts (``n_risk``,
    ``n_event``, ``n_censor``: time x state; ``n_transition``: time x the observed
    transitions of ``transitions``) are the Aalen-Johansen ones of the data, as are
    ``n``, ``n_id`` and ``p0`` (one entry or row per stratum).  Strata stack their
    times, ``strata`` giving each block's length.  ``newdata`` holds the rows the curves
    are for (or the group labels after ``aggregate``).  ``engine`` holds the counts and
    time grid; it is dropped by a stratum or state subset, as is ``cumhaz`` by a state
    subset (``oldstate`` then records the original states).
    """

    n: list[int]
    time: list[float] = field(repr=False)
    n_risk: list[list[float]] = field(repr=False)
    n_event: list[list[float]] = field(repr=False)
    n_censor: list[list[float]] = field(repr=False)
    n_transition: list[list[float]] | None = field(repr=False)
    n_id: list[int]
    pstate: NDArray[np.float64] = field(repr=False, compare=False)
    cumhaz: NDArray[np.float64] | None = field(repr=False, compare=False)
    cumhaz_names: list[str]
    p0: list[list[float]] = field(repr=False)
    states: list[str]
    transitions: NamedMatrix | None
    type: str
    t0: float
    start_time: float | None
    strata: dict[str, int] | None
    newdata: dict[str, list[Any]] | None = field(repr=False)
    stype: int = 2
    ctype: int = 1
    time0: bool = False
    oldstate: tuple[str, ...] | None = None
    engine: _core.SurvfitAJResult | None = field(default=None, repr=False, compare=False)

    @property
    def strata_names(self) -> list[str]:
        return list(self.strata) if self.strata else []

    @property
    def dim(self) -> dict[str, int]:
        """R's ``dim(fit)``: the strata (when stratified), newdata rows and states."""

        dims = {"strata": len(self.strata)} if self.strata else {}
        dims["data"] = int(self.pstate.shape[1])
        dims["states"] = len(self.states)
        return dims


@dataclass(frozen=True)
class SummarySurvfitCoxmsResult:
    """R's ``summary.survfitms`` of a :class:`CoxSurvfitMultiStateResult`: the curves at
    their event times (or at ``times``), ``pstate``/``cumhaz`` arrays of shape (time,
    newdata row, state/transition), and ``survmean2``'s table (one row per stratum,
    newdata row and state, the stratum varying fastest)."""

    time: list[float]
    n_risk: list[list[float]]
    n_event: list[list[float]]
    n_censor: list[list[float]]
    n_transition: list[list[float]] | None
    pstate: NDArray[np.float64] = field(repr=False, compare=False)
    cumhaz: NDArray[np.float64] | None = field(repr=False, compare=False)
    strata: list[str] | None
    table: NamedMatrix
    rmean_endtime: list[float] | None
    states: list[str]
    newdata: dict[str, list[Any]] | None = field(repr=False)


@dataclass(frozen=True)
class SummarySurvfitResult:
    """R's ``summary.survfit``: the fit at its event times or at ``times``, plus the table."""

    time: list[float]
    n_risk: list[float] | list[list[float]]
    n_event: list[float] | list[list[float]]
    n_censor: list[float] | list[list[float]]
    surv: list[float] | list[list[float]] | None
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
    """An R matrix with dimnames: ``summary(fit)$table``, ``fit$transitions``, a
    multi-state fit's ``cmap``."""

    rownames: list[str] | None
    colnames: list[str]
    values: list[list[float]] | list[list[int]]


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
    """R's ``yates`` object (population marginal means).

    ``estimate`` is R's data frame: one column per tested variable listing its levels, then
    ``pmm`` and ``std`` (``pmm`` is NaN for a level the fit cannot estimate).  ``test`` rows
    carry R's row names (``global``, ``1 vs 2``, ...; ``chisq`` NaN and ``df`` None for R's
    NA).  ``cmat`` is the population-averaged design over the coefficient columns
    ``cmat_names`` (empty for a simulated prediction or when no level is estimable).
    ``summary`` holds the simulated curves of ``predict="survival"``: the baseline
    ``survfit`` object with one column of ``surv``, ``cumhaz``, ``std_err``, ``lower`` and
    ``upper`` per level.
    With ``method="sgtt"``, ``sas`` holds the estimable SAS hypothesis matrix,
    with column names in ``sas_names`` and row names in ``sas_row_names``.
    """

    estimate: dict[str, list[Any]]
    test: list[_core.YatesContrast]
    mvar: list[list[float]]
    cmat: list[list[float]]
    cmat_names: list[str]
    summary: CoxSurvfitResult | None = None
    sas: list[list[float]] | None = None
    sas_names: list[str] = field(default_factory=list)
    sas_row_names: list[str] = field(default_factory=list)


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
