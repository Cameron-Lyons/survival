from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ._survival import (
    ConfidenceBands,
    SurvfitAJCounts,
    SurvfitAJInfluence,
    SurvfitAJResult,
    SurvfitCounts,
    SurvfitInfluence,
    SurvfitKMResult,
)
from ._survival import SplineBasisResult as _SplineBasisResult

class StrataFactor:
    codes: list[int | None]
    levels: list[str]
    labels: list[str | None]
    counts: list[int]
    def __iter__(self): ...
    def __len__(self) -> int: ...

class Surv:
    time: tuple[float, ...]
    event: tuple[int | None, ...]
    start: tuple[float, ...] | None
    time2: tuple[float, ...] | None
    type: str
    states: tuple[str, ...]
    def __init__(
        self,
        *args: Any,
        type: str | None = None,
        origin: Any = 0.0,
        time: Any = ...,
        time2: Any = ...,
        event: Any = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    @property
    def status(self) -> tuple[int | None, ...]: ...
    @property
    def ncol(self) -> int: ...
    def as_matrix(self) -> list[list[Any]]: ...
    def replace_times(
        self,
        *,
        time: Sequence[float] | None = None,
        start: Sequence[float] | None = None,
        time2: Sequence[float] | None = None,
    ) -> Surv: ...
    def subset(self, indices: Sequence[int]) -> Surv: ...

class Surv2:
    time: tuple[float, ...]
    status: tuple[int | None, ...]
    states: tuple[str, ...]
    repeated: bool | str
    def __init__(self, time: Any, event: Any, repeated: Any = False) -> None: ...
    def __len__(self) -> int: ...

def Surv2data(
    time: Any,
    status: Any,
    *,
    states: Any | None = None,
    repeated: Any = False,
    id: Any,
) -> Surv2Data: ...
def totimeline(
    start: Any,
    stop: Any,
    status: Any,
    *,
    states: Any,
    id: Any,
    istate: Any | None = None,
    istate_levels: Any | None = None,
) -> Timeline: ...

class Surv2Data:
    row: list[int]
    start: list[float]
    stop: list[float]
    status: list[int | None]
    istate: list[int] | None
    states: list[str]
    type: str

class Timeline:
    time: list[float]
    status: list[int]
    data_row: list[int]
    state_levels: list[str]

class ModelFrame:
    formula: str
    data: Any
    n: int
    spec: Any
    response: Surv | None
    y: list[float] | None
    terms: Any
    weights: list[Any] | None
    offset: list[float] | None
    id: list[Any] | None
    cluster: list[Any] | None
    istate: list[Any] | None
    na_action: str
    extra: dict[str, list[Any]]
    @property
    def response_name(self) -> str | None: ...
    @property
    def response_columns(self) -> tuple[str, ...]: ...

class RateTable:
    dims: list[int]
    dimid: list[str]
    dimnames: list[list[str]]
    cutpoints: list[list[float] | None]
    rates: list[float]
    def type_codes(self) -> list[int]: ...
    def rate(self, index: Sequence[int]) -> float | None: ...

class SurvExpResult:
    time: list[float]
    surv: list[float] | list[list[float]]
    n_risk: list[float] | list[list[float]]
    method: str
    n: int
    strata: list[str] | None
    @property
    def cumhaz(self) -> list[float] | list[list[float]]: ...

class PyearsResult:
    pyears: Any
    n: Any
    offtable: float
    observations: int
    tcut: bool
    dim: list[int]
    dimnames: dict[str, list[str]]
    event: Any
    expected: Any
    data: dict[str, list[Any]] | None
    @property
    def group(self) -> list[str]: ...

class FineGrayOutput:
    row: list[int]
    start: list[float]
    end: list[float]
    wt: list[float]
    add: list[int]

class FineGrayFrame(dict[str, list[Any]]):
    event: Any
    def __init__(
        self,
        columns: Mapping[str, Sequence[Any]] | None = None,
        *,
        event: Any = None,
    ) -> None: ...
    def copy(self) -> FineGrayFrame: ...

class TMergeOperation:
    kind: str
    time: Any
    value: Any | None
    default: Any | None
    censor: Any | None

class TMergeFrame(Mapping[str, list[Any]]):
    columns: dict[str, list[Any]]
    tname: dict[str, str]
    tevent: dict[str, Any]
    tdcvar: tuple[str, ...]
    tcount: dict[str, dict[str, int]]
    def __getitem__(self, key: str) -> list[Any]: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def copy(self) -> TMergeFrame: ...

class TcutResult:
    values: list[float]
    cutpoints: list[float]
    labels: list[str]

class SurvObrienResult:
    statistic: float
    p_value: float
    df: int
    scores: list[float]
    score_sum: float
    expected: float
    variance: float

class YatesResult:
    levels: list[str]
    means: list[float]
    se: list[float]
    lower: list[float]
    upper: list[float]
    n: list[int]
    predict_type: str

class YatesPairwiseResult:
    level1: list[str]
    level2: list[str]
    difference: list[float]
    se: list[float]
    z: list[float]
    p_value: list[float]

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
    dfbeta: list[list[list[float]]] | None
    robust_test_variance: list[list[float]] | None
    formula: str | None
    weights: list[float] | None
    cluster: list[Any] | None
    cluster_levels: list[Any] | None
    model: dict[str, Any] | None
    x: list[list[float]] | None
    y: Surv | None
    @property
    def nrisk(self) -> list[float]: ...
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def tweight(self) -> list[list[float]]: ...
    @property
    def test_var(self) -> list[list[float]]: ...
    @property
    def test_var2(self) -> list[list[float]] | None: ...

def is_surv(value: Any) -> bool: ...
def is_na_surv(x: Any) -> list[bool]: ...
def format_surv(x: Any) -> list[str]: ...
def fromtimeline(
    time: Any,
    status: Any,
    *,
    id: Any,
    states: Any | None = None,
    repeated: Any = False,
) -> Surv2Data: ...
def is_ratetable(x: Any, verbose: bool = False) -> bool | list[str]: ...
def ratetableDate(x: Any) -> float | list[float]: ...
def survexp(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    rmap: Mapping[str, Any] | None = None,
    times: Any | None = None,
    method: str | None = None,
    cohort: bool = True,
    conditional: bool = False,
    ratetable: Any | None = None,
    scale: Any = 1,
    se_fit: bool | None = None,
    model: bool = False,
    x: bool = False,
    y: bool = False,
    *,
    time: Any = None,
    age: Any = None,
    year: Any = None,
    sex: Any = None,
    **kwargs: Any,
) -> SurvExpResult | list[float]: ...
def survexp_individual(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
) -> list[float]: ...
def pyears(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    rmap: Mapping[str, Any] | None = None,
    ratetable: Any | None = None,
    scale: Any = 365.25,
    expect: str = "event",
    model: bool = False,
    x: bool = False,
    y: bool = False,
    data_frame: bool = False,
    *,
    time: Any = None,
    start: Any = None,
    stop: Any = None,
    event: Any = None,
    group: Any = None,
    **kwargs: Any,
) -> PyearsResult: ...
def finegray(
    formula: str,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.pass",
    etype: Any | None = None,
    prefix: str = "fg",
    count: str | None = None,
    id: Any | None = None,
    timefix: bool = True,
    **kwargs: Any,
) -> FineGrayFrame: ...
def tdc(time: Any, value: Any | None = None, init: Any | None = None) -> TMergeOperation: ...
def cumtdc(
    time: Any,
    value: Any | None = None,
    init: Any | None = None,
) -> TMergeOperation: ...
def event(
    time: Any,
    value: Any | None = None,
    censor: Any | None = None,
) -> TMergeOperation: ...
def cumevent(
    time: Any,
    value: Any | None = None,
    censor: Any | None = None,
) -> TMergeOperation: ...
def tmerge(
    data1: Any,
    data2: Any,
    id: Any,
    *,
    tstart: Any | None = None,
    tstop: Any | None = None,
    options: Mapping[str, Any] | None = None,
    operations: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    **args: Any,
) -> TMergeFrame: ...
def cipoisson(
    k: Any,
    time: Any = 1.0,
    p: Any = 0.95,
    method: Any = "exact",
) -> tuple[float, float] | list[tuple[float, float]]: ...
def blogit(x: Any, edge: Any = 0.05) -> float | list[float]: ...
def bprobit(x: Any, edge: Any = 0.05) -> float | list[float]: ...
def bcloglog(x: Any, edge: Any = 0.05) -> float | list[float]: ...
def blog(x: Any, edge: Any = 0.05) -> float | list[float]: ...
def neardate(
    id1: Any,
    id2: Any,
    y1: Any,
    y2: Any,
    best: str = "after",
    nomatch: int | None = None,
) -> list[int | None]: ...
def tcut(
    x: Any,
    breaks: Any,
    labels: Any | None = None,
    scale: Any = 1,
) -> TcutResult: ...
def nsk(
    x: Any,
    df: Any | None = None,
    knots: Any | None = None,
    intercept: Any = False,
    b: Any = 0.05,
    Boundary_knots: Any = ...,
    **kwargs: Any,
) -> _SplineBasisResult: ...
def pspline(
    x: Any,
    df: Any = 4,
    theta: Any | None = None,
    nterm: Any | None = None,
    degree: Any = 3,
    eps: Any = 0.1,
    method: Any | None = None,
    Boundary_knots: Any | None = None,
    *,
    boundary_knots: Any | None = None,
    intercept: Any = False,
    penalty: Any = True,
    combine: Any | None = None,
) -> dict[str, Any]: ...
def survobrien(
    formula: Any,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    transform: Any | None = None,
) -> SurvObrienResult | dict[str, list[Any]]: ...
def yates(
    predictions: Any,
    factor: Any,
    weights: Any | None = None,
    conf_level: Any | None = None,
) -> YatesResult: ...
def yates_contrast(
    x: Any,
    coef: Any,
    n_obs: Any,
    n_vars: Any,
    factor_col: Any,
    factor_levels: Any,
    predict_type: str | None = None,
) -> YatesResult: ...
def yates_pairwise(result: YatesResult) -> YatesPairwiseResult: ...
def survexp_us() -> RateTable: ...
def survexp_mn() -> RateTable: ...
def survexp_usr() -> RateTable: ...
def strata(
    *variables: Any,
    na_group: bool = False,
    shortlabel: bool | None = None,
    sep: str = ", ",
    labels: Sequence[str] | None = None,
) -> StrataFactor: ...

class ConcordanceResult:
    concordance: float | list[float]
    count: dict[str, float] | list[dict[str, float]]
    n: int
    names: list[str] | None
    var: float | list[list[float]] | None
    cvar: float | list[float] | None
    dfbeta: list[float] | list[list[float]] | None
    influence: list[list[float]] | list[list[list[float]]] | None
    ranks: list[dict[str, float]] | list[list[dict[str, float]]] | None
    formula: str | None
    @property
    def std(self) -> float | list[float] | None: ...

class PredictResult:
    fit: Any
    se_fit: Any
    def __iter__(self): ...
    @property
    def predictions(self) -> Any: ...
    @property
    def se(self) -> Any: ...

class CoxZPHResult:
    table: list[dict[str, float | int | str]]
    x: list[float]
    time: list[float]
    y: list[list[float]]
    var: list[list[float]]
    transform: str
    names: tuple[str, ...]
    strata: list[Any] | None
    def subset(self, indices: Sequence[int]) -> CoxZPHResult: ...

class CchModelResult:
    fit: Any
    formula: str
    design: Any
    coef_names: tuple[str, ...]
    y: Surv
    id: tuple[Any, ...]
    subcoh: tuple[int, ...]
    stratum: tuple[Any, ...] | None
    cohort_size: tuple[int, ...]
    subcohort_size: tuple[int, ...]
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def var(self) -> list[list[float]]: ...
    @property
    def naive_var(self) -> list[list[float]]: ...
    @property
    def phase2var(self) -> list[list[float]]: ...
    @property
    def method(self) -> str: ...
    @property
    def stratified(self) -> bool: ...

class CoxPHDetailResult:
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
    strata: dict[str, int] | None
    riskmat: list[list[int]] | None
    sortorder: list[int] | None
    weights: list[float] | None
    nevent_wt: list[float] | None
    nrisk_wt: list[float] | None
    @property
    def cumhaz(self) -> list[float]: ...

class CoxPHWTestResult:
    test: list[float]
    df: int
    solve: list[float] | list[list[float]] | float

class CoxBaseHazardResult:
    hazard: list[float] | list[list[float]]
    time: list[float]
    strata: list[str] | None

class CoxSurvfitResult:
    n: list[int]
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    surv: list[float] | list[list[float]]
    cumhaz: list[float] | list[list[float]]
    type: str
    strata: dict[str, int] | None
    std_err: list[float] | list[list[float]] | None
    std_chaz: list[float] | list[list[float]] | None
    lower: list[float] | list[list[float]] | None
    upper: list[float] | list[list[float]] | None
    logse: bool
    conf_type: str
    conf_int: float | None
    start_time: float | None
    newdata: Any | None
    @property
    def ncurve(self) -> int: ...

class SurvfitCall:
    terms: tuple[str, ...]
    stype: int
    ctype: int
    timefix: bool
    start_time: float | None
    p0: list[float] | None
    id: str | None

class SurvfitResult:
    n: list[int]
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    surv: list[float]
    cumhaz: list[float]
    type: str
    t0: float
    n_enter: list[float] | None
    counts: SurvfitCounts | None
    std_err: list[float] | None
    std_chaz: list[float] | None
    lower: list[float] | None
    upper: list[float] | None
    strata: dict[str, int] | None
    n_id: list[int] | None
    logse: bool | None
    conf_int: float | None
    conf_type: str | None
    conf_lower: str | None
    influence_surv: list[SurvfitInfluence] | None
    influence_chaz: list[SurvfitInfluence] | None
    start_time: float | None
    time0: bool
    call: SurvfitCall
    model: dict[str, Any] | None
    engine: SurvfitKMResult | None
    @property
    def strata_names(self) -> list[str]: ...

class NamedMatrix:
    rownames: list[str] | None
    colnames: list[str]
    values: list[list[float]]

class SurvfitMultiStateResult:
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
    n_enter: list[list[float]] | None
    counts: SurvfitAJCounts | None
    std_err: list[list[float]] | None
    std_chaz: list[list[float]] | None
    std_auc: list[list[float]] | None
    se0: list[list[float]] | None
    lower: list[list[float]] | None
    upper: list[list[float]] | None
    strata: dict[str, int] | None
    logse: bool | None
    conf_int: float | None
    conf_type: str | None
    influence_pstate: list[SurvfitAJInfluence] | None
    start_time: float | None
    time0: bool
    call: SurvfitCall
    model: dict[str, Any] | None
    engine: SurvfitAJResult | None
    @property
    def strata_names(self) -> list[str]: ...

class SummarySurvfitResult:
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    surv: list[float]
    cumhaz: list[float]
    strata: list[str] | None
    table: NamedMatrix
    n: list[int]
    n_enter: list[float] | None
    std_err: list[float] | None
    std_chaz: list[float] | None
    lower: list[float] | None
    upper: list[float] | None
    rmean_endtime: list[float] | None
    conf_int: float | None
    conf_type: str | None

class SurvfitQuantileResult:
    probs: list[float]
    quantile: list[list[float]]
    strata: list[str] | None
    lower: list[list[float]] | None
    upper: list[list[float]] | None

class SurvfitResidualsResult:
    resid: list[Any]
    time: list[float]
    id: list[Any]
    curve: list[int] | None
    columns: list[str] | None
    column_name: str | None
    id_name: str | None

class SurvDiffResult:
    n: list[int]
    obs: list[Any]
    exp: list[Any]
    var: list[list[float]]
    chisq: float
    pvalue: float
    df: int
    groups: list[str]
    strata: dict[str, int] | None

class SurvfitKMInfluence:
    influence_surv: list[list[float]]
    influence_chaz: list[list[float]]

SurvfitConfidenceIntervalResult = ConfidenceBands

def concordance(
    object: Any,
    *more: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    cluster: Any | None = None,
    ymin: Any | None = None,
    ymax: Any | None = None,
    timewt: Any = "n",
    influence: Any = 0,
    ranks: Any = False,
    reverse: Any = False,
    timefix: Any = True,
    keepstrata: Any = 10,
    newdata: Any | None = None,
    scores: Any | None = None,
    strata: Any | None = None,
    **kwargs: Any,
) -> ConcordanceResult: ...
def survConcordance(
    formula: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "fail",
    **kwargs: Any,
) -> ConcordanceResult: ...
def survConcordance_fit(
    y: Any,
    x: Any,
    strata: Any | None = None,
    weight: Any | None = None,
) -> dict[str, float]: ...
def survfit(
    response: Any,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.omit",
    stype: int = 1,
    ctype: int = 1,
    id: Any | None = None,
    cluster: Any | None = None,
    robust: Any | None = None,
    istate: Any | None = None,
    timefix: Any = True,
    etype: Any | None = None,
    model: Any = False,
    error: Any | None = None,
    entry: Any = False,
    time0: Any = False,
    *,
    group: Any | None = None,
    newdata: Any | None = None,
    se_fit: Any = True,
    conf_int: Any = 0.95,
    conf_type: str = "log",
    conf_lower: str = "usual",
    start_time: Any | None = None,
    influence: Any = False,
    p0: Any | None = None,
    type: str | None = None,
    reverse: Any = False,
    censor: Any = True,
    **kwargs: Any,
) -> Any: ...
def survfitkm_influence(
    time: Any,
    status: Any,
    cluster: Any,
    weights: Any | None = None,
    stype: int = 1,
    ctype: int = 1,
    conf_level: float = 0.95,
    conf_type: str = "log",
) -> SurvfitKMInfluence: ...
def survfitkm_counting_influence(
    start: Any,
    stop: Any,
    status: Any,
    cluster: Any,
    weights: Any | None = None,
    stype: int = 1,
    ctype: int = 1,
    conf_level: float = 0.95,
    conf_type: str = "log",
    **kwargs: Any,
) -> SurvfitKMInfluence: ...
def survfit0(x: Any, *args: Any, **kwargs: Any) -> Any: ...
def summary_survfit(
    object: Any,
    times: Any | None = None,
    censored: Any = False,
    scale: Any = 1,
    extend: Any = False,
    rmean: Any | None = None,
) -> SummarySurvfitResult: ...
def quantile_survfit(
    x: Any,
    probs: Any = (0.25, 0.5, 0.75),
    conf_int: Any = True,
    scale: Any = 1,
    tolerance: Any | None = None,
    **kwargs: Any,
) -> SurvfitQuantileResult: ...
def aggregate_survfit(x: Any, by: Any | None = None, FUN: str = "mean") -> Any: ...
def aggregate_survfit_result(result: Any, groups: Any | None = None) -> Any: ...
def survfit_confint(
    p: Any,
    se: Any,
    logse: Any = True,
    conf_type: str | None = None,
    conf_int: Any = 0.95,
    selow: Any | None = None,
    ulimit: Any = True,
    **kwargs: Any,
) -> ConfidenceBands: ...
def survdiff(
    response: Any,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.omit",
    rho: Any = 0,
    timefix: Any = True,
    *,
    group: Any | None = None,
    **kwargs: Any,
) -> SurvDiffResult: ...
def rttright(
    formula: Any = None,
    data: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    times: Any | None = None,
    id: Any | None = None,
    timefix: bool = True,
    renorm: bool = True,
    **kwargs: Any,
) -> list[float] | list[list[float]]: ...
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
) -> dict[str, Any]: ...
def pseudo(
    fit: Any,
    times: Any | None = None,
    type: str | None = None,
    collapse: Any = True,
    data_frame: Any = False,
    **kwargs: Any,
) -> Any: ...
def survcheck(
    formula: Any = ...,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "pass",
    id: Any | None = None,
    istate: Any | None = None,
    istate0: str = "(s0)",
    timefix: bool = True,
    *,
    time1: Any = ...,
    time2: Any = ...,
    status: Any = ...,
    **kwargs: Any,
) -> Any: ...
def basehaz(
    fit: Any,
    newdata: Any | None = None,
    centered: Any = True,
) -> CoxBaseHazardResult: ...
def brier(
    fit: Any,
    times: Any | None = None,
    newdata: Any | None = None,
    ties: Any = True,
    detail: Any = False,
    timefix: Any = True,
    efron: Any = False,
) -> dict[str, Any]: ...
def cox_zph(
    fit: Any,
    transform: Any = "km",
    terms: Any = True,
    singledf: Any = False,
    global_test: Any = True,
    **kwargs: Any,
) -> CoxZPHResult: ...
def coef(fit: Any) -> list[float]: ...
def coef_names(fit: Any, *, complete: Any | None = None) -> list[str]: ...
def confint(
    fit: Any,
    parm: Any | None = None,
    *,
    level: Any = 0.95,
) -> list[dict[str, float | str]]: ...
def vcov(fit: Any, *, complete: Any = True) -> list[list[float]]: ...
def loglik(fit: Any) -> float: ...
def model_formula(fit: Any) -> str: ...
def model_summary(fit: Any, **kwargs: Any) -> dict[str, Any]: ...
def model_term_names(fit: Any, terms: Any | None = None) -> list[str]: ...
def model_weights(fit: Any) -> list[float] | None: ...
def nobs(fit: Any) -> int: ...
def degrees_freedom(fit: Any) -> int: ...
def df_residual(fit: Any) -> int: ...
def aic(fit: Any, *, k: Any = 2.0) -> float: ...
def bic(fit: Any) -> float: ...
def royston(
    fit: Any,
    newdata: Any | None = None,
    ties: Any = True,
    adjust: Any = False,
) -> dict[str, float]: ...
def extract_aic(fit: Any, *, scale: Any = 0.0, k: Any = 2.0) -> list[float]: ...
def model_matrix(fit: Any) -> dict[str, Any]: ...
def model_frame(fit: Any) -> dict[str, list[Any]]: ...
def fitted(fit: Any, **kwargs: Any) -> Any: ...
def as_data_frame(result: Any) -> dict[str, list[Any]]: ...
def anova(*fits: Any, test: Any = "Chisq") -> Any: ...
def aareg(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    qrtol: Any = 1e-7,
    nmin: Any | None = None,
    dfbeta: Any = False,
    taper: Any = 1.0,
    test: Any = "aalen",
    cluster: Any | None = None,
    model: Any = False,
    x: Any = False,
    y: Any = False,
    **kwargs: Any,
) -> AaregModelResult: ...
def coxph_detail(
    fit: Any,
    riskmat: Any = False,
    rorder: str = "data",
) -> CoxPHDetailResult: ...
def coxph_wtest(var: Any, b: Any, toler_chol: Any = 1e-9) -> CoxPHWTestResult: ...
def cch(
    formula: str,
    data: Any = None,
    subcoh: Any = None,
    id: Any = None,
    stratum: Any | None = None,
    cohort_size: Any | None = None,
    method: str = "Prentice",
    robust: Any = False,
    *,
    subset: Any | None = None,
    na_action: str | None = "fail",
    **kwargs: Any,
) -> CchModelResult: ...
def coxph(
    formula: str | None = None,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    init: Any | None = None,
    control: Any | None = None,
    ties: str | None = None,
    method: str | None = None,
    singular_ok: Any = True,
    robust: Any | None = None,
    model: Any = False,
    x: Any = False,
    y: Any = True,
    tt: Any | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    istate: Any | None = None,
    statedata: Any | None = None,
    nocenter: Any = (-1, 0, 1),
    offset: Any | None = None,
    strata: Any | None = None,
    iter_max: Any | None = None,
    eps: Any | None = None,
    toler_chol: Any | None = None,
    timefix: Any | None = None,
    **kwargs: Any,
) -> Any: ...
def clogit(
    formula: str,
    data: Any | None = None,
    *,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "fail",
    method: str = "exact",
    **kwargs: Any,
) -> Any: ...
def predict(fit: Any, newdata: Any | None = None, **kwargs: Any) -> Any: ...
def residuals(fit: Any, *, type: str = "martingale", **kwargs: Any) -> Any: ...
def survfit_residuals(
    object: Any,
    times: Any | None = None,
    type: str = "pstate",
    collapse: Any = False,
    weighted: Any | None = None,
    data_frame: Any = False,
    extra: Any = False,
    **kwargs: Any,
) -> Any: ...
def aeqSurv(x: Any, tolerance: Any | None = None) -> Surv: ...
def survcondense(
    formula: str,
    data: Any | None = None,
    subset: Any | None = None,
    weights: Any | None = None,
    na_action: str | None = "na.pass",
    *,
    id: Any,
    start: str | None = None,
    end: str | None = None,
    event: str | None = None,
    **kwargs: Any,
) -> dict[str, list[Any]]: ...
def survSplit(
    formula: Any = None,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = "na.pass",
    id: Any | None = None,
    *,
    cut: Any,
    zero: Any = 0,
    episode: str | None = None,
    start: str | None = None,
    end: str | None = None,
    event: str | None = None,
    added: str | None = None,
    timefix: bool = True,
    response: Any | None = None,
    **kwargs: Any,
) -> dict[str, list[Any]]: ...
def lvcf(id: Any, x: Any, time: Any | None = None, first: bool = True) -> list[Any]: ...
def nostutter(id: Any, x: Any, censor: Any = 0, single: bool = False) -> list[Any]: ...
def dsurvreg(
    x: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]: ...
def psurvreg(
    q: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]: ...
def qsurvreg(
    p: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]: ...
def rsurvreg(
    n: Any,
    mean: Any,
    scale: Any = 1,
    distribution: str = "weibull",
    parms: Any | None = None,
) -> list[float]: ...
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
) -> Any: ...
