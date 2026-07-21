# ruff: noqa: N802

from typing import Any

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
    event: tuple[int, ...]
    start: tuple[float, ...] | None
    time2: tuple[float, ...] | None
    type: str
    def __init__(
        self,
        *args: Any,
        type: str | None = None,
        origin: Any = 0.0,
        time: Any = ...,
        time1: Any = ...,
        time2: Any = ...,
        event: Any = ...,
        status: Any = ...,
        start: Any = ...,
        stop: Any = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    @property
    def status(self) -> tuple[int, ...]: ...

class Surv2:
    time: tuple[float, ...]
    status: tuple[int | None, ...]
    states: tuple[str, ...]
    repeated: bool
    def __init__(self, time: Any, event: Any, repeated: Any = False) -> None: ...
    def __len__(self) -> int: ...

def Surv2data(
    time: Any,
    status: Any,
    *,
    states: Any | None = None,
    repeated: Any = False,
    id: Any,
) -> dict[str, Any]: ...
def totimeline(
    start: Any,
    stop: Any,
    status: Any,
    *,
    states: Any,
    id: Any,
    istate: Any | None = None,
    istate_levels: Any | None = None,
) -> dict[str, Any]: ...

class RateTable:
    summary: str
    def ndim(self) -> int: ...
    def dim_names(self) -> list[str]: ...
    def lookup(self, coords: dict[str, float]) -> float: ...

class SurvExpResult:
    time: list[float]
    surv: list[float]
    n_risk: list[float]
    cumhaz: list[float]
    method: str
    n: int

class PyearsResult:
    pyears: list[float]
    n: list[float]
    offtable: float
    group: list[str]
    observations: int
    event: list[float] | None
    expected: list[float] | None
    tcut: bool

class FineGrayOutput:
    row: list[int]
    start: list[float]
    end: list[float]
    wt: list[float]
    add: list[int]

class TcutResult:
    codes: list[int]
    levels: list[str]
    breaks: list[float]
    counts: list[int]

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

def is_surv(value: Any) -> bool: ...
def is_na_surv(x: Any) -> list[bool]: ...
def format_surv(x: Any) -> list[str]: ...
def fromtimeline(
    time: Any,
    status: Any,
    *,
    id: Any,
    states: Any | None = None,
    data: Any | None = None,
    id_name: Any = "id",
) -> dict[str, Any]: ...
def is_ratetable(
    x: Any,
    has_rates: Any | None = None,
    has_dims: Any | None = None,
    verbose: Any = False,
) -> bool: ...
def ratetableDate(
    x: Any,
    month: Any | None = None,
    day: Any | None = None,
    *,
    origin_year: Any = 1970,
) -> Any: ...
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
) -> SurvExpResult | list[float]: ...
def survexp_individual(
    time: Any,
    age: Any,
    year: Any,
    ratetable: Any | None = None,
    sex: Any | None = None,
) -> list[float]: ...
def pyears(
    response: Any = None,
    data: Any | None = None,
    *,
    time: Any = ...,
    start: Any = ...,
    stop: Any = ...,
    event: Any = ...,
    group: Any | None = None,
    weights: Any | None = None,
    subset: Any | None = None,
    na_action: str | None = None,
    scale: Any = 365.25,
    data_frame: Any = False,
) -> PyearsResult | dict[str, list[Any]]: ...
def finegray(
    tstart: Any,
    tstop: Any,
    ctime: Any,
    cprob: Any,
    extend: Any,
    keep: Any,
) -> FineGrayOutput: ...
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
    best: Any = "after",
    nomatch: Any | None = None,
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
    Boundary_knots: Any = ...,  # noqa: N803
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
    Boundary_knots: Any | None = None,  # noqa: N803
    *,
    boundary_knots: Any | None = None,
    intercept: Any = False,
    penalty: Any = True,
    combine: Any | None = None,
) -> dict[str, Any]: ...
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
    labels: Any | None = None,
) -> Any: ...

class ConcordanceResult:
    concordance: float | list[float]
    n: int
    n_event: int
    reverse: bool
    concordant: float | list[float]
    comparable: float | list[float]
    tied_x: float | list[float]
    tied_y: float | list[float]
    tied_xy: float | list[float]
    ranks: list[dict[str, float]] | list[list[dict[str, float]] | None] | None
    dfbeta: list[float] | list[list[float] | None] | None
    influence: list[list[float]] | list[list[list[float]] | None] | None
    variance: float | list[float | None] | None
    score_names: list[str] | None
    @property
    def c_index(self) -> float | list[float]: ...
    @property
    def var(self) -> float | list[float | None] | None: ...

class PredictResult:
    fit: Any
    se_fit: Any
    def __iter__(self): ...
    @property
    def predictions(self) -> Any: ...
    @property
    def se(self) -> Any: ...

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
    def table(self) -> list[dict[str, float | int | str]]: ...

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
    strata: dict[int, int] | None
    riskmat: list[list[int]] | None
    weights: list[float] | None
    nevent_wt: list[float] | None
    nrisk_wt: list[float] | None
    sortorder: list[int] | None
    @property
    def n_event(self) -> list[int]: ...
    @property
    def n_risk(self) -> list[int]: ...
    @property
    def var_hazard(self) -> list[float]: ...
    @property
    def cumulative_hazard(self) -> list[float]: ...
    def times(self) -> list[float]: ...
    def hazards(self) -> list[float]: ...
    def cumulative_hazards(self) -> list[float]: ...
    def n_risk_at_times(self) -> list[int]: ...
    def schoenfeld_residuals(self) -> list[list[float]]: ...

class CoxPHWTestResult:
    test: list[float]
    df: int
    solve: list[float] | list[list[float]] | float

class CoxBaseHazardResult:
    time: list[float]
    cumhaz: list[float] | list[list[float]]
    strata: list[int] | None
    centered: bool
    curve_strata: list[int] | None
    strata_labels: list[Any] | None
    curve_strata_labels: list[Any] | None
    def __iter__(self): ...
    @property
    def hazard(self) -> list[float] | list[list[float]]: ...
    @property
    def cumulative_hazard(self) -> list[float] | list[list[float]]: ...

class CoxSurvfitResult:
    time: list[float]
    surv: list[list[float]]
    cumhaz: list[list[float]]
    linear_predictors: list[float]
    centered: bool
    strata: list[int] | None
    strata_labels: list[Any] | None
    start_time: float | None
    std_err: list[list[float]]
    std_chaz: list[list[float]]
    conf_lower: list[list[float]]
    conf_upper: list[list[float]]
    model: dict[str, Any] | None
    def __iter__(self): ...
    @property
    def curves(self) -> list[list[float]]: ...
    @property
    def estimate(self) -> list[list[float]]: ...
    @property
    def cumulative_hazard(self) -> list[list[float]]: ...
    @property
    def cumulative_hazard_std_err(self) -> list[list[float]]: ...

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
    n_enter: list[float] | None
    n_risk_count: list[float] | None
    n_event_count: list[float] | None
    n_censor_count: list[float] | None
    n_enter_count: list[float] | None
    model: dict[str, Any] | None
    @property
    def surv(self) -> list[float]: ...
    @property
    def cumulative_hazard(self) -> list[float]: ...
    @property
    def cumulative_hazard_std_err(self) -> list[float]: ...

class SurvfitConfidenceIntervalResult:
    lower: list[float]
    upper: list[float]
    def __iter__(self): ...

class TurnbullSurvfitResult:
    time_points: list[float]
    survival: list[float]
    survival_lower: list[float]
    survival_upper: list[float]
    n_iter: int
    converged: bool
    model: dict[str, Any] | None

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
    **kwargs: Any,
) -> dict[str, float]: ...
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
    type: str | None = None,
    stype: int | None = None,
    ctype: int | None = None,
    id: Any | None = None,
    cluster: Any | None = None,
    robust: Any | None = None,
    istate: Any | None = None,
    etype: Any | None = None,
    model: Any = False,
    error: Any | None = None,
    entry: Any = False,
    timefix: bool = True,
    **kwargs: Any,
) -> Any: ...
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
) -> Any: ...
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
) -> Any: ...
def survfit0(x: Any, *args: Any, **kwargs: Any) -> Any: ...
def aggregate_survfit_result(
    result: CoxSurvfitResult,
    groups: Any | None = None,
    weights: Any | None = None,
) -> CoxSurvfitResult: ...
def survfit_confint(
    p: Any,
    se: Any,
    logse: Any = True,
    conf_type: str | None = None,
    conf_int: Any = 0.95,
    selow: Any | None = None,
    ulimit: Any = True,
    **kwargs: Any,
) -> SurvfitConfidenceIntervalResult: ...
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
) -> Any: ...
def rttright(
    response: Any,
    status: Any | None = None,
    weights: Any | None = None,
    *,
    data: Any | None = None,
    subset: Any | None = None,
    na_action: Any | None = "pass",
    times: Any | None = None,
    id: Any | None = None,
    timefix: bool = True,
    renorm: bool = True,
    **kwargs: Any,
) -> Any: ...
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
    fit: Any = ...,
    status: Any | None = None,
    eval_times: Any | None = None,
    type_: Any | None = None,
    *,
    times: Any | None = None,
    type: Any | None = None,
    collapse: bool = True,
    data_frame: bool = False,
    time: Any | None = None,
    **kwargs: Any,
) -> Any: ...
def survcheck(
    response: Any = ...,
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
    fit: Any | None = None,
    status: Any | None = None,
    linear_predictors: Any | None = None,
    centered: bool = True,
    *,
    newdata: Any | None = None,
    time: Any | None = None,
    entry_times: Any | None = None,
    weights: Any | None = None,
) -> Any: ...
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
    *,
    terms: bool = True,
    singledf: bool = False,
    global_test: bool = True,
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
def model_summary(fit: Any) -> dict[str, Any]: ...
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
def fitted(
    fit: Any,
    *,
    type: str | None = None,
    centered: bool | None = None,
    terms: Any | None = None,
    collapse: Any = False,
    reference: str | None = None,
    se_fit: bool = False,
    times: Any | None = None,
    p: Any | None = None,
    quantiles: Any | None = None,
    **kwargs: Any,
) -> Any: ...
def as_data_frame(result: Any) -> dict[str, list[Any]]: ...
def anova(*fits: Any, test: str | None = "Chisq") -> Any: ...
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
) -> Any: ...
def coxph_wtest(var: Any, b: Any, toler_chol: Any = 1e-9) -> CoxPHWTestResult: ...
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
    id: Any | None = None,
    istate: Any | None = None,
    statedata: Any | None = None,
    singular_ok: Any = True,
    nocenter: Any = (-1, 0, 1),
    control: Any | None = None,
    **kwargs: Any,
) -> Any: ...
def predict(
    fit: Any,
    newdata: Any | None = None,
    *,
    type: str | None = None,
    centered: bool | None = None,
    terms: Any | None = None,
    collapse: Any = False,
    reference: str | None = None,
    se_fit: bool = False,
    times: Any | None = None,
    p: Any | None = None,
    quantiles: Any | None = None,
    **kwargs: Any,
) -> Any: ...
def residuals(
    fit: Any,
    *,
    type: str = "martingale",
    terms: Any | None = None,
    collapse: Any = False,
    weighted: bool | None = None,
    rsigma: bool | None = None,
) -> Any: ...
def survfit_residuals(
    fit: Any,
    times: Any | None = None,
    *,
    type: str = "pstate",
    collapse: Any = False,
    weighted: Any = False,
    data_frame: Any = False,
    extra: Any = False,
    **kwargs: Any,
) -> dict[str, Any]: ...
def aeqSurv(x: Any, tolerance: Any | None = None) -> Surv: ...
def survcondense(
    formula: Any,
    data: Any | None = None,
    subset: Any | None = None,
    weights: Any | None = None,
    na_action: Any | None = "pass",
    *,
    id: Any | None = None,
    start: str = "tstart",
    end: str = "tstop",
    event: str = "event",
    **kwargs: Any,
) -> Any: ...
def survSplit(
    response: Surv,
    data: Any | None = None,
    *,
    cut: Any,
    start: str = "tstart",
    end: str = "tstop",
    event: str = "event",
    episode: str | None = None,
    id: str | None = None,
    zero: Any = 0,
) -> dict[str, list[Any]]: ...
def lvcf(id: Any, x: Any, time: Any | None = None) -> list[Any]: ...
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
