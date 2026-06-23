from typing import Any

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

def is_surv(value: Any) -> bool: ...

class ConcordanceResult:
    concordance: float | list[float]
    n: int
    n_event: int
    reverse: bool
    concordant: float | list[float]
    comparable: float | list[float]
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
    model: dict[str, Any] | None
    @property
    def surv(self) -> list[float]: ...
    @property
    def cumulative_hazard(self) -> list[float]: ...
    @property
    def cumulative_hazard_std_err(self) -> list[float]: ...

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
def coef_names(fit: Any, *, complete: Any = False) -> list[str]: ...
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
