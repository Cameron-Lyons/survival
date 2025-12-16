from typing import Optional, List, Tuple, Dict, Any
from typing_extensions import Protocol

class AaregOptions:
    def __init__(
        self,
        formula: str,
        data: List[List[float]],
        variable_names: List[str],
        weights: Optional[List[float]] = None,
        subset: Optional[List[int]] = None,
        na_action: Optional[str] = None,
        qrtol: float = 1e-8,
        nmin: Optional[int] = None,
        dfbeta: bool = False,
        taper: float = 0.0,
        test: List[str] = ...,
        cluster: Optional[List[int]] = None,
        model: bool = False,
        x: bool = False,
        y: bool = False,
    ) -> None: ...

class PSpline:
    def __init__(
        self,
        x: List[float],
        df: int,
        theta: float,
        eps: float,
        method: str,
        boundary_knots: Tuple[float, float],
        intercept: bool,
        penalty: bool,
    ) -> None: ...
    def fit(self) -> None: ...

class CoxCountOutput:
    pass

class LinkFunctionParams:
    def __init__(self, edge: float) -> None: ...
    def blogit(self, input: float) -> float: ...
    def bprobit(self, input: float) -> float: ...
    def bcloglog(self, input: float) -> float: ...
    def blog(self, input: float) -> float: ...

class Subject:
    id: int
    covariates: List[float]
    is_case: bool
    is_subcohort: bool
    stratum: int
    def __init__(
        self,
        id: int,
        covariates: List[float],
        is_case: bool,
        is_subcohort: bool,
        stratum: int,
    ) -> None: ...

class CoxPHModel:
    baseline_hazard: List[float]
    risk_scores: List[float]
    event_times: List[float]
    censoring: List[int]
    def __init__(self) -> None: ...
    @staticmethod
    def new_with_data(
        covariates: List[List[float]],
        event_times: List[float],
        censoring: List[int],
    ) -> "CoxPHModel": ...
    def fit(self, n_iters: int = 10) -> None: ...
    def predict(self, covariates: List[List[float]]) -> List[float]: ...
    def get_coefficients(self) -> List[List[float]]: ...
    def brier_score(self) -> float: ...
    def survival_curve(
        self,
        covariates: List[List[float]],
        time_points: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[List[float]]]: ...
    def add_subject(self, subject: Subject) -> None: ...

class SurvFitKMOutput:
    """Output from Kaplan-Meier survival estimation."""
    time: List[float]
    n_risk: List[float]
    n_event: List[float]
    n_censor: List[float]
    estimate: List[float]
    std_err: List[float]
    conf_lower: List[float]
    conf_upper: List[float]

class FineGrayOutput:
    row: List[int]
    start: List[float]
    end: List[float]
    wt: List[float]
    add: List[int]

class SurvivalFit:
    coefficients: List[float]
    iterations: int
    variance_matrix: List[List[float]]
    log_likelihood: float
    convergence_flag: int
    score_vector: List[float]

class DistributionType:
    pass

class SurvDiffResult:
    observed: List[float]
    expected: List[float]
    variance: List[List[float]]
    chi_squared: float
    degrees_of_freedom: int

class CchMethod:
    """Case-cohort analysis method."""
    Prentice: "CchMethod"
    SelfPrentice: "CchMethod"
    LinYing: "CchMethod"
    IBorgan: "CchMethod"
    IIBorgan: "CchMethod"

class CohortData:
    """Case-cohort data structure."""
    @staticmethod
    def new() -> "CohortData": ...
    def add_subject(self, subject: Subject) -> None: ...
    def get_subject(self, id: int) -> Subject: ...
    def fit(self, method: CchMethod) -> CoxPHModel: ...

class SurvFitAJ:
    """Result of Aalen-Johansen multi-state survival estimation."""
    n_risk: List[List[float]]
    n_event: List[List[float]]
    n_censor: List[List[float]]
    pstate: List[List[float]]
    cumhaz: List[List[float]]
    std_err: Optional[List[List[float]]]
    std_chaz: Optional[List[List[float]]]
    std_auc: Optional[List[List[float]]]
    influence: Optional[List[List[float]]]
    n_enter: Optional[List[List[float]]]
    n_transition: List[List[float]]

class SplitResult:
    """Result of splitting survival data at time points."""
    row: List[int]
    interval: List[int]
    start: List[float]
    end: List[float]
    censor: List[bool]

class ClogitDataSet:
    """Dataset for conditional logistic regression (matched case-control studies)."""
    def __init__(self) -> None: ...
    def add_observation(
        self,
        case_control_status: int,
        stratum: int,
        covariates: List[float],
    ) -> None: ...
    def get_num_observations(self) -> int: ...
    def get_num_covariates(self) -> int: ...

class ConditionalLogisticRegression:
    """Conditional logistic regression for matched case-control studies."""
    coefficients: List[float]
    max_iter: int
    tol: float
    iterations: int
    converged: bool
    def __init__(
        self,
        data: ClogitDataSet,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...
    def fit(self) -> None: ...
    def predict(self, covariates: List[float]) -> float: ...
    def odds_ratios(self) -> List[float]: ...

def aareg(options: AaregOptions) -> Dict[str, Any]: ...

def survfitkm(
    time: List[float],
    status: List[float],
    weights: Optional[List[float]] = None,
    entry_times: Optional[List[float]] = None,
    position: Optional[List[int]] = None,
    reverse: Optional[bool] = None,
    computation_type: Optional[int] = None,
) -> SurvFitKMOutput: ...

def survreg(
    time: List[float],
    status: List[float],
    covariates: List[List[float]],
    weights: Optional[List[float]] = None,
    offsets: Optional[List[float]] = None,
    initial_beta: Optional[List[float]] = None,
    strata: Optional[List[int]] = None,
    distribution: Optional[str] = None,
    max_iter: Optional[int] = None,
    eps: Optional[float] = None,
    tol_chol: Optional[float] = None,
) -> SurvivalFit: ...

def survdiff2(
    time: List[float],
    status: List[int],
    group: List[int],
    strata: Optional[List[int]] = None,
    rho: Optional[float] = None,
) -> SurvDiffResult: ...

def coxmart(
    time: List[float],
    status: List[int],
    score: List[float],
    weights: Optional[List[float]] = None,
    strata: Optional[List[int]] = None,
    method: Optional[int] = None,
) -> List[float]: ...

def finegray(
    tstart: List[float],
    tstop: List[float],
    ctime: List[float],
    cprob: List[float],
    extend: List[bool],
    keep: List[bool],
) -> FineGrayOutput: ...

def perform_cox_regression_frailty(*args: Any, **kwargs: Any) -> Any: ...
def perform_pyears_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_concordance1_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_concordance3_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_concordance_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_score_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_agscore3_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_pystep_calculation(*args: Any, **kwargs: Any) -> Any: ...
def perform_pystep_simple_calculation(*args: Any, **kwargs: Any) -> Any: ...
def collapse(*args: Any, **kwargs: Any) -> Any: ...
def cox_callback(*args: Any, **kwargs: Any) -> Any: ...
def coxcount1(*args: Any, **kwargs: Any) -> Any: ...
def coxcount2(*args: Any, **kwargs: Any) -> Any: ...
def norisk(*args: Any, **kwargs: Any) -> Any: ...
def cipoisson(*args: Any, **kwargs: Any) -> Any: ...
def cipoisson_exact(*args: Any, **kwargs: Any) -> Any: ...
def cipoisson_anscombe(*args: Any, **kwargs: Any) -> Any: ...
def concordance(*args: Any, **kwargs: Any) -> Any: ...
def agexact(*args: Any, **kwargs: Any) -> Any: ...
def agsurv4(*args: Any, **kwargs: Any) -> Any: ...
def agsurv5(*args: Any, **kwargs: Any) -> Any: ...
def agmart(*args: Any, **kwargs: Any) -> Any: ...

def brier(
    predictions: List[float],
    outcomes: List[int],
    weights: Optional[List[float]] = None,
) -> float: ...

def integrated_brier(
    predictions: List[List[float]],
    outcomes: List[int],
    times: List[float],
    weights: Optional[List[float]] = None,
) -> float: ...

def tmerge(
    id: List[int],
    time1: List[float],
    newx: List[float],
    nid: List[int],
    ntime: List[float],
    x: List[float],
) -> List[float]: ...

def tmerge2(
    id: List[int],
    time1: List[float],
    nid: List[int],
    ntime: List[float],
) -> List[int]: ...

def tmerge3(
    id: List[int],
    miss: List[bool],
) -> List[int]: ...

def survsplit(
    tstart: List[float],
    tstop: List[float],
    cut: List[float],
) -> SplitResult: ...

def schoenfeld_residuals(
    y: List[float],
    score: List[float],
    strata: List[int],
    covar: List[float],
    nvar: int,
    method: int = 0,
) -> List[float]:
    """Calculate Schoenfeld residuals for Cox proportional hazards model.

    Schoenfeld residuals are used to test the proportional hazards assumption.

    Args:
        y: Survival data as [start_times..., stop_times..., event_indicators...]
        score: Risk scores (exp(linear predictor))
        strata: Stratum indicators (1 = end of stratum, 0 otherwise)
        covar: Covariate matrix in column-major order
        nvar: Number of covariates
        method: Efron (1) or Breslow (0) method for ties

    Returns:
        Schoenfeld residuals matrix in column-major order
    """
    ...

def cox_score_residuals(
    y: List[float],
    strata: List[int],
    covar: List[float],
    score: List[float],
    weights: List[float],
    nvar: int,
    method: int = 0,
) -> List[float]:
    """Calculate Cox score (dfbeta) residuals.

    Score residuals measure the influence of each observation on coefficient estimates.

    Args:
        y: Survival data as [time..., status...] (length 2*n)
        strata: Stratum indicators for each observation
        covar: Covariate matrix in row-major order (n * nvar)
        score: Risk scores (exp(linear predictor))
        weights: Observation weights
        nvar: Number of covariates
        method: Efron (1) or Breslow (0) method for ties

    Returns:
        Score residuals matrix in row-major order (n * nvar)
    """
    ...

def survfitaj(
    y: List[float],
    sort1: List[int],
    sort2: List[int],
    utime: List[float],
    cstate: List[int],
    wt: List[float],
    grp: List[int],
    ngrp: int,
    p0: List[float],
    i0: List[float],
    sefit: int,
    entry: bool,
    position: List[int],
    hindx: List[List[int]],
    trmat: List[List[int]],
    t0: float,
) -> SurvFitAJ: ...

