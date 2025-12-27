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
    coefficients: Optional[List[float]]
    fitted: bool
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
    def fit(self) -> List[float]: ...
    def predict(self, new_x: List[float]) -> List[float]: ...
    @property
    def df(self) -> int: ...
    @property
    def eps(self) -> float: ...

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
    def fit(self, n_iters: int = 20) -> None: ...
    def predict(self, covariates: List[List[float]]) -> List[float]: ...
    def get_coefficients(self) -> List[List[float]]: ...
    def brier_score(self) -> float: ...
    def survival_curve(
        self,
        covariates: List[List[float]],
        time_points: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[List[float]]]: ...
    def add_subject(self, subject: Subject) -> None: ...
    def hazard_ratios(self) -> List[float]: ...
    def hazard_ratios_with_ci(
        self, confidence_level: float = 0.95
    ) -> Tuple[List[float], List[float], List[float]]: ...
    def log_likelihood(self) -> float: ...
    def aic(self) -> float: ...
    def bic(self) -> float: ...
    def cumulative_hazard(
        self, covariates: List[List[float]]
    ) -> Tuple[List[float], List[List[float]]]: ...
    def predicted_survival_time(
        self, covariates: List[List[float]], percentile: float = 0.5
    ) -> List[Optional[float]]: ...
    def restricted_mean_survival_time(
        self, covariates: List[List[float]], tau: float
    ) -> List[float]: ...
    def martingale_residuals(self) -> List[float]: ...
    def deviance_residuals(self) -> List[float]: ...
    def dfbeta(self) -> List[List[float]]: ...
    def n_events(self) -> int: ...
    def n_observations(self) -> int: ...
    def summary(self) -> str: ...

class SurvFitKMOutput:
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
    Prentice: "CchMethod"
    SelfPrentice: "CchMethod"
    LinYing: "CchMethod"
    IBorgan: "CchMethod"
    IIBorgan: "CchMethod"

class CohortData:
    @staticmethod
    def new() -> "CohortData": ...
    def add_subject(self, subject: Subject) -> None: ...
    def get_subject(self, id: int) -> Subject: ...
    def fit(self, method: "CchMethod") -> CoxPHModel: ...

class SurvFitAJ:
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
    row: List[int]
    interval: List[int]
    start: List[float]
    end: List[float]
    censor: List[bool]

class ClogitDataSet:
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

class BootstrapResult:
    coefficients: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    se: List[float]
    n_bootstrap: int

class CVResult:
    scores: List[float]
    mean_score: float
    se_score: float
    n_folds: int

class TestResult:
    statistic: float
    p_value: float
    df: int
    test_type: str

class ProportionalityTest:
    variable_names: List[str]
    chi_squared: List[float]
    p_values: List[float]
    global_chi_squared: float
    global_p_value: float
    global_df: int

class NelsonAalenResult:
    time: List[float]
    cumulative_hazard: List[float]
    variance: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    n_risk: List[int]
    n_events: List[int]
    def survival(self) -> List[float]: ...

class StratifiedKMResult:
    strata: List[int]
    times: List[List[float]]
    survival: List[List[float]]
    ci_lower: List[List[float]]
    ci_upper: List[List[float]]
    n_risk: List[List[int]]
    n_events: List[List[int]]

class LogRankResult:
    statistic: float
    p_value: float
    df: int
    observed: List[float]
    expected: List[float]
    variance: float
    weight_type: str

class TrendTestResult:
    statistic: float
    p_value: float
    trend_direction: str

class SampleSizeResult:
    n_total: int
    n_events: int
    n_per_group: List[int]
    power: float
    alpha: float
    hazard_ratio: float
    method: str

class AccrualResult:
    n_total: int
    accrual_time: float
    followup_time: float
    study_duration: float
    expected_events: float

class CalibrationResult:
    risk_groups: List[float]
    predicted: List[float]
    observed: List[float]
    n_per_group: List[int]
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_pvalue: float
    calibration_slope: float
    calibration_intercept: float

class PredictionResult:
    linear_predictor: List[float]
    risk_score: List[float]
    survival_prob: List[List[float]]
    times: List[float]

class RiskStratificationResult:
    risk_groups: List[int]
    cutpoints: List[float]
    group_sizes: List[int]
    group_event_rates: List[float]
    group_median_risk: List[float]

class TdAUCResult:
    times: List[float]
    auc: List[float]
    integrated_auc: float

class RMSTResult:
    rmst: float
    variance: float
    se: float
    ci_lower: float
    ci_upper: float
    tau: float

class RMSTComparisonResult:
    rmst_diff: float
    rmst_ratio: float
    diff_se: float
    diff_ci_lower: float
    diff_ci_upper: float
    ratio_ci_lower: float
    ratio_ci_upper: float
    p_value: float
    rmst_group1: RMSTResult
    rmst_group2: RMSTResult

class MedianSurvivalResult:
    median: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    quantile: float

class CumulativeIncidenceResult:
    time: List[float]
    cif: List[List[float]]
    variance: List[List[float]]
    event_types: List[int]
    n_risk: List[int]

class NNTResult:
    nnt: float
    nnt_ci_lower: float
    nnt_ci_upper: float
    absolute_risk_reduction: float
    arr_ci_lower: float
    arr_ci_upper: float
    time_horizon: float

class LandmarkResult:
    landmark_time: float
    n_at_risk: int
    n_excluded: int
    time: List[float]
    status: List[int]
    original_indices: List[int]

class ConditionalSurvivalResult:
    given_time: float
    target_time: float
    conditional_survival: float
    ci_lower: float
    ci_upper: float
    n_at_risk: int

class HazardRatioResult:
    hazard_ratio: float
    ci_lower: float
    ci_upper: float
    se_log_hr: float
    z_statistic: float
    p_value: float

class SurvivalAtTimeResult:
    time: float
    survival: float
    ci_lower: float
    ci_upper: float
    n_at_risk: int
    n_events: int

class LifeTableResult:
    interval_start: List[float]
    interval_end: List[float]
    n_at_risk: List[float]
    n_deaths: List[float]
    n_censored: List[float]
    n_effective: List[float]
    hazard: List[float]
    survival: List[float]
    se_survival: List[float]

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

def perform_cox_regression_frailty(
    time: List[float],
    event: List[int],
    covariates: List[List[float]],
    offset: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    strata: Optional[List[int]] = None,
    frail: Optional[List[int]] = None,
    max_iter: Optional[int] = None,
    eps: Optional[float] = None,
) -> Dict[str, Any]: ...

def perform_pyears_calculation(
    n: int,
    ny: int,
    doevent: bool,
    doexpect: bool,
    edim: int,
    efac: List[int],
    edims: List[int],
    ecut: List[float],
    expect: List[float],
    y: List[float],
    wt: List[float],
    data: List[float],
    odim: int,
    ofac: List[int],
    odims: List[int],
    ocut: List[float],
) -> Dict[str, Any]: ...

def perform_concordance1_calculation(
    y: List[float],
    wt: List[float],
    indx: List[int],
    ntree: int,
    sortstop: List[int],
    sortstart: List[int],
) -> Dict[str, Any]: ...

def perform_concordance3_calculation(
    y: List[float],
    wt: List[float],
    indx: List[int],
    ntree: int,
    sortstop: List[int],
    sortstart: List[int],
    nvar: int,
    covar: List[float],
    need_residuals: bool,
) -> Dict[str, Any]: ...

def perform_concordance_calculation(
    y: List[float],
    wt: List[float],
    indx: List[int],
    ntree: int,
    sortstop: List[int],
    sortstart: Optional[List[int]] = None,
    nvar: Optional[int] = None,
    covar: Optional[List[float]] = None,
    need_residuals: bool = False,
) -> Dict[str, Any]: ...

def perform_score_calculation(
    time_data: List[float],
    covariates: List[float],
    strata: List[int],
    score: List[float],
    weights: List[float],
    method: int,
) -> Dict[str, Any]: ...

def perform_agscore3_calculation(
    time_data: List[float],
    covariates: List[float],
    strata: List[int],
    score: List[float],
    weights: List[float],
    method: int,
    sort1: List[int],
) -> Dict[str, Any]: ...

def perform_pystep_calculation(
    edim: int,
    data: List[float],
    efac: List[int],
    edims: List[int],
    ecut: List[List[float]],
    tmax: float,
) -> Dict[str, Any]: ...

def perform_pystep_simple_calculation(
    odim: int,
    data: List[float],
    ofac: List[int],
    odims: List[int],
    ocut: List[List[float]],
    timeleft: float,
) -> Dict[str, Any]: ...

def collapse(
    y: List[float],
    x: List[int],
    istate: List[int],
    id: List[int],
    wt: List[float],
    order: List[int],
) -> Dict[str, Any]: ...

def cox_callback(
    time1: List[float],
    time2: List[float],
    status: List[int],
    covar: List[float],
    offset: List[float],
    weights: List[float],
    strata: List[int],
    sort1: List[int],
    sort2: List[int],
    method: int,
    eps: float,
    tol_chol: float,
    beta: List[float],
) -> Dict[str, Any]: ...

def coxcount1(
    time1: List[float],
    time2: List[float],
    status: List[int],
    strata: List[int],
    sort1: List[int],
    sort2: List[int],
) -> Dict[str, Any]: ...

def coxcount2(
    time1: List[float],
    time2: List[float],
    status: List[int],
    strata: List[int],
    sort1: List[int],
    sort2: List[int],
) -> Dict[str, Any]: ...

def norisk(
    time1: List[float],
    time2: List[float],
    status: List[int],
    sort1: List[int],
    sort2: List[int],
    strata: List[int],
) -> List[int]: ...

def cipoisson(k: int, time: float, p: float, method: str) -> Tuple[float, float]: ...
def cipoisson_exact(k: int, time: float, p: float) -> Tuple[float, float]: ...
def cipoisson_anscombe(k: int, time: float, p: float) -> Tuple[float, float]: ...

def concordance(
    y: List[float],
    wt: List[float],
    indx: List[int],
    ntree: int,
    sortstop: List[int],
    sortstart: List[int],
    strata: List[int],
) -> Dict[str, Any]: ...

def agexact(
    maxiter: int,
    nused: int,
    nvar: int,
    start: List[float],
    stop: List[float],
    event: List[int],
    covar: List[float],
    offset: List[float],
    strata: List[int],
    sort: List[int],
    beta: List[float],
    eps: float,
    tol_chol: float,
) -> Dict[str, Any]: ...

def agsurv4(
    y: List[float],
    wt: List[float],
    surv: List[float],
    varh: List[float],
    nrisk: List[float],
    nevent: List[float],
    ncensor: List[float],
    strata: List[int],
) -> Dict[str, Any]: ...

def agsurv5(
    y: List[float],
    wt: List[float],
    id: List[int],
    cluster: List[int],
    risk: List[float],
    position: List[int],
    strata: List[int],
    se_type: int,
) -> Dict[str, Any]: ...

def agmart(
    time: List[float],
    status: List[int],
    score: List[float],
    weights: List[float],
    strata: List[int],
    method: int,
) -> List[float]: ...

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
) -> List[float]: ...

def cox_score_residuals(
    y: List[float],
    strata: List[int],
    covar: List[float],
    score: List[float],
    weights: List[float],
    nvar: int,
    method: int = 0,
) -> List[float]: ...

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

def bootstrap_cox_ci(
    time: List[float],
    status: List[int],
    covariates: List[List[float]],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> BootstrapResult: ...

def bootstrap_survreg_ci(
    time: List[float],
    status: List[int],
    covariates: List[List[float]],
    distribution: str = "weibull",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> BootstrapResult: ...

def cv_cox_concordance(
    time: List[float],
    status: List[int],
    covariates: List[List[float]],
    n_folds: int = 5,
) -> CVResult: ...

def cv_survreg_loglik(
    time: List[float],
    status: List[int],
    covariates: List[List[float]],
    distribution: str = "weibull",
    n_folds: int = 5,
) -> CVResult: ...

def lrt_test(
    log_likelihood_null: float,
    log_likelihood_full: float,
    df: int,
) -> TestResult: ...

def wald_test_py(
    coefficients: List[float],
    variance_matrix: List[List[float]],
) -> TestResult: ...

def score_test_py(
    score: List[float],
    information_matrix: List[List[float]],
) -> TestResult: ...

def ph_test(
    time: List[float],
    status: List[int],
    schoenfeld_residuals: List[List[float]],
    variable_names: List[str],
) -> ProportionalityTest: ...

def nelson_aalen_estimator(
    time: List[float],
    status: List[int],
    weights: Optional[List[float]] = None,
    confidence_level: Optional[float] = None,
) -> NelsonAalenResult: ...

def stratified_kaplan_meier(
    time: List[float],
    status: List[int],
    strata: List[int],
    confidence_level: Optional[float] = None,
) -> StratifiedKMResult: ...

def logrank_test(
    time: List[float],
    status: List[int],
    group: List[int],
    weight_type: Optional[str] = None,
) -> LogRankResult: ...

def fleming_harrington_test(
    time: List[float],
    status: List[int],
    group: List[int],
    p: float,
    q: float,
) -> LogRankResult: ...

def logrank_trend(
    time: List[float],
    status: List[int],
    group: List[int],
    scores: Optional[List[float]] = None,
) -> TrendTestResult: ...

def sample_size_survival(
    hazard_ratio: float,
    power: Optional[float] = None,
    alpha: Optional[float] = None,
    allocation_ratio: Optional[float] = None,
    sided: Optional[int] = None,
) -> SampleSizeResult: ...

def sample_size_survival_freedman(
    hazard_ratio: float,
    prob_event: float,
    power: Optional[float] = None,
    alpha: Optional[float] = None,
    allocation_ratio: Optional[float] = None,
    sided: Optional[int] = None,
) -> SampleSizeResult: ...

def power_survival(
    n_events: int,
    hazard_ratio: float,
    alpha: Optional[float] = None,
    allocation_ratio: Optional[float] = None,
    sided: Optional[int] = None,
) -> float: ...

def expected_events(
    n_total: int,
    hazard_control: float,
    hazard_ratio: float,
    accrual_time: float,
    followup_time: float,
    allocation_ratio: Optional[float] = None,
    dropout_rate: Optional[float] = None,
) -> AccrualResult: ...

def calibration(
    predicted_risk: List[float],
    observed_event: List[int],
    n_groups: Optional[int] = None,
) -> CalibrationResult: ...

def predict_cox(
    coef: List[float],
    x: List[List[float]],
    baseline_hazard: List[float],
    baseline_times: List[float],
    pred_times: List[float],
) -> PredictionResult: ...

def risk_stratification(
    risk_scores: List[float],
    events: List[int],
    n_groups: Optional[int] = None,
) -> RiskStratificationResult: ...

def td_auc(
    time: List[float],
    status: List[int],
    risk_score: List[float],
    eval_times: List[float],
) -> TdAUCResult: ...

def rmst(
    time: List[float],
    status: List[int],
    tau: float,
    confidence_level: Optional[float] = None,
) -> RMSTResult: ...

def rmst_comparison(
    time: List[float],
    status: List[int],
    group: List[int],
    tau: float,
    confidence_level: Optional[float] = None,
) -> RMSTComparisonResult: ...

def survival_quantile(
    time: List[float],
    status: List[int],
    quantile: Optional[float] = None,
    confidence_level: Optional[float] = None,
) -> MedianSurvivalResult: ...

def cumulative_incidence(
    time: List[float],
    status: List[int],
) -> CumulativeIncidenceResult: ...

def number_needed_to_treat(
    time: List[float],
    status: List[int],
    group: List[int],
    time_horizon: float,
    confidence_level: Optional[float] = None,
) -> NNTResult: ...

def landmark_analysis(
    time: List[float],
    status: List[int],
    landmark_time: float,
) -> LandmarkResult: ...

def conditional_survival(
    time: List[float],
    status: List[int],
    given_time: float,
    target_time: float,
    confidence_level: Optional[float] = None,
) -> ConditionalSurvivalResult: ...

def hazard_ratio(
    time: List[float],
    status: List[int],
    group: List[int],
    confidence_level: Optional[float] = None,
) -> HazardRatioResult: ...

def survival_at_times(
    time: List[float],
    status: List[int],
    eval_times: List[float],
    confidence_level: Optional[float] = None,
) -> List[SurvivalAtTimeResult]: ...

def life_table(
    time: List[float],
    status: List[int],
    breaks: List[float],
) -> LifeTableResult: ...
