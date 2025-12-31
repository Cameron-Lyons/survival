from typing import Any

class AaregOptions:
    def __init__(
        self,
        formula: str,
        data: list[list[float]],
        variable_names: list[str],
        weights: list[float] | None = None,
        subset: list[int] | None = None,
        na_action: str | None = None,
        qrtol: float = 1e-8,
        nmin: int | None = None,
        dfbeta: bool = False,
        taper: float = 0.0,
        test: list[str] = ...,
        cluster: list[int] | None = None,
        model: bool = False,
        x: bool = False,
        y: bool = False,
    ) -> None: ...

class PSpline:
    coefficients: list[float] | None
    fitted: bool
    def __init__(
        self,
        x: list[float],
        df: int,
        theta: float,
        eps: float,
        method: str,
        boundary_knots: tuple[float, float],
        intercept: bool,
        penalty: bool,
    ) -> None: ...
    def fit(self) -> list[float]: ...
    def predict(self, new_x: list[float]) -> list[float]: ...
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
    covariates: list[float]
    is_case: bool
    is_subcohort: bool
    stratum: int
    def __init__(
        self,
        id: int,
        covariates: list[float],
        is_case: bool,
        is_subcohort: bool,
        stratum: int,
    ) -> None: ...

class CoxPHModel:
    baseline_hazard: list[float]
    risk_scores: list[float]
    event_times: list[float]
    censoring: list[int]
    def __init__(self) -> None: ...
    @staticmethod
    def new_with_data(
        covariates: list[list[float]],
        event_times: list[float],
        censoring: list[int],
    ) -> "CoxPHModel": ...
    def fit(self, n_iters: int = 20) -> None: ...
    def predict(self, covariates: list[list[float]]) -> list[float]: ...
    def get_coefficients(self) -> list[list[float]]: ...
    def brier_score(self) -> float: ...
    def survival_curve(
        self,
        covariates: list[list[float]],
        time_points: list[float] | None = None,
    ) -> tuple[list[float], list[list[float]]]: ...
    def add_subject(self, subject: Subject) -> None: ...
    def hazard_ratios(self) -> list[float]: ...
    def hazard_ratios_with_ci(
        self, confidence_level: float = 0.95
    ) -> tuple[list[float], list[float], list[float]]: ...
    def log_likelihood(self) -> float: ...
    def aic(self) -> float: ...
    def bic(self) -> float: ...
    def cumulative_hazard(
        self, covariates: list[list[float]]
    ) -> tuple[list[float], list[list[float]]]: ...
    def predicted_survival_time(
        self, covariates: list[list[float]], percentile: float = 0.5
    ) -> list[float | None]: ...
    def restricted_mean_survival_time(
        self, covariates: list[list[float]], tau: float
    ) -> list[float]: ...
    def martingale_residuals(self) -> list[float]: ...
    def deviance_residuals(self) -> list[float]: ...
    def dfbeta(self) -> list[list[float]]: ...
    def n_events(self) -> int: ...
    def n_observations(self) -> int: ...
    def summary(self) -> str: ...

class SurvFitKMOutput:
    time: list[float]
    n_risk: list[float]
    n_event: list[float]
    n_censor: list[float]
    estimate: list[float]
    std_err: list[float]
    conf_lower: list[float]
    conf_upper: list[float]

class FineGrayOutput:
    row: list[int]
    start: list[float]
    end: list[float]
    wt: list[float]
    add: list[int]

class SurvivalFit:
    coefficients: list[float]
    iterations: int
    variance_matrix: list[list[float]]
    log_likelihood: float
    convergence_flag: int
    score_vector: list[float]

class DistributionType:
    pass

class SurvDiffResult:
    observed: list[float]
    expected: list[float]
    variance: list[list[float]]
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
    n_risk: list[list[float]]
    n_event: list[list[float]]
    n_censor: list[list[float]]
    pstate: list[list[float]]
    cumhaz: list[list[float]]
    std_err: list[list[float]] | None
    std_chaz: list[list[float]] | None
    std_auc: list[list[float]] | None
    influence: list[list[float]] | None
    n_enter: list[list[float]] | None
    n_transition: list[list[float]]

class SplitResult:
    row: list[int]
    interval: list[int]
    start: list[float]
    end: list[float]
    censor: list[bool]

class ClogitDataSet:
    def __init__(self) -> None: ...
    def add_observation(
        self,
        case_control_status: int,
        stratum: int,
        covariates: list[float],
    ) -> None: ...
    def get_num_observations(self) -> int: ...
    def get_num_covariates(self) -> int: ...

class ConditionalLogisticRegression:
    coefficients: list[float]
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
    def predict(self, covariates: list[float]) -> float: ...
    def odds_ratios(self) -> list[float]: ...

class BootstrapResult:
    coefficients: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    se: list[float]
    n_bootstrap: int

class CVResult:
    scores: list[float]
    mean_score: float
    se_score: float
    n_folds: int

class TestResult:
    statistic: float
    p_value: float
    df: int
    test_type: str

class ProportionalityTest:
    variable_names: list[str]
    chi_squared: list[float]
    p_values: list[float]
    global_chi_squared: float
    global_p_value: float
    global_df: int

class NelsonAalenResult:
    time: list[float]
    cumulative_hazard: list[float]
    variance: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    n_risk: list[int]
    n_events: list[int]
    def survival(self) -> list[float]: ...

class StratifiedKMResult:
    strata: list[int]
    times: list[list[float]]
    survival: list[list[float]]
    ci_lower: list[list[float]]
    ci_upper: list[list[float]]
    n_risk: list[list[int]]
    n_events: list[list[int]]

class LogRankResult:
    statistic: float
    p_value: float
    df: int
    observed: list[float]
    expected: list[float]
    variance: float
    weight_type: str

class TrendTestResult:
    statistic: float
    p_value: float
    trend_direction: str

class SampleSizeResult:
    n_total: int
    n_events: int
    n_per_group: list[int]
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
    risk_groups: list[float]
    predicted: list[float]
    observed: list[float]
    n_per_group: list[int]
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_pvalue: float
    calibration_slope: float
    calibration_intercept: float

class PredictionResult:
    linear_predictor: list[float]
    risk_score: list[float]
    survival_prob: list[list[float]]
    times: list[float]

class RiskStratificationResult:
    risk_groups: list[int]
    cutpoints: list[float]
    group_sizes: list[int]
    group_event_rates: list[float]
    group_median_risk: list[float]

class TdAUCResult:
    times: list[float]
    auc: list[float]
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
    median: float | None
    ci_lower: float | None
    ci_upper: float | None
    quantile: float

class CumulativeIncidenceResult:
    time: list[float]
    cif: list[list[float]]
    variance: list[list[float]]
    event_types: list[int]
    n_risk: list[int]

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
    time: list[float]
    status: list[int]
    original_indices: list[int]

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
    interval_start: list[float]
    interval_end: list[float]
    n_at_risk: list[float]
    n_deaths: list[float]
    n_censored: list[float]
    n_effective: list[float]
    hazard: list[float]
    survival: list[float]
    se_survival: list[float]

def aareg(options: AaregOptions) -> dict[str, Any]: ...

def survfitkm(
    time: list[float],
    status: list[float],
    weights: list[float] | None = None,
    entry_times: list[float] | None = None,
    position: list[int] | None = None,
    reverse: bool | None = None,
    computation_type: int | None = None,
) -> SurvFitKMOutput: ...

def survreg(
    time: list[float],
    status: list[float],
    covariates: list[list[float]],
    weights: list[float] | None = None,
    offsets: list[float] | None = None,
    initial_beta: list[float] | None = None,
    strata: list[int] | None = None,
    distribution: str | None = None,
    max_iter: int | None = None,
    eps: float | None = None,
    tol_chol: float | None = None,
) -> SurvivalFit: ...

def survdiff2(
    time: list[float],
    status: list[int],
    group: list[int],
    strata: list[int] | None = None,
    rho: float | None = None,
) -> SurvDiffResult: ...

def coxmart(
    time: list[float],
    status: list[int],
    score: list[float],
    weights: list[float] | None = None,
    strata: list[int] | None = None,
    method: int | None = None,
) -> list[float]: ...

def finegray(
    tstart: list[float],
    tstop: list[float],
    ctime: list[float],
    cprob: list[float],
    extend: list[bool],
    keep: list[bool],
) -> FineGrayOutput: ...

def perform_cox_regression_frailty(
    time: list[float],
    event: list[int],
    covariates: list[list[float]],
    offset: list[float] | None = None,
    weights: list[float] | None = None,
    strata: list[int] | None = None,
    frail: list[int] | None = None,
    max_iter: int | None = None,
    eps: float | None = None,
) -> dict[str, Any]: ...

def perform_pyears_calculation(
    n: int,
    ny: int,
    doevent: bool,
    doexpect: bool,
    edim: int,
    efac: list[int],
    edims: list[int],
    ecut: list[float],
    expect: list[float],
    y: list[float],
    wt: list[float],
    data: list[float],
    odim: int,
    ofac: list[int],
    odims: list[int],
    ocut: list[float],
) -> dict[str, Any]: ...

def perform_concordance1_calculation(
    y: list[float],
    wt: list[float],
    indx: list[int],
    ntree: int,
    sortstop: list[int],
    sortstart: list[int],
) -> dict[str, Any]: ...

def perform_concordance3_calculation(
    y: list[float],
    wt: list[float],
    indx: list[int],
    ntree: int,
    sortstop: list[int],
    sortstart: list[int],
    nvar: int,
    covar: list[float],
    need_residuals: bool,
) -> dict[str, Any]: ...

def perform_concordance_calculation(
    y: list[float],
    wt: list[float],
    indx: list[int],
    ntree: int,
    sortstop: list[int],
    sortstart: list[int] | None = None,
    nvar: int | None = None,
    covar: list[float] | None = None,
    need_residuals: bool = False,
) -> dict[str, Any]: ...

def perform_score_calculation(
    time_data: list[float],
    covariates: list[float],
    strata: list[int],
    score: list[float],
    weights: list[float],
    method: int,
) -> dict[str, Any]: ...

def perform_agscore3_calculation(
    time_data: list[float],
    covariates: list[float],
    strata: list[int],
    score: list[float],
    weights: list[float],
    method: int,
    sort1: list[int],
) -> dict[str, Any]: ...

def perform_pystep_calculation(
    edim: int,
    data: list[float],
    efac: list[int],
    edims: list[int],
    ecut: list[list[float]],
    tmax: float,
) -> dict[str, Any]: ...

def perform_pystep_simple_calculation(
    odim: int,
    data: list[float],
    ofac: list[int],
    odims: list[int],
    ocut: list[list[float]],
    timeleft: float,
) -> dict[str, Any]: ...

def collapse(
    y: list[float],
    x: list[int],
    istate: list[int],
    id: list[int],
    wt: list[float],
    order: list[int],
) -> dict[str, Any]: ...

def cox_callback(
    time1: list[float],
    time2: list[float],
    status: list[int],
    covar: list[float],
    offset: list[float],
    weights: list[float],
    strata: list[int],
    sort1: list[int],
    sort2: list[int],
    method: int,
    eps: float,
    tol_chol: float,
    beta: list[float],
) -> dict[str, Any]: ...

def coxcount1(
    time1: list[float],
    time2: list[float],
    status: list[int],
    strata: list[int],
    sort1: list[int],
    sort2: list[int],
) -> dict[str, Any]: ...

def coxcount2(
    time1: list[float],
    time2: list[float],
    status: list[int],
    strata: list[int],
    sort1: list[int],
    sort2: list[int],
) -> dict[str, Any]: ...

def norisk(
    time1: list[float],
    time2: list[float],
    status: list[int],
    sort1: list[int],
    sort2: list[int],
    strata: list[int],
) -> list[int]: ...

def cipoisson(k: int, time: float, p: float, method: str) -> tuple[float, float]: ...
def cipoisson_exact(k: int, time: float, p: float) -> tuple[float, float]: ...
def cipoisson_anscombe(k: int, time: float, p: float) -> tuple[float, float]: ...

def concordance(
    y: list[float],
    wt: list[float],
    indx: list[int],
    ntree: int,
    sortstop: list[int],
    sortstart: list[int],
    strata: list[int],
) -> dict[str, Any]: ...

def agexact(
    maxiter: int,
    nused: int,
    nvar: int,
    start: list[float],
    stop: list[float],
    event: list[int],
    covar: list[float],
    offset: list[float],
    strata: list[int],
    sort: list[int],
    beta: list[float],
    eps: float,
    tol_chol: float,
) -> dict[str, Any]: ...

def agsurv4(
    y: list[float],
    wt: list[float],
    surv: list[float],
    varh: list[float],
    nrisk: list[float],
    nevent: list[float],
    ncensor: list[float],
    strata: list[int],
) -> dict[str, Any]: ...

def agsurv5(
    y: list[float],
    wt: list[float],
    id: list[int],
    cluster: list[int],
    risk: list[float],
    position: list[int],
    strata: list[int],
    se_type: int,
) -> dict[str, Any]: ...

def agmart(
    time: list[float],
    status: list[int],
    score: list[float],
    weights: list[float],
    strata: list[int],
    method: int,
) -> list[float]: ...

def brier(
    predictions: list[float],
    outcomes: list[int],
    weights: list[float] | None = None,
) -> float: ...

def integrated_brier(
    predictions: list[list[float]],
    outcomes: list[int],
    times: list[float],
    weights: list[float] | None = None,
) -> float: ...

def tmerge(
    id: list[int],
    time1: list[float],
    newx: list[float],
    nid: list[int],
    ntime: list[float],
    x: list[float],
) -> list[float]: ...

def tmerge2(
    id: list[int],
    time1: list[float],
    nid: list[int],
    ntime: list[float],
) -> list[int]: ...

def tmerge3(
    id: list[int],
    miss: list[bool],
) -> list[int]: ...

def survsplit(
    tstart: list[float],
    tstop: list[float],
    cut: list[float],
) -> SplitResult: ...

def schoenfeld_residuals(
    y: list[float],
    score: list[float],
    strata: list[int],
    covar: list[float],
    nvar: int,
    method: int = 0,
) -> list[float]: ...

def cox_score_residuals(
    y: list[float],
    strata: list[int],
    covar: list[float],
    score: list[float],
    weights: list[float],
    nvar: int,
    method: int = 0,
) -> list[float]: ...

def survfitaj(
    y: list[float],
    sort1: list[int],
    sort2: list[int],
    utime: list[float],
    cstate: list[int],
    wt: list[float],
    grp: list[int],
    ngrp: int,
    p0: list[float],
    i0: list[float],
    sefit: int,
    entry: bool,
    position: list[int],
    hindx: list[list[int]],
    trmat: list[list[int]],
    t0: float,
) -> SurvFitAJ: ...

def bootstrap_cox_ci(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> BootstrapResult: ...

def bootstrap_survreg_ci(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    distribution: str = "weibull",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> BootstrapResult: ...

def cv_cox_concordance(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    n_folds: int = 5,
) -> CVResult: ...

def cv_survreg_loglik(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    distribution: str = "weibull",
    n_folds: int = 5,
) -> CVResult: ...

def lrt_test(
    log_likelihood_null: float,
    log_likelihood_full: float,
    df: int,
) -> TestResult: ...

def wald_test_py(
    coefficients: list[float],
    variance_matrix: list[list[float]],
) -> TestResult: ...

def score_test_py(
    score: list[float],
    information_matrix: list[list[float]],
) -> TestResult: ...

def ph_test(
    time: list[float],
    status: list[int],
    schoenfeld_residuals: list[list[float]],
    variable_names: list[str],
) -> ProportionalityTest: ...

def nelson_aalen_estimator(
    time: list[float],
    status: list[int],
    weights: list[float] | None = None,
    confidence_level: float | None = None,
) -> NelsonAalenResult: ...

def stratified_kaplan_meier(
    time: list[float],
    status: list[int],
    strata: list[int],
    confidence_level: float | None = None,
) -> StratifiedKMResult: ...

def logrank_test(
    time: list[float],
    status: list[int],
    group: list[int],
    weight_type: str | None = None,
) -> LogRankResult: ...

def fleming_harrington_test(
    time: list[float],
    status: list[int],
    group: list[int],
    p: float,
    q: float,
) -> LogRankResult: ...

def logrank_trend(
    time: list[float],
    status: list[int],
    group: list[int],
    scores: list[float] | None = None,
) -> TrendTestResult: ...

def sample_size_survival(
    hazard_ratio: float,
    power: float | None = None,
    alpha: float | None = None,
    allocation_ratio: float | None = None,
    sided: int | None = None,
) -> SampleSizeResult: ...

def sample_size_survival_freedman(
    hazard_ratio: float,
    prob_event: float,
    power: float | None = None,
    alpha: float | None = None,
    allocation_ratio: float | None = None,
    sided: int | None = None,
) -> SampleSizeResult: ...

def power_survival(
    n_events: int,
    hazard_ratio: float,
    alpha: float | None = None,
    allocation_ratio: float | None = None,
    sided: int | None = None,
) -> float: ...

def expected_events(
    n_total: int,
    hazard_control: float,
    hazard_ratio: float,
    accrual_time: float,
    followup_time: float,
    allocation_ratio: float | None = None,
    dropout_rate: float | None = None,
) -> AccrualResult: ...

def calibration(
    predicted_risk: list[float],
    observed_event: list[int],
    n_groups: int | None = None,
) -> CalibrationResult: ...

def predict_cox(
    coef: list[float],
    x: list[list[float]],
    baseline_hazard: list[float],
    baseline_times: list[float],
    pred_times: list[float],
) -> PredictionResult: ...

def risk_stratification(
    risk_scores: list[float],
    events: list[int],
    n_groups: int | None = None,
) -> RiskStratificationResult: ...

def td_auc(
    time: list[float],
    status: list[int],
    risk_score: list[float],
    eval_times: list[float],
) -> TdAUCResult: ...

def rmst(
    time: list[float],
    status: list[int],
    tau: float,
    confidence_level: float | None = None,
) -> RMSTResult: ...

def rmst_comparison(
    time: list[float],
    status: list[int],
    group: list[int],
    tau: float,
    confidence_level: float | None = None,
) -> RMSTComparisonResult: ...

def survival_quantile(
    time: list[float],
    status: list[int],
    quantile: float | None = None,
    confidence_level: float | None = None,
) -> MedianSurvivalResult: ...

def cumulative_incidence(
    time: list[float],
    status: list[int],
) -> CumulativeIncidenceResult: ...

def number_needed_to_treat(
    time: list[float],
    status: list[int],
    group: list[int],
    time_horizon: float,
    confidence_level: float | None = None,
) -> NNTResult: ...

def landmark_analysis(
    time: list[float],
    status: list[int],
    landmark_time: float,
) -> LandmarkResult: ...

def conditional_survival(
    time: list[float],
    status: list[int],
    given_time: float,
    target_time: float,
    confidence_level: float | None = None,
) -> ConditionalSurvivalResult: ...

def hazard_ratio(
    time: list[float],
    status: list[int],
    group: list[int],
    confidence_level: float | None = None,
) -> HazardRatioResult: ...

def survival_at_times(
    time: list[float],
    status: list[int],
    eval_times: list[float],
    confidence_level: float | None = None,
) -> list[SurvivalAtTimeResult]: ...

def life_table(
    time: list[float],
    status: list[int],
    breaks: list[float],
) -> LifeTableResult: ...
