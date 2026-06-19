from collections.abc import Callable
from typing import Any

class GBSurvLoss:
    AFT: GBSurvLoss
    CoxPH: GBSurvLoss
    Huber: GBSurvLoss
    SquaredError: GBSurvLoss

class SplitRule:
    Conservation: SplitRule
    Hellinger: SplitRule
    LogRank: SplitRule
    LogRankScore: SplitRule

class Activation:
    ReLU: Activation
    SELU: Activation
    Tanh: Activation
    def __init__(self, name: str) -> None: ...

class DistributionType:
    extreme_value: DistributionType
    logistic: DistributionType
    gaussian: DistributionType
    weibull: DistributionType
    lognormal: DistributionType
    loglogistic: DistributionType

class DimType:
    Age: DimType
    Continuous: DimType
    Factor: DimType
    Year: DimType

class RateDimension:
    def __init__(
        self,
        name: str,
        dim_type: DimType,
        cutpoints: list[float],
        levels: list[str] | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def dim_type(self) -> DimType: ...
    @property
    def levels(self) -> list[str] | None: ...
    @property
    def cutpoints(self) -> list[float]: ...

class RateTable:
    def __init__(
        self,
        dimensions: list[RateDimension],
        rates: list[float],
        summary: str | None = None,
    ) -> None: ...
    @property
    def summary(self) -> str: ...
    def ndim(self) -> int: ...
    def dim_names(self) -> list[str]: ...
    def lookup(self, coords: dict[str, float]) -> float: ...
    def lookup_interpolate(self, coords: dict[str, float]) -> float: ...
    def cumulative_hazard(
        self,
        age_start: float,
        age_end: float,
        year_start: float,
        sex: int | None = None,
    ) -> float: ...
    def expected_survival(
        self,
        age_start: float,
        age_end: float,
        year_start: float,
        sex: int | None = None,
    ) -> float: ...

class NetSurvivalMethod:
    EdererI: NetSurvivalMethod
    EdererII: NetSurvivalMethod
    Hakulinen: NetSurvivalMethod
    Pohar_Perme: NetSurvivalMethod
    def __init__(self, name: str) -> None: ...

class SpatialCorrelationStructure:
    CAR: SpatialCorrelationStructure
    SAR: SpatialCorrelationStructure
    Exponential: SpatialCorrelationStructure
    Matern: SpatialCorrelationStructure
    def __init__(self, name: str) -> None: ...

class SpatialFrailtyResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_ci_lower(self) -> list[float]: ...
    @property
    def hr_ci_upper(self) -> list[float]: ...
    @property
    def spatial_frailties(self) -> list[float]: ...
    @property
    def frailty_variance(self) -> float: ...
    @property
    def spatial_correlation(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def dic(self) -> float: ...
    @property
    def n_regions(self) -> int: ...
    @property
    def converged(self) -> bool: ...

class CentralityType:
    Degree: CentralityType
    Betweenness: CentralityType
    Closeness: CentralityType
    Eigenvector: CentralityType
    PageRank: CentralityType
    def __init__(self, name: str) -> None: ...

class NetworkSurvivalConfig:
    include_peer_effects: bool
    include_centrality: bool
    centrality_type: CentralityType
    peer_lag: int
    max_iter: int
    tol: float
    def __init__(
        self,
        include_peer_effects: bool = True,
        include_centrality: bool = True,
        centrality_type: CentralityType = CentralityType.Degree,
        peer_lag: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...

class NetworkSurvivalResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_ci_lower(self) -> list[float]: ...
    @property
    def hr_ci_upper(self) -> list[float]: ...
    @property
    def peer_effect(self) -> float: ...
    @property
    def peer_effect_se(self) -> float: ...
    @property
    def centrality_effect(self) -> float: ...
    @property
    def centrality_effect_se(self) -> float: ...
    @property
    def centrality_values(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_edges(self) -> int: ...
    @property
    def converged(self) -> bool: ...

class DiffusionSurvivalConfig:
    diffusion_rate: float
    recovery_rate: float
    susceptibility_covariate: bool
    max_iter: int
    tol: float
    def __init__(
        self,
        diffusion_rate: float = 0.1,
        recovery_rate: float = 0.05,
        susceptibility_covariate: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...

class DiffusionSurvivalResult:
    @property
    def diffusion_rate(self) -> float: ...
    @property
    def diffusion_rate_se(self) -> float: ...
    @property
    def recovery_rate(self) -> float: ...
    @property
    def recovery_rate_se(self) -> float: ...
    @property
    def susceptibility_coef(self) -> float: ...
    @property
    def susceptibility_se(self) -> float: ...
    @property
    def infection_probabilities(self) -> list[float]: ...
    @property
    def expected_infection_times(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def r0(self) -> float: ...
    @property
    def converged(self) -> bool: ...

class NetworkHeterogeneityResult:
    @property
    def community_hazard_ratios(self) -> list[float]: ...
    @property
    def within_community_correlation(self) -> float: ...
    @property
    def between_community_effect(self) -> float: ...
    @property
    def modularity(self) -> float: ...
    @property
    def community_assignments(self) -> list[int]: ...
    @property
    def log_likelihood(self) -> float: ...

class ReliabilityScale:
    ClogLog: ReliabilityScale
    Cumhaz: ReliabilityScale
    LogLogistic: ReliabilityScale
    Probit: ReliabilityScale
    Surv: ReliabilityScale
    @staticmethod
    def from_str(s: str) -> ReliabilityScale: ...

class ReliabilityResult:
    def __init__(
        self,
        time: list[float],
        estimate: list[float],
        std_err: list[float] | None = None,
        lower: list[float] | None = None,
        upper: list[float] | None = None,
        scale: str = "surv",
    ) -> None: ...
    @property
    def time(self) -> list[float]: ...
    @property
    def estimate(self) -> list[float]: ...
    @property
    def std_err(self) -> list[float] | None: ...
    @property
    def lower(self) -> list[float] | None: ...
    @property
    def upper(self) -> list[float] | None: ...
    @property
    def scale(self) -> str: ...

class WarrantyConfig:
    def __init__(
        self,
        warranty_period: float,
        cost_per_failure: float,
        cost_per_repair: float | None = None,
        discount_rate: float = 0.0,
    ) -> None: ...
    @property
    def warranty_period(self) -> float: ...
    @warranty_period.setter
    def warranty_period(self, value: float) -> None: ...
    @property
    def cost_per_failure(self) -> float: ...
    @cost_per_failure.setter
    def cost_per_failure(self, value: float) -> None: ...
    @property
    def cost_per_repair(self) -> float: ...
    @cost_per_repair.setter
    def cost_per_repair(self, value: float) -> None: ...
    @property
    def discount_rate(self) -> float: ...
    @discount_rate.setter
    def discount_rate(self, value: float) -> None: ...

class WarrantyResult:
    @property
    def expected_failures(self) -> float: ...
    @property
    def expected_cost(self) -> float: ...
    @property
    def cost_per_unit(self) -> float: ...
    @property
    def failure_probability(self) -> float: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def cumulative_failures(self) -> list[float]: ...
    @property
    def cumulative_cost(self) -> list[float]: ...

class RenewalResult:
    @property
    def expected_renewals(self) -> float: ...
    @property
    def renewal_variance(self) -> float: ...
    @property
    def mtbf(self) -> float: ...
    @property
    def availability(self) -> float: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def renewal_function(self) -> list[float]: ...

class ReliabilityGrowthResult:
    @property
    def initial_mtbf(self) -> float: ...
    @property
    def final_mtbf(self) -> float: ...
    @property
    def growth_rate(self) -> float: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def mtbf_trajectory(self) -> list[float]: ...

class AaregOptions:
    formula: str
    data: list[list[float]]
    variable_names: list[str]
    weights: list[float] | None
    subset: list[int] | None
    na_action: str | None
    qrtol: float
    nmin: int | None
    dfbeta: bool
    taper: float
    test: list[str]
    cluster: dict[str, int] | None
    model: bool
    x: bool
    y: bool
    max_iter: int
    def __init__(
        self,
        formula: str,
        data: list[list[float]],
        variable_names: list[str],
        max_iter: int = 100,
    ) -> None: ...

class AaregConfidenceInterval:
    lower_bound: float
    upper_bound: float

class AaregFitDetails:
    iterations: int
    converged: bool
    final_objective_value: float
    convergence_threshold: float
    change_in_objective: float | None
    max_iterations: int | None
    optimization_method: str | None
    warnings: list[str]

class AaregDiagnostics:
    dfbetas: list[float] | None
    cooks_distance: list[float] | None
    leverage: list[float] | None
    deviance_residuals: list[float] | None
    martingale_residuals: list[float] | None
    schoenfeld_residuals: list[float] | None
    score_residuals: list[float] | None
    additional_measures: list[float] | None

class AaregResult:
    coefficients: list[float]
    standard_errors: list[float]
    confidence_intervals: list[AaregConfidenceInterval]
    p_values: list[float]
    goodness_of_fit: float
    fit_details: AaregFitDetails | None
    residuals: list[float] | None
    diagnostics: AaregDiagnostics | None

def aareg(options: AaregOptions) -> AaregResult: ...

class CoxCountOutput:
    @property
    def time(self) -> list[float]: ...
    @property
    def nrisk(self) -> list[int]: ...
    @property
    def index(self) -> list[int]: ...
    @property
    def status(self) -> list[int]: ...

class SplitResult:
    @property
    def row(self) -> list[int]: ...
    @property
    def interval(self) -> list[int]: ...
    @property
    def start(self) -> list[float]: ...
    @property
    def end(self) -> list[float]: ...
    @property
    def censor(self) -> list[bool]: ...

class ClusterResult:
    @property
    def cluster_ids(self) -> list[int]: ...
    @property
    def n_clusters(self) -> int: ...
    @property
    def cluster_sizes(self) -> list[int]: ...
    @property
    def levels(self) -> list[str]: ...

class StrataResult:
    @property
    def strata(self) -> list[int]: ...
    @property
    def levels(self) -> list[str]: ...
    @property
    def counts(self) -> list[int]: ...
    @property
    def n_strata(self) -> int: ...

class AeqSurvResult:
    @property
    def time(self) -> list[float]: ...
    @property
    def adjusted_count(self) -> int: ...
    @property
    def adjusted_indices(self) -> list[int]: ...

class NearDateResult:
    @property
    def indices(self) -> list[int | None]: ...
    @property
    def distances(self) -> list[float | None]: ...
    @property
    def n_matched(self) -> int: ...

class RttrightResult:
    @property
    def weights(self) -> list[float]: ...
    @property
    def time(self) -> list[float]: ...
    @property
    def status(self) -> list[int]: ...
    @property
    def order(self) -> list[int]: ...

class Surv2DataResult:
    @property
    def id(self) -> list[int]: ...
    @property
    def time1(self) -> list[float]: ...
    @property
    def time2(self) -> list[float]: ...
    @property
    def status(self) -> list[int]: ...
    @property
    def row_index(self) -> list[int]: ...

class CondenseResult:
    @property
    def id(self) -> list[int]: ...
    @property
    def time1(self) -> list[float]: ...
    @property
    def time2(self) -> list[float]: ...
    @property
    def status(self) -> list[int]: ...
    @property
    def row_map(self) -> list[list[int]]: ...

class TcutResult:
    @property
    def codes(self) -> list[int]: ...
    @property
    def levels(self) -> list[str]: ...
    @property
    def breaks(self) -> list[float]: ...
    @property
    def counts(self) -> list[int]: ...

class TimelineResult:
    @property
    def id(self) -> list[int]: ...
    @property
    def states(self) -> list[list[int]]: ...
    @property
    def time_points(self) -> list[float]: ...

class IntervalResult:
    @property
    def id(self) -> list[int]: ...
    @property
    def time1(self) -> list[float]: ...
    @property
    def time2(self) -> list[float]: ...
    @property
    def status(self) -> list[int]: ...

class SurvivalData:
    time: list[float]
    status: list[int]
    def __init__(self, time: list[float], status: list[int]) -> None: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...

class CovariateMatrix:
    values: list[float]
    n_obs: int
    n_vars: int
    def __init__(self, values: list[float], n_obs: int, n_vars: int) -> None: ...
    def __len__(self) -> int: ...
    def shape(self) -> tuple[int, int]: ...

class Weights:
    values: list[float]
    def __init__(self, values: list[float]) -> None: ...
    @staticmethod
    def unit(n_obs: int) -> Weights: ...
    def __len__(self) -> int: ...

class CountingProcessData:
    start: list[float]
    stop: list[float]
    event: list[int]
    def __init__(self, start: list[float], stop: list[float], event: list[int]) -> None: ...
    def __len__(self) -> int: ...

class CoxRegressionInput:
    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    def __init__(
        self,
        covariates: CovariateMatrix,
        survival: SurvivalData,
        weights: Weights | None = None,
        offset: list[float] | None = None,
    ) -> None: ...

class NaturalSplineKnot:
    @property
    def knots(self) -> list[float]: ...
    @property
    def boundary_knots(self) -> tuple[float, float]: ...
    @property
    def intercept(self) -> bool: ...
    @property
    def df(self) -> int: ...
    def __init__(
        self,
        knots: list[float] | None = None,
        boundary_knots: tuple[float, float] | None = None,
        df: int | None = None,
        intercept: bool | None = None,
    ) -> None: ...
    def basis(self, x: list[float]) -> SplineBasisResult: ...
    def predict(self, x: list[float], coef: list[float]) -> list[float]: ...

class SplineBasisResult:
    @property
    def basis(self) -> list[float]: ...
    @property
    def n_rows(self) -> int: ...
    @property
    def n_cols(self) -> int: ...
    @property
    def knots(self) -> list[float]: ...
    @property
    def boundary_knots(self) -> tuple[float, float]: ...

class PSpline:
    @property
    def coefficients(self) -> list[float] | None: ...
    @property
    def fitted(self) -> bool: ...
    @property
    def df(self) -> int: ...
    @property
    def eps(self) -> float: ...
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

class LinkFunctionParams:
    def __init__(self, edge: float) -> None: ...
    def blogit(self, input: float) -> float: ...
    def bprobit(self, input: float) -> float: ...
    def bcloglog(self, input: float) -> float: ...
    def blog(self, input: float) -> float: ...

class CoxMartInput:
    @property
    def n_obs(self) -> int: ...
    def __init__(
        self,
        survival: SurvivalData,
        score: list[float],
        weights: Weights | None = None,
        strata: list[int] | None = None,
    ) -> None: ...

class AndersenGillInput:
    @property
    def n_obs(self) -> int: ...
    def __init__(
        self,
        counting: CountingProcessData,
        score: list[float],
        weights: Weights | None = None,
        strata: list[int] | None = None,
    ) -> None: ...

class GradientBoostSurvivalConfig:
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    subsample: float
    max_features: int | None
    seed: int | None
    loss: GBSurvLoss
    dropout_rate: float

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        max_features: int | None = None,
        seed: int | None = None,
        loss: GBSurvLoss | None = None,
        dropout_rate: float = 0.0,
    ) -> None: ...

class GradientBoostSurvival:
    @property
    def n_estimators(self) -> int: ...
    @property
    def learning_rate(self) -> float: ...
    @property
    def unique_times(self) -> list[float]: ...
    @property
    def feature_importance(self) -> list[float]: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def train_loss(self) -> list[float]: ...
    @staticmethod
    def fit(
        x: list[float],
        n: int,
        p: int,
        time: list[float],
        status: list[int],
        config: GradientBoostSurvivalConfig,
    ) -> GradientBoostSurvival: ...
    def predict_risk(self, x_new: list[float], n_new: int) -> list[float]: ...
    def predict_survival(self, x_new: list[float], n_new: int) -> list[list[float]]: ...
    def predict_cumulative_hazard(self, x_new: list[float], n_new: int) -> list[list[float]]: ...
    def predict_survival_time(
        self, x_new: list[float], n_new: int, percentile: float = 0.5
    ) -> list[float | None]: ...
    def predict_median_survival_time(
        self, x_new: list[float], n_new: int
    ) -> list[float | None]: ...

class SurvivalForestConfig:
    n_trees: int
    max_depth: int | None
    min_node_size: int
    mtry: int | None
    sample_fraction: float
    seed: int | None
    oob_error: bool
    split_rule: SplitRule
    n_random_splits: int

    def __init__(
        self,
        n_trees: int = 500,
        max_depth: int | None = None,
        min_node_size: int = 15,
        mtry: int | None = None,
        sample_fraction: float = 0.632,
        seed: int | None = None,
        oob_error: bool = True,
        split_rule: SplitRule | None = None,
        n_random_splits: int = 10,
    ) -> None: ...

class SurvivalForestInput:
    x: list[float]
    n_obs: int
    n_vars: int
    time: list[float]
    status: list[int]

    def __init__(
        self,
        x: list[float],
        n_obs: int,
        n_vars: int,
        time: list[float],
        status: list[int],
    ) -> None: ...

class SurvivalForest:
    @property
    def n_trees(self) -> int: ...
    @property
    def unique_times(self) -> list[float]: ...
    @property
    def variable_importance(self) -> list[float]: ...
    @property
    def oob_error(self) -> float | None: ...
    @staticmethod
    def fit_typed(
        input: SurvivalForestInput,
        config: SurvivalForestConfig,
    ) -> SurvivalForest: ...
    @staticmethod
    def fit(
        x: list[float],
        n: int,
        p: int,
        time: list[float],
        status: list[int],
        config: SurvivalForestConfig,
    ) -> SurvivalForest: ...
    def predict_risk(self, x_new: list[float], n_new: int) -> list[float]: ...
    def predict_survival(self, x_new: list[float], n_new: int) -> list[list[float]]: ...
    def predict_cumulative_hazard(self, x_new: list[float], n_new: int) -> list[list[float]]: ...
    def predict_survival_time(
        self, x_new: list[float], n_new: int, percentile: float = 0.5
    ) -> list[float | None]: ...
    def predict_median_survival_time(
        self, x_new: list[float], n_new: int
    ) -> list[float | None]: ...

class DeepSurvConfig:
    hidden_layers: list[int]
    activation: Activation
    dropout_rate: float
    learning_rate: float
    batch_size: int
    n_epochs: int
    l2_reg: float
    seed: int | None
    early_stopping_patience: int | None
    validation_fraction: float

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: Activation | None = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_epochs: int = 100,
        l2_reg: float = 0.0001,
        seed: int | None = None,
        early_stopping_patience: int | None = None,
        validation_fraction: float = 0.1,
    ) -> None: ...

class DeepSurv:
    @property
    def n_features(self) -> int: ...
    @property
    def hidden_layers(self) -> list[int]: ...
    @property
    def unique_times(self) -> list[float]: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def train_loss(self) -> list[float]: ...
    @property
    def val_loss(self) -> list[float]: ...
    @staticmethod
    def fit(
        x: list[float],
        n: int,
        p: int,
        time: list[float],
        status: list[int],
        config: DeepSurvConfig,
    ) -> DeepSurv: ...
    def predict_risk(self, x_new: list[float], n_new: int) -> list[float]: ...
    def predict_survival(self, x_new: list[float], n_new: int) -> list[list[float]]: ...
    def predict_cumulative_hazard(self, x_new: list[float], n_new: int) -> list[list[float]]: ...
    def predict_survival_time(
        self, x_new: list[float], n_new: int, percentile: float = 0.5
    ) -> list[float | None]: ...
    def predict_median_survival_time(
        self, x_new: list[float], n_new: int
    ) -> list[float | None]: ...

class CchMethod:
    Prentice: CchMethod
    SelfPrentice: CchMethod
    LinYing: CchMethod
    IBorgan: CchMethod
    IIBorgan: CchMethod

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

class CohortData:
    @staticmethod
    def new() -> CohortData: ...
    def add_subject(self, subject: Subject) -> None: ...
    def get_subject(self, index: int) -> Subject: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def fit(self, method: CchMethod, max_iter: int = 100) -> CoxPHModel: ...

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
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...

class ConditionalLogisticRegression:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def max_iter(self) -> int: ...
    @max_iter.setter
    def max_iter(self, value: int) -> None: ...
    @property
    def tol(self) -> float: ...
    @tol.setter
    def tol(self, value: float) -> None: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    def __init__(
        self,
        data: ClogitDataSet,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...
    def fit(self) -> None: ...
    def predict(self, covariates: list[float]) -> float: ...
    def odds_ratios(self) -> list[float]: ...

class CoxPHFit:
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def means(self) -> list[float]: ...
    @property
    def score_vector(self) -> list[float]: ...
    @property
    def information_matrix(self) -> list[list[float]]: ...
    @property
    def log_likelihood(self) -> list[float]: ...
    @property
    def score_test(self) -> float: ...
    @property
    def convergence_flag(self) -> int: ...
    @property
    def iterations(self) -> int: ...
    @property
    def risk_scores(self) -> list[float]: ...
    @property
    def event_times(self) -> list[float]: ...
    @property
    def status(self) -> list[int]: ...
    @property
    def linear_predictors(self) -> list[float]: ...
    @property
    def entry_times(self) -> list[float] | None: ...
    @property
    def weights(self) -> list[float]: ...
    def predict(self, covariates: list[list[float]]) -> list[float]: ...
    def hazard_ratios(self) -> list[float]: ...
    def basehaz(self, centered: bool = True) -> tuple[list[float], list[float]]: ...
    def basehaz_with_strata(
        self,
        centered: bool = True,
    ) -> tuple[list[float], list[float], list[int]]: ...
    def survival_curve(
        self,
        covariates: list[list[float]] | None = None,
        centered: bool = True,
    ) -> tuple[list[float], list[list[float]]]: ...
    def survival_curve_with_strata(
        self,
        covariates: list[list[float]],
        strata: list[int],
        centered: bool = True,
    ) -> tuple[list[float], list[list[float]]]: ...
    def expected_events(self) -> list[float]: ...
    def martingale_residuals(self) -> list[float]: ...
    def deviance_residuals(self) -> list[float]: ...
    def score_residuals(self) -> list[list[float]]: ...
    def dfbeta(self) -> list[list[float]]: ...
    def dfbetas(self) -> list[list[float]]: ...
    def schoenfeld_residuals(self) -> list[list[float]]: ...
    def scaled_schoenfeld_residuals(self) -> list[list[float]]: ...
    def partial_residuals(self) -> list[list[float]]: ...
    @property
    def covariates(self) -> list[list[float]]: ...
    @property
    def strata(self) -> list[int]: ...
    @property
    def method(self) -> str: ...
    @property
    def nocenter(self) -> list[float]: ...

class CoxphDetailRow:
    @property
    def stratum(self) -> int: ...
    @property
    def time(self) -> float: ...
    @property
    def n_risk(self) -> int: ...
    @property
    def n_event(self) -> int: ...
    @property
    def n_censor(self) -> int: ...
    @property
    def hazard(self) -> float: ...
    @property
    def cumhaz(self) -> float: ...
    @property
    def varhaz(self) -> float: ...
    @property
    def wtrisk(self) -> float: ...
    @property
    def n_event_weight(self) -> float: ...
    @property
    def score(self) -> list[float]: ...
    @property
    def schoenfeld(self) -> list[float] | None: ...
    @property
    def means(self) -> list[float]: ...
    @property
    def imat(self) -> list[list[float]]: ...

class CoxphDetail:
    @property
    def rows(self) -> list[CoxphDetailRow]: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_observations(self) -> int: ...
    @property
    def n_covariates(self) -> int: ...
    def times(self) -> list[float]: ...
    def hazards(self) -> list[float]: ...
    def cumulative_hazards(self) -> list[float]: ...
    def n_risk_at_times(self) -> list[int]: ...
    def scores(self) -> list[list[float]]: ...
    def means(self) -> list[list[float]]: ...
    def information_matrices(self) -> list[list[list[float]]]: ...
    def variance_hazards(self) -> list[float]: ...
    def weighted_risk(self) -> list[float]: ...
    def schoenfeld_residuals(self) -> list[list[float]]: ...

class FineGrayOutput:
    @property
    def row(self) -> list[int]: ...
    @property
    def start(self) -> list[float]: ...
    @property
    def end(self) -> list[float]: ...
    @property
    def wt(self) -> list[float]: ...
    @property
    def add(self) -> list[int]: ...

class FineGrayResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def z_scores(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def variance_matrix(self) -> list[list[float]]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def log_likelihood_null(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_competing(self) -> int: ...
    @property
    def n_censored(self) -> int: ...
    @property
    def n_observations(self) -> int: ...
    @property
    def event_type(self) -> int: ...
    @property
    def convergence(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    def __init__(
        self,
        coefficients: list[float],
        std_errors: list[float],
        z_scores: list[float],
        p_values: list[float],
        ci_lower: list[float],
        ci_upper: list[float],
        variance_matrix: list[list[float]],
        log_likelihood: float,
        log_likelihood_null: float,
        n_events: int,
        n_competing: int,
        n_censored: int,
        n_observations: int,
        event_type: int,
        convergence: bool,
        iterations: int,
    ) -> None: ...
    def hazard_ratio(self) -> list[float]: ...
    def summary(self) -> str: ...

class CompetingRisksCIF:
    @property
    def times(self) -> list[float]: ...
    @property
    def cif(self) -> list[float]: ...
    @property
    def variance(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def n_risk(self) -> list[int]: ...
    @property
    def n_events(self) -> list[int]: ...
    @property
    def event_type(self) -> int: ...
    def __init__(
        self,
        times: list[float],
        cif: list[float],
        variance: list[float],
        ci_lower: list[float],
        ci_upper: list[float],
        n_risk: list[int],
        n_events: list[int],
        event_type: int,
    ) -> None: ...

class MultiStateConfig:
    def __init__(
        self,
        n_states: int,
        state_names: list[str] | None = None,
        transition_matrix: list[list[bool]] | None = None,
        absorbing_states: list[int] | None = None,
    ) -> None: ...
    @property
    def n_states(self) -> int: ...
    @n_states.setter
    def n_states(self, value: int) -> None: ...
    @property
    def state_names(self) -> list[str]: ...
    @state_names.setter
    def state_names(self, value: list[str]) -> None: ...
    @property
    def transition_matrix(self) -> list[list[bool]]: ...
    @transition_matrix.setter
    def transition_matrix(self, value: list[list[bool]]) -> None: ...
    @property
    def absorbing_states(self) -> list[int]: ...
    @absorbing_states.setter
    def absorbing_states(self, value: list[int]) -> None: ...

class TransitionIntensityResult:
    def __init__(
        self,
        intensities: dict[str, list[float]],
        cumulative_intensities: dict[str, list[float]],
        time_points: list[float],
        variance: dict[str, list[float]],
        n_at_risk: dict[str, list[float]],
        n_transitions: dict[str, list[int]],
    ) -> None: ...
    @property
    def intensities(self) -> dict[str, list[float]]: ...
    @property
    def cumulative_intensities(self) -> dict[str, list[float]]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def variance(self) -> dict[str, list[float]]: ...
    @property
    def n_at_risk(self) -> dict[str, list[float]]: ...
    @property
    def n_transitions(self) -> dict[str, list[int]]: ...

class MultiStateResult:
    def __init__(
        self,
        state_probabilities: list[list[float]],
        time_points: list[float],
        transition_intensities: TransitionIntensityResult,
        restricted_mean_times: list[float],
        sojourn_times: list[float],
        state_occupancy: list[list[float]],
    ) -> None: ...
    @property
    def state_probabilities(self) -> list[list[float]]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def transition_intensities(self) -> TransitionIntensityResult: ...
    @property
    def restricted_mean_times(self) -> list[float]: ...
    @property
    def sojourn_times(self) -> list[float]: ...
    @property
    def state_occupancy(self) -> list[list[float]]: ...

class MarkovMSMResult:
    def __init__(
        self,
        transition_matrix: list[list[float]],
        generator_matrix: list[list[float]],
        stationary_distribution: list[float],
        time_points: list[float],
        state_probabilities: list[list[float]],
        log_likelihood: float,
    ) -> None: ...
    @property
    def transition_matrix(self) -> list[list[float]]: ...
    @property
    def generator_matrix(self) -> list[list[float]]: ...
    @property
    def stationary_distribution(self) -> list[float]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def state_probabilities(self) -> list[list[float]]: ...
    @property
    def log_likelihood(self) -> float: ...

class IllnessDeathType:
    Progressive: IllnessDeathType
    Reversible: IllnessDeathType
    MarkovProgressive: IllnessDeathType
    SemiMarkovProgressive: IllnessDeathType

class IllnessDeathConfig:
    def __init__(
        self,
        model_type: IllnessDeathType = IllnessDeathType.Progressive,
        state_names: list[str] | None = None,
        clock_type: str = "forward",
        max_iter: int = 100,
        tol: float = 1e-6,
        n_bootstrap: int = 0,
    ) -> None: ...
    @property
    def model_type(self) -> IllnessDeathType: ...
    @model_type.setter
    def model_type(self, value: IllnessDeathType) -> None: ...
    @property
    def state_names(self) -> list[str]: ...
    @state_names.setter
    def state_names(self, value: list[str]) -> None: ...
    @property
    def clock_type(self) -> str: ...
    @clock_type.setter
    def clock_type(self, value: str) -> None: ...
    @property
    def max_iter(self) -> int: ...
    @max_iter.setter
    def max_iter(self, value: int) -> None: ...
    @property
    def tol(self) -> float: ...
    @tol.setter
    def tol(self, value: float) -> None: ...
    @property
    def n_bootstrap(self) -> int: ...
    @n_bootstrap.setter
    def n_bootstrap(self, value: int) -> None: ...

class TransitionHazard:
    @property
    def from_state(self) -> str: ...
    @property
    def to_state(self) -> str: ...
    @property
    def coefficient(self) -> float: ...
    @property
    def se(self) -> float: ...
    @property
    def hazard_ratio(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def baseline_times(self) -> list[float]: ...

class IllnessDeathResult:
    @property
    def transition_hazards(self) -> list[TransitionHazard]: ...
    @property
    def state_occupation_probs(self) -> list[list[float]]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def cumulative_incidence(self) -> list[list[float]]: ...
    @property
    def sojourn_times(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_transitions(self) -> list[int]: ...
    @property
    def model_type(self) -> IllnessDeathType: ...
    def get_survival_probability(self, time: float) -> float: ...
    def get_illness_probability(self, time: float) -> float: ...
    def get_death_probability(self, time: float) -> float: ...

class IllnessDeathPrediction:
    @property
    def state_probs(self) -> list[list[float]]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def survival_prob(self) -> list[float]: ...
    @property
    def illness_free_survival(self) -> list[float]: ...
    @property
    def death_prob(self) -> list[float]: ...

class SojournDistribution:
    Exponential: SojournDistribution
    Weibull: SojournDistribution
    LogNormal: SojournDistribution
    Gamma: SojournDistribution
    GeneralizedGamma: SojournDistribution

class SemiMarkovConfig:
    def __init__(
        self,
        n_states: int,
        state_names: list[str] | None = None,
        sojourn_distributions: list[SojournDistribution] | None = None,
        absorbing_states: list[int] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...
    @property
    def n_states(self) -> int: ...
    @n_states.setter
    def n_states(self, value: int) -> None: ...
    @property
    def state_names(self) -> list[str]: ...
    @state_names.setter
    def state_names(self, value: list[str]) -> None: ...
    @property
    def sojourn_distributions(self) -> list[SojournDistribution]: ...
    @sojourn_distributions.setter
    def sojourn_distributions(self, value: list[SojournDistribution]) -> None: ...
    @property
    def absorbing_states(self) -> list[int]: ...
    @absorbing_states.setter
    def absorbing_states(self, value: list[int]) -> None: ...
    @property
    def max_iter(self) -> int: ...
    @max_iter.setter
    def max_iter(self, value: int) -> None: ...
    @property
    def tol(self) -> float: ...
    @tol.setter
    def tol(self, value: float) -> None: ...

class SojournTimeParams:
    @property
    def distribution(self) -> SojournDistribution: ...
    @property
    def shape(self) -> float: ...
    @property
    def scale(self) -> float: ...
    @property
    def location(self) -> float: ...
    @property
    def mean(self) -> float: ...
    @property
    def variance(self) -> float: ...
    @property
    def median(self) -> float: ...

class SemiMarkovResult:
    @property
    def transition_probs(self) -> dict[str, float]: ...
    @property
    def sojourn_params(self) -> list[SojournTimeParams]: ...
    @property
    def state_occupation_probs(self) -> list[list[float]]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def mean_sojourn_times(self) -> list[float]: ...
    @property
    def n_transitions(self) -> dict[str, int]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    def get_transition_prob(self, from_state: int, to_state: int) -> float: ...
    def predict_state_at_time(self, time: float) -> list[float]: ...

class SemiMarkovPrediction:
    @property
    def state_probs(self) -> list[list[float]]: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def expected_sojourn(self) -> list[float]: ...
    @property
    def transition_hazards(self) -> dict[str, list[float]]: ...

class StateFigData:
    @property
    def states(self) -> list[str]: ...
    @property
    def positions(self) -> list[tuple[float, float]]: ...
    @property
    def edges(self) -> list[tuple[int, int, int]]: ...
    @property
    def box_sizes(self) -> list[tuple[float, float]]: ...
    @property
    def layout(self) -> list[int]: ...

class DCalibrationResult:
    def __init__(
        self,
        statistic: float,
        p_value: float,
        degrees_of_freedom: int,
        n_bins: int,
        observed_counts: list[int],
        expected_counts: list[float],
        bin_edges: list[float],
        n_events: int,
        is_calibrated: bool,
    ) -> None: ...
    @property
    def statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def degrees_of_freedom(self) -> int: ...
    @property
    def n_bins(self) -> int: ...
    @property
    def observed_counts(self) -> list[int]: ...
    @property
    def expected_counts(self) -> list[float]: ...
    @property
    def bin_edges(self) -> list[float]: ...
    @property
    def n_events(self) -> int: ...
    @property
    def is_calibrated(self) -> bool: ...

class OneCalibrationResult:
    def __init__(
        self,
        time_point: float,
        statistic: float,
        p_value: float,
        degrees_of_freedom: int,
        n_groups: int,
        predicted_survival: list[float],
        observed_survival: list[float],
        n_per_group: list[int],
        n_events_per_group: list[int],
        is_calibrated: bool,
    ) -> None: ...
    @property
    def time_point(self) -> float: ...
    @property
    def statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def degrees_of_freedom(self) -> int: ...
    @property
    def n_groups(self) -> int: ...
    @property
    def predicted_survival(self) -> list[float]: ...
    @property
    def observed_survival(self) -> list[float]: ...
    @property
    def n_per_group(self) -> list[int]: ...
    @property
    def n_events_per_group(self) -> list[int]: ...
    @property
    def is_calibrated(self) -> bool: ...

class CalibrationPlotData:
    def __init__(
        self,
        predicted: list[float],
        observed: list[float],
        n_per_group: list[int],
        ci_lower: list[float],
        ci_upper: list[float],
        ici: float,
        e50: float,
        e90: float,
        emax: float,
    ) -> None: ...
    @property
    def predicted(self) -> list[float]: ...
    @property
    def observed(self) -> list[float]: ...
    @property
    def n_per_group(self) -> list[int]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def ici(self) -> float: ...
    @property
    def e50(self) -> float: ...
    @property
    def e90(self) -> float: ...
    @property
    def emax(self) -> float: ...

class BrierCalibrationResult:
    def __init__(
        self,
        time_point: float,
        brier_score: float,
        calibration_slope: float,
        calibration_intercept: float,
        ici: float,
        e50: float,
        e90: float,
        emax: float,
        predicted: list[float],
        observed: list[float],
        ci_lower: list[float],
        ci_upper: list[float],
        n_per_group: list[int],
    ) -> None: ...
    @property
    def time_point(self) -> float: ...
    @property
    def brier_score(self) -> float: ...
    @property
    def calibration_slope(self) -> float: ...
    @property
    def calibration_intercept(self) -> float: ...
    @property
    def ici(self) -> float: ...
    @property
    def e50(self) -> float: ...
    @property
    def e90(self) -> float: ...
    @property
    def emax(self) -> float: ...
    @property
    def predicted(self) -> list[float]: ...
    @property
    def observed(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def n_per_group(self) -> list[int]: ...

class SmoothedCalibrationCurve:
    def __init__(
        self,
        predicted_grid: list[float],
        smoothed_observed: list[float],
        ci_lower: list[float],
        ci_upper: list[float],
        bandwidth: float,
    ) -> None: ...
    @property
    def predicted_grid(self) -> list[float]: ...
    @property
    def smoothed_observed(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def bandwidth(self) -> float: ...

class KaplanMeierPlotData:
    @property
    def time_points(self) -> list[float]: ...
    @property
    def survival_prob(self) -> list[float]: ...
    @property
    def lower_ci(self) -> list[float]: ...
    @property
    def upper_ci(self) -> list[float]: ...
    @property
    def at_risk(self) -> list[int]: ...
    @property
    def n_events(self) -> list[int]: ...
    @property
    def n_censored(self) -> list[int]: ...
    @property
    def group_name(self) -> str | None: ...
    def to_step_data(self) -> tuple[list[float], list[float]]: ...

class ForestPlotData:
    @property
    def variable_names(self) -> list[str]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def lower_ci(self) -> list[float]: ...
    @property
    def upper_ci(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def weights(self) -> list[float] | None: ...
    def significant_at(self, alpha: float) -> list[bool]: ...

class CalibrationCurveData:
    @property
    def predicted_prob(self) -> list[float]: ...
    @property
    def observed_prob(self) -> list[float]: ...
    @property
    def n_per_bin(self) -> list[int]: ...
    @property
    def bin_boundaries(self) -> list[float]: ...
    @property
    def hosmer_lemeshow_stat(self) -> float: ...
    @property
    def hosmer_lemeshow_p(self) -> float: ...

class SurvivalReport:
    @property
    def title(self) -> str: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_events(self) -> int: ...
    @property
    def median_survival(self) -> float | None: ...
    @property
    def median_ci(self) -> tuple[float, float] | None: ...
    @property
    def survival_rates(self) -> list[tuple[float, float, float, float]]: ...
    @property
    def rmst(self) -> float | None: ...
    @property
    def hazard_ratio(self) -> float | None: ...
    @property
    def hazard_ratio_ci(self) -> tuple[float, float] | None: ...
    @property
    def logrank_p(self) -> float | None: ...
    def to_markdown(self) -> str: ...
    def to_latex(self) -> str: ...

class ROCPlotData:
    @property
    def fpr(self) -> list[float]: ...
    @property
    def tpr(self) -> list[float]: ...
    @property
    def thresholds(self) -> list[float]: ...
    @property
    def auc(self) -> float: ...
    def optimal_threshold(self, method: str) -> float: ...

class DecisionCurveResult:
    @property
    def thresholds(self) -> list[float]: ...
    @property
    def net_benefit(self) -> list[float]: ...
    @property
    def net_benefit_all(self) -> list[float]: ...
    @property
    def net_benefit_none(self) -> list[float]: ...
    @property
    def interventions_avoided(self) -> list[float]: ...
    def optimal_threshold(self) -> float: ...
    def area_under_curve(self) -> float: ...

class ClinicalUtilityResult:
    @property
    def threshold(self) -> float: ...
    @property
    def sensitivity(self) -> float: ...
    @property
    def specificity(self) -> float: ...
    @property
    def ppv(self) -> float: ...
    @property
    def npv(self) -> float: ...
    @property
    def nnt(self) -> float: ...
    @property
    def net_benefit(self) -> float: ...

class ModelComparisonResult:
    @property
    def model_names(self) -> list[str]: ...
    @property
    def net_benefit_difference(self) -> list[list[float]]: ...
    @property
    def thresholds(self) -> list[float]: ...
    @property
    def best_model_per_threshold(self) -> list[str]: ...

class RiskStratificationResult:
    def __init__(
        self,
        risk_groups: list[int],
        cutpoints: list[float],
        group_sizes: list[int],
        group_event_rates: list[float],
        group_median_risk: list[float],
    ) -> None: ...
    @property
    def risk_groups(self) -> list[int]: ...
    @property
    def cutpoints(self) -> list[float]: ...
    @property
    def group_sizes(self) -> list[int]: ...
    @property
    def group_event_rates(self) -> list[float]: ...
    @property
    def group_median_risk(self) -> list[float]: ...

class MedianSurvivalResult:
    def __init__(
        self,
        median: float | None,
        ci_lower: float | None,
        ci_upper: float | None,
        quantile: float,
    ) -> None: ...
    @property
    def median(self) -> float | None: ...
    @property
    def ci_lower(self) -> float | None: ...
    @property
    def ci_upper(self) -> float | None: ...
    @property
    def quantile(self) -> float: ...

class CumulativeIncidenceResult:
    def __init__(
        self,
        time: list[float],
        cif: list[list[float]],
        variance: list[list[float]],
        event_types: list[int],
        n_risk: list[int],
    ) -> None: ...
    @property
    def time(self) -> list[float]: ...
    @property
    def cif(self) -> list[list[float]]: ...
    @property
    def variance(self) -> list[list[float]]: ...
    @property
    def event_types(self) -> list[int]: ...
    @property
    def n_risk(self) -> list[int]: ...

class NNTResult:
    def __init__(
        self,
        nnt: float,
        nnt_ci_lower: float,
        nnt_ci_upper: float,
        absolute_risk_reduction: float,
        arr_ci_lower: float,
        arr_ci_upper: float,
        time_horizon: float,
    ) -> None: ...
    @property
    def nnt(self) -> float: ...
    @property
    def nnt_ci_lower(self) -> float: ...
    @property
    def nnt_ci_upper(self) -> float: ...
    @property
    def absolute_risk_reduction(self) -> float: ...
    @property
    def arr_ci_lower(self) -> float: ...
    @property
    def arr_ci_upper(self) -> float: ...
    @property
    def time_horizon(self) -> float: ...

class MultiTimeCalibrationResult:
    def __init__(
        self,
        time_points: list[float],
        brier_scores: list[float],
        integrated_brier: float,
        calibration_slopes: list[float],
        calibration_intercepts: list[float],
        ici_values: list[float],
        mean_ici: float,
        mean_slope: float,
    ) -> None: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def brier_scores(self) -> list[float]: ...
    @property
    def integrated_brier(self) -> float: ...
    @property
    def calibration_slopes(self) -> list[float]: ...
    @property
    def calibration_intercepts(self) -> list[float]: ...
    @property
    def ici_values(self) -> list[float]: ...
    @property
    def mean_ici(self) -> float: ...
    @property
    def mean_slope(self) -> float: ...

class ImputationMethod:
    PMM: ImputationMethod
    Regression: ImputationMethod
    MICE: ImputationMethod
    KNN: ImputationMethod
    def __init__(self, name: str) -> None: ...

class MultipleImputationResult:
    @property
    def pooled_coefficients(self) -> list[float]: ...
    @property
    def pooled_se(self) -> list[float]: ...
    @property
    def pooled_ci_lower(self) -> list[float]: ...
    @property
    def pooled_ci_upper(self) -> list[float]: ...
    @property
    def within_variance(self) -> list[float]: ...
    @property
    def between_variance(self) -> list[float]: ...
    @property
    def total_variance(self) -> list[float]: ...
    @property
    def fraction_missing_info(self) -> list[float]: ...
    @property
    def relative_efficiency(self) -> list[float]: ...
    @property
    def n_imputations(self) -> int: ...

class PatternMixtureResult:
    @property
    def pattern_coefficients(self) -> list[list[float]]: ...
    @property
    def pattern_se(self) -> list[list[float]]: ...
    @property
    def pattern_weights(self) -> list[float]: ...
    @property
    def averaged_coefficients(self) -> list[float]: ...
    @property
    def averaged_se(self) -> list[float]: ...
    @property
    def averaged_ci_lower(self) -> list[float]: ...
    @property
    def averaged_ci_upper(self) -> list[float]: ...
    @property
    def n_patterns(self) -> int: ...
    @property
    def pattern_sizes(self) -> list[int]: ...

class SensitivityAnalysisType:
    TiltingModel: SensitivityAnalysisType
    SelectionModel: SensitivityAnalysisType
    DeltaAdjustment: SensitivityAnalysisType
    def __init__(self, name: str) -> None: ...

class GapTimeResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_ci_lower(self) -> list[float]: ...
    @property
    def hr_ci_upper(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def baseline_times(self) -> list[float]: ...

class FrailtyDistribution:
    Gamma: FrailtyDistribution
    LogNormal: FrailtyDistribution
    Positive_Stable: FrailtyDistribution
    def __init__(self, name: str) -> None: ...

class JointFrailtyResult:
    @property
    def recurrent_coef(self) -> list[float]: ...
    @property
    def recurrent_se(self) -> list[float]: ...
    @property
    def terminal_coef(self) -> list[float]: ...
    @property
    def terminal_se(self) -> list[float]: ...
    @property
    def frailty_variance(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def frailty_values(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def n_recurrent_events(self) -> int: ...
    @property
    def n_terminal_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...

class PWPTimescale:
    Gap: PWPTimescale
    Total: PWPTimescale
    def __init__(self, name: str) -> None: ...

class PWPConfig:
    def __init__(
        self,
        timescale: PWPTimescale = PWPTimescale.Gap,
        max_iter: int = 100,
        tol: float = 1e-6,
        stratify_by_event: bool = True,
        robust_variance: bool = True,
    ) -> None: ...
    @property
    def timescale(self) -> PWPTimescale: ...
    @timescale.setter
    def timescale(self, value: PWPTimescale) -> None: ...
    @property
    def max_iter(self) -> int: ...
    @max_iter.setter
    def max_iter(self, value: int) -> None: ...
    @property
    def tol(self) -> float: ...
    @tol.setter
    def tol(self, value: float) -> None: ...
    @property
    def stratify_by_event(self) -> bool: ...
    @stratify_by_event.setter
    def stratify_by_event(self, value: bool) -> None: ...
    @property
    def robust_variance(self) -> bool: ...
    @robust_variance.setter
    def robust_variance(self, value: bool) -> None: ...

class PWPResult:
    @property
    def coef(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def robust_std_errors(self) -> list[float]: ...
    @property
    def z_scores(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_lower(self) -> list[float]: ...
    @property
    def hr_upper(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def event_specific_coef(self) -> list[list[float]]: ...
    @property
    def baseline_cumhaz(self) -> list[float]: ...

class AndersonGillResult:
    @property
    def coef(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def robust_std_errors(self) -> list[float]: ...
    @property
    def z_scores(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_lower(self) -> list[float]: ...
    @property
    def hr_upper(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def mean_event_rate(self) -> float: ...

class WLWConfig:
    max_iter: int
    tol: float
    robust_variance: bool
    common_baseline: bool
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        robust_variance: bool = True,
        common_baseline: bool = False,
    ) -> None: ...

class WLWResult:
    @property
    def coef(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def robust_std_errors(self) -> list[float]: ...
    @property
    def z_scores(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_lower(self) -> list[float]: ...
    @property
    def hr_upper(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_strata(self) -> int: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def stratum_coef(self) -> list[list[float]]: ...
    @property
    def global_test_stat(self) -> float: ...
    @property
    def global_test_pvalue(self) -> float: ...

class NegativeBinomialFrailtyConfig:
    max_iter: int
    tol: float
    em_max_iter: int
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        em_max_iter: int = 50,
    ) -> None: ...

class NegativeBinomialFrailtyResult:
    @property
    def coef(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def z_scores(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def rate_ratios(self) -> list[float]: ...
    @property
    def rr_lower(self) -> list[float]: ...
    @property
    def rr_upper(self) -> list[float]: ...
    @property
    def theta(self) -> float: ...
    @property
    def theta_se(self) -> float: ...
    @property
    def frailty_variance(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def frailty_estimates(self) -> list[float]: ...

class MarginalMethod:
    AndersenGill: MarginalMethod
    WeiLinWeissfeld: MarginalMethod
    def __init__(self, name: str) -> None: ...

class MarginalModelResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def robust_se(self) -> list[float]: ...
    @property
    def naive_se(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_ci_lower(self) -> list[float]: ...
    @property
    def hr_ci_upper(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def score_test(self) -> float: ...
    @property
    def wald_test(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_subjects(self) -> int: ...
    @property
    def mean_events_per_subject(self) -> float: ...

class CoxPHModel:
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def event_times(self) -> list[float]: ...
    @event_times.setter
    def event_times(self, value: list[float]) -> None: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def risk_scores(self) -> list[float]: ...
    @property
    def censoring(self) -> list[int]: ...
    @censoring.setter
    def censoring(self, value: list[int]) -> None: ...
    @staticmethod
    def new_with_data(
        covariates: list[list[float]],
        event_times: list[float],
        censoring: list[int],
    ) -> CoxPHModel: ...
    def add_subject(self, subject: Subject) -> None: ...
    def fit(self, n_iters: int = 20) -> None: ...
    def predict(self, covariates: list[list[float]]) -> list[float]: ...
    def predict_survival(self, time: float) -> float: ...
    def predicted_survival_time(
        self, covariates: list[list[float]], percentile: float = 0.5
    ) -> list[float | None]: ...
    def survival_curve(
        self, covariates: list[list[float]], time_points: list[float] | None
    ) -> tuple[list[float], list[list[float]]]: ...
    def cumulative_hazard(
        self, covariates: list[list[float]]
    ) -> tuple[list[float], list[list[float]]]: ...
    def hazard_ratios(self) -> list[float]: ...
    def hazard_ratios_with_ci(
        self, confidence_level: float = 0.95
    ) -> tuple[list[float], list[float], list[float]]: ...
    def martingale_residuals(self) -> list[float]: ...
    def deviance_residuals(self) -> list[float]: ...
    def dfbeta(self) -> list[list[float]]: ...
    def brier_score(self) -> float: ...
    def restricted_mean_survival_time(
        self, covariates: list[list[float]], tau: float
    ) -> list[float]: ...
    def summary(self) -> str: ...
    def log_likelihood(self) -> float: ...
    def n_observations(self) -> int: ...
    def n_events(self) -> int: ...
    def std_errors(self) -> list[float]: ...
    def vcov(self) -> list[list[float]]: ...
    def aic(self) -> float: ...
    def bic(self) -> float: ...
    def calculate_baseline_hazard(self) -> None: ...
    def compute_standard_errors(self) -> list[float]: ...
    def compute_fisher_information(self) -> list[list[float]]: ...

class SurvFitKMOutput:
    @property
    def time(self) -> list[float]: ...
    @property
    def n_risk(self) -> list[float]: ...
    @property
    def n_event(self) -> list[float]: ...
    @property
    def n_censor(self) -> list[float]: ...
    @property
    def estimate(self) -> list[float]: ...
    @property
    def std_err(self) -> list[float]: ...
    @property
    def cumhaz(self) -> list[float]: ...
    @property
    def std_chaz(self) -> list[float]: ...
    @property
    def cumulative_hazard(self) -> list[float]: ...
    @property
    def cumulative_hazard_std_err(self) -> list[float]: ...
    @property
    def conf_lower(self) -> list[float]: ...
    @property
    def conf_upper(self) -> list[float]: ...

class SurvFitAJ:
    @property
    def n_risk(self) -> list[list[float]]: ...
    @property
    def n_event(self) -> list[list[float]]: ...
    @property
    def n_censor(self) -> list[list[float]]: ...
    @property
    def pstate(self) -> list[list[float]]: ...
    @property
    def cumhaz(self) -> list[list[float]]: ...
    @property
    def std_err(self) -> list[list[float]] | None: ...
    @property
    def std_chaz(self) -> list[list[float]] | None: ...
    @property
    def std_auc(self) -> list[list[float]] | None: ...
    @property
    def influence(self) -> list[list[float]] | None: ...
    @property
    def n_enter(self) -> list[list[float]] | None: ...
    @property
    def n_transition(self) -> list[list[float]]: ...

class VarianceEstimator:
    Greenwood: VarianceEstimator
    Aalen: VarianceEstimator
    Bootstrap: VarianceEstimator
    def __init__(self, name: str) -> None: ...

class TransitionType:
    Standard: TransitionType
    MarkovIllnessDeath: TransitionType
    Progressive: TransitionType
    Custom: TransitionType
    def __init__(self, name: str) -> None: ...

class AalenJohansenExtendedConfig:
    variance_estimator: VarianceEstimator
    transition_type: TransitionType
    n_bootstrap: int
    confidence_level: float
    compute_sojourn: bool
    seed: int | None
    def __init__(
        self,
        variance_estimator: VarianceEstimator = ...,
        transition_type: TransitionType = ...,
        n_bootstrap: int = 200,
        confidence_level: float = 0.95,
        compute_sojourn: bool = True,
        seed: int | None = None,
    ) -> None: ...

class TransitionMatrix:
    @property
    def time(self) -> float: ...
    @property
    def matrix(self) -> list[list[float]]: ...
    @property
    def n_at_risk(self) -> list[int]: ...
    @property
    def n_transitions(self) -> list[list[int]]: ...

class AalenJohansenExtendedResult:
    @property
    def time(self) -> list[float]: ...
    @property
    def state_probs(self) -> list[list[list[float]]]: ...
    @property
    def variance(self) -> list[list[list[float]]]: ...
    @property
    def ci_lower(self) -> list[list[list[float]]]: ...
    @property
    def ci_upper(self) -> list[list[list[float]]]: ...
    @property
    def transition_matrices(self) -> list[TransitionMatrix]: ...
    @property
    def cumulative_incidence(self) -> list[list[float]]: ...
    @property
    def expected_sojourn(self) -> list[float] | None: ...
    @property
    def n_states(self) -> int: ...
    @property
    def n_obs(self) -> int: ...
    @property
    def n_events(self) -> int: ...
    def __repr__(self) -> str: ...
    def get_cif(self, to_state: int) -> list[float]: ...
    def get_state_prob(self, from_state: int, to_state: int) -> list[float]: ...
    def interpolate_at(self, query_time: float) -> list[list[float]]: ...

class KaplanMeierConfig:
    reverse: bool
    computation_type: int
    conf_level: float
    conf_type: str

    def __init__(
        self,
        reverse: bool | None = None,
        computation_type: int | None = None,
        conf_level: float | None = None,
        conf_type: str | None = None,
    ) -> None: ...

class SurvfitKMOptions:
    weights: list[float] | None
    entry_times: list[float] | None
    position: list[int] | None
    reverse: bool | None
    computation_type: int | None
    conf_level: float | None
    conf_type: str | None

    def __init__(
        self,
        weights: list[float] | None = None,
        entry_times: list[float] | None = None,
        position: list[int] | None = None,
        reverse: bool | None = None,
        computation_type: int | None = None,
        conf_level: float | None = None,
        conf_type: str | None = None,
    ) -> None: ...
    def with_weights(self, weights: list[float]) -> SurvfitKMOptions: ...
    def with_entry_times(self, entry_times: list[float]) -> SurvfitKMOptions: ...
    def with_position(self, position: list[int]) -> SurvfitKMOptions: ...
    def with_reverse(self, reverse: bool) -> SurvfitKMOptions: ...
    def with_computation_type(self, computation_type: int) -> SurvfitKMOptions: ...
    def with_conf_level(self, conf_level: float) -> SurvfitKMOptions: ...
    def with_conf_type(self, conf_type: str) -> SurvfitKMOptions: ...

class SurvfitMatrixResult:
    @property
    def time(self) -> list[float]: ...
    @property
    def surv(self) -> list[list[float]]: ...
    @property
    def cumhaz(self) -> list[list[float]]: ...
    @property
    def std_err(self) -> list[list[float]] | None: ...
    @property
    def n_risk(self) -> list[float]: ...
    @property
    def n_event(self) -> list[float]: ...
    @property
    def n_states(self) -> int: ...
    def __init__(
        self,
        time: list[float],
        surv: list[list[float]],
        cumhaz: list[list[float]],
        std_err: list[list[float]] | None = None,
        n_risk: list[float] = ...,
        n_event: list[float] = ...,
        n_states: int = 1,
    ) -> None: ...
    def get_surv_at_state(self, state: int) -> list[float]: ...
    def get_cumhaz_at_state(self, state: int) -> list[float]: ...

class PseudoResult:
    @property
    def pseudo(self) -> list[list[float]]: ...
    @property
    def time(self) -> list[float]: ...
    @property
    def type_(self) -> str: ...
    @property
    def n(self) -> int: ...

class GEEConfig:
    correlation_structure: str
    link_function: str
    max_iter: int
    tol: float
    def __init__(
        self,
        correlation_structure: str = "independence",
        link_function: str = "identity",
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...

class GEEResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def z_values(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def confidence_intervals(self) -> list[tuple[float, float]]: ...
    @property
    def qic(self) -> float: ...
    @property
    def n_iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    def __init__(
        self,
        coefficients: list[float],
        std_errors: list[float],
        z_values: list[float],
        p_values: list[float],
        confidence_intervals: list[tuple[float, float]],
        qic: float,
        n_iterations: int,
        converged: bool,
    ) -> None: ...

class AggregateSurvfitResult:
    @property
    def time(self) -> list[float]: ...
    @property
    def surv(self) -> list[float]: ...
    @property
    def std_err(self) -> list[float]: ...
    @property
    def lower(self) -> list[float]: ...
    @property
    def upper(self) -> list[float]: ...
    @property
    def n_curves(self) -> int: ...
    @property
    def weights(self) -> list[float]: ...

class SurvCheckResult:
    @property
    def n_subjects(self) -> int: ...
    @property
    def n_transitions(self) -> int: ...
    @property
    def n_problems(self) -> int: ...
    @property
    def overlap_ids(self) -> list[int]: ...
    @property
    def gap_ids(self) -> list[int]: ...
    @property
    def teleport_ids(self) -> list[int]: ...
    @property
    def invalid_ids(self) -> list[int]: ...
    @property
    def transitions(self) -> dict[str, int]: ...
    @property
    def flags(self) -> list[int]: ...
    @property
    def is_valid(self) -> bool: ...
    @property
    def messages(self) -> list[str]: ...

class NelsonAalenResult:
    @property
    def time(self) -> list[float]: ...
    @property
    def cumulative_hazard(self) -> list[float]: ...
    @property
    def variance(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def n_risk(self) -> list[int]: ...
    @property
    def n_events(self) -> list[int]: ...
    def survival(self) -> list[float]: ...

class StratifiedKMResult:
    @property
    def strata(self) -> list[int]: ...
    @property
    def times(self) -> list[list[float]]: ...
    @property
    def survival(self) -> list[list[float]]: ...
    @property
    def ci_lower(self) -> list[list[float]]: ...
    @property
    def ci_upper(self) -> list[list[float]]: ...
    @property
    def n_risk(self) -> list[list[int]]: ...
    @property
    def n_events(self) -> list[list[int]]: ...

class SurvDiffResult:
    @property
    def observed(self) -> list[float]: ...
    @property
    def expected(self) -> list[float]: ...
    @property
    def variance(self) -> list[list[float]]: ...
    @property
    def chi_squared(self) -> float: ...
    @property
    def degrees_of_freedom(self) -> int: ...

class LogRankResult:
    def __init__(
        self,
        statistic: float,
        p_value: float,
        df: int,
        observed: list[float],
        expected: list[float],
        variance: float,
        weight_type: str,
    ) -> None: ...
    @property
    def statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def df(self) -> int: ...
    @property
    def observed(self) -> list[float]: ...
    @property
    def expected(self) -> list[float]: ...
    @property
    def variance(self) -> float: ...
    @property
    def weight_type(self) -> str: ...

class TrendTestResult:
    @property
    def statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def trend_direction(self) -> str: ...

class SurvObrienResult:
    @property
    def statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def df(self) -> int: ...
    @property
    def scores(self) -> list[float]: ...
    @property
    def score_sum(self) -> float: ...
    @property
    def expected(self) -> float: ...
    @property
    def variance(self) -> float: ...

class SampleSizeResult:
    def __init__(
        self,
        n_total: int,
        n_events: int,
        n_per_group: list[int],
        power: float,
        alpha: float,
        hazard_ratio: float,
        method: str,
    ) -> None: ...
    @property
    def n_total(self) -> int: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_per_group(self) -> list[int]: ...
    @property
    def power(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def hazard_ratio(self) -> float: ...
    @property
    def method(self) -> str: ...

class AccrualResult:
    def __init__(
        self,
        n_total: int,
        accrual_time: float,
        followup_time: float,
        study_duration: float,
        expected_events: float,
    ) -> None: ...
    @property
    def n_total(self) -> int: ...
    @property
    def accrual_time(self) -> float: ...
    @property
    def followup_time(self) -> float: ...
    @property
    def study_duration(self) -> float: ...
    @property
    def expected_events(self) -> float: ...

class RoystonResult:
    @property
    def d(self) -> float: ...
    @property
    def se(self) -> float: ...
    @property
    def r_squared_d(self) -> float: ...
    @property
    def r_squared_ko(self) -> float: ...
    @property
    def z(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def n_events(self) -> int: ...

class YatesResult:
    @property
    def levels(self) -> list[str]: ...
    @property
    def means(self) -> list[float]: ...
    @property
    def se(self) -> list[float]: ...
    @property
    def lower(self) -> list[float]: ...
    @property
    def upper(self) -> list[float]: ...
    @property
    def n(self) -> list[int]: ...
    @property
    def predict_type(self) -> str: ...

class YatesPairwiseResult:
    @property
    def level1(self) -> list[str]: ...
    @property
    def level2(self) -> list[str]: ...
    @property
    def difference(self) -> list[float]: ...
    @property
    def se(self) -> list[float]: ...
    @property
    def z(self) -> list[float]: ...
    @property
    def p_value(self) -> list[float]: ...

class RatetableDateResult:
    @property
    def days(self) -> float: ...
    @property
    def years(self) -> float: ...
    @property
    def origin_year(self) -> int: ...

class SurvExpResult:
    @property
    def time(self) -> list[float]: ...
    @property
    def surv(self) -> list[float]: ...
    @property
    def n_risk(self) -> list[float]: ...
    @property
    def cumhaz(self) -> list[float]: ...
    @property
    def method(self) -> str: ...
    @property
    def n(self) -> int: ...

class PyearsSummary:
    @property
    def total_person_years(self) -> float: ...
    @property
    def total_events(self) -> float: ...
    @property
    def total_expected(self) -> float: ...
    @property
    def n_observations(self) -> float: ...
    @property
    def offtable(self) -> float: ...
    @property
    def observed_rate(self) -> float: ...
    @property
    def expected_rate(self) -> float: ...
    @property
    def smr(self) -> float: ...
    @property
    def sir(self) -> float: ...
    def to_table(self) -> str: ...

class PyearsCell:
    @property
    def person_years(self) -> float: ...
    @property
    def n(self) -> float: ...
    @property
    def events(self) -> float: ...
    @property
    def expected(self) -> float: ...
    @property
    def rate(self) -> float: ...
    @property
    def smr(self) -> float: ...

class ExpectedSurvivalResult:
    @property
    def expected_survival(self) -> list[float]: ...
    @property
    def time(self) -> list[float]: ...
    @property
    def n(self) -> int: ...

class NetSurvivalResult:
    @property
    def time_points(self) -> list[float]: ...
    @property
    def net_survival(self) -> list[float]: ...
    @property
    def net_survival_se(self) -> list[float]: ...
    @property
    def net_survival_lower(self) -> list[float]: ...
    @property
    def net_survival_upper(self) -> list[float]: ...
    @property
    def cumulative_excess_hazard(self) -> list[float]: ...
    @property
    def n_at_risk(self) -> list[int]: ...
    @property
    def n_events(self) -> list[int]: ...
    @property
    def method(self) -> str: ...

class RelativeSurvivalResult:
    @property
    def time_points(self) -> list[float]: ...
    @property
    def observed_survival(self) -> list[float]: ...
    @property
    def expected_survival(self) -> list[float]: ...
    @property
    def relative_survival(self) -> list[float]: ...
    @property
    def relative_survival_se(self) -> list[float]: ...
    @property
    def cumulative_excess_hazard(self) -> list[float]: ...
    @property
    def excess_mortality_rate(self) -> list[float]: ...
    @property
    def n_at_risk(self) -> list[int]: ...
    @property
    def n_events(self) -> list[int]: ...

class ExcessHazardModelResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def excess_hazard_ratio(self) -> list[float]: ...
    @property
    def ehr_ci_lower(self) -> list[float]: ...
    @property
    def ehr_ci_upper(self) -> list[float]: ...
    @property
    def baseline_excess_hazard(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...

class AnovaRow:
    def __init__(
        self,
        model_name: str,
        loglik: float,
        df: int,
        chisq: float | None,
        p_value: float | None,
    ) -> None: ...
    @property
    def model_name(self) -> str: ...
    @property
    def loglik(self) -> float: ...
    @property
    def df(self) -> int: ...
    @property
    def chisq(self) -> float | None: ...
    @property
    def p_value(self) -> float | None: ...

class AnovaCoxphResult:
    def __init__(self, rows: list[AnovaRow], test_type: str) -> None: ...
    @property
    def rows(self) -> list[AnovaRow]: ...
    @property
    def test_type(self) -> str: ...
    def to_table(self) -> str: ...

class RMSTResult:
    def __init__(
        self,
        rmst: float,
        variance: float,
        se: float,
        ci_lower: float,
        ci_upper: float,
        tau: float,
    ) -> None: ...
    @property
    def rmst(self) -> float: ...
    @property
    def variance(self) -> float: ...
    @property
    def se(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...
    @property
    def tau(self) -> float: ...

class RMSTComparisonResult:
    def __init__(
        self,
        rmst_diff: float,
        rmst_ratio: float,
        diff_se: float,
        diff_ci_lower: float,
        diff_ci_upper: float,
        ratio_ci_lower: float,
        ratio_ci_upper: float,
        p_value: float,
        rmst_group1: RMSTResult,
        rmst_group2: RMSTResult,
    ) -> None: ...
    @property
    def rmst_diff(self) -> float: ...
    @property
    def rmst_ratio(self) -> float: ...
    @property
    def diff_se(self) -> float: ...
    @property
    def diff_ci_lower(self) -> float: ...
    @property
    def diff_ci_upper(self) -> float: ...
    @property
    def ratio_ci_lower(self) -> float: ...
    @property
    def ratio_ci_upper(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def rmst_group1(self) -> RMSTResult: ...
    @property
    def rmst_group2(self) -> RMSTResult: ...

class ChangepointInfo:
    def __init__(
        self,
        time: float,
        hazard_before: float,
        hazard_after: float,
        likelihood_ratio: float,
        p_value: float,
    ) -> None: ...
    @property
    def time(self) -> float: ...
    @property
    def hazard_before(self) -> float: ...
    @property
    def hazard_after(self) -> float: ...
    @property
    def likelihood_ratio(self) -> float: ...
    @property
    def p_value(self) -> float: ...

class RMSTOptimalThresholdResult:
    def __init__(
        self,
        optimal_tau: float,
        max_followup: float,
        changepoints: list[ChangepointInfo],
        n_changepoints: int,
        rmst_at_optimal: RMSTResult,
    ) -> None: ...
    @property
    def optimal_tau(self) -> float: ...
    @property
    def max_followup(self) -> float: ...
    @property
    def changepoints(self) -> list[ChangepointInfo]: ...
    @property
    def n_changepoints(self) -> int: ...
    @property
    def rmst_at_optimal(self) -> RMSTResult: ...

class ConditionalSurvivalResult:
    def __init__(
        self,
        given_time: float,
        target_time: float,
        conditional_survival: float,
        ci_lower: float,
        ci_upper: float,
        n_at_risk: int,
    ) -> None: ...
    @property
    def given_time(self) -> float: ...
    @property
    def target_time(self) -> float: ...
    @property
    def conditional_survival(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...
    @property
    def n_at_risk(self) -> int: ...

class HazardRatioResult:
    def __init__(
        self,
        hazard_ratio: float,
        ci_lower: float,
        ci_upper: float,
        se_log_hr: float,
        z_statistic: float,
        p_value: float,
    ) -> None: ...
    @property
    def hazard_ratio(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...
    @property
    def se_log_hr(self) -> float: ...
    @property
    def z_statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...

class SurvivalAtTimeResult:
    def __init__(
        self,
        time: float,
        survival: float,
        ci_lower: float,
        ci_upper: float,
        n_at_risk: int,
        n_events: int,
    ) -> None: ...
    @property
    def time(self) -> float: ...
    @property
    def survival(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...
    @property
    def n_at_risk(self) -> int: ...
    @property
    def n_events(self) -> int: ...

class LifeTableResult:
    def __init__(
        self,
        interval_start: list[float],
        interval_end: list[float],
        n_at_risk: list[float],
        n_deaths: list[float],
        n_censored: list[float],
        n_effective: list[float],
        hazard: list[float],
        survival: list[float],
        se_survival: list[float],
    ) -> None: ...
    @property
    def interval_start(self) -> list[float]: ...
    @property
    def interval_end(self) -> list[float]: ...
    @property
    def n_at_risk(self) -> list[float]: ...
    @property
    def n_deaths(self) -> list[float]: ...
    @property
    def n_censored(self) -> list[float]: ...
    @property
    def n_effective(self) -> list[float]: ...
    @property
    def hazard(self) -> list[float]: ...
    @property
    def survival(self) -> list[float]: ...
    @property
    def se_survival(self) -> list[float]: ...

class CalibrationResult:
    @property
    def observed(self) -> list[float]: ...
    @property
    def predicted(self) -> list[float]: ...
    @property
    def n_groups(self) -> int: ...
    @property
    def hosmer_lemeshow_stat(self) -> float: ...
    @property
    def hosmer_lemeshow_p(self) -> float: ...

class UnoCIndexResult:
    @property
    def c_index(self) -> float: ...
    @property
    def concordant(self) -> float: ...
    @property
    def discordant(self) -> float: ...
    @property
    def tied_risk(self) -> float: ...
    @property
    def comparable_pairs(self) -> float: ...
    @property
    def variance(self) -> float: ...
    @property
    def std_error(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...
    @property
    def tau(self) -> float: ...

class ConcordanceComparisonResult:
    @property
    def c_index_1(self) -> float: ...
    @property
    def c_index_2(self) -> float: ...
    @property
    def difference(self) -> float: ...
    @property
    def variance_diff(self) -> float: ...
    @property
    def std_error_diff(self) -> float: ...
    @property
    def z_statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...

class CIndexDecompositionResult:
    @property
    def c_index(self) -> float: ...
    @property
    def c_index_ee(self) -> float: ...
    @property
    def c_index_ec(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def n_event_event_pairs(self) -> int: ...
    @property
    def n_event_censored_pairs(self) -> int: ...
    @property
    def concordant_ee(self) -> float: ...
    @property
    def concordant_ec(self) -> float: ...
    @property
    def discordant_ee(self) -> float: ...
    @property
    def discordant_ec(self) -> float: ...
    @property
    def tied_ee(self) -> float: ...
    @property
    def tied_ec(self) -> float: ...

class GonenHellerResult:
    @property
    def cpe(self) -> float: ...
    @property
    def n_pairs(self) -> int: ...
    @property
    def n_ties(self) -> int: ...
    @property
    def variance(self) -> float: ...
    @property
    def std_error(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...

class TimeDepAUCResult:
    @property
    def auc(self) -> float: ...
    @property
    def time(self) -> float: ...
    @property
    def n_cases(self) -> int: ...
    @property
    def n_controls(self) -> int: ...
    @property
    def std_error(self) -> float: ...
    @property
    def ci_lower(self) -> float: ...
    @property
    def ci_upper(self) -> float: ...

class CumulativeDynamicAUCResult:
    @property
    def times(self) -> list[float]: ...
    @property
    def auc(self) -> list[float]: ...
    @property
    def mean_auc(self) -> float: ...
    @property
    def integrated_auc(self) -> float: ...
    @property
    def n_cases(self) -> list[int]: ...
    @property
    def n_controls(self) -> list[int]: ...

class SurvregPrediction:
    @property
    def predictions(self) -> list[float]: ...
    @property
    def se(self) -> list[float] | None: ...
    @property
    def prediction_type(self) -> str: ...
    @property
    def n(self) -> int: ...

class SurvregQuantilePrediction:
    @property
    def quantiles(self) -> list[float]: ...
    @property
    def predictions(self) -> list[list[float]]: ...
    @property
    def n(self) -> int: ...

class SurvregResiduals:
    @property
    def residuals(self) -> list[float]: ...
    @property
    def residual_type(self) -> str: ...
    @property
    def n(self) -> int: ...

class SurvfitResiduals:
    @property
    def residuals(self) -> list[float]: ...
    @property
    def time(self) -> list[float]: ...
    @property
    def residual_type(self) -> str: ...

class DfbetaResult:
    @property
    def dfbeta(self) -> list[list[float]]: ...
    @property
    def dfbetas(self) -> list[list[float]]: ...
    @property
    def max_dfbeta(self) -> list[float]: ...
    @property
    def influential_obs(self) -> list[int]: ...
    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...

class LeverageResult:
    @property
    def leverage(self) -> list[float]: ...
    @property
    def lmax(self) -> list[float]: ...
    @property
    def mean_leverage(self) -> float: ...
    @property
    def high_leverage_obs(self) -> list[int]: ...
    @property
    def n_obs(self) -> int: ...

class SchoenfeldSmoothResult:
    @property
    def times(self) -> list[float]: ...
    @property
    def smoothed_residuals(self) -> list[list[float]]: ...
    @property
    def coefficient_path(self) -> list[list[float]]: ...
    @property
    def slope_test_stats(self) -> list[float]: ...
    @property
    def slope_p_values(self) -> list[float]: ...
    @property
    def non_proportional_vars(self) -> list[int]: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_vars(self) -> int: ...

class OutlierDetectionResult:
    @property
    def martingale_residuals(self) -> list[float]: ...
    @property
    def deviance_residuals(self) -> list[float]: ...
    @property
    def standardized_deviance(self) -> list[float]: ...
    @property
    def outlier_indices(self) -> list[int]: ...
    @property
    def extreme_survivor_indices(self) -> list[int]: ...
    @property
    def outlier_scores(self) -> list[float]: ...
    @property
    def threshold(self) -> float: ...
    @property
    def n_outliers(self) -> int: ...

class ModelInfluenceResult:
    @property
    def cooks_distance(self) -> list[float]: ...
    @property
    def covratio(self) -> list[float]: ...
    @property
    def dffits(self) -> list[float]: ...
    @property
    def likelihood_displacement(self) -> list[float]: ...
    @property
    def influential_by_cooks(self) -> list[int]: ...
    @property
    def influential_by_covratio(self) -> list[int]: ...
    @property
    def influential_by_dffits(self) -> list[int]: ...
    @property
    def overall_influential(self) -> list[int]: ...
    @property
    def n_obs(self) -> int: ...

class GofTestResult:
    @property
    def global_test_stat(self) -> float: ...
    @property
    def global_p_value(self) -> float: ...
    @property
    def variable_test_stats(self) -> list[float]: ...
    @property
    def variable_p_values(self) -> list[float]: ...
    @property
    def linear_test_stat(self) -> float: ...
    @property
    def linear_p_value(self) -> float: ...
    @property
    def df(self) -> int: ...
    @property
    def n_obs(self) -> int: ...

class SurvivalFit:
    coefficients: list[float]
    location_coefficients: list[float]
    scale: float
    scales: list[float]
    distribution: str
    n_covariates: int
    n_strata: int
    linear_predictors: list[float]
    time: list[float]
    time2: list[float] | None
    status: list[int]
    covariates: list[list[float]]
    strata: list[int]
    weights: list[float]
    iterations: int
    variance_matrix: list[list[float]]
    log_likelihood: float
    convergence_flag: int
    score_vector: list[float]
    def predict(
        self,
        covariates: list[list[float]] | None = None,
        predict_type: str = "response",
        offset: list[float] | None = None,
        se_fit: bool = False,
    ) -> SurvregPrediction: ...
    def predict_quantile(
        self,
        covariates: list[list[float]] | None = None,
        quantiles: list[float] | None = None,
        offset: list[float] | None = None,
    ) -> SurvregQuantilePrediction: ...
    def residuals(self, residual_type: str = "deviance") -> SurvregResiduals: ...
    def dfbeta(self) -> list[list[float]]: ...

class SurvregConfig:
    max_iter: int
    eps: float
    tol_chol: float
    distribution: DistributionType

    def __init__(
        self,
        distribution: DistributionType | None = None,
        max_iter: int | None = None,
        eps: float | None = None,
        tol_chol: float | None = None,
    ) -> None: ...

class BootstrapResult:
    def __init__(
        self,
        coefficients: list[float],
        std_errors: list[float],
        ci_lower: list[float],
        ci_upper: list[float],
        bootstrap_samples: list[list[float]],
    ) -> None: ...
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def bootstrap_samples(self) -> list[list[float]]: ...

class CVResult:
    def __init__(
        self,
        fold_scores: list[float],
        mean_score: float,
        std_score: float,
        fold_coefficients: list[list[float]],
    ) -> None: ...
    @property
    def fold_scores(self) -> list[float]: ...
    @property
    def mean_score(self) -> float: ...
    @property
    def std_score(self) -> float: ...
    @property
    def fold_coefficients(self) -> list[list[float]]: ...

class TestResult:
    def __init__(self, statistic: float, df: int, p_value: float, test_name: str) -> None: ...
    @property
    def statistic(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def df(self) -> int: ...
    @property
    def test_name(self) -> str: ...

class ProportionalityTest:
    def __init__(
        self,
        variable_names: list[str],
        chi2_values: list[float],
        p_values: list[float],
        global_chi2: float,
        global_df: int,
        global_p_value: float,
    ) -> None: ...
    @property
    def variable_names(self) -> list[str]: ...
    @property
    def chi2_values(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def global_chi2(self) -> float: ...
    @property
    def global_df(self) -> int: ...
    @property
    def global_p_value(self) -> float: ...

class PriorType:
    Normal: PriorType
    Laplace: PriorType
    Cauchy: PriorType
    Horseshoe: PriorType
    Flat: PriorType
    def __init__(self, name: str) -> None: ...

class BayesianCoxConfig:
    prior_type: PriorType
    prior_scale: float
    n_samples: int
    n_warmup: int
    n_chains: int
    target_accept: float
    seed: int | None
    def __init__(
        self,
        prior_type: PriorType = PriorType.Normal,
        prior_scale: float = 2.5,
        n_samples: int = 2000,
        n_warmup: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.8,
        seed: int | None = None,
    ) -> None: ...

class BayesianCoxResult:
    @property
    def posterior_mean(self) -> list[float]: ...
    @property
    def posterior_sd(self) -> list[float]: ...
    @property
    def credible_lower(self) -> list[float]: ...
    @property
    def credible_upper(self) -> list[float]: ...
    @property
    def hazard_ratio_mean(self) -> list[float]: ...
    @property
    def hazard_ratio_lower(self) -> list[float]: ...
    @property
    def hazard_ratio_upper(self) -> list[float]: ...
    @property
    def samples(self) -> list[list[float]]: ...
    @property
    def log_posterior(self) -> list[float]: ...
    @property
    def waic(self) -> float: ...
    @property
    def loo(self) -> float: ...
    @property
    def rhat(self) -> list[float]: ...
    @property
    def n_eff(self) -> list[float]: ...

class BayesianDistribution:
    Weibull: BayesianDistribution
    LogNormal: BayesianDistribution
    LogLogistic: BayesianDistribution
    Exponential: BayesianDistribution
    def __init__(self, name: str) -> None: ...

class BayesianParametricConfig:
    distribution: BayesianDistribution
    beta_prior_scale: float
    shape_prior_mean: float
    shape_prior_sd: float
    n_samples: int
    n_warmup: int
    n_chains: int
    seed: int | None
    def __init__(
        self,
        distribution: BayesianDistribution = BayesianDistribution.Weibull,
        beta_prior_scale: float = 2.5,
        shape_prior_mean: float = 1.0,
        shape_prior_sd: float = 1.0,
        n_samples: int = 2000,
        n_warmup: int = 1000,
        n_chains: int = 4,
        seed: int | None = None,
    ) -> None: ...

class BayesianParametricResult:
    @property
    def beta_mean(self) -> list[float]: ...
    @property
    def beta_sd(self) -> list[float]: ...
    @property
    def beta_lower(self) -> list[float]: ...
    @property
    def beta_upper(self) -> list[float]: ...
    @property
    def shape_mean(self) -> float: ...
    @property
    def shape_sd(self) -> float: ...
    @property
    def shape_lower(self) -> float: ...
    @property
    def shape_upper(self) -> float: ...
    @property
    def acceleration_factor_mean(self) -> list[float]: ...
    @property
    def acceleration_factor_lower(self) -> list[float]: ...
    @property
    def acceleration_factor_upper(self) -> list[float]: ...
    @property
    def beta_samples(self) -> list[list[float]]: ...
    @property
    def shape_samples(self) -> list[float]: ...
    @property
    def log_posterior(self) -> list[float]: ...
    @property
    def dic(self) -> float: ...
    @property
    def waic(self) -> float: ...

class DirichletProcessConfig:
    concentration: float
    n_components: int
    n_iter: int
    burnin: int
    seed: int | None
    def __init__(
        self,
        concentration: float = 1.0,
        n_components: int = 10,
        n_iter: int = 1000,
        burnin: int = 500,
        seed: int | None = None,
    ) -> None: ...

class DirichletProcessResult:
    @property
    def cluster_assignments(self) -> list[int]: ...
    @property
    def cluster_sizes(self) -> list[int]: ...
    @property
    def cluster_survival(self) -> list[list[float]]: ...
    @property
    def eval_times(self) -> list[float]: ...
    @property
    def posterior_mean_survival(self) -> list[float]: ...
    @property
    def posterior_lower(self) -> list[float]: ...
    @property
    def posterior_upper(self) -> list[float]: ...
    @property
    def n_clusters(self) -> int: ...
    @property
    def concentration_posterior(self) -> float: ...

class BayesianModelAveragingConfig:
    n_iter: int
    burnin: int
    prior_inclusion_prob: float
    seed: int | None
    def __init__(
        self,
        n_iter: int = 2000,
        burnin: int = 1000,
        prior_inclusion_prob: float = 0.5,
        seed: int | None = None,
    ) -> None: ...

class BayesianModelAveragingResult:
    @property
    def posterior_inclusion_prob(self) -> list[float]: ...
    @property
    def posterior_mean_coef(self) -> list[float]: ...
    @property
    def posterior_sd_coef(self) -> list[float]: ...
    @property
    def model_posterior_probs(self) -> list[float]: ...
    @property
    def best_model_indices(self) -> list[int]: ...
    @property
    def bayes_factor_vs_null(self) -> list[float]: ...
    @property
    def n_models_visited(self) -> int: ...
    @property
    def n_vars(self) -> int: ...

class SpikeSlabConfig:
    spike_var: float
    slab_var: float
    prior_inclusion: float
    n_iter: int
    burnin: int
    seed: int | None
    def __init__(
        self,
        spike_var: float = 0.001,
        slab_var: float = 10.0,
        prior_inclusion: float = 0.5,
        n_iter: int = 2000,
        burnin: int = 1000,
        seed: int | None = None,
    ) -> None: ...

class SpikeSlabResult:
    @property
    def posterior_inclusion_prob(self) -> list[float]: ...
    @property
    def posterior_mean(self) -> list[float]: ...
    @property
    def posterior_sd(self) -> list[float]: ...
    @property
    def credible_lower(self) -> list[float]: ...
    @property
    def credible_upper(self) -> list[float]: ...
    @property
    def selected_variables(self) -> list[int]: ...
    @property
    def n_selected(self) -> int: ...
    @property
    def log_marginal_likelihood(self) -> float: ...

class HorseshoeConfig:
    tau_global: float
    n_iter: int
    burnin: int
    seed: int | None
    def __init__(
        self,
        tau_global: float = 1.0,
        n_iter: int = 2000,
        burnin: int = 1000,
        seed: int | None = None,
    ) -> None: ...

class HorseshoeResult:
    @property
    def posterior_mean(self) -> list[float]: ...
    @property
    def posterior_sd(self) -> list[float]: ...
    @property
    def credible_lower(self) -> list[float]: ...
    @property
    def credible_upper(self) -> list[float]: ...
    @property
    def shrinkage_factors(self) -> list[float]: ...
    @property
    def local_scales(self) -> list[float]: ...
    @property
    def global_scale(self) -> float: ...
    @property
    def effective_df(self) -> float: ...

class PenaltyType:
    ElasticNet: PenaltyType
    Lasso: PenaltyType
    Ridge: PenaltyType
    def __init__(self, name: str) -> None: ...

class ElasticNetConfig:
    alpha: float
    l1_ratio: float
    max_iter: int
    tol: float
    standardize: bool
    warm_start: bool
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-7,
        standardize: bool = True,
        warm_start: bool = False,
    ) -> None: ...
    @staticmethod
    def lasso(alpha: float) -> ElasticNetConfig: ...
    @staticmethod
    def ridge(alpha: float) -> ElasticNetConfig: ...

class RidgePenalty:
    theta: float
    scale: bool
    @property
    def df(self) -> float | None: ...
    def __init__(self, theta: float, scale: bool | None = None) -> None: ...
    @staticmethod
    def from_df(df: float, n_vars: int, scale: bool | None = None) -> RidgePenalty: ...
    def penalty_value(self, beta: list[float]) -> float: ...
    def penalty_gradient(self, beta: list[float]) -> list[float]: ...
    def apply_to_information(self, info_diag: list[float]) -> list[float]: ...

class RidgeResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_err(self) -> list[float]: ...
    @property
    def df(self) -> float: ...
    @property
    def gcv(self) -> float: ...
    @property
    def theta(self) -> float: ...
    @property
    def scale_factors(self) -> list[float] | None: ...

class ElasticNetPathConfig:
    l1_ratio: float
    n_lambda: int
    lambda_min_ratio: float | None
    max_iter: int
    tol: float
    def __init__(
        self,
        l1_ratio: float = 0.5,
        n_lambda: int = 100,
        lambda_min_ratio: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-7,
    ) -> None: ...

class ElasticNetCVConfig:
    l1_ratio: float
    n_lambda: int
    n_folds: int
    def __init__(self, l1_ratio: float = 0.5, n_lambda: int = 100, n_folds: int = 10) -> None: ...

class ElasticNetCoxResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def nonzero_indices(self) -> list[int]: ...
    @property
    def lambda_used(self) -> float: ...
    @property
    def l1_ratio(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def deviance(self) -> float: ...
    @property
    def df(self) -> float: ...
    @property
    def scale_factors(self) -> list[float] | None: ...
    @property
    def intercept(self) -> float: ...

class ElasticNetCoxPath:
    @property
    def lambdas(self) -> list[float]: ...
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def deviances(self) -> list[float]: ...
    @property
    def df(self) -> list[float]: ...
    @property
    def n_iters(self) -> list[int]: ...

class ScreeningRule:
    EDPP: ScreeningRule
    None_: ScreeningRule
    Safe: ScreeningRule
    Strong: ScreeningRule
    def __init__(self, name: str) -> None: ...

class FastCoxSolverConfig:
    max_iter: int
    tol: float
    screening: ScreeningRule
    working_set_size: int | None
    active_set_update_freq: int
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-7,
        screening: ScreeningRule | None = None,
        working_set_size: int | None = None,
        active_set_update_freq: int = 10,
    ) -> None: ...

class FastCoxConfig:
    lambda_: float
    l1_ratio: float
    max_iter: int
    tol: float
    screening: ScreeningRule
    working_set_size: int | None
    active_set_update_freq: int
    standardize: bool
    use_simd: bool
    def __init__(
        self,
        lambda_: float = 0.1,
        l1_ratio: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-7,
        screening: ScreeningRule | None = None,
        working_set_size: int | None = None,
        active_set_update_freq: int = 10,
        standardize: bool = True,
        use_simd: bool = True,
    ) -> None: ...

class FastCoxPathConfig:
    l1_ratio: float
    n_lambda: int
    lambda_min_ratio: float | None
    max_iter: int
    tol: float
    screening: ScreeningRule
    def __init__(
        self,
        l1_ratio: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-7,
        screening: ScreeningRule | None = None,
    ) -> None: ...

class FastCoxCVConfig:
    l1_ratio: float
    n_lambda: int
    n_folds: int
    screening: ScreeningRule
    seed: int | None
    def __init__(
        self,
        l1_ratio: float = 1.0,
        n_lambda: int = 100,
        n_folds: int = 5,
        screening: ScreeningRule | None = None,
        seed: int | None = None,
    ) -> None: ...

class FastCoxResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def nonzero_indices(self) -> list[int]: ...
    @property
    def lambda_used(self) -> float: ...
    @property
    def l1_ratio(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def deviance(self) -> float: ...
    @property
    def df(self) -> float: ...
    @property
    def scale_factors(self) -> list[float] | None: ...
    @property
    def center_values(self) -> list[float] | None: ...
    @property
    def screened_out(self) -> int: ...
    @property
    def active_set_size(self) -> int: ...

class FastCoxPath:
    @property
    def lambdas(self) -> list[float]: ...
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def deviances(self) -> list[float]: ...
    @property
    def df(self) -> list[float]: ...
    @property
    def n_iters(self) -> list[int]: ...
    @property
    def converged(self) -> list[bool]: ...

class GroupLassoConfig:
    lambda_: float
    max_iter: int
    tol: float
    standardize: bool
    group_weights: list[float] | None
    def __init__(
        self,
        lambda_: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        standardize: bool = True,
        group_weights: list[float] | None = None,
    ) -> None: ...

class GroupLassoResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def selected_groups(self) -> list[int]: ...
    @property
    def group_norms(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def lambda_(self) -> float: ...
    @property
    def n_groups(self) -> int: ...
    @property
    def df(self) -> int: ...

class SparseBoostingConfig:
    n_iterations: int
    learning_rate: float
    subsample_ratio: float
    early_stopping_rounds: int
    l1_penalty: float
    seed: int | None
    def __init__(
        self,
        n_iterations: int = 100,
        learning_rate: float = 0.1,
        subsample_ratio: float = 0.8,
        early_stopping_rounds: int = 10,
        l1_penalty: float = 0.0,
        seed: int | None = None,
    ) -> None: ...

class SparseBoostingResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def selected_features(self) -> list[int]: ...
    @property
    def feature_importance(self) -> list[float]: ...
    @property
    def iteration_scores(self) -> list[float]: ...
    @property
    def best_iteration(self) -> int: ...
    @property
    def n_selected(self) -> int: ...

class SISConfig:
    n_select: int
    iterative: bool
    max_iter: int
    threshold: float
    def __init__(
        self,
        n_select: int | None = None,
        iterative: bool = False,
        max_iter: int = 5,
        threshold: float = 0.0,
    ) -> None: ...

class SISResult:
    @property
    def selected_features(self) -> list[int]: ...
    @property
    def marginal_scores(self) -> list[float]: ...
    @property
    def ranking(self) -> list[int]: ...
    @property
    def n_selected(self) -> int: ...
    @property
    def iteration_selections(self) -> list[list[int]]: ...

class StabilitySelectionConfig:
    n_bootstrap: int
    subsample_ratio: float
    lambda_range: list[float]
    threshold: float
    seed: int | None
    def __init__(
        self,
        n_bootstrap: int = 100,
        subsample_ratio: float = 0.5,
        lambda_range: list[float] | None = None,
        threshold: float = 0.6,
        seed: int | None = None,
    ) -> None: ...

class StabilitySelectionResult:
    @property
    def selected_features(self) -> list[int]: ...
    @property
    def selection_probabilities(self) -> list[float]: ...
    @property
    def stable_features(self) -> list[int]: ...
    @property
    def per_lambda_selections(self) -> list[list[float]]: ...
    @property
    def n_selected(self) -> int: ...

class CensoringType:
    Censored: CensoringType
    Competing: CensoringType
    def __init__(self, name: str) -> None: ...

class CauseSpecificCoxConfig:
    cause_of_interest: int
    treat_other_causes_as: CensoringType
    max_iter: int
    tol: float
    ties: str
    def __init__(
        self,
        cause_of_interest: int = 1,
        treat_other_causes_as: CensoringType | None = None,
        max_iter: int = 100,
        tol: float = 1e-9,
        ties: str = "breslow",
    ) -> None: ...

class CauseSpecificCoxResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def hr_ci_lower(self) -> list[float]: ...
    @property
    def hr_ci_upper(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_at_risk(self) -> int: ...
    @property
    def n_competing(self) -> int: ...
    @property
    def n_censored(self) -> int: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def cause_of_interest(self) -> int: ...
    @property
    def baseline_hazard_times(self) -> list[float]: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def cumulative_baseline_hazard(self) -> list[float]: ...
    def predict_cumulative_hazard(self, x: list[float], n_obs: int) -> list[list[float]]: ...
    def predict_survival(self, x: list[float], n_obs: int) -> list[list[float]]: ...
    def predict_cif(self, x: list[float], n_obs: int) -> list[list[float]]: ...

class CorrelationType:
    Independent: CorrelationType
    SharedFrailty: CorrelationType
    CopulaBased: CorrelationType
    def __init__(self, name: str) -> None: ...

class JointCompetingRisksConfig:
    num_causes: int
    correlation_structure: CorrelationType
    frailty_variance: float
    max_iter: int
    tol: float
    estimate_correlation: bool
    def __init__(
        self,
        num_causes: int = 2,
        correlation_structure: CorrelationType | None = None,
        frailty_variance: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        estimate_correlation: bool = True,
    ) -> None: ...

class CauseResult:
    @property
    def cause(self) -> int: ...
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def hazard_ratios(self) -> list[float]: ...
    @property
    def baseline_hazard_times(self) -> list[float]: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def cumulative_baseline_hazard(self) -> list[float]: ...

class JointCompetingRisksResult:
    @property
    def cause_specific_results(self) -> list[CauseResult]: ...
    @property
    def subdistribution_results(self) -> list[CauseResult]: ...
    @property
    def correlation_matrix(self) -> list[list[float]] | None: ...
    @property
    def frailty_variance(self) -> float | None: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_events_by_cause(self) -> list[int]: ...
    @property
    def n_obs(self) -> int: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    def predict_cif(self, x: list[float], n_obs: int, cause_idx: int) -> list[list[float]]: ...
    def predict_overall_survival(self, x: list[float], n_obs: int) -> list[list[float]]: ...

class JointModelConfig:
    n_quadrature_points: int
    max_iter: int
    tolerance: float
    association_type: str
    random_effects_structure: str
    def __init__(
        self,
        n_quadrature_points: int = 15,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        association_type: str = "value",
        random_effects_structure: str = "intercept_slope",
    ) -> None: ...

class JointLongSurvResult:
    @property
    def longitudinal_fixed_effects(self) -> list[float]: ...
    @property
    def survival_coefficients(self) -> list[float]: ...
    @property
    def association_parameter(self) -> float: ...
    @property
    def random_effects_variance(self) -> list[float]: ...
    @property
    def residual_variance(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def convergence_iterations(self) -> int: ...
    def predict_longitudinal(
        self,
        time: list[float],
        covariates: list[list[float]],
    ) -> list[float]: ...
    def predict_survival(
        self,
        time: list[float],
        covariates: list[list[float]],
    ) -> list[float]: ...

class LandmarkAnalysisResult:
    @property
    def landmark_time(self) -> float: ...
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def standard_errors(self) -> list[float]: ...
    @property
    def n_at_risk(self) -> int: ...
    @property
    def n_events(self) -> int: ...
    @property
    def prediction_times(self) -> list[float]: ...
    @property
    def survival_probabilities(self) -> list[float]: ...

class LongDynamicPredResult:
    @property
    def prediction_time(self) -> float: ...
    @property
    def horizon(self) -> float: ...
    @property
    def survival_probabilities(self) -> list[float]: ...
    @property
    def confidence_lower(self) -> list[float]: ...
    @property
    def confidence_upper(self) -> list[float]: ...
    @property
    def risk_scores(self) -> list[float]: ...

class TimeVaryingCoxResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def coefficient_times(self) -> list[float]: ...
    @property
    def standard_errors(self) -> list[list[float]]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    def coefficients_at_time(self, t: float) -> list[float]: ...

class BasisType:
    BSpline: BasisType
    Fourier: BasisType
    Wavelet: BasisType
    FunctionalPCA: BasisType
    def __init__(self, name: str) -> None: ...

class FunctionalSurvivalConfig:
    basis_type: BasisType
    n_basis: int
    n_pca_components: int
    regularization: float
    max_iter: int
    tol: float
    def __init__(
        self,
        basis_type: BasisType = BasisType.BSpline,
        n_basis: int = 10,
        n_pca_components: int = 5,
        regularization: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None: ...

class FunctionalPCAResult:
    @property
    def eigenvalues(self) -> list[float]: ...
    @property
    def explained_variance_ratio(self) -> list[float]: ...
    @property
    def cumulative_variance(self) -> list[float]: ...
    @property
    def mean_function(self) -> list[float]: ...
    @property
    def principal_components(self) -> list[list[float]]: ...
    @property
    def scores(self) -> list[list[float]]: ...

class FunctionalSurvivalResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def coefficient_se(self) -> list[float]: ...
    @property
    def coefficient_function(self) -> list[float]: ...
    @property
    def coefficient_times(self) -> list[float]: ...
    @property
    def hazard_ratio(self) -> list[float]: ...
    @property
    def ci_lower(self) -> list[float]: ...
    @property
    def ci_upper(self) -> list[float]: ...
    @property
    def p_values(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def functional_pca(self) -> FunctionalPCAResult | None: ...
    @property
    def basis_coefficients(self) -> list[list[float]]: ...
    def predict_coefficient(self, t: float) -> float: ...

class SplineConfig:
    n_knots: int
    degree: int
    knot_placement: str
    boundary_knots: tuple[float, float] | None
    def __init__(
        self,
        n_knots: int = 4,
        degree: int = 3,
        knot_placement: str = "quantile",
        boundary_knots: tuple[float, float] | None = None,
    ) -> None: ...

class FlexibleParametricResult:
    def __init__(
        self,
        coefficients: list[float],
        spline_coefficients: list[float],
        std_errors: list[float],
        knots: list[float],
        log_likelihood: float,
        aic: float,
        bic: float,
        n_iterations: int,
        converged: bool,
    ) -> None: ...
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def spline_coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def knots(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...

class RestrictedCubicSplineResult:
    def __init__(
        self,
        knots: list[float],
        basis_matrix: list[list[float]],
        coefficients: list[float],
        std_errors: list[float],
    ) -> None: ...
    @property
    def knots(self) -> list[float]: ...
    @property
    def basis_matrix(self) -> list[list[float]]: ...
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...

class HazardSplineResult:
    def __init__(
        self,
        time_points: list[float],
        hazard: list[float],
        cumulative_hazard: list[float],
        survival: list[float],
        lower_ci: list[float],
        upper_ci: list[float],
    ) -> None: ...
    @property
    def time_points(self) -> list[float]: ...
    @property
    def hazard(self) -> list[float]: ...
    @property
    def cumulative_hazard(self) -> list[float]: ...
    @property
    def survival(self) -> list[float]: ...
    @property
    def lower_ci(self) -> list[float]: ...
    @property
    def upper_ci(self) -> list[float]: ...

class CureDistribution:
    Weibull: CureDistribution
    LogNormal: CureDistribution
    LogLogistic: CureDistribution
    Exponential: CureDistribution
    Gamma: CureDistribution
    def __init__(self, name: str) -> None: ...

class LinkFunction:
    Logit: LinkFunction
    Probit: LinkFunction
    CLogLog: LinkFunction
    Identity: LinkFunction
    def __init__(self, name: str) -> None: ...
    def link(self, p: float) -> float: ...
    def inv_link(self, eta: float) -> float: ...
    def deriv(self, eta: float) -> float: ...

class MixtureCureConfig:
    distribution: CureDistribution
    link: LinkFunction
    max_iter: int
    tol: float
    em_max_iter: int
    def __init__(
        self,
        distribution: CureDistribution | None = None,
        link: LinkFunction | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        em_max_iter: int = 500,
    ) -> None: ...

class MixtureCureResult:
    @property
    def cure_coef(self) -> list[float]: ...
    @property
    def survival_coef(self) -> list[float]: ...
    @property
    def scale(self) -> float: ...
    @property
    def shape(self) -> float: ...
    @property
    def cure_fraction(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def cure_prob(self) -> list[float]: ...

class PromotionTimeCureResult:
    @property
    def theta(self) -> float: ...
    @property
    def coef(self) -> list[float]: ...
    @property
    def scale(self) -> float: ...
    @property
    def shape(self) -> float: ...
    @property
    def cure_fraction(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...

class BoundedCumulativeHazardConfig:
    distribution: CureDistribution
    max_iter: int
    tol: float
    alpha: float
    def __init__(
        self,
        distribution: CureDistribution | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        alpha: float = 1.0,
    ) -> None: ...

class BoundedCumulativeHazardResult:
    @property
    def coef(self) -> list[float]: ...
    @property
    def scale(self) -> float: ...
    @property
    def shape(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def cure_fraction(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def cumulative_hazard_bound(self) -> float: ...
    @property
    def std_errors(self) -> list[float]: ...

class NonMixtureType:
    GeometricGeneralized: NonMixtureType
    NegativeBinomial: NonMixtureType
    Poisson: NonMixtureType
    Destructive: NonMixtureType
    def __init__(self, name: str) -> None: ...

class NonMixtureCureConfig:
    model_type: NonMixtureType
    distribution: CureDistribution
    max_iter: int
    tol: float
    dispersion: float
    def __init__(
        self,
        model_type: NonMixtureType | None = None,
        distribution: CureDistribution | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        dispersion: float = 1.0,
    ) -> None: ...

class NonMixtureCureResult:
    @property
    def coef(self) -> list[float]: ...
    @property
    def theta(self) -> float: ...
    @property
    def scale(self) -> float: ...
    @property
    def shape(self) -> float: ...
    @property
    def dispersion(self) -> float: ...
    @property
    def cure_fraction(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def survival_probs(self) -> list[float]: ...

class CureModelComparisonResult:
    @property
    def model_names(self) -> list[str]: ...
    @property
    def log_likelihoods(self) -> list[float]: ...
    @property
    def aic_values(self) -> list[float]: ...
    @property
    def bic_values(self) -> list[float]: ...
    @property
    def cure_fractions(self) -> list[float]: ...
    @property
    def best_model_aic(self) -> str: ...
    @property
    def best_model_bic(self) -> str: ...

class AssociationStructure:
    Value: AssociationStructure
    Slope: AssociationStructure
    ValueSlope: AssociationStructure
    Area: AssociationStructure
    SharedRandomEffects: AssociationStructure
    def __init__(self, name: str) -> None: ...

class JointSurvivalModelConfig:
    association: AssociationStructure
    n_quadrature: int
    max_iter: int
    tol: float
    baseline_hazard_knots: int
    def __init__(
        self,
        association: AssociationStructure = AssociationStructure.Value,
        n_quadrature: int = 15,
        max_iter: int = 500,
        tol: float = 1e-4,
        baseline_hazard_knots: int = 5,
    ) -> None: ...

class JointModelResult:
    @property
    def longitudinal_fixed(self) -> list[float]: ...
    @property
    def longitudinal_fixed_se(self) -> list[float]: ...
    @property
    def survival_fixed(self) -> list[float]: ...
    @property
    def survival_fixed_se(self) -> list[float]: ...
    @property
    def association_param(self) -> float: ...
    @property
    def association_se(self) -> float: ...
    @property
    def random_effects_var(self) -> list[float]: ...
    @property
    def residual_var(self) -> float: ...
    @property
    def baseline_hazard(self) -> list[float]: ...
    @property
    def baseline_hazard_times(self) -> list[float]: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def random_effects(self) -> list[list[float]]: ...

class DynamicPredictionResult:
    @property
    def time_points(self) -> list[float]: ...
    @property
    def survival_mean(self) -> list[float]: ...
    @property
    def survival_lower(self) -> list[float]: ...
    @property
    def survival_upper(self) -> list[float]: ...
    @property
    def cumulative_risk(self) -> list[float]: ...
    @property
    def conditional_survival(self) -> list[float]: ...
    @property
    def auc(self) -> float: ...
    @property
    def brier_score(self) -> float: ...

class TimeVaryingAUCResult:
    @property
    def times(self) -> list[float]: ...
    @property
    def auc_values(self) -> list[float]: ...
    @property
    def auc_lower(self) -> list[float]: ...
    @property
    def auc_upper(self) -> list[float]: ...
    @property
    def integrated_auc(self) -> float: ...
    @property
    def n_cases(self) -> list[int]: ...
    @property
    def n_controls(self) -> list[int]: ...

class DynamicCIndexResult:
    @property
    def c_index(self) -> float: ...
    @property
    def se(self) -> float: ...
    @property
    def lower(self) -> float: ...
    @property
    def upper(self) -> float: ...
    @property
    def n_concordant(self) -> int: ...
    @property
    def n_discordant(self) -> int: ...
    @property
    def n_tied(self) -> int: ...
    @property
    def n_pairs(self) -> int: ...
    @property
    def time_dependent_c(self) -> list[float]: ...
    @property
    def eval_times(self) -> list[float]: ...

class IPCWAUCResult:
    @property
    def times(self) -> list[float]: ...
    @property
    def auc_values(self) -> list[float]: ...
    @property
    def auc_se(self) -> list[float]: ...
    @property
    def integrated_auc(self) -> float: ...
    @property
    def ipcw_weights(self) -> list[float]: ...

class SuperLandmarkResult:
    @property
    def landmark_times(self) -> list[float]: ...
    @property
    def coefficients(self) -> list[list[float]]: ...
    @property
    def std_errors(self) -> list[list[float]]: ...
    @property
    def c_indices(self) -> list[float]: ...
    @property
    def brier_scores(self) -> list[float]: ...
    @property
    def n_at_risk(self) -> list[int]: ...
    @property
    def n_events(self) -> list[int]: ...
    @property
    def pooled_coef(self) -> list[float]: ...
    @property
    def pooled_se(self) -> list[float]: ...

class TimeDependentROCResult:
    @property
    def times(self) -> list[float]: ...
    @property
    def sensitivity(self) -> list[list[float]]: ...
    @property
    def specificity(self) -> list[list[float]]: ...
    @property
    def thresholds(self) -> list[float]: ...
    @property
    def auc(self) -> list[float]: ...
    @property
    def optimal_threshold(self) -> list[float]: ...

class IPCWInput:
    time: list[float]
    status: list[int]
    x_censoring: list[float]
    n_obs: int
    n_vars: int

    def __init__(
        self,
        time: list[float],
        status: list[int],
        x_censoring: list[float],
        n_obs: int,
        n_vars: int,
    ) -> None: ...

class IPCWConfig:
    stabilized: bool
    trim: float | None

    def __init__(self, stabilized: bool = True, trim: float | None = None) -> None: ...

class IPCWResult:
    weights: list[float]
    censoring_probs: list[float]
    treatment_effect: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_effective: float

class GComputationResult:
    ate: float
    ate_se: float
    ate_ci_lower: float
    ate_ci_upper: float
    potential_outcome_treated: float
    potential_outcome_control: float
    survival_treated: list[float]
    survival_control: list[float]
    time_points: list[float]
    rmst_treated: float
    rmst_control: float
    rmst_difference: float

class CausalForestConfig:
    n_trees: int
    max_depth: int
    min_samples_leaf: int
    min_samples_split: int
    max_features: int | None
    honesty: bool
    honesty_fraction: float
    seed: int | None
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        min_samples_split: int = 10,
        max_features: int | None = None,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        seed: int | None = None,
    ) -> None: ...

class CausalForestResult:
    cate_estimates: list[float]
    cate_se: list[float]
    feature_importance: list[float]
    ate: float
    ate_se: float

class CausalForestSurvival:
    def predict_cate(self, covariates: list[list[float]]) -> list[float]: ...
    def predict_variance(self, covariates: list[list[float]]) -> list[float]: ...
    def feature_importance(self) -> list[float]: ...

class CounterfactualSurvivalConfig:
    representation_dim: int
    hidden_dims: list[int]
    balance_alpha: float
    learning_rate: float
    n_epochs: int
    batch_size: int
    dropout_rate: float
    seed: int | None
    def __init__(
        self,
        representation_dim: int = 64,
        hidden_dims: list[int] | None = None,
        balance_alpha: float = 1.0,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 64,
        dropout_rate: float = 0.1,
        seed: int | None = None,
    ) -> None: ...

class CounterfactualSurvivalResult:
    ite: list[float]
    survival_treated: list[list[float]]
    survival_control: list[list[float]]
    time_points: list[float]
    ate: float
    ate_rmst: float

class TVSurvCausConfig:
    hidden_dim: int
    num_rnn_layers: int
    balance_lambda: float
    learning_rate: float
    n_epochs: int
    dropout_rate: float
    def __init__(
        self,
        hidden_dim: int = 64,
        num_rnn_layers: int = 2,
        balance_lambda: float = 1.0,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        dropout_rate: float = 0.1,
    ) -> None: ...

class TVSurvCausResult:
    time_varying_ite: list[list[float]]
    counterfactual_survival: list[list[list[float]]]
    time_points: list[float]
    treatment_times: list[float]

class CopulaType:
    Clayton: CopulaType
    Frank: CopulaType
    Gumbel: CopulaType
    Gaussian: CopulaType
    Independent: CopulaType
    def __init__(self, name: str) -> None: ...

class CopulaCensoringConfig:
    copula_type: CopulaType
    theta: float | None
    max_iter: int
    tol: float
    n_grid: int
    def __init__(
        self,
        copula_type: CopulaType = ...,
        theta: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_grid: int = 100,
    ) -> None: ...

class CopulaCensoringResult:
    theta: float
    theta_se: float
    kendall_tau: float
    marginal_survival_t: list[float]
    marginal_survival_c: list[float]
    joint_survival: list[list[float]]
    eval_times: list[float]
    log_likelihood: float
    aic: float
    n_iter: int
    converged: bool

class SensitivityBoundsConfig:
    gamma_range: list[float]
    n_grid: int
    method: str
    def __init__(
        self,
        gamma_range: list[float] | None = None,
        n_grid: int = 100,
        method: str = "rosenbaum",
    ) -> None: ...

class SensitivityBoundsResult:
    gamma_values: list[float]
    survival_lower: list[list[float]]
    survival_upper: list[list[float]]
    rmst_lower: list[float]
    rmst_upper: list[float]
    hazard_ratio_lower: list[float]
    hazard_ratio_upper: list[float]
    eval_times: list[float]
    point_estimate: float

class MNARSurvivalConfig:
    delta_range: list[float]
    pattern: str
    def __init__(
        self,
        delta_range: list[float] | None = None,
        pattern: str = "tilt",
    ) -> None: ...

class MNARSurvivalResult:
    delta_values: list[float]
    adjusted_survival: list[list[float]]
    adjusted_rmst: list[float]
    adjusted_median: list[float]
    eval_times: list[float]
    reference_survival: list[float]

class DoubleMLConfig:
    n_folds: int
    n_rep: int
    score: str
    trimming_threshold: float
    seed: int | None
    def __init__(
        self,
        n_folds: int = 5,
        n_rep: int = 1,
        score: str | None = None,
        trimming_threshold: float = 0.01,
        seed: int | None = None,
    ) -> None: ...

class DoubleMLResult:
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_obs: int
    scores: list[float]
    def is_significant(self, alpha: float) -> bool: ...

class CATEResult:
    cate_estimates: list[float]
    cate_se: list[float]
    group_labels: list[str]
    group_sizes: list[int]

class IVCoxConfig:
    max_iter: int
    tol: float
    two_stage: bool
    robust_variance: bool
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        two_stage: bool = True,
        robust_variance: bool = True,
    ) -> None: ...

class IVCoxResult:
    treatment_coef: float
    treatment_se: float
    treatment_z: float
    treatment_pvalue: float
    covariate_coef: list[float]
    covariate_se: list[float]
    log_likelihood: float
    first_stage_f: float
    first_stage_r2: float
    weak_instrument_test: float
    sargan_test: float
    sargan_pvalue: float
    n_iter: int
    converged: bool

class RDSurvivalConfig:
    bandwidth: float
    kernel: str
    polynomial_order: int
    fuzzy: bool
    def __init__(
        self,
        bandwidth: float | None = None,
        kernel: str = "triangular",
        polynomial_order: int = 1,
        fuzzy: bool = False,
    ) -> None: ...

class RDSurvivalResult:
    treatment_effect: float
    se: float
    ci_lower: float
    ci_upper: float
    z_score: float
    p_value: float
    bandwidth_used: float
    n_left: int
    n_right: int
    survival_left: float
    survival_right: float

class MediationSurvivalConfig:
    max_iter: int
    tol: float
    n_bootstrap: int
    seed: int | None
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_bootstrap: int = 500,
        seed: int | None = None,
    ) -> None: ...

class MediationSurvivalResult:
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    total_se: float
    direct_se: float
    indirect_se: float
    total_pvalue: float
    direct_pvalue: float
    indirect_pvalue: float
    treatment_to_mediator: float
    mediator_to_outcome: float

class GEstimationConfig:
    max_iter: int
    tol: float
    model_type: str
    def __init__(self, max_iter: int = 100, tol: float = 1e-6, model_type: str = "aft") -> None: ...

class GEstimationResult:
    psi: list[float]
    se: list[float]
    z_scores: list[float]
    p_values: list[float]
    counterfactual_times: list[float]
    treatment_effect_ratio: float
    n_iter: int
    converged: bool

class MSMResult:
    coefficients: list[float]
    std_errors: list[float]
    hazard_ratios: list[float]
    hr_ci_lower: list[float]
    hr_ci_upper: list[float]
    weights: list[float]
    effective_n: float
    log_likelihood: float

class TrialEmulationConfig:
    grace_period: float
    max_followup: float
    clone_censor_weighting: bool
    stabilized_weights: bool
    trim_weights: float
    n_bootstrap: int
    def __init__(
        self,
        grace_period: float = 0.0,
        max_followup: float | None = None,
        clone_censor_weighting: bool = True,
        stabilized_weights: bool = True,
        trim_weights: float = 0.01,
        n_bootstrap: int = 200,
    ) -> None: ...

class TargetTrialResult:
    hazard_ratio: float
    hr_ci_lower: float
    hr_ci_upper: float
    risk_difference: float
    rd_ci_lower: float
    rd_ci_upper: float
    survival_treated: list[float]
    survival_control: list[float]
    time_points: list[float]
    n_eligible: int
    n_treated: int
    n_control: int
    n_clones: int
    weights: list[float]

class TMLEConfig:
    n_folds: int
    trimming: float
    max_iter: int
    tol: float
    seed: int | None
    def __init__(
        self,
        n_folds: int = 5,
        trimming: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> None: ...

class TMLEResult:
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    influence_function: list[float]
    n_obs: int
    def is_significant(self, alpha: float) -> bool: ...

class TMLESurvivalResult:
    time_points: list[float]
    survival_diff: list[float]
    se: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    rmst_diff: float
    rmst_se: float

class DriftConfig:
    window_size: int
    threshold_psi: float
    threshold_ks: float
    n_bins: int
    def __init__(
        self,
        window_size: int = 1000,
        threshold_psi: float = 0.2,
        threshold_ks: float = 0.05,
        n_bins: int = 10,
    ) -> None: ...

class FeatureDriftResult:
    @property
    def feature_name(self) -> str: ...
    @property
    def psi(self) -> float: ...
    @property
    def ks_statistic(self) -> float: ...
    @property
    def ks_pvalue(self) -> float: ...
    @property
    def has_drift(self) -> bool: ...
    @property
    def drift_severity(self) -> str: ...

class DriftReport:
    @property
    def feature_results(self) -> list[FeatureDriftResult]: ...
    @property
    def overall_drift_detected(self) -> bool: ...
    @property
    def n_features_drifted(self) -> int: ...
    @property
    def prediction_drift_psi(self) -> float: ...
    @property
    def prediction_drift_detected(self) -> bool: ...
    def to_summary(self) -> str: ...

class PerformanceDriftResult:
    @property
    def time_periods(self) -> list[str]: ...
    @property
    def c_indices(self) -> list[float]: ...
    @property
    def calibration_slopes(self) -> list[float]: ...
    @property
    def drift_detected(self) -> bool: ...
    @property
    def c_index_change(self) -> float: ...
    @property
    def recommendation(self) -> str: ...

class ModelPerformanceMetrics:
    c_index: float
    brier_score: float
    integrated_brier: float
    calibration_slope: float
    calibration_intercept: float
    ci_lower_c_index: float
    ci_upper_c_index: float
    def __init__(
        self,
        c_index: float,
        brier_score: float | None = None,
        integrated_brier: float | None = None,
        calibration_slope: float | None = None,
        calibration_intercept: float | None = None,
        ci_lower_c_index: float | None = None,
        ci_upper_c_index: float | None = None,
    ) -> None: ...

class SubgroupPerformance:
    @property
    def subgroup_name(self) -> str: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def c_index(self) -> float: ...
    @property
    def event_rate(self) -> float: ...

class ModelCard:
    @property
    def model_name(self) -> str: ...
    @property
    def model_type(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def intended_use(self) -> str: ...
    @property
    def limitations(self) -> list[str]: ...
    @property
    def training_data_description(self) -> str: ...
    @property
    def n_training_samples(self) -> int: ...
    @property
    def n_events(self) -> int: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def overall_performance(self) -> ModelPerformanceMetrics: ...
    @property
    def subgroup_performance(self) -> list[SubgroupPerformance]: ...
    @property
    def ethical_considerations(self) -> list[str]: ...
    @property
    def caveats(self) -> list[str]: ...
    def to_markdown(self) -> str: ...
    def to_json(self) -> str: ...

class FairnessAuditResult:
    @property
    def protected_attribute(self) -> str: ...
    @property
    def group_names(self) -> list[str]: ...
    @property
    def group_c_indices(self) -> list[float]: ...
    @property
    def group_sizes(self) -> list[int]: ...
    @property
    def max_disparity(self) -> float: ...
    @property
    def passes_threshold(self) -> bool: ...

class QALYResult:
    @property
    def qaly(self) -> float: ...
    @property
    def life_years(self) -> float: ...
    @property
    def mean_utility(self) -> float: ...
    @property
    def qaly_by_period(self) -> list[float]: ...
    @property
    def discounted_qaly(self) -> float: ...
    @property
    def qaly_se(self) -> float: ...
    @property
    def qaly_ci_lower(self) -> float: ...
    @property
    def qaly_ci_upper(self) -> float: ...

class QTWISTResult:
    @property
    def qtwist(self) -> float: ...
    @property
    def tox(self) -> float: ...
    @property
    def twistt(self) -> float: ...
    @property
    def rel(self) -> float: ...
    @property
    def total_time(self) -> float: ...
    @property
    def utility_tox(self) -> float: ...
    @property
    def utility_rel(self) -> float: ...
    @property
    def qtwist_difference(self) -> float | None: ...
    @property
    def ci_lower(self) -> float | None: ...
    @property
    def ci_upper(self) -> float | None: ...

class RCLLResult:
    def __init__(
        self,
        rcll: float,
        mean_rcll: float,
        n_events: int,
        n_censored: int,
        event_contribution: float,
        censored_contribution: float,
    ) -> None: ...
    @property
    def rcll(self) -> float: ...
    @property
    def mean_rcll(self) -> float: ...
    @property
    def n_events(self) -> int: ...
    @property
    def n_censored(self) -> int: ...
    @property
    def event_contribution(self) -> float: ...
    @property
    def censored_contribution(self) -> float: ...

class TurnbullResult:
    @property
    def time_points(self) -> list[float]: ...
    @property
    def survival(self) -> list[float]: ...
    @property
    def survival_lower(self) -> list[float]: ...
    @property
    def survival_upper(self) -> list[float]: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...

class IntervalDistribution:
    Weibull: IntervalDistribution
    LogNormal: IntervalDistribution
    LogLogistic: IntervalDistribution
    Exponential: IntervalDistribution
    Generalized: IntervalDistribution
    def __init__(self, name: str) -> None: ...

class IntervalCensoredResult:
    @property
    def coefficients(self) -> list[float]: ...
    @property
    def std_errors(self) -> list[float]: ...
    @property
    def scale(self) -> float: ...
    @property
    def shape(self) -> float: ...
    @property
    def log_likelihood(self) -> float: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def n_iter(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def survival_prob(self) -> list[float]: ...

def survsplit(tstart: list[float], tstop: list[float], cut: list[float]) -> SplitResult: ...
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
def tmerge3(id: list[int], miss: list[bool]) -> list[int]: ...
def collapse(
    y: list[float],
    x: list[int],
    istate: list[int],
    id: list[int],
    wt: list[float],
    order: list[int],
) -> dict[str, list[list[int]] | list[str]]: ...
def cluster(id: list[int]) -> ClusterResult: ...
def cluster_str(id: list[str]) -> ClusterResult: ...
def strata(variables: list[list[int]]) -> StrataResult: ...
def strata_str(variables: list[list[str]]) -> StrataResult: ...
def aeq_surv(time: list[float], tolerance: float | None = None) -> AeqSurvResult: ...
def neardate(
    id1: list[int],
    date1: list[float],
    id2: list[int],
    date2: list[float],
    best: str | None = None,
    nomatch: int | None = None,
) -> NearDateResult: ...
def neardate_str(
    id1: list[str],
    date1: list[float],
    id2: list[str],
    date2: list[float],
    best: str | None = None,
    nomatch: int | None = None,
) -> NearDateResult: ...
def rttright(
    time: list[float],
    status: list[int],
    weights: list[float] | None = None,
) -> RttrightResult: ...
def rttright_stratified(
    time: list[float],
    status: list[int],
    strata: list[int],
    weights: list[float] | None = None,
) -> RttrightResult: ...
def surv2data(
    id: list[int],
    time: list[float],
    event_time: list[float] | None = None,
    event_status: list[int] | None = None,
) -> Surv2DataResult: ...
def survcondense(
    id: list[int],
    time1: list[float],
    time2: list[float],
    status: list[int],
) -> CondenseResult: ...
def tcut(
    value: list[float],
    breaks: list[float],
    labels: list[str] | None = None,
) -> TcutResult: ...
def tcut_expand(
    start: list[float],
    stop: list[float],
    cuts: list[float],
) -> tuple[list[float], list[float], list[int], list[int]]: ...
def to_timeline(
    id: list[int],
    time1: list[float],
    time2: list[float],
    status: list[int],
    time_points: list[float] | None = None,
) -> TimelineResult: ...
def from_timeline(
    id: list[int],
    states: list[list[int]],
    time_points: list[float],
) -> IntervalResult: ...
def turnbull_estimator(
    left: list[float],
    right: list[float],
    max_iter: int = 1000,
    tol: float = 1e-6,
    weights: list[float] | None = None,
) -> TurnbullResult: ...
def npmle_interval(
    left: list[float],
    right: list[float],
    weights: list[float] | None = None,
) -> tuple[list[float], list[float]]: ...
def interval_censored_regression(
    left: list[float],
    right: list[float],
    censor_type: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    distribution: IntervalDistribution,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> IntervalCensoredResult: ...
def compute_baseline_survival_steps(
    ndeath: list[int],
    risk: list[float],
    wt: list[float],
    sn: int,
    denom: list[float],
) -> list[float]: ...
def agsurv4(
    ndeath: list[int],
    risk: list[float],
    wt: list[float],
    sn: int,
    denom: list[float],
) -> list[float]: ...
def compute_tied_baseline_summaries(
    n: int,
    nvar: int,
    dd: list[int],
    x1: list[float],
    x2: list[float],
    xsum: list[float],
    xsum2: list[float],
) -> dict[str, list[float]]: ...
def agsurv5(
    n: int,
    nvar: int,
    dd: list[int],
    x1: list[float],
    x2: list[float],
    xsum: list[float],
    xsum2: list[float],
) -> dict[str, list[float]]: ...
def coxcount1(
    time: list[float],
    status: list[float],
    strata: list[int],
) -> CoxCountOutput: ...
def coxcount2(
    time1: list[float],
    time2: list[float],
    status: list[float],
    sort1: list[int],
    sort2: list[int],
    strata: list[int],
) -> CoxCountOutput: ...
def perform_concordance1_calculation(
    time_data: list[float],
    weights: list[float],
    indices: list[int],
    ntree: int,
) -> dict[str, float | list[float]]: ...
def perform_concordance3_calculation(
    time_data: list[float],
    indices: list[int],
    weights: list[float],
    time_weights: list[float],
    sort_stop: list[int],
    do_residuals: bool,
) -> dict[str, float | int | list[float]]: ...
def perform_concordance_calculation(
    time_data: list[float],
    predictor_values: list[int],
    weights: list[float],
    time_weights: list[float],
    sort_stop: list[int],
    sort_start: list[int] | None = None,
    do_residuals: bool | None = None,
) -> dict[str, float | int | list[float]]: ...
def perform_score_calculation(
    time_data: list[float],
    covariates: list[float],
    strata: list[int],
    score: list[float],
    weights: list[float],
    method: int,
) -> dict[str, list[float] | int | str]: ...
def perform_agscore3_calculation(
    time_data: list[float],
    covariates: list[float],
    strata: list[int],
    score: list[float],
    weights: list[float],
    method: int,
    sort1: list[int],
) -> dict[str, list[float] | int | str]: ...
def perform_pystep_simple_calculation(
    odim: int,
    data: list[float],
    ofac: list[int],
    odims: list[int],
    ocut: list[list[float]],
    timeleft: float,
) -> dict[str, float | int]: ...
def perform_pystep_calculation(
    edim: int,
    data: list[float],
    efac: list[int],
    edims: list[int],
    ecut: list[list[float]],
    tmax: float,
) -> dict[str, float | int | list[float]]: ...
def perform_pyears_calculation(
    time_data: list[float],
    weights: list[float],
    expected_dim: int,
    expected_factors: list[int],
    expected_dims: list[int],
    expected_cuts: list[float],
    expected_rates: list[float],
    expected_data: list[float],
    observed_dim: int,
    observed_factors: list[int],
    observed_dims: list[int],
    observed_cuts: list[float],
    method: int,
    observed_data: list[float],
    do_event: int | None,
    ny: int | None,
) -> dict[str, list[float] | float]: ...
def cox_callback(
    which: int,
    coef: list[float],
    first: list[float],
    second: list[float],
    penalty: list[float],
    flag: list[int],
    fexpr: Callable[..., dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float], list[int]]: ...
def survfit_from_hazard(
    time: list[float],
    hazard: list[float],
    n_risk: list[float] | None = None,
    n_event: list[float] | None = None,
) -> SurvfitMatrixResult: ...
def survfit_from_cumhaz(
    time: list[float],
    cumhaz: list[float],
    n_risk: list[float] | None = None,
    n_event: list[float] | None = None,
) -> SurvfitMatrixResult: ...
def survfit_from_matrix(
    time: list[float],
    hazard_matrix: list[list[float]],
) -> SurvfitMatrixResult: ...
def survfit_multistate(
    time: list[float],
    transition_hazards: list[list[list[float]]],
    initial_state: int,
) -> SurvfitMatrixResult: ...
def survfitkm(
    time: list[float],
    status: list[int],
    weights: list[float] | None = None,
    entry_times: list[float] | None = None,
    position: list[int] | None = None,
    reverse: bool | None = None,
    computation_type: int | None = None,
    conf_level: float | None = None,
    conf_type: str | None = None,
) -> SurvFitKMOutput: ...
def survfitkm_with_options(
    time: list[float],
    status: list[int],
    options: SurvfitKMOptions | None = None,
) -> SurvFitKMOutput: ...
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
def survfitaj_extended(
    from_state: list[int],
    to_state: list[int],
    time: list[float],
    config: AalenJohansenExtendedConfig,
    weights: list[float] | None = None,
) -> AalenJohansenExtendedResult: ...
def pseudo(
    time: list[float],
    status: list[int],
    eval_times: list[float] | None = None,
    type_: str | None = None,
) -> PseudoResult: ...
def pseudo_fast(
    time: list[float],
    status: list[int],
    eval_times: list[float] | None = None,
    type_: str | None = None,
) -> PseudoResult: ...
def pseudo_gee_regression(
    pseudo_values: list[list[float]],
    covariates: list[list[float]],
    cluster_id: list[int] | None = None,
    config: GEEConfig | None = None,
) -> GEEResult: ...
def aggregate_survfit(
    times: list[list[float]],
    survs: list[list[float]],
    std_errs: list[list[float]] | None = None,
    weights: list[float] | None = None,
    conf_level: float | None = None,
) -> AggregateSurvfitResult: ...
def aggregate_survfit_by_group(
    times: list[list[float]],
    survs: list[list[float]],
    groups: list[int],
    weights: list[float] | None = None,
) -> list[AggregateSurvfitResult]: ...
def survcheck(
    id: list[int],
    time1: list[float],
    time2: list[float],
    status: list[int],
    istate: list[int] | None = None,
) -> SurvCheckResult: ...
def survcheck_simple(time: list[float], status: list[int]) -> SurvCheckResult: ...
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
def norisk(
    time1: list[float],
    time2: list[float],
    status: list[int],
    sort1: list[int],
    sort2: list[int],
    strata: list[int],
) -> list[int]: ...
def compute_logrank_components(
    time: list[float],
    status: list[int],
    group: list[int],
    strata: list[int] | None,
    rho: float | None,
) -> SurvDiffResult: ...
def survdiff2(
    time: list[float],
    status: list[int],
    group: list[int],
    strata: list[int] | None,
    rho: float | None,
) -> SurvDiffResult: ...
def logrank_test(
    time: list[float],
    status: list[int],
    group: list[int],
    weight_type: str | None = None,
    entry_times: list[float] | None = None,
) -> LogRankResult: ...
def fleming_harrington_test(
    time: list[float],
    status: list[int],
    group: list[int],
    rho: float = 0.0,
    gamma: float = 0.0,
    entry_times: list[float] | None = None,
) -> LogRankResult: ...
def logrank_trend(
    time: list[float],
    status: list[int],
    group: list[int],
    scores: list[float] | None = None,
) -> TrendTestResult: ...
def cipoisson_exact(k: int, time: float, p: float) -> tuple[float, float]: ...
def cipoisson_anscombe(k: int, time: float, p: float) -> tuple[float, float]: ...
def cipoisson(k: int, time: float, p: float, method: str) -> tuple[float, float]: ...
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
    means: list[float],
    beta: list[float],
    u: list[float],
    imat: list[float],
    loglik: list[float],
    work: list[float],
    work2: list[int],
    eps: float,
    tol_chol: float,
    nocenter: list[int],
) -> dict[str, Any]: ...
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
def royston(
    linear_predictor: list[float],
    time: list[float],
    status: list[int],
) -> RoystonResult: ...
def royston_from_model(
    x: list[float],
    coef: list[float],
    n_obs: int,
    time: list[float],
    status: list[int],
) -> RoystonResult: ...
def yates(
    predictions: list[float],
    factor: list[str],
    weights: list[float] | None = None,
    conf_level: float | None = None,
) -> YatesResult: ...
def yates_contrast(
    x: list[float],
    coef: list[float],
    n_obs: int,
    n_vars: int,
    factor_col: int,
    factor_levels: list[float],
    predict_type: str | None = None,
) -> YatesResult: ...
def yates_pairwise(yates_result: YatesResult) -> YatesPairwiseResult: ...
def create_simple_ratetable(
    age_breaks: list[float],
    year_breaks: list[float],
    rates_male: list[float],
    rates_female: list[float],
) -> RateTable: ...
def is_ratetable(ndim: int, has_rates: bool, has_dims: bool) -> bool: ...
def ratetable_date(
    year: int,
    month: int = 1,
    day: int = 1,
    origin_year: int = 1960,
) -> RatetableDateResult: ...
def days_to_date(days: float, origin_year: int) -> tuple[int, int, int]: ...
def survexp(
    time: list[float],
    age: list[float],
    year: list[float],
    ratetable: RateTable,
    sex: list[int] | None = None,
    times: list[float] | None = None,
    method: str | None = None,
) -> SurvExpResult: ...
def survexp_individual(
    time: list[float],
    age: list[float],
    year: list[float],
    ratetable: RateTable,
    sex: list[int] | None = None,
) -> list[float]: ...
def summary_pyears(
    pyears: list[float],
    pn: list[float],
    pcount: list[float],
    pexpect: list[float],
    offtable: float,
) -> PyearsSummary: ...
def pyears_by_cell(
    pyears: list[float],
    pn: list[float],
    pcount: list[float],
    pexpect: list[float],
) -> list[PyearsCell]: ...
def pyears_ci(
    observed: float, expected: float, conf_level: float
) -> tuple[float, float, float]: ...
def survexp_us() -> RateTable: ...
def survexp_mn() -> RateTable: ...
def survexp_usr() -> RateTable: ...
def compute_expected_survival(
    age: list[float],
    sex: list[int],
    year: list[float],
    times: list[float],
    ratetable: RateTable | None = None,
) -> ExpectedSurvivalResult: ...
def net_survival(
    time: list[float],
    status: list[int],
    expected_survival: list[float],
    method: NetSurvivalMethod = ...,
    weights: list[float] | None = None,
) -> NetSurvivalResult: ...
def crude_probability_of_death(
    time: list[float],
    status: list[int],
    _expected_survival: list[float],
    cause: list[int],
    time_points: list[float],
) -> tuple[list[float], list[float], list[float]]: ...
def relative_survival(
    time: list[float],
    status: list[int],
    expected_hazard: list[float],
    age_at_diagnosis: list[float],
    follow_up_years: list[float] | None = None,
) -> RelativeSurvivalResult: ...
def excess_hazard_regression(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    expected_hazard: list[float],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ExcessHazardModelResult: ...
def spatial_frailty_model(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    region_id: list[int],
    adjacency_matrix: list[float],
    n_regions: int,
    correlation_structure: SpatialCorrelationStructure = ...,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SpatialFrailtyResult: ...
def compute_spatial_smoothed_rates(
    observed_events: list[float],
    expected_events: list[float],
    adjacency_matrix: list[float],
    n_regions: int,
    smoothing_param: float,
) -> tuple[list[float], list[float]]: ...
def moran_i_test(
    values: list[float],
    adjacency_matrix: list[float],
    n_regions: int,
) -> tuple[float, float, float]: ...
def network_survival_model(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    adjacency_matrix: list[float],
    n_nodes: int,
    config: NetworkSurvivalConfig,
) -> NetworkSurvivalResult: ...
def diffusion_survival_model(
    infection_time: list[float],
    infected: list[int],
    covariates: list[float],
    n_covariates: int,
    adjacency_matrix: list[float],
    n_nodes: int,
    config: DiffusionSurvivalConfig,
) -> DiffusionSurvivalResult: ...
def network_heterogeneity_survival(
    time: list[float],
    event: list[int],
    adjacency_matrix: list[float],
    n_nodes: int,
    n_communities: int | None = None,
) -> NetworkHeterogeneityResult: ...
def survobrien(
    time: list[float],
    status: list[int],
    covariate: list[float],
    strata: list[int] | None = None,
) -> SurvObrienResult: ...
def finegray(
    tstart: list[float],
    tstop: list[float],
    ctime: list[float],
    cprob: list[float],
    extend: list[bool],
    keep: list[bool],
) -> FineGrayOutput: ...
def finegray_regression(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    event_type: int,
    max_iter: int = 25,
    eps: float = 1e-9,
) -> FineGrayResult: ...
def competing_risks_cif(
    time: list[float],
    status: list[int],
    event_type: int,
    confidence_level: float = 0.95,
) -> CompetingRisksCIF: ...
def estimate_transition_intensities(
    entry_time: list[float],
    exit_time: list[float],
    from_state: list[int],
    to_state: list[int],
    event: list[int],
    config: MultiStateConfig,
) -> TransitionIntensityResult: ...
def fit_multi_state_model(
    entry_time: list[float],
    exit_time: list[float],
    from_state: list[int],
    to_state: list[int],
    event: list[int],
    eval_times: list[float],
    config: MultiStateConfig,
) -> MultiStateResult: ...
def fit_markov_msm(
    entry_time: list[float],
    exit_time: list[float],
    from_state: list[int],
    to_state: list[int],
    event: list[int],
    eval_times: list[float],
    config: MultiStateConfig,
) -> MarkovMSMResult: ...
def fit_illness_death(
    entry_time: list[float],
    transition_time: list[float],
    exit_time: list[float],
    from_state: list[int],
    to_state: list[int],
    covariates: list[list[float]] | None = None,
    config: IllnessDeathConfig | None = None,
) -> IllnessDeathResult: ...
def predict_illness_death(
    model: IllnessDeathResult,
    current_state: int,
    time_in_state: float,
    prediction_times: list[float],
    covariates: list[float] | None = None,
) -> IllnessDeathPrediction: ...
def fit_semi_markov(
    entry_times: list[float],
    exit_times: list[float],
    from_states: list[int],
    to_states: list[int],
    config: SemiMarkovConfig,
) -> SemiMarkovResult: ...
def predict_semi_markov(
    model: SemiMarkovResult,
    current_state: int,
    time_in_state: float,
    prediction_times: list[float],
) -> SemiMarkovPrediction: ...
def statefig(
    states: list[str],
    transitions: dict[tuple[str, str], int],
    layout: list[int] | None = None,
) -> StateFigData: ...
def statefig_matplotlib_code(data: StateFigData) -> str: ...
def statefig_transition_matrix(data: StateFigData) -> list[list[int]]: ...
def statefig_validate(
    data: StateFigData,
    allowed_transitions: dict[tuple[str, str], bool],
) -> list[str]: ...
def d_calibration(
    survival_probs: list[float],
    status: list[int],
    n_bins: int | None = None,
) -> DCalibrationResult: ...
def one_calibration(
    time: list[float],
    status: list[int],
    predicted_survival_at_t: list[float],
    time_point: float,
    n_groups: int | None = None,
) -> OneCalibrationResult: ...
def calibration_plot(
    time: list[float],
    status: list[int],
    predicted_survival_at_t: list[float],
    time_point: float,
    n_groups: int | None = None,
) -> CalibrationPlotData: ...
def brier_calibration(
    time: list[float],
    status: list[int],
    predicted_survival_at_t: list[float],
    time_point: float,
    n_groups: int | None = None,
) -> BrierCalibrationResult: ...
def smoothed_calibration(
    time: list[float],
    status: list[int],
    predicted_survival_at_t: list[float],
    time_point: float,
    n_grid_points: int | None = None,
    bandwidth: float | None = None,
) -> SmoothedCalibrationCurve: ...
def km_plot_data(
    time: list[float],
    event: list[int],
    confidence_level: float = 0.95,
    group_name: str | None = None,
) -> KaplanMeierPlotData: ...
def forest_plot_data(
    variable_names: list[str],
    coefficients: list[float],
    standard_errors: list[float],
    confidence_level: float = 0.95,
) -> ForestPlotData: ...
def calibration_plot_data(
    predicted: list[float],
    observed: list[int],
    n_bins: int = 10,
) -> CalibrationCurveData: ...
def generate_survival_report(
    title: str,
    time: list[float],
    event: list[int],
    landmark_times: list[float] | None = None,
) -> SurvivalReport: ...
def roc_plot_data(scores: list[float], labels: list[int]) -> ROCPlotData: ...
def decision_curve_analysis(
    predicted_risk: list[float],
    time: list[float],
    event: list[int],
    time_horizon: float,
    thresholds: list[float] | None = None,
) -> DecisionCurveResult: ...
def clinical_utility_at_threshold(
    predicted_risk: list[float],
    time: list[float],
    event: list[int],
    time_horizon: float,
    threshold: float,
) -> ClinicalUtilityResult: ...
def compare_decision_curves(
    model_predictions: list[list[float]],
    model_names: list[str],
    time: list[float],
    event: list[int],
    time_horizon: float,
    thresholds: list[float] | None = None,
) -> ModelComparisonResult: ...
def risk_stratification(
    risk_scores: list[float],
    events: list[int],
    n_groups: int | None = None,
) -> RiskStratificationResult: ...
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
def multi_time_calibration(
    time: list[float],
    status: list[int],
    survival_predictions: list[list[float]],
    prediction_times: list[float],
    n_groups: int | None = None,
) -> MultiTimeCalibrationResult: ...
def analyze_missing_pattern(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    missing_indicators: list[bool],
    n_imputations: int = 5,
) -> tuple[list[float], list[str], bool]: ...
def multiple_imputation_survival(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    missing_indicators: list[bool],
    n_imputations: int = 5,
    method: ImputationMethod = ImputationMethod.PMM,
    max_iter: int = 20,
    seed: int | None = None,
) -> MultipleImputationResult: ...
def pattern_mixture_model(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    dropout_pattern: list[int],
    dropout_time: list[float] | None = None,
) -> PatternMixtureResult: ...
def sensitivity_analysis(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    dropout_pattern: list[int],
    delta_values: list[float],
    analysis_type: SensitivityAnalysisType = SensitivityAnalysisType.DeltaAdjustment,
) -> list[tuple[float, list[float], list[float]]]: ...
def tipping_point_analysis(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    dropout_pattern: list[int],
    coef_index: int,
    target_value: float,
    delta_range: tuple[float, float],
    n_steps: int,
) -> float | None: ...
def gap_time_model(
    subject_id: list[int],
    start_time: list[float],
    stop_time: list[float],
    event_status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> GapTimeResult: ...
def pwp_gap_time(
    subject_id: list[int],
    event_time: list[float],
    event_status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    stratify_by_event_number: bool = False,
) -> GapTimeResult: ...
def pwp_model(
    id: list[int],
    start: list[float],
    stop: list[float],
    event: list[int],
    event_number: list[int],
    covariates: list[float],
    config: PWPConfig,
) -> PWPResult: ...
def anderson_gill_model(
    id: list[int],
    start: list[float],
    stop: list[float],
    event: list[int],
    covariates: list[float],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> AndersonGillResult: ...
def wlw_model(
    id: list[int],
    time: list[float],
    event: list[int],
    stratum: list[int],
    covariates: list[float],
    config: WLWConfig,
) -> WLWResult: ...
def negative_binomial_frailty(
    id: list[int],
    time: list[float],
    event: list[int],
    covariates: list[float],
    offset: list[float] | None,
    config: NegativeBinomialFrailtyConfig,
) -> NegativeBinomialFrailtyResult: ...
def marginal_recurrent_model(
    subject_id: list[int],
    start_time: list[float],
    stop_time: list[float],
    event_status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    method: MarginalMethod,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MarginalModelResult: ...
def andersen_gill(
    subject_id: list[int],
    start_time: list[float],
    stop_time: list[float],
    event_status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
) -> MarginalModelResult: ...
def wei_lin_weissfeld(
    subject_id: list[int],
    event_time: list[float],
    event_status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
) -> MarginalModelResult: ...
def joint_frailty_model(
    subject_id: list[int],
    rec_start: list[float],
    rec_stop: list[float],
    rec_status: list[int],
    x_recurrent: list[float],
    n_rec_obs: int,
    n_rec_vars: int,
    term_time: list[float],
    term_status: list[int],
    x_terminal: list[float],
    n_subjects: int,
    n_term_vars: int,
    frailty_dist: FrailtyDistribution = FrailtyDistribution.Gamma,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> JointFrailtyResult: ...
def basehaz(
    time: list[float],
    status: list[int],
    linear_predictors: list[float],
    centered: bool,
    entry_times: list[float] | None = None,
    weights: list[float] | None = None,
) -> tuple[list[float], list[float]]: ...
def anova_coxph(
    logliks: list[float],
    dfs: list[int],
    model_names: list[str] | None = None,
    test: str = "LRT",
) -> AnovaCoxphResult: ...
def anova_coxph_single(
    loglik_null: float,
    loglik_full: float,
    df_null: int,
    df_full: int,
) -> AnovaCoxphResult: ...
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
def rmst_optimal_threshold(
    time: list[float],
    status: list[int],
    alpha: float | None = None,
    min_events_per_interval: int | None = None,
    confidence_level: float | None = None,
) -> RMSTOptimalThresholdResult: ...
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
def life_table(time: list[float], status: list[int], breaks: list[float]) -> LifeTableResult: ...
def reliability(
    time: list[float],
    surv: list[float],
    std_err: list[float] | None = None,
    conf_level: float = 0.95,
    scale: str = "cumhaz",
) -> ReliabilityResult: ...
def reliability_inverse(estimate: list[float], scale: str) -> list[float]: ...
def hazard_to_reliability(time: list[float], hazard: list[float]) -> ReliabilityResult: ...
def failure_probability(surv: list[float]) -> list[float]: ...
def conditional_reliability(
    time: list[float],
    surv: list[float],
    t0: float,
) -> ReliabilityResult: ...
def mean_residual_life(time: list[float], surv: list[float], at_time: float) -> float: ...
def warranty_analysis(
    time: list[float],
    event: list[int],
    n_units: int,
    config: WarrantyConfig,
) -> WarrantyResult: ...
def renewal_analysis(
    failure_times: list[float],
    event: list[int],
    time_horizon: float,
    repair_time: float | None = None,
) -> RenewalResult: ...
def reliability_growth(
    failure_times: list[float],
    cumulative_time: list[float],
) -> ReliabilityGrowthResult: ...
def coxph_fit(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    strata: list[int] | None = None,
    weights: list[float] | None = None,
    offset: list[float] | None = None,
    initial_beta: list[float] | None = None,
    max_iter: int | None = None,
    eps: float | None = None,
    toler: float | None = None,
    method: str | None = None,
    entry_times: list[float] | None = None,
    nocenter: list[float] | None = None,
) -> CoxPHFit: ...
def coxph_detail(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    coefficients: list[float],
    weights: list[float] | None = None,
    entry_times: list[float] | None = None,
    strata: list[int] | None = None,
    offset: list[float] | None = None,
    method: str = "breslow",
    center: float = 0.0,
) -> CoxphDetail: ...
def dfbeta_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    coefficients: list[float],
    threshold: float | None = None,
) -> DfbetaResult: ...
def leverage_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    coefficients: list[float],
    threshold_multiplier: float = 2.0,
) -> LeverageResult: ...
def smooth_schoenfeld(
    event_times: list[float],
    schoenfeld_residuals: list[float],
    n_covariates: int,
    coefficients: list[float],
    bandwidth: float | None = None,
    transform: str = "identity",
) -> SchoenfeldSmoothResult: ...
def outlier_detection_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    coefficients: list[float],
    outlier_threshold: float = 3.0,
) -> OutlierDetectionResult: ...
def model_influence_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    coefficients: list[float],
) -> ModelInfluenceResult: ...
def goodness_of_fit_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    coefficients: list[float],
) -> GofTestResult: ...
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
    time2: list[float] | None = None,
    fixed_scale: float | None = None,
) -> SurvivalFit: ...
def predict_survreg(
    covariates: list[list[float]],
    coefficients: list[float],
    scale: float,
    distribution: str,
    predict_type: str = "response",
    offset: list[float] | None = None,
    var_matrix: list[list[float]] | None = None,
    se_fit: bool = False,
) -> SurvregPrediction: ...
def predict_survreg_quantile(
    covariates: list[list[float]],
    coefficients: list[float],
    scale: float,
    distribution: str,
    quantiles: list[float],
    offset: list[float] | None = None,
) -> SurvregQuantilePrediction: ...
def residuals_survfit(
    time: list[float],
    status: list[int],
    surv_time: list[float],
    surv: list[float],
    residual_type: str = "martingale",
) -> SurvfitResiduals: ...
def residuals_survreg(
    time: list[float],
    status: list[int],
    linear_pred: list[float],
    scale: float,
    distribution: str,
    residual_type: str = "deviance",
    time2: list[float] | None = None,
) -> SurvregResiduals: ...
def survreg_residual_matrix(
    time: list[float],
    status: list[int],
    linear_pred: list[float],
    scale: float,
    distribution: str,
    time2: list[float] | None = None,
) -> list[list[float]]: ...
def survreg_influence_residuals(
    derivative_matrix: list[list[float]],
    covariates: list[list[float]],
    scales: list[float],
    strata: list[int],
    var_matrix: list[list[float]],
    residual_type: str,
    rsigma: bool = True,
) -> list[float]: ...
def survreg_dfbeta_residuals(
    derivative_matrix: list[list[float]],
    covariates: list[list[float]],
    scales: list[float],
    strata: list[int],
    var_matrix: list[list[float]],
    rsigma: bool = True,
    standardized: bool = False,
) -> list[list[float]]: ...
def dfbeta_survreg(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    linear_pred: list[float],
    scale: float,
    var_matrix: list[list[float]],
    distribution: str,
    time2: list[float] | None = None,
) -> list[list[float]]: ...
def calibration(
    predicted: list[float],
    time: list[float],
    status: list[int],
    n_groups: int = 10,
) -> CalibrationResult: ...
def uno_c_index(
    time: list[float],
    status: list[int],
    risk_score: list[float],
    tau: float | None = None,
) -> UnoCIndexResult: ...
def compare_uno_c_indices(
    time: list[float],
    status: list[int],
    risk_score_1: list[float],
    risk_score_2: list[float],
    tau: float | None = None,
) -> ConcordanceComparisonResult: ...
def c_index_decomposition(
    time: list[float],
    status: list[int],
    risk_score: list[float],
    tau: float | None = None,
) -> CIndexDecompositionResult: ...
def gonen_heller_concordance(linear_predictor: list[float]) -> GonenHellerResult: ...
def time_dependent_auc(
    time: list[float],
    status: list[int],
    marker: list[float],
    t: float,
) -> TimeDepAUCResult: ...
def cumulative_dynamic_auc(
    time: list[float],
    status: list[int],
    marker: list[float],
    times: list[float],
) -> CumulativeDynamicAUCResult: ...
def nsk(
    x: list[float],
    df: int | None = None,
    knots: list[float] | None = None,
    boundary_knots: tuple[float, float] | None = None,
) -> SplineBasisResult: ...
def cox_score_residuals(
    y: list[float],
    strata: list[int],
    covar: list[float],
    score: list[float],
    weights: list[float],
    nvar: int,
    method: int = 0,
) -> list[float]: ...
def schoenfeld_residuals(
    y: list[float],
    score: list[float],
    strata: list[int],
    covar: list[float],
    nvar: int,
    method: int = 0,
) -> list[float]: ...
def concordance(
    y: list[float],
    x: list[int],
    wt: list[float],
    timewt: list[float],
    sortstart: list[int] | None,
    sortstop: list[int],
) -> dict[str, list[float]]: ...
def concordance_index(
    time: list[float],
    status: list[int],
    risk_scores: list[float],
    weights: list[float] | None = None,
    timewt: str = "n",
) -> float: ...
def concordance_summary(
    time: list[float],
    status: list[int],
    risk_scores: list[float],
    weights: list[float] | None = None,
    timewt: str = "n",
) -> dict[str, float]: ...
def counting_concordance_index(
    start: list[float],
    stop: list[float],
    status: list[int],
    risk_scores: list[float],
    weights: list[float] | None = None,
    timewt: str = "n",
) -> float: ...
def counting_concordance_summary(
    start: list[float],
    stop: list[float],
    status: list[int],
    risk_scores: list[float],
    weights: list[float] | None = None,
    timewt: str = "n",
) -> dict[str, float]: ...
def brier(
    time: list[float],
    status: list[int],
    predicted: list[float],
    eval_time: float,
) -> float: ...
def integrated_brier(
    time: list[float],
    status: list[int],
    predicted: list[list[float]],
    eval_times: list[float],
) -> float: ...
def bootstrap_cox_ci(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    weights: list[float] | None = None,
    n_bootstrap: int | None = None,
    confidence_level: float | None = None,
    seed: int | None = None,
) -> BootstrapResult: ...
def cv_cox_concordance(
    time: list[float],
    status: list[int],
    covariates: list[list[float]],
    weights: list[float] | None = None,
    n_folds: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
) -> CVResult: ...
def lrt_test(
    loglik_full: float,
    loglik_reduced: float,
    df: int,
) -> TestResult: ...
def wald_test_py(
    coefficients: list[float],
    std_errors: list[float],
) -> TestResult: ...
def score_test_py(
    score_vector: list[float],
    information_matrix: list[list[float]],
) -> TestResult: ...
def ph_test(
    schoenfeld_residuals: list[list[float]],
    event_times: list[float],
    weights: list[float] | None = None,
) -> ProportionalityTest: ...
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
def bayesian_cox(
    x: list[float],
    n_obs: int,
    n_vars: int,
    time: list[float],
    status: list[int],
    config: BayesianCoxConfig,
) -> BayesianCoxResult: ...
def bayesian_cox_predict_survival(
    result: BayesianCoxResult,
    x_new: list[float],
    n_new: int,
    n_vars: int,
    baseline_hazard: list[float],
    time_points: list[float],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]: ...
def bayesian_parametric(
    time: list[float],
    status: list[int],
    x: list[float],
    n_obs: int,
    n_vars: int,
    config: BayesianParametricConfig,
) -> BayesianParametricResult: ...
def bayesian_parametric_predict(
    result: BayesianParametricResult,
    x_new: list[float],
    n_new: int,
    n_vars: int,
    time_points: list[float],
    distribution: BayesianDistribution,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]: ...
def dirichlet_process_survival(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    config: DirichletProcessConfig,
) -> DirichletProcessResult: ...
def bayesian_model_averaging_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    config: BayesianModelAveragingConfig,
) -> BayesianModelAveragingResult: ...
def spike_slab_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    config: SpikeSlabConfig,
) -> SpikeSlabResult: ...
def horseshoe_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    n_covariates: int,
    config: HorseshoeConfig,
) -> HorseshoeResult: ...
def elastic_net_cox(
    input: CoxRegressionInput,
    config: ElasticNetConfig,
) -> ElasticNetCoxResult: ...
def elastic_net_cox_path(
    input: CoxRegressionInput,
    config: ElasticNetPathConfig | None = None,
) -> ElasticNetCoxPath: ...
def elastic_net_cox_cv(
    input: CoxRegressionInput,
    config: ElasticNetCVConfig | None = None,
) -> tuple[float, float, list[float], list[float]]: ...
def fast_cox(input: CoxRegressionInput, config: FastCoxConfig) -> FastCoxResult: ...
def fast_cox_numpy(
    x: Any,
    time: Any,
    status: Any,
    config: FastCoxConfig,
    weights: Any | None = None,
    offset: Any | None = None,
) -> FastCoxResult: ...
def fast_cox_path(
    input: CoxRegressionInput,
    config: FastCoxPathConfig | None = None,
) -> FastCoxPath: ...
def fast_cox_cv(
    input: CoxRegressionInput,
    config: FastCoxCVConfig | None = None,
) -> tuple[float, float, list[float], list[float]]: ...
def group_lasso_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    groups: list[int],
    config: GroupLassoConfig,
) -> GroupLassoResult: ...
def sparse_boosting_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    config: SparseBoostingConfig,
) -> SparseBoostingResult: ...
def sis_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    config: SISConfig,
) -> SISResult: ...
def stability_selection_cox(
    time: list[float],
    event: list[int],
    covariates: list[float],
    config: StabilitySelectionConfig,
) -> StabilitySelectionResult: ...
def cause_specific_cox(
    x: list[float],
    n_obs: int,
    n_vars: int,
    time: list[float],
    cause: list[int],
    config: CauseSpecificCoxConfig,
    weights: list[float] | None = None,
) -> CauseSpecificCoxResult: ...
def cause_specific_cox_all(
    x: list[float],
    n_obs: int,
    n_vars: int,
    time: list[float],
    cause: list[int],
    max_cause: int,
    weights: list[float] | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> list[CauseSpecificCoxResult]: ...
def joint_competing_risks(
    x: list[float],
    n_obs: int,
    n_vars: int,
    time: list[float],
    cause: list[int],
    config: JointCompetingRisksConfig,
    weights: list[float] | None = None,
) -> JointCompetingRisksResult: ...
def joint_longitudinal_model(
    subject_id: list[int],
    longitudinal_times: list[float],
    longitudinal_values: list[float],
    survival_time: list[float],
    survival_event: list[int],
    longitudinal_covariates: list[list[float]],
    survival_covariates: list[list[float]],
    config: JointModelConfig,
) -> JointLongSurvResult: ...
def landmark_cox_analysis(
    time: list[float],
    event: list[int],
    covariates: list[list[float]],
    landmark_time: float,
    horizon: float,
) -> LandmarkAnalysisResult: ...
def longitudinal_dynamic_pred(
    subject_id: list[int],
    measurement_times: list[float],
    measurement_values: list[float],
    prediction_time: float,
    horizon: float,
    model_coefficients: list[float],
) -> LongDynamicPredResult: ...
def time_varying_cox(
    start_time: list[float],
    stop_time: list[float],
    event: list[int],
    covariates: list[list[float]],
    n_time_points: int = 10,
) -> TimeVaryingCoxResult: ...
def functional_cox(
    functional_covariates: list[list[float]],
    curve_times: list[float],
    time: list[float],
    event: list[int],
    scalar_covariates: list[list[float]] | None = None,
    config: FunctionalSurvivalConfig | None = None,
) -> FunctionalSurvivalResult: ...
def fpca_survival(curves: list[list[float]], n_components: int) -> FunctionalPCAResult: ...
def flexible_parametric_model(
    time: list[float],
    event: list[int],
    covariates: list[list[float]],
    config: SplineConfig | None = None,
) -> FlexibleParametricResult: ...
def restricted_cubic_spline(
    x: list[float],
    n_knots: int | None = None,
    knots: list[float] | None = None,
) -> RestrictedCubicSplineResult: ...
def predict_hazard_spline(
    model_result: FlexibleParametricResult,
    eval_times: list[float],
    covariate_values: list[float] | None = None,
) -> HazardSplineResult: ...
def coxmart(input: CoxMartInput, method: int | None = None) -> list[float]: ...
def agmart(input: AndersenGillInput, method: int | None = None) -> list[float]: ...
def bounded_cumulative_hazard_model(
    time: list[float],
    status: list[int],
    covariates: list[float],
    config: BoundedCumulativeHazardConfig,
) -> BoundedCumulativeHazardResult: ...
def compare_cure_models(
    time: list[float],
    status: list[int],
    covariates: list[float],
    distributions: list[str] | None = None,
) -> CureModelComparisonResult: ...
def mixture_cure_model(
    time: list[float],
    status: list[int],
    x_cure: list[float],
    x_surv: list[float],
    config: MixtureCureConfig,
) -> MixtureCureResult: ...
def non_mixture_cure_model(
    time: list[float],
    status: list[int],
    covariates: list[float],
    config: NonMixtureCureConfig,
) -> NonMixtureCureResult: ...
def predict_bounded_cumulative_hazard(
    result: BoundedCumulativeHazardResult,
    time_points: list[float],
    covariates: list[float],
    n_subjects: int,
    distribution: CureDistribution,
) -> list[list[float]]: ...
def predict_non_mixture_survival(
    result: NonMixtureCureResult,
    time_points: list[float],
    covariates: list[float],
    n_subjects: int,
    model_type: NonMixtureType,
    distribution: CureDistribution,
) -> list[list[float]]: ...
def promotion_time_cure_model(
    time: list[float],
    status: list[int],
    x: list[float],
    distribution: CureDistribution | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> PromotionTimeCureResult: ...
def joint_model(
    y_longitudinal: list[float],
    times_longitudinal: list[float],
    x_longitudinal: list[float],
    n_long_obs: int,
    n_long_vars: int,
    subject_ids_long: list[int],
    event_time: list[float],
    event_status: list[int],
    x_survival: list[float],
    n_subjects: int,
    n_surv_vars: int,
    config: JointSurvivalModelConfig,
) -> JointModelResult: ...
def dynamic_prediction(
    beta_long: list[float],
    gamma_surv: list[float],
    alpha: float,
    random_effects: list[float],
    baseline_hazard: list[float],
    baseline_times: list[float],
    y_history: list[float],
    times_history: list[float],
    x_long_fixed: list[float],
    n_history: int,
    n_long_vars: int,
    x_surv: list[float],
    n_surv_vars: int,
    landmark_time: float,
    prediction_times: list[float],
    n_monte_carlo: int = 500,
) -> DynamicPredictionResult: ...
def dynamic_auc(
    beta_long: list[float],
    gamma_surv: list[float],
    alpha: float,
    baseline_hazard: list[float],
    baseline_times: list[float],
    y_observed: list[float],
    times_observed: list[float],
    x_long_fixed: list[float],
    n_obs: int,
    n_long_vars: int,
    x_surv: list[float],
    n_surv_vars: int,
    event_time: list[float],
    event_status: list[int],
    horizon: float,
) -> float: ...
def dynamic_brier_score(
    survival_predictions: list[list[float]],
    event_time: list[float],
    event_status: list[int],
    prediction_times: list[float],
) -> list[float]: ...
def landmarking_analysis(
    event_time: list[float],
    event_status: list[int],
    covariates: list[float],
    n_subjects: int,
    n_vars: int,
    landmark_times: list[float],
    horizon: float,
) -> list[tuple[float, list[float], float]]: ...
def time_varying_auc(
    risk_scores: list[float],
    event_time: list[float],
    event_status: list[int],
    eval_times: list[float],
    prediction_window: float,
    method: str = "cumulative/dynamic",
) -> TimeVaryingAUCResult: ...
def dynamic_c_index(
    risk_scores: list[float],
    event_time: list[float],
    event_status: list[int],
    landmark_time: float,
    horizon: float,
    eval_times: list[float] | None = None,
) -> DynamicCIndexResult: ...
def ipcw_auc(
    risk_scores: list[float],
    event_time: list[float],
    event_status: list[int],
    eval_times: list[float],
) -> IPCWAUCResult: ...
def super_landmark_model(
    event_time: list[float],
    event_status: list[int],
    covariates: list[float],
    n_vars: int,
    landmark_times: list[float],
    horizon: float,
    max_iter: int = 50,
) -> SuperLandmarkResult: ...
def time_dependent_roc(
    risk_scores: list[float],
    event_time: list[float],
    event_status: list[int],
    eval_times: list[float],
    n_thresholds: int = 100,
) -> TimeDependentROCResult: ...

class AggregationMethod:
    Mean: AggregationMethod
    Integral: AggregationMethod
    MaxAbsolute: AggregationMethod
    TimeWeighted: AggregationMethod
    def __init__(self, name: str) -> None: ...

class SurvShapConfig:
    n_coalitions: int
    n_background: int
    seed: int | None
    parallel: bool
    def __init__(
        self,
        n_coalitions: int = 2048,
        n_background: int = 100,
        seed: int | None = None,
        parallel: bool = True,
    ) -> None: ...

class FeatureImportance:
    feature_idx: int
    importance: float
    importance_std: float | None

class SurvShapResult:
    shap_values: list[list[list[float]]]
    base_value: list[float]
    time_points: list[float]
    aggregated_importance: list[float] | None
    def get_sample_shap(self, sample_idx: int) -> list[list[float]]: ...
    def get_feature_shap(self, feature_idx: int) -> list[list[float]]: ...
    def get_shap_at_time(self, time_idx: int) -> list[list[float]]: ...
    def feature_ranking(
        self,
        method: AggregationMethod = ...,
        top_k: int | None = None,
    ) -> list[FeatureImportance]: ...
    def mean_absolute_shap(self) -> list[float]: ...
    def check_additivity(self, predictions: list[float], tolerance: float) -> list[bool]: ...

class SurvShapExplanation:
    shap_values: list[list[float]]
    base_value: list[float]
    time_points: list[float]
    feature_values: list[float]
    aggregated_importance: list[float] | None

class BootstrapSurvShapResult:
    shap_values_mean: list[list[list[float]]]
    shap_values_std: list[list[list[float]]]
    shap_values_lower: list[list[list[float]]]
    shap_values_upper: list[list[list[float]]]
    base_value: list[float]
    time_points: list[float]
    n_bootstrap: int
    confidence_level: float

class PermutationImportanceResult:
    importance: list[float]
    importance_std: list[float]
    baseline_score: float
    n_repeats: int
    def feature_ranking(self, top_k: int | None) -> list[FeatureImportance]: ...

class ShapInteractionResult:
    interaction_values: list[list[list[float]]]
    time_points: list[float]
    aggregated_interactions: list[list[float]] | None
    def get_interaction(self, feature_i: int, feature_j: int) -> list[float]: ...
    def top_interactions(self, top_k: int) -> list[tuple[int, int, float]]: ...

class TimeVaryingTestType:
    SlopeTest: TimeVaryingTestType
    VarianceTest: TimeVaryingTestType
    BreakpointTest: TimeVaryingTestType
    def __init__(self, name: str) -> None: ...

class TimeVaryingTestConfig:
    test_type: TimeVaryingTestType
    n_windows: int
    min_window_size: int
    significance_level: float
    n_permutations: int
    def __init__(
        self,
        test_type: TimeVaryingTestType = ...,
        n_windows: int = 5,
        min_window_size: int = 10,
        significance_level: float = 0.05,
        n_permutations: int = 1000,
    ) -> None: ...

class TimeVaryingTestResult:
    feature_idx: int
    is_time_varying: bool
    test_statistic: float
    p_value: float
    slope: float | None
    slope_se: float | None
    window_means: list[float] | None
    window_variances: list[float] | None
    breakpoint_time: float | None
    effect_size: float

class TimeVaryingAnalysis:
    results: list[TimeVaryingTestResult]
    time_varying_features: list[int]
    stable_features: list[int]
    feature_rankings: list[tuple[int, float]]
    def get_feature_result(self, feature_idx: int) -> TimeVaryingTestResult | None: ...

class ChangepointMethod:
    PELT: ChangepointMethod
    BinarySegment: ChangepointMethod
    BottomUp: ChangepointMethod
    def __init__(self, name: str) -> None: ...

class CostFunction:
    L2: CostFunction
    L1: CostFunction
    Normal: CostFunction
    Poisson: CostFunction
    def __init__(self, name: str) -> None: ...

class ChangepointConfig:
    method: ChangepointMethod
    cost: CostFunction
    penalty: float
    min_size: int
    max_changepoints: int | None
    def __init__(
        self,
        method: ChangepointMethod = ...,
        cost: CostFunction = ...,
        penalty: float = 1.0,
        min_size: int = 2,
        max_changepoints: int | None = None,
    ) -> None: ...

class Changepoint:
    index: int
    time: float
    cost_improvement: float
    mean_before: float
    mean_after: float

class ChangepointResult:
    feature_idx: int
    changepoints: list[Changepoint]
    segments: list[tuple[int, int]]
    segment_means: list[float]
    total_cost: float
    n_changepoints: int
    def get_segment_at(self, time_idx: int) -> int: ...

class AllChangepointsResult:
    results: list[ChangepointResult]
    features_with_changes: list[int]
    most_unstable_features: list[tuple[int, int]]

class GroupingMethod:
    Hierarchical: GroupingMethod
    KMeans: GroupingMethod
    Domain: GroupingMethod
    Automatic: GroupingMethod
    def __init__(self, name: str) -> None: ...

class LinkageType:
    Single: LinkageType
    Complete: LinkageType
    Average: LinkageType
    Ward: LinkageType
    def __init__(self, name: str) -> None: ...

class VariableGroupingConfig:
    method: GroupingMethod
    n_groups: int | None
    correlation_threshold: float
    linkage: LinkageType
    max_iter: int
    seed: int | None
    def __init__(
        self,
        method: GroupingMethod = ...,
        n_groups: int | None = None,
        correlation_threshold: float = 0.7,
        linkage: LinkageType = ...,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> None: ...

class FeatureGroup:
    group_id: int
    feature_indices: list[int]
    representative_feature: int
    group_importance: float
    internal_correlation: float

class VariableGroupingResult:
    groups: list[FeatureGroup]
    feature_to_group: list[int]
    correlation_matrix: list[list[float]]
    dendrogram: list[tuple[int, int, float, int]] | None
    n_groups: int
    n_features: int
    def get_group(self, group_id: int) -> FeatureGroup | None: ...
    def get_feature_group(self, feature_idx: int) -> int | None: ...
    def get_group_by_feature(self, feature_idx: int) -> FeatureGroup | None: ...

class ViewRecommendation:
    UseGlobal: ViewRecommendation
    UseLocal: ViewRecommendation
    UseBoth: ViewRecommendation
    def __init__(self, name: str) -> None: ...

class LocalGlobalConfig:
    heterogeneity_threshold: float
    monotonicity_threshold: float
    interaction_threshold: float
    n_bootstrap: int
    confidence_level: float
    def __init__(
        self,
        heterogeneity_threshold: float = 0.3,
        monotonicity_threshold: float = 0.8,
        interaction_threshold: float = 0.2,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
    ) -> None: ...

class FeatureViewAnalysis:
    feature_idx: int
    recommendation: ViewRecommendation
    heterogeneity_score: float
    monotonicity_score: float
    interaction_score: float
    global_mean_importance: float
    local_importance_std: float
    confidence_interval_width: float
    explanation: str

class LocalGlobalResult:
    analyses: list[FeatureViewAnalysis]
    global_features: list[int]
    local_features: list[int]
    both_features: list[int]
    summary_statistics: LocalGlobalSummary
    def get_feature_analysis(self, feature_idx: int) -> FeatureViewAnalysis | None: ...
    def features_by_recommendation(self, rec: ViewRecommendation) -> list[int]: ...

class LocalGlobalSummary:
    mean_heterogeneity: float
    mean_monotonicity: float
    mean_interaction: float
    n_features: int
    proportion_global: float
    proportion_local: float
    proportion_both: float

class ALEResult:
    feature_values: list[float]
    ale_values: list[float]
    feature_index: int
    num_intervals: int

class ALE2DResult:
    feature1_values: list[float]
    feature2_values: list[float]
    ale_values: list[list[float]]
    feature1_index: int
    feature2_index: int

class FriedmanHResult:
    feature1_index: int
    feature2_index: int
    h_statistic: float
    interaction_strength: float

class FeatureImportanceResult:
    feature_index: int
    main_effect: float
    total_effect: float
    interaction_effect: float

class ICEResult:
    grid_values: list[float]
    ice_curves: list[list[float]]
    pdp_curve: list[float]
    feature_index: int
    centered: bool
    def get_curve(self, index: int) -> list[float]: ...

class DICEResult:
    grid_values: list[float]
    derivative_curves: list[list[float]]
    mean_derivative: list[float]
    feature_index: int

def survshap(
    x_explain: list[float],
    x_background: list[float],
    predictions_explain: list[float],
    predictions_background: list[float],
    time_points: list[float],
    n_explain: int,
    n_background: int,
    n_features: int,
    config: SurvShapConfig | None = None,
    aggregation_method: AggregationMethod | None = None,
) -> SurvShapResult: ...
def survshap_from_model(
    x_explain: list[float],
    x_background: list[float],
    time_points: list[float],
    n_explain: int,
    n_background: int,
    n_features: int,
    predict_fn: Callable[[list[float], int], list[float]],
    config: SurvShapConfig | None = None,
    aggregation_method: AggregationMethod | None = None,
) -> SurvShapResult: ...
def survshap_bootstrap(
    x_explain: list[float],
    x_background: list[float],
    predictions_explain: list[float],
    predictions_background: list[float],
    time_points: list[float],
    n_explain: int,
    n_background: int,
    n_features: int,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    config: SurvShapConfig | None = None,
) -> BootstrapSurvShapResult: ...
def aggregate_survshap(
    shap_values: list[list[list[float]]],
    time_points: list[float],
    method: AggregationMethod,
) -> list[float]: ...
def permutation_importance(
    predictions: list[float],
    time_points: list[float],
    times: list[float],
    events: list[int],
    n_samples: int,
    n_features: int,
    n_repeats: int = 10,
    seed: int | None = None,
    parallel: bool = True,
) -> PermutationImportanceResult: ...
def compute_shap_interactions(
    shap_values: list[list[list[float]]],
    time_points: list[float],
    n_features: int,
    aggregation_method: AggregationMethod | None = None,
) -> ShapInteractionResult: ...
def detect_time_varying_features(
    shap_values: list[list[list[float]]],
    time_points: list[float],
    n_samples: int,
    n_features: int,
    config: TimeVaryingTestConfig,
) -> TimeVaryingAnalysis: ...
def detect_changepoints(
    shap_values: list[list[list[float]]],
    time_points: list[float],
    n_samples: int,
    n_features: int,
    config: ChangepointConfig,
) -> AllChangepointsResult: ...
def detect_changepoints_single_series(
    data: list[float],
    time_points: list[float],
    config: ChangepointConfig,
) -> ChangepointResult: ...
def group_variables(
    shap_values: list[list[list[float]]],
    n_samples: int,
    n_features: int,
    n_times: int,
    config: VariableGroupingConfig,
) -> VariableGroupingResult: ...
def analyze_local_global(
    shap_values: list[list[list[float]]],
    feature_values: list[float],
    n_samples: int,
    n_features: int,
    n_times: int,
    config: LocalGlobalConfig,
    seed: int | None = None,
) -> LocalGlobalResult: ...
def compute_ale(
    covariates: list[list[float]],
    predictions: list[float],
    feature_index: int,
    num_intervals: int = 20,
) -> ALEResult: ...
def compute_ale_2d(
    covariates: list[list[float]],
    predictions: list[float],
    feature1_index: int,
    feature2_index: int,
    num_intervals: int = 10,
) -> ALE2DResult: ...
def compute_time_varying_ale(
    covariates: list[list[float]],
    predictions: list[list[float]],
    time_points: list[float],
    feature_index: int,
    num_intervals: int = 20,
) -> list[ALEResult]: ...
def compute_friedman_h(
    covariates: list[list[float]],
    predictions: list[float],
    feature1_index: int,
    feature2_index: int,
    n_grid: int = 20,
) -> FriedmanHResult: ...
def compute_all_pairwise_interactions(
    covariates: list[list[float]],
    predictions: list[float],
    feature_indices: list[int] | None = None,
    n_grid: int = 20,
) -> list[FriedmanHResult]: ...
def compute_feature_importance_decomposition(
    covariates: list[list[float]],
    predictions: list[float],
    n_grid: int = 20,
) -> list[FeatureImportanceResult]: ...
def compute_ice(
    covariates: list[list[float]],
    predictions: list[float],
    feature_index: int,
    n_grid: int = 50,
    centered: bool = False,
    sample_size: int | None = None,
) -> ICEResult: ...
def compute_dice(
    covariates: list[list[float]],
    predictions: list[float],
    feature_index: int,
    n_grid: int = 50,
    sample_size: int | None = None,
) -> DICEResult: ...
def compute_survival_ice(
    covariates: list[list[float]],
    survival_predictions: list[list[float]],
    time_points: list[float],
    feature_index: int,
    n_grid: int = 50,
    sample_size: int | None = None,
) -> list[ICEResult]: ...
def detect_heterogeneity(
    ice_result: ICEResult,
    threshold: float = 0.1,
) -> list[int]: ...
def cluster_ice_curves(
    ice_result: ICEResult,
    n_clusters: int = 3,
) -> list[int]: ...
def ipcw_kaplan_meier(
    time: list[float],
    status: list[int],
    x_censoring: list[float],
    n_obs: int,
    n_vars: int,
    time_points: list[float],
) -> tuple[list[float], list[float], list[float]]: ...
def g_computation(
    time: list[float],
    status: list[int],
    treatment: list[int],
    x_confounders: list[float],
    n_obs: int,
    n_vars: int,
    tau: float | None = None,
    n_bootstrap: int | None = None,
) -> GComputationResult: ...
def g_computation_survival_curves(
    time: list[float],
    status: list[int],
    treatment: list[int],
    x_confounders: list[float],
    n_obs: int,
    n_vars: int,
    time_points: list[float],
) -> tuple[list[float], list[float], list[float]]: ...
def compute_ipcw_weights(
    time: list[float],
    status: list[int],
    x_censoring: list[float],
    n_obs: int,
    n_vars: int,
    stabilized: bool = True,
    trim: float | None = None,
) -> IPCWResult: ...
def ipcw_treatment_effect(
    time: list[float],
    status: list[int],
    treatment: list[int],
    outcome: list[float],
    x_confounders: list[float],
    n_obs: int,
    n_vars: int,
    tau: float | None = None,
) -> IPCWResult: ...
def marginal_structural_model(
    time: list[float],
    status: list[int],
    treatment: list[int],
    x_outcome: list[float],
    x_propensity: list[float],
    n_obs: int,
    n_outcome_vars: int,
    n_propensity_vars: int,
    stabilized: bool = True,
    trim: float | None = None,
) -> MSMResult: ...
def compute_longitudinal_iptw(
    treatment_history: list[int],
    x_time_varying: list[float],
    n_obs: int,
    n_times: int,
    n_vars: int,
    stabilized: bool = True,
    trim: float | None = None,
) -> list[float]: ...
def target_trial_emulation(
    time: list[float],
    status: list[int],
    treatment_time: list[float | None],
    x_baseline: list[float],
    x_censoring: list[float],
    n_obs: int,
    n_vars_baseline: int,
    n_vars_censoring: int,
    config: TrialEmulationConfig,
) -> TargetTrialResult: ...
def sequential_trial_emulation(
    enrollment_times: list[float],
    treatment_times: list[float | None],
    event_times: list[float],
    event_status: list[int],
    x_baseline: list[float],
    n_obs: int,
    n_vars: int,
    trial_starts: list[float],
) -> list[TargetTrialResult]: ...
def estimate_counterfactual_survival(
    covariates: list[list[float]],
    treatment: list[int],
    time: list[float],
    event: list[int],
    time_points: list[float],
    config: CounterfactualSurvivalConfig | None = None,
) -> CounterfactualSurvivalResult: ...
def estimate_tv_survcaus(
    covariates_sequence: list[list[list[float]]],
    treatment_sequence: list[list[int]],
    _time: list[float],
    _event: list[int],
    time_points: list[float],
    config: TVSurvCausConfig | None = None,
) -> TVSurvCausResult: ...
def tmle_ate(
    covariates: list[list[float]],
    treatment: list[int],
    outcome: list[float],
    config: TMLEConfig | None = None,
) -> TMLEResult: ...
def tmle_survival(
    covariates: list[list[float]],
    treatment: list[int],
    time: list[float],
    event: list[int],
    time_points: list[float] | None = None,
    config: TMLEConfig | None = None,
) -> TMLESurvivalResult: ...
def causal_forest_survival(
    covariates: list[list[float]],
    treatment: list[int],
    time: list[float],
    event: list[int],
    time_horizon: float,
    config: CausalForestConfig | None = None,
) -> tuple[CausalForestSurvival, CausalForestResult]: ...
def iv_cox(
    time: list[float],
    event: list[int],
    treatment: list[float],
    instruments: list[float],
    covariates: list[float],
    config: IVCoxConfig,
) -> IVCoxResult: ...
def rd_survival(
    time: list[float],
    event: list[int],
    running_var: list[float],
    cutoff: float,
    treatment: list[float],
    covariates: list[float],
    config: RDSurvivalConfig,
) -> RDSurvivalResult: ...
def mediation_survival(
    time: list[float],
    event: list[int],
    treatment: list[float],
    mediator: list[float],
    covariates: list[float],
    config: MediationSurvivalConfig,
) -> MediationSurvivalResult: ...
def g_estimation_aft(
    time: list[float],
    event: list[int],
    treatment: list[float],
    covariates: list[float],
    config: GEstimationConfig,
) -> GEstimationResult: ...
def copula_censoring_model(
    time: list[float],
    event: list[int],
    censoring_indicator: list[int],
    covariates: list[float],
    config: CopulaCensoringConfig,
) -> CopulaCensoringResult: ...
def sensitivity_bounds_survival(
    time: list[float],
    event: list[int],
    treatment: list[int],
    covariates: list[float],
    tau: float,
    config: SensitivityBoundsConfig,
) -> SensitivityBoundsResult: ...
def mnar_sensitivity_survival(
    time: list[float],
    event: list[int],
    covariates: list[float],
    config: MNARSurvivalConfig,
) -> MNARSurvivalResult: ...
def double_ml_survival(
    covariates: list[list[float]],
    treatment: list[int],
    outcome: list[float],
    time: list[float],
    event: list[int],
    config: DoubleMLConfig | None = None,
) -> DoubleMLResult: ...
def double_ml_cate(
    covariates: list[list[float]],
    treatment: list[int],
    outcome: list[float],
    time: list[float],
    event: list[int],
    group_variable: list[str],
    config: DoubleMLConfig | None = None,
) -> CATEResult: ...
def detect_drift(
    reference_features: list[list[float]],
    current_features: list[list[float]],
    feature_names: list[str],
    reference_predictions: list[float] | None = None,
    current_predictions: list[float] | None = None,
    config: DriftConfig | None = None,
) -> DriftReport: ...
def monitor_performance(
    predictions: list[float],
    time: list[float],
    event: list[int],
    period_labels: list[str],
    c_index_threshold: float = 0.05,
) -> PerformanceDriftResult: ...
def create_model_card(
    model_name: str,
    model_type: str,
    version: str,
    description: str,
    intended_use: str,
    training_data_description: str,
    n_training_samples: int,
    n_events: int,
    feature_names: list[str],
    overall_performance: ModelPerformanceMetrics,
    subgroup_performance: list[SubgroupPerformance] | None = None,
    limitations: list[str] | None = None,
    ethical_considerations: list[str] | None = None,
    caveats: list[str] | None = None,
) -> ModelCard: ...
def fairness_audit(
    predictions: list[float],
    time: list[float],
    event: list[int],
    protected_attribute: str,
    group_labels: list[str],
    disparity_threshold: float = 0.1,
) -> FairnessAuditResult: ...
def qaly_calculation(
    time: list[float],
    status: list[int],
    utility_values: list[float],
    utility_times: list[float],
    discount_rate: float = 0.03,
    horizon: float | None = None,
) -> QALYResult: ...
def qaly_comparison(
    time_treated: list[float],
    status_treated: list[int],
    utility_treated: list[float],
    time_control: list[float],
    status_control: list[int],
    utility_control: list[float],
    utility_times: list[float],
    discount_rate: float = 0.03,
    horizon: float | None = None,
    n_bootstrap: int = 1000,
) -> tuple[QALYResult, QALYResult, float, float, float]: ...
def incremental_cost_effectiveness(
    qaly_treated: float,
    qaly_control: float,
    cost_treated: float,
    cost_control: float,
    wtp_threshold: float | None,
) -> tuple[float, float, bool]: ...
def qtwist_analysis(
    time: list[float],
    status: list[int],
    toxicity_start: list[float | None],
    toxicity_end: list[float | None],
    relapse_time: list[float | None],
    utility_tox: float = 0.5,
    utility_rel: float = 0.5,
    tau: float | None = None,
) -> QTWISTResult: ...
def qtwist_comparison(
    time_treated: list[float],
    status_treated: list[int],
    tox_start_treated: list[float | None],
    tox_end_treated: list[float | None],
    relapse_treated: list[float | None],
    time_control: list[float],
    status_control: list[int],
    tox_start_control: list[float | None],
    tox_end_control: list[float | None],
    relapse_control: list[float | None],
    utility_tox: float = 0.5,
    utility_rel: float = 0.5,
    tau: float | None = None,
    n_bootstrap: int = 1000,
) -> tuple[QTWISTResult, QTWISTResult, float, float, float]: ...
def qtwist_sensitivity(
    time: list[float],
    status: list[int],
    toxicity_start: list[float | None],
    toxicity_end: list[float | None],
    relapse_time: list[float | None],
    utility_tox_range: list[float],
    utility_rel_range: list[float],
    tau: float | None,
) -> list[tuple[float, float, float]]: ...
def rcll(
    survival_predictions: list[list[float]],
    prediction_times: list[float],
    event_times: list[float],
    status: list[int],
    weights: list[float] | None = None,
) -> RCLLResult: ...
def ridge_fit(
    x: list[float],
    n_obs: int,
    n_vars: int,
    time: list[float],
    status: list[int],
    penalty: RidgePenalty,
    weights: list[float] | None = None,
) -> RidgeResult: ...
def ridge_cv(
    x: list[float],
    n_obs: int,
    n_vars: int,
    time: list[float],
    status: list[int],
    theta_grid: list[float] | None = None,
    n_folds: int | None = None,
) -> tuple[float, list[float]]: ...
def gradient_boost_survival(
    x: list[float],
    n: int,
    p: int,
    time: list[float],
    status: list[int],
    config: GradientBoostSurvivalConfig | None = None,
) -> GradientBoostSurvival: ...
def survival_forest(
    x: list[float],
    n: int,
    p: int,
    time: list[float],
    status: list[int],
    config: SurvivalForestConfig | None = None,
) -> SurvivalForest: ...
def deep_surv(
    x: list[float],
    n: int,
    p: int,
    time: list[float],
    status: list[int],
    config: DeepSurvConfig | None = None,
) -> DeepSurv: ...
def load_lung() -> dict[str, list[Any]]: ...
def load_aml() -> dict[str, list[Any]]: ...
def load_veteran() -> dict[str, list[Any]]: ...
def load_ovarian() -> dict[str, list[Any]]: ...
def load_colon() -> dict[str, list[Any]]: ...
def load_pbc() -> dict[str, list[Any]]: ...
def load_cgd() -> dict[str, list[Any]]: ...
def load_bladder() -> dict[str, list[Any]]: ...
def load_heart() -> dict[str, list[Any]]: ...
def load_kidney() -> dict[str, list[Any]]: ...
def load_rats() -> dict[str, list[Any]]: ...
def load_stanford2() -> dict[str, list[Any]]: ...
def load_udca() -> dict[str, list[Any]]: ...
def load_myeloid() -> dict[str, list[Any]]: ...
def load_flchain() -> dict[str, list[Any]]: ...
def load_transplant() -> dict[str, list[Any]]: ...
def load_mgus() -> dict[str, list[Any]]: ...
def load_mgus2() -> dict[str, list[Any]]: ...
def load_diabetic() -> dict[str, list[Any]]: ...
def load_retinopathy() -> dict[str, list[Any]]: ...
def load_gbsg() -> dict[str, list[Any]]: ...
def load_rotterdam() -> dict[str, list[Any]]: ...
def load_logan() -> dict[str, list[Any]]: ...
def load_nwtco() -> dict[str, list[Any]]: ...
def load_solder() -> dict[str, list[Any]]: ...
def load_tobin() -> dict[str, list[Any]]: ...
def load_rats2() -> dict[str, list[Any]]: ...
def load_nafld() -> dict[str, list[Any]]: ...
def load_cgd0() -> dict[str, list[Any]]: ...
def load_pbcseq() -> dict[str, list[Any]]: ...
def load_hoel() -> dict[str, list[Any]]: ...
def load_myeloma() -> dict[str, list[Any]]: ...
def load_rhdnase() -> dict[str, list[Any]]: ...
