# survival

[![Crates.io](https://img.shields.io/crates/v/survival.svg)](https://crates.io/crates/survival)
[![PyPI version](https://img.shields.io/pypi/v/survival.svg)](https://pypi.org/project/survival/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance survival analysis library written in Rust, with a Python API powered by [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

## Features

- Core survival analysis routines
- Cox proportional hazards models with frailty
- Kaplan-Meier and Aalen-Johansen (multi-state) survival curves
- Nelson-Aalen estimator
- Parametric accelerated failure time models
- Fine-Gray competing risks model
- Penalized splines (P-splines) for smooth covariate effects
- Concordance index calculations
- Person-years calculations
- Score calculations for survival models
- Residual analysis (martingale, Schoenfeld, score residuals)
- Bootstrap confidence intervals
- Cross-validation for model assessment
- Statistical tests (log-rank, likelihood ratio, Wald, score, proportional hazards)
- Sample size and power calculations
- RMST (Restricted Mean Survival Time) analysis
- Landmark analysis
- Calibration and risk stratification
- Time-dependent AUC
- Conditional logistic regression
- Time-splitting utilities

## Installation

### From PyPI (Recommended)

```sh
pip install survival
```

### From Source

#### Prerequisites

- Python 3.9+
- Rust (see [rustup.rs](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin)

Install maturin:
```sh
pip install maturin
```

#### Build and Install

Build the Python wheel:
```sh
maturin build --release
```

Install the wheel:
```sh
pip install target/wheels/survival-*.whl
```

For development:
```sh
maturin develop
```

## Usage

### Aalen's Additive Regression Model

```python
from survival import AaregOptions, aareg

data = [
    [1.0, 0.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 0.0, 2.5],
]
variable_names = ["time", "event", "covariate1"]

# Create options with required parameters (formula, data, variable_names)
options = AaregOptions(
    formula="time + event ~ covariate1",
    data=data,
    variable_names=variable_names,
)

# Optional: modify default values via setters
# options.weights = [1.0, 1.0, 1.0]
# options.qrtol = 1e-8
# options.dfbeta = True

result = aareg(options)
print(result)
```

### Penalized Splines (P-splines)

```python
from survival import PSpline

x = [0.1 * i for i in range(100)]
pspline = PSpline(
    x=x,
    df=10,
    theta=1.0,
    eps=1e-6,
    method="GCV",
    boundary_knots=(0.0, 10.0),
    intercept=True,
    penalty=True,
)
pspline.fit()
```

### Concordance Index

```python
from survival import perform_concordance1_calculation

time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
weights = [1.0, 1.0, 1.0, 1.0, 1.0]
indices = [0, 1, 2, 3, 4]
ntree = 5

result = perform_concordance1_calculation(time_data, weights, indices, ntree)
print(f"Concordance index: {result['concordance_index']}")
```

### Cox Regression with Frailty

```python
from survival import perform_cox_regression_frailty

result = perform_cox_regression_frailty(
    time_data=[...],
    status_data=[...],
    covariates=[...],
    # ... other parameters
)
```

### Person-Years Calculation

```python
from survival import perform_pyears_calculation

result = perform_pyears_calculation(
    time_data=[...],
    weights=[...],
    # ... other parameters
)
```

### Kaplan-Meier Survival Curves

```python
from survival import survfitkm, SurvFitKMOutput

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]  # 1 = event, 0 = censored
weights = [1.0] * len(time)  # Optional: equal weights

result = survfitkm(
    time=time,
    status=status,
    weights=weights,
    entry_times=None,  # Optional: entry times for left-truncation
    position=None,     # Optional: position flags
    reverse=False,     # Optional: reverse time order
    computation_type=0 # Optional: computation type
)

print(f"Time points: {result.time}")
print(f"Survival estimates: {result.estimate}")
print(f"Standard errors: {result.std_err}")
print(f"Number at risk: {result.n_risk}")
```

### Fine-Gray Competing Risks Model

```python
from survival import finegray, FineGrayOutput

# Example competing risks data
tstart = [0.0, 0.0, 0.0, 0.0]
tstop = [1.0, 2.0, 3.0, 4.0]
ctime = [0.5, 1.5, 2.5, 3.5]  # Cut points
cprob = [0.1, 0.2, 0.3, 0.4]  # Cumulative probabilities
extend = [True, True, False, False]  # Whether to extend intervals
keep = [True, True, True, True]      # Which cut points to keep

result = finegray(
    tstart=tstart,
    tstop=tstop,
    ctime=ctime,
    cprob=cprob,
    extend=extend,
    keep=keep
)

print(f"Row indices: {result.row}")
print(f"Start times: {result.start}")
print(f"End times: {result.end}")
print(f"Weights: {result.wt}")
```

### Parametric Survival Regression (Accelerated Failure Time Models)

```python
from survival import survreg, SurvivalFit, DistributionType

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]  # 1 = event, 0 = censored
covariates = [
    [1.0, 2.0],
    [1.5, 2.5],
    [2.0, 3.0],
    [2.5, 3.5],
    [3.0, 4.0],
    [3.5, 4.5],
    [4.0, 5.0],
    [4.5, 5.5],
]

# Fit parametric survival model
result = survreg(
    time=time,
    status=status,
    covariates=covariates,
    weights=None,          # Optional: observation weights
    offsets=None,          # Optional: offset values
    initial_beta=None,     # Optional: initial coefficient values
    strata=None,           # Optional: stratification variable
    distribution="weibull",  # "extreme_value", "logistic", "gaussian", "weibull", or "lognormal"
    max_iter=20,          # Optional: maximum iterations
    eps=1e-5,             # Optional: convergence tolerance
    tol_chol=1e-9,        # Optional: Cholesky tolerance
)

print(f"Coefficients: {result.coefficients}")
print(f"Log-likelihood: {result.log_likelihood}")
print(f"Iterations: {result.iterations}")
print(f"Variance matrix: {result.variance_matrix}")
print(f"Convergence flag: {result.convergence_flag}")
```

### Cox Proportional Hazards Model

```python
from survival import CoxPHModel, Subject

# Create a Cox PH model
model = CoxPHModel()

# Or create with data
covariates = [[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]]
event_times = [1.0, 2.0, 3.0]
censoring = [1, 1, 0]  # 1 = event, 0 = censored

model = CoxPHModel.new_with_data(covariates, event_times, censoring)

# Fit the model
model.fit(n_iters=10)

# Get results
print(f"Baseline hazard: {model.baseline_hazard}")
print(f"Risk scores: {model.risk_scores}")
print(f"Coefficients: {model.get_coefficients()}")

# Predict on new data
new_covariates = [[1.0, 2.0], [2.0, 3.0]]
predictions = model.predict(new_covariates)
print(f"Predictions: {predictions}")

# Calculate Brier score
brier = model.brier_score()
print(f"Brier score: {brier}")

# Compute survival curves for new covariates
new_covariates = [[1.0, 2.0], [2.0, 3.0]]
time_points = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Optional: specific time points
times, survival_curves = model.survival_curve(new_covariates, time_points)
print(f"Time points: {times}")
print(f"Survival curves: {survival_curves}")  # One curve per covariate set

# Create and add subjects
subject = Subject(
    id=1,
    covariates=[1.0, 2.0],
    is_case=True,
    is_subcohort=True,
    stratum=0
)
model.add_subject(subject)
```

### Cox Martingale Residuals

```python
from survival import coxmart

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 0, 1, 0, 1, 1, 0]  # 1 = event, 0 = censored
score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Risk scores

# Calculate martingale residuals
residuals = coxmart(
    time=time,
    status=status,
    score=score,
    weights=None,      # Optional: observation weights
    strata=None,       # Optional: stratification variable
    method=0,          # Optional: method (0 = Breslow, 1 = Efron)
)

print(f"Martingale residuals: {residuals}")
```

### Survival Difference Tests (Log-Rank Test)

```python
from survival import survdiff2, SurvDiffResult

# Example: Compare survival between two groups
time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
status = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
group = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # Group 1 and Group 2

# Perform log-rank test (rho=0 for standard log-rank)
result = survdiff2(
    time=time,
    status=status,
    group=group,
    strata=None,  # Optional: stratification variable
    rho=0.0,      # 0.0 = log-rank, 1.0 = Wilcoxon, other = generalized
)

print(f"Observed events: {result.observed}")
print(f"Expected events: {result.expected}")
print(f"Chi-squared statistic: {result.chi_squared}")
print(f"Degrees of freedom: {result.degrees_of_freedom}")
print(f"Variance matrix: {result.variance}")
```

### Built-in Datasets

The library includes 30 classic survival analysis datasets:

```python
from survival import load_lung, load_aml, load_veteran

# Load the lung cancer dataset
lung = load_lung()
print(f"Columns: {lung['columns']}")
print(f"Number of rows: {len(lung['data'])}")

# Load the acute myelogenous leukemia dataset
aml = load_aml()

# Load the veteran's lung cancer dataset
veteran = load_veteran()
```

**Available datasets:**
- `load_lung()` - NCCTG Lung Cancer Data
- `load_aml()` - Acute Myelogenous Leukemia Survival Data
- `load_veteran()` - Veterans' Administration Lung Cancer Study
- `load_ovarian()` - Ovarian Cancer Survival Data
- `load_colon()` - Colon Cancer Data
- `load_pbc()` - Primary Biliary Cholangitis Data
- `load_cgd()` - Chronic Granulomatous Disease Data
- `load_bladder()` - Bladder Cancer Recurrences
- `load_heart()` - Stanford Heart Transplant Data
- `load_kidney()` - Kidney Catheter Data
- `load_rats()` - Rat Treatment Data
- `load_stanford2()` - Stanford Heart Transplant Data (Extended)
- `load_udca()` - UDCA Clinical Trial Data
- `load_myeloid()` - Acute Myeloid Leukemia Clinical Trial
- `load_flchain()` - Free Light Chain Data
- `load_transplant()` - Liver Transplant Data
- `load_mgus()` - Monoclonal Gammopathy Data
- `load_mgus2()` - Monoclonal Gammopathy Data (Updated)
- `load_diabetic()` - Diabetic Retinopathy Data
- `load_retinopathy()` - Retinopathy Data
- `load_gbsg()` - German Breast Cancer Study Group Data
- `load_rotterdam()` - Rotterdam Tumor Bank Data
- `load_logan()` - Logan Unemployment Data
- `load_nwtco()` - National Wilms Tumor Study Data
- `load_solder()` - Solder Joint Data
- `load_tobin()` - Tobin's Tobit Data
- `load_rats2()` - Rat Tumorigenesis Data
- `load_nafld()` - Non-Alcoholic Fatty Liver Disease Data
- `load_cgd0()` - CGD Baseline Data
- `load_pbcseq()` - PBC Sequential Data

## API Reference

### Classes

**Core Models:**
- `AaregOptions`: Configuration options for Aalen's additive regression model
- `PSpline`: Penalized spline class for smooth covariate effects
- `CoxPHModel`: Cox proportional hazards model class
- `Subject`: Subject data structure for Cox PH models
- `ConditionalLogisticRegression`: Conditional logistic regression model
- `ClogitDataSet`: Dataset for conditional logistic regression

**Survival Curves:**
- `SurvFitKMOutput`: Output from Kaplan-Meier survival curve fitting
- `SurvfitKMOptions`: Options for Kaplan-Meier fitting
- `KaplanMeierConfig`: Configuration for Kaplan-Meier
- `SurvFitAJ`: Output from Aalen-Johansen survival curve fitting
- `NelsonAalenResult`: Output from Nelson-Aalen estimator
- `StratifiedKMResult`: Output from stratified Kaplan-Meier

**Parametric Models:**
- `SurvivalFit`: Output from parametric survival regression
- `SurvregConfig`: Configuration for parametric survival regression
- `DistributionType`: Distribution types for parametric models (extreme_value, logistic, gaussian, weibull, lognormal)
- `FineGrayOutput`: Output from Fine-Gray competing risks model

**Statistical Tests:**
- `SurvDiffResult`: Output from survival difference tests
- `LogRankResult`: Output from log-rank test
- `TrendTestResult`: Output from trend tests
- `TestResult`: General test result output
- `ProportionalityTest`: Output from proportional hazards test
- `SurvObrienResult`: Output from O'Brien transformation

**Validation:**
- `BootstrapResult`: Output from bootstrap confidence interval calculations
- `CVResult`: Output from cross-validation
- `CalibrationResult`: Output from calibration analysis
- `PredictionResult`: Output from prediction functions
- `RiskStratificationResult`: Output from risk stratification
- `TdAUCResult`: Output from time-dependent AUC calculation

**RMST and Survival Metrics:**
- `RMSTResult`: Output from RMST calculation
- `RMSTComparisonResult`: Output from RMST comparison between groups
- `MedianSurvivalResult`: Output from median survival calculation
- `CumulativeIncidenceResult`: Output from cumulative incidence calculation
- `NNTResult`: Number needed to treat result

**Landmark Analysis:**
- `LandmarkResult`: Output from landmark analysis
- `ConditionalSurvivalResult`: Output from conditional survival calculation
- `HazardRatioResult`: Output from hazard ratio calculation
- `SurvivalAtTimeResult`: Output from survival at specific times
- `LifeTableResult`: Output from life table calculation

**Power and Sample Size:**
- `SampleSizeResult`: Output from sample size calculations
- `AccrualResult`: Output from accrual calculations

**Utilities:**
- `CoxCountOutput`: Output from Cox counting functions
- `SplitResult`: Output from time-splitting
- `CondenseResult`: Output from data condensing
- `Surv2DataResult`: Output from survival-to-data conversion
- `TimelineResult`: Output from timeline conversion
- `IntervalResult`: Output from interval calculations
- `LinkFunctionParams`: Link function parameters
- `CchMethod`: Case-cohort method specification
- `CohortData`: Cohort data structure

### Functions

**Model Fitting:**
- `aareg(options)`: Fit Aalen's additive regression model
- `survreg(...)`: Fit parametric accelerated failure time models
- `perform_cox_regression_frailty(...)`: Fit Cox proportional hazards model with frailty

**Survival Curves:**
- `survfitkm(...)`: Fit Kaplan-Meier survival curves
- `survfitkm_with_options(...)`: Fit Kaplan-Meier with configuration options
- `survfitaj(...)`: Fit Aalen-Johansen survival curves (multi-state)
- `nelson_aalen_estimator(...)`: Calculate Nelson-Aalen estimator
- `stratified_kaplan_meier(...)`: Calculate stratified Kaplan-Meier curves
- `agsurv4(...)`: Anderson-Gill survival calculations (version 4)
- `agsurv5(...)`: Anderson-Gill survival calculations (version 5)

**Statistical Tests:**
- `survdiff2(...)`: Perform survival difference tests (log-rank, Wilcoxon, etc.)
- `logrank_test(...)`: Perform log-rank test
- `fleming_harrington_test(...)`: Perform Fleming-Harrington weighted test
- `logrank_trend(...)`: Perform log-rank trend test
- `lrt_test(...)`: Likelihood ratio test
- `wald_test_py(...)`: Wald test
- `score_test_py(...)`: Score test
- `ph_test(...)`: Proportional hazards assumption test
- `survobrien(...)`: O'Brien transformation for survival data

**Residuals:**
- `coxmart(...)`: Calculate Cox martingale residuals
- `agmart(...)`: Calculate Anderson-Gill martingale residuals
- `schoenfeld_residuals(...)`: Calculate Schoenfeld residuals
- `cox_score_residuals(...)`: Calculate Cox score residuals

**Concordance:**
- `perform_concordance1_calculation(...)`: Calculate concordance index (version 1)
- `perform_concordance3_calculation(...)`: Calculate concordance index (version 3)
- `perform_concordance_calculation(...)`: Calculate concordance index (version 5)
- `compute_concordance(...)`: General concordance calculation

**Validation:**
- `bootstrap_cox_ci(...)`: Bootstrap confidence intervals for Cox models
- `bootstrap_survreg_ci(...)`: Bootstrap confidence intervals for parametric models
- `cv_cox_concordance(...)`: Cross-validation for Cox model concordance
- `cv_survreg_loglik(...)`: Cross-validation for parametric model log-likelihood
- `calibration(...)`: Model calibration assessment
- `predict_cox(...)`: Predictions from Cox models
- `risk_stratification(...)`: Risk group stratification
- `td_auc(...)`: Time-dependent AUC calculation
- `brier(...)`: Calculate Brier score
- `integrated_brier(...)`: Calculate integrated Brier score

**RMST and Survival Metrics:**
- `rmst(...)`: Calculate restricted mean survival time
- `rmst_comparison(...)`: Compare RMST between groups
- `survival_quantile(...)`: Calculate survival quantiles (median, etc.)
- `cumulative_incidence(...)`: Calculate cumulative incidence
- `number_needed_to_treat(...)`: Calculate NNT

**Landmark Analysis:**
- `landmark_analysis(...)`: Perform landmark analysis
- `landmark_analysis_batch(...)`: Perform batch landmark analysis at multiple time points
- `conditional_survival(...)`: Calculate conditional survival
- `hazard_ratio(...)`: Calculate hazard ratios
- `survival_at_times(...)`: Calculate survival at specific time points
- `life_table(...)`: Generate life table

**Power and Sample Size:**
- `sample_size_survival(...)`: Calculate required sample size
- `sample_size_survival_freedman(...)`: Sample size using Freedman's method
- `power_survival(...)`: Calculate statistical power
- `expected_events(...)`: Calculate expected number of events

**Utilities:**
- `finegray(...)`: Fine-Gray competing risks model data preparation
- `perform_pyears_calculation(...)`: Calculate person-years of observation
- `perform_pystep_calculation(...)`: Perform step calculations
- `perform_pystep_simple_calculation(...)`: Perform simple step calculations
- `perform_score_calculation(...)`: Calculate score statistics
- `perform_agscore3_calculation(...)`: Calculate score statistics (version 3)
- `survsplit(...)`: Split survival data at specified times
- `survcondense(...)`: Condense survival data by collapsing adjacent intervals
- `surv2data(...)`: Convert survival objects to data format
- `to_timeline(...)`: Convert data to timeline format
- `from_timeline(...)`: Convert from timeline format to intervals
- `tmerge(...)`: Merge time-dependent covariates
- `tmerge2(...)`: Merge time-dependent covariates (version 2)
- `tmerge3(...)`: Merge time-dependent covariates (version 3)
- `collapse(...)`: Collapse survival data
- `coxcount1(...)`: Cox counting process calculations
- `coxcount2(...)`: Cox counting process calculations (version 2)
- `agexact(...)`: Exact Anderson-Gill calculations
- `norisk(...)`: No-risk calculations
- `cipoisson(...)`: Poisson confidence intervals
- `cipoisson_exact(...)`: Exact Poisson confidence intervals
- `cipoisson_anscombe(...)`: Anscombe Poisson confidence intervals
- `cox_callback(...)`: Cox model callback for iterative fitting

## PSpline Options

The `PSpline` class provides penalized spline smoothing:

**Constructor Parameters:**
- `x`: Covariate vector (list of floats)
- `df`: Degrees of freedom (integer)
- `theta`: Roughness penalty (float)
- `eps`: Accuracy for degrees of freedom (float)
- `method`: Penalty method for tuning parameter selection. Supported methods:
  - `"GCV"` - Generalized Cross-Validation
  - `"UBRE"` - Unbiased Risk Estimator
  - `"REML"` - Restricted Maximum Likelihood
  - `"AIC"` - Akaike Information Criterion
  - `"BIC"` - Bayesian Information Criterion
- `boundary_knots`: Tuple of (min, max) for the spline basis
- `intercept`: Whether to include an intercept in the basis
- `penalty`: Whether to apply the penalty

**Methods:**
- `fit()`: Fit the spline model, returns coefficients
- `predict(new_x)`: Predict values at new x points

**Properties:**
- `coefficients`: Fitted coefficients (None if not fitted)
- `fitted`: Whether the model has been fitted
- `df`: Degrees of freedom
- `eps`: Convergence tolerance

## Development

Build the Rust library:
```sh
cargo build
```

Run tests:
```sh
cargo test
```

Format code:
```sh
cargo fmt
```

The codebase is organized with:
- Core routines in `src/`
- Tests and examples in `test/`
- Python bindings using PyO3

## Dependencies

- [PyO3](https://github.com/PyO3/pyo3) - Python bindings
- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays
- [faer](https://github.com/sarah-ek/faer-rs) - Pure-Rust linear algebra
- [itertools](https://github.com/rust-itertools/itertools) - Iterator utilities
- [rayon](https://github.com/rayon-rs/rayon) - Parallel computation

## Compatibility

- This build is for Python only. R/extendr bindings are currently disabled.
- macOS users: Ensure you are using the correct Python version and have Homebrew-installed Python if using Apple Silicon.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
