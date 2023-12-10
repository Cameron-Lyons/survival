// Aalen’s additive regression model for censored data

extern crate pyo3;
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::HashMap;

enum NaAction {
    Fail,
    Exclude,
}

#[pyclass]
struct AaregOptions {
    formula: String,                       // Formula as a string
    data: Array2<f64>,                     // 2D array for the dataset
    weights: Option<Vec<f64>>,             // Optional weights
    subset: Option<Vec<usize>>,            // Optional subset of data indices
    na_action: Option<NaAction>,           // Action for NA values
    qrtol: f64,                            // Tolerance for singularity detection
    nmin: Option<usize>,                   // Minimum number of observations
    dfbeta: bool,                          // Whether to compute dfbeta residuals
    taper: f64,                            // Taper parameter
    test: Vec<String>,                     // List of tests to perform
    cluster: Option<HashMap<String, i32>>, // Optional clustering
    model: bool,                           // Whether to include the model frame in the output
    x: bool,                               // Whether to include the matrix of predictors
    y: bool,                               // Whether to include the response vector
}

impl AaregOptions {
    fn new(formula: String, data: Array2<f64>) -> AaregOptions {
        AaregOptions {
            formula,
            data,
            weights: None,
            subset: None,
            na_action: None,
            qrtol: 1e-07,
            nmin: None,
            dfbeta: false,
            taper: 1.0,
            test: vec![],
            cluster: None,
            model: false,
            x: false,
            y: false,
        }
    }
}

#[pyclass]
struct Surv {
    time: Vec<f64>,
    event: Vec<u8>,
}

impl Surv {
    /// Constructs a new `Surv` instance.
    /// # Arguments
    /// * `time` - A vector of event times.
    /// * `event` - A vector indicating event occurrence (1 for observed, 0 for censored).
    /// # Panics
    /// Panics if `time` and `event` have different lengths.
    pub fn new(time: Vec<f64>, event: Vec<u8>) -> Self {
        assert_eq!(
            time.len(),
            event.len(),
            "Time and event vectors must be of the same length."
        );

        Surv { time, event }
    }
}

#[pyclass]
struct AaregResult {
    // Estimated coefficients for each predictor variable
    coefficients: Vec<f64>,
    // Standard errors for the estimated coefficients
    standard_errors: Vec<f64>,
    // Confidence intervals for the coefficients
    confidence_intervals: Vec<ConfidenceInterval>,
    // P-values for testing the hypothesis that each coefficient is zero
    p_values: Vec<f64>,
    // The overall goodness-of-fit statistic for the model
    goodness_of_fit: f64,
    // Optional: Information about the model fit, convergence details, etc.
    fit_details: Option<FitDetails>,
    // Optional: Residuals from the model
    residuals: Option<Vec<f64>>,
    // Optional: Additional diagnostic information
    diagnostics: Option<Diagnostics>,
}

#[pyclass]
struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

#[pyclass]
struct FitDetails {
    // Number of iterations taken by the fitting algorithm
    iterations: u32,
    // Whether the fitting algorithm successfully converged
    converged: bool,
    // The final value of the objective function (e.g., log-likelihood, residual sum of squares)
    final_objective_value: f64,
    // The threshold for convergence, showing how close the algorithm needs to get to the solution
    convergence_threshold: f64,
    // Optional: The rate of change of the objective function in the last iteration
    change_in_objective: Option<f64>,
    // Optional: The maximum number of iterations allowed
    max_iterations: Option<u32>,
    // Optional: Information about the optimization method used (e.g., gradient descent, Newton-Raphson)
    optimization_method: Option<String>,
    // Any warnings or notes generated during the fitting process
    warnings: Vec<String>,
}

#[pyclass]
struct Diagnostics {
    // DFBetas for each predictor variable to assess their influence on the model
    dfbetas: Option<Vec<f64>>,
    // Cook's distance for each observation, indicating its influence on the fitted values
    cooks_distance: Option<Vec<f64>>,
    // Leverage values for each observation, indicating their influence on the model fit
    leverage: Option<Vec<f64>>,
    // Deviance residuals, which are useful for identifying outliers
    deviance_residuals: Option<Vec<f64>>,
    // Martingale residuals, which are particularly relevant in survival analysis
    martingale_residuals: Option<Vec<f64>>,
    // Schoenfeld residuals, useful for checking the proportional hazards assumption in survival models
    schoenfeld_residuals: Option<Vec<f64>>,
    // Score residuals, useful for diagnostic checks in various regression models
    score_residuals: Option<Vec<f64>>,
    // Optional: Additional model-specific diagnostic measures
    additional_measures: Option<Vec<f64>>,
}

enum AaregError {
    DataError(String),        // e.g., "Data dimensions mismatch"
    FormulaError(String),     // e.g., "Formula parsing error"
    WeightsError(String),     // e.g., "Weights length does not match number of observations"
    CalculationError(String), // e.g., "Singular matrix encountered in calculations"
    InputError(String),       // e.g., "Invalid input: negative values in data"
    InternalError(String),    // e.g., "Unexpected internal error"
}

#[pymethods]
impl std::fmt::Debug for AaregError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AaregError::DataError(msg) => write!(f, "Data Error: {}", msg),
            AaregError::FormulaError(msg) => write!(f, "Formula Error: {}", msg),
            AaregError::WeightsError(msg) => write!(f, "Weights Error: {}", msg),
            AaregError::CalculationError(msg) => write!(f, "Calculation Error: {}", msg),
            AaregError::InputError(msg) => write!(f, "Input Error: {}", msg),
            AaregError::InternalError(msg) => write!(f, "Internal Error: {}", msg),
        }
    }
}

#[pymethods]
fn aareg(options: AaregOptions) -> Result<AaregResult, AaregError> {
    let (response, covariates) = parse_formula(&options.formula)?;
    let subset_data = apply_subset(&options.data, &options.subset)?;
    let weighted_data = apply_weights(&subset_data, &options.weights)?;
    let filtered_data = handle_missing_data(weighted_data, options.na_action)?;
    let (y, x) = prepare_data_for_regression(&filtered_data, &response, &covariates)?;
    let regression_result = perform_aalen_regression(&y, &x, &options)?;
    let processed_result = post_process_results(regression_result, &options)?;

    Ok(processed_result)
}

#[pymethods]
fn parse_formula(formula: &str) -> Result<(String, Vec<String>), AaregError> {
    let mut formula_parts = formula.split("~");
    let response = formula_parts.next().unwrap().trim().to_string();
    let covariates = formula_parts
        .next()
        .unwrap()
        .trim()
        .split("+")
        .map(|s| s.trim().to_string())
        .collect();
    Ok((response, covariates))
}

#[pymethods]
fn apply_subset(
    data: &Array2<f64>,
    subset: &Option<Vec<usize>>,
) -> Result<Array2<f64>, AaregError> {
    match subset {
        Some(s) => {
            let subset_data = data.select(ndarray::Axis(0), s);
            Ok(subset_data)
        }
        None => Ok(data.to_owned()),
    }
}

#[pymethods]
fn apply_weights(
    data: &Array2<f64>,
    weights: &Option<Vec<f64>>,
) -> Result<Array2<f64>, AaregError> {
    match weights {
        Some(w) => {
            if w.len() != data.nrows() {
                Err(AaregError::WeightsError(
                    "Weights length does not match number of observations".to_string(),
                ))
            } else {
                let weighted_data = data * w;
                Ok(weighted_data)
            }
        }
        None => Ok(data.to_owned()),
    }
}

#[pymethods]
fn handle_missing_data(
    data: Array2<f64>,
    na_action: Option<NaAction>,
) -> Result<Array2<f64>, AaregError> {
    match na_action {
        Some(NaAction::Fail) => {
            if data.iter().any(|&x| x.is_nan()) {
                Err(AaregError::InputError(
                    "Invalid input: missing values in data".to_string(),
                ))
            } else {
                Ok(data)
            }
        }
        Some(NaAction::Exclude) => {
            let filtered_data = data.select(ndarray::Axis(1), |&x| !x.is_nan());
            Ok(filtered_data)
        }
        None => Ok(data),
    }
}

#[pymethods]
fn prepare_data_for_regression(
    data: &Array2<f64>,
    response: &str,
    covariates: &[String],
) -> Result<(Array2<f64>, Array2<f64>), AaregError> {
    let response_index = match data.column_names().iter().position(|s| s == response) {
        Some(i) => i,
        None => {
            return Err(AaregError::FormulaError(
                "Response variable not found in data".to_string(),
            ))
        }
    };
    let covariate_indices = covariates
        .iter()
        .map(|c| match data.column_names().iter().position(|s| s == c) {
            Some(i) => Ok(i),
            None => Err(AaregError::FormulaError(
                "Covariate not found in data".to_string(),
            )),
        })
        .collect::<Result<Vec<usize>, AaregError>>()?;
    let y = data.select(ndarray::Axis(1), response_index);
    let x = data.select(ndarray::Axis(1), &covariate_indices);
    Ok((y, x))
}

#[pymethods]
fn perform_aalen_regression(
    y: &Array2<f64>,
    x: &Array2<f64>,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    let mut coefficients = vec![];
    let mut standard_errors = vec![];
    let mut confidence_intervals = vec![];
    let mut p_values = vec![];
    let mut goodness_of_fit = 0.0;
    let mut fit_details = None;
    let mut residuals = None;
    let mut diagnostics = None;
    Ok(AaregResult {
        coefficients,
        standard_errors,
        confidence_intervals,
        p_values,
        goodness_of_fit,
        fit_details,
        residuals,
        diagnostics,
    })
}

#[pymethods]
fn post_process_results(
    regression_result: AaregResult,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    let mut processed_result = regression_result;
    if options.dfbeta {
        let dfbetas = calculate_dfbetas(&regression_result);
        processed_result.diagnostics = Some(Diagnostics {
            dfbetas: Some(dfbetas),
            cooks_distance: None,
            leverage: None,
            deviance_residuals: None,
            martingale_residuals: None,
            schoenfeld_residuals: None,
            score_residuals: None,
            additional_measures: None,
        });
    }
    Ok(processed_result)
}

#[pymodule]
fn pyAaregError(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AaregError>()?;
    Ok(())
}
