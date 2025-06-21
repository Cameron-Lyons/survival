// src/lib.rs


use ndarray::{Array1, Array2, Axis};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;

#[pyclass]
#[derive(Clone)]
struct AaregOptions {
    #[pyo3(get, set)]
    formula: String,                   // Formula as a string
    #[pyo3(get, set)]
    data: Vec<Vec<f64>>,               // 2D array for the dataset
    #[pyo3(get, set)]
    variable_names: Vec<String>,       // Names of the variables (columns in data)
    #[pyo3(get, set)]
    weights: Option<Vec<f64>>,         // Optional weights
    #[pyo3(get, set)]
    subset: Option<Vec<usize>>,        // Optional subset of data indices
    #[pyo3(get, set)]
    na_action: Option<String>,         // Action for NA values ("Fail" or "Exclude")
    #[pyo3(get, set)]
    qrtol: f64,                        // Tolerance for singularity detection
    #[pyo3(get, set)]
    nmin: Option<usize>,               // Minimum number of observations
    #[pyo3(get, set)]
    dfbeta: bool,                      // Whether to compute dfbeta residuals
    #[pyo3(get, set)]
    taper: f64,                        // Taper parameter
    #[pyo3(get, set)]
    test: Vec<String>,                 // List of tests to perform
    #[pyo3(get, set)]
    cluster: Option<HashMap<String, i32>>, // Optional clustering
    #[pyo3(get, set)]
    model: bool,                       // Whether to include the model frame in the output
    #[pyo3(get, set)]
    x: bool,                           // Whether to include the matrix of predictors
    #[pyo3(get, set)]
    y: bool,                           // Whether to include the response vector
}

#[pymethods]
impl AaregOptions {
    /// Creates a new `AaregOptions` instance.
    #[new]
    fn new(formula: String, data: Vec<Vec<f64>>, variable_names: Vec<String>) -> Self {
        AaregOptions {
            formula,
            data,
            variable_names,
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

/// Represents the result of the Aalen's additive regression.
#[pyclass]
#[derive(Clone)]
struct AaregResult {
    #[pyo3(get, set)]
    coefficients: Vec<f64>,                  // Estimated coefficients
    #[pyo3(get, set)]
    standard_errors: Vec<f64>,               // Standard errors
    #[pyo3(get, set)]
    confidence_intervals: Vec<ConfidenceInterval>, // Confidence intervals
    #[pyo3(get, set)]
    p_values: Vec<f64>,                      // P-values
    #[pyo3(get, set)]
    goodness_of_fit: f64,                    // Goodness-of-fit statistic
    #[pyo3(get, set)]
    fit_details: Option<FitDetails>,         // Fit details
    #[pyo3(get, set)]
    residuals: Option<Vec<f64>>,             // Residuals
    #[pyo3(get, set)]
    diagnostics: Option<Diagnostics>,        // Diagnostics
}

/// Confidence interval for a coefficient.
#[pyclass]
#[derive(Clone)]
struct ConfidenceInterval {
    #[pyo3(get, set)]
    lower_bound: f64,
    #[pyo3(get, set)]
    upper_bound: f64,
}

/// Details about the fit of the model.
#[pyclass]
#[derive(Clone)]
struct FitDetails {
    #[pyo3(get, set)]
    iterations: u32,                  // Number of iterations
    #[pyo3(get, set)]
    converged: bool,                  // Convergence status
    #[pyo3(get, set)]
    final_objective_value: f64,       // Final objective value
    #[pyo3(get, set)]
    convergence_threshold: f64,       // Convergence threshold
    #[pyo3(get, set)]
    change_in_objective: Option<f64>, // Change in objective
    #[pyo3(get, set)]
    max_iterations: Option<u32>,      // Max iterations allowed
    #[pyo3(get, set)]
    optimization_method: Option<String>, // Optimization method used
    #[pyo3(get, set)]
    warnings: Vec<String>,            // Warnings during fitting
}

/// Diagnostic information from the model.
#[pyclass]
#[derive(Clone)]
struct Diagnostics {
    #[pyo3(get, set)]
    dfbetas: Option<Vec<f64>>,            // DFBetas
    #[pyo3(get, set)]
    cooks_distance: Option<Vec<f64>>,     // Cook's distance
    #[pyo3(get, set)]
    leverage: Option<Vec<f64>>,           // Leverage values
    #[pyo3(get, set)]
    deviance_residuals: Option<Vec<f64>>, // Deviance residuals
    #[pyo3(get, set)]
    martingale_residuals: Option<Vec<f64>>, // Martingale residuals
    #[pyo3(get, set)]
    schoenfeld_residuals: Option<Vec<f64>>, // Schoenfeld residuals
    #[pyo3(get, set)]
    score_residuals: Option<Vec<f64>>,    // Score residuals
    #[pyo3(get, set)]
    additional_measures: Option<Vec<f64>>, // Additional measures
}

/// Custom error type for the Aareg model.
#[derive(Debug)]
enum AaregError {
    DataError(String),
    FormulaError(String),
    WeightsError(String),
    CalculationError(String),
    InputError(String),
    InternalError(String),
    GenericError(String),
}

impl fmt::Display for AaregError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AaregError::DataError(msg) => write!(f, "Data Error: {}", msg),
            AaregError::FormulaError(msg) => write!(f, "Formula Error: {}", msg),
            AaregError::WeightsError(msg) => write!(f, "Weights Error: {}", msg),
            AaregError::CalculationError(msg) => write!(f, "Calculation Error: {}", msg),
            AaregError::InputError(msg) => write!(f, "Input Error: {}", msg),
            AaregError::InternalError(msg) => write!(f, "Internal Error: {}", msg),
            AaregError::GenericError(msg) => write!(f, "Generic Error: {}", msg),
        }
    }
}

impl From<pyo3::PyErr> for AaregError {
    fn from(err: pyo3::PyErr) -> AaregError {
        AaregError::GenericError(err.to_string())
    }
}

impl From<AaregError> for PyErr {
    fn from(err: AaregError) -> PyErr {
        PyRuntimeError::new_err(format!("Aareg error: {}", err))
    }
}

/// Main function to perform Aalen's additive regression.
#[pyfunction]
fn aareg(options: AaregOptions) -> PyResult<AaregResult> {
    // Convert data to Array2<f64>
    let data_array = Array2::from_shape_vec(
        (options.data.len(), options.data[0].len()),
        options.data.clone().into_iter().flatten().collect(),
    )
    .map_err(|e| AaregError::DataError(e.to_string()))?;

    // Parse the formula to get response and covariate names
    let (response_name, covariate_names) = parse_formula(&options.formula)?;

    // Apply subset if provided
    let subset_data = apply_subset(&data_array, &options.subset)?;

    // Apply weights if provided
    let weighted_data = apply_weights(&subset_data, options.weights.clone())?;

    // Handle missing data according to na_action
    let filtered_data = handle_missing_data(&weighted_data, options.na_action.clone())?;

    // Prepare data for regression
    let (y, x) = prepare_data_for_regression(
        &filtered_data,
        &response_name,
        &covariate_names,
        &options.variable_names,
    )?;

    // Perform Aalen's additive regression
    let regression_result = perform_aalen_regression(&y, &x, &options)?;

    // Post-process results
    let processed_result = post_process_results(regression_result, &options)?;

    Ok(processed_result)
}

/// Parses the formula string to extract the response variable and covariates.
fn parse_formula(formula: &str) -> Result<(String, Vec<String>), AaregError> {
    let mut formula_parts = formula.splitn(2, '~');
    let response = formula_parts
        .next()
        .ok_or_else(|| AaregError::FormulaError("Formula is missing a response variable.".to_string()))?
        .trim()
        .to_string();
    let covariates_str = formula_parts
        .next()
        .ok_or_else(|| AaregError::FormulaError("Formula is missing covariates.".to_string()))?
        .trim();
    let covariates = covariates_str
        .split('+')
        .map(|s| s.trim().to_string())
        .collect();
    Ok((response, covariates))
}

/// Applies subset selection to the data.
fn apply_subset(
    data: &Array2<f64>,
    subset: &Option<Vec<usize>>,
) -> Result<Array2<f64>, AaregError> {
    match subset {
        Some(s) => {
            if s.iter().any(|&i| i >= data.nrows()) {
                return Err(AaregError::DataError(
                    "Subset indices are out of bounds".to_string(),
                ));
            }
            let subset_data = data.select(Axis(0), s);
            Ok(subset_data)
        }
        None => Ok(data.clone()),
    }
}

/// Applies weights to the data if provided.
fn apply_weights(
    data: &Array2<f64>,
    weights: Option<Vec<f64>>,
) -> Result<Array2<f64>, AaregError> {
    match weights {
        Some(w) => {
            if w.len() != data.nrows() {
                return Err(AaregError::WeightsError(
                    "Weights length does not match number of observations".to_string(),
                ));
            }
            let weights_array = Array1::from_vec(w);
            let weighted_data = data * &weights_array.insert_axis(Axis(1));
            Ok(weighted_data)
        }
        None => Ok(data.clone()),
    }
}

/// Handles missing data according to the specified action.
fn handle_missing_data(
    data: &Array2<f64>,
    na_action: Option<String>,
) -> Result<Array2<f64>, AaregError> {
    match na_action.as_deref() {
        Some("Fail") => {
            if data.iter().any(|x| x.is_nan()) {
                Err(AaregError::InputError(
                    "Invalid input: missing values in data".to_string(),
                ))
            } else {
                Ok(data.clone())
            }
        }
        Some("Exclude") => {
            let not_nan_rows: Vec<usize> = data
                .axis_iter(Axis(0))
                .enumerate()
                .filter(|(_, row)| !row.iter().any(|x| x.is_nan()))
                .map(|(i, _)| i)
                .collect();

            if not_nan_rows.is_empty() {
                Err(AaregError::InputError(
                    "All rows contain NaN values".to_string(),
                ))
            } else {
                Ok(data.select(Axis(0), &not_nan_rows))
            }
        }
        Some(other) => Err(AaregError::InputError(format!(
            "Invalid na_action '{}'. Expected 'Fail' or 'Exclude'.",
            other
        ))),
        None => Ok(data.clone()),
    }
}

/// Prepares data for regression by selecting the response and predictor variables.
fn prepare_data_for_regression(
    data: &Array2<f64>,
    response_name: &String,
    covariate_names: &[String],
    variable_names: &[String],
) -> Result<(Array1<f64>, Array2<f64>), AaregError> {
    // Map variable names to their column indices
    let mut name_to_index = HashMap::new();
    for (i, name) in variable_names.iter().enumerate() {
        name_to_index.insert(name.clone(), i);
    }

    // Get index of the response variable
    let response_index = name_to_index
        .get(response_name)
        .ok_or_else(|| AaregError::FormulaError(format!("Response variable '{}' not found.", response_name)))?;

    // Get indices of covariate variables
    let mut covariate_indices = Vec::new();
    for cov_name in covariate_names {
        let idx = name_to_index.get(cov_name).ok_or_else(|| {
            AaregError::FormulaError(format!("Covariate '{}' not found.", cov_name))
        })?;
        covariate_indices.push(*idx);
    }

    // Extract response vector and predictor matrix
    let y = data.column(*response_index).to_owned();
    let x = data.select(Axis(1), &covariate_indices);

    Ok((y, x))
}

/// Performs Aalen's additive regression analysis.
fn perform_aalen_regression(
    y: &Array1<f64>,
    x: &Array2<f64>,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    // Placeholder for the actual regression implementation
    // TODO: Implement Aalen's additive regression algorithm here.

    // For demonstration, we will return dummy values
    let num_covariates = x.ncols();

    Ok(AaregResult {
        coefficients: vec![0.0; num_covariates],
        standard_errors: vec![0.0; num_covariates],
        confidence_intervals: vec![
            ConfidenceInterval {
                lower_bound: 0.0,
                upper_bound: 0.0,
            };
            num_covariates
        ],
        p_values: vec![1.0; num_covariates],
        goodness_of_fit: 1.0,
        fit_details: Some(FitDetails {
            iterations: 0,
            converged: true,
            final_objective_value: 0.0,
            convergence_threshold: options.qrtol,
            change_in_objective: None,
            max_iterations: None,
            optimization_method: Some("Placeholder".to_string()),
            warnings: vec![],
        }),
        residuals: Some(vec![0.0; y.len()]),
        diagnostics: Some(Diagnostics {
            dfbetas: None,
            cooks_distance: None,
            leverage: None,
            deviance_residuals: None,
            martingale_residuals: None,
            schoenfeld_residuals: None,
            score_residuals: None,
            additional_measures: None,
        }),
    })
}

/// Post-processes the results, applying any additional computations as specified in options.
fn post_process_results(
    mut regression_result: AaregResult,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    if options.dfbeta {
        if let Some(ref mut diagnostics) = regression_result.diagnostics {
            diagnostics.dfbetas = Some(vec![0.0; regression_result.coefficients.len()]);
        }
    }
    Ok(regression_result)
}

/// Python module definition
#[pymodule]
fn py_aareg(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AaregOptions>()?;
    m.add_class::<AaregResult>()?;
    m.add_class::<ConfidenceInterval>()?;
    m.add_class::<FitDetails>()?;
    m.add_class::<Diagnostics>()?;
    m.add_function(wrap_pyfunction!(aareg, m)?)?;
    Ok(())
}
