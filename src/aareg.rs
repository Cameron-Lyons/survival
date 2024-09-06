// Aalen’s additive regression model for censored dat

use ndarray::{Array1, Array2, Axis, s};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone)]
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
    coefficients: Vec<f64>,                  // Estimated coefficients for each predictor variable
    standard_errors: Vec<f64>,               // Standard errors for the estimated coefficients
    confidence_intervals: Vec<ConfidenceInterval>, // Confidence intervals for the coefficients
    p_values: Vec<f64>,                      // P-values for testing the hypothesis that each coefficient is zero
    goodness_of_fit: f64,                    // The overall goodness-of-fit statistic for the model
    fit_details: Option<FitDetails>,         // Optional: Information about the model fit, convergence details, etc.
    residuals: Option<Vec<f64>>,             // Optional: Residuals from the model
    diagnostics: Option<Diagnostics>,        // Optional: Additional diagnostic information
}

#[pyclass]
struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

#[pyclass]
struct FitDetails {
    iterations: u32,                  // Number of iterations taken by the fitting algorithm
    converged: bool,                  // Whether the fitting algorithm successfully converged
    final_objective_value: f64,       // The final value of the objective function
    convergence_threshold: f64,       // The threshold for convergence
    change_in_objective: Option<f64>, // Optional: The rate of change of the objective function in the last iteration
    max_iterations: Option<u32>,      // Optional: The maximum number of iterations allowed
    optimization_method: Option<String>, // Optional: Information about the optimization method used
    warnings: Vec<String>,            // Any warnings or notes generated during the fitting process
}

#[pyclass]
struct Diagnostics {
    dfbetas: Option<Vec<f64>>,            // DFBetas for each predictor variable to assess their influence on the model
    cooks_distance: Option<Vec<f64>>,     // Cook's distance for each observation
    leverage: Option<Vec<f64>>,           // Leverage values for each observation
    deviance_residuals: Option<Vec<f64>>, // Deviance residuals
    martingale_residuals: Option<Vec<f64>>, // Martingale residuals
    schoenfeld_residuals: Option<Vec<f64>>, // Schoenfeld residuals
    score_residuals: Option<Vec<f64>>,    // Score residuals
    additional_measures: Option<Vec<f64>>, // Optional: Additional model-specific diagnostic measures
}

#[derive(Debug)]
enum AaregError {
    DataError(String),        // e.g., "Data dimensions mismatch"
    FormulaError(String),     // e.g., "Formula parsing error"
    WeightsError(String),     // e.g., "Weights length does not match number of observations"
    CalculationError(String), // e.g., "Singular matrix encountered in calculations"
    InputError(String),       // e.g., "Invalid input: negative values in data"
    InternalError(String),    // e.g., "Unexpected internal error"
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

#[pyfunction]
fn aareg(options: &AaregOptions) -> Result<AaregResult, AaregError> {
    let (response, covariates) = parse_formula(&options.formula)?;
    let subset_data = apply_subset(&options.data, &options.subset)?;

    let py = unsafe { Python::assume_gil_acquired() };
    let py_subset_data = PyList::new(py, subset_data.outer_iter().map(|x| x.to_vec()).collect::<Vec<_>>());

    let weighted_data = apply_weights(py, &py_subset_data, options.weights.clone())?;
    let filtered_data = handle_missing_data(weighted_data, options.na_action.clone())?;
    let (y, x) = prepare_data_for_regression(&filtered_data, &response, &covariates)?;
    let regression_result = perform_aalen_regression(&y, &x, options)?;
    let processed_result = post_process_results(regression_result, options)?;

    Ok(processed_result)
}

fn parse_formula(formula: &str) -> Result<(String, Vec<String>), AaregError> {
    let mut formula_parts = formula.splitn(2, '~');
    let response = formula_parts.next().unwrap().trim().to_string();
    let covariates = formula_parts
        .next()
        .unwrap_or("")
        .trim()
        .split('+')
        .map(|s| s.trim().to_string())
        .collect();
    Ok((response, covariates))
}

fn apply_subset(
    data: &Array2<f64>,
    subset: &Option<Vec<usize>>,
) -> Result<Array2<f64>, AaregError> {
    match subset {
        Some(s) => {
            let subset_data = data.select(Axis(0), s);
            Ok(subset_data)
        }
        None => Ok(data.to_owned()),
    }
}

fn apply_weights(
    py: Python,
    data: &PyList,
    weights: Option<Vec<f64>>,
) -> PyResult<Array2<f64>> {
    let data_vec: Vec<Vec<f64>> = data.extract()?;
    let data_array = Array2::from_shape_vec(
        (data_vec.len(), data_vec[0].len()),
        data_vec.into_iter().flatten().collect::<Vec<_>>(),
    ).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to convert to Array2: {}",
            e
        ))
    })?;

    let result_array = match weights {
        Some(w) => {
            if w.len() != data_array.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Weights length does not match number of observations",
                ));
            }

            let weights_array = Array1::from_vec(w);
            let weighted_data = data_array * &weights_array.insert_axis(Axis(1));

            weighted_data
        }
        None => data_array,
    };

    Ok(result_array)
}

fn handle_missing_data(
    data: Array2<f64>,
    na_action: Option<NaAction>,
) -> Result<Array2<f64>, AaregError> {
    match na_action {
        Some(NaAction::Fail) => {
            if data.iter().any(|x| x.is_nan()) {
                Err(AaregError::InputError(
                    "Invalid input: missing values in data".to_string(),
                ))
            } else {
                Ok(data)
            }
        }
        Some(NaAction::Exclude) => {
            let filtered_data = data
                .axis_iter(Axis(0))
                .filter(|row| !row.iter().any(|x| x.is_nan()))
                .collect::<Vec<_>>();

            if filtered_data.is_empty() {
                Err(AaregError::InputError(
                    "All rows contain NaN values".to_string(),
                ))
            } else {
                let rows = filtered_data.len();
                let cols = filtered_data[0].len();
                let flat_data: Vec<f64> = filtered_data
                    .into_iter()
                    .flat_map(|r| r.iter().cloned().collect::<Vec<f64>>())
                    .collect();
                Ok(Array2::from_shape_vec((rows, cols), flat_data).unwrap())
            }
        }
        None => Ok(data),
    }
}

fn prepare_data_for_regression(
    data: &Array2<f64>,
    response: &str,
    covariates: &[String],
) -> Result<(Array2<f64>, Array2<f64>), AaregError> {
    let response_index = 0; // Assuming the response is always the first column; this will need to be adjusted
    let covariate_indices: Vec<usize> = (1..covariates.len() + 1).collect(); // Assuming covariates follow the response

    let y = data.slice(s![.., response_index..response_index + 1]);
    let x = data.select(Axis(1), &covariate_indices);
    Ok((y.to_owned(), x))
}

fn perform_aalen_regression(
    y: &Array2<f64>,
    x: &Array2<f64>,
    _options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    // Placeholder for actual regression logic
    let coefficients = vec![0.0; x.ncols()];
    let standard_errors = vec![0.0; x.ncols()];
    let confidence_intervals = coefficients
        .iter()
        .map(|&_| ConfidenceInterval {
            lower_bound: 0.0,
            upper_bound: 0.0,
        })
        .collect();
    let p_values = vec![0.0; x.ncols()];
    let goodness_of_fit = 1.0;
    let fit_details = Some(FitDetails {
        iterations: 10,
        converged: true,
        final_objective_value: 0.0,
        convergence_threshold: 1e-7,
        change_in_objective: Some(0.0),
        max_iterations: Some(100),
        optimization_method: Some("Gradient Descent".to_string()),
        warnings: vec![],
    });
    let residuals = Some(vec![0.0; y.len_of(Axis(0))]);
    let diagnostics = Some(Diagnostics {
        dfbetas: Some(vec![0.0; x.ncols()]),
        cooks_distance: Some(vec![0.0; y.len_of(Axis(0))]),
        leverage: Some(vec![0.0; y.len_of(Axis(0))]),
        deviance_residuals: Some(vec![0.0; y.len_of(Axis(0))]),
        martingale_residuals: Some(vec![0.0; y.len_of(Axis(0))]),
        schoenfeld_residuals: Some(vec![0.0; y.len_of(Axis(0))]),
        score_residuals: Some(vec![0.0; y.len_of(Axis(0))]),
        additional_measures: None,
    });

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

fn post_process_results(
    mut regression_result: AaregResult,
    options: &AaregOptions,
) -> Result<AaregResult, AaregError> {
    if options.dfbeta {
        if let Some(ref mut diagnostics) = regression_result.diagnostics {
            diagnostics.dfbetas = Some(vec![0.1; regression_result.coefficients.len()]); // Placeholder logic
        }
    }
    Ok(regression_result)
}

#[pymodule]
fn pyAareg(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AaregOptions>()?;
    m.add_class::<Surv>()?;
    m.add_class::<AaregResult>()?;
    m.add_class::<ConfidenceInterval>()?;
    m.add_class::<FitDetails>()?;
    m.add_class::<Diagnostics>()?;
    m.add_function(wrap_pyfunction!(aareg, m)?)?;
    Ok(())
}
