// Aalen’s additive regression model for censored data

use ndarray::Array2;
use std::collections::HashMap;

enum NaAction {
    Fail,
    Exclude,
}

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

struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

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
