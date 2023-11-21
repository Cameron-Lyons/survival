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
