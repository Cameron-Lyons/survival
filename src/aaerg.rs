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
