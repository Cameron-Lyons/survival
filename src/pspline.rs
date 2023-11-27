// Smoothing splines using a pspline basis

struct PSpline {
    x: Vec<f64>,                // Covariate vector
    df: u32,                    // Degrees of freedom
    theta: f64,                 // Roughness penalty
    nterm: u32,                 // Number of splines in the basis
    degree: u32,                // Degree of splines
    eps: f64,                   // Accuracy for df
    method: String,             // Method for choosing tuning parameter theta
    boundary_knots: (f64, f64), // Boundary knots
    intercept: bool,            // Include intercept in basis functions or not
    penalty: bool,              // Apply penalty or not
}
