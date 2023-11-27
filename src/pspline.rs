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

impl PSpline {
    fn new(
        x: Vec<f64>,
        df: u32,
        theta: f64,
        eps: f64,
        method: String,
        boundary_knots: (f64, f64),
        intercept: bool,
        penalty: bool,
    ) -> PSpline {
        let nterm = df + 1;
        let degree = 3;
        PSpline {
            x: x,
            df: df,
            theta: theta,
            nterm: nterm,
            degree: degree,
            eps: eps,
            method: method,
            boundary_knots: boundary_knots,
            intercept: intercept,
            penalty: penalty,
        }
    }
    pub fn fit(&self) {
        let basis = self.create_basis();
        let penalized_basis = self.apply_penalty(basis);
        let coefficients = self.optimize_fit(penalized_basis);
    }
    fn create_basis(&self) -> Vec<Vec<f64>> {
        let n = self.x.len();
        let mut basis = vec![vec![0.0; self.nterm as usize]; n];
        for i in 0..n {
            for j in 0..self.nterm {
                basis[i][j as usize] = self.basis_function(self.x[i], j);
            }
        }
        basis
    }
    fn basis_function(&self, x: f64, j: u32) -> f64 {
        let mut b = 0.0;
        if self.intercept {
            b += 1.0;
        }
        if j == 0 {
            return b;
        }
        let (a, b) = self.boundary_knots;
        let t = (x - a) / (b - a);
        let knots = self.knots();
        let mut d = vec![0.0; self.degree as usize + 1];
        d[0] = 1.0;
        for k in 1..self.degree + 1 {
            for i in 0..self.degree - k + 1 {
                let w = (t - knots[i] / (knots[i + k] - knots[i])).max(0.0);
                d[i as usize] = (1.0 - w) * d[i as usize] + w * d[i as usize + 1];
            }
        }
        b * d[0]
    }
    fn knots(&self) -> Vec<f64> {
        let (a, b) = self.boundary_knots;
        let mut knots = vec![0.0; self.nterm as usize + self.degree as usize + 1];
        for i in 0..self.degree + 1 {
            knots[i as usize] = a;
            knots[(self.nterm + self.degree - i) as usize] = b;
        }
        knots
    }
}
