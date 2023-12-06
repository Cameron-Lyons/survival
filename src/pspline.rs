// Smoothing splines using a pspline basis
use pyo3::prelude::*;

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
    fn apply_penalty(&self, basis: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut penalized_basis = basis.clone();
        if self.penalty {
            let mut penalty = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
            for i in 0..self.nterm {
                for j in 0..self.nterm {
                    penalty[i as usize][j as usize] = self.penalty_function(i, j);
                }
            }
            for i in 0..self.nterm {
                for j in 0..self.nterm {
                    for k in 0..self.nterm {
                        penalized_basis[i as usize][j as usize] -= self.theta
                            * penalty[i as usize][k as usize]
                            * basis[k as usize][j as usize];
                    }
                }
            }
        }
        penalized_basis
    }
    fn penalty_function(&self, i: u32, j: u32) -> f64 {
        match self.method {
            "GCV" => self.gcv(i, j),
            "UBRE" => self.ubre(i, j),
            _ => panic!("Method not implemented"),
        }
    }
    fn gcv(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut gcv = 0.0;
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        gcv = df / (1.0 - trace / self.nterm as f64).powi(2);
        gcv
    }
    fn ubre(&self, i: u32, j: u32) -> f64 {
        if i == j {
            return self.nterm as f64;
        }
        let mut ubre = 0.0;
        let mut df = 0.0;
        let mut trace = 0.0;
        let mut basis = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                basis[k as usize][l as usize] = self.basis_function(self.x[k as usize], l);
            }
        }
        for k in 0..self.nterm {
            for l in 0..self.nterm {
                df += basis[k as usize][i as usize]
                    * basis[l as usize][j as usize]
                    * basis[k as usize][l as usize];
                trace += basis[k as usize][i as usize]
                    * basis[k as usize][l as usize]
                    * basis[l as usize][j as usize];
            }
        }
        ubre = df / (1.0 - trace / self.nterm as f64).powi(2);
        ubre
    }
    fn optimize_fit(&self, basis: Vec<Vec<f64>>) -> Vec<f64> {
        let mut coefficients = vec![0.0; self.nterm as usize];
        let mut a = vec![vec![0.0; self.nterm as usize]; self.nterm as usize];
        let mut b = vec![0.0; self.nterm as usize];
        for i in 0..self.nterm {
            for j in 0..self.nterm {
                for k in 0..self.nterm {
                    a[i as usize][j as usize] +=
                        basis[k as usize][i as usize] * basis[k as usize][j as usize];
                }
            }
        }
        for i in 0..self.nterm {
            for j in 0..self.nterm {
                b[i as usize] += basis[j as usize][i as usize];
            }
        }
        coefficients = self.solve(a, b);
        coefficients
    }
}
