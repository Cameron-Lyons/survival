#![allow(dead_code)]
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

#[allow(dead_code)]
fn probit(p: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    normal.inverse_cdf(p)
}

#[allow(dead_code)]
fn cloglog(p: f64) -> f64 {
    (-(1.0 - p).ln()).ln()
}

#[pyclass]
pub struct LinkFunctionParams {
    #[allow(dead_code)]
    edge: f64,
}

#[pymethods]
impl LinkFunctionParams {
    #[new]
    fn new(edge: f64) -> Self {
        LinkFunctionParams { edge }
    }

    fn blogit(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else if input > 1.0 - self.edge {
            1.0 - self.edge
        } else {
            input
        };
        adjusted_input.ln() - (1.0 - adjusted_input).ln()
    }

    fn bprobit(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else if input > 1.0 - self.edge {
            1.0 - self.edge
        } else {
            input
        };
        probit(adjusted_input) - probit(1.0 - adjusted_input)
    }

    fn bcloglog(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge {
            self.edge
        } else if input > 1.0 - self.edge {
            1.0 - self.edge
        } else {
            input
        };
        cloglog(adjusted_input) - cloglog(1.0 - adjusted_input)
    }

    fn blog(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge { self.edge } else { input };
        adjusted_input.ln()
    }
}
