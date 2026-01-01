use crate::utilities::statistical::normal_inverse_cdf;
use pyo3::prelude::*;

fn probit(p: f64) -> f64 {
    normal_inverse_cdf(p)
}
fn cloglog(p: f64) -> f64 {
    (-(1.0 - p).ln()).ln()
}
#[pyclass]
pub struct LinkFunctionParams {
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
