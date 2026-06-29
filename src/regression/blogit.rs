use crate::internal::statistical::probit;
use pyo3::prelude::*;
fn cloglog(p: f64) -> f64 {
    (-(1.0 - p).ln()).ln()
}

fn bounded_unit_interval(input: f64, edge: f64) -> f64 {
    input.min(1.0 - edge).max(edge)
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
        let adjusted_input = bounded_unit_interval(input, self.edge);
        adjusted_input.ln() - (1.0 - adjusted_input).ln()
    }
    fn bprobit(&self, input: f64) -> f64 {
        let adjusted_input = bounded_unit_interval(input, self.edge);
        probit(adjusted_input)
    }
    fn bcloglog(&self, input: f64) -> f64 {
        let adjusted_input = bounded_unit_interval(input, self.edge);
        cloglog(adjusted_input)
    }
    fn blog(&self, input: f64) -> f64 {
        let adjusted_input = if input < self.edge { self.edge } else { input };
        adjusted_input.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_2;

    #[test]
    fn bounded_links_match_r_survival_reference_values() {
        let link = LinkFunctionParams { edge: 0.05 };
        assert!((link.blogit(0.0) - -2.9444389791664403).abs() < 1e-9);
        assert!((link.bprobit(0.0) - -1.6448536269514729).abs() < 1e-8);
        assert!((link.bcloglog(0.0) - -2.9701952490421637).abs() < 1e-9);
        assert!((link.blog(0.0) - -2.995732273553991).abs() < 1e-9);

        assert!((link.blogit(0.5)).abs() < 1e-9);
        assert!((link.bprobit(0.5)).abs() < 1e-8);
        assert!((link.bcloglog(0.5) - -0.36651292058166435).abs() < 1e-9);
        assert!((link.blog(0.5) - -LN_2).abs() < 1e-9);
    }

    #[test]
    fn bounded_links_follow_r_clamp_order_for_large_edge() {
        let link = LinkFunctionParams { edge: 0.6 };
        for input in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((link.blogit(input) - 0.4054651081081642).abs() < 1e-9);
            assert!((link.bprobit(input) - 0.2533471031357997).abs() < 1e-8);
            assert!((link.bcloglog(input) - -0.08742157179075517).abs() < 1e-9);
        }
    }
}
