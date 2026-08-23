use crate::internal::statistical::probit;
use pyo3::prelude::*;
fn cloglog(p: f64) -> f64 {
    (-(1.0 - p).ln()).ln()
}

fn bounded_unit_interval(input: f64, edge: f64) -> f64 {
    if input.is_nan() || edge.is_nan() {
        f64::NAN
    } else {
        input.min(1.0 - edge).max(edge)
    }
}

#[pyclass]
pub struct LinkFunctionParams {
    edge: f64,
}

impl LinkFunctionParams {
    fn transform_many(
        &self,
        input: Vec<Option<f64>>,
        transform: fn(&Self, f64) -> f64,
    ) -> Vec<f64> {
        input
            .into_iter()
            .map(|value| value.map_or(f64::NAN, |value| transform(self, value)))
            .collect()
    }
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
    fn blogit_many(&self, input: Vec<Option<f64>>) -> Vec<f64> {
        self.transform_many(input, Self::blogit)
    }
    fn bprobit(&self, input: f64) -> f64 {
        let adjusted_input = bounded_unit_interval(input, self.edge);
        if !(0.0..=1.0).contains(&adjusted_input) {
            f64::NAN
        } else {
            probit(adjusted_input)
        }
    }
    fn bprobit_many(&self, input: Vec<Option<f64>>) -> Vec<f64> {
        self.transform_many(input, Self::bprobit)
    }
    fn bcloglog(&self, input: f64) -> f64 {
        let adjusted_input = bounded_unit_interval(input, self.edge);
        cloglog(adjusted_input)
    }
    fn bcloglog_many(&self, input: Vec<Option<f64>>) -> Vec<f64> {
        self.transform_many(input, Self::bcloglog)
    }
    fn blog(&self, input: f64) -> f64 {
        let adjusted_input = if self.edge.is_nan() {
            f64::NAN
        } else if input < self.edge {
            self.edge
        } else {
            input
        };
        adjusted_input.ln()
    }
    fn blog_many(&self, input: Vec<Option<f64>>) -> Vec<f64> {
        self.transform_many(input, Self::blog)
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

    #[test]
    fn bounded_link_vectors_match_scalars_and_preserve_missing_values() {
        let link = LinkFunctionParams { edge: 0.05 };
        let input = vec![Some(0.0), None, Some(0.5), Some(1.0)];

        for (actual, transform) in [
            (
                link.blogit_many(input.clone()),
                LinkFunctionParams::blogit as fn(&LinkFunctionParams, f64) -> f64,
            ),
            (
                link.bprobit_many(input.clone()),
                LinkFunctionParams::bprobit,
            ),
            (
                link.bcloglog_many(input.clone()),
                LinkFunctionParams::bcloglog,
            ),
            (link.blog_many(input.clone()), LinkFunctionParams::blog),
        ] {
            assert_eq!(actual.len(), input.len());
            assert!(actual[1].is_nan());
            for idx in [0, 2, 3] {
                assert!((actual[idx] - transform(&link, input[idx].unwrap())).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn bounded_links_propagate_nan_inputs_and_edges() {
        let link = LinkFunctionParams { edge: 0.05 };
        let nan_edge = LinkFunctionParams { edge: f64::NAN };

        for transform in [
            LinkFunctionParams::blogit as fn(&LinkFunctionParams, f64) -> f64,
            LinkFunctionParams::bprobit,
            LinkFunctionParams::bcloglog,
            LinkFunctionParams::blog,
        ] {
            assert!(transform(&link, f64::NAN).is_nan());
            assert!(transform(&nan_edge, 0.5).is_nan());
        }

        for actual in [
            link.blogit_many(vec![Some(f64::NAN)]),
            link.bprobit_many(vec![Some(f64::NAN)]),
            link.bcloglog_many(vec![Some(f64::NAN)]),
            link.blog_many(vec![Some(f64::NAN)]),
        ] {
            assert!(actual[0].is_nan());
        }

        assert!(LinkFunctionParams { edge: 2.0 }.bprobit(0.5).is_nan());
        assert!(
            LinkFunctionParams { edge: -1.0 }
                .bprobit(f64::NEG_INFINITY)
                .is_nan()
        );
    }
}
