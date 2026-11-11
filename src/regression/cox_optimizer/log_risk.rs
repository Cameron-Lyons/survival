//! Logarithmic Breslow/Efron moments for right-censored risk sets.
//!
//! The reverse-time sweep only adds rows, so a growing accumulator retains a
//! late, tiny risk set before earlier, much larger risks enter it. Keeping the
//! current deaths separate also forms Efron's denominator as a positive mixture
//! rather than subtracting nearly equal risk sums.

use super::{CoxData, ExactRiskAccumulator};
use crate::regression::log_scale::LogScale;
use ndarray::Array2;

impl CoxData {
    pub(super) fn coxfit6_log(&self, beta: &[f64], u: &mut [f64], imat: &mut Array2<f64>) -> f64 {
        let eta = self.predictor_scales(beta);
        let nvar = u.len();
        let mut survivors = ExactRiskAccumulator::new(nvar);
        let mut deaths = ExactRiskAccumulator::new(nvar);
        let mut delta = vec![0.0; nvar];
        let mut loglik = 0.0;
        let mut end = eta.len();
        while end > 0 {
            if self.stratum_end[end - 1] == 1 {
                survivors.clear();
            }
            let mut start = end - 1;
            while start > 0
                && self.stratum_end[start - 1] != 1
                && self.time[start - 1] == self.time[end - 1]
            {
                start -= 1;
            }
            deaths.clear();
            let mut ndead = 0usize;
            let mut death_weight = 0.0;
            let mut death_eta = LogScale::EMPTY;
            let mut centered_eta_sum = 0.0;
            for person in (start..end).rev() {
                let weight = self.weights[person];
                let log_risk = eta[person].shifted(weight.ln());
                if self.status[person] == 0 {
                    survivors.add_scale(person, log_risk, &self.covar);
                } else {
                    if ndead == 0 {
                        death_eta = eta[person];
                    } else {
                        let relative = eta[person].ratio_log(death_eta);
                        if relative > 0.0 {
                            // Reference the largest death, keeping every
                            // centered contribution nonpositive even when
                            // opposite finite origins differ by infinity.
                            centered_eta_sum -= death_weight * relative;
                            death_eta = eta[person];
                        }
                    }
                    ndead += 1;
                    death_weight += weight;
                    centered_eta_sum += weight * eta[person].ratio_log(death_eta);
                    deaths.add_scale(person, log_risk, &self.covar);
                    for (variable, value) in u.iter_mut().enumerate() {
                        *value += weight * self.covar[(person, variable)];
                    }
                }
            }
            if ndead > 0 {
                loglik += centered_eta_sum;
                for (variable, difference) in delta.iter_mut().enumerate() {
                    *difference = deaths.mean[variable] - survivors.mean[variable];
                }
                let steps = if self.efron { ndead } else { 1 };
                let average_weight = death_weight / steps as f64;
                for step in 0..steps {
                    let remaining = (ndead - step) as f64 / ndead as f64;
                    let (denom, a, b) = survivors.weight.mix(deaths.weight.shifted(remaining.ln()));
                    // Subtract before multiplying by case weights: the two
                    // uncentered likelihood terms can overflow separately.
                    loglik += average_weight * death_eta.ratio_log(denom);
                    for i in 0..nvar {
                        let mean = if a >= b {
                            survivors.mean[i] + b * delta[i]
                        } else {
                            deaths.mean[i] - a * delta[i]
                        };
                        u[i] -= average_weight * mean;
                        for k in 0..=i {
                            let covariance = a * survivors.covariance[(i, k)]
                                + b * deaths.covariance[(i, k)]
                                + a * b * delta[i] * delta[k];
                            imat[(k, i)] += average_weight * covariance;
                        }
                    }
                }
                for person in (start..end).rev() {
                    if self.status[person] != 0 {
                        survivors.add_scale(
                            person,
                            eta[person].shifted(self.weights[person].ln()),
                            &self.covar,
                        );
                    }
                }
            }
            end = start;
        }
        loglik
    }
}
