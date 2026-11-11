//! Predictor-precision fallbacks. The ordinary and historical counting-process
//! recentering paths remain in use below the predictor precision threshold.

use super::{AgWalk, CoxData, ExactRiskAccumulator, ExactRiskTree};
use crate::error::SurvivalResult;
use crate::regression::exact_ties::exact_tied_moments_scales;
use crate::regression::log_risk_sweep::LogRiskSweep;
use ndarray::{Array2, ArrayView2};

impl CoxData {
    pub(super) fn agfit4_scales(
        &self,
        beta: &[f64],
        walk: &AgWalk,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) -> SurvivalResult<f64> {
        let eta = self.predictor_scales(beta);
        let mut entry = vec![0.0; eta.len()];
        for &(time, row) in &walk.by_entry {
            entry[row] = time;
        }
        let mut mean = vec![0.0; u.len()];
        let mut covariance = Array2::zeros(imat.raw_dim());
        let mut loglik = 0.0;
        for &(start, end) in &walk.bounds {
            let rows: Vec<_> = walk.by_stop[start..end].iter().map(|row| row.row).collect();
            let sweep = LogRiskSweep {
                stop: self.time.as_slice().expect("contiguous"),
                entry: Some(&entry),
                status: self.status.as_slice().expect("contiguous"),
                x: self.covar.view(),
                weights: self.weights.as_slice().expect("contiguous"),
                log_risk: &[],
                rows: &rows,
            };
            sweep.for_each_time_with_covariance_scales(&eta, |event| {
                let Some(&first) = event.deaths.first() else {
                    return Ok(());
                };
                let reference = event.deaths.iter().fold(eta[first], |reference, &row| {
                    if eta[row].ratio_log(reference) > 0.0 {
                        eta[row]
                    } else {
                        reference
                    }
                });
                for &person in event.deaths {
                    let weight = self.weights[person];
                    loglik += weight * eta[person].ratio_log(reference);
                    for (variable, value) in u.iter_mut().enumerate() {
                        *value += weight * self.covar[(person, variable)];
                    }
                }
                let steps = if self.efron { event.deaths.len() } else { 1 };
                let average_weight = event.death_weight / steps as f64;
                for step in 0..steps {
                    let denom = event.efron_denom(step);
                    loglik += average_weight * reference.ratio_log(denom);
                    event.efron_mean(step, &mut mean);
                    event.efron_covariance(step, &mut covariance);
                    for (i, value) in u.iter_mut().enumerate() {
                        *value -= average_weight * mean[i];
                        for k in 0..=i {
                            imat[(k, i)] += average_weight * covariance[(i, k)];
                        }
                    }
                }
                Ok(())
            })?;
        }
        Ok(loglik)
    }

    pub(super) fn exact_scales(
        &self,
        beta: &[f64],
        walk: &AgWalk,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) -> f64 {
        let eta = self.predictor_scales(beta);
        if eta.iter().any(|value| {
            !value.origin.is_finite()
                || !value.origin_correction.is_finite()
                || !value.log_mass.is_finite()
        }) {
            return f64::NAN;
        }
        let mut loglik = 0.0;
        let mut deaths = Vec::new();
        let mut rows = Vec::new();
        for (stratum, &(start, end)) in walk.bounds.iter().enumerate() {
            if start == end {
                continue;
            }
            let growing = walk.growing[stratum];
            let mut singleton = growing.then(|| ExactRiskAccumulator::new(self.covar.ncols()));
            let mut singleton_rows = 0;
            let mut tree = (!growing).then(|| {
                let (first, last) = walk.row_bounds[stratum];
                ExactRiskTree::new(first, last - first, self.covar.ncols())
            });
            rows.clear();
            let (mut joined, mut left) = (start, start);
            while joined < end {
                let time = walk.by_stop[joined].time;
                if let Some(tree) = &mut tree {
                    while left < end && walk.by_entry[left].0 >= time {
                        tree.set_active(walk.by_entry[left].1, false);
                        left += 1;
                    }
                }
                deaths.clear();
                while joined < end && walk.by_stop[joined].time == time {
                    let person = walk.by_stop[joined].row;
                    if let Some(tree) = &mut tree {
                        tree.set_active(person, true);
                    } else {
                        rows.push(person);
                    }
                    if walk.by_stop[joined].death {
                        deaths.push(person);
                    }
                    joined += 1;
                }
                let nrisk = tree.as_ref().map_or(rows.len(), ExactRiskTree::len);
                if deaths.is_empty() || deaths.len() == nrisk {
                    continue;
                }
                if deaths.len() == 1 {
                    if let Some(tree) = &mut tree {
                        tree.refresh_scales(&eta, &self.covar);
                        loglik += eta[deaths[0]].ratio_log(tree.weight());
                        self.exact_scaled_score(&deaths, tree.mean(), tree.covariance(), u, imat);
                    } else {
                        let moments = singleton.as_mut().expect("growing risk set");
                        for &person in &rows[singleton_rows..] {
                            moments.add_scale(person, eta[person], &self.covar);
                        }
                        singleton_rows = rows.len();
                        loglik += eta[deaths[0]].ratio_log(moments.weight);
                        self.exact_scaled_score(
                            &deaths,
                            &moments.mean,
                            moments.covariance.view(),
                            u,
                            imat,
                        );
                    }
                } else {
                    if let Some(tree) = &tree {
                        rows.clear();
                        rows.extend(tree.active_rows());
                    }
                    let moments = exact_tied_moments_scales(&rows, deaths.len(), &eta, &self.covar);
                    let death_risks: Vec<_> = deaths.iter().map(|&row| eta[row]).collect();
                    loglik += moments.log_likelihood(&death_risks);
                    self.exact_scaled_score(
                        &deaths,
                        &moments.mean,
                        moments.covariance.view(),
                        u,
                        imat,
                    );
                }
            }
        }
        loglik
    }

    fn exact_scaled_score(
        &self,
        deaths: &[usize],
        mean: &[f64],
        covariance: ArrayView2<'_, f64>,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) {
        for &person in deaths {
            for (variable, value) in u.iter_mut().enumerate() {
                *value += self.covar[(person, variable)];
            }
        }
        for (variable, value) in u.iter_mut().enumerate() {
            *value -= mean[variable];
            for other in 0..self.covar.ncols() {
                imat[(variable, other)] += covariance[(variable, other)];
            }
        }
    }
}
