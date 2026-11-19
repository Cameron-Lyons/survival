use super::survfit_confint::{ConfType, survfit_confint, validate_conf_int};
use super::survfitkm::{HazardType, SurvfitKMData, SurvfitKMOptions, fit_with_counts};
use crate::error::SurvivalResult;
use crate::internal::validation::validate_length;

#[doc = " Nelson-Aalen estimate at the distinct event times."]
#[doc = ""]
#[doc = " `n_risk` and `n_events` are unweighted counts; `cumulative_hazard` and"]
#[doc = " `variance` (`std.chaz^2`, the Poisson variance `sum d / n^2`) use the"]
#[doc = " case weights when given."]
#[derive(Debug, Clone, PartialEq)]
pub struct NelsonAalenResult {
    pub time: Vec<f64>,
    pub cumulative_hazard: Vec<f64>,
    pub variance: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub n_risk: Vec<usize>,
    pub n_events: Vec<usize>,
}

impl NelsonAalenResult {
    #[doc = " `exp(-cumulative_hazard)`, the Fleming-Harrington survival estimate."]
    pub fn survival(&self) -> Vec<f64> {
        self.cumulative_hazard.iter().map(|&h| (-h).exp()).collect()
    }
}

pub fn nelson_aalen(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    confidence_level: f64,
) -> SurvivalResult<NelsonAalenResult> {
    validate_length(time.len(), status.len(), "status")?;
    if time.is_empty() {
        if let Some(weights) = weights {
            validate_length(0, weights.len(), "weights")?;
        }
        validate_conf_int(confidence_level)?;
        return Ok(NelsonAalenResult {
            time: Vec::new(),
            cumulative_hazard: Vec::new(),
            variance: Vec::new(),
            ci_lower: Vec::new(),
            ci_upper: Vec::new(),
            n_risk: Vec::new(),
            n_events: Vec::new(),
        });
    }
    let data = SurvfitKMData::try_new(
        None,
        time.to_vec(),
        status.to_vec(),
        weights.map(<[f64]>::to_vec),
        None,
        None,
        None,
    )?;
    let options = SurvfitKMOptions {
        ctype: HazardType::NelsonAalen,
        conf_int: confidence_level,
        conf_type: ConfType::None,
        robust: Some(false),
        ..Default::default()
    };
    let fit = fit_with_counts(&data, &options, true)?;
    let std_chaz = fit.std_chaz.as_deref().expect("se.fit is on");
    let (n_risk, n_event) = match &fit.counts {
        Some(counts) => (&counts.n_risk, &counts.n_event),
        None => (&fit.n_risk, &fit.n_event),
    };
    let rows: Vec<usize> = (0..fit.time.len()).filter(|&i| n_event[i] > 0.0).collect();
    let cumulative_hazard: Vec<f64> = rows.iter().map(|&i| fit.cumhaz[i]).collect();
    let std_err: Vec<f64> = rows.iter().map(|&i| std_chaz[i]).collect();
    let bands = survfit_confint(
        &cumulative_hazard,
        &std_err,
        false,
        ConfType::Plain,
        confidence_level,
        None,
        false,
    )?;
    Ok(NelsonAalenResult {
        time: rows.iter().map(|&i| fit.time[i]).collect(),
        cumulative_hazard,
        variance: std_err.iter().map(|se| se * se).collect(),
        ci_lower: bands.lower,
        ci_upper: bands.upper,
        n_risk: rows.iter().map(|&i| n_risk[i] as usize).collect(),
        n_events: rows.iter().map(|&i| n_event[i] as usize).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn empty_and_all_censored_inputs_give_empty_curves() {
        let result = nelson_aalen(&[], &[], None, 0.95).unwrap();
        assert!(result.time.is_empty());
        let result = nelson_aalen(&[1.0, 2.0, 3.0], &[0, 0, 0], None, 0.95).unwrap();
        assert!(result.time.is_empty());
        assert!(result.cumulative_hazard.is_empty());
    }

    #[test]
    fn matches_the_hand_computed_estimate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 0, 1, 0, 1, 0];
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(result.time, vec![1.0, 3.0, 5.0]);
        assert_eq!(result.n_risk, vec![6, 4, 2]);
        assert_eq!(result.n_events, vec![1, 1, 1]);
        assert_vec_close(
            &result.cumulative_hazard,
            &[
                1.0 / 6.0,
                1.0 / 6.0 + 1.0 / 4.0,
                1.0 / 6.0 + 1.0 / 4.0 + 1.0 / 2.0,
            ],
        );
        assert_vec_close(
            &result.variance,
            &[
                1.0 / 36.0,
                1.0 / 36.0 + 1.0 / 16.0,
                1.0 / 36.0 + 1.0 / 16.0 + 1.0 / 4.0,
            ],
        );
        for i in 0..3 {
            assert!(result.ci_lower[i] <= result.cumulative_hazard[i]);
            assert!(result.cumulative_hazard[i] <= result.ci_upper[i]);
            assert!(result.ci_lower[i] >= 0.0);
        }
        assert_vec_close(
            &result.survival(),
            &[
                (-1.0_f64 / 6.0).exp(),
                (-5.0_f64 / 12.0).exp(),
                (-11.0_f64 / 12.0).exp(),
            ],
        );
    }

    #[test]
    fn weights_scale_the_hazard_but_not_the_counts() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let weighted = nelson_aalen(&time, &status, Some(&[2.0, 1.0, 1.0]), 0.95).unwrap();
        assert_eq!(weighted.n_risk, vec![3, 2, 1]);
        assert!((weighted.cumulative_hazard[0] - 0.5).abs() < 1e-12);
        let unit = nelson_aalen(&time, &status, Some(&[1.0; 3]), 0.95).unwrap();
        let plain = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(unit, plain);
    }

    #[test]
    fn near_tied_event_times_are_grouped() {
        let time = vec![1.0 + 5e-10, 2.0, 1.0];
        let status = vec![1, 0, 1];
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(result.time, vec![1.0]);
        assert_eq!(result.n_risk, vec![3]);
        assert_eq!(result.n_events, vec![2]);
        assert!((result.cumulative_hazard[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((result.variance[0] - 2.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn near_unit_weights_keep_exact_counts_at_event_times() {
        let weights = [1.0 - 1e-10; 3];
        let fit = nelson_aalen(&[1.0, 2.0, 3.0], &[1, 0, 1], Some(&weights), 0.95).unwrap();
        assert_eq!(fit.time, [1.0, 3.0]);
        assert_eq!(fit.n_risk, [3, 1]);
        assert_eq!(fit.n_events, [1, 1]);
        assert_vec_close(&fit.cumulative_hazard, &[1.0 / 3.0, 4.0 / 3.0]);
    }

    #[test]
    fn rejects_malformed_inputs() {
        assert!(nelson_aalen(&[], &[], None, 1.0).is_err());
        assert!(nelson_aalen(&[], &[], Some(&[1.0]), 0.95).is_err());
        assert!(nelson_aalen(&[1.0, 2.0], &[1, 2], None, 0.95).is_err());
        assert!(nelson_aalen(&[1.0, 2.0], &[1, 0], Some(&[1.0, f64::INFINITY]), 0.95).is_err());
        assert!(nelson_aalen(&[1.0, 2.0], &[1, 0], None, 1.0).is_err());
        assert!(nelson_aalen(&[1.0], &[1, 0], None, 0.95).is_err());
    }
}
