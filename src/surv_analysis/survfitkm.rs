use std::collections::BTreeSet;

pub struct SurvFitKMOutput {
    pub time: Vec<f64>,
    pub n_risk: Vec<f64>,
    pub n_event: Vec<f64>,
    pub n_censor: Vec<f64>,
    pub estimate: Vec<f64>,
    pub std_err: Vec<f64>,
}

pub fn survfitkm(
    time: &[f64],
    status: &[f64],
    weights: &[f64],
    entry_times: Option<&[f64]>,
    position: &[i32],
    reverse: bool,
    computation_type: i32,
) -> SurvFitKMOutput {
    let mut unique_times = BTreeSet::new();
    for (&t, &s) in time.iter().zip(status) {
        if s > 0.0 {
            unique_times.insert(t);
        }
    }
    let mut dtime: Vec<f64> = unique_times.into_iter().collect();
    dtime.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ntime = dtime.len();
    let mut n_risk = vec![0.0; ntime];
    let mut n_event = vec![0.0; ntime];
    let mut n_censor = vec![0.0; ntime];
    let mut estimate = vec![1.0; ntime];
    let mut std_err = vec![0.0; ntime];

    let mut current_risk = weights.iter().sum();
    let mut current_estimate = 1.0;
    let mut cumulative_variance = 0.0;

    for (i, &t) in dtime.iter().enumerate().rev() {
        let mut events = 0.0;
        let mut censored = 0.0;
        let mut weighted_events = 0.0;
        let mut weighted_risk = current_risk;

        for (j, (&time_j, &status_j)) in time.iter().zip(status).enumerate() {
            if (time_j - t).abs() < 1e-9 {
                if status_j > 0.0 {
                    events += 1.0;
                    weighted_events += weights[j];
                } else if position[j] & 2 != 0 {
                    censored += 1.0;
                }
            }
        }

        if i < ntime - 1 {
            current_risk -= weighted_events + censored;
        }

        n_risk[i] = weighted_risk;
        n_event[i] = weighted_events;
        n_censor[i] = censored;

        if weighted_risk > 0.0 && weighted_events > 0.0 {
            let hazard = weighted_events / weighted_risk;
            current_estimate *= 1.0 - hazard;
            cumulative_variance += hazard / (weighted_risk - weighted_events);
        }

        estimate[i] = current_estimate;
        std_err[i] = (current_estimate * current_estimate * cumulative_variance).sqrt();
    }

    SurvFitKMOutput {
        time: dtime,
        n_risk,
        n_event,
        n_censor,
        estimate,
        std_err,
    }
}

fn process_entry_times(entry_times: Option<&[f64]>, position: &[i32]) -> Vec<f64> {
    let mut entries = Vec::new();
    if let Some(entries) = entry_times {
        for (&time, &pos) in entries.iter().zip(position) {
            if pos & 1 != 0 {
                entries.push(time);
            }
        }
    }
    entries
}
