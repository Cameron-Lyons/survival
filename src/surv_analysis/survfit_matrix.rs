use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvfitMatrixResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub surv: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_states: usize,
}

#[pymethods]
impl SurvfitMatrixResult {
    #[new]
    #[pyo3(signature = (time, surv, cumhaz, std_err=None, n_risk=vec![], n_event=vec![], n_states=1))]
    pub fn new(
        time: Vec<f64>,
        surv: Vec<Vec<f64>>,
        cumhaz: Vec<Vec<f64>>,
        std_err: Option<Vec<Vec<f64>>>,
        n_risk: Vec<f64>,
        n_event: Vec<f64>,
        n_states: usize,
    ) -> Self {
        Self {
            time,
            surv,
            cumhaz,
            std_err,
            n_risk,
            n_event,
            n_states,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvfitMatrixResult(n_times={}, n_states={}, has_stderr={})",
            self.time.len(),
            self.n_states,
            self.std_err.is_some()
        )
    }

    pub fn get_surv_at_state(&self, state: usize) -> PyResult<Vec<f64>> {
        if state >= self.n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "State index {} out of range (n_states={})",
                state, self.n_states
            )));
        }
        Ok(self.surv.iter().map(|row| row[state]).collect())
    }

    pub fn get_cumhaz_at_state(&self, state: usize) -> PyResult<Vec<f64>> {
        if state >= self.n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "State index {} out of range (n_states={})",
                state, self.n_states
            )));
        }
        Ok(self.cumhaz.iter().map(|row| row[state]).collect())
    }
}

#[pyfunction]
#[pyo3(signature = (time, hazard, n_risk=None, n_event=None))]
pub fn survfit_from_hazard(
    time: Vec<f64>,
    hazard: Vec<f64>,
    n_risk: Option<Vec<f64>>,
    n_event: Option<Vec<f64>>,
) -> PyResult<SurvfitMatrixResult> {
    if time.len() != hazard.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and hazard must have the same length",
        ));
    }

    let n = time.len();
    let mut cumhaz = Vec::with_capacity(n);
    let mut surv = Vec::with_capacity(n);

    let mut cum = 0.0;
    for &h in &hazard {
        cum += h;
        cumhaz.push(vec![cum]);
        surv.push(vec![(-cum).exp()]);
    }

    let n_risk_vec = n_risk.unwrap_or_else(|| vec![0.0; n]);
    let n_event_vec = n_event.unwrap_or_else(|| vec![0.0; n]);

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz,
        std_err: None,
        n_risk: n_risk_vec,
        n_event: n_event_vec,
        n_states: 1,
    })
}

#[pyfunction]
#[pyo3(signature = (time, cumhaz, n_risk=None, n_event=None))]
pub fn survfit_from_cumhaz(
    time: Vec<f64>,
    cumhaz: Vec<f64>,
    n_risk: Option<Vec<f64>>,
    n_event: Option<Vec<f64>>,
) -> PyResult<SurvfitMatrixResult> {
    if time.len() != cumhaz.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and cumhaz must have the same length",
        ));
    }

    let n = time.len();
    let surv: Vec<Vec<f64>> = cumhaz.iter().map(|&h| vec![(-h).exp()]).collect();
    let cumhaz_matrix: Vec<Vec<f64>> = cumhaz.iter().map(|&h| vec![h]).collect();

    let n_risk_vec = n_risk.unwrap_or_else(|| vec![0.0; n]);
    let n_event_vec = n_event.unwrap_or_else(|| vec![0.0; n]);

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz: cumhaz_matrix,
        std_err: None,
        n_risk: n_risk_vec,
        n_event: n_event_vec,
        n_states: 1,
    })
}

#[pyfunction]
pub fn survfit_from_matrix(
    time: Vec<f64>,
    hazard_matrix: Vec<Vec<f64>>,
) -> PyResult<SurvfitMatrixResult> {
    if time.is_empty() || hazard_matrix.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and hazard_matrix cannot be empty",
        ));
    }

    let n_times = time.len();
    let n_states = hazard_matrix[0].len();

    if hazard_matrix.len() != n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_matrix rows must match time length",
        ));
    }

    for row in &hazard_matrix {
        if row.len() != n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All rows in hazard_matrix must have the same number of columns",
            ));
        }
    }

    let mut cumhaz = Vec::with_capacity(n_times);
    let mut surv = Vec::with_capacity(n_times);

    let mut cum = vec![0.0; n_states];

    for row in &hazard_matrix {
        for (j, &h) in row.iter().enumerate() {
            cum[j] += h;
        }
        cumhaz.push(cum.clone());
        surv.push(cum.iter().map(|&c| (-c).exp()).collect());
    }

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz,
        std_err: None,
        n_risk: vec![0.0; n_times],
        n_event: vec![0.0; n_times],
        n_states,
    })
}

#[pyfunction]
pub fn survfit_multistate(
    time: Vec<f64>,
    transition_hazards: Vec<Vec<Vec<f64>>>,
    initial_state: usize,
) -> PyResult<SurvfitMatrixResult> {
    if time.is_empty() || transition_hazards.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and transition_hazards cannot be empty",
        ));
    }

    let n_times = time.len();
    let n_states = transition_hazards[0].len();

    if transition_hazards.len() != n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "transition_hazards length must match time length",
        ));
    }

    for haz in &transition_hazards {
        if haz.len() != n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each time point must have n_states x n_states transition matrix",
            ));
        }
        for row in haz {
            if row.len() != n_states {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Transition matrices must be square",
                ));
            }
        }
    }

    if initial_state >= n_states {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "initial_state must be less than n_states",
        ));
    }

    let mut prob = vec![0.0; n_states];
    prob[initial_state] = 1.0;

    let mut surv = Vec::with_capacity(n_times);
    let mut cumhaz = Vec::with_capacity(n_times);

    for haz_matrix in &transition_hazards {
        let mut new_prob = vec![0.0; n_states];

        for i in 0..n_states {
            let mut out_rate = 0.0;
            for j in 0..n_states {
                if i != j {
                    out_rate += haz_matrix[i][j];
                    new_prob[j] += prob[i] * haz_matrix[i][j];
                }
            }
            new_prob[i] += prob[i] * (1.0 - out_rate).max(0.0);
        }

        prob = new_prob;
        surv.push(prob.clone());

        let ch: Vec<f64> = prob
            .iter()
            .map(|&p| if p > 0.0 { -p.ln() } else { f64::INFINITY })
            .collect();
        cumhaz.push(ch);
    }

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz,
        std_err: None,
        n_risk: vec![0.0; n_times],
        n_event: vec![0.0; n_times],
        n_states,
    })
}

#[pyfunction]
pub fn basehaz(
    time: Vec<f64>,
    status: Vec<i32>,
    linear_predictors: Vec<f64>,
    centered: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n = time.len();
    if status.len() != n || linear_predictors.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and linear_predictors must have the same length",
        ));
    }

    let center = if centered {
        linear_predictors.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let risk_scores: Vec<f64> = linear_predictors
        .iter()
        .map(|&lp| (lp - center).exp())
        .collect();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut hazard = Vec::new();

    let mut cumulative_risk: Vec<f64> = vec![0.0; n];
    let mut running_sum = 0.0;
    for i in (0..n).rev() {
        running_sum += risk_scores[indices[i]];
        cumulative_risk[i] = running_sum;
    }

    let mut i = 0;
    let mut cum_hazard = 0.0;

    while i < n {
        let idx = indices[i];
        if status[idx] == 0 {
            i += 1;
            continue;
        }

        let current_time = time[idx];
        let mut events = 0.0;
        let start_i = i;

        while i < n && (time[indices[i]] - current_time).abs() < 1e-9 {
            if status[indices[i]] == 1 {
                events += 1.0;
            }
            i += 1;
        }

        let risk_sum = cumulative_risk[start_i];
        if risk_sum > 0.0 {
            cum_hazard += events / risk_sum;
        }

        unique_times.push(current_time);
        hazard.push(cum_hazard);
    }

    Ok((unique_times, hazard))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survfit_from_hazard() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hazard = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let result = survfit_from_hazard(time, hazard, None, None).unwrap();

        assert_eq!(result.time.len(), 5);
        assert_eq!(result.n_states, 1);
        assert!((result.surv[0][0] - (-0.1_f64).exp()).abs() < 1e-10);
        assert!((result.surv[4][0] - (-0.5_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_survfit_from_cumhaz() {
        let time = vec![1.0, 2.0, 3.0];
        let cumhaz = vec![0.1, 0.3, 0.6];
        let result = survfit_from_cumhaz(time, cumhaz, None, None).unwrap();

        assert_eq!(result.time.len(), 3);
        assert!((result.surv[2][0] - (-0.6_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_survfit_from_matrix() {
        let time = vec![1.0, 2.0, 3.0];
        let hazard_matrix = vec![vec![0.1, 0.05], vec![0.1, 0.05], vec![0.1, 0.05]];
        let result = survfit_from_matrix(time, hazard_matrix).unwrap();

        assert_eq!(result.n_states, 2);
        assert!((result.cumhaz[2][0] - 0.3).abs() < 1e-10);
        assert!((result.cumhaz[2][1] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_basehaz() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let lp = vec![0.0, 0.1, -0.1, 0.2, 0.0];
        let (times, haz) = basehaz(time, status, lp, true).unwrap();

        assert_eq!(times.len(), 3);
        assert_eq!(haz.len(), 3);
        assert!(haz[0] > 0.0);
        assert!(haz[1] > haz[0]);
        assert!(haz[2] > haz[1]);
    }
}
