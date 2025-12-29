use pyo3::prelude::*;
#[pyclass]
#[derive(Clone)]
pub struct SplitResult {
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub interval: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub end: Vec<f64>,
    #[pyo3(get)]
    pub censor: Vec<bool>,
}
#[pyfunction]
pub fn survsplit(tstart: Vec<f64>, tstop: Vec<f64>, cut: Vec<f64>) -> SplitResult {
    let n = tstart.len();
    let ncut = cut.len();
    let mut extra = 0;
    for i in 0..n {
        if tstart[i].is_nan() || tstop[i].is_nan() {
            continue;
        }
        for &c in &cut {
            if c > tstart[i] && c < tstop[i] {
                extra += 1;
            }
        }
    }
    let n2 = n + extra;
    let mut result = SplitResult {
        row: Vec::with_capacity(n2),
        interval: Vec::with_capacity(n2),
        start: Vec::with_capacity(n2),
        end: Vec::with_capacity(n2),
        censor: Vec::with_capacity(n2),
    };
    for i in 0..n {
        let current_start = tstart[i];
        let current_stop = tstop[i];
        if current_start.is_nan() || current_stop.is_nan() {
            result.row.push(i + 1);
            result.interval.push(1);
            result.start.push(current_start);
            result.end.push(current_stop);
            result.censor.push(false);
            continue;
        }
        let mut cuts_in_interval: Vec<f64> = cut
            .iter()
            .copied()
            .filter(|&c| c > current_start && c < current_stop)
            .collect();
        cuts_in_interval
            .sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut current = current_start;
        let mut interval_num = 1;
        let mut j = 0;
        while j < ncut && cut[j] <= current_start {
            j += 1;
        }
        while j < ncut && cut[j] < current_stop {
            if cut[j] > current {
                result.row.push(i + 1);
                result.interval.push(interval_num);
                result.start.push(current);
                result.end.push(cut[j]);
                result.censor.push(true);
                current = cut[j];
                interval_num += 1;
            }
            j += 1;
        }
        result.row.push(i + 1);
        result.interval.push(interval_num);
        result.start.push(current);
        result.end.push(current_stop);
        result.censor.push(false);
    }
    result
}
