pub struct SplitResult {
    pub row: Vec<usize>,
    pub interval: Vec<usize>,
    pub start: Vec<f64>,
    pub end: Vec<f64>,
    pub censor: Vec<bool>,
}

pub fn survsplit(tstart: &[f64], tstop: &[f64], cut: &[f64]) -> SplitResult {
    let n = tstart.len();
    let ncut = cut.len();
    let mut extra = 0;

    for i in 0..n {
        if tstart[i].is_nan() || tstop[i].is_nan() {
            continue;
        }
        for &c in cut {
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

        let mut cuts_in_interval = Vec::new();
        for &c in cut {
            if c > current_start && c < current_stop {
                cuts_in_interval.push(c);
            }
        }
        cuts_in_interval.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
