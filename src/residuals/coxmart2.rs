#![allow(dead_code)]
pub fn coxmart2(
    time: &[f64],
    status: &[i32],
    strata: &[i32],
    score: &[f64],
    wt: &[f64],
    resid: &mut [f64],
) {
    let n = time.len();
    assert_eq!(status.len(), n);
    assert_eq!(strata.len(), n);
    assert_eq!(score.len(), n);
    assert_eq!(wt.len(), n);
    assert_eq!(resid.len(), n);

    let mut denom = 0.0;
    let mut i = 0;
    while i < n {
        if strata[i] == 1 {
            denom = 0.0;
        }

        denom += score[i] * wt[i];
        let mut deaths = status[i] as f64 * wt[i];
        let mut j = i + 1;
        while j < n && time[j] == time[i] && strata[j] == 0 {
            denom += score[j] * wt[j];
            deaths += status[j] as f64 * wt[j];
            j += 1;
        }

        let hazard = if denom == 0.0 { 0.0 } else { deaths / denom };
        resid[j - 1] = hazard;

        i = j;
    }

    let mut expected = 0.0;
    for i in (0..n).rev() {
        expected += resid[i];
        resid[i] = status[i] as f64 - score[i] * expected;
        if strata[i] == 1 {
            expected = 0.0;
        }
    }
}
