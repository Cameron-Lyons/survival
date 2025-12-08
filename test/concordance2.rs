use approx::relative_eq;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct SurvivalData {
    time: Vec<f64>,
    status: Vec<i32>,
}

#[derive(Debug)]
struct ConcordanceResult {
    count: [f64; 5],
    influence: Option<Vec<[f64; 5]>>,
    cvar: f64,
    var: f64,
}

fn aeq(x: &[f64], y: &[f64], tolerance: f64) -> bool {
    x.iter()
        .zip(y.iter())
        .all(|(a, b)| (a - b).abs() < tolerance)
}

fn allpair(time: &[f64], status: &[i32], x: &[f64], wt: &[f64]) -> [f64; 5] {
    let n = status.len();
    let mut count = [0.0; 5];

    for i in 0..n {
        if status[i] != 1 {
            continue;
        }
        let xi = x[i];
        let wi = wt[i];
        let ti = time[i];

        let mut concordant = 0.0;
        let mut discordant = 0.0;
        let mut tied_x = 0.0;
        let mut tied_y = 0.0;
        let mut tied_xy = 0.0;

        for j in 0..n {
            if i == j {
                continue;
            }
            let at_risk = if time[j] > ti {
                true
            } else if (time[j] - ti).abs() < 1e-9 {
                status[j] == 0
            } else {
                false
            };

            if at_risk {
                let xj = x[j];
                let wj = wt[j];
                let diff = xj - xi;
                if diff > 1e-9 {
                    concordant += wj;
                } else if diff < -1e-9 {
                    discordant += wj;
                } else {
                    tied_x += wj;
                }
            }
        }

        for j in (i + 1)..n {
            if status[j] == 1 && (time[j] - ti).abs() < 1e-9 {
                let xj = x[j];
                let wj = wt[j];
                if (xj - xi).abs() < 1e-9 {
                    tied_xy += wj;
                } else {
                    tied_y += wj;
                }
            }
        }

        count[0] += wi * concordant;
        count[1] += wi * discordant;
        count[2] += wi * tied_x;
        count[3] += wi * tied_y;
        count[4] += wi * tied_xy;
    }

    count
}

fn leverage(time: &[f64], status: &[i32], x: &[f64], wt: &[f64], eps: f64) -> Vec<[f64; 5]> {
    let n = time.len();
    let mut influence = vec![[0.0; 5]; n];

    let t2: Vec<f64> = time
        .iter()
        .zip(status.iter())
        .map(|(&t, &s)| if s == 0 { t + eps } else { t })
        .collect();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let comparable = if status[i] == 0 {
                time[j] <= time[i] && status[j] == 1
            } else {
                (status[j] == 0 && t2[j] >= t2[i])
                    || (status[j] == 1 && (t2[j] - t2[i]).abs() > 1e-9)
            };

            if comparable {
                let temp = (x[i] - x[j]).signum();
                let idx = match temp {
                    x if x > 0.0 => 0,
                    x if x < 0.0 => 1,
                    _ => 2,
                };
                influence[i][idx] += wt[j];
            }
        }

        if status[i] == 1 {
            for j in (i + 1)..n {
                if status[j] == 1 && (time[j] - time[i]).abs() < 1e-9 {
                    let xj = x[j];
                    let wj = wt[j];
                    if (xj - x[i]).abs() < 1e-9 {
                        influence[i][4] += wj;
                    } else {
                        influence[i][3] += wj;
                    }
                }
            }
        }
    }

    influence
}

fn concordance(
    data: &SurvivalData,
    x: &[f64],
    weights: Option<&[f64]>,
    influence: i32,
) -> ConcordanceResult {
    let wt = weights.unwrap_or_else(|| vec![1.0; x.len()].as_slice());

    let count = allpair(&data.time, &data.status, x, wt);
    let influence_matrix = if influence > 0 {
        Some(leverage(&data.time, &data.status, x, wt, 1e-5))
    } else {
        None
    };

    let npair = count[0] + count[1] + count[2];
    let d = (count[0] - count[1]).abs();
    let cvar = 0.0; 
    let var = 0.0;

    ConcordanceResult {
        count,
        influence: influence_matrix,
        cvar,
        var,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_tdata() -> (SurvivalData, Vec<f64>, Vec<f64>) {
        let time = vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
        ];
        let status = vec![1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0];
        let x = vec![1.0, 6.0, 2.0, 7.0, 3.0, 7.0, 3.0, 8.0, 4.0, 4.0, 5.0];
        let wt = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let survival_data = SurvivalData { time, status };
        (survival_data, x, wt)
    }

    #[test]
    fn test_allpair() {
        let (tdata, x, wt) = setup_tdata();
        let count = allpair(&tdata.time, &tdata.status, &x, &wt);
        let expected = [91.0, 70.0, 7.0, 0.0, 0.0];
        assert!(aeq(&count, &expected, 1e-9), "Counts do not match");
    }

    #[test]
    fn test_leverage() {
        let (tdata, x, wt) = setup_tdata();
        let influence = leverage(&tdata.time, &tdata.status, &x, &wt, 1e-5);
        assert_eq!(influence.len(), tdata.time.len());
    }

    #[test]
    fn test_concordance() {
        let (tdata, x, wt) = setup_tdata();
        let fit = concordance(&tdata, &x, Some(&wt), 2);
        let expected_count = allpair(&tdata.time, &tdata.status, &x, &wt);
        assert!(
            aeq(&fit.count, &expected_count, 1e-9),
            "Counts do not match"
        );
    }
}
