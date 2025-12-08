pub fn survfit4(dd: &[i32], x1: &mut [f64], x2: &mut [f64]) {
    assert_eq!(dd.len(), x1.len(), "dd and x1 must have the same length");
    assert_eq!(dd.len(), x2.len(), "dd and x2 must have the same length");

    for i in 0..dd.len() {
        let d = dd[i];
        match d {
            0 => {
                x1[i] = 1.0;
                x2[i] = 1.0;
            }
            1 => {
                let inv = 1.0 / x1[i];
                x1[i] = inv;
                x2[i] = inv.powi(2);
            }
            _ => {
                let mut sum = 1.0 / x1[i];
                let mut sum_sq = sum.powi(2);
                let d_f64 = d as f64;

                for j in 1..d {
                    let j_f64 = j as f64;
                    let denominator = x1[i] - x2[i] * j_f64 / d_f64;
                    let term = 1.0 / denominator;
                    sum += term;
                    sum_sq += term.powi(2);
                }

                x1[i] = sum / d_f64;
                x2[i] = sum_sq / d_f64;
            }
        }
    }
}
