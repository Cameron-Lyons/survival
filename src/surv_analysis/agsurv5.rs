#[allow(dead_code)]
pub fn agsurv5(
    n: usize,
    nvar: usize,
    dd: &[i32],
    x1: &[f64],
    x2: &[f64],
    xsum: &[f64],
    xsum2: &[f64],
    sum1: &mut [f64],
    sum2: &mut [f64],
    xbar: &mut [f64],
) {
    for i in 0..n {
        let d = dd[i] as f64;

        if d == 1.0 {
            let temp = 1.0 / x1[i];
            sum1[i] = temp;
            sum2[i] = temp.powi(2);

            for k in 0..nvar {
                let idx = i + n * k;
                xbar[idx] = xsum[idx] * temp.powi(2);
            }
        } else {
            let d_int = dd[i] as i32;
            let mut temp;

            for j in 0..d_int {
                let j_f64 = j as f64;
                temp = 1.0 / (x1[i] - x2[i] * j_f64 / d);

                sum1[i] += temp / d;
                sum2[i] += temp.powi(2) / d;

                for k in 0..nvar {
                    let idx = i + n * k;
                    let weighted_x = xsum[idx] - xsum2[idx] * j_f64 / d;
                    xbar[idx] += (weighted_x * temp.powi(2)) / d;
                }
            }
        }
    }
}
