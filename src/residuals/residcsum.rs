#[allow(dead_code)]
pub(crate) fn residcsum(y2: &[f64], strata: &[i32], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut csum = y2.to_vec();

    for j in 0..ncols {
        let mut current_stratum = None;
        let mut temp = 0.0;

        for i in 0..nrows {
            let idx = j * nrows + i;
            let stratum = strata[i];

            if i == 0 || Some(stratum) != current_stratum {
                temp = 0.0;
                current_stratum = Some(stratum);
            }

            temp += csum[idx];
            csum[idx] = temp;
        }
    }

    csum
}
