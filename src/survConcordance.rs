use pyo3::prelude::*;

#[pyfunction]
fn surv_concordance(
    np: usize,
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n2p: usize,
    x2: &[f64],
    temp: &mut [i32],
    result: &mut [i32],
) {
    let (count1, count2) = temp.split_at_mut(n2p);
    let n = np;
    let n2 = n2p;

    for i in result.iter_mut() {
        *i = 0;
    }

    for count in count1.iter_mut() {
        *count = 0;
    }

    for i in 0..n {
        let mut tdeath = 0;
        let mut nright = 0;
        let mut nsame = 0;

        if status[i] != 0 {
            tdeath = 1;
            nsame = 1;

            for j in (i + 1)..n {
                if time[j] < time[i] {
                    break;
                }
                if status[j] != 0 {
                    nsame += 1;
                    if time[j] == time[i] {
                        tdeath += 1;
                    }
                }
            }

            let mut k = (n2 - 1) / 2;
            let mut start = 0;
            let mut end = n2 - 1;

            count2.copy_from_slice(count1);

            while start < end {
                if x[i] < x2[k] {
                    end = k - 1;
                    k = (start + end) / 2;
                } else {
                    nright += count1[k] - count2[k];
                    if x[i] > x2[k] {
                        start = k + 1;
                        k = (start + end + 1) / 2;
                    } else {
                        break;
                    }
                }
            }

            count2[k] += nsame;
            result[0] += nsame * (count1[k] - count2[k]);
            result[1] += nsame * nright;
            result[2] += tdeath * (n - i - nsame);
            result[3] += tdeath * (nright + count1[k] - count2[k]);
            result[4] += tdeath * (i - nright - count1[k]);
        }
    }
}
