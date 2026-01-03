use super::common::{build_concordance_result, validate_concordance_inputs};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn node_weight(nwt: &[f64], index: usize, ntree: usize) -> f64 {
    if index >= ntree { 0.0 } else { nwt[index] }
}

pub fn concordance1(y: &[f64], wt: &[f64], indx: &[i32], ntree: i32) -> Vec<f64> {
    let n = wt.len();
    let ntree = ntree as usize;
    let mut count = vec![0.0; 5];
    let mut twt = vec![0.0; 2 * ntree];
    let (time, status) = (&y[0..n], &y[n..2 * n]);
    let mut vss = 0.0;
    let mut i = n as i32 - 1;

    while i >= 0 {
        let mut ndeath = 0.0;
        let mut j = i;

        if status[i as usize] == 1.0 {
            let mut tied_indices = Vec::new();
            let mut temp_j = i;
            while temp_j >= 0
                && status[temp_j as usize] == 1.0
                && time[temp_j as usize] == time[i as usize]
            {
                tied_indices.push(temp_j as usize);
                temp_j -= 1;
            }
            j = temp_j;

            let num_tied = tied_indices.len();

            if num_tied > 10 {
                let parallel_counts: (f64, f64, f64, f64, f64) = tied_indices
                    .par_iter()
                    .enumerate()
                    .map(|(idx_in_tied, &j_idx)| {
                        let mut local_count0 = 0.0;
                        let mut local_count1 = 0.0;
                        let mut local_count2 = 0.0;
                        let mut local_count3 = 0.0;
                        let local_ndeath = wt[j_idx];

                        for &k_idx in tied_indices.iter().skip(idx_in_tied + 1) {
                            local_count3 += wt[j_idx] * wt[k_idx];
                        }

                        let index = indx[j_idx] as usize;
                        local_count2 += wt[j_idx] * node_weight(&twt[ntree..], index, ntree);

                        let mut child = 2 * index + 1;
                        if child < ntree {
                            local_count0 += wt[j_idx] * twt[child];
                        }
                        child += 1;
                        if child < ntree {
                            local_count1 += wt[j_idx] * twt[child];
                        }

                        let mut current = index;
                        while current > 0 {
                            let parent = (current - 1) / 2;
                            if current % 2 == 1 {
                                local_count1 += wt[j_idx] * (twt[parent] - twt[current]);
                            } else {
                                local_count0 += wt[j_idx] * (twt[parent] - twt[current]);
                            }
                            current = parent;
                        }

                        (
                            local_count0,
                            local_count1,
                            local_count2,
                            local_count3,
                            local_ndeath,
                        )
                    })
                    .reduce(
                        || (0.0, 0.0, 0.0, 0.0, 0.0),
                        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
                    );

                count[0] += parallel_counts.0;
                count[1] += parallel_counts.1;
                count[2] += parallel_counts.2;
                count[3] += parallel_counts.3;
                ndeath = parallel_counts.4;
            } else {
                for (idx_in_tied, &j_idx) in tied_indices.iter().enumerate() {
                    ndeath += wt[j_idx];
                    let index = indx[j_idx] as usize;

                    for &k_idx in tied_indices.iter().skip(idx_in_tied + 1) {
                        count[3] += wt[j_idx] * wt[k_idx];
                    }

                    count[2] += wt[j_idx] * node_weight(&twt[ntree..], index, ntree);

                    let mut child = 2 * index + 1;
                    if child < ntree {
                        count[0] += wt[j_idx] * twt[child];
                    }
                    child += 1;
                    if child < ntree {
                        count[1] += wt[j_idx] * twt[child];
                    }

                    let mut current = index;
                    while current > 0 {
                        let parent = (current - 1) / 2;
                        if current % 2 == 1 {
                            count[1] += wt[j_idx] * (twt[parent] - twt[current]);
                        } else {
                            count[0] += wt[j_idx] * (twt[parent] - twt[current]);
                        }
                        current = parent;
                    }
                }
            }

            for &i_idx in tied_indices.iter().rev() {
                let mut wsum1 = 0.0;
                let oldmean = twt[0] / 2.0;
                let index = indx[i_idx] as usize;
                twt[ntree + index] += wt[i_idx];
                twt[index] += wt[i_idx];
                let wsum2 = twt[ntree + index];
                let child = 2 * index + 1;
                if child < ntree {
                    wsum1 += twt[child];
                }
                let mut current = index;
                while current > 0 {
                    let parent = (current - 1) / 2;
                    twt[parent] += wt[i_idx];
                    if current.is_multiple_of(2) {
                        wsum1 += twt[parent] - twt[current];
                    }
                    current = parent;
                }
                let wsum3 = twt[0] - (wsum1 + wsum2);
                let lmean = wsum1 / 2.0;
                let umean = wsum1 + wsum2 + wsum3 / 2.0;
                let newmean = twt[0] / 2.0;
                let myrank = wsum1 + wsum2 / 2.0;
                vss += wsum1 * (newmean + oldmean - 2.0 * lmean) * (newmean - oldmean);
                vss += wsum3 * (newmean + oldmean + wt[i_idx] - 2.0 * umean) * (oldmean - newmean);
                vss += wt[i_idx] * (myrank - newmean).powi(2);
            }
        }

        if twt[0] > 0.0 {
            count[4] += ndeath * vss / twt[0];
        }

        if j == i {
            i -= 1;
        } else {
            i = j;
        }
    }
    count
}

#[pyfunction]
pub fn perform_concordance1_calculation(
    time_data: Vec<f64>,
    weights: Vec<f64>,
    indices: Vec<i32>,
    ntree: i32,
) -> PyResult<Py<PyAny>> {
    let n = weights.len();
    validate_concordance_inputs(time_data.len(), n, indices.len(), weights.len())?;
    let count = concordance1(&time_data, &weights, &indices, ntree);
    Python::attach(|py| {
        let dict = build_concordance_result(py, &count, None, None, None)?;
        dict.bind(py).set_item("counts", count.clone())?;
        Ok(dict.into())
    })
}
