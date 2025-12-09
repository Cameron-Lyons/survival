use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[allow(dead_code)]
struct BinaryTree {
    nwt: Vec<f64>,
    twt: Vec<f64>,
}

impl BinaryTree {
    #[allow(dead_code)]
    fn new(ntree: usize) -> Self {
        BinaryTree {
            nwt: vec![0.0; ntree],
            twt: vec![0.0; 2 * ntree],
        }
    }
}

#[allow(dead_code)]
fn addin(nwt: &mut [f64], twt: &mut [f64], index: usize, weight: f64) {
    nwt[index] += weight;
    let mut node_index = index;
    while node_index != 0 {
        let parent_index = (node_index - 1) / 2;
        twt[parent_index] += weight;
        node_index = parent_index;
    }
    twt[index] += weight;
}

fn walkup(nwt: &[f64], twt: &[f64], index: usize, ntree: usize) -> [f64; 3] {
    let mut wsum = [0.0; 3];

    if index >= ntree {
        return wsum;
    }

    wsum[2] = nwt[index];

    let mut child = 2 * index + 1;
    if child < ntree {
        wsum[0] += twt[child];
    }

    child += 1;
    if child < ntree {
        wsum[0] += twt[child];
    }

    let mut current = index;
    while current > 0 {
        let parent = (current - 1) / 2;
        wsum[1] += twt[parent] - twt[current];
        current = parent;
    }

    wsum
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
            while j >= 0 && status[j as usize] == 1.0 && time[j as usize] == time[i as usize] {
                let j_idx = j as usize;
                ndeath += wt[j_idx];
                let index = indx[j_idx] as usize;

                for k in (j + 1)..=i {
                    count[3] += wt[j_idx] * wt[k as usize];
                }

                let wsum = walkup(&twt[ntree..], &twt, index, ntree);
                count[2] += wt[j_idx] * wsum[2];

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

                j -= 1;
            }
        }

        for idx in (j + 1)..=i {
            let i_idx = idx as usize;
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

        if twt[0] > 0.0 {
            count[4] += ndeath * vss / twt[0];
        }

        i = j;
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
    if n == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }

    if time_data.len() != 2 * n {
        return Err(PyRuntimeError::new_err(
            "Time data should have 2*n elements (time, status)",
        ));
    }

    if weights.len() != n {
        return Err(PyRuntimeError::new_err(
            "Weights length does not match observations",
        ));
    }

    if indices.len() != n {
        return Err(PyRuntimeError::new_err(
            "Indices length does not match observations",
        ));
    }

    let count = concordance1(&time_data, &weights, &indices, ntree);

    let concordant = count[0];
    let discordant = count[1];
    let tied_x = count[2];
    let tied_y = count[3];
    let tied_xy = count[4];

    let total_pairs = concordant + discordant + tied_x + tied_y + tied_xy;
    let concordance_index = if total_pairs > 0.0 {
        (concordant + 0.5 * (tied_x + tied_y + tied_xy)) / total_pairs
    } else {
        0.0
    };

    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("concordant", concordant).unwrap();
        dict.set_item("discordant", discordant).unwrap();
        dict.set_item("tied_x", tied_x).unwrap();
        dict.set_item("tied_y", tied_y).unwrap();
        dict.set_item("tied_xy", tied_xy).unwrap();
        dict.set_item("concordance_index", concordance_index)
            .unwrap();
        dict.set_item("total_pairs", total_pairs).unwrap();
        dict.set_item("counts", count).unwrap();
        Ok(dict.into())
    })
}
