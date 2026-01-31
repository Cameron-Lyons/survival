use pyo3::prelude::*;
#[pyfunction]
pub fn tmerge(
    id: Vec<i32>,
    time1: Vec<f64>,
    newx: Vec<f64>,
    nid: Vec<i32>,
    ntime: Vec<f64>,
    x: Vec<f64>,
) -> Vec<f64> {
    let n1 = id.len();
    let n2 = nid.len();
    let mut result = newx;
    let mut k = 0;
    let mut current_id = -1;
    let mut csum = 0.0;
    let mut has_one = false;
    for i in 0..n1 {
        if id[i] != current_id {
            current_id = id[i];
            csum = 0.0;
            has_one = false;
            while k < n2 && nid[k] < current_id {
                k += 1;
            }
        }
        let start_time = time1[i];
        let mut local_k = k;
        while local_k < n2 && nid[local_k] == current_id && ntime[local_k] <= start_time {
            csum += x[local_k];
            has_one = true;
            local_k += 1;
        }
        if has_one {
            result[i] = if result[i].is_nan() {
                csum
            } else {
                result[i] + csum
            };
        }
    }
    result
}
#[pyfunction]
pub fn tmerge2(id: Vec<i32>, time1: Vec<f64>, nid: Vec<i32>, ntime: Vec<f64>) -> Vec<usize> {
    let n1 = id.len();
    let n2 = nid.len();
    let mut result = vec![0; n1];
    let mut k = 0;
    for i in 0..n1 {
        let current_id = id[i];
        let start_time = time1[i];
        result[i] = 0;
        while k < n2 && nid[k] < current_id {
            k += 1;
        }
        let mut last_valid = 0;
        let mut local_k = k;
        while local_k < n2 && nid[local_k] == current_id && ntime[local_k] <= start_time {
            last_valid = local_k + 1;
            local_k += 1;
        }
        result[i] = last_valid;
    }
    result
}
#[pyfunction]
pub fn tmerge3(id: Vec<i32>, miss: Vec<bool>) -> Vec<usize> {
    let n = id.len();
    let mut result = vec![0; n];
    let mut last_good = 0;
    let mut current_id = -1;
    for (i, (&current, is_missing)) in id.iter().zip(miss).enumerate() {
        if current != current_id {
            current_id = current;
            last_good = 0;
        }
        if is_missing {
            result[i] = last_good;
        } else {
            result[i] = i + 1;
            last_good = i + 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmerge_single_id_cumulative_sum() {
        let result = tmerge(
            vec![1, 1],
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![1],
            vec![0.5],
            vec![10.0],
        );
        assert!((result[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge_multiple_ids_dont_mix() {
        let result = tmerge(
            vec![1, 2],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1, 2],
            vec![0.5, 0.5],
            vec![10.0, 20.0],
        );
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge_nan_replacement() {
        let result = tmerge(
            vec![1, 1],
            vec![1.0, 2.0],
            vec![f64::NAN, f64::NAN],
            vec![1],
            vec![0.5],
            vec![5.0],
        );
        assert!((result[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge2_basic_matching() {
        let result = tmerge2(vec![1, 1], vec![1.0, 2.0], vec![1, 1], vec![0.5, 1.5]);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn tmerge2_no_matches() {
        let result = tmerge2(vec![1], vec![0.0], vec![2], vec![1.0]);
        assert_eq!(result[0], 0);
    }

    #[test]
    fn tmerge3_no_missing() {
        let result = tmerge3(vec![1, 1, 1], vec![false, false, false]);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn tmerge3_missing_uses_last_good() {
        let result = tmerge3(vec![1, 1, 1], vec![false, true, false]);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 3);
    }

    #[test]
    fn tmerge3_id_transition_resets() {
        let result = tmerge3(vec![1, 2, 2], vec![false, true, false]);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 3);
    }
}
