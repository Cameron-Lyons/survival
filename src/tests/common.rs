#![cfg(test)]
#![allow(dead_code)]

pub const STRICT_TOL: f64 = 1e-4;
pub const STANDARD_TOL: f64 = 1e-3;
pub const LOOSE_TOL: f64 = 0.05;

pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() < tol
}

pub fn rel_approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let max_abs = a.abs().max(b.abs());
    if max_abs < 1e-10 {
        return true;
    }
    (a - b).abs() / max_abs < rel_tol
}

pub fn aml_maintained() -> (Vec<f64>, Vec<i32>) {
    (
        vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
        ],
        vec![1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    )
}

pub fn aml_nonmaintained() -> (Vec<f64>, Vec<i32>) {
    (
        vec![
            5.0, 5.0, 8.0, 8.0, 12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
        ],
        vec![1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    )
}

pub fn aml_combined() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let (t1, s1) = aml_maintained();
    let (t2, s2) = aml_nonmaintained();
    let mut time = t1.clone();
    time.extend(t2.clone());
    let mut status = s1.clone();
    status.extend(s2.clone());
    let mut group = vec![1i32; t1.len()];
    group.extend(vec![0i32; t2.len()]);
    (time, status, group)
}

pub fn aml_combined_sorted() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let (time, status, group) = aml_combined();
    let mut indices: Vec<usize> = (0..time.len()).collect();
    indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
    let time_sorted: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
    let status_sorted: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
    let group_sorted: Vec<i32> = indices.iter().map(|&i| group[i]).collect();
    (time_sorted, status_sorted, group_sorted)
}

pub struct LungData {
    pub time: Vec<f64>,
    pub status: Vec<i32>,
    pub sex: Vec<i32>,
    pub age: Vec<f64>,
    pub ph_ecog: Vec<i32>,
}

pub fn lung_data() -> LungData {
    LungData {
        time: vec![
            306.0, 455.0, 1010.0, 210.0, 883.0, 1022.0, 310.0, 361.0, 218.0, 166.0, 170.0, 654.0,
            728.0, 71.0, 567.0, 144.0, 613.0, 707.0, 61.0, 88.0,
        ],
        status: vec![2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        sex: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        age: vec![
            74.0, 68.0, 56.0, 57.0, 60.0, 74.0, 68.0, 71.0, 53.0, 61.0, 63.0, 52.0, 47.0, 60.0,
            75.0, 77.0, 64.0, 58.0, 64.0, 71.0,
        ],
        ph_ecog: vec![1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2],
    }
}

pub fn lung_subset() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    (
        vec![
            306.0, 455.0, 1010.0, 210.0, 883.0, 1022.0, 310.0, 361.0, 218.0, 166.0, 170.0, 654.0,
            728.0, 71.0, 567.0, 144.0, 613.0, 707.0, 61.0, 88.0,
        ],
        vec![1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    )
}

pub fn ovarian_data() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    (
        vec![
            59.0, 115.0, 156.0, 421.0, 431.0, 448.0, 464.0, 475.0, 477.0, 563.0, 638.0, 744.0,
            769.0, 770.0, 803.0, 855.0, 1040.0, 1106.0, 1129.0, 1206.0, 268.0, 329.0, 353.0, 365.0,
            377.0, 506.0,
        ],
        vec![
            1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        ],
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        ],
    )
}
