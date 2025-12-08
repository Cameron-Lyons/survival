#![allow(dead_code)]
pub fn dmatrix(array: &[f64], nrow: usize, ncol: usize) -> Vec<&[f64]> {
    let mut columns = Vec::with_capacity(ncol);
    for col in 0..ncol {
        let start = col * nrow;
        let end = start + nrow;
        columns.push(&array[start..end]);
    }
    columns
}

pub fn imatrix(array: &[i32], nrow: usize, ncol: usize) -> Vec<&[i32]> {
    let mut columns = Vec::with_capacity(ncol);
    for col in 0..ncol {
        let start = col * nrow;
        let end = start + nrow;
        columns.push(&array[start..end]);
    }
    columns
}
