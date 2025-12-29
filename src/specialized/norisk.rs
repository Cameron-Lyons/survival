use pyo3::prelude::*;
#[pyfunction]
pub fn norisk(
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    sort1: Vec<i32>,
    sort2: Vec<i32>,
    strata: Vec<i32>,
) -> PyResult<Vec<i32>> {
    let time1_slice = &time1;
    let time2_slice = &time2;
    let status_slice = &status;
    let sort1_slice = &sort1;
    let sort2_slice = &sort2;
    let strata_slice = &strata;
    let n = time1_slice.len();
    assert_eq!(time2_slice.len(), n);
    assert_eq!(status_slice.len(), n);
    assert_eq!(sort1_slice.len(), n);
    assert_eq!(sort2_slice.len(), n);
    assert!(strata_slice.iter().all(|&s| s >= 0 && s <= n as i32));
    let mut notused = vec![0; n];
    let mut ndeath = 0;
    let mut istrat = 0;
    let mut j = 0;
    for (i, &sort2_i) in sort2_slice.iter().enumerate() {
        let p2 = sort2_i as usize;
        let dtime = time2_slice[p2];
        if i == strata_slice.get(istrat).copied().unwrap_or(n as i32) as usize {
            while j < i {
                let p1 = sort1_slice[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
            ndeath = 0;
            istrat += 1;
        } else {
            while j < i && time1_slice[sort1_slice[j] as usize] >= dtime {
                let p1 = sort1_slice[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
        }
        ndeath += status_slice[p2];
        if j < n {
            let p1 = sort1_slice[j] as usize;
            notused[p1] = ndeath;
        }
    }
    while j < n {
        let p1 = sort1_slice[j] as usize;
        notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
        j += 1;
    }
    Ok(notused)
}
