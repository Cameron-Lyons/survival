use pyo3::prelude::*;
#[pyclass]
pub struct CoxCountOutput {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub nrisk: Vec<i32>,
    #[pyo3(get)]
    pub index: Vec<i32>,
    #[pyo3(get)]
    pub status: Vec<i32>,
}
#[pyfunction]
pub fn coxcount1(
    time: Vec<f64>,
    status: Vec<f64>,
    strata: Vec<i32>,
) -> PyResult<Py<CoxCountOutput>> {
    let time_slice = &time;
    let status_slice = &status;
    let strata_slice = &strata;
    let n = time_slice.len();
    let mut ntime = 0;
    let mut nrow = 0;
    let mut _stratastart = 0;
    let mut nrisk = 0;
    let mut i = 0;
    while i < n {
        if strata_slice[i] == 1 {
            _stratastart = i;
            nrisk = 0;
        }
        nrisk += 1;
        if status_slice[i] == 1.0 {
            let dtime = time_slice[i];
            let mut j = i + 1;
            while j < n
                && (time_slice[j] - dtime).abs() < f64::EPSILON
                && status_slice[j] == 1.0
                && strata_slice[j] == 0
            {
                nrisk += 1;
                j += 1;
            }
            ntime += 1;
            nrow += nrisk;
            i = j - 1;
        }
        i += 1;
    }
    let mut time_vec = Vec::with_capacity(ntime);
    let mut nrisk_vec = Vec::with_capacity(ntime);
    let mut index_vec = Vec::with_capacity(nrow);
    let mut status_vec = Vec::with_capacity(nrow);
    let mut _stratastart = 0;
    let mut i = 0;
    while i < n {
        if strata_slice[i] == 1 {
            _stratastart = i;
        }
        if status_slice[i] == 1.0 {
            let dtime = time_slice[i];
            let mut j = i + 1;
            while j < n
                && (time_slice[j] - dtime).abs() < f64::EPSILON
                && status_slice[j] == 1.0
                && strata_slice[j] == 0
            {
                j += 1;
            }
            for k in _stratastart..i {
                status_vec.push(0);
                index_vec.push((k + 1) as i32);
            }
            for k in i..j {
                status_vec.push(1);
                index_vec.push((k + 1) as i32);
            }
            time_vec.push(dtime);
            nrisk_vec.push((j - _stratastart) as i32);
            i = j - 1;
        }
        i += 1;
    }
    Python::attach(|py| {
        Py::new(
            py,
            CoxCountOutput {
                time: time_vec,
                nrisk: nrisk_vec,
                index: index_vec,
                status: status_vec,
            },
        )
    })
}
#[pyfunction]
pub fn coxcount2(
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    strata: Vec<i32>,
) -> PyResult<Py<CoxCountOutput>> {
    let time1_slice = &time1;
    let time2_slice = &time2;
    let status_slice = &status;
    let sort1_slice = &sort1;
    let sort2_slice = &sort2;
    let strata_slice = &strata;
    let n = time1_slice.len();
    let mut ntime = 0;
    let mut nrow = 0;
    let mut j = 0;
    let mut i = 0;
    let mut nrisk = 0;
    while i < n {
        let iptr = sort2_slice[i];
        if strata_slice[i] == 1 {
            nrisk = 0;
            j = i;
        }
        if status_slice[iptr] == 1.0 {
            let dtime = time2_slice[iptr];
            while j < i && time1_slice[sort1_slice[j]] >= dtime {
                nrisk -= 1;
                j += 1;
            }
            nrisk += 1;
            i += 1;
            while i < n
                && strata_slice[i] == 0
                && (time2_slice[sort2_slice[i]] - dtime).abs() < f64::EPSILON
            {
                nrisk += 1;
                i += 1;
            }
            nrow += nrisk;
            ntime += 1;
        } else {
            nrisk += 1;
            i += 1;
        }
    }
    let mut time_vec = Vec::with_capacity(ntime);
    let mut nrisk_vec = Vec::with_capacity(ntime);
    let mut index_vec = Vec::with_capacity(nrow);
    let mut status_vec = Vec::with_capacity(nrow);
    let mut atrisk = vec![None; n];
    let mut who = Vec::with_capacity(n);
    let mut j = 0;
    let mut i = 0;
    while i < n {
        let iptr = sort2_slice[i];
        if strata_slice[i] == 1 {
            atrisk.iter_mut().for_each(|x| *x = None);
            who.clear();
            j = i;
        }
        if status_slice[iptr] == 0.0 {
            if atrisk[iptr].is_none() {
                atrisk[iptr] = Some(who.len());
                who.push(iptr);
            }
            i += 1;
        } else {
            let dtime = time2_slice[iptr];
            while j < i {
                let jptr = sort1_slice[j];
                if time1_slice[jptr] >= dtime {
                    if let Some(pos) = atrisk[jptr]
                        && pos < who.len()
                    {
                        if let Some(last) = who.pop()
                            && pos < who.len()
                        {
                            who[pos] = last;
                            atrisk[last] = Some(pos);
                        }
                        atrisk[jptr] = None;
                    }
                    j += 1;
                } else {
                    break;
                }
            }
            for &k in &who {
                status_vec.push(0);
                index_vec.push((k + 1) as i32);
            }
            let mut events = vec![iptr];
            i += 1;
            while i < n
                && strata_slice[i] == 0
                && (time2_slice[sort2_slice[i]] - dtime).abs() < f64::EPSILON
            {
                events.push(sort2_slice[i]);
                i += 1;
            }
            for &k in &events {
                status_vec.push(1);
                index_vec.push((k + 1) as i32);
                if atrisk[k].is_none() {
                    atrisk[k] = Some(who.len());
                    who.push(k);
                }
            }
            time_vec.push(dtime);
            nrisk_vec.push(who.len() as i32);
        }
    }
    Python::attach(|py| {
        Py::new(
            py,
            CoxCountOutput {
                time: time_vec,
                nrisk: nrisk_vec,
                index: index_vec,
                status: status_vec,
            },
        )
    })
}
