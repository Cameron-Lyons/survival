use coxph::CoxPHModel;
use pyo3::prelude::*;

enum Method {
    Prentice,
    SelfPrentice,
    LinYing,
    IBorgan,
    IIBorgan,
}

#[pyclass]
struct Subject {
    id: usize,
    covariates: Vec<f64>,
    is_case: bool,
    is_subcohort: bool,
    stratum: usize,
}

#[pyclass]
struct CohortData {
    subjects: Vec<Subject>,
}

#[pymethods]
impl CohortData {
    pub fn new() -> CohortData {
        CohortData {
            subjects: Vec::new(),
        }
    }
    pub fn add_subject(&mut self, subject: Subject) {
        self.subjects.push(subject);
    }
    pub fn get_subject(&self, id: usize) -> &Subject {
        &self.subjects[id]
    }
    pub fn fit(&self, method: Method) -> CoxPHModel {
        let mut model = CoxPHModel::new();
        for subject in &self.subjects {
            if subject.is_subcohort {
                model.add_subject(subject);
            }
        }
        model.fit(method);
        model
    }
}

#[pymodule]
fn pyCohortData(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CohortData>()?;
    Ok(())
}
