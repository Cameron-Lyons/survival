use crate::regression::coxph::{CoxPHModel, Subject};
use pyo3::prelude::*;
#[derive(Clone)]
#[pyclass]
pub enum CchMethod {
    Prentice,
    SelfPrentice,
    LinYing,
    IBorgan,
    IIBorgan,
}
#[pyclass]
pub struct CohortData {
    subjects: Vec<Subject>,
}
impl Default for CohortData {
    fn default() -> Self {
        Self::new()
    }
}
#[pymethods]
impl CohortData {
    #[staticmethod]
    pub fn new() -> CohortData {
        CohortData {
            subjects: Vec::new(),
        }
    }
    pub fn add_subject(&mut self, subject: Subject) {
        self.subjects.push(subject);
    }
    pub fn get_subject(&self, id: usize) -> Subject {
        self.subjects[id].clone()
    }
    #[pyo3(signature = (_method, max_iter=100))]
    pub fn fit(&self, _method: CchMethod, max_iter: u16) -> PyResult<CoxPHModel> {
        let mut model = CoxPHModel::new();
        for subject in &self.subjects {
            if subject.is_subcohort {
                model.add_subject(subject)?;
            }
        }
        model.fit(max_iter)?;
        Ok(model)
    }
}
