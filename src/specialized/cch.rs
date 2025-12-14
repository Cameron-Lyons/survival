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
    pub fn fit(&self, _method: CchMethod) -> CoxPHModel {
        let mut model = CoxPHModel::new();
        for subject in &self.subjects {
            if subject.is_subcohort {
                model.add_subject(subject);
            }
        }
        model.fit(100);
        model
    }
}
