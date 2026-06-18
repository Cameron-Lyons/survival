use crate::regression::coxph::{CoxPHModel, Subject};
use pyo3::prelude::*;

fn index_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyIndexError::new_err(message.into())
}

#[derive(Clone, Debug)]
#[pyclass(from_py_object)]
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
    pub fn get_subject(&self, index: usize) -> PyResult<Subject> {
        self.subjects.get(index).cloned().ok_or_else(|| {
            index_error(format!(
                "subject index {index} out of range for cohort of size {}",
                self.subjects.len()
            ))
        })
    }
    pub fn __len__(&self) -> usize {
        self.subjects.len()
    }
    pub fn is_empty(&self) -> bool {
        self.subjects.is_empty()
    }
    #[pyo3(signature = (method, max_iter=100))]
    pub fn fit(&self, method: CchMethod, max_iter: u16) -> PyResult<CoxPHModel> {
        if !matches!(method, CchMethod::Prentice) {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "CchMethod::{method:?} is not implemented; only Prentice is currently supported"
            )));
        }

        let mut model = CoxPHModel::new();
        for subject in &self.subjects {
            if subject.is_subcohort || subject.is_case {
                model.add_subject(subject)?;
            }
        }
        model.fit(max_iter)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(id: usize) -> Subject {
        Subject::new(id, vec![id as f64], true, true, 0)
    }

    #[test]
    fn cohort_data_len_and_get_subject_are_safe() {
        let mut cohort = CohortData::new();
        assert_eq!(cohort.__len__(), 0);
        assert!(cohort.is_empty());

        cohort.add_subject(subject(7));
        assert_eq!(cohort.__len__(), 1);
        assert!(!cohort.is_empty());
        assert_eq!(
            cohort
                .get_subject(0)
                .expect("subject at index 0 should exist")
                .id,
            7
        );
        assert!(cohort.get_subject(1).is_err());
    }
}
