// Fits proportional hazards regression model to case-cohort data

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
    id: usize,            // Unique identifier for each subject
    covariates: Vec<f64>, // Covariates (e.g., age, blood pressure, etc.)
    is_case: bool,        // True if the subject is a case, false if a control
    is_subcohort: bool,   // True if the subject is in the sub-cohort
    stratum: usize,       // Stratum identifier, if applicable
}

#[pyclass]
struct CohortData {
    subjects: Vec<Subject>, // Collection of all subjects
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
