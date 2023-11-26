// Fits proportional hazards regression model to case-cohort data

enum Method {
    Prentice,
    SelfPrentice,
    LinYing,
    IBorgan,
    IIBorgan,
}

struct Subject {
    id: usize,            // Unique identifier for each subject
    covariates: Vec<f64>, // Covariates (e.g., age, blood pressure, etc.)
    is_case: bool,        // True if the subject is a case, false if a control
    is_subcohort: bool,   // True if the subject is in the sub-cohort
    stratum: usize,       // Stratum identifier, if applicable
}

struct CohortData {
    subjects: Vec<Subject>, // Collection of all subjects
}
