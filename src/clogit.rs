// Conditional logistic regression

struct DataSet {
    case_control_status: Vec<u8>, // 0 = control, 1 = case
    strata: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}
