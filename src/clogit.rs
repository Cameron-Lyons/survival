// Conditional logistic regression

struct DataSet {
    case_control_status: Vec<u8>, // 0 = control, 1 = case
    strata: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}

impl DataSet {
    pub fn new() -> DataSet {
        DataSet {
            case_control_status: Vec::new(),
            strata: Vec::new(),
            covariates: Vec::new(),
        }
    }
    pub fn add_observation(&mut self, case_control_status: u8, stratum: u8, covariates: Vec<f64>) {
        self.case_control_status.push(case_control_status);
        self.strata.push(stratum);
        self.covariates.push(covariates);
    }
    pub fn get_case_control_status(&self, id: usize) -> u8 {
        self.case_control_status[id]
    }
    pub fn get_stratum(&self, id: usize) -> u8 {
        self.strata[id]
    }
    pub fn get_covariates(&self, id: usize) -> &Vec<f64> {
        &self.covariates[id]
    }
    pub fn get_num_observations(&self) -> usize {
        self.case_control_status.len()
    }
    pub fn get_num_covariates(&self) -> usize {
        self.covariates[0].len()
    }
}

struct ConditionalLogisticRegression {
    data: DataSet,
    coefficients: Vec<f64>,
}
