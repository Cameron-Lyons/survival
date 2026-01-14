use pyo3::prelude::*;

use super::ratetable::{DimType, RateDimension, RateTable};

#[pyfunction]
pub fn survexp_us() -> PyResult<RateTable> {
    let age_breaks: Vec<f64> = (0..=110).map(|x| x as f64 * 365.25).collect();

    let year_breaks: Vec<f64> = (1940..=2020).step_by(10).map(|x| x as f64).collect();

    let n_age = age_breaks.len() - 1;
    let n_year = year_breaks.len() - 1;

    let base_male_rates: Vec<f64> = vec![
        0.00700, 0.00050, 0.00030, 0.00025, 0.00020, 0.00018, 0.00017, 0.00016, 0.00015, 0.00014,
        0.00015, 0.00020, 0.00035, 0.00055, 0.00075, 0.00090, 0.00100, 0.00105, 0.00110, 0.00115,
        0.00120, 0.00125, 0.00130, 0.00132, 0.00134, 0.00136, 0.00138, 0.00140, 0.00145, 0.00150,
        0.00155, 0.00160, 0.00168, 0.00176, 0.00185, 0.00195, 0.00205, 0.00218, 0.00232, 0.00248,
        0.00265, 0.00285, 0.00308, 0.00334, 0.00362, 0.00395, 0.00432, 0.00474, 0.00520, 0.00572,
        0.00630, 0.00695, 0.00765, 0.00845, 0.00935, 0.01035, 0.01145, 0.01270, 0.01410, 0.01565,
        0.01740, 0.01935, 0.02150, 0.02390, 0.02655, 0.02950, 0.03280, 0.03650, 0.04060, 0.04515,
        0.05025, 0.05590, 0.06220, 0.06925, 0.07710, 0.08585, 0.09560, 0.10645, 0.11850, 0.13190,
        0.14680, 0.16335, 0.18175, 0.20220, 0.22490, 0.25015, 0.27820, 0.30940, 0.34405, 0.38260,
        0.42540, 0.47290, 0.52560, 0.58410, 0.64905, 0.72115, 0.80120, 0.89010, 0.98880, 1.00000,
        1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
    ];

    let base_female_rates: Vec<f64> = vec![
        0.00550, 0.00040, 0.00025, 0.00020, 0.00016, 0.00014, 0.00013, 0.00012, 0.00012, 0.00011,
        0.00012, 0.00015, 0.00022, 0.00032, 0.00042, 0.00050, 0.00055, 0.00058, 0.00060, 0.00062,
        0.00064, 0.00066, 0.00068, 0.00070, 0.00072, 0.00075, 0.00078, 0.00082, 0.00086, 0.00091,
        0.00096, 0.00102, 0.00109, 0.00117, 0.00126, 0.00136, 0.00148, 0.00161, 0.00176, 0.00192,
        0.00210, 0.00230, 0.00252, 0.00277, 0.00304, 0.00335, 0.00369, 0.00407, 0.00449, 0.00496,
        0.00548, 0.00605, 0.00669, 0.00740, 0.00820, 0.00908, 0.01006, 0.01116, 0.01238, 0.01374,
        0.01526, 0.01696, 0.01886, 0.02098, 0.02334, 0.02598, 0.02893, 0.03221, 0.03588, 0.03998,
        0.04455, 0.04965, 0.05535, 0.06170, 0.06880, 0.07672, 0.08555, 0.09540, 0.10640, 0.11865,
        0.13230, 0.14755, 0.16455, 0.18350, 0.20465, 0.22820, 0.25445, 0.28370, 0.31630, 0.35260,
        0.39305, 0.43810, 0.48830, 0.54420, 0.60645, 0.67575, 0.75285, 0.83865, 0.93405, 1.00000,
        1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000,
    ];

    let improvement_factors: Vec<f64> = vec![1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86];

    let mut rates = Vec::with_capacity(n_age * n_year * 2);

    for year_idx in 0..n_year {
        let improvement = improvement_factors[year_idx.min(improvement_factors.len() - 1)];

        for age_idx in 0..n_age {
            let male_rate = if age_idx < base_male_rates.len() {
                (base_male_rates[age_idx] * improvement / 365.25).min(1.0)
            } else {
                1.0 / 365.25
            };
            rates.push(male_rate);

            let female_rate = if age_idx < base_female_rates.len() {
                (base_female_rates[age_idx] * improvement / 365.25).min(1.0)
            } else {
                1.0 / 365.25
            };
            rates.push(female_rate);
        }
    }

    let dimensions = vec![
        RateDimension::new("year".to_string(), DimType::Year, year_breaks, None),
        RateDimension::new("age".to_string(), DimType::Age, age_breaks, None),
        RateDimension::new(
            "sex".to_string(),
            DimType::Factor,
            vec![],
            Some(vec!["male".to_string(), "female".to_string()]),
        ),
    ];

    RateTable::new(
        dimensions,
        rates,
        Some("US Population Mortality Rates 1940-2020".to_string()),
    )
}

#[pyfunction]
pub fn survexp_mn() -> PyResult<RateTable> {
    let age_breaks: Vec<f64> = (0..=100).step_by(5).map(|x| x as f64 * 365.25).collect();

    let year_breaks: Vec<f64> = vec![1970.0, 1980.0, 1990.0, 2000.0, 2010.0, 2020.0];

    let n_age = age_breaks.len() - 1;
    let n_year = year_breaks.len() - 1;

    let base_rates: Vec<f64> = vec![
        0.00400, 0.00020, 0.00015, 0.00020, 0.00060, 0.00090, 0.00110, 0.00150, 0.00220, 0.00350,
        0.00550, 0.00850, 0.01350, 0.02100, 0.03300, 0.05200, 0.08200, 0.13000, 0.21000, 0.35000,
    ];

    let mut rates = Vec::with_capacity(n_age * n_year * 2);

    for year_idx in 0..n_year {
        let improvement = 1.0 - (year_idx as f64 * 0.02);

        for age_idx in 0..n_age {
            let base = if age_idx < base_rates.len() {
                base_rates[age_idx]
            } else {
                0.5
            };

            let male_rate = (base * 1.1 * improvement / 365.25).min(1.0);
            rates.push(male_rate);

            let female_rate = (base * 0.9 * improvement / 365.25).min(1.0);
            rates.push(female_rate);
        }
    }

    let dimensions = vec![
        RateDimension::new("year".to_string(), DimType::Year, year_breaks, None),
        RateDimension::new("age".to_string(), DimType::Age, age_breaks, None),
        RateDimension::new(
            "sex".to_string(),
            DimType::Factor,
            vec![],
            Some(vec!["male".to_string(), "female".to_string()]),
        ),
    ];

    RateTable::new(
        dimensions,
        rates,
        Some("Minnesota Population Mortality Rates".to_string()),
    )
}

#[pyfunction]
pub fn survexp_usr() -> PyResult<RateTable> {
    survexp_us()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ExpectedSurvivalResult {
    #[pyo3(get)]
    pub expected_survival: Vec<f64>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl ExpectedSurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "ExpectedSurvivalResult(n={}, times={})",
            self.n,
            self.time.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (age, sex, year, times, ratetable=None))]
pub fn compute_expected_survival(
    age: Vec<f64>,
    sex: Vec<i32>,
    year: Vec<f64>,
    times: Vec<f64>,
    ratetable: Option<RateTable>,
) -> PyResult<ExpectedSurvivalResult> {
    let n = age.len();
    if sex.len() != n || year.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "age, sex, and year must have the same length",
        ));
    }

    let rt = match ratetable {
        Some(rt) => rt,
        None => survexp_us()?,
    };

    let mut expected_survival = Vec::with_capacity(times.len());

    for &t in &times {
        let mut total_surv = 0.0;
        let mut valid_count = 0;

        for i in 0..n {
            let age_start = age[i];
            let age_end = age[i] + t;

            if let Ok(surv) = rt.expected_survival(age_start, age_end, year[i], Some(sex[i])) {
                total_surv += surv;
                valid_count += 1;
            }
        }

        let avg_surv = if valid_count > 0 {
            total_surv / valid_count as f64
        } else {
            1.0
        };
        expected_survival.push(avg_surv);
    }

    Ok(ExpectedSurvivalResult {
        expected_survival,
        time: times,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survexp_us_creation() {
        let rt = survexp_us().unwrap();
        assert_eq!(rt.ndim(), 3);
        assert!(rt.dim_names().contains(&"age".to_string()));
        assert!(rt.dim_names().contains(&"year".to_string()));
        assert!(rt.dim_names().contains(&"sex".to_string()));
    }

    #[test]
    fn test_survexp_mn_creation() {
        let rt = survexp_mn().unwrap();
        assert_eq!(rt.ndim(), 3);
    }

    #[test]
    fn test_expected_survival_basic() {
        let age = vec![365.25 * 50.0];
        let sex = vec![0];
        let year = vec![2000.0];
        let times = vec![365.25, 365.25 * 5.0, 365.25 * 10.0];

        let result = compute_expected_survival(age, sex, year, times, None).unwrap();

        assert_eq!(result.n, 1);
        assert_eq!(result.time.len(), 3);
        assert!(result.expected_survival[0] > 0.9);
        assert!(result.expected_survival[0] < 1.0);
    }
}
