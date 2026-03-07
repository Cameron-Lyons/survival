use crate::utilities::statistical::chi2_sf;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AnovaRow {
    #[pyo3(get)]
    pub model_name: String,
    #[pyo3(get)]
    pub loglik: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub chisq: Option<f64>,
    #[pyo3(get)]
    pub p_value: Option<f64>,
}

#[pymethods]
impl AnovaRow {
    #[new]
    pub fn new(
        model_name: String,
        loglik: f64,
        df: usize,
        chisq: Option<f64>,
        p_value: Option<f64>,
    ) -> Self {
        Self {
            model_name,
            loglik,
            df,
            chisq,
            p_value,
        }
    }

    fn __repr__(&self) -> String {
        match (self.chisq, self.p_value) {
            (Some(chi), Some(p)) => format!(
                "AnovaRow(model='{}', loglik={:.4}, df={}, chisq={:.4}, p={:.4})",
                self.model_name, self.loglik, self.df, chi, p
            ),
            _ => format!(
                "AnovaRow(model='{}', loglik={:.4}, df={})",
                self.model_name, self.loglik, self.df
            ),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AnovaCoxphResult {
    #[pyo3(get)]
    pub rows: Vec<AnovaRow>,
    #[pyo3(get)]
    pub test_type: String,
}

#[pymethods]
impl AnovaCoxphResult {
    #[new]
    pub fn new(rows: Vec<AnovaRow>, test_type: String) -> Self {
        Self { rows, test_type }
    }

    fn __repr__(&self) -> String {
        let rows_str: Vec<String> = self.rows.iter().map(|r| r.__repr__()).collect();
        format!(
            "AnovaCoxphResult(test='{}', models=[\n  {}\n])",
            self.test_type,
            rows_str.join(",\n  ")
        )
    }

    pub fn to_table(&self) -> String {
        let mut table = String::new();
        table.push_str(&format!(
            "Analysis of Deviance Table ({})\n",
            self.test_type
        ));
        table.push_str(&format!(
            "{:<20} {:>12} {:>6} {:>12} {:>12}\n",
            "Model", "loglik", "df", "Chisq", "Pr(>|Chi|)"
        ));
        table.push_str(&"-".repeat(64));
        table.push('\n');

        for row in &self.rows {
            let chisq_str = row
                .chisq
                .map(|c| format!("{:.4}", c))
                .unwrap_or_else(|| "".to_string());
            let p_str = row
                .p_value
                .map(|p| format!("{:.4}", p))
                .unwrap_or_else(|| "".to_string());
            table.push_str(&format!(
                "{:<20} {:>12.4} {:>6} {:>12} {:>12}\n",
                row.model_name, row.loglik, row.df, chisq_str, p_str
            ));
        }
        table
    }
}

#[pyfunction]
#[pyo3(signature = (logliks, dfs, model_names=None, test="LRT".to_string()))]
pub fn anova_coxph(
    logliks: Vec<f64>,
    dfs: Vec<usize>,
    model_names: Option<Vec<String>>,
    test: String,
) -> PyResult<AnovaCoxphResult> {
    if logliks.len() != dfs.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "logliks and dfs must have the same length",
        ));
    }

    if logliks.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Need at least 2 models for comparison",
        ));
    }

    let names = model_names.unwrap_or_else(|| {
        (1..=logliks.len())
            .map(|i| format!("Model {}", i))
            .collect()
    });

    if names.len() != logliks.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "model_names must match logliks length",
        ));
    }

    let mut rows = Vec::with_capacity(logliks.len());

    rows.push(AnovaRow {
        model_name: names[0].clone(),
        loglik: logliks[0],
        df: dfs[0],
        chisq: None,
        p_value: None,
    });

    for i in 1..logliks.len() {
        let chisq = 2.0 * (logliks[i] - logliks[i - 1]);
        let df_diff = dfs[i].abs_diff(dfs[i - 1]);

        let p_value = if df_diff > 0 && chisq >= 0.0 {
            chi2_sf(chisq, df_diff)
        } else {
            f64::NAN
        };

        rows.push(AnovaRow {
            model_name: names[i].clone(),
            loglik: logliks[i],
            df: dfs[i],
            chisq: Some(chisq),
            p_value: Some(p_value),
        });
    }

    Ok(AnovaCoxphResult {
        rows,
        test_type: test,
    })
}

#[pyfunction]
pub fn anova_coxph_single(
    loglik_null: f64,
    loglik_full: f64,
    df_null: usize,
    df_full: usize,
) -> PyResult<AnovaCoxphResult> {
    anova_coxph(
        vec![loglik_null, loglik_full],
        vec![df_null, df_full],
        Some(vec!["Null".to_string(), "Full".to_string()]),
        "LRT".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxph::CoxPHModel;
    use crate::validation::model_selection::compare_models;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn synthetic_nested_cox_data() -> (Vec<f64>, Vec<u8>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![
                19.0, 4.0, 30.0, 15.0, 6.0, 24.0, 9.0, 21.0, 25.0, 16.0, 14.0, 26.0, 12.0, 17.0,
                29.0, 23.0, 27.0, 13.0, 28.0, 3.0, 7.0, 2.0, 20.0, 5.0, 1.0, 8.0, 22.0, 10.0, 11.0,
                18.0,
            ],
            vec![
                0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
                1, 1,
            ],
            vec![
                -0.22147620175631144,
                0.7602932575724843,
                -0.9231655212836085,
                -0.069374019584312,
                0.6597046787856315,
                -0.7463730340538133,
                0.42097512302323836,
                -0.3437683161794367,
                -0.9513974428461089,
                -0.05255013822137822,
                0.04338547789066993,
                -0.9168274986523088,
                0.13183870710356138,
                -0.3051323240843211,
                -0.991013585487406,
                -0.6184532386468693,
                -0.7783786554426502,
                0.08124390946480986,
                -0.9137596739153231,
                0.8562651401961585,
                0.6901239447967451,
                0.8905952874629122,
                -0.3703979344271331,
                0.8105347771237885,
                0.9686248450400701,
                0.5294628685052931,
                -0.449834772848833,
                0.3417786082943073,
                0.19132630746795987,
                -0.19159339567111333,
            ],
            vec![
                -0.3378174893476555,
                0.4962021719131225,
                -1.147936036429733,
                -0.2890003259812402,
                0.6482403975782177,
                -0.6612370032176764,
                0.5794161944899268,
                -0.6157400601961036,
                -0.7571415192786584,
                -0.32646740424680315,
                0.07635357592547731,
                -0.7703387887675024,
                0.21057153018009817,
                -0.03532511867451987,
                -1.0841945736197047,
                -0.5669232255043457,
                -1.0286992178020604,
                0.11712186222418908,
                -0.7257803932695498,
                0.6772278484914469,
                0.5467026450170536,
                1.0108386715947284,
                -0.5180687542654951,
                0.6660820615366315,
                1.2299340178036509,
                0.8285886870111618,
                -0.6567157144335142,
                0.5818760406491075,
                0.22296219905220327,
                -0.4684327102248028,
            ],
            vec![
                0.17100543047437067,
                0.28309930134150996,
                -0.9324086025957814,
                0.5153838443172007,
                0.63560028294837,
                -0.8567135156260877,
                0.2967998801323577,
                -0.0869050381944676,
                -0.5225574253161578,
                -0.0826592367550314,
                -0.6812205804954357,
                -0.3326681865354073,
                0.310414599401895,
                -0.04702888769625324,
                0.11184018935508333,
                0.08688558760906062,
                0.6411884802232783,
                -0.3132344036927748,
                0.6259241815636314,
                -0.840025825739185,
                -0.1445339082547894,
                -0.295359766926256,
                -0.09683872258950199,
                0.6670196410725331,
                0.024798800975902147,
                0.9744932925896734,
                0.7229214405502136,
                -0.7623065093739558,
                -0.36621692887666457,
                -0.9545489969462266,
            ],
        )
    }

    #[test]
    fn test_anova_coxph() {
        let logliks = vec![-100.0, -95.0, -90.0];
        let dfs = vec![0, 1, 2];
        let result = anova_coxph(logliks, dfs, None, "LRT".to_string()).unwrap();

        assert_eq!(result.rows.len(), 3);
        assert!(result.rows[0].chisq.is_none());
        assert!(result.rows[1].chisq.is_some());
        assert!((result.rows[1].chisq.unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_anova_coxph_matches_current_nested_models() {
        let (time, status, x1, x2, x3) = synthetic_nested_cox_data();
        let covariate_sets = vec![
            x1.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
            x1.iter()
                .zip(x2.iter())
                .map(|(&a, &b)| vec![a, b])
                .collect::<Vec<_>>(),
            x1.iter()
                .zip(x2.iter())
                .zip(x3.iter())
                .map(|((&a, &b), &c)| vec![a, b, c])
                .collect::<Vec<_>>(),
        ];

        let mut logliks = Vec::new();
        for covariates in covariate_sets {
            let mut model = CoxPHModel::new_with_data(covariates, time.clone(), status.clone());
            model.fit(100).expect("nested Cox model should fit");
            logliks.push(model.log_likelihood());
        }

        assert!(logliks.windows(2).all(|pair| pair[1] >= pair[0]));

        let model_names = vec![
            "x1".to_string(),
            "x1 + x2".to_string(),
            "x1 + x2 + x3".to_string(),
        ];
        let dfs = vec![1, 2, 3];
        let anova = anova_coxph(
            logliks.clone(),
            dfs.clone(),
            Some(model_names.clone()),
            "LRT".to_string(),
        )
        .expect("anova should succeed for nested Cox models");

        assert_eq!(anova.rows.len(), 3);
        assert!(anova.rows[0].chisq.is_none());

        let step_12 = 2.0 * (logliks[1] - logliks[0]);
        let step_23 = 2.0 * (logliks[2] - logliks[1]);
        assert_close(anova.rows[1].chisq.expect("step 1 chisq"), step_12, 1e-10);
        assert_close(anova.rows[2].chisq.expect("step 2 chisq"), step_23, 1e-10);

        let single = anova_coxph_single(logliks[1], logliks[2], dfs[1], dfs[2])
            .expect("single-step anova should succeed");
        assert_close(
            single.rows[1].chisq.expect("single-step chisq"),
            anova.rows[2].chisq.expect("sequential chisq"),
            1e-10,
        );
        assert_close(
            single.rows[1].p_value.expect("single-step p-value"),
            anova.rows[2].p_value.expect("sequential p-value"),
            1e-10,
        );

        let comparison = compare_models(model_names, logliks.clone(), dfs, time.len())
            .expect("model comparison should succeed");
        assert_eq!(comparison.likelihood_ratio_tests.len(), 3);

        let lr_12 = &comparison.likelihood_ratio_tests[0];
        assert_eq!(lr_12.0, "x1");
        assert_eq!(lr_12.1, "x1 + x2");
        assert_close(lr_12.2, anova.rows[1].chisq.expect("step 1 chisq"), 1e-10);
        assert_close(
            lr_12.4,
            anova.rows[1].p_value.expect("step 1 p-value"),
            1e-10,
        );

        let lr_23 = &comparison.likelihood_ratio_tests[2];
        assert_eq!(lr_23.0, "x1 + x2");
        assert_eq!(lr_23.1, "x1 + x2 + x3");
        assert_close(lr_23.2, anova.rows[2].chisq.expect("step 2 chisq"), 1e-10);
        assert_close(
            lr_23.4,
            anova.rows[2].p_value.expect("step 2 p-value"),
            1e-10,
        );
    }
}
