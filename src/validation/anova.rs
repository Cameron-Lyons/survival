use crate::utilities::statistical::chi2_sf;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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

#[allow(dead_code)]
pub fn compute_anova_lrt(loglik1: f64, loglik2: f64, df1: usize, df2: usize) -> (f64, f64, usize) {
    let chisq = 2.0 * (loglik2 - loglik1).abs();
    let df_diff = df2.abs_diff(df1);
    let p_value = if df_diff > 0 {
        chi2_sf(chisq, df_diff)
    } else {
        1.0
    };
    (chisq, p_value, df_diff)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_compute_anova_lrt() {
        let (chisq, p_value, df) = compute_anova_lrt(-100.0, -95.0, 0, 1);
        assert!((chisq - 10.0).abs() < 1e-10);
        assert_eq!(df, 1);
        assert!(p_value < 0.01);
    }
}
