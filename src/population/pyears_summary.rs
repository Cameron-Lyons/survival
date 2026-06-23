use crate::internal::statistical::gamma_inverse_cdf;
use crate::internal::validation::{validate_finite, validate_non_negative};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
#[pyclass(str, from_py_object)]
pub struct PyearsSummary {
    #[pyo3(get)]
    pub total_person_years: f64,
    #[pyo3(get)]
    pub total_events: f64,
    #[pyo3(get)]
    pub total_expected: f64,
    #[pyo3(get)]
    pub n_observations: f64,
    #[pyo3(get)]
    pub offtable: f64,
    #[pyo3(get)]
    pub observed_rate: f64,
    #[pyo3(get)]
    pub expected_rate: f64,
    #[pyo3(get)]
    pub smr: f64,
    #[pyo3(get)]
    pub sir: f64,
}

impl fmt::Display for PyearsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PyearsSummary(person_years={:.2}, events={:.0}, expected={:.2}, SMR={:.3})",
            self.total_person_years, self.total_events, self.total_expected, self.smr
        )
    }
}

#[pymethods]
impl PyearsSummary {
    pub fn to_table(&self) -> String {
        let mut table = String::new();
        table.push_str("Person-Years Summary\n");
        table.push_str("====================\n\n");
        table.push_str(&format!(
            "Total person-years: {:>12.2}\n",
            self.total_person_years
        ));
        table.push_str(&format!(
            "Total observations: {:>12.0}\n",
            self.n_observations
        ));
        table.push_str(&format!("Off-table:          {:>12.2}\n", self.offtable));
        table.push('\n');
        table.push_str(&format!(
            "Observed events:    {:>12.0}\n",
            self.total_events
        ));
        table.push_str(&format!(
            "Expected events:    {:>12.2}\n",
            self.total_expected
        ));
        table.push('\n');
        table.push_str(&format!(
            "Observed rate:      {:>12.6}\n",
            self.observed_rate
        ));
        table.push_str(&format!(
            "Expected rate:      {:>12.6}\n",
            self.expected_rate
        ));
        table.push('\n');
        table.push_str(&format!("SMR (O/E):          {:>12.3}\n", self.smr));
        table.push_str(&format!("SIR (O/E):          {:>12.3}\n", self.sir));
        table
    }
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_same_lengths(
    pyears: &[f64],
    pn: &[f64],
    pcount: &[f64],
    pexpect: &[f64],
) -> PyResult<()> {
    let n = pyears.len();
    if pn.len() != n || pcount.len() != n || pexpect.len() != n {
        return Err(value_error(
            "pyears, pn, pcount, and pexpect must have the same length",
        ));
    }
    Ok(())
}

fn validate_pyears_components(
    pyears: &[f64],
    pn: &[f64],
    pcount: &[f64],
    pexpect: &[f64],
) -> PyResult<()> {
    validate_same_lengths(pyears, pn, pcount, pexpect)?;
    validate_finite(pyears, "pyears")?;
    validate_non_negative(pyears, "pyears")?;
    validate_finite(pn, "pn")?;
    validate_non_negative(pn, "pn")?;
    validate_finite(pcount, "pcount")?;
    validate_non_negative(pcount, "pcount")?;
    validate_finite(pexpect, "pexpect")?;
    validate_non_negative(pexpect, "pexpect")?;
    Ok(())
}

#[pyfunction]
pub fn summary_pyears(
    pyears: Vec<f64>,
    pn: Vec<f64>,
    pcount: Vec<f64>,
    pexpect: Vec<f64>,
    offtable: f64,
) -> PyResult<PyearsSummary> {
    validate_pyears_components(&pyears, &pn, &pcount, &pexpect)?;
    if !offtable.is_finite() || offtable < 0.0 {
        return Err(value_error("offtable must be a finite non-negative value"));
    }

    let total_person_years: f64 = pyears.iter().sum();
    let total_events: f64 = pcount.iter().sum();
    let total_expected: f64 = pexpect.iter().sum();
    let n_observations: f64 = pn.iter().sum();

    let observed_rate = if total_person_years > 0.0 {
        total_events / total_person_years
    } else {
        0.0
    };

    let expected_rate = if total_person_years > 0.0 {
        total_expected / total_person_years
    } else {
        0.0
    };

    let smr = if total_expected > 0.0 {
        total_events / total_expected
    } else {
        f64::NAN
    };

    let sir = smr;

    Ok(PyearsSummary {
        total_person_years,
        total_events,
        total_expected,
        n_observations,
        offtable,
        observed_rate,
        expected_rate,
        smr,
        sir,
    })
}

#[derive(Debug, Clone)]
#[pyclass(str, from_py_object)]
pub struct PyearsCell {
    #[pyo3(get)]
    pub person_years: f64,
    #[pyo3(get)]
    pub n: f64,
    #[pyo3(get)]
    pub events: f64,
    #[pyo3(get)]
    pub expected: f64,
    #[pyo3(get)]
    pub rate: f64,
    #[pyo3(get)]
    pub smr: f64,
}

impl fmt::Display for PyearsCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PyearsCell(py={:.2}, events={:.0}, expected={:.2})",
            self.person_years, self.events, self.expected
        )
    }
}

#[pyfunction]
pub fn pyears_by_cell(
    pyears: Vec<f64>,
    pn: Vec<f64>,
    pcount: Vec<f64>,
    pexpect: Vec<f64>,
) -> PyResult<Vec<PyearsCell>> {
    validate_pyears_components(&pyears, &pn, &pcount, &pexpect)?;
    let n = pyears.len();
    let mut cells = Vec::with_capacity(n);

    for i in 0..n {
        let py = pyears[i];
        let events = pcount[i];
        let expected = pexpect[i];

        let rate = if py > 0.0 { events / py } else { 0.0 };
        let smr = if expected > 0.0 {
            events / expected
        } else {
            f64::NAN
        };

        cells.push(PyearsCell {
            person_years: py,
            n: pn[i],
            events,
            expected,
            rate,
            smr,
        });
    }

    Ok(cells)
}

#[pyfunction]
pub fn pyears_ci(observed: f64, expected: f64, conf_level: f64) -> PyResult<(f64, f64, f64)> {
    if !observed.is_finite() || observed < 0.0 {
        return Err(value_error("observed must be a finite non-negative value"));
    }
    if !expected.is_finite() || expected <= 0.0 {
        return Err(value_error("expected must be a finite positive value"));
    }
    if !conf_level.is_finite() || conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(value_error(
            "conf_level must be a finite value between 0 and 1",
        ));
    }

    let smr = observed / expected;
    let alpha = (1.0 - conf_level) / 2.0;
    let lower = if observed == 0.0 {
        0.0
    } else {
        gamma_inverse_cdf(alpha, observed) / expected
    };
    let upper = gamma_inverse_cdf(1.0 - alpha, observed + 1.0) / expected;

    Ok((smr, lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_pyears() {
        let pyears = vec![100.0, 200.0, 150.0];
        let pn = vec![50.0, 80.0, 60.0];
        let pcount = vec![5.0, 10.0, 7.0];
        let pexpect = vec![4.0, 8.0, 6.0];
        let offtable = 5.0;

        let summary = summary_pyears(pyears, pn, pcount, pexpect, offtable).unwrap();

        assert!((summary.total_person_years - 450.0).abs() < 1e-10);
        assert!((summary.total_events - 22.0).abs() < 1e-10);
        assert!((summary.total_expected - 18.0).abs() < 1e-10);
        assert!((summary.smr - 22.0 / 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_pyears_ci() {
        let (smr, lower, upper) = pyears_ci(20.0, 10.0, 0.95).unwrap();

        assert!((smr - 2.0).abs() < 1e-10);
        assert!((lower - 1.221652).abs() < 1e-6);
        assert!((upper - 3.088838).abs() < 1e-6);

        let (smr, lower, upper) = pyears_ci(0.0, 10.0, 0.95).unwrap();
        assert_eq!(smr, 0.0);
        assert_eq!(lower, 0.0);
        assert!((upper - 0.3688879).abs() < 1e-6);
    }

    #[test]
    fn pyears_helpers_validate_public_inputs() {
        assert!(
            summary_pyears(vec![1.0], vec![], vec![1.0], vec![1.0], 0.0)
                .expect_err("length mismatch should fail")
                .to_string()
                .contains("must have the same length")
        );
        assert!(
            summary_pyears(vec![f64::INFINITY], vec![1.0], vec![1.0], vec![1.0], 0.0)
                .expect_err("non-finite pyears should fail")
                .to_string()
                .contains("pyears contains non-finite")
        );
        assert!(
            pyears_by_cell(vec![1.0], vec![1.0], vec![-1.0], vec![1.0])
                .expect_err("negative event count should fail")
                .to_string()
                .contains("pcount contains negative")
        );
        assert!(
            pyears_ci(1.0, 0.0, 0.95)
                .expect_err("zero expected should fail")
                .to_string()
                .contains("expected must be a finite positive value")
        );
        assert!(
            pyears_ci(1.0, 1.0, 1.0)
                .expect_err("invalid confidence should fail")
                .to_string()
                .contains("conf_level")
        );
    }
}
