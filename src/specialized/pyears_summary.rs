use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
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

#[pymethods]
impl PyearsSummary {
    fn __repr__(&self) -> String {
        format!(
            "PyearsSummary(person_years={:.2}, events={:.0}, expected={:.2}, SMR={:.3})",
            self.total_person_years, self.total_events, self.total_expected, self.smr
        )
    }

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

#[pyfunction]
pub fn summary_pyears(
    pyears: Vec<f64>,
    pn: Vec<f64>,
    pcount: Vec<f64>,
    pexpect: Vec<f64>,
    offtable: f64,
) -> PyResult<PyearsSummary> {
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
#[pyclass]
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

#[pymethods]
impl PyearsCell {
    fn __repr__(&self) -> String {
        format!(
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
    let smr = if expected > 0.0 {
        observed / expected
    } else {
        f64::NAN
    };

    let z = if conf_level >= 0.99 {
        2.576
    } else if conf_level >= 0.95 {
        1.96
    } else {
        1.645
    };

    let se_log = if observed > 0.0 {
        1.0 / observed.sqrt()
    } else {
        f64::INFINITY
    };

    let lower = if observed > 0.0 {
        smr * (-z * se_log).exp()
    } else {
        0.0
    };

    let upper = smr * (z * se_log).exp();

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
        assert!(lower < smr);
        assert!(upper > smr);
    }
}
