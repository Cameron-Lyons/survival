//! Person-years tabulation: a port of R survival's `src/pyears1.c` (with
//! `src/pyears2.c` as its no-rate-table case) and of the data side of
//! `R/pyears.R` that runs after `model.frame`.
//!
//! Follow-up time is split across the cells of the observed table (factor
//! or `tcut` categories) and, when a rate table is supplied, the expected
//! number of events or expected person-years in each cell are accumulated
//! from the table's hazards.  Cells are stored in R's column-major order.

use super::match_ratetable::align_us_year_axis;
use super::pystep::{PystepTable, pystep};
use super::ratetable::RateTable;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use ndarray::Array2;
use pyo3::prelude::*;

/// Round-off protection of `pyears1.c`: events are counted in the last
/// cell that received person-years, so a follow-up that ends exactly on a
/// boundary must not spill into the next cell.
const EPS_FRACTION: f64 = 1e-8;

/// Follow-up as R's `pyears` sees the response: `stop` is the time (the
/// second column of a counting-process `Surv`), `start` the entry time of
/// counting-process data and `event` the status column of a `Surv`
/// response (absent for a plain numeric response).
#[derive(Debug, Clone, PartialEq)]
pub struct PyearsFollowup {
    pub start: Option<Vec<f64>>,
    pub stop: Vec<f64>,
    pub event: Option<Vec<f64>>,
}

impl PyearsFollowup {
    fn validate(&self) -> SurvivalResult<usize> {
        let n = self.stop.len();
        validate_finite(&self.stop, "time")?;
        if let Some(start) = &self.start {
            validate_length(n, start.len(), "start")?;
            validate_finite(start, "start")?;
        } else {
            validate_non_negative(&self.stop, "time")?;
        }
        if let Some(event) = &self.event {
            validate_length(n, event.len(), "event")?;
            validate_finite(event, "event")?;
        }
        Ok(n)
    }
}

/// The observed (output) table: one entry per formula term.  A `tcut`
/// term is time based (`factor == 0`) with `dims[i] + 1` cutpoints; a
/// factor term (`factor == 1`) has no cutpoints and its `data` column holds
/// R's one-based level codes.  `data` is `n x ndim`.  An empty table (no
/// terms) tabulates everything into a single cell.
#[derive(Debug, Clone, PartialEq)]
pub struct PyearsCategories {
    pub factors: Vec<i32>,
    pub dims: Vec<usize>,
    pub cuts: Vec<Vec<f64>>,
    pub data: Array2<f64>,
}

impl PyearsCategories {
    fn validate(&self, n: usize) -> SurvivalResult<usize> {
        let ndim = self.factors.len();
        validate_length(ndim, self.dims.len(), "dims")?;
        validate_length(ndim, self.cuts.len(), "cuts")?;
        if self.data.nrows() != n || self.data.ncols() != ndim {
            return Err(SurvivalError::invalid_input(format!(
                "category data must be {n} x {ndim}, got {} x {}",
                self.data.nrows(),
                self.data.ncols()
            )));
        }
        for (d, (&factor, &dim)) in self.factors.iter().zip(&self.dims).enumerate() {
            if dim == 0 {
                return Err(SurvivalError::invalid_input(
                    "every category dimension must have at least one level",
                ));
            }
            match factor {
                1 => {
                    if self.data.column(d).iter().any(|&code| {
                        !code.is_finite() || code.fract() != 0.0 || code < 1.0 || code > dim as f64
                    }) {
                        return Err(SurvivalError::invalid_input(format!(
                            "factor codes of category {} must be integers between 1 and {dim}",
                            d + 1
                        )));
                    }
                }
                0 => {
                    let cuts = &self.cuts[d];
                    validate_length(dim + 1, cuts.len(), "cutpoints")?;
                    validate_finite(cuts, "cutpoints")?;
                    if cuts.windows(2).any(|w| w[1] <= w[0]) {
                        return Err(SurvivalError::invalid_input(
                            "tcut cutpoints must be strictly increasing",
                        ));
                    }
                    if self.data.column(d).iter().any(|v| !v.is_finite()) {
                        return Err(SurvivalError::invalid_input("tcut values must be finite"));
                    }
                }
                _ => {
                    return Err(SurvivalError::invalid_input(
                        "category factor flags must be 0 (tcut) or 1 (factor)",
                    ));
                }
            }
        }
        Ok(self.dims.iter().product())
    }
}

/// Which expected quantity to accumulate (R's `expect` argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyearsExpect {
    /// Expected number of events: the integrated hazard.
    Event,
    /// Expected person-years: the integrated expected survival.
    Pyears,
}

impl PyearsExpect {
    fn parse(value: &str) -> SurvivalResult<Self> {
        match value {
            "event" => Ok(Self::Event),
            "pyears" => Ok(Self::Pyears),
            _ => Err(SurvivalError::invalid_input(
                "expect must be 'event' or 'pyears'",
            )),
        }
    }
}

/// The per-cell tables of R's `pyears` (column-major over `dims`).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct PyearsResult {
    /// Person-years per cell, divided by `scale`.
    #[pyo3(get)]
    pub pyears: Vec<f64>,
    /// Number of observations contributing to each cell.
    #[pyo3(get)]
    pub n: Vec<f64>,
    /// Observed events per cell (a `Surv` response only).
    #[pyo3(get)]
    pub event: Option<Vec<f64>>,
    /// Expected events (or expected person-years) per cell (rate table only).
    #[pyo3(get)]
    pub expected: Option<Vec<f64>>,
    /// Person-years that fell outside the observed table, divided by `scale`.
    #[pyo3(get)]
    pub offtable: f64,
    /// Extent of each category dimension.
    #[pyo3(get)]
    pub dims: Vec<usize>,
    /// Number of observations tabulated.
    #[pyo3(get)]
    pub observations: usize,
}

/// A rate table together with the matched starting positions of every
/// subject in it (`match.ratetable`'s `R`, one row per subject).
pub struct PyearsRatetable<'a> {
    pub table: &'a RateTable,
    pub positions: Array2<f64>,
}

/// Move the time-based coordinates of a table position forward.
fn advance(position: &mut [f64], factors: &[i32], time: f64) {
    for (value, &factor) in position.iter_mut().zip(factors) {
        if factor != 1 {
            *value += time;
        }
    }
}

/// Raw accumulators of `pyears1.c`.
struct PyearsCells {
    pyears: Vec<f64>,
    pn: Vec<f64>,
    pcount: Vec<f64>,
    pexpect: Vec<f64>,
    offtable: f64,
}

/// `pyears1.c`: `expected` is `None` for the `pyears2.c` case.
fn pyears1(
    followup: &PyearsFollowup,
    weights: &[f64],
    categories: &PyearsCategories,
    expected: Option<(&PystepTable<'_>, &[f64], &Array2<f64>)>,
    method: PyearsExpect,
    n_cells: usize,
) -> PyearsCells {
    let n = followup.stop.len();
    let start = followup.start.as_deref();
    let timeleft_of = |i: usize| start.map_or(followup.stop[i], |s| followup.stop[i] - s[i]);
    let odim = categories.factors.len();
    let ocut_slices: Vec<&[f64]> = categories.cuts.iter().map(Vec::as_slice).collect();
    let observed = PystepTable {
        factors: &categories.factors,
        dims: &categories.dims,
        cuts: &ocut_slices,
        edge: false,
    };
    let edim = expected.map_or(0, |(table, _, _)| table.factors.len());

    // eps = min(time[time > 0]) * 1e-8, zero when every follow-up is zero.
    let eps = (0..n)
        .map(timeleft_of)
        .filter(|&t| t > 0.0)
        .fold(0.0, |acc: f64, t| if acc == 0.0 { t } else { acc.min(t) })
        * EPS_FRACTION;

    let mut cells = PyearsCells {
        pyears: vec![0.0; n_cells],
        pn: vec![0.0; n_cells],
        pcount: vec![0.0; n_cells],
        pexpect: vec![0.0; n_cells],
        offtable: 0.0,
    };
    let mut data = vec![0.0; odim];
    let mut data2 = vec![0.0; edim];
    for i in 0..n {
        let entry = start.map_or(0.0, |s| s[i]);
        for (j, value) in data.iter_mut().enumerate() {
            *value = categories.data[[i, j]];
            if categories.factors[j] != 1 {
                *value += entry;
            }
        }
        if let Some((table, _, positions)) = expected {
            for (j, value) in data2.iter_mut().enumerate() {
                *value = positions[[i, j]];
                if table.factors[j] != 1 {
                    *value += entry;
                }
            }
        }
        let mut timeleft = timeleft_of(i);
        let mut cumhaz: f64 = 0.0;
        let mut index = None;

        if timeleft <= eps && followup.event.is_some() {
            // Call pystep at least once so the event lands in a cell.
            index = pystep(&observed, &data, 1.0).index;
        }

        while timeleft > eps {
            let step = pystep(&observed, &data, timeleft);
            let thiscell = step.time;
            index = step.index;
            if let Some(cell) = step.index {
                cells.pyears[cell] += thiscell * weights[i];
                cells.pn[cell] += 1.0;

                if let Some((table, rates, _)) = expected {
                    let mut etime = thiscell;
                    let mut hazard: f64 = 0.0;
                    let mut temp = 0.0;
                    while etime > 0.0 {
                        let expected_step = pystep(table, &data2, etime);
                        let et2 = expected_step.time;
                        let first = rates[expected_step.index.unwrap_or(0)];
                        let lambda = if expected_step.weight < 1.0 {
                            expected_step.weight * first
                                + (1.0 - expected_step.weight) * rates[expected_step.index2]
                        } else {
                            first
                        };
                        if method == PyearsExpect::Pyears {
                            // (1 - exp(-lambda t)) / lambda, whose limit at
                            // lambda = 0 is t (the C code divides by zero).
                            let survival_loss = if lambda == 0.0 {
                                et2
                            } else {
                                -(-lambda * et2).exp_m1() / lambda
                            };
                            temp += (-hazard).exp() * survival_loss;
                        }
                        hazard += lambda * et2;
                        advance(&mut data2, table.factors, et2);
                        etime -= et2;
                    }
                    cells.pexpect[cell] += match method {
                        PyearsExpect::Event => hazard * weights[i],
                        PyearsExpect::Pyears => (-cumhaz).exp() * temp * weights[i],
                    };
                    cumhaz += hazard;
                }
            } else {
                cells.offtable += thiscell * weights[i];
                if let Some((table, _, _)) = expected {
                    advance(&mut data2, table.factors, thiscell);
                }
            }
            advance(&mut data, &categories.factors, thiscell);
            timeleft -= thiscell;
        }
        if let (Some(cell), Some(event)) = (index, &followup.event) {
            cells.pcount[cell] += event[i] * weights[i];
        }
    }
    cells
}

/// The data side of R's `pyears` (`R/pyears.R`) once the model frame has
/// been evaluated: tabulate person-years, observations, events and, with a
/// rate table, expected events over the category table.
pub fn pyears(
    followup: &PyearsFollowup,
    weights: Option<&[f64]>,
    categories: &PyearsCategories,
    ratetable: Option<PyearsRatetable<'_>>,
    expect: PyearsExpect,
    scale: f64,
) -> SurvivalResult<PyearsResult> {
    let n = followup.validate()?;
    if n == 0 {
        return Err(SurvivalError::invalid_input("Data set has 0 observations"));
    }
    if scale.is_nan() || scale <= 0.0 || !scale.is_finite() {
        return Err(SurvivalError::invalid_input("scale must be a value > 0"));
    }
    let weights = match weights {
        Some(values) => {
            validate_length(n, values.len(), "weights")?;
            validate_finite(values, "weights")?;
            values.to_vec()
        }
        None => vec![1.0; n],
    };
    let n_cells = categories.validate(n)?;

    let positions = match &ratetable {
        Some(rt) => {
            if rt.positions.nrows() != n {
                return Err(SurvivalError::invalid_input(
                    "ratetable positions must have one row per observation",
                ));
            }
            rt.table.validate_positions(&rt.positions)?;
            let mut positions = rt.positions.clone();
            align_us_year_axis(rt.table, &mut positions)?;
            Some(positions)
        }
        None => None,
    };
    let cells = match (&ratetable, &positions) {
        (Some(rt), Some(positions)) => {
            let factors = rt.table.factor_flags();
            let cuts = rt.table.cut_slices();
            let table = PystepTable {
                factors: &factors,
                dims: &rt.table.dims,
                cuts: &cuts,
                edge: true,
            };
            pyears1(
                followup,
                &weights,
                categories,
                Some((&table, &rt.table.rates, positions)),
                expect,
                n_cells,
            )
        }
        _ => pyears1(followup, &weights, categories, None, expect, n_cells),
    };

    let expected = ratetable.map(|_| {
        let divisor = if expect == PyearsExpect::Pyears {
            scale
        } else {
            1.0
        };
        cells.pexpect.iter().map(|e| e / divisor).collect()
    });
    Ok(PyearsResult {
        pyears: cells.pyears.iter().map(|p| p / scale).collect(),
        n: cells.pn,
        event: followup.event.as_ref().map(|_| cells.pcount),
        expected,
        offtable: cells.offtable / scale,
        dims: categories.dims.clone(),
        observations: n,
    })
}

/// Python entry point of [`pyears`]: `categories_data` and
/// `ratetable_positions` are row-major nested lists (one row per
/// observation), the latter being `match_ratetable(...).r`.
#[pyfunction(name = "pyears")]
#[pyo3(signature = (
    stop,
    start=None,
    event=None,
    weights=None,
    factors=Vec::new(),
    dims=Vec::new(),
    cuts=Vec::new(),
    categories_data=Vec::new(),
    ratetable=None,
    ratetable_positions=None,
    expect="event",
    scale=365.25,
))]
#[allow(clippy::too_many_arguments)]
pub fn pyears_py(
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    event: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    factors: Vec<i32>,
    dims: Vec<usize>,
    cuts: Vec<Vec<f64>>,
    categories_data: Vec<Vec<f64>>,
    ratetable: Option<&RateTable>,
    ratetable_positions: Option<Vec<Vec<f64>>>,
    expect: &str,
    scale: f64,
) -> PyResult<PyearsResult> {
    let n = stop.len();
    let followup = PyearsFollowup { start, stop, event };
    let categories = PyearsCategories {
        data: rows_to_matrix(&categories_data, n, factors.len(), "categories_data")?,
        factors,
        dims,
        cuts,
    };
    let ratetable = match (ratetable, ratetable_positions) {
        (Some(table), Some(positions)) => Some(PyearsRatetable {
            positions: rows_to_matrix(&positions, n, table.ndim(), "ratetable_positions")?,
            table,
        }),
        (None, None) => None,
        _ => {
            return Err(SurvivalError::invalid_input(
                "ratetable and ratetable_positions must be given together",
            )
            .into());
        }
    };
    Ok(pyears(
        &followup,
        weights.as_deref(),
        &categories,
        ratetable,
        PyearsExpect::parse(expect)?,
        scale,
    )?)
}

/// Assemble a row-major nested list into an `n x ncols` matrix.
pub(crate) fn rows_to_matrix(
    rows: &[Vec<f64>],
    n: usize,
    ncols: usize,
    name: &str,
) -> SurvivalResult<Array2<f64>> {
    if rows.len() != n || rows.iter().any(|row| row.len() != ncols) {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must be {n} x {ncols}"
        )));
    }
    let mut matrix = Array2::<f64>::zeros((n, ncols));
    for (i, row) in rows.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            matrix[[i, j]] = value;
        }
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::ratetable::DimType;
    use crate::population::ratetable_data::survexp_us_table;

    fn tcut_categories(cuts: Vec<f64>, values: Vec<f64>) -> PyearsCategories {
        let n = values.len();
        PyearsCategories {
            factors: vec![0],
            dims: vec![cuts.len() - 1],
            cuts: vec![cuts],
            data: Array2::from_shape_vec((n, 1), values).unwrap(),
        }
    }

    fn right(stop: Vec<f64>, event: Vec<f64>) -> PyearsFollowup {
        PyearsFollowup {
            start: None,
            stop,
            event: Some(event),
        }
    }

    #[test]
    fn tcut_person_time_and_events_match_reference_cells() {
        let out = pyears(
            &right(vec![25.0, 8.0], vec![1.0, 0.0]),
            None,
            &tcut_categories(vec![0.0, 10.0, 20.0, 30.0], vec![0.0, 5.0]),
            None,
            PyearsExpect::Event,
            1.0,
        )
        .unwrap();
        assert_eq!(out.pyears, vec![15.0, 13.0, 5.0]);
        assert_eq!(out.n, vec![2.0, 2.0, 1.0]);
        assert_eq!(out.event, Some(vec![0.0, 0.0, 1.0]));
        assert_eq!(out.expected, None);
        assert_eq!(out.offtable, 0.0);
        assert_eq!(out.observations, 2);
    }

    #[test]
    fn off_table_time_is_tracked_and_scaled() {
        let out = pyears(
            &right(vec![10.0, 10.0, 10.0], vec![1.0, 1.0, 1.0]),
            None,
            &tcut_categories(vec![0.0, 10.0, 20.0, 30.0], vec![-5.0, 25.0, 35.0]),
            None,
            PyearsExpect::Event,
            2.0,
        )
        .unwrap();
        assert_eq!(out.pyears, vec![2.5, 0.0, 2.5]);
        assert_eq!(out.n, vec![1.0, 0.0, 1.0]);
        assert_eq!(out.event, Some(vec![1.0, 0.0, 0.0]));
        assert_eq!(out.offtable, 10.0);
    }

    #[test]
    fn tcut_and_factor_dimensions_use_column_major_output_order() {
        let categories = PyearsCategories {
            factors: vec![0, 1],
            dims: vec![4, 2],
            cuts: vec![vec![0.0, 10.0, 20.0, 30.0, 40.0], vec![]],
            data: ndarray::arr2(&[[0.0, 1.0], [5.0, 2.0], [15.0, 1.0]]),
        };
        let out = pyears(
            &right(vec![25.0, 8.0, 12.0], vec![1.0, 0.0, 1.0]),
            None,
            &categories,
            None,
            PyearsExpect::Event,
            1.0,
        )
        .unwrap();
        assert_eq!(out.pyears, vec![10.0, 15.0, 12.0, 0.0, 5.0, 3.0, 0.0, 0.0]);
        assert_eq!(out.n, vec![1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            out.event,
            Some(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(out.dims, vec![4, 2]);
    }

    #[test]
    fn no_categories_tabulate_into_a_single_cell_with_weights_and_start() {
        let categories = PyearsCategories {
            factors: vec![],
            dims: vec![],
            cuts: vec![],
            data: Array2::zeros((2, 0)),
        };
        let followup = PyearsFollowup {
            start: Some(vec![2.0, 0.0]),
            stop: vec![12.0, 3.0],
            event: Some(vec![1.0, 1.0]),
        };
        let out = pyears(
            &followup,
            Some(&[2.0, 1.0]),
            &categories,
            None,
            PyearsExpect::Event,
            1.0,
        )
        .unwrap();
        assert_eq!(out.pyears, vec![23.0]);
        assert_eq!(out.n, vec![2.0]);
        assert_eq!(out.event, Some(vec![3.0]));
    }

    #[test]
    fn zero_follow_up_events_are_still_counted_in_their_cell() {
        let out = pyears(
            &right(vec![0.0, 5.0], vec![1.0, 0.0]),
            None,
            &tcut_categories(vec![0.0, 10.0], vec![3.0, 3.0]),
            None,
            PyearsExpect::Event,
            1.0,
        )
        .unwrap();
        assert_eq!(out.pyears, vec![5.0]);
        assert_eq!(out.n, vec![1.0]);
        assert_eq!(out.event, Some(vec![1.0]));
    }

    #[test]
    fn expected_events_integrate_the_rate_table_hazard() {
        // One-dimensional age table: rate 0.1/day below 10 days, 0.3 after.
        let table = RateTable::try_new(
            vec![2],
            vec!["age".into()],
            vec![vec!["0".into(), "10".into()]],
            vec![Some(vec![0.0, 10.0])],
            vec![DimType::Continuous],
            vec![0.1, 0.3],
        )
        .unwrap();
        let categories = tcut_categories(vec![0.0, 100.0], vec![0.0]);
        let ratetable = PyearsRatetable {
            table: &table,
            positions: ndarray::arr2(&[[5.0]]),
        };
        let events = pyears(
            &right(vec![10.0], vec![1.0]),
            None,
            &categories,
            Some(ratetable),
            PyearsExpect::Event,
            1.0,
        )
        .unwrap();
        // 5 days at 0.1 then 5 days at 0.3.
        assert!((events.expected.as_ref().unwrap()[0] - 2.0).abs() < 1e-12);

        let ratetable = PyearsRatetable {
            table: &table,
            positions: ndarray::arr2(&[[5.0]]),
        };
        let person_years = pyears(
            &right(vec![10.0], vec![1.0]),
            None,
            &categories,
            Some(ratetable),
            PyearsExpect::Pyears,
            1.0,
        )
        .unwrap();
        let expected =
            (1.0 - (-0.5f64).exp()) / 0.1 + (-0.5f64).exp() * (1.0 - (-1.5f64).exp()) / 0.3;
        assert!((person_years.expected.as_ref().unwrap()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn rate_table_positions_are_checked_before_indexing_the_rates() {
        // survexp.us is age x sex x year; sex has two levels.
        let table = survexp_us_table();
        let categories = tcut_categories(vec![0.0, 1000.0], vec![0.0]);
        let followup = right(vec![100.0], vec![0.0]);
        let attempt = |row: [f64; 3]| {
            pyears(
                &followup,
                None,
                &categories,
                Some(PyearsRatetable {
                    table,
                    positions: ndarray::arr2(&[row]),
                }),
                PyearsExpect::Event,
                1.0,
            )
        };
        let entry = 109.0 * 365.25;
        assert!(attempt([entry, 1.0, 18262.0]).is_ok());
        for sex in [5.0, 0.0, 1.5, f64::NAN] {
            let message = attempt([entry, sex, 18262.0]).unwrap_err().to_string();
            assert!(message.contains("The variable sex"), "{message}");
        }
        assert!(
            attempt([f64::NAN, 1.0, 18262.0])
                .unwrap_err()
                .to_string()
                .contains("age contains missing values")
        );
        assert!(
            attempt([entry, 1.0, f64::INFINITY])
                .unwrap_err()
                .to_string()
                .contains("year must be finite")
        );
        let wrong_width = pyears(
            &followup,
            None,
            &categories,
            Some(PyearsRatetable {
                table,
                positions: ndarray::arr2(&[[entry, 1.0]]),
            }),
            PyearsExpect::Event,
            1.0,
        );
        assert!(wrong_width.is_err());
    }

    #[test]
    fn rejects_malformed_inputs() {
        let categories = tcut_categories(vec![0.0, 10.0], vec![3.0]);
        let err = |followup: PyearsFollowup, weights: Option<&[f64]>, scale: f64| {
            pyears(
                &followup,
                weights,
                &categories,
                None,
                PyearsExpect::Event,
                scale,
            )
            .unwrap_err()
            .to_string()
        };
        assert!(err(right(vec![], vec![]), None, 1.0).contains("0 observations"));
        assert!(err(right(vec![-1.0], vec![0.0]), None, 1.0).contains("negative"));
        assert!(err(right(vec![1.0], vec![0.0]), Some(&[1.0, 2.0]), 1.0).contains("weights"));
        assert!(err(right(vec![1.0], vec![0.0]), None, 0.0).contains("scale"));
        let bad_codes = PyearsCategories {
            factors: vec![1],
            dims: vec![2],
            cuts: vec![vec![]],
            data: ndarray::arr2(&[[3.0]]),
        };
        assert!(
            pyears(
                &right(vec![1.0], vec![0.0]),
                None,
                &bad_codes,
                None,
                PyearsExpect::Event,
                1.0
            )
            .is_err()
        );
        assert!(PyearsExpect::parse("events").is_err());
    }
}
