use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_cutpoints(cutpoints: &[f64], field: &str) -> PyResult<()> {
    for (index, &value) in cutpoints.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{field} contains non-finite value at index {index}"
            )));
        }
    }
    for (index, pair) in cutpoints.windows(2).enumerate() {
        if pair[1] <= pair[0] {
            return Err(value_error(format!(
                "{field} must be strictly increasing; index {} is not greater than index {}",
                index + 1,
                index
            )));
        }
    }
    Ok(())
}

fn validate_rates(rates: &[f64], field: &str) -> PyResult<()> {
    for (index, &rate) in rates.iter().enumerate() {
        if !rate.is_finite() {
            return Err(value_error(format!(
                "{field} contains non-finite value at index {index}"
            )));
        }
        if rate < 0.0 {
            return Err(value_error(format!(
                "{field} contains negative value {rate} at index {index}"
            )));
        }
    }
    Ok(())
}

/// Type of dimension in a rate table
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub enum DimType {
    /// Categorical factor (e.g., sex)
    Factor,
    /// Age in days/years
    Age,
    /// Calendar year
    Year,
    /// General continuous variable
    Continuous,
}

/// A dimension in the rate table
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RateDimension {
    /// Name of the dimension
    #[pyo3(get)]
    pub name: String,
    /// Type of dimension
    #[pyo3(get)]
    pub dim_type: DimType,
    /// Factor levels (for Factor type)
    #[pyo3(get)]
    pub levels: Option<Vec<String>>,
    /// Cutpoints for continuous dimensions
    #[pyo3(get)]
    pub cutpoints: Vec<f64>,
}

#[pymethods]
impl RateDimension {
    #[new]
    #[pyo3(signature = (name, dim_type, cutpoints, levels=None))]
    pub fn new(
        name: String,
        dim_type: DimType,
        cutpoints: Vec<f64>,
        levels: Option<Vec<String>>,
    ) -> Self {
        RateDimension {
            name,
            dim_type,
            levels,
            cutpoints,
        }
    }
}

/// Population mortality rate table.
///
/// A multi-dimensional table of mortality rates indexed by age, calendar year,
/// sex, and potentially other factors. Used with survexp() to compute expected
/// survival based on population mortality.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RateTable {
    /// Dimensions of the table
    dimensions: Vec<RateDimension>,
    /// Flattened array of mortality rates (daily hazard rates)
    rates: Vec<f64>,
    /// Shape of the multi-dimensional array
    shape: Vec<usize>,
    /// Summary/description of the table
    #[pyo3(get)]
    pub summary: String,
}

#[pymethods]
impl RateTable {
    /// Create a new rate table
    ///
    /// # Arguments
    /// * `dimensions` - Vector of dimension definitions
    /// * `rates` - Flattened array of mortality rates
    /// * `summary` - Description of the table
    #[new]
    #[pyo3(signature = (dimensions, rates, summary=None))]
    pub fn new(
        dimensions: Vec<RateDimension>,
        rates: Vec<f64>,
        summary: Option<String>,
    ) -> PyResult<Self> {
        if dimensions.is_empty() {
            return Err(value_error("dimensions cannot be empty"));
        }
        for dim in &dimensions {
            if dim.name.trim().is_empty() {
                return Err(value_error("dimension names cannot be empty"));
            }
            if dim.dim_type == DimType::Factor {
                if let Some(levels) = &dim.levels
                    && levels.is_empty()
                {
                    return Err(value_error(format!(
                        "factor dimension '{}' must have at least one level",
                        dim.name
                    )));
                }
            } else {
                validate_cutpoints(&dim.cutpoints, &format!("{} cutpoints", dim.name))?;
            }
        }
        validate_rates(&rates, "rates")?;

        let shape: Vec<usize> = dimensions
            .iter()
            .map(|d| {
                if d.dim_type == DimType::Factor {
                    d.levels.as_ref().map_or(1, |l| l.len())
                } else {
                    d.cutpoints.len().saturating_sub(1).max(1)
                }
            })
            .collect();

        let expected_size: usize = shape.iter().product();
        if rates.len() != expected_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "rates length ({}) doesn't match dimensions (expected {})",
                rates.len(),
                expected_size
            )));
        }

        Ok(RateTable {
            dimensions,
            rates,
            shape,
            summary: summary.unwrap_or_else(|| "Custom rate table".to_string()),
        })
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    /// Get dimension names
    pub fn dim_names(&self) -> Vec<String> {
        self.dimensions.iter().map(|d| d.name.clone()).collect()
    }

    /// Lookup mortality rate for given coordinates
    ///
    /// # Arguments
    /// * `coords` - HashMap mapping dimension names to values
    ///
    /// # Returns
    /// Daily hazard rate at the specified coordinates
    pub fn lookup(&self, coords: HashMap<String, f64>) -> PyResult<f64> {
        let indices = self.coords_to_indices(&coords)?;
        let flat_idx = self.indices_to_flat(&indices);
        Ok(self.rates[flat_idx])
    }

    /// Lookup with interpolation for continuous dimensions
    pub fn lookup_interpolate(&self, coords: HashMap<String, f64>) -> PyResult<f64> {
        self.lookup(coords)
    }

    /// Get cumulative hazard over a time interval
    ///
    /// # Arguments
    /// * `age_start` - Starting age (in days)
    /// * `age_end` - Ending age (in days)
    /// * `year_start` - Starting calendar year
    /// * `sex` - Sex (0=male, 1=female typically)
    ///
    /// # Returns
    /// Cumulative hazard over the interval
    #[pyo3(signature = (age_start, age_end, year_start, sex=None))]
    pub fn cumulative_hazard(
        &self,
        age_start: f64,
        age_end: f64,
        year_start: f64,
        sex: Option<i32>,
    ) -> PyResult<f64> {
        if !age_start.is_finite() || !age_end.is_finite() || !year_start.is_finite() {
            return Err(value_error(
                "age_start, age_end, and year_start must be finite",
            ));
        }
        if age_start < 0.0 || age_end < 0.0 {
            return Err(value_error("age_start and age_end must be non-negative"));
        }
        if matches!(sex, Some(value) if value < 0) {
            return Err(value_error("sex must be non-negative"));
        }
        if age_end <= age_start {
            return Ok(0.0);
        }

        let mut cumhaz = 0.0;
        let mut current_age = age_start;
        let mut current_year = year_start;

        let mut coords = HashMap::with_capacity(self.dimensions.len());
        let mut age_key = None;
        let mut year_key = None;

        for dim in &self.dimensions {
            match dim.dim_type {
                DimType::Age => {
                    age_key = Some(dim.name.clone());
                    coords.insert(dim.name.clone(), current_age);
                }
                DimType::Year => {
                    year_key = Some(dim.name.clone());
                    coords.insert(dim.name.clone(), current_year);
                }
                DimType::Factor if dim.name.to_lowercase().contains("sex") => {
                    coords.insert(dim.name.clone(), sex.unwrap_or(0) as f64);
                }
                _ => {}
            }
        }

        let step = 1.0;
        while current_age < age_end {
            if let Some(ref key) = age_key {
                coords.insert(key.clone(), current_age);
            }
            if let Some(ref key) = year_key {
                coords.insert(key.clone(), current_year);
            }

            if let Ok(rate) = self.lookup(coords.clone()) {
                let actual_step = (age_end - current_age).min(step);
                cumhaz += rate * actual_step;
            }

            current_age += step;
            current_year += step / 365.25;
        }

        Ok(cumhaz)
    }

    /// Get expected survival probability over a time interval
    #[pyo3(signature = (age_start, age_end, year_start, sex=None))]
    pub fn expected_survival(
        &self,
        age_start: f64,
        age_end: f64,
        year_start: f64,
        sex: Option<i32>,
    ) -> PyResult<f64> {
        let cumhaz = self.cumulative_hazard(age_start, age_end, year_start, sex)?;
        Ok((-cumhaz).exp())
    }
}

impl RateTable {
    /// Convert coordinate values to array indices
    fn coords_to_indices(&self, coords: &HashMap<String, f64>) -> PyResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(self.dimensions.len());

        for dim in &self.dimensions {
            let value = coords.get(&dim.name).copied().unwrap_or(0.0);
            if !value.is_finite() {
                return Err(value_error(format!(
                    "{} coordinate must be finite",
                    dim.name
                )));
            }

            let idx = match dim.dim_type {
                DimType::Factor => {
                    if value < 0.0 {
                        return Err(value_error(format!(
                            "{} coordinate must be non-negative",
                            dim.name
                        )));
                    }
                    let max_idx = dim.levels.as_ref().map_or(0, |l| l.len().saturating_sub(1));
                    (value as usize).min(max_idx)
                }
                DimType::Age | DimType::Year | DimType::Continuous => {
                    find_interval(&dim.cutpoints, value)
                }
            };
            indices.push(idx);
        }

        Ok(indices)
    }

    /// Convert multi-dimensional indices to flat index
    fn indices_to_flat(&self, indices: &[usize]) -> usize {
        let mut flat_idx = 0;
        let mut multiplier = 1;

        for (i, &idx) in indices.iter().rev().enumerate() {
            let dim_idx = self.shape.len() - 1 - i;
            flat_idx += idx.min(self.shape[dim_idx].saturating_sub(1)) * multiplier;
            multiplier *= self.shape[dim_idx];
        }

        flat_idx.min(self.rates.len().saturating_sub(1))
    }
}

/// Find which interval a value belongs to using binary search
fn find_interval(cutpoints: &[f64], value: f64) -> usize {
    if cutpoints.len() < 2 {
        return 0;
    }

    match cutpoints.binary_search_by(|probe| probe.total_cmp(&value)) {
        Ok(i) => {
            if i >= cutpoints.len() - 1 {
                cutpoints.len() - 2
            } else {
                i
            }
        }
        Err(i) => {
            if i == 0 {
                0
            } else if i >= cutpoints.len() {
                cutpoints.len() - 2
            } else {
                i - 1
            }
        }
    }
}

/// Create a simple rate table for testing/demonstration
#[pyfunction]
pub fn create_simple_ratetable(
    age_breaks: Vec<f64>,
    year_breaks: Vec<f64>,
    rates_male: Vec<f64>,
    rates_female: Vec<f64>,
) -> PyResult<RateTable> {
    if age_breaks.len() < 2 {
        return Err(value_error(
            "age_breaks must contain at least two cutpoints",
        ));
    }
    if year_breaks.len() < 2 {
        return Err(value_error(
            "year_breaks must contain at least two cutpoints",
        ));
    }
    validate_cutpoints(&age_breaks, "age_breaks")?;
    validate_cutpoints(&year_breaks, "year_breaks")?;
    validate_rates(&rates_male, "rates_male")?;
    validate_rates(&rates_female, "rates_female")?;

    let n_age = age_breaks.len().saturating_sub(1).max(1);
    let n_year = year_breaks.len().saturating_sub(1).max(1);

    if rates_male.len() != n_age * n_year || rates_female.len() != n_age * n_year {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "rates arrays must match age x year dimensions",
        ));
    }

    let mut rates = Vec::with_capacity(rates_male.len() + rates_female.len());
    for i in 0..(n_age * n_year) {
        rates.push(rates_male[i]);
        rates.push(rates_female[i]);
    }

    let dimensions = vec![
        RateDimension::new("age".to_string(), DimType::Age, age_breaks, None),
        RateDimension::new("year".to_string(), DimType::Year, year_breaks, None),
        RateDimension::new(
            "sex".to_string(),
            DimType::Factor,
            vec![],
            Some(vec!["male".to_string(), "female".to_string()]),
        ),
    ];

    RateTable::new(dimensions, rates, Some("Simple rate table".to_string()))
}

#[pyfunction]
pub fn is_ratetable(ndim: usize, has_rates: bool, has_dims: bool) -> bool {
    ndim > 0 && has_rates && has_dims
}

#[derive(Debug, Clone)]
#[pyclass(str, from_py_object)]
pub struct RatetableDateResult {
    #[pyo3(get)]
    pub days: f64,
    #[pyo3(get)]
    pub years: f64,
    #[pyo3(get)]
    pub origin_year: i32,
}

impl fmt::Display for RatetableDateResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RatetableDateResult(days={:.1}, years={:.4}, origin={})",
            self.days, self.years, self.origin_year
        )
    }
}

#[pyfunction]
#[pyo3(signature = (year, month=1, day=1, origin_year=1960))]
pub fn ratetable_date(
    year: i32,
    month: u32,
    day: u32,
    origin_year: i32,
) -> PyResult<RatetableDateResult> {
    if !(1..=12).contains(&month) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "month must be between 1 and 12",
        ));
    }
    if !(1..=31).contains(&day) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "day must be between 1 and 31",
        ));
    }

    let days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    fn is_leap_year(y: i32) -> bool {
        (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
    }

    fn days_in_year(y: i32) -> f64 {
        if is_leap_year(y) { 366.0 } else { 365.0 }
    }

    let max_day = if month == 2 && is_leap_year(year) {
        29
    } else {
        days_per_month[(month - 1) as usize]
    };
    if day > max_day {
        return Err(value_error("day is invalid for the given month and year"));
    }

    let mut total_days: f64 = 0.0;

    for y in origin_year..year {
        total_days += days_in_year(y);
    }

    for m in 1..month {
        let mut d = days_per_month[(m - 1) as usize] as f64;
        if m == 2 && is_leap_year(year) {
            d += 1.0;
        }
        total_days += d;
    }

    total_days += (day - 1) as f64;

    let years = total_days / 365.25;

    Ok(RatetableDateResult {
        days: total_days,
        years,
        origin_year,
    })
}

#[pyfunction]
pub fn days_to_date(days: f64, origin_year: i32) -> PyResult<(i32, u32, u32)> {
    if !days.is_finite() || days < 0.0 {
        return Err(value_error("days must be a finite non-negative value"));
    }

    fn is_leap_year(y: i32) -> bool {
        (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
    }

    fn days_in_year(y: i32) -> i32 {
        if is_leap_year(y) { 366 } else { 365 }
    }

    let days_per_month_normal = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let days_per_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    let mut remaining_days = days as i32;
    let mut year = origin_year;

    while remaining_days >= days_in_year(year) {
        remaining_days -= days_in_year(year);
        year += 1;
    }

    let days_per_month = if is_leap_year(year) {
        &days_per_month_leap
    } else {
        &days_per_month_normal
    };

    let mut month = 1u32;
    for &d in days_per_month.iter() {
        if remaining_days < d {
            break;
        }
        remaining_days -= d;
        month += 1;
    }

    let day = (remaining_days + 1) as u32;

    Ok((year, month, day))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ratetable_basic() {
        let age_breaks = vec![0.0, 365.0, 3650.0, 36500.0];
        let year_breaks = vec![1990.0, 2000.0, 2010.0];

        let rates_male = vec![0.001, 0.0008, 0.0005, 0.0004, 0.0003, 0.0002];
        let rates_female = vec![0.0008, 0.0006, 0.0004, 0.0003, 0.0002, 0.00015];

        let rt = create_simple_ratetable(age_breaks, year_breaks, rates_male, rates_female);
        assert!(rt.is_ok());

        let rt = rt.unwrap();
        assert_eq!(rt.ndim(), 3);
    }

    #[test]
    fn test_ratetable_lookup() {
        let dimensions = vec![RateDimension::new(
            "age".to_string(),
            DimType::Age,
            vec![0.0, 10.0, 20.0],
            None,
        )];
        let rates = vec![0.01, 0.02];

        let rt = RateTable::new(dimensions, rates, None).unwrap();

        let mut coords = HashMap::new();
        coords.insert("age".to_string(), 5.0);
        assert_eq!(rt.lookup(coords).unwrap(), 0.01);

        let mut coords = HashMap::new();
        coords.insert("age".to_string(), 15.0);
        assert_eq!(rt.lookup(coords).unwrap(), 0.02);
    }

    #[test]
    fn ratetable_validates_public_inputs() {
        assert!(
            RateTable::new(vec![], vec![], None)
                .expect_err("empty dimensions should fail")
                .to_string()
                .contains("dimensions cannot be empty")
        );
        assert!(
            create_simple_ratetable(vec![0.0], vec![1990.0, 2000.0], vec![0.1], vec![0.1])
                .expect_err("short age breaks should fail")
                .to_string()
                .contains("age_breaks")
        );
        assert!(
            create_simple_ratetable(
                vec![0.0, 10.0, 5.0],
                vec![1990.0, 2000.0],
                vec![0.1, 0.2],
                vec![0.1, 0.2],
            )
            .expect_err("unsorted age breaks should fail")
            .to_string()
            .contains("age_breaks must be strictly increasing")
        );
        assert!(
            create_simple_ratetable(
                vec![0.0, 10.0],
                vec![1990.0, 2000.0],
                vec![f64::NAN],
                vec![0.1],
            )
            .expect_err("non-finite rate should fail")
            .to_string()
            .contains("rates_male contains non-finite")
        );

        let rt = create_simple_ratetable(
            vec![0.0, 365.0],
            vec![1990.0, 2000.0],
            vec![0.001],
            vec![0.0008],
        )
        .unwrap();
        let mut coords = HashMap::new();
        coords.insert("age".to_string(), f64::NAN);
        assert!(
            rt.lookup(coords)
                .expect_err("non-finite coordinate should fail")
                .to_string()
                .contains("age coordinate must be finite")
        );
        assert!(
            rt.cumulative_hazard(0.0, f64::INFINITY, 2000.0, Some(0))
                .expect_err("non-finite age end should fail")
                .to_string()
                .contains("must be finite")
        );
        assert!(
            days_to_date(-1.0, 1960)
                .expect_err("negative days should fail")
                .to_string()
                .contains("days must be a finite non-negative value")
        );
        assert!(
            ratetable_date(2001, 2, 29, 1960)
                .expect_err("invalid calendar date should fail")
                .to_string()
                .contains("day is invalid")
        );
    }
}
