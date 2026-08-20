use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;

const DAYS_PER_YEAR: f64 = 365.25;
const DAYS_BEFORE_MONTH: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_year(year: i32) -> i32 {
    if is_leap_year(year) { 366 } else { 365 }
}

fn day_number(year: i32, month: u32, day: u32) -> i64 {
    let previous_year = i64::from(year) - 1;
    let days_before_year = 365 * previous_year + previous_year.div_euclid(4)
        - previous_year.div_euclid(100)
        + previous_year.div_euclid(400);
    let leap_day = i64::from(month > 2 && is_leap_year(year));
    days_before_year + DAYS_BEFORE_MONTH[(month - 1) as usize] + i64::from(day - 1) + leap_day
}

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

#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub enum DimType {
    Factor,
    Age,
    Year,
    Continuous,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RateDimension {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub dim_type: DimType,
    #[pyo3(get)]
    pub levels: Option<Vec<String>>,
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

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RateTable {
    dimensions: Vec<RateDimension>,
    rates: Vec<f64>,
    shape: Vec<usize>,
    #[pyo3(get)]
    pub summary: String,
}

#[pymethods]
impl RateTable {
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

    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    pub fn dim_names(&self) -> Vec<String> {
        self.dimensions.iter().map(|d| d.name.clone()).collect()
    }

    pub fn dimension_specs(&self) -> Vec<RateDimension> {
        self.dimensions.clone()
    }

    pub fn lookup(&self, coords: HashMap<String, f64>) -> PyResult<f64> {
        let indices = self.coords_to_indices(&coords)?;
        let flat_idx = self.indices_to_flat(&indices);
        Ok(self.rates[flat_idx])
    }

    pub fn lookup_many(&self, coords: HashMap<String, Vec<f64>>) -> PyResult<Vec<f64>> {
        let Some(row_count) = coords.values().next().map(Vec::len) else {
            return Err(value_error(
                "coords must contain at least one coordinate column",
            ));
        };
        for (name, values) in &coords {
            if values.len() != row_count {
                return Err(value_error(format!(
                    "coordinate column '{name}' has length {}; expected {row_count}",
                    values.len()
                )));
            }
        }

        let columns: Vec<Option<&Vec<f64>>> = self
            .dimensions
            .iter()
            .map(|dimension| coords.get(&dimension.name))
            .collect();
        let mut result = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let mut flat_idx = 0usize;
            for (dimension_idx, (dimension, column)) in
                self.dimensions.iter().zip(&columns).enumerate()
            {
                let value = column.map_or(0.0, |values| values[row]);
                let index = coordinate_index(dimension, value)?
                    .min(self.shape[dimension_idx].saturating_sub(1));
                flat_idx = flat_idx * self.shape[dimension_idx] + index;
            }
            result.push(self.rates[flat_idx.min(self.rates.len().saturating_sub(1))]);
        }
        Ok(result)
    }

    pub fn lookup_interpolate(&self, coords: HashMap<String, f64>) -> PyResult<f64> {
        self.lookup(coords)
    }

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

        let mut indices: Vec<usize> = self
            .dimensions
            .iter()
            .map(|dimension| match dimension.dim_type {
                DimType::Factor if is_sex_dimension(&dimension.name) => {
                    factor_index(dimension, sex.unwrap_or(0) as usize)
                }
                DimType::Factor => factor_index(dimension, 0),
                DimType::Continuous => find_interval(&dimension.cutpoints, 0.0),
                DimType::Age | DimType::Year => 0,
            })
            .collect();
        let mut cumhaz = 0.0;
        let mut current_age = age_start;
        let mut current_year = year_start;
        let mut remaining = age_end - age_start;

        while remaining > 0.0 {
            let mut interval = remaining;
            for (index, dimension) in self.dimensions.iter().enumerate() {
                match dimension.dim_type {
                    DimType::Age => {
                        indices[index] = find_interval(&dimension.cutpoints, current_age);
                        if let Some(boundary) = next_cutpoint(&dimension.cutpoints, current_age) {
                            interval = interval.min(boundary - current_age);
                        }
                    }
                    DimType::Year => {
                        indices[index] = find_interval(&dimension.cutpoints, current_year);
                        if let Some(boundary) = next_cutpoint(&dimension.cutpoints, current_year) {
                            interval = interval.min((boundary - current_year) * DAYS_PER_YEAR);
                        }
                    }
                    DimType::Factor | DimType::Continuous => {}
                }
            }

            cumhaz += self.rates[self.indices_to_flat(&indices)] * interval;
            remaining -= interval;
            current_age += interval;
            current_year += interval / DAYS_PER_YEAR;
        }

        Ok(cumhaz)
    }

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
    fn coords_to_indices(&self, coords: &HashMap<String, f64>) -> PyResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(self.dimensions.len());

        for dim in &self.dimensions {
            let value = coords.get(&dim.name).copied().unwrap_or(0.0);
            indices.push(coordinate_index(dim, value)?);
        }

        Ok(indices)
    }

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

    pub(crate) fn aligned_coordinate_columns(
        &self,
        coordinates: &HashMap<String, Vec<f64>>,
        row_count: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        for name in coordinates.keys() {
            if !self
                .dimensions
                .iter()
                .any(|dimension| dimension.name == *name)
            {
                return Err(value_error(format!(
                    "coordinate {name:?} is not a rate-table dimension"
                )));
            }
        }

        self.dimensions
            .iter()
            .map(|dimension| {
                let values = coordinates.get(&dimension.name).ok_or_else(|| {
                    value_error(format!(
                        "coordinate {:?} is required by the rate table",
                        dimension.name
                    ))
                })?;
                if values.len() != row_count {
                    return Err(value_error(format!(
                        "coordinate {:?} must have length {row_count}",
                        dimension.name
                    )));
                }
                for (row, &value) in values.iter().enumerate() {
                    coordinate_index(dimension, value)?;
                    if dimension.dim_type == DimType::Age && value < 0.0 {
                        return Err(value_error(format!(
                            "{} coordinate must be non-negative; got {value} at row {row}",
                            dimension.name
                        )));
                    }
                    if dimension.dim_type == DimType::Factor {
                        let level_count = dimension.levels.as_ref().map_or(0, Vec::len);
                        if value.fract() != 0.0 || value as usize >= level_count {
                            return Err(value_error(format!(
                                "{} coordinate must be an integer from 0 to {}; got {value} at row {row}",
                                dimension.name,
                                level_count.saturating_sub(1)
                            )));
                        }
                    }
                }
                Ok(values.clone())
            })
            .collect()
    }

    pub(crate) fn cumulative_hazard_from_values(
        &self,
        base_coordinates: &[f64],
        duration: f64,
    ) -> PyResult<f64> {
        if base_coordinates.len() != self.dimensions.len() {
            return Err(value_error(
                "coordinate count must match rate-table dimensions",
            ));
        }
        if !duration.is_finite() || duration < 0.0 {
            return Err(value_error("duration must be finite and non-negative"));
        }
        if duration == 0.0 {
            return Ok(0.0);
        }

        let mut indices = vec![0usize; self.dimensions.len()];
        let mut cumulative_hazard = 0.0;
        let mut elapsed = 0.0;
        while elapsed < duration {
            let mut interval = duration - elapsed;
            for (dimension_index, dimension) in self.dimensions.iter().enumerate() {
                let value = match dimension.dim_type {
                    DimType::Age => base_coordinates[dimension_index] + elapsed,
                    DimType::Year => base_coordinates[dimension_index] + elapsed / DAYS_PER_YEAR,
                    DimType::Factor | DimType::Continuous => base_coordinates[dimension_index],
                };
                indices[dimension_index] = coordinate_index(dimension, value)?;
                match dimension.dim_type {
                    DimType::Age => {
                        if let Some(boundary) = next_cutpoint(&dimension.cutpoints, value) {
                            interval = interval.min(boundary - value);
                        }
                    }
                    DimType::Year => {
                        if let Some(boundary) = next_cutpoint(&dimension.cutpoints, value) {
                            interval = interval.min((boundary - value) * DAYS_PER_YEAR);
                        }
                    }
                    DimType::Factor | DimType::Continuous => {}
                }
            }
            if interval <= 0.0 {
                return Err(value_error("rate-table integration did not advance"));
            }
            cumulative_hazard += self.rates[self.indices_to_flat(&indices)] * interval;
            elapsed += interval;
        }
        Ok(cumulative_hazard)
    }

    pub(crate) fn cumulative_hazard_interval_from_values(
        &self,
        base_coordinates: &[f64],
        start: f64,
        stop: f64,
    ) -> PyResult<f64> {
        if !start.is_finite() || !stop.is_finite() || start < 0.0 || stop < start {
            return Err(value_error(
                "integration start and stop must be finite with 0 <= start <= stop",
            ));
        }
        let mut advanced = base_coordinates.to_vec();
        for (value, dimension) in advanced.iter_mut().zip(&self.dimensions) {
            match dimension.dim_type {
                DimType::Age => *value += start,
                DimType::Year => *value += start / DAYS_PER_YEAR,
                DimType::Factor | DimType::Continuous => {}
            }
        }
        self.cumulative_hazard_from_values(&advanced, stop - start)
    }
}

fn coordinate_index(dimension: &RateDimension, value: f64) -> PyResult<usize> {
    if !value.is_finite() {
        return Err(value_error(format!(
            "{} coordinate must be finite",
            dimension.name
        )));
    }
    match dimension.dim_type {
        DimType::Factor => {
            if value < 0.0 {
                return Err(value_error(format!(
                    "{} coordinate must be non-negative",
                    dimension.name
                )));
            }
            let max_idx = dimension
                .levels
                .as_ref()
                .map_or(0, |levels| levels.len().saturating_sub(1));
            Ok((value as usize).min(max_idx))
        }
        DimType::Age | DimType::Year | DimType::Continuous => {
            Ok(find_interval(&dimension.cutpoints, value))
        }
    }
}

fn is_sex_dimension(name: &str) -> bool {
    name.as_bytes()
        .windows(3)
        .any(|window| window.eq_ignore_ascii_case(b"sex"))
}

fn factor_index(dimension: &RateDimension, value: usize) -> usize {
    value.min(
        dimension
            .levels
            .as_ref()
            .map_or(0, |levels| levels.len().saturating_sub(1)),
    )
}

fn next_cutpoint(cutpoints: &[f64], value: f64) -> Option<f64> {
    let index = match cutpoints.binary_search_by(|probe| probe.total_cmp(&value)) {
        Ok(index) => index + 1,
        Err(index) => index,
    };
    cutpoints.get(index).copied()
}

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

    let max_day = if month == 2 && is_leap_year(year) {
        29
    } else {
        days_per_month[(month - 1) as usize]
    };
    if day > max_day {
        return Err(value_error("day is invalid for the given month and year"));
    }

    let total_days = (day_number(year, month, day) - day_number(origin_year, 1, 1)) as f64;

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
    fn ratetable_batch_lookup_matches_scalar_coordinates() {
        let dimensions = vec![
            RateDimension::new("age".to_string(), DimType::Age, vec![0.0, 10.0, 20.0], None),
            RateDimension::new(
                "sex".to_string(),
                DimType::Factor,
                vec![],
                Some(vec!["male".to_string(), "female".to_string()]),
            ),
        ];
        let table = RateTable::new(dimensions, vec![0.01, 0.02, 0.03, 0.04], None).unwrap();
        let mut columns = HashMap::new();
        columns.insert("age".to_string(), vec![5.0, 15.0, 15.0]);
        columns.insert("sex".to_string(), vec![0.0, 0.0, 1.0]);

        assert_eq!(table.lookup_many(columns).unwrap(), vec![0.01, 0.03, 0.04]);

        let mut age_only = HashMap::new();
        age_only.insert("age".to_string(), vec![5.0, 15.0]);
        assert_eq!(table.lookup_many(age_only).unwrap(), vec![0.01, 0.03]);
    }

    #[test]
    fn ratetable_batch_lookup_validates_column_shape_and_values() {
        let dimensions = vec![RateDimension::new(
            "age".to_string(),
            DimType::Age,
            vec![0.0, 10.0, 20.0],
            None,
        )];
        let table = RateTable::new(dimensions, vec![0.01, 0.02], None).unwrap();

        assert!(
            table
                .lookup_many(HashMap::new())
                .expect_err("empty coordinate columns should fail")
                .to_string()
                .contains("at least one coordinate column")
        );
        let mut ragged = HashMap::new();
        ragged.insert("age".to_string(), vec![5.0, 15.0]);
        ragged.insert("unused".to_string(), vec![1.0]);
        assert!(
            table
                .lookup_many(ragged)
                .expect_err("ragged coordinate columns should fail")
                .to_string()
                .contains("coordinate column")
        );
        let mut non_finite = HashMap::new();
        non_finite.insert("age".to_string(), vec![f64::NAN]);
        assert!(
            table
                .lookup_many(non_finite)
                .expect_err("non-finite batch coordinate should fail")
                .to_string()
                .contains("age coordinate must be finite")
        );
    }

    #[test]
    fn cumulative_hazard_integrates_exactly_across_age_cutpoints() {
        let dimensions = vec![RateDimension::new(
            "age".to_string(),
            DimType::Age,
            vec![0.0, 10.0, 20.0],
            None,
        )];
        let table = RateTable::new(dimensions, vec![0.1, 0.2], None).unwrap();

        assert!((table.cumulative_hazard(8.0, 12.0, 2000.0, None).unwrap() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn cumulative_hazard_integrates_exactly_across_year_cutpoints() {
        let dimensions = vec![RateDimension::new(
            "year".to_string(),
            DimType::Year,
            vec![2000.0, 2001.0, 2002.0],
            None,
        )];
        let table = RateTable::new(dimensions, vec![0.1, 0.2], None).unwrap();
        let expected = 0.5 * DAYS_PER_YEAR * 0.1 + 0.5 * DAYS_PER_YEAR * 0.2;

        assert!(
            (table
                .cumulative_hazard(0.0, DAYS_PER_YEAR, 2000.5, None)
                .unwrap()
                - expected)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn cumulative_hazard_splits_at_the_earliest_dimension_boundary() {
        let dimensions = vec![
            RateDimension::new("age".to_string(), DimType::Age, vec![0.0, 10.0, 20.0], None),
            RateDimension::new(
                "year".to_string(),
                DimType::Year,
                vec![2000.0, 2001.0, 2002.0],
                None,
            ),
            RateDimension::new(
                "sex".to_string(),
                DimType::Factor,
                vec![],
                Some(vec!["male".to_string(), "female".to_string()]),
            ),
        ];
        let table = RateTable::new(
            dimensions,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            None,
        )
        .unwrap();
        let year_interval = 0.01 * DAYS_PER_YEAR;
        let expected = 2.0 * 2.0 + (year_interval - 2.0) * 6.0 + (4.0 - year_interval) * 8.0;

        assert!(
            (table
                .cumulative_hazard(8.0, 12.0, 2000.99, Some(1))
                .unwrap()
                - expected)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn ratetable_date_supports_pre_origin_and_century_boundaries() {
        let pre_origin = ratetable_date(1940, 1, 1, 1960).unwrap();
        assert_eq!(pre_origin.days, -7305.0);
        assert_eq!(ratetable_date(1959, 12, 31, 1960).unwrap().days, -1.0);

        assert_eq!(ratetable_date(1900, 3, 1, 1900).unwrap().days, 59.0);
        assert_eq!(ratetable_date(2000, 3, 1, 2000).unwrap().days, 60.0);
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
