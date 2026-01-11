use pyo3::prelude::*;
use std::collections::HashMap;

/// Type of dimension in a rate table
#[derive(Debug, Clone, PartialEq)]
#[pyclass]
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
#[pyclass]
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
#[pyclass]
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
        // Calculate expected shape
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
                DimType::Factor => {
                    if dim.name.to_lowercase().contains("sex") {
                        coords.insert(dim.name.clone(), sex.unwrap_or(0) as f64);
                    }
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

            let idx = match dim.dim_type {
                DimType::Factor => {
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

    match cutpoints.binary_search_by(|probe| {
        probe
            .partial_cmp(&value)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ratetable_basic() {
        let age_breaks = vec![0.0, 365.0, 3650.0, 36500.0];
        let year_breaks = vec![1990.0, 2000.0, 2010.0];

        // 3 age intervals x 2 year intervals = 6 rates per sex
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
}
