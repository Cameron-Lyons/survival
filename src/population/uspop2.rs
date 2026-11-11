use ndarray::{Array3, ShapeBuilder};
use pyo3::prelude::*;
use std::sync::OnceLock;

const USPOP2: &str = include_str!("data/uspop2.tsv");

#[doc = " R's projected US population by age, sex and calendar year."]
#[doc = " These are the Census Bureau's 2008 projections, not observed census counts."]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[pyclass(module = "survival._survival", from_py_object)]
pub struct Uspop2Data {
    #[doc = " Population counts indexed by age, sex and year, with shape (101, 2, 21)."]
    #[doc = " Ages are 0 through 100 (100 includes all older ages), sexes are male/female,"]
    #[doc = " and years are 2000 through 2020."]
    pub counts: Array3<i32>,
    #[doc = " R's three dimnames vectors, in age, sex and year order."]
    #[pyo3(get)]
    pub dimnames: Vec<Vec<String>>,
}

#[pymethods]
impl Uspop2Data {
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.counts.dim()
    }

    #[doc = " An owned, mutable int32 NumPy snapshot with age, sex and year axes."]
    #[cfg(feature = "python")]
    #[getter(counts)]
    fn counts_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray3<i32>> {
        use numpy::IntoPyArray;
        self.counts.clone().into_pyarray(py)
    }

    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }
}

fn parse() -> Uspop2Data {
    let mut lines = USPOP2.lines();
    let dimnames: Vec<Vec<String>> = ["age", "sex", "year"]
        .into_iter()
        .map(|axis| {
            let mut fields = lines.next().expect("embedded uspop2 axis").split('\t');
            assert_eq!(fields.next(), Some(axis), "embedded uspop2 axis order");
            fields.map(str::to_owned).collect()
        })
        .collect();
    let shape = (dimnames[0].len(), dimnames[1].len(), dimnames[2].len());
    assert_eq!(shape, (101, 2, 21), "embedded uspop2 dimensions");
    let values = lines
        .map(|line| line.parse::<i32>().expect("embedded uspop2 integer count"))
        .collect();
    let counts = Array3::from_shape_vec(shape.f(), values).expect("embedded uspop2 count length");
    Uspop2Data { counts, dimnames }
}

#[doc = " Load R's `uspop2` population projections, preserving its values and axis labels."]
#[doc = " The embedded table is parsed once; each call returns an independent owned copy."]
pub fn uspop2() -> Uspop2Data {
    static DATA: OnceLock<Uspop2Data> = OnceLock::new();
    DATA.get_or_init(parse).clone()
}

#[doc = " Load R's projected US population as an age-by-sex-by-year array and dimnames."]
#[pyfunction(name = "load_uspop2")]
pub(crate) fn load_uspop2_py() -> Uspop2Data {
    uspop2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn reference() -> Value {
        serde_json::from_str(include_str!(
            "../../python/tests/fixtures/uspop2_reference.json"
        ))
        .unwrap()
    }

    #[test]
    fn every_population_count_and_axis_matches_stock_r() {
        let data = uspop2();
        let expected = reference();
        assert_eq!(data.counts.shape(), &[101, 2, 21]);
        assert_eq!(
            serde_json::to_value(&data.dimnames).unwrap(),
            expected["dimnames"]
        );
        let counts = expected["counts"].as_array().unwrap();
        assert_eq!(counts.len(), data.counts.len());
        for (index, value) in counts.iter().enumerate() {
            let age = index % 101;
            let sex = (index / 101) % 2;
            let year = index / 202;
            assert_eq!(
                i64::from(data.counts[[age, sex, year]]),
                value.as_i64().unwrap()
            );
        }
    }

    #[test]
    fn standard_population_age_groups_match_r_example() {
        let data = uspop2();
        let expected = reference();
        let groups = expected["age50_groups_2000"]["counts"].as_array().unwrap();
        for (group, expected) in groups.iter().enumerate() {
            let start = 50 + 5 * group;
            let end = (start + 5).min(101);
            for sex in 0..2 {
                let total: i64 = (start..end)
                    .map(|age| i64::from(data.counts[[age, sex, 0]]))
                    .sum();
                assert_eq!(total, expected[sex].as_i64().unwrap());
            }
        }
        assert_eq!(
            data.counts
                .iter()
                .map(|&value| i64::from(value))
                .sum::<i64>(),
            6_526_228_928
        );
    }

    #[test]
    fn loads_do_not_share_mutable_counts_or_labels() {
        let mut changed = uspop2();
        changed.counts[[0, 0, 0]] = 0;
        changed.dimnames[1][0] = "changed".to_owned();
        let original = uspop2();
        assert_eq!(original.counts[[0, 0, 0]], 1_965_147);
        assert_eq!(original.dimnames[1], ["male", "female"]);
    }
}
