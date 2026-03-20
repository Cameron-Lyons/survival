use super::parser::{parse_csv, parse_f64, parse_i32};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(super) const LUNG_CSV: &str = include_str!("data/lung.csv");
pub(super) const AML_CSV: &str = include_str!("data/aml.csv");
pub(super) const VETERAN_CSV: &str = include_str!("data/veteran.csv");
pub(super) const OVARIAN_CSV: &str = include_str!("data/ovarian.csv");
pub(super) const COLON_CSV: &str = include_str!("data/colon.csv");
pub(super) const PBC_CSV: &str = include_str!("data/pbc.csv");
pub(super) const CGD_CSV: &str = include_str!("data/cgd.csv");
pub(super) const BLADDER_CSV: &str = include_str!("data/bladder.csv");
pub(super) const HEART_CSV: &str = include_str!("data/heart.csv");
pub(super) const KIDNEY_CSV: &str = include_str!("data/kidney.csv");
pub(super) const RATS_CSV: &str = include_str!("data/rats.csv");
pub(super) const STANFORD2_CSV: &str = include_str!("data/stanford2.csv");
pub(super) const UDCA_CSV: &str = include_str!("data/udca.csv");
pub(super) const MYELOID_CSV: &str = include_str!("data/myeloid.csv");
pub(super) const FLCHAIN_CSV: &str = include_str!("data/flchain.csv");
pub(super) const TRANSPLANT_CSV: &str = include_str!("data/transplant.csv");
pub(super) const MGUS_CSV: &str = include_str!("data/mgus.csv");
pub(super) const MGUS2_CSV: &str = include_str!("data/mgus2.csv");
pub(super) const DIABETIC_CSV: &str = include_str!("data/diabetic.csv");
pub(super) const RETINOPATHY_CSV: &str = include_str!("data/retinopathy.csv");
pub(super) const GBSG_CSV: &str = include_str!("data/gbsg.csv");
pub(super) const ROTTERDAM_CSV: &str = include_str!("data/rotterdam.csv");
pub(super) const LOGAN_CSV: &str = include_str!("data/logan.csv");
pub(super) const NWTCO_CSV: &str = include_str!("data/nwtco.csv");
pub(super) const SOLDER_CSV: &str = include_str!("data/solder.csv");
pub(super) const TOBIN_CSV: &str = include_str!("data/tobin.csv");
pub(super) const RATS2_CSV: &str = include_str!("data/rats2.csv");
pub(super) const NAFLD_CSV: &str = include_str!("data/nafld.csv");
pub(super) const CGDRAW_CSV: &str = include_str!("data/cgd0.csv");
pub(super) const PBCSEQ_CSV: &str = include_str!("data/pbcseq.csv");
pub(super) const HOEL_CSV: &str = include_str!("data/hoel.csv");
pub(super) const MYELOMA_CSV: &str = include_str!("data/myeloma.csv");
pub(super) const RHDNASE_CSV: &str = include_str!("data/rhDNase.csv");

#[derive(Clone, Copy)]
pub(super) enum ColType {
    Float,
    Int,
    Str,
}

pub(super) fn csv_to_dict(
    py: Python<'_>,
    csv_data: &str,
    schema: &[(&str, ColType)],
) -> PyResult<Py<PyDict>> {
    let (headers, rows) = parse_csv(csv_data).map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);

    for (col_name, col_type) in schema {
        let idx = headers.iter().position(|h| h == *col_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Column '{}' not found in CSV",
                col_name
            ))
        })?;

        match col_type {
            ColType::Float => {
                let values: Vec<Option<f64>> =
                    rows.iter().map(|row| parse_f64(&row[idx])).collect();
                let list = PyList::new(py, values.iter().map(|v| v.map(|x| x)))?;
                dict.set_item(*col_name, list)?;
            }
            ColType::Int => {
                let values: Vec<Option<i32>> =
                    rows.iter().map(|row| parse_i32(&row[idx])).collect();
                let list = PyList::new(py, values.iter().map(|v| v.map(|x| x)))?;
                dict.set_item(*col_name, list)?;
            }
            ColType::Str => {
                let values: Vec<&str> = rows.iter().map(|row| row[idx].as_str()).collect();
                let list = PyList::new(py, values)?;
                dict.set_item(*col_name, list)?;
            }
        }
    }

    dict.set_item("_nrow", rows.len())?;
    dict.set_item("_ncol", schema.len())?;

    Ok(dict.into())
}
