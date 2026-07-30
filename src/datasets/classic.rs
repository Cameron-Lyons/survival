use super::common::{
    AML_CSV, BLADDER_CSV, CGD_CSV, COLON_CSV, ColType, FLCHAIN_CSV, HEART_CSV, KIDNEY_CSV,
    LUNG_CSV, MGUS_CSV, MGUS2_CSV, MYELOID_CSV, OVARIAN_CSV, PBC_CSV, RATS_CSV, STANFORD2_CSV,
    TRANSPLANT_CSV, UDCA_CSV, VETERAN_CSV, csv_to_dict,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub(crate) fn load_lung(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("inst", ColType::Int),
        ("time", ColType::Int),
        ("status", ColType::Int),
        ("age", ColType::Int),
        ("sex", ColType::Int),
        ("ph.ecog", ColType::Int),
        ("ph.karno", ColType::Int),
        ("pat.karno", ColType::Int),
        ("meal.cal", ColType::Int),
        ("wt.loss", ColType::Int),
    ];
    csv_to_dict(py, LUNG_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_aml(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("time", ColType::Int),
        ("cens", ColType::Int),
        ("group", ColType::Int),
    ];
    csv_to_dict(py, AML_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_veteran(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("trt", ColType::Int),
        ("celltype", ColType::Str),
        ("time", ColType::Float),
        ("status", ColType::Int),
        ("karno", ColType::Int),
        ("diagtime", ColType::Int),
        ("age", ColType::Int),
        ("prior", ColType::Int),
    ];
    csv_to_dict(py, VETERAN_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_ovarian(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("futime", ColType::Float),
        ("fustat", ColType::Int),
        ("age", ColType::Float),
        ("resid.ds", ColType::Int),
        ("rx", ColType::Int),
        ("ecog.ps", ColType::Int),
    ];
    csv_to_dict(py, OVARIAN_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_colon(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("study", ColType::Int),
        ("rx", ColType::Str),
        ("sex", ColType::Int),
        ("age", ColType::Int),
        ("obstruct", ColType::Int),
        ("perfor", ColType::Int),
        ("adhere", ColType::Int),
        ("nodes", ColType::Int),
        ("time", ColType::Int),
        ("status", ColType::Int),
        ("differ", ColType::Int),
        ("extent", ColType::Int),
        ("surg", ColType::Int),
        ("node4", ColType::Int),
        ("etype", ColType::Int),
    ];
    csv_to_dict(py, COLON_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_pbc(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("time", ColType::Int),
        ("status", ColType::Int),
        ("trt", ColType::Int),
        ("age", ColType::Float),
        ("sex", ColType::Str),
        ("ascites", ColType::Int),
        ("hepato", ColType::Int),
        ("spiders", ColType::Int),
        ("edema", ColType::Float),
        ("bili", ColType::Float),
        ("chol", ColType::Int),
        ("albumin", ColType::Float),
        ("copper", ColType::Int),
        ("alk.phos", ColType::Float),
        ("ast", ColType::Float),
        ("trig", ColType::Int),
        ("platelet", ColType::Int),
        ("protime", ColType::Float),
        ("stage", ColType::Int),
    ];
    csv_to_dict(py, PBC_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_cgd(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("center", ColType::Int),
        ("random", ColType::Str),
        ("treat", ColType::Str),
        ("sex", ColType::Str),
        ("age", ColType::Float),
        ("height", ColType::Float),
        ("weight", ColType::Float),
        ("inherit", ColType::Str),
        ("steroids", ColType::Int),
        ("propylac", ColType::Int),
        ("hos.cat", ColType::Str),
        ("tstart", ColType::Int),
        ("enum", ColType::Int),
        ("tstop", ColType::Int),
        ("status", ColType::Int),
    ];
    csv_to_dict(py, CGD_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_bladder(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("rx", ColType::Int),
        ("number", ColType::Int),
        ("size", ColType::Int),
        ("stop", ColType::Int),
        ("event", ColType::Int),
        ("enum", ColType::Int),
    ];
    csv_to_dict(py, BLADDER_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_heart(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("start", ColType::Int),
        ("stop", ColType::Int),
        ("event", ColType::Int),
        ("age", ColType::Float),
        ("year", ColType::Float),
        ("surgery", ColType::Int),
        ("transplant", ColType::Int),
        ("id", ColType::Int),
    ];
    csv_to_dict(py, HEART_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_kidney(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("time", ColType::Float),
        ("status", ColType::Int),
        ("age", ColType::Int),
        ("sex", ColType::Int),
        ("disease", ColType::Str),
        ("frail", ColType::Float),
    ];
    csv_to_dict(py, KIDNEY_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_rats(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("group", ColType::Int),
        ("n", ColType::Int),
        ("y", ColType::Int),
    ];
    csv_to_dict(py, RATS_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_stanford2(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("time", ColType::Float),
        ("status", ColType::Int),
        ("age", ColType::Float),
        ("t5", ColType::Float),
    ];
    csv_to_dict(py, STANFORD2_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_udca(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("trt", ColType::Int),
        ("entry.dt", ColType::Str),
        ("last.dt", ColType::Str),
        ("stage", ColType::Int),
        ("bili", ColType::Float),
        ("riskscore", ColType::Float),
        ("death.dt", ColType::Str),
        ("tx.dt", ColType::Str),
        ("hprogress.dt", ColType::Str),
        ("varices.dt", ColType::Str),
        ("ascites.dt", ColType::Str),
        ("enceph.dt", ColType::Str),
        ("double.dt", ColType::Str),
        ("worsen.dt", ColType::Str),
    ];
    csv_to_dict(py, UDCA_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_myeloid(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("trt", ColType::Str),
        ("sex", ColType::Str),
        ("flt3", ColType::Str),
        ("futime", ColType::Int),
        ("death", ColType::Int),
        ("txtime", ColType::Int),
        ("crtime", ColType::Int),
        ("rltime", ColType::Int),
    ];
    csv_to_dict(py, MYELOID_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_flchain(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("age", ColType::Int),
        ("sex", ColType::Str),
        ("sample.yr", ColType::Int),
        ("kappa", ColType::Float),
        ("lambda", ColType::Float),
        ("flc.grp", ColType::Int),
        ("creatinine", ColType::Float),
        ("mgus", ColType::Int),
        ("futime", ColType::Int),
        ("death", ColType::Int),
        ("chapter", ColType::Str),
    ];
    csv_to_dict(py, FLCHAIN_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_transplant(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("age", ColType::Float),
        ("sex", ColType::Str),
        ("abo", ColType::Str),
        ("year", ColType::Int),
        ("futime", ColType::Int),
        ("event", ColType::Str),
    ];
    csv_to_dict(py, TRANSPLANT_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_mgus(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("age", ColType::Int),
        ("sex", ColType::Str),
        ("dxyr", ColType::Int),
        ("pcdx", ColType::Float),
        ("pctime", ColType::Int),
        ("futime", ColType::Int),
        ("death", ColType::Int),
        ("alb", ColType::Float),
        ("creat", ColType::Float),
        ("hgb", ColType::Float),
        ("mspike", ColType::Float),
    ];
    csv_to_dict(py, MGUS_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_mgus2(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("age", ColType::Int),
        ("sex", ColType::Str),
        ("dxyr", ColType::Int),
        ("hgb", ColType::Float),
        ("creat", ColType::Float),
        ("mspike", ColType::Float),
        ("ptime", ColType::Int),
        ("pstat", ColType::Int),
        ("futime", ColType::Int),
        ("death", ColType::Int),
    ];
    csv_to_dict(py, MGUS2_CSV, SCHEMA)
}
