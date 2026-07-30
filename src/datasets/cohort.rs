use super::common::{
    CGDRAW_CSV, ColType, DIABETIC_CSV, GBSG_CSV, HOEL_CSV, LOGAN_CSV, MYELOMA_CSV, NAFLD_CSV,
    NWTCO_CSV, PBCSEQ_CSV, RATS2_CSV, RETINOPATHY_CSV, RHDNASE_CSV, ROTTERDAM_CSV, SOLDER_CSV,
    TOBIN_CSV, csv_to_dict,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub(crate) fn load_diabetic(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("laser", ColType::Str),
        ("age", ColType::Int),
        ("eye", ColType::Str),
        ("trt", ColType::Int),
        ("risk", ColType::Int),
        ("time", ColType::Float),
        ("status", ColType::Int),
    ];
    csv_to_dict(py, DIABETIC_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_retinopathy(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("laser", ColType::Str),
        ("eye", ColType::Str),
        ("age", ColType::Int),
        ("type", ColType::Str),
        ("trt", ColType::Int),
        ("futime", ColType::Float),
        ("status", ColType::Int),
        ("risk", ColType::Int),
    ];
    csv_to_dict(py, RETINOPATHY_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_gbsg(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("pid", ColType::Int),
        ("age", ColType::Int),
        ("meno", ColType::Int),
        ("size", ColType::Int),
        ("grade", ColType::Int),
        ("nodes", ColType::Int),
        ("pgr", ColType::Int),
        ("er", ColType::Int),
        ("hormon", ColType::Int),
        ("rfstime", ColType::Int),
        ("status", ColType::Int),
    ];
    csv_to_dict(py, GBSG_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_rotterdam(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("pid", ColType::Int),
        ("year", ColType::Int),
        ("age", ColType::Int),
        ("meno", ColType::Int),
        ("size", ColType::Int),
        ("grade", ColType::Int),
        ("nodes", ColType::Int),
        ("pgr", ColType::Int),
        ("er", ColType::Int),
        ("hormon", ColType::Int),
        ("chemo", ColType::Int),
        ("rtime", ColType::Int),
        ("recur", ColType::Int),
        ("dtime", ColType::Int),
        ("death", ColType::Int),
    ];
    csv_to_dict(py, ROTTERDAM_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_logan(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("occupation", ColType::Int),
        ("focc", ColType::Int),
        ("education", ColType::Int),
        ("race", ColType::Str),
    ];
    csv_to_dict(py, LOGAN_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_nwtco(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("seqno", ColType::Int),
        ("instit", ColType::Int),
        ("histol", ColType::Int),
        ("stage", ColType::Int),
        ("study", ColType::Int),
        ("rel", ColType::Int),
        ("edrel", ColType::Float),
        ("age", ColType::Int),
        ("in.subcohort", ColType::Int),
    ];
    csv_to_dict(py, NWTCO_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_solder(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("Opening", ColType::Str),
        ("Solder", ColType::Str),
        ("Mask", ColType::Str),
        ("PadType", ColType::Str),
        ("Panel", ColType::Int),
        ("skips", ColType::Int),
    ];
    csv_to_dict(py, SOLDER_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_tobin(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("durable", ColType::Float),
        ("age", ColType::Int),
        ("quant", ColType::Int),
    ];
    csv_to_dict(py, TOBIN_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_rats2(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("trt", ColType::Int),
        ("obs", ColType::Int),
        ("time1", ColType::Int),
        ("time2", ColType::Int),
        ("status", ColType::Int),
    ];
    csv_to_dict(py, RATS2_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_nafld(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("age", ColType::Float),
        ("male", ColType::Int),
        ("weight", ColType::Float),
        ("height", ColType::Float),
        ("bmi", ColType::Float),
        ("case.id", ColType::Int),
        ("futime", ColType::Int),
        ("status", ColType::Int),
    ];
    csv_to_dict(py, NAFLD_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_cgd0(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
        ("futime", ColType::Int),
        ("etime1", ColType::Int),
        ("etime2", ColType::Int),
        ("etime3", ColType::Int),
        ("etime4", ColType::Int),
        ("etime5", ColType::Int),
        ("etime6", ColType::Int),
        ("etime7", ColType::Int),
    ];
    csv_to_dict(py, CGDRAW_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_pbcseq(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("futime", ColType::Int),
        ("status", ColType::Int),
        ("trt", ColType::Int),
        ("age", ColType::Float),
        ("sex", ColType::Str),
        ("day", ColType::Int),
        ("ascites", ColType::Int),
        ("hepato", ColType::Int),
        ("spiders", ColType::Int),
        ("edema", ColType::Float),
        ("bili", ColType::Float),
        ("chol", ColType::Int),
        ("albumin", ColType::Float),
        ("alk.phos", ColType::Float),
        ("ast", ColType::Float),
        ("platelet", ColType::Int),
        ("protime", ColType::Float),
        ("stage", ColType::Int),
    ];
    csv_to_dict(py, PBCSEQ_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_hoel(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("time", ColType::Int),
        ("status", ColType::Int),
        ("cause", ColType::Int),
    ];
    csv_to_dict(py, HOEL_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_myeloma(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("time", ColType::Int),
        ("status", ColType::Int),
        ("hgb", ColType::Float),
        ("bun", ColType::Int),
        ("ca", ColType::Int),
        ("protein", ColType::Int),
        ("pcells", ColType::Int),
        ("age", ColType::Int),
    ];
    csv_to_dict(py, MYELOMA_CSV, SCHEMA)
}

#[pyfunction]
pub(crate) fn load_rhdnase(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("inst", ColType::Int),
        ("trt", ColType::Int),
        ("fev", ColType::Float),
        ("entry", ColType::Int),
        ("fev.last", ColType::Float),
        ("ivstart", ColType::Str),
        ("ivstop", ColType::Str),
    ];
    csv_to_dict(py, RHDNASE_CSV, SCHEMA)
}
