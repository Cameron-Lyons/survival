use super::common::{
    CGDRAW_CSV, ColType, DIABETIC_CSV, GBSG_CSV, HOEL_CSV, LOGAN_CSV, MYELOMA_CSV, NAFLD_CSV,
    NWTCO_CSV, PBCSEQ_CSV, RATS2_CSV, RETINOPATHY_CSV, RHDNASE_CSV, ROTTERDAM_CSV, SOLDER_CSV,
    TOBIN_CSV, csv_to_dict,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Diabetic Retinopathy
///
/// Partial results from a trial of laser coagulation for diabetic retinopathy.
///
/// Variables:
/// - id: subject id
/// - laser: type of laser (xenon or argon)
/// - age: age at diagnosis
/// - eye: eye treated (left or right)
/// - trt: 0=no treatment, 1=treatment
/// - risk: risk group (6-12)
/// - time: time to vision loss or censoring
/// - status: 0=censored, 1=vision loss
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

/// Diabetic Retinopathy Study
///
/// Alternative formatting of the diabetic retinopathy data.
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

/// German Breast Cancer Study Group
///
/// Data from the German Breast Cancer Study Group 2 trial.
///
/// Variables:
/// - pid: patient id
/// - age: age in years
/// - meno: menopausal status (0=pre, 1=post)
/// - size: tumour size in mm
/// - grade: tumour grade (1-3)
/// - nodes: number of positive nodes
/// - pgr: progesterone receptors (fmol/l)
/// - er: estrogen receptors (fmol/l)
/// - hormon: hormone therapy (0=no, 1=yes)
/// - rfstime: recurrence-free survival time in days
/// - status: 0=censored, 1=recurrence
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

/// Rotterdam Tumor Bank data
///
/// Breast cancer patients from the Rotterdam tumor bank.
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

/// Data from the 1972-78 General Social Survey
///
/// Used by Logan (1983) to illustrate partial likelihood estimation.
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

/// Data from the National Wilms Tumor Study
///
/// Histology data from the National Wilms Tumor Study (NWTCO).
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

/// Soldering Experiment
///
/// Data from a designed experiment on wave soldering.
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

/// Tobin's Tobit data
///
/// Data from Tobin (1958), used to illustrate the tobit model.
#[pyfunction]
pub(crate) fn load_tobin(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("durable", ColType::Float),
        ("age", ColType::Int),
        ("quant", ColType::Int),
    ];
    csv_to_dict(py, TOBIN_CSV, SCHEMA)
}

/// Rat data from Gail et al
///
/// Litter-matched data on time to tumour in rats.
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

/// Non-alcoholic fatty liver disease
///
/// Subjects with NAFLD from the Rochester Epidemiology Project.
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

/// Chronic Granulomatous Disease (raw data)
///
/// The raw data version of the CGD dataset, before conversion to counting process format.
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

/// Mayo Clinic Primary Biliary Cirrhosis (sequential data)
///
/// Sequential measurements for the PBC dataset.
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

/// Hoel (1972) data on causes of death in RFM mice
///
/// Data from a radiation experiment on RFM mice. Each mouse was followed until
/// death and the cause of death was recorded.
///
/// Variables:
/// - time: time to death in days
/// - status: 1=died, 0=censored
/// - cause: cause of death (1=thymic lymphoma, 2=reticulum cell sarcoma,
///          3=other causes, 4=lung tumour, 0=censored)
#[pyfunction]
pub(crate) fn load_hoel(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("time", ColType::Int),
        ("status", ColType::Int),
        ("cause", ColType::Int),
    ];
    csv_to_dict(py, HOEL_CSV, SCHEMA)
}

/// Multiple Myeloma Data
///
/// Survival of multiple myeloma patients, used in Krall, Uthoff, and Harley (1975).
///
/// Variables:
/// - time: survival time in months from diagnosis
/// - status: 1=died, 0=censored
/// - hgb: hemoglobin at diagnosis
/// - bun: blood urea nitrogen at diagnosis
/// - ca: serum calcium at diagnosis
/// - protein: proteinuria at diagnosis
/// - pcells: percent of plasma cells in bone marrow
/// - age: age in years
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

/// rhDNase clinical trial data
///
/// Data from a randomized trial of recombinant human deoxyribonuclease (rhDNase)
/// for treatment of cystic fibrosis. The endpoint was time to first pulmonary
/// exacerbation requiring intravenous (IV) antibiotic therapy.
///
/// Variables:
/// - id: patient id
/// - inst: institution
/// - trt: treatment (0=placebo, 1=rhDNase)
/// - fev: baseline forced expiratory volume (FEV) as percent predicted
/// - entry: entry time (0 for all)
/// - fev.last: FEV at last observation
/// - ivstart: start of IV therapy (days from entry), NA if no event
/// - ivstop: end of IV therapy (days from entry), NA if no event
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
