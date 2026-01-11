//! Embedded survival analysis datasets from the R survival package
//!
//! Each dataset is accessible via a `load_*` function that returns a Python dict
//! with column names as keys and lists/arrays as values.

mod parser;

use parser::{parse_csv, parse_f64, parse_i32};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const LUNG_CSV: &str = include_str!("data/lung.csv");
const AML_CSV: &str = include_str!("data/aml.csv");
const VETERAN_CSV: &str = include_str!("data/veteran.csv");
const OVARIAN_CSV: &str = include_str!("data/ovarian.csv");
const COLON_CSV: &str = include_str!("data/colon.csv");
const PBC_CSV: &str = include_str!("data/pbc.csv");
const CGD_CSV: &str = include_str!("data/cgd.csv");
const BLADDER_CSV: &str = include_str!("data/bladder.csv");
const HEART_CSV: &str = include_str!("data/heart.csv");
const KIDNEY_CSV: &str = include_str!("data/kidney.csv");
const RATS_CSV: &str = include_str!("data/rats.csv");
const STANFORD2_CSV: &str = include_str!("data/stanford2.csv");
const UDCA_CSV: &str = include_str!("data/udca.csv");
const MYELOID_CSV: &str = include_str!("data/myeloid.csv");
const FLCHAIN_CSV: &str = include_str!("data/flchain.csv");
const TRANSPLANT_CSV: &str = include_str!("data/transplant.csv");
const MGUS_CSV: &str = include_str!("data/mgus.csv");
const MGUS2_CSV: &str = include_str!("data/mgus2.csv");
const DIABETIC_CSV: &str = include_str!("data/diabetic.csv");
const RETINOPATHY_CSV: &str = include_str!("data/retinopathy.csv");
const GBSG_CSV: &str = include_str!("data/gbsg.csv");
const ROTTERDAM_CSV: &str = include_str!("data/rotterdam.csv");
const LOGAN_CSV: &str = include_str!("data/logan.csv");
const NWTCO_CSV: &str = include_str!("data/nwtco.csv");
const SOLDER_CSV: &str = include_str!("data/solder.csv");
const TOBIN_CSV: &str = include_str!("data/tobin.csv");
const RATS2_CSV: &str = include_str!("data/rats2.csv");
const NAFLD_CSV: &str = include_str!("data/nafld.csv");
const CGDRAW_CSV: &str = include_str!("data/cgd0.csv");
const PBCSEQ_CSV: &str = include_str!("data/pbcseq.csv");

/// Column type specification
#[derive(Clone, Copy)]
enum ColType {
    Float,
    Int,
    Str,
}

/// Helper function to parse CSV and return as Python dict
fn csv_to_dict(py: Python<'_>, csv_data: &str, schema: &[(&str, ColType)]) -> PyResult<Py<PyDict>> {
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

/// NCCTG Lung Cancer Data
///
/// Survival in patients with advanced lung cancer from the North Central Cancer
/// Treatment Group. Performance scores rate how well the patient can perform
/// usual daily activities.
///
/// Variables:
/// - inst: Institution code
/// - time: Survival time in days
/// - status: censoring status 1=censored, 2=dead
/// - age: Age in years
/// - sex: Male=1 Female=2
/// - ph.ecog: ECOG performance score (0=good 5=dead)
/// - ph.karno: Karnofsky performance score (bad=0-good=100) rated by physician
/// - pat.karno: Karnofsky performance score rated by patient
/// - meal.cal: Calories consumed at meals
/// - wt.loss: Weight loss in last six months
#[pyfunction]
pub fn load_lung(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Acute Myelogenous Leukemia survival data
///
/// Survival times in weeks for patients with acute myelogenous leukemia.
/// The main question was whether maintenance chemotherapy prolonged remission.
///
/// Variables:
/// - time: survival or censoring time
/// - cens: censoring status (1=event, 0=censored)
/// - group: maintenance chemotherapy group (1 or 2)
#[pyfunction]
pub fn load_aml(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("time", ColType::Int),
        ("cens", ColType::Int),
        ("group", ColType::Int),
    ];
    csv_to_dict(py, AML_CSV, SCHEMA)
}

/// Veterans' Administration Lung Cancer study
///
/// Randomised trial of two treatment regimens for lung cancer.
///
/// Variables:
/// - trt: 1=standard 2=test
/// - celltype: 1=squamous, 2=smallcell, 3=adeno, 4=large
/// - time: survival time
/// - status: censoring status
/// - karno: Karnofsky performance score
/// - diagtime: months from diagnosis to randomisation
/// - age: in years
/// - prior: prior therapy 0=no, 10=yes
#[pyfunction]
pub fn load_veteran(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Ovarian Cancer Survival Data
///
/// Survival in a randomised trial comparing two treatments for ovarian cancer.
///
/// Variables:
/// - futime: survival or censoring time
/// - fustat: censoring status
/// - age: in years
/// - resid.ds: residual disease present (1=no, 2=yes)
/// - rx: treatment group
/// - ecog.ps: ECOG performance status
#[pyfunction]
pub fn load_ovarian(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Chemotherapy for Stage B/C colon cancer
///
/// Survival data from a trial of adjuvant chemotherapy for colon cancer.
///
/// Variables:
/// - id: patient id
/// - study: 1 for all patients
/// - rx: treatment - Obs(ervation), Lev(amisole), Lev+5FU
/// - sex: 1=male
/// - age: in years
/// - obstruct: obstruction of colon by tumour
/// - perfor: perforation of colon
/// - adhere: adherence to nearby organs
/// - nodes: number of lymph nodes with detectable cancer
/// - time: days until event or censoring
/// - status: censoring status
/// - differ: differentiation of tumour (1=well, 2=moderate, 3=poor)
/// - extent: extent of local spread (1=submucosa, 2=muscle, 3=serosa, 4=contiguous)
/// - surg: time from surgery to registration (0=short, 1=long)
/// - node4: more than 4 positive lymph nodes
/// - etype: event type: 1=recurrence, 2=death
#[pyfunction]
pub fn load_colon(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Mayo Clinic Primary Biliary Cholangitis Data
///
/// Primary biliary cholangitis is a rare autoimmune disease of the liver.
/// This data is from a Mayo Clinic trial conducted between 1974 and 1984.
///
/// Variables include demographics, lab values, and clinical measurements.
#[pyfunction]
pub fn load_pbc(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Chronic Granulomatous Disease data
///
/// CGD is a rare inherited disorder affecting the immune system.
/// This is a placebo-controlled trial of gamma interferon.
///
/// Variables:
/// - id: subject id
/// - center: enrolling center
/// - random: date of randomization
/// - treatment: gamma interferon or placebo
/// - sex: male/female
/// - age: age in years at study entry
/// - height: height in cm at study entry
/// - weight: weight in kg at study entry
/// - inherit: pattern of inheritance
/// - steression: use of corticosteroids at study entry
/// - propession: use of prophylactic antibiotics at study entry
/// - hos.cat: institution category
/// - tstart: start of interval
/// - tstop: end of interval
/// - status: infection status (1=infection, 0=censored)
/// - enum: observation number within subject
#[pyfunction]
pub fn load_cgd(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Bladder Cancer Recurrences
///
/// Data on recurrences of bladder cancer.
///
/// Variables:
/// - id: patient id
/// - rx: treatment (1=placebo, 2=thiotepa)
/// - number: initial number of tumours
/// - size: initial size of largest tumour
/// - stop: recurrence or censoring time
/// - event: indicator of recurrence
/// - enum: event number
#[pyfunction]
pub fn load_bladder(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Stanford Heart Transplant data
///
/// Survival of patients on the waiting list for the Stanford heart transplant program.
///
/// Variables:
/// - start: start of interval
/// - stop: end of interval
/// - event: status (1=dead, 0=alive)
/// - age: age - 48 years
/// - year: year of acceptance (in years after Nov 1, 1967)
/// - surgery: prior bypass surgery (1=yes, 0=no)
/// - transplant: received transplant (1=yes, 0=no)
/// - id: patient id
#[pyfunction]
pub fn load_heart(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Kidney catheter data
///
/// Times to first and second infection in kidney patients using portable dialysis.
///
/// Variables:
/// - id: patient id
/// - time: time to infection
/// - status: event status (1=infection, 0=censored)
/// - age: patient age
/// - sex: 1=male, 2=female
/// - disease: disease type (GN, AN, PKD, Other)
/// - frail: frailty estimate from penalised model
#[pyfunction]
pub fn load_kidney(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Rat treatment data from Mantel et al
///
/// Summarized data from a 3 treatment experiment on rats.
///
/// Variables:
/// - group: treatment group
/// - n: number of rats
/// - y: number with tumour
#[pyfunction]
pub fn load_rats(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("group", ColType::Int),
        ("n", ColType::Int),
        ("y", ColType::Int),
    ];
    csv_to_dict(py, RATS_CSV, SCHEMA)
}

/// More Stanford Heart Transplant data
///
/// Contains additional information from the Stanford transplant program.
///
/// Variables:
/// - id: patient id
/// - time: survival or censoring time
/// - status: event status
/// - age: age at transplant
/// - t5: T5 mismatch score
#[pyfunction]
pub fn load_stanford2(py: Python<'_>) -> PyResult<Py<PyDict>> {
    const SCHEMA: &[(&str, ColType)] = &[
        ("id", ColType::Int),
        ("time", ColType::Float),
        ("status", ColType::Int),
        ("age", ColType::Float),
        ("t5", ColType::Float),
    ];
    csv_to_dict(py, STANFORD2_CSV, SCHEMA)
}

/// Data from a trial of ursodeoxycholic acid
///
/// A double-blinded randomised trial comparing UDCA to placebo for primary
/// biliary cirrhosis.
///
/// Variables:
/// - id: case number
/// - trt: 0=placebo, 1=UDCA
/// - entry.dt: entry date
/// - last.dt: date of last follow-up
/// - stage: histologic disease stage
/// - bili: serum bilirubin at entry
/// - riskscore: risk score
/// - death: death (0=no, 1=yes)
/// - tx: liver transplant (0=no, 1=yes)
/// - hprogress: histologic progression
/// - varices: varices (0=no, 1=yes)
/// - ascites: ascites (0=no, 1=yes)
/// - enceph: hepatic encephalopathy (0=no, 1=yes)
/// - double: doubling of bilirubin
/// - worsen: 2 point worsening of histology
#[pyfunction]
pub fn load_udca(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Acute myeloid leukemia
///
/// Subjects with acute myeloid leukemia, at 5 clinical sites.
///
/// Variables include treatment, response, relapse times, and transplant info.
#[pyfunction]
pub fn load_myeloid(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Assay of serum free light chain for 7874 subjects
///
/// This is a stratified random sample from residents of Olmsted County, MN.
///
/// Variables:
/// - age: age in years
/// - sex: F=female, M=male
/// - sample.yr: calendar year of blood sample
/// - kappa: serum free light chain, kappa portion
/// - lambda: serum free light chain, lambda portion
/// - flc.grp: FLC group for analysis
/// - creatinine: serum creatinine
/// - mgus: 1 if MGUS at baseline, 0 otherwise
/// - futime: days from enrollment to death or last follow-up
/// - death: 0=alive, 1=dead
/// - chapter: for those who died, the chapter in ICD-9/10 code
#[pyfunction]
pub fn load_flchain(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Liver transplant waiting list
///
/// Subjects on a liver transplant waiting list from 1990-1999.
///
/// Variables:
/// - age: age at registration
/// - sex: m=male, f=female
/// - abo: blood type A, B, AB, or O
/// - year: year of registration
/// - futime: time to death, censoring, or transplant
/// - event: ltx=transplant, death, censor
#[pyfunction]
pub fn load_transplant(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Monoclonal gammopathy data
///
/// Natural history of 241 subjects with monoclonal gammopathy of undetermined
/// significance (MGUS).
///
/// Variables:
/// - id: subject id
/// - age: age at diagnosis
/// - sex: male or female
/// - dxyr: year of diagnosis
/// - pcdx: plasma cell percentage at diagnosis
/// - mspike: size of monoclonal spike
/// - futime: follow-up time in months
/// - death: 1=death, 0=alive
/// - alession: 1=progression to AL amyloidosis, 0=no
/// - mmdx: 1=progression to myeloma, 0=no
#[pyfunction]
pub fn load_mgus(py: Python<'_>) -> PyResult<Py<PyDict>> {
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

/// Monoclonal gammopathy data (extended)
///
/// Updated and expanded MGUS dataset with additional patients and follow-up.
#[pyfunction]
pub fn load_mgus2(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_diabetic(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_retinopathy(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_gbsg(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_rotterdam(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_logan(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_nwtco(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_solder(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_tobin(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_rats2(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_nafld(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_cgd0(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
pub fn load_pbcseq(py: Python<'_>) -> PyResult<Py<PyDict>> {
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
