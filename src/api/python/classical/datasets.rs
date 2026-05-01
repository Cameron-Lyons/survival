use super::*;
use crate::datasets::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_lung, m)?)?;
    m.add_function(wrap_pyfunction!(load_aml, m)?)?;
    m.add_function(wrap_pyfunction!(load_veteran, m)?)?;
    m.add_function(wrap_pyfunction!(load_ovarian, m)?)?;
    m.add_function(wrap_pyfunction!(load_colon, m)?)?;
    m.add_function(wrap_pyfunction!(load_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(load_cgd, m)?)?;
    m.add_function(wrap_pyfunction!(load_bladder, m)?)?;
    m.add_function(wrap_pyfunction!(load_heart, m)?)?;
    m.add_function(wrap_pyfunction!(load_kidney, m)?)?;
    m.add_function(wrap_pyfunction!(load_rats, m)?)?;
    m.add_function(wrap_pyfunction!(load_stanford2, m)?)?;
    m.add_function(wrap_pyfunction!(load_udca, m)?)?;
    m.add_function(wrap_pyfunction!(load_myeloid, m)?)?;
    m.add_function(wrap_pyfunction!(load_flchain, m)?)?;
    m.add_function(wrap_pyfunction!(load_transplant, m)?)?;
    m.add_function(wrap_pyfunction!(load_mgus, m)?)?;
    m.add_function(wrap_pyfunction!(load_mgus2, m)?)?;
    m.add_function(wrap_pyfunction!(load_diabetic, m)?)?;
    m.add_function(wrap_pyfunction!(load_retinopathy, m)?)?;
    m.add_function(wrap_pyfunction!(load_gbsg, m)?)?;
    m.add_function(wrap_pyfunction!(load_rotterdam, m)?)?;
    m.add_function(wrap_pyfunction!(load_logan, m)?)?;
    m.add_function(wrap_pyfunction!(load_nwtco, m)?)?;
    m.add_function(wrap_pyfunction!(load_solder, m)?)?;
    m.add_function(wrap_pyfunction!(load_tobin, m)?)?;
    m.add_function(wrap_pyfunction!(load_rats2, m)?)?;
    m.add_function(wrap_pyfunction!(load_nafld, m)?)?;
    m.add_function(wrap_pyfunction!(load_cgd0, m)?)?;
    m.add_function(wrap_pyfunction!(load_pbcseq, m)?)?;
    m.add_function(wrap_pyfunction!(load_hoel, m)?)?;
    m.add_function(wrap_pyfunction!(load_myeloma, m)?)?;
    m.add_function(wrap_pyfunction!(load_rhdnase, m)?)?;

    Ok(())
}
