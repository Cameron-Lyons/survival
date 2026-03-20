use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        crate::datasets::load_lung,
        crate::datasets::load_aml,
        crate::datasets::load_veteran,
        crate::datasets::load_ovarian,
        crate::datasets::load_colon,
        crate::datasets::load_pbc,
        crate::datasets::load_cgd,
        crate::datasets::load_bladder,
        crate::datasets::load_heart,
        crate::datasets::load_kidney,
        crate::datasets::load_rats,
        crate::datasets::load_stanford2,
        crate::datasets::load_udca,
        crate::datasets::load_myeloid,
        crate::datasets::load_flchain,
        crate::datasets::load_transplant,
        crate::datasets::load_mgus,
        crate::datasets::load_mgus2,
        crate::datasets::load_diabetic,
        crate::datasets::load_retinopathy,
        crate::datasets::load_gbsg,
        crate::datasets::load_rotterdam,
        crate::datasets::load_logan,
        crate::datasets::load_nwtco,
        crate::datasets::load_solder,
        crate::datasets::load_tobin,
        crate::datasets::load_rats2,
        crate::datasets::load_nafld,
        crate::datasets::load_cgd0,
        crate::datasets::load_pbcseq,
        crate::datasets::load_hoel,
        crate::datasets::load_myeloma,
        crate::datasets::load_rhdnase,
    );

    Ok(())
}
