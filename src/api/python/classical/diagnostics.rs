use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coxph_wtest_py, m)?)?;

    register_classes!(m, CoxphWtest,);

    Ok(())
}
