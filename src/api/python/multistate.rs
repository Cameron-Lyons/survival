use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survfitaj_extended, m)?)?;
    m.add_class::<AalenJohansenExtendedConfig>()?;
    m.add_class::<AalenJohansenExtendedResult>()?;
    m.add_class::<TransitionMatrix>()?;
    m.add_class::<TransitionType>()?;
    m.add_class::<VarianceEstimator>()?;

    m.add_function(wrap_pyfunction!(estimate_transition_intensities, m)?)?;
    m.add_function(wrap_pyfunction!(fit_multi_state_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_markov_msm, m)?)?;
    m.add_class::<MultiStateConfig>()?;
    m.add_class::<MultiStateResult>()?;
    m.add_class::<TransitionIntensityResult>()?;
    m.add_class::<MarkovMSMResult>()?;

    m.add_function(wrap_pyfunction!(fit_semi_markov, m)?)?;
    m.add_function(wrap_pyfunction!(predict_semi_markov, m)?)?;
    m.add_class::<SemiMarkovConfig>()?;
    m.add_class::<SemiMarkovResult>()?;
    m.add_class::<SemiMarkovPrediction>()?;
    m.add_class::<SojournDistribution>()?;
    m.add_class::<SojournTimeParams>()?;

    m.add_function(wrap_pyfunction!(fit_illness_death, m)?)?;
    m.add_function(wrap_pyfunction!(predict_illness_death, m)?)?;
    m.add_class::<IllnessDeathConfig>()?;
    m.add_class::<IllnessDeathResult>()?;
    m.add_class::<IllnessDeathPrediction>()?;
    m.add_class::<IllnessDeathType>()?;
    m.add_class::<TransitionHazard>()?;

    Ok(())
}
