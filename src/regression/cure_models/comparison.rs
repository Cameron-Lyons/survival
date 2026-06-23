
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CureModelComparisonResult {
    #[pyo3(get)]
    pub model_names: Vec<String>,
    #[pyo3(get)]
    pub log_likelihoods: Vec<f64>,
    #[pyo3(get)]
    pub aic_values: Vec<f64>,
    #[pyo3(get)]
    pub bic_values: Vec<f64>,
    #[pyo3(get)]
    pub cure_fractions: Vec<f64>,
    #[pyo3(get)]
    pub best_model_aic: String,
    #[pyo3(get)]
    pub best_model_bic: String,
}

fn parse_comparison_distribution(name: &str) -> PyResult<(CureDistribution, &'static str)> {
    match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "weibull" => Ok((CureDistribution::Weibull, "weibull")),
        "lognormal" | "log_normal" => Ok((CureDistribution::LogNormal, "lognormal")),
        "loglogistic" | "log_logistic" => {
            Ok((CureDistribution::LogLogistic, "loglogistic"))
        }
        "exponential" | "exp" => Ok((CureDistribution::Exponential, "exponential")),
        "gamma" => Ok((CureDistribution::Gamma, "gamma")),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "distribution must be one of weibull, lognormal, loglogistic, exponential, or gamma",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, distributions=None))]
pub fn compare_cure_models(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<f64>,
    distributions: Option<Vec<String>>,
) -> PyResult<CureModelComparisonResult> {
    let dists = distributions.unwrap_or_else(|| {
        vec![
            "weibull".to_string(),
            "lognormal".to_string(),
            "loglogistic".to_string(),
        ]
    });
    if dists.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "distributions must not be empty",
        ));
    }

    let mut model_names = Vec::new();
    let mut log_likelihoods = Vec::new();
    let mut aic_values = Vec::new();
    let mut bic_values = Vec::new();
    let mut cure_fractions = Vec::new();

    for dist_name in &dists {
        let (dist, dist_label) = parse_comparison_distribution(dist_name)?;

        let mixture_config = MixtureCureConfig::new(dist, LinkFunction::Logit, 50, 1e-5, 200);
        if let Ok(result) = mixture_cure_model(
            time.clone(),
            status.clone(),
            covariates.clone(),
            vec![],
            &mixture_config,
        ) {
            model_names.push(format!("Mixture-{dist_label}"));
            log_likelihoods.push(result.log_likelihood);
            aic_values.push(result.aic);
            bic_values.push(result.bic);
            cure_fractions.push(result.cure_fraction);
        }

        let bch_config = BoundedCumulativeHazardConfig::new(dist, 200, 1e-5, 1.0);
        if let Ok(result) = bounded_cumulative_hazard_model(
            time.clone(),
            status.clone(),
            covariates.clone(),
            &bch_config,
        ) {
            model_names.push(format!("BCH-{dist_label}"));
            log_likelihoods.push(result.log_likelihood);
            aic_values.push(result.aic);
            bic_values.push(result.bic);
            cure_fractions.push(result.cure_fraction);
        }

        let nm_config =
            NonMixtureCureConfig::new(NonMixtureType::GeometricGeneralized, dist, 200, 1e-5, 1.0);
        if let Ok(result) =
            non_mixture_cure_model(time.clone(), status.clone(), covariates.clone(), &nm_config)
        {
            model_names.push(format!("NonMixture-{dist_label}"));
            log_likelihoods.push(result.log_likelihood);
            aic_values.push(result.aic);
            bic_values.push(result.bic);
            cure_fractions.push(result.cure_fraction);
        }
    }

    let best_model_aic = if !aic_values.is_empty() {
        let min_idx = aic_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        model_names[min_idx].clone()
    } else {
        "None".to_string()
    };

    let best_model_bic = if !bic_values.is_empty() {
        let min_idx = bic_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        model_names[min_idx].clone()
    } else {
        "None".to_string()
    };

    Ok(CureModelComparisonResult {
        model_names,
        log_likelihoods,
        aic_values,
        bic_values,
        cure_fractions,
        best_model_aic,
        best_model_bic,
    })
}
