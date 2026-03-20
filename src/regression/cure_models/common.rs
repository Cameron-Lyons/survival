

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

pub enum CureDistribution {
    Weibull,
    LogNormal,
    LogLogistic,
    Exponential,
    Gamma,
}

#[pymethods]
impl CureDistribution {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "weibull" => Ok(CureDistribution::Weibull),
            "lognormal" | "log_normal" => Ok(CureDistribution::LogNormal),
            "loglogistic" | "log_logistic" => Ok(CureDistribution::LogLogistic),
            "exponential" | "exp" => Ok(CureDistribution::Exponential),
            "gamma" => Ok(CureDistribution::Gamma),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown distribution",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum LinkFunction {
    Logit,
    Probit,
    CLogLog,
    Identity,
}

#[pymethods]
impl LinkFunction {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "logit" => Ok(LinkFunction::Logit),
            "probit" => Ok(LinkFunction::Probit),
            "cloglog" | "c_log_log" => Ok(LinkFunction::CLogLog),
            "identity" => Ok(LinkFunction::Identity),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown link function",
            )),
        }
    }

    fn link(&self, p: f64) -> f64 {
        let p_clamped = p.clamp(
            crate::constants::DIVISION_FLOOR,
            1.0 - crate::constants::DIVISION_FLOOR,
        );
        match self {
            LinkFunction::Logit => (p_clamped / (1.0 - p_clamped)).ln(),
            LinkFunction::Probit => probit(p_clamped),
            LinkFunction::CLogLog => (-(-p_clamped).ln_1p()).ln(),
            LinkFunction::Identity => p_clamped,
        }
    }

    fn inv_link(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Logit => 1.0 / (1.0 + (-eta).exp()),
            LinkFunction::Probit => normal_cdf(eta),
            LinkFunction::CLogLog => 1.0 - (-eta.exp()).exp(),
            LinkFunction::Identity => eta.clamp(0.0, 1.0),
        }
    }

    fn deriv(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Logit => {
                let p = 1.0 / (1.0 + (-eta).exp());
                p * (1.0 - p)
            }
            LinkFunction::Probit => normal_pdf(eta),
            LinkFunction::CLogLog => {
                let exp_eta = eta.exp();
                exp_eta * (-exp_eta).exp()
            }
            LinkFunction::Identity => 1.0,
        }
    }
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn weibull_surv(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    (-(t / scale).powf(shape)).exp()
}

fn weibull_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let z = t / scale;
    (shape / scale) * z.powf(shape - 1.0) * (-z.powf(shape)).exp()
}

fn lognormal_surv(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    1.0 - normal_cdf((t.ln() - mu) / sigma)
}

fn lognormal_pdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let z = (t.ln() - mu) / sigma;
    normal_pdf(z) / (t * sigma)
}

fn loglogistic_surv(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + (t / scale).powf(shape))
}

fn loglogistic_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let z = (t / scale).powf(shape);
    (shape / scale) * (t / scale).powf(shape - 1.0) / (1.0 + z).powi(2)
}

fn exponential_surv(t: f64, rate: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    (-rate * t).exp()
}

fn exponential_pdf(t: f64, rate: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    rate * (-rate * t).exp()
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MixtureCureConfig {
    #[pyo3(get, set)]
    pub distribution: CureDistribution,
    #[pyo3(get, set)]
    pub link: LinkFunction,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub em_max_iter: usize,
}

#[pymethods]
impl MixtureCureConfig {
    #[new]
    #[pyo3(signature = (distribution=CureDistribution::Weibull, link=LinkFunction::Logit, max_iter=100, tol=1e-6, em_max_iter=500))]
    pub fn new(
        distribution: CureDistribution,
        link: LinkFunction,
        max_iter: usize,
        tol: f64,
        em_max_iter: usize,
    ) -> Self {
        MixtureCureConfig {
            distribution,
            link,
            max_iter,
            tol,
            em_max_iter,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MixtureCureResult {
    #[pyo3(get)]
    pub cure_coef: Vec<f64>,
    #[pyo3(get)]
    pub survival_coef: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub cure_fraction: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub cure_prob: Vec<f64>,
}

fn compute_surv_density(t: f64, scale: f64, shape: f64, dist: &CureDistribution) -> (f64, f64) {
    match dist {
        CureDistribution::Weibull => (weibull_surv(t, scale, shape), weibull_pdf(t, scale, shape)),
        CureDistribution::LogNormal => (
            lognormal_surv(t, scale, shape),
            lognormal_pdf(t, scale, shape),
        ),
        CureDistribution::LogLogistic => (
            loglogistic_surv(t, scale, shape),
            loglogistic_pdf(t, scale, shape),
        ),
        CureDistribution::Exponential => (exponential_surv(t, scale), exponential_pdf(t, scale)),
        CureDistribution::Gamma => (weibull_surv(t, scale, shape), weibull_pdf(t, scale, shape)),
    }
}
