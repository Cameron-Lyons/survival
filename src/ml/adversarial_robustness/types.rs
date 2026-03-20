
#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int, from_py_object)]

pub enum AttackType {
    FGSM,
    PGD,
    CarliniWagner,
    DeepFool,
    BoundaryAttack,
}

#[pymethods]
impl AttackType {
    fn __repr__(&self) -> String {
        match self {
            AttackType::FGSM => "AttackType.FGSM".to_string(),
            AttackType::PGD => "AttackType.PGD".to_string(),
            AttackType::CarliniWagner => "AttackType.CarliniWagner".to_string(),
            AttackType::DeepFool => "AttackType.DeepFool".to_string(),
            AttackType::BoundaryAttack => "AttackType.BoundaryAttack".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int, from_py_object)]
pub enum DefenseType {
    AdversarialTraining,
    InputPreprocessing,
    CertifiedDefense,
    Ensembling,
    Randomization,
}

#[pymethods]
impl DefenseType {
    fn __repr__(&self) -> String {
        match self {
            DefenseType::AdversarialTraining => "DefenseType.AdversarialTraining".to_string(),
            DefenseType::InputPreprocessing => "DefenseType.InputPreprocessing".to_string(),
            DefenseType::CertifiedDefense => "DefenseType.CertifiedDefense".to_string(),
            DefenseType::Ensembling => "DefenseType.Ensembling".to_string(),
            DefenseType::Randomization => "DefenseType.Randomization".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AdversarialAttackConfig {
    #[pyo3(get, set)]
    pub attack_type: AttackType,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub n_iterations: usize,
    #[pyo3(get, set)]
    pub step_size: f64,
    #[pyo3(get, set)]
    pub targeted: bool,
    #[pyo3(get, set)]
    pub clip_min: f64,
    #[pyo3(get, set)]
    pub clip_max: f64,
}

#[pymethods]
impl AdversarialAttackConfig {
    #[new]
    #[pyo3(signature = (attack_type=AttackType::FGSM, epsilon=0.1, n_iterations=10, step_size=0.01, targeted=false, clip_min=f64::NEG_INFINITY, clip_max=f64::INFINITY))]
    pub fn new(
        attack_type: AttackType,
        epsilon: f64,
        n_iterations: usize,
        step_size: f64,
        targeted: bool,
        clip_min: f64,
        clip_max: f64,
    ) -> Self {
        Self {
            attack_type,
            epsilon,
            n_iterations,
            step_size,
            targeted,
            clip_min,
            clip_max,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AdversarialDefenseConfig {
    #[pyo3(get, set)]
    pub defense_type: DefenseType,
    #[pyo3(get, set)]
    pub adversarial_ratio: f64,
    #[pyo3(get, set)]
    pub n_ensemble: usize,
    #[pyo3(get, set)]
    pub noise_scale: f64,
    #[pyo3(get, set)]
    pub certified_radius: f64,
}

#[pymethods]
impl AdversarialDefenseConfig {
    #[new]
    #[pyo3(signature = (defense_type=DefenseType::AdversarialTraining, adversarial_ratio=0.5, n_ensemble=5, noise_scale=0.1, certified_radius=0.1))]
    pub fn new(
        defense_type: DefenseType,
        adversarial_ratio: f64,
        n_ensemble: usize,
        noise_scale: f64,
        certified_radius: f64,
    ) -> Self {
        Self {
            defense_type,
            adversarial_ratio,
            n_ensemble,
            noise_scale,
            certified_radius,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AdversarialExample {
    #[pyo3(get)]
    pub original: Vec<f64>,
    #[pyo3(get)]
    pub perturbed: Vec<f64>,
    #[pyo3(get)]
    pub perturbation: Vec<f64>,
    #[pyo3(get)]
    pub original_prediction: f64,
    #[pyo3(get)]
    pub adversarial_prediction: f64,
    #[pyo3(get)]
    pub perturbation_norm: f64,
    #[pyo3(get)]
    pub success: bool,
}

#[pymethods]
impl AdversarialExample {
    fn __repr__(&self) -> String {
        format!(
            "AdversarialExample(success={}, norm={:.4}, pred_change={:.4})",
            self.success,
            self.perturbation_norm,
            (self.adversarial_prediction - self.original_prediction).abs()
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AdversarialAttackResult {
    #[pyo3(get)]
    pub adversarial_examples: Vec<AdversarialExample>,
    #[pyo3(get)]
    pub success_rate: f64,
    #[pyo3(get)]
    pub mean_perturbation_norm: f64,
    #[pyo3(get)]
    pub mean_prediction_change: f64,
    #[pyo3(get)]
    pub attack_type: AttackType,
}

#[pymethods]
impl AdversarialAttackResult {
    fn __repr__(&self) -> String {
        format!(
            "AdversarialAttackResult(success_rate={:.2}%, mean_norm={:.4})",
            self.success_rate * 100.0,
            self.mean_perturbation_norm
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RobustSurvivalModel {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub robust_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub certified_radius: f64,
    #[pyo3(get)]
    pub empirical_robustness: f64,
    #[pyo3(get)]
    pub defense_type: DefenseType,
    #[pyo3(get)]
    pub training_loss: f64,
    #[pyo3(get)]
    pub adversarial_loss: f64,
}

#[pymethods]
impl RobustSurvivalModel {
    fn __repr__(&self) -> String {
        format!(
            "RobustSurvivalModel(certified_radius={:.4}, empirical_robustness={:.4})",
            self.certified_radius, self.empirical_robustness
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RobustnessEvaluation {
    #[pyo3(get)]
    pub clean_accuracy: f64,
    #[pyo3(get)]
    pub robust_accuracy: f64,
    #[pyo3(get)]
    pub accuracy_drop: f64,
    #[pyo3(get)]
    pub certified_accuracy: f64,
    #[pyo3(get)]
    pub attack_success_rates: Vec<f64>,
    #[pyo3(get)]
    pub epsilon_values: Vec<f64>,
}

#[pymethods]
impl RobustnessEvaluation {
    fn __repr__(&self) -> String {
        format!(
            "RobustnessEvaluation(clean={:.2}%, robust={:.2}%, drop={:.2}%)",
            self.clean_accuracy * 100.0,
            self.robust_accuracy * 100.0,
            self.accuracy_drop * 100.0
        )
    }
}

