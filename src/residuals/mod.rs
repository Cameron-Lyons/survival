#[path = "agmart.rs"]
pub(crate) mod agmart_module;
#[path = "coxmart.rs"]
pub(crate) mod coxmart_module;
pub(crate) mod diagnostics;
pub(crate) mod survfit_resid;
pub(crate) mod survreg_resid;

// Public facade exports
pub use agmart_module::agmart;
pub use coxmart_module::coxmart;
pub use diagnostics::{
    DfbetaResult, GofTestResult, LeverageResult, ModelInfluenceResult, OutlierDetectionResult,
    SchoenfeldSmoothResult, dfbeta_cox, goodness_of_fit_cox, leverage_cox, model_influence_cox,
    outlier_detection_cox, smooth_schoenfeld,
};
pub use survfit_resid::{SurvfitResiduals, residuals_survfit};
pub use survreg_resid::{SurvregResiduals, dfbeta_survreg, residuals_survreg};
