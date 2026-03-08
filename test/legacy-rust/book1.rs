use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use survival::prelude::*;

const TOL: f64 = 1e-6;

#[test]
fn test_cox_model() -> Result<(), Box<dyn std::error::Error>> {
    let time = array![9.0, 3.0, 1.0, 1.0, 6.0, 6.0, 8.0];
    let status = vec![Some(1), None, Some(1), Some(0), Some(1), Some(1), Some(0)];
    let x = array![0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0];

    let surv = Surv::new(time, status)
        .with_missing(MissingData::Exclude)
        .with_contrasts(Contrasts::Treatment);

    let mut fit = CoxPH::new()
        .method(TieMethod::Breslow)
        .max_iter(0)
        .fit(&surv, &x)?;

    let truth = byhand1(0.0, 0.0);
    assert_abs_diff_eq!(truth.loglik, fit.loglik()[0], epsilon = TOL);
    assert_abs_diff_eq!(1.0 / truth.imat, fit.variance()[(0, 0)], epsilon = TOL);

    let expected_residuals = Array1::from(vec![
        truth.mart[0],
        truth.mart[1],
        truth.mart[2],
        truth.mart[3],
        truth.mart[4],
        truth.mart[5],
    ]);
    assert_abs_diff_eq!(fit.residuals(), &expected_residuals, epsilon = TOL);

    fit = CoxPH::new()
        .method(TieMethod::Breslow)
        .max_iter(1)
        .fit(&surv, &x)?;
    assert_abs_diff_eq!(fit.coefficients()[0], 8.0 / 5.0, epsilon = TOL);

    fit = CoxPH::new()
        .method(TieMethod::Breslow)
        .tolerance(1e-8)
        .fit(&surv, &x)?;
    let beta = fit.coefficients()[0];
    let expected_beta = (1.5 + (33.0).sqrt() / 2.0).ln();
    assert_abs_diff_eq!(beta, expected_beta, epsilon = TOL);

    let truth_full = byhand1(beta, 0.0);
    assert_abs_diff_eq!(truth_full.loglik, fit.loglik()[1], epsilon = TOL);
    assert_abs_diff_eq!(1.0 / truth_full.imat, fit.variance()[(0, 0)], epsilon = TOL);

    let sfit = fit.survival_curve(&array![0.0]);
    assert_abs_diff_eq!(
        -sfit.surv().mapv(|v| v.ln()),
        truth_full.haz.accumulate(),
        epsilon = TOL
    );

    Ok(())
}

fn byhand1(beta: f64, newx: f64) -> ModelTruth {
    let r = beta.exp();
    let loglik = 2.0 * beta - (3.0 * r + 3.0).ln() - 2.0 * (r + 3.0).ln();

    ModelTruth {
        loglik,
        imat: 0.36,                                 
        mart: array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
        haz: array![0.1, 0.2, 0.0, 0.3],            
    }
}

struct ModelTruth {
    loglik: f64,
    imat: f64,
    mart: Array1<f64>,
    haz: Array1<f64>,
    
}

trait SurvivalAnalysis {
    fn residuals(&self) -> Array1<f64>;
    fn survival_curve(&self, x: &Array1<f64>) -> SurvivalCurve;
}

struct SurvivalCurve {
    surv: Array1<f64>,
    cumhaz: Array1<f64>,
    std_err: Array1<f64>,
}

impl SurvivalCurve {
    fn surv(&self) -> &Array1<f64> {
        &self.surv
    }
}
