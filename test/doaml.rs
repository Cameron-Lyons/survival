use approx::assert_relative_eq;
use ndarray::{Array1, Array2};
use statrs::distribution::ChiSquared;

#[derive(Debug, Clone)]
struct SurvivalData {
    time: Array1<f64>,
    status: Array1<i32>,
    covariates: Array2<f64>,
    weights: Option<Array1<f64>>,
    strata: Option<Array1<i32>>,
}

struct CoxModel {
    coefficients: Array1<f64>,
    log_likelihood: f64,
    variance: Array2<f64>,
    residuals: CoxResiduals,
}

struct CoxResiduals {
    martingale: Array1<f64>,
    score: Array1<f64>,
    schoenfeld: Array2<f64>,
}

struct KaplanMeier {
    time_points: Array1<f64>,
    survival: Array1<f64>,
    std_err: Array1<f64>,
}

impl SurvivalData {
    fn new(time: Array1<f64>, status: Array1<i32>, covariates: Array2<f64>) -> Self {
        SurvivalData {
            time,
            status,
            covariates,
            weights: None,
            strata: None,
        }
    }

    fn coxph(&self, method: TieMethod) -> CoxModel {
        unimplemented!()
    }

    fn survfit(&self) -> KaplanMeier {
        unimplemented!()
    }

    fn survdiff(&self) -> LogRankTest {
        unimplemented!()
    }
}

enum TieMethod {
    Breslow,
    Efron,
}

struct LogRankTest {
    chi_squared: f64,
    df: usize,
    p_value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn create_aml_data() -> SurvivalData {
        let time = arr1(&[
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0, 5.0, 5.0, 8.0, 8.0,
            12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
        ]);
        let status = arr1(&[
            1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        ]);
        let covariates = Array2::eye(23); // Dummy covariates

        SurvivalData::new(time, status, covariates)
    }

    #[test]
    fn test_coxph_equality() {
        let aml = create_aml_data();

        let fit1 = aml.coxph(TieMethod::Breslow);
        let fit2 = aml.coxph(TieMethod::Breslow); // Should match

        assert_relative_eq!(fit1.log_likelihood, fit2.log_likelihood, epsilon = 1e-6);
        assert_relative_eq!(
            fit1.coefficients.mean().unwrap(),
            fit2.coefficients.mean().unwrap(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_km_variance() {
        let aml = create_aml_data();

        let fit = aml.survfit();

        let weighted_data = SurvivalData {
            weights: Some(Array1::from_elem(23, 2.0)),
            ..aml.clone()
        };
        let weighted_fit = weighted_data.survfit();

        for (std1, std2) in fit.std_err.iter().zip(weighted_fit.std_err.iter()) {
            assert_relative_eq!(std1.powi(2), 2.0 * std2.powi(2), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_log_rank() {
        let aml = create_aml_data();
        let result = aml.survdiff();

        let dist = ChiSquared::new(result.df as f64).unwrap();
        let p = 1.0 - dist.cdf(result.chi_squared);

        assert_relative_eq!(result.p_value, p, epsilon = 1e-6);
        assert!(result.chi_squared > 5.0); // Expect significant result
    }
}

fn main() {
    let aml_data = create_aml_data();

    let cox_model = aml_data.coxph(TieMethod::Breslow);
    println!("Cox Model Coefficients: {:?}", cox_model.coefficients);

    let km = aml_data.survfit();
    println!("KM Survival Estimates: {:?}", km.survival);

    let lr = aml_data.survdiff();
    println!(
        "Log-Rank Test: χ²({}) = {:.2}, p = {:.4}",
        lr.df, lr.chi_squared, lr.p_value
    );
}
