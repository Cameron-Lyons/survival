use approx::assert_relative_eq;
use ndarray::{Array1, Array2};
use statrs::distribution::ChiSquared;
use survival::regression::coxfit6::{CoxFit, CoxError, Method as CoxMethod};
use survival::surv_analysis::survfitkm::{KaplanMeierConfig, compute_survfitkm};
use survival::surv_analysis::survdiff2::{compute_survdiff, SurvDiffInput, SurvDiffOutput, SurvDiffParams};
use survival::residuals::coxmart::compute_coxmart;

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
        let n = self.time.len();
        let nvar = self.covariates.ncols();
        
        let strata = self.strata.clone().unwrap_or_else(|| Array1::zeros(n));
        let offset = Array1::zeros(n);
        let weights = self.weights.clone().unwrap_or_else(|| Array1::from_elem(n, 1.0));
        
        let cox_method = match method {
            TieMethod::Breslow => CoxMethod::Breslow,
            TieMethod::Efron => CoxMethod::Efron,
        };
        
        let mut cox_fit = CoxFit::new(
            self.time.clone(),
            self.status.clone(),
            self.covariates.clone(),
            strata,
            offset,
            weights,
            cox_method,
            20,
            1e-5,
            1e-9,
            vec![true; nvar],
            vec![0.0; nvar],
        ).expect("Failed to create CoxFit");
        
        cox_fit.fit().expect("Failed to fit Cox model");
        
        let (beta, _means, _u, imat, loglik, _sctest, _flag, _iter) = cox_fit.results();
        
        let risk_scores: Vec<f64> = self.covariates
            .outer_iter()
            .map(|row| {
                row.iter()
                    .zip(beta.iter())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    .exp()
            })
            .collect();
        
        let mut martingale = vec![0.0; n];
        let status_vec: Vec<i32> = self.status.to_vec();
        let time_vec: Vec<f64> = self.time.to_vec();
        let mut strata_vec: Vec<i32> = vec![0; n];
        strata_vec[n - 1] = 1;
        let weights_vec: Vec<f64> = self.weights.as_ref()
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0; n]);
        
        use survival::residuals::coxmart::{SurvivalData, Weights};
        compute_coxmart(
            n,
            0,
            SurvivalData {
                time: &time_vec,
                status: &status_vec,
                strata: &mut strata_vec,
            },
            Weights {
                score: &risk_scores,
                wt: &weights_vec,
            },
            &mut martingale,
        );
        
        let score = martingale.clone();
        
        let schoenfeld = Array2::zeros((n, nvar));
        
        CoxModel {
            coefficients: Array1::from_vec(beta),
            log_likelihood: loglik[1],
            variance: imat,
            residuals: CoxResiduals {
                martingale: Array1::from_vec(martingale),
                score: Array1::from_vec(score),
                schoenfeld,
            },
        }
    }

    fn survfit(&self) -> KaplanMeier {
        let n = self.time.len();
        let time_vec: Vec<f64> = self.time.to_vec();
        let status_vec: Vec<f64> = self.status.mapv(|x| x as f64).to_vec();
        let weights_vec: Vec<f64> = self.weights.as_ref()
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0; n]);
        let position_vec: Vec<i32> = vec![0; n];
        
        let result = compute_survfitkm(
            &time_vec,
            &status_vec,
            &weights_vec,
            None,
            &position_vec,
            &KaplanMeierConfig::default(),
        );
        
        KaplanMeier {
            time_points: Array1::from_vec(result.time),
            survival: Array1::from_vec(result.estimate),
            std_err: Array1::from_vec(result.std_err),
        }
    }

    fn survdiff(&self) -> LogRankTest {
        let n = self.time.len();
        let time_vec: Vec<f64> = self.time.to_vec();
        let status_vec: Vec<i32> = self.status.to_vec();
        
        let median_time = {
            let mut sorted_times = self.time.to_vec();
            sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_times[n / 2]
        };
        let group_vec: Vec<i32> = self.time.iter()
            .map(|&t| if t <= median_time { 1 } else { 2 })
            .collect();
        
        let strata_vec: Vec<i32> = vec![0; n];
        
        let ngroup = 2;
        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];
        
        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };
        
        let input = SurvDiffInput {
            time: &time_vec,
            status: &status_vec,
            group: &group_vec,
            strata: &strata_vec,
        };
        
        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };
        
        compute_survdiff(params, input, &mut output);
        
        let mut chi_sq = 0.0;
        let mut df = 0;
        for i in 0..ngroup {
            let diff = obs[i] - exp[i];
            if exp[i] > 0.0 {
                chi_sq += diff * diff / exp[i];
                df += 1;
            }
        }
        df = (df - 1).max(0);
        
        let dist = ChiSquared::new(df as f64).unwrap();
        let p_value = 1.0 - dist.cdf(chi_sq);
        
        LogRankTest {
            chi_squared: chi_sq,
            df,
            p_value,
        }
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

fn create_aml_data() -> SurvivalData {
    use ndarray::arr1;
        let time = arr1(&[
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0, 5.0, 5.0, 8.0, 8.0,
            12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
        ]);
        let status = arr1(&[
            1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        ]);
        let covariates = Array2::eye(23); 

    SurvivalData::new(time, status, covariates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_coxph_equality() {
        let aml = create_aml_data();

        let fit1 = aml.coxph(TieMethod::Breslow);
        let fit2 = aml.coxph(TieMethod::Breslow); 

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
        assert!(result.chi_squared > 5.0); 
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
