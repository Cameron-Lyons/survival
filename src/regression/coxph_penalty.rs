use pyo3::prelude::*;

use super::coxph::CoxPHFit;
use super::coxph_wtest_module::coxph_wtest_core;

/// Uncertainty and effective degrees of freedom for a diagonal-penalized Cox fit.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CoxPenaltyDiagnostics {
    /// Penalty curvature in the original coefficient units.
    #[pyo3(get)]
    pub penalty_diagonal: Vec<f64>,
    /// One half of the fitted quadratic penalty.
    #[pyo3(get)]
    pub penalty: f64,
    /// Sampling covariance V H V, distinct from the inverse penalized information V.
    #[pyo3(get)]
    pub variance2: Vec<Vec<f64>>,
    /// Effective degrees of freedom in formula term order.
    #[pyo3(get)]
    pub term_df: Vec<f64>,
}

pub(crate) fn validate_penalty(
    diagonal: &[f64],
    term_groups: &[Vec<usize>],
    nvar: usize,
) -> PyResult<()> {
    if diagonal.len() != nvar {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "penalty has {} values but covariates has {} columns",
            diagonal.len(),
            nvar
        )));
    }
    if diagonal
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "penalty values must be finite and non-negative",
        ));
    }
    let mut seen = vec![false; nvar];
    for group in term_groups {
        if group.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "term_groups must not contain empty groups",
            ));
        }
        for &column in group {
            if column >= nvar || seen[column] {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "term_groups must contain each covariate column exactly once",
                ));
            }
            seen[column] = true;
        }
    }
    if seen.iter().any(|value| !value) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "term_groups must contain each covariate column exactly once",
        ));
    }
    Ok(())
}

impl CoxPenaltyDiagnostics {
    pub(crate) fn from_fit(
        fit: &CoxPHFit,
        diagonal: Vec<f64>,
        term_groups: &[Vec<usize>],
        toler: f64,
    ) -> PyResult<Self> {
        let variance = &fit.information_matrix;
        let nvar = diagonal.len();
        let mut variance2 = variance.clone();
        let active_penalties: Vec<(usize, f64)> = diagonal
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, penalty)| *penalty != 0.0)
            .collect();
        // V H V = V - V P V; diagonal P avoids an information inversion.
        for row in 0..nvar {
            for col in 0..=row {
                let correction: f64 = active_penalties
                    .iter()
                    .map(|&(k, penalty)| variance[row][k] * penalty * variance[k][col])
                    .sum();
                let value = variance[row][col] - correction;
                variance2[row][col] = value;
                variance2[col][row] = value;
            }
        }
        let mut term_df = Vec::with_capacity(term_groups.len());
        if term_groups.len() == 1 {
            // The optimizer's inverse has an exactly zero diagonal at aliased
            // pivots. trace(H V) = rank(H + P) - trace(P V), including aliases.
            // A second factorization of V would depend on the covariate units.
            let rank = (0..nvar).filter(|&i| variance[i][i] > 0.0).count();
            let shrinkage: f64 = diagonal
                .iter()
                .enumerate()
                .map(|(i, &penalty)| penalty * variance[i][i])
                .sum();
            term_df.push(rank as f64 - shrinkage);
        }
        for group in term_groups {
            if term_groups.len() == 1 {
                break;
            }
            if group.len() == 1 {
                let column = group[0];
                // R's scalar Wald solve is division, including 0/0 for an
                // entirely aliased singleton term. Matrix blocks use the
                // rank-aware solve below.
                term_df.push(variance2[column][column] / variance[column][column]);
                continue;
            }
            // Scale both sides of the block solve to correlation units. This
            // is a similarity transform of V_gg^-1 V2_gg, so its trace stays
            // unchanged without treating differing measurement units as rank loss.
            let scales: Vec<f64> = group
                .iter()
                .map(|&i| {
                    if variance[i][i] > 0.0 {
                        variance[i][i].sqrt()
                    } else {
                        1.0
                    }
                })
                .collect();
            let block: Vec<Vec<f64>> = group
                .iter()
                .enumerate()
                .map(|(i, &row)| {
                    group
                        .iter()
                        .enumerate()
                        .map(|(j, &col)| variance[row][col] / scales[i] / scales[j])
                        .collect()
                })
                .collect();
            let rhs_columns: Vec<Vec<f64>> = group
                .iter()
                .enumerate()
                .map(|(j, &col)| {
                    group
                        .iter()
                        .enumerate()
                        .map(|(i, &row)| variance2[row][col] / scales[i] / scales[j])
                        .collect()
                })
                .collect();
            let (_, _, solved) = coxph_wtest_core(&block, &rhs_columns, toler)?;
            term_df.push(solved.iter().enumerate().map(|(i, row)| row[i]).sum());
        }
        let penalty = diagonal
            .iter()
            .zip(&fit.coefficients[0])
            .map(|(&curvature, &coefficient)| 0.5 * curvature * coefficient * coefficient)
            .sum();
        Ok(Self {
            penalty_diagonal: diagonal,
            penalty,
            variance2,
            term_df,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::{coxph_fit, coxph_penalized_fit};

    fn fixture() -> (Vec<f64>, Vec<i32>, Vec<Vec<f64>>, Vec<f64>) {
        let x = [0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.3, 0.9, 0.6, 1.2];
        let z = [0.3, 1.2, 0.4, 0.8, 1.1, 0.6, 0.2, 0.7, 1.4, 0.9];
        (
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            z.iter().zip(x).map(|(&z, x)| vec![z, x]).collect(),
            vec![1.0, 2.0, 1.0, 0.5, 1.5, 1.0, 2.0, 1.0, 1.0, 1.0],
        )
    }

    fn fit(
        penalty: Vec<f64>,
        groups: Vec<Vec<usize>>,
        method: &str,
        initial: Option<Vec<f64>>,
        max_iter: usize,
    ) -> (CoxPHFit, CoxPenaltyDiagnostics) {
        let (time, status, covariates, weights) = fixture();
        coxph_penalized_fit(
            time,
            status,
            covariates,
            penalty,
            groups,
            None,
            Some(weights),
            None,
            initial,
            Some(max_iter),
            Some(1e-12),
            Some(1e-12),
            Some(method),
            None,
            None,
        )
        .expect("penalized fixture should fit")
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn fixed_ridge_matches_r_weighted_coefficients_uncertainty_and_term_df() {
        // survival 3.8-11: Surv(time,event) ~ z + ridge(x,theta=2), weights=w.
        let diagonal = vec![0.0, 2.0 * 0.115_666_666_666_666_67];
        let (fit, diagnostics) = fit(diagonal.clone(), vec![vec![0], vec![1]], "efron", None, 40);
        let expected_beta = [-0.231_167_528_265_580, -1.544_516_336_277_5];
        let expected_var = [
            [0.698_882_460_593_138, -0.256_432_417_306_670],
            [-0.256_432_417_306_670, 1.358_289_225_623_05],
        ];
        let expected_var2 = [
            [0.683_670_539_345_090, -0.175_856_845_195_830],
            [-0.175_856_845_195_830, 0.931_490_880_093_751],
        ];
        for i in 0..2 {
            close(fit.coefficients[0][i], expected_beta[i], 2e-8);
            close(
                fit.score_vector[i],
                diagonal[i] * fit.coefficients[0][i],
                1e-9,
            );
            for j in 0..2 {
                close(fit.information_matrix[i][j], expected_var[i][j], 2e-8);
                close(diagnostics.variance2[i][j], expected_var2[i][j], 2e-8);
            }
        }
        close(diagnostics.term_df[0], 0.978_233_934_737_555, 2e-8);
        close(diagnostics.term_df[1], 0.685_782_425_805_867, 2e-8);
        close(diagnostics.penalty, 0.275_926_385_806_914, 2e-8);
        close(fit.log_likelihood[0], -16.798_529_901_245, 2e-8);
        close(fit.log_likelihood[1], -15.334_569_950_845_5, 2e-8);
        close(fit.means[0], 0.76, 1e-12);
        close(fit.means[1], 0.57, 1e-12);
    }

    #[test]
    fn diagonal_penalties_jointly_solve_the_score_for_every_column() {
        for method in ["efron", "breslow"] {
            for penalty in [vec![0.0, 0.0], vec![0.0, 2.0], vec![0.7, 200.0]] {
                let (fitted, _) = fit(penalty.clone(), vec![vec![0, 1]], method, None, 40);
                let (time, status, covariates, weights) = fixture();
                let evaluated = coxph_fit(
                    time,
                    status,
                    covariates,
                    None,
                    Some(weights),
                    None,
                    Some(fitted.coefficients[0].clone()),
                    Some(0),
                    None,
                    None,
                    Some(method),
                    None,
                    None,
                )
                .unwrap();
                for (i, &curvature) in penalty.iter().enumerate() {
                    close(
                        evaluated.score_vector[i],
                        curvature * fitted.coefficients[0][i],
                        1e-8,
                    );
                    close(fitted.score_vector[i], evaluated.score_vector[i], 1e-12);
                }
                close(fitted.log_likelihood[1], evaluated.log_likelihood[1], 1e-12);
            }
        }
    }

    #[test]
    fn zero_penalty_recovers_unpenalized_fit_and_full_rank_df() {
        let (time, status, covariates, weights) = fixture();
        let ordinary = coxph_fit(
            time,
            status,
            covariates,
            None,
            Some(weights),
            None,
            None,
            Some(40),
            Some(1e-12),
            Some(1e-12),
            Some("efron"),
            None,
            None,
        )
        .unwrap();
        let (penalized, diagnostics) = fit(vec![0.0, 0.0], vec![vec![0, 1]], "efron", None, 40);
        assert_eq!(penalized.coefficients, ordinary.coefficients);
        assert_eq!(penalized.information_matrix, ordinary.information_matrix);
        assert_eq!(penalized.log_likelihood, ordinary.log_likelihood);
        assert_eq!(diagnostics.variance2, ordinary.information_matrix);
        close(diagnostics.term_df[0], 2.0, 1e-12);
        assert_eq!(diagnostics.penalty, 0.0);
    }

    #[test]
    fn term_df_uses_group_blocks_instead_of_global_trace() {
        let penalty = vec![0.0, 2.0 * 0.115_666_666_666_666_67];
        let (_, separate) = fit(penalty.clone(), vec![vec![0], vec![1]], "efron", None, 40);
        let (_, joint) = fit(penalty, vec![vec![0, 1]], "efron", None, 40);
        close(joint.term_df[0], 1.685_782_425_805_867, 2e-8);
        assert!((separate.term_df.iter().sum::<f64>() - joint.term_df[0]).abs() > 0.02);
    }

    #[test]
    fn exact_labeled_penalty_uses_the_weighted_breslow_route() {
        let (breslow, b_diagnostics) =
            fit(vec![0.0, 2.0], vec![vec![0], vec![1]], "breslow", None, 40);
        let (exact, e_diagnostics) = fit(vec![0.0, 2.0], vec![vec![0], vec![1]], "exact", None, 40);
        assert_eq!(exact.method, "breslow");
        assert_eq!(exact.coefficients, breslow.coefficients);
        assert_eq!(exact.information_matrix, breslow.information_matrix);
        assert_eq!(e_diagnostics.variance2, b_diagnostics.variance2);
    }

    #[test]
    fn zero_iteration_and_nonconverged_fit_report_final_penalized_curvature() {
        let penalty = vec![0.3, 2.0];
        for max_iter in [0, 1] {
            let (fitted, diagnostics) = fit(
                penalty.clone(),
                vec![vec![0, 1]],
                "efron",
                Some(vec![0.4, -0.2]),
                max_iter,
            );
            let (evaluated, evaluated_diagnostics) = fit(
                penalty.clone(),
                vec![vec![0, 1]],
                "efron",
                Some(fitted.coefficients[0].clone()),
                0,
            );
            assert_eq!(fitted.information_matrix, evaluated.information_matrix);
            assert_eq!(fitted.score_vector, evaluated.score_vector);
            assert_eq!(fitted.log_likelihood[1], evaluated.log_likelihood[1]);
            assert_eq!(diagnostics.variance2, evaluated_diagnostics.variance2);
            assert_eq!(diagnostics.penalty, evaluated_diagnostics.penalty);
        }
    }

    #[test]
    fn validates_diagonal_and_complete_disjoint_term_groups() {
        assert!(validate_penalty(&[1.0], &[vec![0]], 2).is_err());
        assert!(validate_penalty(&[-1.0], &[vec![0]], 1).is_err());
        assert!(validate_penalty(&[f64::INFINITY], &[vec![0]], 1).is_err());
        assert!(validate_penalty(&[f64::NAN], &[vec![0]], 1).is_err());
        for groups in [vec![vec![]], vec![vec![1]], vec![vec![0], vec![0]], vec![]] {
            assert!(validate_penalty(&[1.0], &groups, 1).is_err());
        }
        assert!(validate_penalty(&[], &[], 0).is_ok());
        assert!(validate_penalty(&[0.0, 1.0], &[vec![1, 0]], 2).is_ok());
    }

    #[test]
    fn constant_columns_preserve_alias_variance_and_term_df() {
        for curvature in [0.0, 2.0] {
            let (time, status, mut covariates, weights) = fixture();
            for row in &mut covariates {
                row[0] = 1.0;
            }
            let (fitted, diagnostics) = coxph_penalized_fit(
                time,
                status,
                covariates,
                vec![curvature, 2.0],
                vec![vec![0], vec![1]],
                None,
                Some(weights),
                None,
                None,
                None,
                None,
                None,
                Some("efron"),
                None,
                Some(vec![-1.0, 0.0, 1.0]),
            )
            .unwrap();
            close(fitted.coefficients[0][0], 0.0, 1e-12);
            close(
                fitted.information_matrix[0][0],
                if curvature == 0.0 {
                    0.0
                } else {
                    1.0 / curvature
                },
                1e-12,
            );
            close(diagnostics.variance2[0][0], 0.0, 1e-12);
            if curvature == 0.0 {
                assert!(diagnostics.term_df[0].is_nan());
            } else {
                close(diagnostics.term_df[0], 0.0, 1e-12);
            }
            assert_eq!(fitted.means[0], 0.0);
        }
    }

    #[test]
    fn aliased_grouped_and_sole_terms_keep_zero_effective_df() {
        for (nvar, groups) in [
            (1, vec![vec![0]]),
            (2, vec![vec![0, 1]]),
            (3, vec![vec![0], vec![1, 2]]),
        ] {
            let (time, status, covariates, weights) = fixture();
            let covariates = covariates
                .into_iter()
                .map(|row| {
                    let mut constant = vec![1.0; nvar];
                    if nvar == 3 {
                        constant[0] = row[0];
                    }
                    constant
                })
                .collect();
            let (_, diagnostics) = coxph_penalized_fit(
                time,
                status,
                covariates,
                vec![0.0; nvar],
                groups,
                None,
                Some(weights),
                None,
                None,
                None,
                None,
                None,
                Some("efron"),
                None,
                None,
            )
            .unwrap();
            close(*diagnostics.term_df.last().unwrap(), 0.0, 1e-12);
        }
    }

    #[test]
    fn grouped_effective_df_is_invariant_to_covariate_measurement_units() {
        for separate_term in [false, true] {
            let mut dfs = Vec::new();
            for scale in [1.0, 1e-8, 1e8] {
                let (time, status, mut covariates, weights) = fixture();
                for (i, row) in covariates.iter_mut().enumerate() {
                    row[0] *= scale;
                    if separate_term {
                        row.push((i as f64 * 0.7).sin());
                    }
                }
                let mut penalty = vec![0.5 * scale * scale, 0.3];
                let mut groups = vec![vec![0, 1]];
                if separate_term {
                    penalty.push(0.0);
                    groups.push(vec![2]);
                }
                let (_, diagnostics) = coxph_penalized_fit(
                    time,
                    status,
                    covariates,
                    penalty,
                    groups,
                    None,
                    Some(weights),
                    None,
                    None,
                    Some(40),
                    Some(1e-12),
                    Some(1e-12),
                    Some("efron"),
                    None,
                    None,
                )
                .unwrap();
                dfs.push(diagnostics.term_df);
            }
            for row in &dfs[1..] {
                for (&actual, &expected) in row.iter().zip(&dfs[0]) {
                    close(actual, expected, 1e-10);
                }
            }
        }
    }

    #[test]
    fn counting_breslow_matches_r_with_offset_weights_and_strata() {
        // survival 3.8-11, fixed theta=2, scale=FALSE; z remains unpenalized.
        let covariates = vec![
            vec![0.5, -1.2],
            vec![-1.0, 0.4],
            vec![0.3, 1.1],
            vec![1.2, -0.3],
            vec![-0.7, 0.8],
            vec![0.9, 1.7],
            vec![0.1, -0.9],
            vec![-1.3, 0.2],
        ];
        let (fitted, diagnostics) = coxph_penalized_fit(
            vec![2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0],
            vec![1, 1, 0, 1, 0, 1, 1, 0],
            covariates,
            vec![0.0, 2.0],
            vec![vec![0], vec![1]],
            Some(vec![0, 0, 0, 0, 0, 1, 1, 1]),
            Some(vec![1.0, 1.5, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3]),
            Some(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.2, -0.15, 0.0]),
            None,
            Some(40),
            Some(1e-12),
            Some(1e-12),
            Some("breslow"),
            Some(vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]),
            None,
        )
        .unwrap();
        close(fitted.coefficients[0][0], 0.194_347_507_934_330, 2e-8);
        close(fitted.coefficients[0][1], 0.009_894_882_091_005_14, 2e-8);
        close(fitted.information_matrix[0][0], 0.275_204_671_017_473, 2e-8);
        close(
            fitted.information_matrix[1][0],
            0.038_473_746_168_445_7,
            2e-8,
        );
        close(fitted.information_matrix[1][1], 0.195_682_824_932_415, 2e-8);
        close(diagnostics.variance2[0][0], 0.272_244_212_729_005, 2e-8);
        close(diagnostics.variance2[1][0], 0.023_416_443_496_497_4, 2e-8);
        close(diagnostics.variance2[1][1], 0.119_099_288_985_355, 2e-8);
        close(diagnostics.term_df[0], 0.989_242_703_339_581, 2e-8);
        close(diagnostics.term_df[1], 0.608_634_350_135_170, 2e-8);
        close(fitted.log_likelihood[0], -6.657_129_981_715_62, 2e-8);
        close(fitted.log_likelihood[1], -6.585_136_411_887_28, 2e-8);
        close(fitted.means[0], 0.0, 1e-12);
        close(fitted.means[1], 0.225, 1e-12);
        close(fitted.score_vector[0], 0.0, 1e-9);
        close(
            fitted.score_vector[1],
            2.0 * fitted.coefficients[0][1],
            1e-9,
        );
    }
}
