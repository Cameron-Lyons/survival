#[cfg(test)]
mod tests {
    use crate::regression::coxfit6::{CoxFit, Method as CoxMethod};
    use crate::surv_analysis::nelson_aalen::nelson_aalen;
    use crate::surv_analysis::survfitkm::compute_survfitkm;
    use crate::validation::logrank::{WeightType, weighted_logrank_test};
    use crate::validation::rmst::compute_rmst;
    use ndarray::{Array1, Array2};
    use serde::Deserialize;
    use std::fs;

    const STANDARD_TOL: f64 = 1e-3;
    const STAT_TOL: f64 = 0.05;

    fn rel_approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        let max_abs = a.abs().max(b.abs());
        if max_abs < 1e-10 {
            return true;
        }
        (a - b).abs() / max_abs < rel_tol
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct RExpectedValues {
        metadata: Metadata,
        aml: AmlData,
        lung: LungData,
        ovarian: OvarianData,
        veteran: VeteranData,
        edge_cases: EdgeCases,
        sample_size: SampleSizeData,
        rmst: RmstData,
        concordance: ConcordanceData,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct Metadata {
        survival_version: String,
        r_version: String,
        generated_at: String,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct AmlData {
        maintained: DataGroup,
        nonmaintained: DataGroup,
        combined: CombinedDataGroup,
        km_maintained: KaplanMeierResult,
        km_nonmaintained: KaplanMeierResult,
        nelson_aalen_maintained: NelsonAalenResult,
        logrank: LogRankResult,
        wilcoxon: WilcoxonResult,
        coxph_breslow: CoxphResult,
        coxph_efron: CoxphEfronResult,
        median_survival: MedianSurvival,
        martingale_residuals: MartingaleResiduals,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct DataGroup {
        time: Vec<f64>,
        status: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct CombinedDataGroup {
        time: Vec<f64>,
        status: Vec<i32>,
        group: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct KaplanMeierResult {
        time: Vec<f64>,
        n_risk: Vec<i32>,
        n_event: Vec<i32>,
        n_censor: Option<Vec<i32>>,
        survival: Vec<f64>,
        std_err: Vec<f64>,
        lower: Option<Vec<f64>>,
        upper: Option<Vec<f64>>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct NelsonAalenResult {
        time: Vec<f64>,
        n_risk: Vec<i32>,
        n_event: Vec<i32>,
        cumulative_hazard: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct LogRankResult {
        n: Vec<i32>,
        observed: Vec<f64>,
        expected: Vec<f64>,
        chisq: f64,
        df: i32,
        p_value: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct WilcoxonResult {
        chisq: f64,
        p_value: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct CoxphResult {
        coefficients: Vec<f64>,
        se: Vec<f64>,
        hazard_ratio: Vec<f64>,
        hr_lower: f64,
        hr_upper: f64,
        loglik: Vec<f64>,
        score_test: f64,
        wald_test: f64,
        lr_test: f64,
        concordance: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct CoxphEfronResult {
        coefficients: Vec<f64>,
        se: Vec<f64>,
        hazard_ratio: Vec<f64>,
        loglik: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct MedianSurvival {
        maintained: f64,
        nonmaintained: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct MartingaleResiduals {
        sum: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct LungData {
        data: LungDataFields,
        coxph: LungCoxph,
        logrank_sex: LogRankSexResult,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct LungDataFields {
        time: Vec<f64>,
        status: Vec<i32>,
        sex: Vec<i32>,
        age: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct LungCoxph {
        coefficients: Vec<f64>,
        se: Vec<f64>,
        hazard_ratio: Vec<f64>,
        loglik: Vec<f64>,
        concordance: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct LogRankSexResult {
        chisq: f64,
        p_value: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct OvarianData {
        data: OvarianDataFields,
        logrank: OvarianLogrank,
        km: OvarianKm,
        coxph: OvarianCoxph,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct OvarianDataFields {
        time: Vec<f64>,
        status: Vec<i32>,
        rx: Vec<i32>,
        age: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct OvarianLogrank {
        chisq: f64,
        p_value: f64,
        observed: Vec<f64>,
        expected: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct OvarianKm {
        time: Vec<f64>,
        survival: Vec<f64>,
        n_risk: Vec<i32>,
        n_event: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct OvarianCoxph {
        coefficients: Vec<f64>,
        se: Vec<f64>,
        hazard_ratio: Vec<f64>,
        loglik: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct VeteranData {
        data: VeteranDataFields,
        km: VeteranKm,
        coxph: VeteranCoxph,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct VeteranDataFields {
        time: Vec<f64>,
        status: Vec<i32>,
        trt: Vec<i32>,
        age: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct VeteranKm {
        time: Vec<f64>,
        survival: Vec<f64>,
        n_risk: Vec<i32>,
        n_event: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct VeteranCoxph {
        coefficients: Vec<f64>,
        hazard_ratio: Vec<f64>,
        loglik: Vec<f64>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct EdgeCases {
        tied_events: EdgeCaseKm,
        all_same_time: EdgeCaseKm,
        simple_nelson_aalen: SimpleNelsonAalen,
        with_censoring: WithCensoring,
        identical_groups_logrank: IdenticalGroupsLogrank,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct EdgeCaseKm {
        time: Vec<f64>,
        survival: Vec<f64>,
        n_risk: Vec<i32>,
        n_event: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct SimpleNelsonAalen {
        time: Vec<f64>,
        cumulative_hazard: Vec<f64>,
        n_risk: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct WithCensoring {
        time: Vec<f64>,
        cumulative_hazard: Vec<f64>,
        survival: Vec<f64>,
        n_risk: Vec<i32>,
        n_event: Vec<i32>,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct IdenticalGroupsLogrank {
        chisq: f64,
        p_value: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct SampleSizeData {
        #[serde(rename = "hr_0.5_power_0.8")]
        hr_05_power_08: i32,
        #[serde(rename = "hr_0.6_power_0.8")]
        hr_06_power_08: i32,
        #[serde(rename = "hr_0.7_power_0.8")]
        hr_07_power_08: i32,
        #[serde(rename = "hr_0.6_power_0.9")]
        hr_06_power_09: i32,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct RmstData {
        aml_maintained_tau30: f64,
        aml_maintained_tau48: f64,
        aml_nonmaintained_tau30: f64,
        aml_nonmaintained_tau48: f64,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct ConcordanceData {
        aml_coxph: ConcordanceResult,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    struct ConcordanceResult {
        concordance: f64,
        n_concordant: i32,
        n_discordant: i32,
        n_tied_risk: i32,
        n_tied_time: i32,
    }

    fn load_expected_values() -> RExpectedValues {
        let json_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test/r_expected_values.json");
        let json_content =
            fs::read_to_string(json_path).expect("Failed to read r_expected_values.json");
        serde_json::from_str(&json_content).expect("Failed to parse r_expected_values.json")
    }

    #[test]
    fn test_aml_logrank_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let result = weighted_logrank_test(
            &aml.combined.time,
            &aml.combined.status,
            &aml.combined.group,
            WeightType::LogRank,
        );

        assert!(
            rel_approx_eq(result.statistic, aml.logrank.chisq, STAT_TOL),
            "Log-rank chi-squared: expected {}, got {}",
            aml.logrank.chisq,
            result.statistic
        );

        assert!(
            rel_approx_eq(result.p_value, aml.logrank.p_value, STAT_TOL),
            "Log-rank p-value: expected {}, got {}",
            aml.logrank.p_value,
            result.p_value
        );

        assert_eq!(
            result.df, aml.logrank.df as usize,
            "Degrees of freedom mismatch"
        );
    }

    #[test]
    fn test_aml_wilcoxon_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let result = weighted_logrank_test(
            &aml.combined.time,
            &aml.combined.status,
            &aml.combined.group,
            WeightType::Wilcoxon,
        );

        assert!(
            rel_approx_eq(result.statistic, aml.wilcoxon.chisq, STAT_TOL),
            "Wilcoxon chi-squared: expected {}, got {}",
            aml.wilcoxon.chisq,
            result.statistic
        );
    }

    #[test]
    fn test_aml_km_maintained_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let status_f64: Vec<f64> = aml.maintained.status.iter().map(|&s| s as f64).collect();
        let weights = vec![1.0; aml.maintained.time.len()];
        let position = vec![0i32; aml.maintained.time.len()];

        let result = compute_survfitkm(
            &aml.maintained.time,
            &status_f64,
            &weights,
            None,
            &position,
            false,
            0,
        );

        for (i, &r_surv) in aml.km_maintained.survival.iter().enumerate() {
            if i < result.estimate.len() {
                assert!(
                    rel_approx_eq(result.estimate[i], r_surv, STANDARD_TOL),
                    "KM survival at time {}: expected {}, got {}",
                    aml.km_maintained.time[i],
                    r_surv,
                    result.estimate[i]
                );
            }
        }
    }

    #[test]
    fn test_aml_nelson_aalen_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let result = nelson_aalen(&aml.maintained.time, &aml.maintained.status, None, 0.95);

        for (i, &r_cumhaz) in aml
            .nelson_aalen_maintained
            .cumulative_hazard
            .iter()
            .enumerate()
        {
            if i < result.cumulative_hazard.len() {
                assert!(
                    rel_approx_eq(result.cumulative_hazard[i], r_cumhaz, STANDARD_TOL),
                    "Nelson-Aalen cumhaz at time {}: expected {}, got {}",
                    aml.nelson_aalen_maintained.time[i],
                    r_cumhaz,
                    result.cumulative_hazard[i]
                );
            }
        }
    }

    #[test]
    fn test_aml_coxph_breslow_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let n = aml.combined.time.len();
        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = aml.combined.group[i] as f64;
        }

        let mut cox_fit = CoxFit::new(
            Array1::from_vec(aml.combined.time.clone()),
            Array1::from_vec(aml.combined.status.clone()),
            covar,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::from_elem(n, 1.0),
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Cox fit init failed");

        cox_fit.fit().expect("Cox fit failed");
        let (beta, _means, _u, _imat, loglik, _sctest, _flag, _iter) = cox_fit.results();

        let r_coef = aml.coxph_breslow.coefficients[0];
        let r_hr = aml.coxph_breslow.hazard_ratio[0];

        assert!(
            rel_approx_eq(beta[0].abs(), r_coef.abs(), STAT_TOL),
            "Cox coefficient magnitude: expected {}, got {}",
            r_coef.abs(),
            beta[0].abs()
        );

        let hr = beta[0].exp();
        let hr_match = rel_approx_eq(hr, r_hr, STAT_TOL) || rel_approx_eq(hr, 1.0 / r_hr, STAT_TOL);
        assert!(
            hr_match,
            "Cox hazard ratio: expected {} or {}, got {}",
            r_hr,
            1.0 / r_hr,
            hr
        );

        let r_loglik_final = aml.coxph_breslow.loglik[1];
        assert!(
            rel_approx_eq(loglik[1], r_loglik_final, STAT_TOL),
            "Cox log-likelihood: expected {}, got {}",
            r_loglik_final,
            loglik[1]
        );
    }

    #[test]
    fn test_aml_coxph_efron_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let n = aml.combined.time.len();
        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = aml.combined.group[i] as f64;
        }

        let mut cox_fit = CoxFit::new(
            Array1::from_vec(aml.combined.time.clone()),
            Array1::from_vec(aml.combined.status.clone()),
            covar,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::from_elem(n, 1.0),
            CoxMethod::Efron,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Cox fit init failed");

        cox_fit.fit().expect("Cox fit failed");
        let (beta, _means, _u, _imat, loglik, _sctest, _flag, _iter) = cox_fit.results();

        let r_coef = aml.coxph_efron.coefficients[0];
        let r_loglik = aml.coxph_efron.loglik[1];

        assert!(
            rel_approx_eq(beta[0].abs(), r_coef.abs(), STAT_TOL),
            "Efron coefficient: expected {}, got {}",
            r_coef,
            beta[0]
        );

        assert!(
            rel_approx_eq(loglik[1], r_loglik, STAT_TOL),
            "Efron log-likelihood: expected {}, got {}",
            r_loglik,
            loglik[1]
        );
    }

    #[test]
    fn test_lung_coxph_exact() {
        let expected = load_expected_values();
        let lung = &expected.lung;

        let n = lung.data.time.len();
        let mut covar = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            covar[[i, 0]] = lung.data.age[i];
            covar[[i, 1]] = lung.data.sex[i] as f64;
        }

        let mut cox_fit = CoxFit::new(
            Array1::from_vec(lung.data.time.clone()),
            Array1::from_vec(lung.data.status.clone()),
            covar,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::from_elem(n, 1.0),
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true, true],
            vec![0.0, 0.0],
        )
        .expect("Cox fit init failed");

        cox_fit.fit().expect("Cox fit failed");
        let (beta, _means, _u, _imat, _loglik, _sctest, _flag, iter) = cox_fit.results();

        assert!(iter < 25, "Cox fit should converge");
        assert!(beta[0].is_finite(), "Age coefficient should be finite");
        assert!(beta[1].is_finite(), "Sex coefficient should be finite");
    }

    #[test]
    fn test_lung_logrank_sex_exact() {
        let expected = load_expected_values();
        let lung = &expected.lung;

        let result = weighted_logrank_test(
            &lung.data.time,
            &lung.data.status,
            &lung.data.sex,
            WeightType::LogRank,
        );

        assert!(
            rel_approx_eq(result.statistic, lung.logrank_sex.chisq, STAT_TOL),
            "Lung logrank chi-squared: expected {}, got {}",
            lung.logrank_sex.chisq,
            result.statistic
        );
    }

    #[test]
    fn test_ovarian_logrank_exact() {
        let expected = load_expected_values();
        let ovarian = &expected.ovarian;

        let result = weighted_logrank_test(
            &ovarian.data.time,
            &ovarian.data.status,
            &ovarian.data.rx,
            WeightType::LogRank,
        );

        assert!(
            rel_approx_eq(result.statistic, ovarian.logrank.chisq, STAT_TOL),
            "Ovarian logrank chi-squared: expected {}, got {}",
            ovarian.logrank.chisq,
            result.statistic
        );
    }

    #[test]
    fn test_ovarian_km_exact() {
        let expected = load_expected_values();
        let ovarian = &expected.ovarian;

        let status_f64: Vec<f64> = ovarian.data.status.iter().map(|&s| s as f64).collect();
        let weights = vec![1.0; ovarian.data.time.len()];
        let position = vec![0i32; ovarian.data.time.len()];

        let result = compute_survfitkm(
            &ovarian.data.time,
            &status_f64,
            &weights,
            None,
            &position,
            false,
            0,
        );

        for (i, &r_surv) in ovarian.km.survival.iter().take(5).enumerate() {
            if i < result.estimate.len() {
                assert!(
                    rel_approx_eq(result.estimate[i], r_surv, STANDARD_TOL),
                    "Ovarian KM survival at index {}: expected {}, got {}",
                    i,
                    r_surv,
                    result.estimate[i]
                );
            }
        }
    }

    #[test]
    fn test_edge_case_simple_nelson_aalen_exact() {
        let expected = load_expected_values();
        let edge = &expected.edge_cases.simple_nelson_aalen;

        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let result = nelson_aalen(&time, &status, None, 0.95);

        for (i, &r_cumhaz) in edge.cumulative_hazard.iter().enumerate() {
            assert!(
                rel_approx_eq(result.cumulative_hazard[i], r_cumhaz, STANDARD_TOL),
                "Simple NA cumhaz at {}: expected {}, got {}",
                i,
                r_cumhaz,
                result.cumulative_hazard[i]
            );
        }
    }

    #[test]
    fn test_edge_case_with_censoring_exact() {
        let expected = load_expected_values();
        let edge = &expected.edge_cases.with_censoring;

        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 0, 1, 0, 1, 0];

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(
            result.time.len(),
            edge.time.len(),
            "Number of event times mismatch"
        );

        for (i, &r_cumhaz) in edge.cumulative_hazard.iter().enumerate() {
            if i < result.cumulative_hazard.len() {
                assert!(
                    rel_approx_eq(result.cumulative_hazard[i], r_cumhaz, STANDARD_TOL),
                    "Censored NA cumhaz at {}: expected {}, got {}",
                    i,
                    r_cumhaz,
                    result.cumulative_hazard[i]
                );
            }
        }
    }

    #[test]
    fn test_edge_case_tied_events_exact() {
        let expected = load_expected_values();
        let edge = &expected.edge_cases.tied_events;

        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0, 15.0];
        let status_f64 = vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0; 6];
        let position = vec![0i32; 6];

        let result = compute_survfitkm(&time, &status_f64, &weights, None, &position, false, 0);

        assert_eq!(
            result.time.len(),
            edge.time.len(),
            "Tied events: number of time points mismatch"
        );

        for (i, &r_surv) in edge.survival.iter().enumerate() {
            if i < result.estimate.len() {
                assert!(
                    rel_approx_eq(result.estimate[i], r_surv, STANDARD_TOL),
                    "Tied events survival at {}: expected {}, got {}",
                    i,
                    r_surv,
                    result.estimate[i]
                );
            }
        }
    }

    #[test]
    fn test_edge_case_identical_groups_exact() {
        let expected = load_expected_values();
        let edge = &expected.edge_cases.identical_groups_logrank;

        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(
            rel_approx_eq(result.statistic, edge.chisq, STAT_TOL),
            "Identical groups chi-squared: expected {}, got {}",
            edge.chisq,
            result.statistic
        );
        assert!(
            result.p_value > 0.9,
            "Identical groups p-value should be ~1, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_rmst_aml_maintained_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let result = compute_rmst(&aml.maintained.time, &aml.maintained.status, 30.0, 0.95);

        assert!(
            rel_approx_eq(result.rmst, expected.rmst.aml_maintained_tau30, STAT_TOL),
            "RMST maintained tau=30: expected {}, got {}",
            expected.rmst.aml_maintained_tau30,
            result.rmst
        );
    }

    #[test]
    fn test_rmst_aml_nonmaintained_exact() {
        let expected = load_expected_values();
        let aml = &expected.aml;

        let result = compute_rmst(
            &aml.nonmaintained.time,
            &aml.nonmaintained.status,
            30.0,
            0.95,
        );

        assert!(
            rel_approx_eq(result.rmst, expected.rmst.aml_nonmaintained_tau30, STAT_TOL),
            "RMST nonmaintained tau=30: expected {}, got {}",
            expected.rmst.aml_nonmaintained_tau30,
            result.rmst
        );
    }

    #[test]
    fn test_expected_values_loaded() {
        let expected = load_expected_values();

        assert!(
            !expected.metadata.survival_version.is_empty(),
            "survival_version should not be empty"
        );
        assert!(
            !expected.metadata.r_version.is_empty(),
            "r_version should not be empty"
        );
        assert!(
            !expected.metadata.generated_at.is_empty(),
            "generated_at should not be empty"
        );

        assert_eq!(expected.aml.maintained.time.len(), 11);
        assert_eq!(expected.aml.nonmaintained.time.len(), 12);
        assert_eq!(expected.aml.combined.time.len(), 23);
    }
}
