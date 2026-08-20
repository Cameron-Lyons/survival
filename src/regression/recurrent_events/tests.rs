#[cfg(test)]
mod tests {
    use super::*;

    struct LegacyBladderData {
        id: Vec<i32>,
        start: Vec<f64>,
        stop: Vec<f64>,
        event: Vec<i32>,
        event_number: Vec<i32>,
        covariates: Vec<f64>,
        wlw_id: Vec<i32>,
        wlw_time: Vec<f64>,
        wlw_event: Vec<i32>,
        wlw_stratum: Vec<i32>,
        wlw_covariates: Vec<f64>,
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn load_legacy_bladder_data() -> LegacyBladderData {
        let csv = include_str!("../../datasets/data/bladder.csv");
        let mut rows: Vec<(i32, i32, i32, i32, i32, i32, i32)> = csv
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let values: Vec<i32> = line
                    .split(',')
                    .map(|value| value.trim().parse::<i32>().expect("valid bladder integer"))
                    .collect();
                assert_eq!(values.len(), 8, "unexpected bladder row width");
                (
                    values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                )
            })
            .collect();

        rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.6.cmp(&b.6)));

        let mut id = Vec::new();
        let mut start = Vec::new();
        let mut stop = Vec::new();
        let mut event = Vec::new();
        let mut event_number = Vec::new();
        let mut covariates = Vec::new();

        let mut wlw_id = Vec::with_capacity(rows.len());
        let mut wlw_time = Vec::with_capacity(rows.len());
        let mut wlw_event = Vec::with_capacity(rows.len());
        let mut wlw_stratum = Vec::with_capacity(rows.len());
        let mut wlw_covariates = Vec::with_capacity(rows.len() * 3);

        let mut idx = 0;
        while idx < rows.len() {
            let current_id = rows[idx].0;
            let mut subject_rows = Vec::new();
            while idx < rows.len() && rows[idx].0 == current_id {
                let row = rows[idx];
                subject_rows.push(row);
                wlw_id.push(row.0);
                wlw_time.push(row.4 as f64);
                wlw_event.push(row.5);
                wlw_stratum.push(row.6);
                wlw_covariates.extend([row.1 as f64, row.3 as f64, row.2 as f64]);
                idx += 1;
            }

            let mut previous_stop = 0.0;
            for &(subject_id, rx, number, size, subject_stop, subject_event, subject_enum) in
                &subject_rows
            {
                let subject_stop = subject_stop as f64;
                if subject_event == 1 || subject_stop > previous_stop {
                    id.push(subject_id);
                    start.push(previous_stop);
                    stop.push(subject_stop);
                    event.push(subject_event);
                    event_number.push(subject_enum);
                    covariates.extend([rx as f64, size as f64, number as f64]);
                    previous_stop = subject_stop;
                }
            }
        }

        assert_eq!(wlw_id.len(), rows.len());

        LegacyBladderData {
            id,
            start,
            stop,
            event,
            event_number,
            covariates,
            wlw_id,
            wlw_time,
            wlw_event,
            wlw_stratum,
            wlw_covariates,
        }
    }

    #[test]
    fn test_pwp_gap_time() {
        let id = vec![1, 1, 2, 2, 3];
        let start = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let stop = vec![10.0, 20.0, 5.0, 15.0, 25.0];
        let event = vec![1, 0, 1, 1, 0];
        let event_number = vec![1, 2, 1, 2, 1];

        let config = PWPConfig::new(PWPTimescale::Gap, 50, 1e-4, true, true);
        let result = pwp_model(id, start, stop, event, event_number, vec![], &config).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert_eq!(result.n_events, 3);
    }

    #[test]
    fn test_pwp_total_time() {
        let id = vec![1, 1, 2, 2, 3];
        let start = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let stop = vec![10.0, 20.0, 5.0, 15.0, 25.0];
        let event = vec![1, 0, 1, 1, 0];
        let event_number = vec![1, 2, 1, 2, 1];

        let config = PWPConfig::new(PWPTimescale::Total, 50, 1e-4, false, true);
        let result = pwp_model(id, start, stop, event, event_number, vec![], &config).unwrap();

        assert_eq!(result.n_subjects, 3);
    }

    #[test]
    fn pwp_matches_stratified_cluster_references() {
        let n_subjects = 40;
        let mut id = Vec::new();
        let mut start = Vec::new();
        let mut stop = Vec::new();
        let mut event = Vec::new();
        let mut event_number = Vec::new();
        let mut covariates = Vec::new();
        for subject in 1..=n_subjects {
            let mut current_time = 0.0;
            for order in 1..=3 {
                let duration = ((subject * 3 + order * 5) % 7 + 1) as f64;
                let next_time = current_time + duration;
                let event_status = match order {
                    1 => subject % 5 != 0,
                    2 => subject % 4 != 0,
                    _ => subject % 3 != 0,
                };
                id.push(subject);
                start.push(current_time);
                stop.push(next_time);
                event.push(i32::from(event_status));
                event_number.push(order);
                covariates.push(((subject * 3) % 17) as f64 * 0.1);
                covariates.push(((subject * 7) % 13) as f64 * 0.1);
                current_time = next_time;
                if !event_status {
                    break;
                }
            }
        }

        let gap = pwp_model(
            id.clone(),
            start.clone(),
            stop.clone(),
            event.clone(),
            event_number.clone(),
            covariates.clone(),
            &PWPConfig::new(PWPTimescale::Gap, 50, 1e-9, true, true),
        )
        .expect("gap-time PWP reference fit should succeed");
        assert_close(gap.coef[0], -0.111_988_650_520_143_1, 1e-10);
        assert_close(gap.coef[1], -0.096_099_608_408_612_62, 1e-10);
        assert_close(gap.std_errors[0], 0.234_278_613_765_360_63, 1e-10);
        assert_close(gap.std_errors[1], 0.327_116_853_801_628_36, 1e-10);
        assert_close(gap.robust_std_errors[0], 0.187_038_839_798_890_3, 1e-10);
        assert_close(gap.robust_std_errors[1], 0.237_532_268_695_615_9, 1e-10);
        assert_close(gap.log_likelihood, -196.712_435_835_508_53, 1e-10);
        assert_eq!(gap.n_iter, 3);
        assert_eq!(gap.baseline_cumhaz.len(), 21);
        assert_eq!(gap.baseline_times.len(), 21);
        assert_eq!(gap.baseline_strata, [vec![1; 7], vec![2; 7], vec![3; 7]].concat());
        assert_close(gap.baseline_cumhaz[0], 0.152_289_350_786_669_1, 1e-10);
        assert_close(gap.baseline_cumhaz[6], 2.772_453_259_118_328_6, 1e-10);
        assert_close(gap.baseline_cumhaz[7], 0.152_081_988_866_538_95, 1e-10);
        assert_close(gap.baseline_cumhaz[20], 1.817_480_673_529_770_2, 1e-10);

        let total = pwp_model(
            id,
            start,
            stop,
            event,
            event_number,
            covariates,
            &PWPConfig::new(PWPTimescale::Total, 50, 1e-9, true, true),
        )
        .expect("total-time PWP reference fit should succeed");
        assert_close(total.coef[0], -0.233_837_087_127_739_32, 1e-10);
        assert_close(total.coef[1], -0.028_246_466_279_691_063, 1e-10);
        assert_close(total.std_errors[0], 0.240_991_214_994_265_32, 1e-10);
        assert_close(total.std_errors[1], 0.330_126_875_800_493_27, 1e-10);
        assert_close(total.robust_std_errors[0], 0.254_798_307_439_802_4, 1e-10);
        assert_close(total.robust_std_errors[1], 0.335_484_735_328_979_5, 1e-10);
        assert_close(total.log_likelihood, -183.340_957_383_822_74, 1e-10);
        assert_eq!(total.n_iter, 3);
        assert_eq!(total.baseline_cumhaz.len(), 21);
        assert_eq!(total.baseline_times.len(), 21);
        assert_eq!(total.baseline_strata, [vec![1; 7], vec![2; 7], vec![3; 7]].concat());
        assert_close(total.baseline_cumhaz[0], 0.160_568_133_525_993_22, 1e-10);
        assert_close(total.baseline_cumhaz[6], 2.915_964_520_738_541, 1e-10);
        assert_close(total.baseline_cumhaz[7], 0.378_009_324_927_201_63, 1e-10);
        assert_close(total.baseline_cumhaz[20], 3.023_371_741_032_096_6, 1e-10);
    }

    #[test]
    fn test_wlw_model() {
        let id = vec![1, 1, 2, 2, 3, 3];
        let time = vec![10.0, 20.0, 5.0, 15.0, 8.0, 25.0];
        let event = vec![1, 0, 1, 1, 0, 0];
        let stratum = vec![1, 2, 1, 2, 1, 2];

        let config = WLWConfig::new(50, 1e-4, true, false);
        let result = wlw_model(id, time, event, stratum, vec![], &config).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert_eq!(result.n_strata, 2);
        assert_eq!(result.n_events, 3);
    }

    #[test]
    fn wlw_matches_stratified_cluster_reference() {
        let n_subjects = 30;
        let mut id = Vec::with_capacity(n_subjects * 3);
        let mut time = Vec::with_capacity(n_subjects * 3);
        let mut event = Vec::with_capacity(n_subjects * 3);
        let mut stratum = Vec::with_capacity(n_subjects * 3);
        let mut covariates = Vec::with_capacity(n_subjects * 6);
        for subject in 1..=n_subjects {
            for event_order in 1..=3 {
                let row = id.len();
                id.push(subject as i32);
                stratum.push(event_order as i32);
                time.push(((row * 5 + subject * 3 + event_order) % 40 + 1) as f64);
                event.push(i32::from((row + subject + event_order) % 4 != 0));
                covariates.push(((row * 3 + subject * 2) % 17) as f64 * 0.1);
                covariates.push(((row * 7 + subject) % 13) as f64 * 0.1);
            }
        }

        let result = wlw_model(
            id.clone(),
            time.clone(),
            event.clone(),
            stratum.clone(),
            covariates.clone(),
            &WLWConfig::new(50, 1e-9, true, false),
        )
        .expect("WLW reference fit should succeed");

        assert_close(result.coef[0], -0.262_198_198_154_192, 1e-10);
        assert_close(result.coef[1], 0.207_452_690_273_597, 1e-10);
        assert_close(result.std_errors[0], 0.267_872_851_791_448, 1e-10);
        assert_close(result.std_errors[1], 0.352_068_828_626_283, 1e-10);
        assert_close(result.robust_std_errors[0], 0.251_017_039_175_710, 1e-10);
        assert_close(result.robust_std_errors[1], 0.273_495_948_040_538, 1e-10);
        assert_close(result.log_likelihood, -148.584_208_908_079, 1e-10);
        assert_close(result.global_test_stat, 1.695_741_210_843_69, 1e-10);
        assert_eq!(result.n_iter, 3);
        assert!(result.converged);

        let common_baseline = wlw_model(
            id,
            time,
            event,
            stratum,
            covariates,
            &WLWConfig::new(50, 1e-9, true, true),
        )
        .expect("common-baseline WLW reference fit should succeed");
        assert_close(common_baseline.coef[0], -0.040_828_254_348_441_3, 1e-10);
        assert_close(common_baseline.coef[1], 0.024_655_056_553_873_553, 1e-10);
        assert_close(common_baseline.log_likelihood, -215.996_228_891_643_6, 1e-10);
        assert_close(common_baseline.global_test_stat, 0.046_431_337_489_869_93, 1e-10);
        assert_eq!(common_baseline.n_iter, 2);
    }

    #[test]
    fn test_bladder_recurrent_event_models() {
        let bladder = load_legacy_bladder_data();

        assert_eq!(bladder.id.len(), 178);
        assert_eq!(bladder.event.iter().filter(|&&e| e == 1).count(), 112);

        let gap_config = PWPConfig::new(PWPTimescale::Gap, 100, 1e-9, true, true);
        let gap = pwp_model(
            bladder.id.clone(),
            bladder.start.clone(),
            bladder.stop.clone(),
            bladder.event.clone(),
            bladder.event_number.clone(),
            bladder.covariates.clone(),
            &gap_config,
        );
        assert!(gap.is_ok());
        let gap = gap.expect("gap-time PWP result should be present");

        let total_config = PWPConfig::new(PWPTimescale::Total, 100, 1e-9, true, true);
        let total = pwp_model(
            bladder.id.clone(),
            bladder.start.clone(),
            bladder.stop.clone(),
            bladder.event.clone(),
            bladder.event_number.clone(),
            bladder.covariates.clone(),
            &total_config,
        );
        assert!(total.is_ok());
        let total = total.expect("total-time PWP result should be present");

        let ag = anderson_gill_model(
            bladder.id.clone(),
            bladder.start.clone(),
            bladder.stop.clone(),
            bladder.event.clone(),
            bladder.covariates.clone(),
            100,
            1e-6,
        );
        assert!(ag.is_ok());
        let ag = ag.expect("Anderson-Gill result should be present");

        let wlw_config = WLWConfig::new(100, 1e-6, true, false);
        let wlw = wlw_model(
            bladder.wlw_id,
            bladder.wlw_time,
            bladder.wlw_event,
            bladder.wlw_stratum,
            bladder.wlw_covariates,
            &wlw_config,
        );
        assert!(wlw.is_ok());
        let wlw = wlw.expect("WLW result should be present");

        assert_eq!(gap.n_subjects, 85);
        assert_eq!(gap.n_events, 112);
        assert!(gap.converged);
        assert_eq!(gap.event_specific_coef.len(), 4);
        assert_eq!(gap.baseline_cumhaz.len(), 88);
        assert_eq!(gap.baseline_times.len(), 88);
        assert_eq!(gap.baseline_strata.len(), 88);
        assert_close(gap.coef[0], -0.279_004_515_497_076_56, 1e-9);
        assert_close(gap.coef[1], 0.007_415_087_501_994_803, 1e-9);
        assert_close(gap.coef[2], 0.158_045_899_958_246_67, 1e-9);
        assert_close(gap.std_errors[0], 0.207_347_725_416_916_23, 1e-9);
        assert_close(gap.robust_std_errors[0], 0.215_623_869_338_403_88, 1e-9);
        assert_close(gap.log_likelihood, -358.968_485_369_266_4, 1e-9);
        assert_eq!(gap.n_iter, 4);

        assert_eq!(total.n_subjects, 85);
        assert_eq!(total.n_events, 112);
        assert!(total.converged);
        assert_eq!(total.baseline_cumhaz.len(), 100);
        assert_eq!(total.baseline_times.len(), 100);
        assert_eq!(total.baseline_strata.len(), 100);
        assert_close(total.coef[0], -0.333_488_726_434_080_34, 1e-9);
        assert_close(total.coef[1], -0.008_494_666_106_611_051, 1e-9);
        assert_close(total.coef[2], 0.119_617_237_977_565_67, 1e-9);
        assert_close(total.std_errors[0], 0.216_167_915_402_341_34, 1e-9);
        assert_close(total.robust_std_errors[0], 0.204_786_771_935_949_46, 1e-9);
        assert_close(total.log_likelihood, -315.990_824_871_805_84, 1e-9);
        assert_eq!(total.n_iter, 4);

        assert_eq!(ag.n_subjects, 85);
        assert_eq!(ag.n_events, 112);
        assert!(ag.converged);
        assert_close(ag.coef[0], -0.464_686_952_943_429, 1e-9);
        assert_close(ag.coef[1], -0.043_660_276_089_821_9, 1e-9);
        assert_close(ag.coef[2], 0.174_960_381_319_399, 1e-9);
        assert_close(ag.std_errors[0], 0.199_732_188_827_129, 1e-9);
        assert_close(ag.std_errors[1], 0.069_050_852_511_246, 1e-9);
        assert_close(ag.std_errors[2], 0.047_074_058_440_440_7, 1e-9);
        assert_close(ag.robust_std_errors[0], 0.265_560_811_898_887, 1e-9);
        assert_close(ag.robust_std_errors[1], 0.077_616_112_681_007_7, 1e-9);
        assert_close(ag.robust_std_errors[2], 0.063_040_543_200_655_3, 1e-9);
        assert_close(ag.log_likelihood, -449.980_642_001_765, 1e-9);
        assert_eq!(ag.n_iter, 4);

        assert_eq!(wlw.n_subjects, 85);
        assert_eq!(wlw.n_events, 112);
        assert_eq!(wlw.n_strata, 4);
        assert!(wlw.converged);
        assert_close(wlw.coef[0], -0.584_793_457_410_789_8, 1e-9);
        assert_close(wlw.coef[1], -0.051_616_982_656_100_99, 1e-9);
        assert_close(wlw.coef[2], 0.210_293_707_384_945_54, 1e-9);
        assert_close(wlw.std_errors[0], 0.201_050_602_379_388_63, 1e-9);
        assert_close(wlw.std_errors[1], 0.069_734_320_003_082_08, 1e-9);
        assert_close(wlw.std_errors[2], 0.046_754_789_784_791_15, 1e-9);
        assert_close(wlw.robust_std_errors[0], 0.307_946_258_183_588_84, 1e-9);
        assert_close(wlw.robust_std_errors[1], 0.094_586_644_025_190_86, 1e-9);
        assert_close(wlw.robust_std_errors[2], 0.066_641_686_829_874_4, 1e-9);
        assert_close(wlw.log_likelihood, -426.146_832_546_882_1, 1e-9);
        assert_close(wlw.global_test_stat, 15.537_046_791_588_393, 1e-9);
        assert_close(wlw.global_test_pvalue, 0.001_410_738_150_921_807, 1e-12);
        assert_eq!(wlw.n_iter, 4);

        assert!(gap.hazard_ratios[0] < 1.0);
        assert!(total.hazard_ratios[0] < gap.hazard_ratios[0]);
        assert!(wlw.hazard_ratios[2] > 1.0);
    }

    #[test]
    fn test_negative_binomial_frailty() {
        let id = vec![1, 1, 2, 2, 2, 3];
        let time = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let event = vec![1, 0, 1, 1, 0, 0];

        let config = NegativeBinomialFrailtyConfig::new(50, 1e-4, 20);
        let result = negative_binomial_frailty(id, time, event, vec![], None, &config).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert!(result.theta > 0.0);
        assert_eq!(result.frailty_estimates.len(), 3);
    }

    #[test]
    fn test_anderson_gill() {
        let id = vec![1, 1, 2, 2, 3];
        let start = vec![0.0, 10.0, 0.0, 5.0, 0.0];
        let stop = vec![10.0, 20.0, 5.0, 15.0, 25.0];
        let event = vec![1, 0, 1, 1, 0];

        let result = anderson_gill_model(id, start, stop, event, vec![], 50, 1e-4).unwrap();

        assert_eq!(result.n_subjects, 3);
        assert_eq!(result.n_events, 3);
        assert!(result.mean_event_rate > 0.0);
    }

    #[test]
    fn anderson_gill_matches_tied_cluster_reference() {
        let n_subjects = 20;
        let mut id = Vec::with_capacity(n_subjects * 3);
        let mut start = Vec::with_capacity(n_subjects * 3);
        let mut stop = Vec::with_capacity(n_subjects * 3);
        let mut event = Vec::with_capacity(n_subjects * 3);
        let mut covariates = Vec::with_capacity(n_subjects * 6);
        for subject in 1..=n_subjects {
            for interval in 0..3 {
                let row = id.len();
                id.push(subject as i32);
                start.push(interval as f64);
                stop.push((interval + 1) as f64);
                event.push(i32::from(row % 4 != 0));
                covariates.push(((row * 3 + subject * 2) % 17) as f64 * 0.1);
                covariates.push(((row * 7 + subject) % 13) as f64 * 0.1);
            }
        }

        let result = anderson_gill_model(id, start, stop, event, covariates, 50, 1e-9).unwrap();

        assert_close(result.coef[0], -0.176_545_463_952_229, 1e-10);
        assert_close(result.coef[1], 0.038_681_549_424_786_2, 1e-10);
        assert_close(result.std_errors[0], 0.309_771_133_238_645, 1e-10);
        assert_close(result.std_errors[1], 0.407_236_351_321_3, 1e-10);
        assert_close(result.robust_std_errors[0], 0.272_213_095_533_931, 1e-10);
        assert_close(result.robust_std_errors[1], 0.341_977_219_513_29, 1e-10);
        assert_close(result.log_likelihood, -112.465_975_135_154, 1e-10);
        assert_eq!(result.n_iter, 3);
        assert!(result.converged);
    }

    #[test]
    fn recurrent_models_validate_public_inputs() {
        assert!(anderson_gill_model(vec![], vec![], vec![], vec![], vec![], 10, 1e-6).is_err());
        assert!(
            anderson_gill_model(
                vec![1, 2],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![1, 0],
                vec![0.5],
                10,
                1e-6,
            )
            .is_err()
        );
        assert!(
            anderson_gill_model(vec![1], vec![0.0], vec![0.0], vec![1], vec![], 10, 1e-6)
                .is_err()
        );
        assert!(
            anderson_gill_model(vec![1], vec![0.0], vec![1.0], vec![2], vec![], 10, 1e-6)
                .is_err()
        );
        assert!(
            anderson_gill_model(
                vec![1],
                vec![0.0],
                vec![1.0],
                vec![1],
                vec![f64::NAN],
                10,
                1e-6,
            )
            .is_err()
        );
        assert!(
            anderson_gill_model(vec![1], vec![0.0], vec![1.0], vec![1], vec![], 0, 1e-6)
                .is_err()
        );

        let pwp_config = PWPConfig::new(PWPTimescale::Gap, 10, 1e-6, true, true);
        assert!(
            pwp_model(
                vec![1],
                vec![0.0],
                vec![1.0],
                vec![1],
                vec![0],
                vec![],
                &pwp_config,
            )
            .is_err()
        );
        let mut bad_pwp_config = PWPConfig::new(PWPTimescale::Gap, 10, 1e-6, true, true);
        bad_pwp_config.tol = f64::NAN;
        assert!(
            pwp_model(
                vec![1],
                vec![0.0],
                vec![1.0],
                vec![1],
                vec![1],
                vec![],
                &bad_pwp_config,
            )
            .is_err()
        );

        let wlw_config = WLWConfig::new(10, 1e-6, true, false);
        assert!(wlw_model(vec![1], vec![f64::INFINITY], vec![1], vec![1], vec![], &wlw_config).is_err());
        assert!(wlw_model(vec![1], vec![1.0], vec![2], vec![1], vec![], &wlw_config).is_err());
        let bad_wlw_config = WLWConfig::new(0, 1e-6, true, false);
        assert!(wlw_model(vec![1], vec![1.0], vec![1], vec![1], vec![], &bad_wlw_config).is_err());

        let nb_config = NegativeBinomialFrailtyConfig::new(10, 1e-6, 10);
        assert!(
            negative_binomial_frailty(vec![1], vec![-1.0], vec![1], vec![], None, &nb_config)
                .is_err()
        );
        assert!(
            negative_binomial_frailty(vec![1], vec![1.0], vec![-1], vec![], None, &nb_config)
                .is_err()
        );
        assert!(
            negative_binomial_frailty(
                vec![1, 2],
                vec![1.0, 1.0],
                vec![1, 0],
                vec![],
                Some(vec![0.0]),
                &nb_config,
            )
            .is_err()
        );
        assert!(
            negative_binomial_frailty(
                vec![1],
                vec![1.0],
                vec![1],
                vec![],
                Some(vec![f64::NAN]),
                &nb_config,
            )
            .is_err()
        );
        let bad_nb_config = NegativeBinomialFrailtyConfig::new(10, 1e-6, 0);
        assert!(
            negative_binomial_frailty(vec![1], vec![1.0], vec![1], vec![], None, &bad_nb_config)
                .is_err()
        );
    }
}
