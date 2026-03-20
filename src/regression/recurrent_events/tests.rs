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
    fn test_bladder_recurrent_event_models() {
        let bladder = load_legacy_bladder_data();

        assert_eq!(bladder.id.len(), 178);
        assert_eq!(bladder.event.iter().filter(|&&e| e == 1).count(), 112);

        let gap_config = PWPConfig::new(PWPTimescale::Gap, 100, 1e-6, true, true);
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

        let total_config = PWPConfig::new(PWPTimescale::Total, 100, 1e-6, true, true);
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
        assert_eq!(gap.baseline_cumhaz.len(), 28);
        assert_close(gap.coef[0], -0.2695101867681618, 1e-9);
        assert_close(gap.coef[1], 0.0068363097719597865, 1e-9);
        assert_close(gap.coef[2], 0.15353662917417513, 1e-9);

        assert_eq!(total.n_subjects, 85);
        assert_eq!(total.n_events, 112);
        assert!(total.converged);
        assert_close(total.coef[0], -0.5167094835791394, 1e-9);
        assert_close(total.coef[1], -0.007743184659185533, 1e-9);
        assert_close(total.coef[2], 0.10287711173954855, 1e-9);

        assert_eq!(ag.n_subjects, 85);
        assert_eq!(ag.n_events, 112);
        assert!(ag.converged);
        assert_close(ag.coef[0], -0.45978826074398993, 1e-9);
        assert_close(ag.coef[1], -0.04256340004595282, 1e-9);
        assert_close(ag.coef[2], 0.17164542460626836, 1e-9);

        assert_eq!(wlw.n_subjects, 85);
        assert_eq!(wlw.n_events, 112);
        assert_eq!(wlw.n_strata, 4);
        assert!(wlw.converged);
        assert_close(wlw.coef[0], -0.5798694870405632, 1e-9);
        assert_close(wlw.coef[1], -0.050935433404071695, 1e-9);
        assert_close(wlw.coef[2], 0.20849094150265948, 1e-9);
        assert_close(wlw.global_test_stat, 12.37081136240878, 1e-9);
        assert_close(wlw.global_test_pvalue, 0.006215083463136151, 1e-12);

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
}
