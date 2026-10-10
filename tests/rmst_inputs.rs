//! RMST comparison validates row alignment before selecting group observations.

use survival::surv_analysis::{RmeanOption, SurvfitKMData, SurvfitKMOptions, survfitkm, survmean};
use survival::validation::rmst_comparison;

const TIME: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
const STATUS: [i32; 4] = [1, 1, 0, 1];
const GROUP: [i32; 4] = [-3, 7, -3, 7];

#[test]
fn comparison_rejects_short_and_long_status_before_subsetting() {
    for len in [0, 1, 3, 5] {
        let mut status = vec![1; len];
        if len > TIME.len() {
            status[TIME.len()] = 2; // This trailing value must not be discarded.
        }
        let error = rmst_comparison(&TIME, &status, &GROUP, None, 4.0, 0.95)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("status length mismatch"),
            "length {len}: {error}"
        );
    }
}

#[test]
fn comparison_rejects_short_and_long_weights_before_subsetting() {
    for len in [0, 1, 3, 5] {
        let mut weights = vec![1.0; len];
        if len > TIME.len() {
            weights[TIME.len()] = -1.0;
        }
        let error = rmst_comparison(&TIME, &STATUS, &GROUP, Some(&weights), 4.0, 0.95)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("weights length mismatch"),
            "length {len}: {error}"
        );
    }
}

#[test]
fn weighted_group_comparison_agrees_with_independently_fitted_curves() {
    let weights = [0.5, 1.5, 2.0, 1.0];
    let result = rmst_comparison(&TIME, &STATUS, &GROUP, Some(&weights), 4.0, 0.95).unwrap();
    assert_eq!(
        result
            .groups
            .iter()
            .map(|group| group.group)
            .collect::<Vec<_>>(),
        [-3, 7]
    );
    assert_eq!(result.df, 1);
    for (group, rows) in result.groups.iter().zip([[0, 2], [1, 3]]) {
        let data = SurvfitKMData::try_new(
            None,
            rows.iter().map(|&row| TIME[row]).collect(),
            rows.iter().map(|&row| STATUS[row]).collect(),
            Some(rows.iter().map(|&row| weights[row]).collect()),
            None,
            None,
            None,
        )
        .unwrap();
        let fit = survfitkm(&data, &SurvfitKMOptions::default()).unwrap();
        let table = survmean(&fit, 1.0, RmeanOption::At(4.0)).unwrap();
        assert_eq!(group.n, 2);
        assert_eq!(group.events, table.events[0]);
        assert_eq!(group.rmean, table.rmean.as_ref().unwrap()[0]);
        assert_eq!(group.se_rmean, table.se_rmean.as_ref().unwrap()[0]);
    }
    assert!((result.groups[0].rmean - 3.4).abs() < 1e-12);
    assert!((result.groups[1].rmean - 2.8).abs() < 1e-12);
    assert!((result.difference[0] + 0.6).abs() < 1e-12);
    assert!(result.difference_se[0].is_finite());
    assert!(result.p_value.is_finite());
}

#[test]
fn interleaved_groups_with_ties_agree_with_independently_fitted_curves() {
    let labels = [21, -8, 4, -2, 11, 0, 7];
    let group: Vec<i32> = (0..140).map(|row| labels[row % labels.len()]).collect();
    let time: Vec<f64> = (0..140).map(|row| ((row * 13) % 19 + 1) as f64).collect();
    let status: Vec<i32> = (0..140).map(|row| i32::from(row % 5 != 0)).collect();
    let weights: Vec<f64> = (0..140).map(|row| 0.25 + (row % 11) as f64 / 8.0).collect();
    let result = rmst_comparison(&time, &status, &group, Some(&weights), 16.0, 0.9).unwrap();
    let mut ordered_labels = labels;
    ordered_labels.sort_unstable();
    assert_eq!(
        result.groups.iter().map(|g| g.group).collect::<Vec<_>>(),
        ordered_labels
    );
    for actual in result.groups {
        // Select from the original input independently, including its tied
        // times and fractional-weight addition order within each group.
        let rows: Vec<usize> = (0..time.len())
            .filter(|&row| group[row] == actual.group)
            .collect();
        let data = SurvfitKMData::try_new(
            None,
            rows.iter().map(|&row| time[row]).collect(),
            rows.iter().map(|&row| status[row]).collect(),
            Some(rows.iter().map(|&row| weights[row]).collect()),
            None,
            None,
            None,
        )
        .unwrap();
        let fit = survfitkm(
            &data,
            &SurvfitKMOptions {
                conf_int: 0.9,
                ..SurvfitKMOptions::default()
            },
        )
        .unwrap();
        let table = survmean(&fit, 1.0, RmeanOption::At(16.0)).unwrap();
        assert_eq!(actual.n, rows.len());
        assert_eq!(actual.events, table.events[0]);
        assert_eq!(actual.rmean, table.rmean.as_ref().unwrap()[0]);
        assert_eq!(actual.se_rmean, table.se_rmean.as_ref().unwrap()[0]);
    }
}
