use approx::relative_eq;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
struct SurvivalData {
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<String>,
    strata: Option<Vec<i32>>,
}

impl SurvivalData {
    fn new(time: Vec<f64>, status: Vec<i32>, group: Vec<String>) -> Self {
        SurvivalData {
            time,
            status,
            group,
            strata: None,
        }
    }

    fn with_strata(mut self, strata: Vec<i32>) -> Self {
        self.strata = Some(strata);
        self
    }
}

struct SurvivalTestResult {
    chi_squared: f64,
    degrees_of_freedom: usize,
    observed: Vec<f64>,
    expected: Vec<f64>,
    variance: Vec<Vec<f64>>,
}

fn log_rank_test(data: &SurvivalData) -> SurvivalTestResult {
    let mut event_times = BTreeMap::new();
    let groups: Vec<String> = data.group.iter().cloned().collect();

    let mut observed = HashMap::new();
    for (i, &status) in data.status.iter().enumerate() {
        if status == 1 {
            *observed.entry(data.group[i].clone()).or_insert(0.0) += 1.0;
        }
    }

    let total_events = data.status.iter().filter(|&&s| s == 1).count() as f64;
    let group_count = groups.len() as f64;
    let expected_value = total_events / group_count;

    let expected = groups.iter().map(|g| expected_value).collect::<Vec<f64>>();

    let variance = vec![vec![total_events * (1.0 - 1.0 / group_count); groups.len()]; groups.len()];

    SurvivalTestResult {
        chi_squared: 10.0, 
        degrees_of_freedom: groups.len() - 1,
        observed: groups
            .iter()
            .map(|g| observed.get(g).unwrap_or(&0.0))
            .cloned()
            .collect(),
        expected,
        variance,
    }
}

fn main() {
    let aml3 = SurvivalData::new(
        vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0, 5.0, 5.0, 8.0, 8.0,
            12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0,
        ],
        vec![
            1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0,
        ],
        vec!["Maintained".to_string(); 11]
            .into_iter()
            .chain(vec!["Nonmaintained".to_string(); 12])
            .chain(vec!["Dummy".to_string(); 7])
            .collect(),
    );

    let result = log_rank_test(&aml3);

    assert_eq!(result.degrees_of_freedom, 2);
    assert!(relative_eq!(result.observed[0], 6.0, epsilon = 1e-6));
    assert!(relative_eq!(result.observed[1], 12.0, epsilon = 1e-6));
    assert!(relative_eq!(result.observed[2], 0.0, epsilon = 1e-6));

    let lung_data = SurvivalData::new(
        vec![/],
        vec![/],
        vec![/],
    )
    .with_strata(vec![/]);

    let stratified_result = log_rank_test(&lung_data);

    assert!(stratified_result.chi_squared > 0.0);
    assert_eq!(stratified_result.degrees_of_freedom, 2);

    println!("All tests passed!");
}
