#[derive(Debug, Clone, PartialEq)]
enum EventType {
    Censored,
    Type1,
    Type2,
}

#[derive(Debug)]
struct Observation {
    time: f64,
    status: EventType,
    x: f64,
    id: usize,
}

#[derive(Debug)]
struct ExpandedObservation {
    id: usize,
    fgstart: f64,
    fgstop: f64,
    fgstatus: EventType,
    fgwt: f64,
    x: f64,
}

struct CensoringDistribution {
    times: Vec<f64>,
    survival: Vec<f64>,
}

impl CensoringDistribution {
    fn get_survival(&self, t: f64) -> f64 {
        self.times
            .iter()
            .rposition(|&x| x < t)
            .map_or(1.0, |i| self.survival[i])
    }
}

fn compute_censoring_distribution() -> CensoringDistribution {
    CensoringDistribution {
        times: vec![0.0, 3.0, 4.0, 6.0, 8.0, 9.0],
        survival: vec![
            1.0,
            11.0 / 12.0,
            (11.0 / 12.0) * (8.0 / 10.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0) * (3.0 / 4.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0) * (3.0 / 4.0) * (2.0 / 3.0),
        ],
    }
}

fn finegray(data: Vec<Observation>, target_event: EventType) -> Vec<ExpandedObservation> {
    let cdist = compute_censoring_distribution();
    let mut expanded = Vec::new();

    for obs in data {
        match &obs.status {
            EventType::Type2 if target_event == EventType::Type1 => {
                let intervals = vec![0.0, 3.0, 4.0, 6.0, 8.0, 9.0, obs.time];
                let mut prev = 0.0;
                for &t in &intervals {
                    if t > prev && t <= obs.time {
                        let g_prev = cdist.get_survival(prev);
                        let g_t = cdist.get_survival(t);
                        let weight = if g_t == 0.0 { 0.0 } else { g_prev / g_t };
                        expanded.push(ExpandedObservation {
                            id: obs.id,
                            fgstart: prev,
                            fgstop: t,
                            fgstatus: if t == obs.time {
                                EventType::Type1
                            } else {
                                EventType::Censored
                            },
                            fgwt: weight,
                            x: obs.x,
                        });
                        prev = t;
                    }
                }
            }
            _ => {
                expanded.push(ExpandedObservation {
                    id: obs.id,
                    fgstart: 0.0,
                    fgstop: obs.time,
                    fgstatus: obs.status.clone(),
                    fgwt: 1.0,
                    x: obs.x,
                });
            }
        }
    }

    expanded
}

fn create_test_data() -> Vec<Observation> {
    let time = vec![1., 2., 3., 4., 4., 4., 5., 5., 6., 8., 8., 9., 10., 12.];
    let status = vec![1, 2, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0];
    let x = vec![5., 4., 3., 1., 2., 1., 1., 2., 2., 4., 6., 1., 2., 0.];
    let ids = 1..=14;

    status
        .into_iter()
        .zip(time)
        .zip(x)
        .zip(ids)
        .map(|(((s, t), x), id)| Observation {
            time: t,
            status: match s {
                0 => EventType::Censored,
                1 => EventType::Type1,
                2 => EventType::Type2,
                _ => panic!("Invalid status code"),
            },
            x,
            id,
        })
        .collect()
}

fn main() {
    let test_data = create_test_data();
    let expanded = finegray(test_data, EventType::Type1);

    assert_eq!(expanded.len(), 19);

    let expected_weights = vec![
        1.0, 1.0, 11.0 / 12.0, 8.0 / 10.0, 5.0 / 6.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0, 5.0 / 12.0,
        1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
    ];
    for (i, (obs, &expected)) in expanded.iter().zip(expected_weights.iter()).enumerate() {
        assert!(
            (obs.fgwt - expected).abs() < 1e-6,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected,
            obs.fgwt
        );
    }

    println!("All assertions passed!");
}
