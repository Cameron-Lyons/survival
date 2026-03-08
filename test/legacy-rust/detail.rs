struct Test2Row {
    start: f64,
    stop: f64,
    event: i32,
    x: f64,
}

fn main() {
    let test2_data = vec![
        Test2Row {
            start: 1.0,
            stop: 2.0,
            event: 1,
            x: 1.0,
        },
        Test2Row {
            start: 2.0,
            stop: 3.0,
            event: 1,
            x: 0.0,
        },
        Test2Row {
            start: 5.0,
            stop: 6.0,
            event: 1,
            x: 0.0,
        },
        Test2Row {
            start: 2.0,
            stop: 7.0,
            event: 1,
            x: 1.0,
        },
        Test2Row {
            start: 1.0,
            stop: 8.0,
            event: 1,
            x: 0.0,
        },
        Test2Row {
            start: 7.0,
            stop: 9.0,
            event: 1,
            x: 1.0,
        },
        Test2Row {
            start: 3.0,
            stop: 9.0,
            event: 1,
            x: 1.0,
        },
        Test2Row {
            start: 4.0,
            stop: 9.0,
            event: 0,
            x: 1.0,
        },
        Test2Row {
            start: 8.0,
            stop: 14.0,
            event: 0,
            x: 0.0,
        },
        Test2Row {
            start: 8.0,
            stop: 17.0,
            event: 0,
            x: 0.0,
        },
    ];

    let beta = -1.0;
    let r = beta.exp();

    let mut event_times: Vec<f64> = test2_data
        .iter()
        .filter(|row| row.event == 1)
        .map(|row| row.stop)
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut computed_hazards = Vec::new();
    for &time in &event_times {
        let sum_risk: f64 = test2_data
            .iter()
            .filter(|row| row.start < time && time <= row.stop)
            .map(|row| (beta * row.x).exp())
            .sum();
        let hazard = 1.0 / sum_risk;
        computed_hazards.push(hazard);
    }

    let expected_hazards = vec![
        1.0 / (r + 1.0),
        1.0 / (r + 2.0),
        1.0 / (3.0 * r + 2.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 2.0),
        1.0 / (2.0 * r + 2.0),
    ];

    let tolerance = 1e-6;
    for (i, (computed, expected)) in computed_hazards
        .iter()
        .zip(expected_hazards.iter())
        .enumerate()
    {
        assert!(
            (computed - expected).abs() < tolerance,
            "Hazard {} mismatch: computed {:.6}, expected {:.6}",
            i,
            computed,
            expected
        );
    }

    let computed_sum = computed_hazards[5] + computed_hazards[6];
    let expected_sum = expected_hazards[5] + expected_hazards[6];
    assert!(
        (computed_sum - expected_sum).abs() < tolerance,
        "Summed hazards mismatch: computed {:.6}, expected {:.6}",
        computed_sum,
        expected_sum
    );

    println!("All tests passed!");
}
