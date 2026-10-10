use std::panic::{AssertUnwindSafe, catch_unwind};

use survival::population::{PyearsResult, PyearsSummaryOptions, summary_pyears};

fn empty(dims: Vec<usize>) -> PyearsResult {
    PyearsResult {
        pyears: vec![],
        n: vec![],
        event: Some(vec![]),
        expected: Some(vec![]),
        offtable: 0.0,
        dims,
        observations: 0,
    }
}

#[test]
fn summary_rejects_unaddressable_extents_and_malformed_cells_without_panicking() {
    let maximum = isize::MAX as usize / size_of::<f64>();
    let overflowing = 1usize << (usize::BITS / 2);
    for dims in [
        vec![usize::MAX],
        vec![usize::MAX, 0],
        vec![0, usize::MAX],
        vec![0, 2, usize::MAX],
        vec![usize::MAX, 0, 1],
        vec![maximum + 1, 0],
        vec![overflowing, overflowing],
        vec![maximum, 2],
    ] {
        for totals in [false, true] {
            for tcut in [false, true] {
                let answer = catch_unwind(AssertUnwindSafe(|| {
                    summary_pyears(
                        &empty(dims.clone()),
                        tcut,
                        PyearsSummaryOptions {
                            totals,
                            ..Default::default()
                        },
                    )
                }))
                .expect("public Result API must return an error rather than panic");
                assert!(
                    answer
                        .unwrap_err()
                        .to_string()
                        .contains("addressable memory")
                );
            }
        }
    }

    // These input tables have zero cells and individually addressable extents.
    // Adding margins would require an unaddressable array, so no allocation is tried.
    for dims in [vec![0, maximum], vec![maximum, 0], vec![0, 2, maximum]] {
        for tcut in [false, true] {
            let answer = catch_unwind(AssertUnwindSafe(|| {
                summary_pyears(
                    &empty(dims.clone()),
                    tcut,
                    PyearsSummaryOptions {
                        totals: true,
                        ..Default::default()
                    },
                )
            }))
            .expect("expanded shape must be checked before allocating margins");
            assert!(
                answer
                    .unwrap_err()
                    .to_string()
                    .contains("addressable memory")
            );
        }
    }

    let fitted = PyearsResult {
        pyears: vec![10.0, 20.0],
        n: vec![2.0, 3.0],
        event: Some(vec![1.0, 2.0]),
        expected: Some(vec![0.5, 1.0]),
        offtable: 0.0,
        dims: vec![2],
        observations: 5,
    };
    for field in ["pyears", "n", "event", "expected"] {
        let mut short = fitted.clone();
        match field {
            "pyears" => short.pyears.pop(),
            "n" => short.n.pop(),
            "event" => short.event.as_mut().unwrap().pop(),
            _ => short.expected.as_mut().unwrap().pop(),
        };
        for totals in [false, true] {
            for tcut in [false, true] {
                assert!(
                    summary_pyears(
                        &short,
                        tcut,
                        PyearsSummaryOptions {
                            totals,
                            ..Default::default()
                        },
                    )
                    .unwrap_err()
                    .to_string()
                    .contains(field)
                );
            }
        }
    }
}

#[test]
fn summary_preserves_zero_cell_tables_and_scalar_conventions() {
    // Stock survival 3.8-12 summary.pyears(totals=TRUE) accepts the first
    // three zero-cell arrays and returns these zero margins. The remaining
    // cases preserve the native data summary's existing multidimensional behavior.
    for (dims, expanded, cells) in [
        (vec![0], vec![1], 1),
        (vec![0, 2], vec![1, 3], 3),
        (vec![2, 0], vec![3, 1], 3),
        (vec![0, 0], vec![1, 1], 1),
        (vec![2, 2, 0], vec![3, 3, 0], 0),
        (vec![0, 2, 2], vec![1, 3, 2], 6),
    ] {
        for totals in [false, true] {
            for tcut in [false, true] {
                let actual = summary_pyears(
                    &empty(dims.clone()),
                    tcut,
                    PyearsSummaryOptions {
                        totals,
                        rate: true,
                        ci_r: true,
                        ci_rr: true,
                        ..Default::default()
                    },
                )
                .unwrap();
                let count = if totals { cells } else { 0 };
                assert_eq!(actual.dims, if totals { &expanded } else { &dims }.clone());
                assert_eq!(actual.pyears, vec![0.0; count]);
                assert_eq!(actual.event, Some(vec![0.0; count]));
                assert_eq!(actual.expected, Some(vec![0.0; count]));
                assert_eq!(actual.n.len(), count);
                assert!(
                    actual
                        .n
                        .iter()
                        .all(|n| if tcut { n.is_nan() } else { *n == 0.0 })
                );
                for values in [
                    actual.rate,
                    actual.rr,
                    actual.ci_r_lower,
                    actual.ci_r_upper,
                    actual.ci_rr_lower,
                    actual.ci_rr_upper,
                ] {
                    let values = values.unwrap();
                    assert_eq!(values.len(), count);
                    assert!(values.iter().all(|value| value.is_nan()));
                }
                assert_eq!(actual.total_events, 0.0);
                assert_eq!(actual.total_pyears, 0.0);
            }
        }
    }

    let maximum = isize::MAX as usize / size_of::<f64>();
    for totals in [false, true] {
        // No cells or margins exist in this trailing-zero table. Avoid
        // overflowing intermediate products of the unused large extents.
        let actual = summary_pyears(
            &empty(vec![maximum - 1, maximum - 1, 0]),
            false,
            PyearsSummaryOptions {
                totals,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(actual.pyears.is_empty());
        let width = if totals { maximum } else { maximum - 1 };
        assert_eq!(actual.dims, vec![width, width, 0]);
    }

    let scalar = PyearsResult {
        pyears: vec![5.0],
        n: vec![2.0],
        event: Some(vec![1.0]),
        expected: Some(vec![0.5]),
        offtable: 0.25,
        dims: vec![],
        observations: 2,
    };
    for totals in [false, true] {
        let actual = summary_pyears(
            &scalar,
            false,
            PyearsSummaryOptions {
                totals,
                rate: true,
                ..Default::default()
            },
        )
        .unwrap();
        let cells = if totals { 2 } else { 1 };
        assert_eq!(actual.dims, vec![cells]);
        assert_eq!(actual.pyears, vec![5.0; cells]);
        assert_eq!(actual.n, vec![2.0; cells]);
        assert_eq!(actual.rate, Some(vec![0.2; cells]));
        assert_eq!(actual.rr, Some(vec![2.0; cells]));
    }
}

#[test]
fn summary_margins_preserve_independent_cell_sums_and_rates() {
    // Hand-summed column-major 2 x 3 slabs; each receives its own row,
    // column and grand totals. These are not generated by the kernel.
    let expected_n = [
        1., 2., 3., 3., 4., 7., 5., 6., 11., 9., 12., 21., 7., 8., 15., 9., 10., 19., 11., 12.,
        23., 27., 30., 57.,
    ];
    let expected_events = [
        0., 1., 1., 2., 3., 5., 4., 5., 9., 6., 9., 15., 6., 7., 13., 8., 9., 17., 10., 11., 21.,
        24., 27., 51.,
    ];
    let margins = [
        false, false, true, false, false, true, false, false, true, true, true, true,
    ];
    for (dims, cells, expanded, out_cells) in [
        (vec![2, 3], 6, vec![3, 4], 12),
        (vec![2, 3, 2], 12, vec![3, 4, 2], 24),
    ] {
        let result = PyearsResult {
            pyears: (1..=cells).map(|n| n as f64 * 10.0).collect(),
            n: (1..=cells).map(|n| n as f64).collect(),
            event: Some((0..cells).map(|n| n as f64).collect()),
            expected: Some((1..=cells).map(|n| n as f64 * 0.5).collect()),
            offtable: 0.25,
            dims,
            observations: cells,
        };
        for tcut in [false, true] {
            let actual = summary_pyears(
                &result,
                tcut,
                PyearsSummaryOptions {
                    totals: true,
                    rate: true,
                    ci_r: true,
                    ci_rr: true,
                    scale: 1000.0,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(actual.dims, expanded);
            assert_eq!(
                actual.event.as_deref().unwrap(),
                &expected_events[..out_cells]
            );
            for (i, &n) in expected_n[..out_cells].iter().enumerate() {
                if tcut && margins[i % 12] {
                    assert!(actual.n[i].is_nan());
                } else {
                    assert_eq!(actual.n[i], n);
                }
                assert_eq!(actual.pyears[i], n * 10.0);
                assert_eq!(actual.expected.as_ref().unwrap()[i], n * 0.5);
            }
            assert_eq!(actual.total_events, if cells == 6 { 15.0 } else { 66.0 });
            assert_eq!(actual.total_pyears, if cells == 6 { 210.0 } else { 780.0 });
            assert!((actual.rate.as_ref().unwrap()[11] - 500.0 / 7.0).abs() < 1e-12);
            assert_eq!(actual.rr.as_ref().unwrap()[11], 15.0 / 10.5);
            assert_eq!(actual.ci_r_lower.as_ref().unwrap()[0], 0.0);
            let zero_event_upper = -0.025f64.ln() / 10.0 * 1000.0;
            assert!((actual.ci_r_upper.as_ref().unwrap()[0] - zero_event_upper).abs() < 1e-9);
            if cells == 12 {
                assert!((actual.rate.as_ref().unwrap()[23] - 1700.0 / 19.0).abs() < 1e-12);
                assert_eq!(actual.rr.as_ref().unwrap()[23], 51.0 / 28.5);
            }
        }
    }
}
