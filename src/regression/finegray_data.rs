use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FineGrayOutput {
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub end: Vec<f64>,
    #[pyo3(get)]
    pub wt: Vec<f64>,
    #[pyo3(get)]
    pub add: Vec<usize>,
}
#[pyfunction]
pub fn finegray(
    tstart: Vec<f64>,
    tstop: Vec<f64>,
    ctime: Vec<f64>,
    cprob: Vec<f64>,
    extend: Vec<bool>,
    keep: Vec<bool>,
) -> PyResult<FineGrayOutput> {
    validate_finegray_inputs(&tstart, &tstop, &ctime, &cprob, &extend, &keep)?;
    Ok(compute_finegray(
        &tstart, &tstop, &ctime, &cprob, &extend, &keep,
    ))
}

fn validate_finegray_inputs(
    tstart: &[f64],
    tstop: &[f64],
    ctime: &[f64],
    cprob: &[f64],
    extend: &[bool],
    keep: &[bool],
) -> PyResult<()> {
    let n = tstart.len();
    if tstop.len() != n {
        return Err(value_error(format!(
            "tstop length ({}) must match tstart length ({})",
            tstop.len(),
            n
        )));
    }
    if extend.len() != n {
        return Err(value_error(format!(
            "extend length ({}) must match tstart length ({})",
            extend.len(),
            n
        )));
    }

    for (idx, (&start, &stop)) in tstart.iter().zip(tstop.iter()).enumerate() {
        if !start.is_finite() {
            return Err(value_error(format!(
                "tstart contains non-finite value at index {}",
                idx
            )));
        }
        if !stop.is_finite() {
            return Err(value_error(format!(
                "tstop contains non-finite value at index {}",
                idx
            )));
        }
        if start > stop {
            return Err(value_error(format!(
                "tstart value {} exceeds tstop value {} at index {}",
                start, stop, idx
            )));
        }
    }

    let ncut = ctime.len();
    if cprob.len() != ncut {
        return Err(value_error(format!(
            "cprob length ({}) must match ctime length ({})",
            cprob.len(),
            ncut
        )));
    }
    if keep.len() != ncut {
        return Err(value_error(format!(
            "keep length ({}) must match ctime length ({})",
            keep.len(),
            ncut
        )));
    }

    for (idx, &time) in ctime.iter().enumerate() {
        if !time.is_finite() {
            return Err(value_error(format!(
                "ctime contains non-finite value at index {}",
                idx
            )));
        }
        if idx > 0 && time < ctime[idx - 1] {
            return Err(value_error("ctime must be sorted in nondecreasing order"));
        }
    }
    for (idx, &probability) in cprob.iter().enumerate() {
        if !probability.is_finite() {
            return Err(value_error(format!(
                "cprob contains non-finite value at index {}",
                idx
            )));
        }
        if !(0.0..=1.0).contains(&probability) || probability == 0.0 {
            return Err(value_error(format!(
                "cprob must contain values in (0, 1]; found {} at index {}",
                probability, idx
            )));
        }
    }

    Ok(())
}

pub(crate) fn compute_finegray(
    tstart: &[f64],
    tstop: &[f64],
    ctime: &[f64],
    cprob: &[f64],
    extend: &[bool],
    keep: &[bool],
) -> FineGrayOutput {
    let n = tstart.len();
    assert_eq!(tstop.len(), n);
    assert_eq!(extend.len(), n);
    let ncut = ctime.len();
    assert_eq!(cprob.len(), ncut);
    assert_eq!(keep.len(), ncut);
    let kept_indices: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_kept)| is_kept.then_some(idx))
        .collect();
    let mut extension_plans = Vec::with_capacity(n);
    let mut extra = 0usize;
    for (&ext, (&ts, &te)) in extend.iter().zip(tstart.iter().zip(tstop.iter())) {
        if ext && !ts.is_nan() && !te.is_nan() {
            let initial_cut = ctime.partition_point(|&cut_time| cut_time < te);
            let first_kept = kept_indices.partition_point(|&idx| idx <= initial_cut);
            extra += kept_indices.len() - first_kept;
            extension_plans.push((initial_cut, first_kept));
        } else {
            extension_plans.push((ncut, kept_indices.len()));
        }
    }
    let total = n + extra;
    let mut row = Vec::with_capacity(total);
    let mut start = Vec::with_capacity(total);
    let mut end = Vec::with_capacity(total);
    let mut wt = Vec::with_capacity(total);
    let mut add = Vec::with_capacity(total);
    for (i, (&original_start, &original_end)) in tstart.iter().zip(tstop.iter()).enumerate() {
        let (initial_cut, first_kept) = extension_plans[i];
        let (current_end, temp_wt) = if initial_cut < ncut {
            (ctime[initial_cut], cprob[initial_cut])
        } else {
            (original_end, 1.0)
        };
        row.push(i + 1);
        start.push(original_start);
        end.push(current_end);
        wt.push(1.0);
        add.push(0);
        if initial_cut < ncut {
            for (iadd, &cut_idx) in kept_indices[first_kept..].iter().enumerate() {
                row.push(i + 1);
                start.push(ctime[cut_idx - 1]);
                end.push(ctime[cut_idx]);
                wt.push(cprob[cut_idx] / temp_wt);
                add.push(iadd + 1);
            }
        }
    }
    FineGrayOutput {
        row,
        start,
        end,
        wt,
        add,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn compute_finegray_naive(
        tstart: &[f64],
        tstop: &[f64],
        ctime: &[f64],
        cprob: &[f64],
        extend: &[bool],
        keep: &[bool],
    ) -> FineGrayOutput {
        let ncut = ctime.len();
        let mut row = Vec::new();
        let mut start = Vec::new();
        let mut end = Vec::new();
        let mut wt = Vec::new();
        let mut add = Vec::new();

        for (i, ((&original_start, &original_end), &ext)) in tstart
            .iter()
            .zip(tstop.iter())
            .zip(extend.iter())
            .enumerate()
        {
            let is_valid = !original_start.is_nan() && !original_end.is_nan();
            let is_extended = ext && is_valid;
            let (current_end, temp_wt, initial_cut) = if is_extended {
                let mut cut_idx = 0;
                while cut_idx < ncut && ctime[cut_idx] < original_end {
                    cut_idx += 1;
                }
                if cut_idx < ncut {
                    (ctime[cut_idx], cprob[cut_idx], cut_idx)
                } else {
                    (original_end, 1.0, ncut)
                }
            } else {
                (original_end, 1.0, ncut)
            };

            row.push(i + 1);
            start.push(original_start);
            end.push(current_end);
            wt.push(1.0);
            add.push(0);
            if is_extended && initial_cut < ncut {
                let mut iadd = 0;
                for cut_idx in (initial_cut + 1)..ncut {
                    if keep[cut_idx] {
                        iadd += 1;
                        row.push(i + 1);
                        start.push(ctime[cut_idx - 1]);
                        end.push(ctime[cut_idx]);
                        wt.push(cprob[cut_idx] / temp_wt);
                        add.push(iadd);
                    }
                }
            }
        }
        FineGrayOutput {
            row,
            start,
            end,
            wt,
            add,
        }
    }

    fn assert_output_eq(actual: &FineGrayOutput, expected: &FineGrayOutput) {
        assert_eq!(actual.row, expected.row);
        assert_eq!(actual.add, expected.add);
        assert_eq!(
            actual
                .start
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .start
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .end
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .end
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .wt
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .wt
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_finegray_matches_legacy_extension_weights() {
        let ctime = vec![3.0, 4.0, 6.0, 8.0, 9.0];
        let cprob = vec![
            11.0 / 12.0,
            (11.0 / 12.0) * (8.0 / 10.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0) * (3.0 / 4.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0) * (3.0 / 4.0) * (2.0 / 3.0),
        ];

        let result = compute_finegray(
            &[0.0, 0.0],
            &[4.0, 2.0],
            &ctime,
            &cprob,
            &[true, false],
            &[true, true, true, true, true],
        );

        assert_eq!(result.row, vec![1, 1, 1, 1, 2]);
        assert_eq!(result.start, vec![0.0, 4.0, 6.0, 8.0, 0.0]);
        assert_eq!(result.end, vec![4.0, 6.0, 8.0, 9.0, 2.0]);
        assert_eq!(result.add, vec![0, 1, 2, 3, 0]);

        let expected_wt = [1.0, 5.0 / 6.0, 5.0 / 8.0, 5.0 / 12.0, 1.0];
        assert_eq!(result.wt.len(), expected_wt.len());
        for (&actual, &expected) in result.wt.iter().zip(expected_wt.iter()) {
            assert_close(actual, expected, 1e-12);
        }
    }

    #[test]
    fn test_finegray_preserves_duplicate_and_boundary_cut_semantics() {
        let tstart = vec![0.0; 5];
        let tstop = vec![1.0, 1.5, 4.0, 8.0, 0.5];
        let ctime = vec![1.0, 1.0, 2.0, 4.0, 4.0, 7.0];
        let cprob = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        let extend = vec![true; 5];
        let keep = vec![false, true, false, false, true, false];

        let result = compute_finegray(&tstart, &tstop, &ctime, &cprob, &extend, &keep);

        assert_eq!(result.row, vec![1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5]);
        assert_eq!(
            result.start,
            vec![0.0, 1.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 1.0, 4.0]
        );
        assert_eq!(
            result.end,
            vec![1.0, 1.0, 4.0, 2.0, 4.0, 4.0, 4.0, 8.0, 1.0, 1.0, 4.0]
        );
        assert_eq!(result.add, vec![0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 2]);
        let expected_wt = vec![
            1.0,
            0.9,
            0.6,
            1.0,
            0.6 / 0.8,
            1.0,
            0.6 / 0.7,
            1.0,
            1.0,
            0.9,
            0.6,
        ];
        assert_eq!(result.wt, expected_wt);
    }

    #[test]
    fn test_finegray_optimized_expansion_matches_naive_reference() {
        for seed in 0..512 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let n = rng.usize(0..24);
            let ncut = rng.usize(0..20);
            let mut current_cut = -2.0;
            let ctime: Vec<f64> = (0..ncut)
                .map(|_| {
                    current_cut += rng.usize(0..4) as f64;
                    current_cut
                })
                .collect();
            let cprob: Vec<f64> = (0..ncut).map(|idx| 1.0 / (idx as f64 + 1.0)).collect();
            let keep: Vec<bool> = (0..ncut).map(|_| rng.bool()).collect();
            let mut tstop = Vec::with_capacity(n);
            let mut tstart = Vec::with_capacity(n);
            let mut extend = Vec::with_capacity(n);
            for _ in 0..n {
                let stop = if ncut == 0 {
                    rng.usize(0..10) as f64
                } else {
                    match rng.usize(0..4) {
                        0 => ctime[rng.usize(0..ncut)],
                        1 => ctime[0] - 1.0,
                        2 => ctime[ncut - 1] + 1.0,
                        _ => ctime[rng.usize(0..ncut)] + 0.5,
                    }
                };
                tstop.push(stop);
                tstart.push(stop - rng.usize(0..4) as f64);
                extend.push(rng.bool());
            }

            let actual = compute_finegray(&tstart, &tstop, &ctime, &cprob, &extend, &keep);
            let expected = compute_finegray_naive(&tstart, &tstop, &ctime, &cprob, &extend, &keep);
            assert_output_eq(&actual, &expected);
        }
    }

    #[test]
    fn test_finegray_public_api_rejects_malformed_inputs() {
        assert!(finegray(vec![0.0], vec![], vec![], vec![], vec![true], vec![]).is_err());
        assert!(finegray(vec![2.0], vec![1.0], vec![], vec![], vec![true], vec![]).is_err());
        assert!(
            finegray(
                vec![0.0],
                vec![1.0],
                vec![2.0, 1.0],
                vec![1.0, 1.0],
                vec![true],
                vec![true, true]
            )
            .is_err()
        );
        assert!(
            finegray(
                vec![0.0],
                vec![1.0],
                vec![1.0],
                vec![0.0],
                vec![true],
                vec![true]
            )
            .is_err()
        );
    }
}
