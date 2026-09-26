//! R's `src/finegray.c`: the interval expansion behind `finegray()`.  A
//! row to extend (a competing event on a subject's last row) keeps its
//! interval up to the next time of the censoring curve and gains a row for
//! each later interval of the curve that is kept, weighted by the curve's
//! probability relative to the row's own.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_finite;
use pyo3::prelude::*;

/// The expanded data; every vector has one entry per output row.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FineGrayOutput {
    /// One-based row of the input each output row came from (R's `row`).
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub end: Vec<f64>,
    /// The probability weight, 1 except on the added intervals.
    #[pyo3(get)]
    pub wt: Vec<f64>,
    /// 0 on a row's own interval, `k` on its `k`-th added one.
    #[pyo3(get)]
    pub add: Vec<usize>,
}

/// `finegray.c`: the `(tstart, tstop]` rows, the censoring curve (the
/// interval that ends at `ctime[j]` has probability `cprob[j]`), which rows
/// to `extend` and which curve intervals to `keep`.
pub fn finegray(
    tstart: &[f64],
    tstop: &[f64],
    ctime: &[f64],
    cprob: &[f64],
    extend: &[bool],
    keep: &[bool],
) -> SurvivalResult<FineGrayOutput> {
    validate_finegray_inputs(tstart, tstop, ctime, cprob, extend, keep)?;
    let ncut = ctime.len();
    let kept: Vec<usize> = (0..ncut).filter(|&idx| keep[idx]).collect();
    // Each row's curve interval (the first `ctime >= tstop`) and its first
    // later kept interval; a row that is not extended adds nothing.  The
    // added rows divide by the probability of that interval.
    let plans: Vec<(usize, usize)> = tstop
        .iter()
        .zip(extend)
        .map(|(&stop, &extended)| {
            if extended {
                let cut = ctime.partition_point(|&time| time < stop);
                (cut, kept.partition_point(|&idx| idx <= cut))
            } else {
                (ncut, kept.len())
            }
        })
        .collect();
    if plans
        .iter()
        .any(|&(cut, first_kept)| first_kept < kept.len() && cprob[cut] == 0.0)
    {
        return Err(SurvivalError::invalid_input(
            "censoring probability is zero before a selected event",
        ));
    }
    let total = tstart.len()
        + plans
            .iter()
            .map(|&(_, first_kept)| kept.len() - first_kept)
            .sum::<usize>();
    let mut out = FineGrayOutput {
        row: Vec::with_capacity(total),
        start: Vec::with_capacity(total),
        end: Vec::with_capacity(total),
        wt: Vec::with_capacity(total),
        add: Vec::with_capacity(total),
    };
    for (i, &(cut, first_kept)) in plans.iter().enumerate() {
        out.row.push(i + 1);
        out.start.push(tstart[i]);
        out.wt.push(1.0);
        out.add.push(0);
        if cut == ncut {
            out.end.push(tstop[i]);
            continue;
        }
        // Extended to the end of its interval, then one row per kept one.
        out.end.push(ctime[cut]);
        for (iadd, &idx) in kept[first_kept..].iter().enumerate() {
            out.row.push(i + 1);
            out.start.push(ctime[idx - 1]);
            out.end.push(ctime[idx]);
            out.wt.push(cprob[idx] / cprob[cut]);
            out.add.push(iadd + 1);
        }
    }
    Ok(out)
}

fn validate_finegray_inputs(
    tstart: &[f64],
    tstop: &[f64],
    ctime: &[f64],
    cprob: &[f64],
    extend: &[bool],
    keep: &[bool],
) -> SurvivalResult<()> {
    let n = tstart.len();
    for (name, len) in [("tstop", tstop.len()), ("extend", extend.len())] {
        if len != n {
            return Err(SurvivalError::invalid_input(format!(
                "{name} length ({len}) must match tstart length ({n})"
            )));
        }
    }
    validate_finite(tstart, "tstart")?;
    validate_finite(tstop, "tstop")?;
    if let Some(idx) = (0..n).find(|&idx| tstart[idx] > tstop[idx]) {
        return Err(SurvivalError::invalid_input(format!(
            "tstart value {} exceeds tstop value {} at index {idx}",
            tstart[idx], tstop[idx]
        )));
    }
    let ncut = ctime.len();
    for (name, len) in [("cprob", cprob.len()), ("keep", keep.len())] {
        if len != ncut {
            return Err(SurvivalError::invalid_input(format!(
                "{name} length ({len}) must match ctime length ({ncut})"
            )));
        }
    }
    validate_finite(ctime, "ctime")?;
    if ctime.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(SurvivalError::invalid_input(
            "ctime must be sorted in nondecreasing order",
        ));
    }
    validate_finite(cprob, "cprob")?;
    if let Some(idx) = cprob.iter().position(|p| !(0.0..=1.0).contains(p)) {
        return Err(SurvivalError::invalid_input(format!(
            "cprob must contain values in [0, 1]; found {} at index {idx}",
            cprob[idx]
        )));
    }
    Ok(())
}

/// Python entry point of [`finegray`].
#[pyfunction(name = "finegray")]
pub fn finegray_py(
    tstart: Vec<f64>,
    tstop: Vec<f64>,
    ctime: Vec<f64>,
    cprob: Vec<f64>,
    extend: Vec<bool>,
    keep: Vec<bool>,
) -> PyResult<FineGrayOutput> {
    Ok(finegray(&tstart, &tstop, &ctime, &cprob, &extend, &keep)?)
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

        let result = finegray(
            &[0.0, 0.0],
            &[4.0, 2.0],
            &ctime,
            &cprob,
            &[true, false],
            &[true, true, true, true, true],
        )
        .unwrap();

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

        let result = finegray(&tstart, &tstop, &ctime, &cprob, &extend, &keep).unwrap();

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
            let mut rng = crate::internal::rng::Rng::with_seed(seed);
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

            let actual = finegray(&tstart, &tstop, &ctime, &cprob, &extend, &keep).unwrap();
            let expected = compute_finegray_naive(&tstart, &tstop, &ctime, &cprob, &extend, &keep);
            assert_output_eq(&actual, &expected);
        }
    }

    #[test]
    fn test_finegray_public_api_rejects_malformed_inputs() {
        let message = |result: SurvivalResult<FineGrayOutput>| result.unwrap_err().to_string();
        assert!(message(finegray(&[0.0], &[], &[], &[], &[true], &[])).contains("tstop length"));
        assert!(
            message(finegray(&[2.0], &[1.0], &[], &[], &[true], &[])).contains("exceeds tstop")
        );
        assert!(
            message(finegray(
                &[0.0],
                &[1.0],
                &[2.0, 1.0],
                &[1.0, 1.0],
                &[true],
                &[true, true]
            ))
            .contains("ctime must be sorted")
        );
        assert!(
            message(finegray(&[0.0], &[1.0], &[1.0], &[-0.1], &[true], &[true]))
                .contains("cprob must contain values")
        );
        assert!(message(finegray(&[f64::NAN], &[1.0], &[], &[], &[true], &[])).contains("tstart"));
        assert!(
            message(finegray(
                &[0.0],
                &[1.0],
                &[1.0],
                &[f64::NAN],
                &[true],
                &[true]
            ))
            .contains("cprob contains non-finite value")
        );
    }

    #[test]
    fn test_finegray_allows_harmless_zero_probability_tails() {
        let result = finegray(
            &[0.0, 0.0],
            &[1.0, 3.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 0.5, 0.0],
            &[true, true],
            &[true, true, true],
        )
        .unwrap();

        assert_eq!(result.row, vec![1, 1, 1, 2]);
        assert_eq!(result.start, vec![0.0, 1.0, 2.0, 0.0]);
        assert_eq!(result.end, vec![1.0, 2.0, 3.0, 3.0]);
        assert_eq!(result.wt, vec![1.0, 0.5, 0.0, 1.0]);
        assert_eq!(result.add, vec![0, 1, 2, 0]);
    }

    #[test]
    fn test_finegray_rejects_zero_probability_before_later_kept_cut() {
        let error = finegray(
            &[0.0],
            &[2.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 0.0, 0.0],
            &[true],
            &[true, false, true],
        )
        .unwrap_err();

        assert!(error.to_string().contains("probability is zero"));
    }
}
