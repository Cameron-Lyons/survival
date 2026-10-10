//! Expected survival of one or more cohorts from a rate table: a port of R
//! survival's `src/pyears3b.c` and of `R/survexp.fit.R`, the routine
//! `survexp` calls once the model frame has been matched to the table.

use super::match_ratetable::align_us_year_axis_validated;
use super::pystep::{PystepTable, pystep_cell};
use super::ratetable::RateTable;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use ndarray::Array2;

/// The per-interval output of `pyears3b.c`, `ntime x ngroup`.
struct Pyears3bOutput {
    /// Conditional survival over each interval (before `cumprod`).
    surv: Array2<f64>,
    /// Number of subjects contributing to each cell.
    n: Array2<usize>,
}

/// The subjects handed to `pyears3b`: zero-based curve numbers, starting
/// positions in the table (one row each) and follow-up times.
struct Pyears3bSubjects<'a> {
    group: &'a [usize],
    positions: &'a Array2<f64>,
    y: &'a [f64],
}

/// `pyears3b(death, efac, edims, ecut, expect, grpx, x, y, times, ngrp)`.
///
/// The C code calls `pystep` for every step, searching the cutpoints again
/// even when the subject is still in the same cell of the table, and
/// evaluates both `exp(-cumhaz)` and `exp(-(cumhaz + hazard))` for every
/// output interval.  Here each subject keeps its cell's rate and the
/// cutpoints ending that cell (`pystep_cell`), so the cutpoints are searched
/// once per cell entered while every step still has `pystep`'s length, and
/// carries `exp(-cumhaz)` from one interval to the next.
fn pyears3b(
    conditional: bool,
    table: &PystepTable<'_>,
    rates: &[f64],
    subjects: &Pyears3bSubjects<'_>,
    times: &[f64],
    n_groups: usize,
) -> Pyears3bOutput {
    let Pyears3bSubjects {
        group,
        positions,
        y,
    } = *subjects;
    let n = y.len();
    let ntime = times.len();
    let edim = table.factors.len();
    let mut esurv = Array2::<f64>::zeros((ntime, n_groups));
    let mut wvec = Array2::<f64>::zeros((ntime, n_groups));
    let mut nsurv = Array2::<usize>::zeros((ntime, n_groups));
    let mut data2 = vec![0.0; edim];
    let mut limits = vec![0.0; edim];

    for i in 0..n {
        // `data2` is the subject's current position in the expected table,
        // `cell_rate` the hazard of its cell, `limits` the cutpoints ending
        // that cell and `cell_left` the time until the first of them.
        for (j, value) in data2.iter_mut().enumerate() {
            *value = positions[[i, j]];
        }
        let mut cell_rate = 0.0;
        let mut cell_left = 0.0;
        let mut cumhaz: f64 = 0.0;
        let mut survival = 1.0;
        let mut timeleft = y[i];
        let g = group[i];
        let mut time = 0.0;

        for (j, &output_time) in times.iter().enumerate() {
            if timeleft <= 0.0 {
                break;
            }
            let thiscell = (output_time - time).min(timeleft);

            let mut etime = thiscell;
            let mut hazard = 0.0;
            while etime > 0.0 {
                if cell_left <= 0.0 {
                    let cell = pystep_cell(table, &data2, &mut limits);
                    cell_rate = rates[cell.index.unwrap_or(0)];
                    cell_left = cell.time;
                }
                let dt = etime.min(cell_left);
                hazard += dt * cell_rate;
                // `cell_left` is measured from the new position, as `pystep`
                // would measure it.
                cell_left = f64::INFINITY;
                for ((value, &limit), &factor) in data2.iter_mut().zip(&limits).zip(table.factors) {
                    if factor != 1 {
                        *value += dt;
                        cell_left = cell_left.min(limit - *value);
                    }
                }
                etime -= dt;
            }
            if output_time == 0.0 {
                wvec[[j, g]] = 1.0;
                esurv[[j, g]] = if conditional { 0.0 } else { 1.0 };
            } else if conditional {
                esurv[[j, g]] += hazard * thiscell;
                wvec[[j, g]] += thiscell;
            } else {
                // `survival` is exp(-cumhaz) at the start of the interval.
                let next = (-(cumhaz + hazard)).exp();
                esurv[[j, g]] += next * thiscell;
                wvec[[j, g]] += survival * thiscell;
                survival = next;
            }
            nsurv[[j, g]] += 1;
            cumhaz += hazard;
            time += thiscell;
            timeleft -= thiscell;
        }
    }

    for (value, &weight) in esurv.iter_mut().zip(wvec.iter()) {
        if weight > 0.0 {
            if conditional {
                *value = (-*value / weight).exp();
            } else {
                *value /= weight;
            }
        } else if conditional {
            *value = (-*value).exp();
        }
    }
    Pyears3bOutput {
        surv: esurv,
        n: nsurv,
    }
}

/// The output of `survexp.fit`: one curve per group, `ntime x ngroup`.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvexpFit {
    /// The sorted unique output times.
    pub times: Vec<f64>,
    /// Expected survival at each time (rows) for each group (columns).
    pub surv: Array2<f64>,
    /// Number of subjects contributing to each cell.
    pub n: Array2<usize>,
}

/// `survexp.fit(group, x, y, times, death, ratetable)` (`R/survexp.fit.R`).
///
/// `group` holds zero-based curve numbers, `positions` is the matrix from
/// `match_ratetable` (one row per subject), `y` the follow-up days of each
/// subject (`None` follows everyone to the last time point) and
/// `conditional` is R's `death` argument.
pub fn survexp_fit(
    group: &[usize],
    positions: &Array2<f64>,
    y: Option<&[f64]>,
    times: &[f64],
    conditional: bool,
    ratetable: &RateTable,
) -> SurvivalResult<SurvexpFit> {
    let n = positions.nrows();
    ratetable.validate_positions(positions)?;
    validate_length(n, group.len(), "group")?;
    let n_groups = match group.iter().max() {
        Some(g) => g
            .checked_add(1)
            .ok_or_else(|| SurvivalError::invalid_input("group codes exceed addressable memory"))?,
        None => 0,
    };
    validate_finite(times, "times")?;
    validate_non_negative(times, "times")?;
    let mut times = times.to_vec();
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup();
    if times.is_empty() {
        return Err(SurvivalError::invalid_input("times must not be empty"));
    }
    n_groups
        .checked_mul(times.len())
        .filter(|&cells| cells <= isize::MAX as usize / size_of::<f64>())
        .ok_or_else(|| SurvivalError::invalid_input("group codes exceed addressable memory"))?;
    let y = match y {
        Some(values) => {
            validate_length(n, values.len(), "y")?;
            validate_finite(values, "y")?;
            values.to_vec()
        }
        None => vec![times[times.len() - 1]; n],
    };

    let mut positions = positions.clone();
    align_us_year_axis_validated(ratetable, &mut positions)?;
    let factors = ratetable.factor_flags();
    let cuts = ratetable.cut_slices();
    let table = PystepTable {
        factors: &factors,
        dims: &ratetable.dims,
        cuts: &cuts,
        edge: true,
    };
    let subjects = Pyears3bSubjects {
        group,
        positions: &positions,
        y: &y,
    };
    let Pyears3bOutput { mut surv, n } = pyears3b(
        conditional,
        &table,
        &ratetable.rates,
        &subjects,
        &times,
        n_groups,
    );
    // cumprod down each column, except for a single time point.
    if times.len() > 1 {
        for mut column in surv.columns_mut() {
            let mut running = 1.0;
            for value in column.iter_mut() {
                running *= *value;
                *value = running;
            }
        }
    }
    Ok(SurvexpFit { times, surv, n })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::pystep::pystep;
    use crate::population::ratetable::DimType;
    use crate::population::ratetable_data::survexp_us_table;

    fn constant_table(rates: Vec<f64>, cuts: Vec<f64>) -> RateTable {
        let labels = cuts.iter().map(|c| c.to_string()).collect();
        RateTable::try_new(
            vec![rates.len()],
            vec!["age".into()],
            vec![labels],
            vec![Some(cuts)],
            vec![DimType::Continuous],
            rates,
        )
        .unwrap()
    }

    #[test]
    fn constant_rate_returns_cumulative_survival_and_counts() {
        let table = constant_table(vec![0.1], vec![0.0]);
        let fit = survexp_fit(
            &[0, 0],
            &ndarray::arr2(&[[0.0], [0.0]]),
            Some(&[3.0, 2.0]),
            &[1.0, 2.0, 3.0],
            false,
            &table,
        )
        .unwrap();
        for (k, value) in fit.surv.column(0).iter().enumerate() {
            assert!((value - (-0.1 * (k + 1) as f64).exp()).abs() < 1e-14);
        }
        assert_eq!(fit.n.column(0).to_vec(), vec![2, 2, 1]);
        assert_eq!(fit.times, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cohort_and_conditional_reductions_are_distinct() {
        let table = constant_table(vec![0.1, 0.3], vec![0.0, 1.0]);
        let x = ndarray::arr2(&[[0.0], [1.0]]);
        let cohort = survexp_fit(&[0, 0], &x, Some(&[1.0, 1.0]), &[1.0], false, &table).unwrap();
        let conditional =
            survexp_fit(&[0, 0], &x, Some(&[1.0, 1.0]), &[1.0], true, &table).unwrap();
        let expected_cohort = ((-0.1f64).exp() + (-0.3f64).exp()) / 2.0;
        assert!((cohort.surv[[0, 0]] - expected_cohort).abs() < 1e-14);
        assert!((conditional.surv[[0, 0]] - (-0.2f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn groups_form_separate_columns_and_times_are_sorted_unique() {
        let table = constant_table(vec![0.1, 0.3], vec![0.0, 1.0]);
        let fit = survexp_fit(
            &[0, 1],
            &ndarray::arr2(&[[0.0], [1.0]]),
            None,
            &[2.0, 1.0, 1.0],
            false,
            &table,
        )
        .unwrap();
        assert_eq!(fit.times, vec![1.0, 2.0]);
        assert!((fit.surv[[0, 0]] - (-0.1f64).exp()).abs() < 1e-14);
        assert!((fit.surv[[0, 1]] - (-0.3f64).exp()).abs() < 1e-14);
        assert!((fit.surv[[1, 0]] - (-0.4f64).exp()).abs() < 1e-14);
        assert_eq!(fit.n, ndarray::arr2(&[[1, 1], [1, 1]]));
    }

    #[test]
    fn time_zero_starts_every_curve_at_one() {
        let table = constant_table(vec![0.1], vec![0.0]);
        let fit = survexp_fit(
            &[0],
            &ndarray::arr2(&[[0.0]]),
            Some(&[5.0]),
            &[0.0, 1.0],
            true,
            &table,
        )
        .unwrap();
        assert_eq!(fit.surv[[0, 0]], 1.0);
        assert!((fit.surv[[1, 0]] - (-0.1f64).exp()).abs() < 1e-14);
    }

    /// Each interval's (hazard, length) as `pyears3b.c` accumulates them,
    /// with a `pystep` call at every step.
    fn pyears3b_c_hazards(
        table: &PystepTable<'_>,
        rates: &[f64],
        start: &[f64],
        y: f64,
        times: &[f64],
    ) -> Vec<(f64, f64)> {
        let mut data2 = start.to_vec();
        let (mut time, mut timeleft) = (0.0, y);
        let mut intervals = Vec::new();
        for &output_time in times {
            if timeleft <= 0.0 {
                break;
            }
            let thiscell = (output_time - time).min(timeleft);
            let (mut etime, mut hazard) = (thiscell, 0.0);
            while etime > 0.0 {
                let step = pystep(table, &data2, etime);
                hazard += step.time * rates[step.index.unwrap()];
                for (k, value) in data2.iter_mut().enumerate() {
                    if table.factors[k] != 1 {
                        *value += step.time;
                    }
                }
                etime -= step.time;
            }
            intervals.push((hazard, thiscell));
            time += thiscell;
            timeleft -= thiscell;
        }
        intervals
    }

    #[test]
    fn cached_cells_take_the_c_codes_steps_exactly() {
        let ratetable = survexp_us_table();
        let factors = ratetable.factor_flags();
        let cuts = ratetable.cut_slices();
        let table = PystepTable {
            factors: &factors,
            dims: &ratetable.dims,
            cuts: &cuts,
            edge: true,
        };
        // Output intervals that end inside the yearly age and calendar cells.
        let times: Vec<f64> = (1..=400).map(|k| k as f64 * 29.7).collect();
        let positions = ndarray::arr2(&[
            [-400.25, 1.0, -3652.5],
            [12345.678, 2.0, 1234.5],
            [21000.1, 1.0, 7305.3],
            [36524.9, 2.0, 11322.75],
        ]);
        let y = [11000.3, 6000.0, 9999.99, 11880.0];
        let subjects = Pyears3bSubjects {
            group: &[0, 1, 2, 3],
            positions: &positions,
            y: &y,
        };
        let fit = pyears3b(true, &table, &ratetable.rates, &subjects, &times, 4);
        for (g, start) in positions.rows().into_iter().enumerate() {
            let intervals =
                pyears3b_c_hazards(&table, &ratetable.rates, &start.to_vec(), y[g], &times);
            for (j, (hazard, thiscell)) in intervals.into_iter().enumerate() {
                assert_eq!(fit.surv[[j, g]], (-(hazard * thiscell) / thiscell).exp());
            }
        }
    }

    #[test]
    fn positions_outside_the_table_are_rejected_not_indexed() {
        let table = survexp_us_table();
        let entry = 109.0 * 365.25;
        let attempt = |row: [f64; 3]| {
            survexp_fit(
                &[0],
                &ndarray::arr2(&[row]),
                None,
                &[0.0, 100.0],
                false,
                table,
            )
        };
        assert!(attempt([entry, 2.0, 18262.0]).is_ok());
        for sex in [5.0, 0.0, 1.5, f64::NAN] {
            let message = attempt([entry, sex, 18262.0]).unwrap_err().to_string();
            assert!(message.contains("The variable sex"), "{message}");
        }
        assert!(
            attempt([f64::NAN, 1.0, 18262.0])
                .unwrap_err()
                .to_string()
                .contains("age contains missing values")
        );
    }

    #[test]
    fn rejects_bad_shapes_and_negative_times() {
        let table = constant_table(vec![0.1], vec![0.0]);
        let x = ndarray::arr2(&[[0.0]]);
        assert!(survexp_fit(&[0, 0], &x, None, &[1.0], false, &table).is_err());
        assert!(survexp_fit(&[0], &x, None, &[-1.0], false, &table).is_err());
        assert!(survexp_fit(&[0], &x, None, &[], false, &table).is_err());
        assert!(
            survexp_fit(
                &[0],
                &ndarray::arr2(&[[0.0, 1.0]]),
                None,
                &[1.0],
                false,
                &table
            )
            .is_err()
        );
    }
}
