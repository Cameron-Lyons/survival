//! Port of R survival's `src/pystep.c`: how long a subject stays in the
//! current cell of a multi-way table before crossing a cutpoint, and the
//! linear index of that cell.  Shared by `pyears1` and `pyears3b`.

/// The time spent in the current cell and where that cell is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PystepResult {
    /// Time spent in the cell (or off the table when `index` is `None`).
    pub time: f64,
    /// Linear (column-major) index of the cell, `None` when off table.
    pub index: Option<usize>,
    /// Second index for the linear interpolation of old-style US tables.
    pub index2: usize,
    /// Weight of `index` in that interpolation (1 without interpolation).
    pub weight: f64,
}

/// One dimension of the table as `pystep` sees it.
///
/// `factors[i]` is 1 for a factor dimension, 0 for a continuous one and
/// `>= 2` for the interpolated calendar-year axis of old US tables (with
/// `1 + (factors[i] - 1) * dims[i]` cutpoints).  For `edge == false` the
/// cutpoints of a continuous dimension hold one extra upper limit
/// (`dims[i] + 1` values) and time outside them is reported off table; for
/// `edge == true` the table extends infinitely at both ends.
pub struct PystepTable<'a> {
    pub factors: &'a [i32],
    pub dims: &'a [usize],
    pub cuts: &'a [&'a [f64]],
    pub edge: bool,
}

/// `pystep(nc, index, index2, wt, data, fac, dims, cuts, step, edge)`.
pub fn pystep(table: &PystepTable<'_>, data: &[f64], step: f64) -> PystepResult {
    let mut index = 0isize;
    let mut index2 = 0isize;
    let mut stride = 1isize;
    let mut weight = 1.0;
    let mut shortfall = 0.0;
    let mut max_time = step;

    for (i, (&factor, &dim)) in table.factors.iter().zip(table.dims).enumerate() {
        if factor == 1 {
            index += (data[i] as isize - 1) * stride;
        } else {
            let cuts = table.cuts[i];
            let cut_count = if factor > 1 {
                1 + (factor as usize - 1) * dim
            } else {
                dim
            };
            let mut j = cuts[..cut_count].partition_point(|&cut| data[i] >= cut);

            if j == 0 {
                // Less than the first cutpoint.
                let temp = cuts[0] - data[i];
                if !table.edge && temp > shortfall {
                    shortfall = temp.min(step);
                }
                max_time = max_time.min(temp);
            } else if j == cut_count {
                // Beyond the last cutpoint.
                if !table.edge {
                    let temp = cuts[j] - data[i];
                    if temp <= 0.0 {
                        shortfall = step;
                    } else {
                        max_time = max_time.min(temp);
                    }
                }
                j = if factor > 1 { dim - 1 } else { j - 1 };
            } else {
                max_time = max_time.min(cuts[j] - data[i]);
                j -= 1;
                if factor > 1 {
                    weight = 1.0 - (j % factor as usize) as f64 / factor as f64;
                    j /= factor as usize;
                    index2 = stride;
                }
            }
            index += j as isize * stride;
        }
        stride *= dim as isize;
    }

    index2 += index;
    if shortfall == 0.0 {
        PystepResult {
            time: max_time,
            index: Some(index as usize),
            index2: index2 as usize,
            weight,
        }
    } else {
        PystepResult {
            time: shortfall,
            index: None,
            index2: index2 as usize,
            weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table<'a>(
        factors: &'a [i32],
        dims: &'a [usize],
        cuts: &'a [&'a [f64]],
        edge: bool,
    ) -> PystepTable<'a> {
        PystepTable {
            factors,
            dims,
            cuts,
            edge,
        }
    }

    #[test]
    fn assigns_elapsed_time_to_the_current_cell() {
        let cuts: [&[f64]; 1] = [&[0.0, 1.0]];
        let result = pystep(&table(&[0], &[2], &cuts, true), &[0.25], 1.0);
        assert_eq!(
            result,
            PystepResult {
                time: 0.75,
                index: Some(0),
                index2: 0,
                weight: 1.0
            }
        );
    }

    #[test]
    fn reports_time_below_and_above_observed_cuts_as_off_table() {
        let cuts: [&[f64]; 1] = [&[0.0, 10.0, 20.0, 30.0]];
        let strict = table(&[0], &[3], &cuts, false);
        let below = pystep(&strict, &[-5.0], 10.0);
        let final_cell = pystep(&strict, &[25.0], 10.0);
        let above = pystep(&strict, &[30.0], 5.0);

        assert_eq!((below.time, below.index), (5.0, None));
        assert_eq!((final_cell.time, final_cell.index), (5.0, Some(2)));
        assert_eq!((above.time, above.index), (5.0, None));
    }

    #[test]
    fn extends_infinitely_at_the_edges_for_rate_tables() {
        let cuts: [&[f64]; 2] = [&[0.0, 10.0, 20.0], &[]];
        let open = table(&[0, 1], &[3, 2], &cuts, true);
        let below = pystep(&open, &[-5.0, 2.0], 100.0);
        assert_eq!((below.time, below.index), (5.0, Some(3)));
        let above = pystep(&open, &[25.0, 1.0], 100.0);
        assert_eq!((above.time, above.index), (100.0, Some(2)));
    }

    #[test]
    fn interpolates_old_style_us_year_axes() {
        // Two decades, interpolated in 10 steps: 21 cutpoints.
        let year_cuts: Vec<f64> = (0..21).map(|i| i as f64 * 365.25).collect();
        let cuts: [&[f64]; 1] = [&year_cuts];
        let result = pystep(
            &table(&[10], &[2], &cuts, true),
            &[3.0 * 365.25 + 1.0],
            5000.0,
        );
        assert_eq!(result.index, Some(0));
        assert_eq!(result.index2, 1);
        assert!((result.weight - 0.7).abs() < 1e-12);
        assert!((result.time - (365.25 - 1.0)).abs() < 1e-9);
    }
}
