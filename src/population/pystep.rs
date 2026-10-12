#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PystepResult {
    pub time: f64,
    pub index: Option<usize>,
}

pub struct PystepTable<'a> {
    pub factors: &'a [i32],
    pub dims: &'a [usize],
    pub cuts: &'a [&'a [f64]],
    pub edge: bool,
}

pub fn pystep(table: &PystepTable<'_>, data: &[f64], step: f64) -> PystepResult {
    let mut index = 0usize;
    let mut stride = 1usize;
    let mut shortfall = 0.0;
    let mut max_time = step;

    for (i, (&factor, &dim)) in table.factors.iter().zip(table.dims).enumerate() {
        if factor == 1 {
            index += (data[i] as usize - 1) * stride;
        } else {
            let cuts = table.cuts[i];
            let mut j = cuts[..dim].partition_point(|&cut| data[i] >= cut);

            if j == 0 {
                let temp = cuts[0] - data[i];
                if !table.edge && temp > shortfall {
                    shortfall = temp.min(step);
                }
                max_time = max_time.min(temp);
            } else if j == dim {
                if !table.edge {
                    let temp = cuts[j] - data[i];
                    if temp <= 0.0 {
                        shortfall = step;
                    } else {
                        max_time = max_time.min(temp);
                    }
                }
                j -= 1;
            } else {
                max_time = max_time.min(cuts[j] - data[i]);
                j -= 1;
            }
            index += j * stride;
        }
        stride *= dim;
    }

    if shortfall == 0.0 {
        PystepResult {
            time: max_time,
            index: Some(index),
        }
    } else {
        PystepResult {
            time: shortfall,
            index: None,
        }
    }
}

pub fn pystep_cell(table: &PystepTable<'_>, data: &[f64], limits: &mut [f64]) -> PystepResult {
    debug_assert!(table.edge);
    let mut index = 0usize;
    let mut stride = 1usize;
    let mut max_time = f64::INFINITY;

    for (i, (&factor, &dim)) in table.factors.iter().zip(table.dims).enumerate() {
        limits[i] = f64::INFINITY;
        if factor == 1 {
            index += (data[i] as usize - 1) * stride;
        } else {
            let cuts = table.cuts[i];
            let j = cuts[..dim].partition_point(|&cut| data[i] >= cut);
            if j < dim {
                limits[i] = cuts[j];
                max_time = max_time.min(cuts[j] - data[i]);
            }
            index += j.saturating_sub(1) * stride;
        }
        stride *= dim;
    }
    PystepResult {
        time: max_time,
        index: Some(index),
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
                index: Some(0)
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
    fn cell_limits_give_pysteps_step_anywhere_in_the_cell() {
        let cuts: [&[f64]; 3] = [&[0.0, 10.0, 20.0], &[], &[-3.0, 4.5]];
        let open = table(&[0, 1, 0], &[3, 2, 2], &cuts, true);
        let mut limits = [0.0; 3];
        let time_to_limits = |limits: &[f64], data: &[f64]| {
            (0..3).fold(f64::INFINITY, |t, i| t.min(limits[i] - data[i]))
        };
        for data in [
            [-5.0, 2.0, -7.0],
            [0.0, 1.0, 4.5],
            [12.5, 2.0, 0.1],
            [19.9, 1.0, 4.4],
            [25.0, 2.0, -3.0],
        ] {
            let cell = pystep_cell(&open, &data, &mut limits);
            assert_eq!(cell, pystep(&open, &data, f64::INFINITY));
            assert_eq!(cell.time, time_to_limits(&limits, &data));
            for step in [0.05, 1.0, 100.0] {
                let moved = [data[0] + step, data[1], data[2] + step];
                if (0..3).all(|i| moved[i] < limits[i]) {
                    let expected = pystep(&open, &moved, f64::INFINITY);
                    assert_eq!(expected.index, cell.index);
                    assert_eq!(expected.time, time_to_limits(&limits, &moved));
                }
            }
        }
    }
}
