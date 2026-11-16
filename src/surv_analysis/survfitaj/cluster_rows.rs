//! The passes over the clusters that the general Aalen-Johansen influence update makes at
//! every event time, on row-major `rows x clusters` matrices.
//!
//! The plain loops are bound by latency, not memory: each row's sum is one chain of
//! dependent additions, and the hazard updates branch on a weight pattern that is random
//! across the clusters.  Here up to four rows share one pass, so their chains overlap, and
//! the updates select their result instead of branching.  Every sum still runs over the
//! clusters in order from the same start, and every update stores the value the plain
//! loop stores, so the results are bitwise the same.

/// Rows that share one pass over the clusters.
const BLOCK: usize = 4;

/// The Euclidean norm of each of the `norms.len()` rows of `width` values in `rows`.
pub(super) fn row_norms(rows: &[f64], width: usize, norms: &mut [f64]) {
    for (block, norms) in norms.chunks_mut(BLOCK).enumerate() {
        let rows = &rows[block * BLOCK * width..];
        let row = |r: usize| &rows[r * width..(r + 1) * width];
        let sums = match norms.len() {
            1 => padded(square_sums([row(0)])),
            2 => padded(square_sums([row(0), row(1)])),
            3 => padded(square_sums([row(0), row(1), row(2)])),
            _ => square_sums([row(0), row(1), row(2), row(3)]),
        };
        for (norm, sum) in norms.iter_mut().zip(sums) {
            *norm = sum.sqrt();
        }
    }
}

/// `area += delta * u`, row by row, with `norms` the Euclidean norms of the new rows of
/// `area` (both matrices have `norms.len()` rows of `width` values).
pub(super) fn integrate_rows(
    area: &mut [f64],
    u: &[f64],
    width: usize,
    delta: f64,
    norms: &mut [f64],
) {
    for (block, norms) in norms.chunks_mut(BLOCK).enumerate() {
        let start = block * BLOCK * width;
        let (area, u) = (&mut area[start..], &u[start..]);
        let row = |r: usize| &u[r * width..(r + 1) * width];
        let sums = match norms.len() {
            1 => padded(integrated(split_rows(area, width), [row(0)], delta)),
            2 => padded(integrated(split_rows(area, width), [row(0), row(1)], delta)),
            3 => padded(integrated(
                split_rows(area, width),
                [row(0), row(1), row(2)],
                delta,
            )),
            _ => integrated(
                split_rows(area, width),
                [row(0), row(1), row(2), row(3)],
                delta,
            ),
        };
        for (norm, sum) in norms.iter_mut().zip(sums) {
            *norm = sum.sqrt();
        }
    }
}

/// `values -= weight * htemp` wherever the weight at risk is positive.
pub(super) fn remove_hazard(values: &mut [f64], weights: &[f64], htemp: f64) {
    for (value, &weight) in values.iter_mut().zip(weights) {
        let updated = *value - weight * htemp;
        *value = if weight > 0.0 { updated } else { *value };
    }
}

/// `from += term` and `to -= term` with `term = weight * probability * htemp`, wherever
/// the weight at risk is positive.
pub(super) fn move_hazard(
    from: &mut [f64],
    to: &mut [f64],
    weights: &[f64],
    probability: f64,
    htemp: f64,
) {
    for ((source, target), &weight) in from.iter_mut().zip(to).zip(weights) {
        let term = weight * probability * htemp;
        let (left, arrived) = (*source + term, *target - term);
        let at_risk = weight > 0.0;
        *source = if at_risk { left } else { *source };
        *target = if at_risk { arrived } else { *target };
    }
}

/// The sum of squares of each row, from `-0.0` as `Iterator::sum` starts.
fn square_sums<const N: usize>(rows: [&[f64]; N]) -> [f64; N] {
    let width = rows[0].len();
    let rows = rows.map(|row| &row[..width]);
    let mut sums = [-0.0; N];
    for g in 0..width {
        for (sum, row) in sums.iter_mut().zip(&rows) {
            *sum += row[g] * row[g];
        }
    }
    sums
}

/// `area += delta * u` for each row, and the sum of squares of each new row from `0.0`.
fn integrated<const N: usize>(area: [&mut [f64]; N], u: [&[f64]; N], delta: f64) -> [f64; N] {
    let width = u[0].len();
    let u = u.map(|row| &row[..width]);
    let mut area = area.map(|row| &mut row[..width]);
    let mut sums = [0.0; N];
    for g in 0..width {
        for ((sum, area), u) in sums.iter_mut().zip(&mut area).zip(&u) {
            area[g] += delta * u[g];
            *sum += area[g] * area[g];
        }
    }
    sums
}

/// The first `N` rows of `width` values of `values`.
fn split_rows<const N: usize>(values: &mut [f64], width: usize) -> [&mut [f64]; N] {
    let mut rest = values;
    std::array::from_fn(|_| {
        let (row, tail) = std::mem::take(&mut rest).split_at_mut(width);
        rest = tail;
        row
    })
}

fn padded<const N: usize>(sums: [f64; N]) -> [f64; BLOCK] {
    let mut block = [0.0; BLOCK];
    block[..N].copy_from_slice(&sums);
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(seed: u64, n: usize) -> Vec<f64> {
        let special = [0.0, -0.0, f64::NAN, f64::INFINITY, -1e300, 1e-310, 2.5];
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let draw = state >> 11;
                if draw.is_multiple_of(9) {
                    special[(draw / 9) as usize % special.len()]
                } else {
                    (draw % 2001) as f64 / 1000.0 - 1.0
                }
            })
            .collect()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn row_passes_match_the_plain_loops_bitwise() {
        for nrow in 0..10 {
            for width in 0..8 {
                let seed = (nrow * 8 + width) as u64;
                let u = values(seed, nrow * width);
                let mut norms = vec![0.0; nrow];
                row_norms(&u, width, &mut norms);
                let reference: Vec<f64> = (0..nrow)
                    .map(|j| {
                        u[j * width..(j + 1) * width]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect();
                assert_eq!(bits(&norms), bits(&reference));

                let delta = [0.75, 0.0, f64::INFINITY][nrow % 3];
                let mut area = values(seed + 100, nrow * width);
                let mut reference_area = area.clone();
                integrate_rows(&mut area, &u, width, delta, &mut norms);
                let reference: Vec<f64> = (0..nrow)
                    .map(|j| {
                        let mut temp = 0.0;
                        for (a, &x) in reference_area[j * width..(j + 1) * width]
                            .iter_mut()
                            .zip(&u[j * width..(j + 1) * width])
                        {
                            *a += delta * x;
                            temp += *a * *a;
                        }
                        temp.sqrt()
                    })
                    .collect();
                assert_eq!(bits(&area), bits(&reference_area));
                assert_eq!(bits(&norms), bits(&reference));
            }
        }
    }

    #[test]
    fn hazard_updates_match_the_branching_loops_bitwise() {
        for (seed, n) in [(1, 0), (2, 1), (3, 17), (4, 64)] {
            let weights = values(seed, n);
            let htemp = [0.125, -0.0, f64::NAN, 3.0][seed as usize % 4];
            let probability = [0.5, f64::INFINITY][seed as usize % 2];

            let mut c = values(seed + 10, n);
            let mut reference = c.clone();
            remove_hazard(&mut c, &weights, htemp);
            for (x, &w) in reference.iter_mut().zip(&weights) {
                if w > 0.0 {
                    *x -= w * htemp;
                }
            }
            assert_eq!(bits(&c), bits(&reference));

            let (mut from, mut to) = (values(seed + 20, n), values(seed + 30, n));
            let (mut reference_from, mut reference_to) = (from.clone(), to.clone());
            move_hazard(&mut from, &mut to, &weights, probability, htemp);
            for ((x, y), &w) in reference_from
                .iter_mut()
                .zip(&mut reference_to)
                .zip(&weights)
            {
                if w > 0.0 {
                    let term = w * probability * htemp;
                    *x += term;
                    *y -= term;
                }
            }
            assert_eq!(bits(&from), bits(&reference_from));
            assert_eq!(bits(&to), bits(&reference_to));
        }
    }
}
