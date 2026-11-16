//! Row orders by stratum and time.
//!
//! Sorting packed integer keys, rather than row indices through a comparator
//! that indexes the data, keeps the sort's working set contiguous and lets a
//! stable sort merge each run of tied times in one step. Tied times are the
//! norm for the rounded follow-up times of survival data.

use crate::core::strata_order::order_within_strata;

/// An integer whose unsigned order is [`f64::total_cmp`]'s order: negative
/// values (sign bit set) have every bit flipped, all others only the sign bit.
#[inline]
pub(crate) fn total_order_key(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

/// Rows ordered by stratum code, then by time under [`f64::total_cmp`], with
/// tied rows in input order: the permutation a stable sort of the row indices
/// by that comparator gives.
pub(crate) fn stratum_time_order(time: &[f64], strata: Option<&[i32]>) -> Vec<usize> {
    match strata {
        None => {
            let mut keyed: Vec<(u64, usize)> =
                time.iter().map(|&t| total_order_key(t)).zip(0..).collect();
            keyed.sort_by_key(|&(key, _)| key);
            keyed.into_iter().map(|(_, row)| row).collect()
        }
        // A 32-bit row index keeps each entry at 16 bytes.
        Some(strata) if u32::try_from(time.len()).is_ok() => {
            debug_assert_eq!(strata.len(), time.len());
            let mut keyed: Vec<(i32, u64, u32)> = strata
                .iter()
                .zip(time)
                .zip(0..)
                .map(|((&code, &t), row)| (code, total_order_key(t), row))
                .collect();
            keyed.sort_by_key(|&(code, key, _)| (code, key));
            keyed.into_iter().map(|(_, _, row)| row as usize).collect()
        }
        // Rows beyond 32-bit indices sort through the comparator instead.
        Some(strata) => order_within_strata(strata, |a, b| time[a].total_cmp(&time[b])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::rng::Rng;

    /// The comparator sort `CoxFitBuilder::build` used before the packed keys.
    fn indirect_order(time: &[f64], strata: Option<&[i32]>) -> Vec<usize> {
        let mut order: Vec<usize> = (0..time.len()).collect();
        order.sort_by(|&lhs, &rhs| {
            strata
                .map_or(std::cmp::Ordering::Equal, |c| c[lhs].cmp(&c[rhs]))
                .then_with(|| time[lhs].total_cmp(&time[rhs]))
                .then_with(|| lhs.cmp(&rhs))
        });
        order
    }

    const EDGE_VALUES: [f64; 12] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::MIN_POSITIVE / 4.0,
        -f64::MIN_POSITIVE / 4.0,
        f64::MAX,
        f64::MIN,
    ];

    #[test]
    fn key_order_is_total_cmp_order() {
        let payload_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let mut values = EDGE_VALUES.to_vec();
        values.extend([payload_nan, -payload_nan, 2.5, -2.5, 1e-300, -1e-300]);
        for &a in &values {
            for &b in &values {
                assert_eq!(
                    total_order_key(a).cmp(&total_order_key(b)),
                    a.total_cmp(&b),
                    "{a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn packed_order_matches_the_indirect_comparator_sort() {
        let mut rng = Rng::with_seed(20_261_115);
        let code_sets: [&[i32]; 3] = [&[0], &[3, -1, 7], &[i32::MIN, -5, 0, i32::MAX]];
        for case in 0..400 {
            let n = rng.usize(0..60);
            let distinct = rng.usize(1..12);
            let time: Vec<f64> = (0..n)
                .map(|_| match rng.usize(0..8) {
                    0 => EDGE_VALUES[rng.usize(0..EDGE_VALUES.len())],
                    1 => rng.f64() * 100.0 - 50.0,
                    _ => (rng.usize(0..distinct) as f64) / 4.0,
                })
                .collect();
            let codes = code_sets[case % code_sets.len()];
            let strata: Vec<i32> = (0..n).map(|_| codes[rng.usize(0..codes.len())]).collect();
            assert_eq!(stratum_time_order(&time, None), indirect_order(&time, None));
            assert_eq!(
                stratum_time_order(&time, Some(&strata)),
                indirect_order(&time, Some(&strata))
            );
        }
    }

    #[test]
    fn presorted_reversed_and_empty_inputs() {
        let time: Vec<f64> = (0..50).map(|i| f64::from(i / 3)).collect();
        let reversed: Vec<f64> = time.iter().rev().copied().collect();
        let strata: Vec<i32> = (0..50).map(|i| i % 2).collect();
        for time in [&time, &reversed] {
            assert_eq!(stratum_time_order(time, None), indirect_order(time, None));
            assert_eq!(
                stratum_time_order(time, Some(&strata)),
                indirect_order(time, Some(&strata))
            );
        }
        assert!(stratum_time_order(&[], None).is_empty());
        assert!(stratum_time_order(&[], Some(&[])).is_empty());
        assert_eq!(stratum_time_order(&[2.0], Some(&[4])), vec![0]);
    }
}
