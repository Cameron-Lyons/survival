use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::{Bound, RangeBounds};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static SEED_SEQUENCE: AtomicU64 = AtomicU64::new(0x6a09_e667_f3bc_c909);

/// Small non-cryptographic generator for simulation, resampling, and model initialization.
///
/// The generator uses WyRand and Lemire's unbiased bounded-integer mapping. Seeded streams retain
/// the deterministic behavior the public statistical APIs rely on, while `new` mixes process-local
/// timing, thread identity, and a monotonic sequence to create independent streams.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rng {
    state: u64,
}

impl Default for Rng {
    fn default() -> Self {
        Self::new()
    }
}

impl Rng {
    #[inline]
    pub(crate) const fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub(crate) fn seed(&mut self, seed: u64) {
        self.state = seed;
    }

    pub(crate) fn new() -> Self {
        let mut hasher = DefaultHasher::new();
        Instant::now().hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);
        SEED_SEQUENCE
            .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
            .hash(&mut hasher);
        Self::with_seed(hasher.finish())
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        const WY_CONST_0: u64 = 0x2d35_8dcc_aa6c_78a5;
        const WY_CONST_1: u64 = 0x8bb8_4b93_962e_acc9;

        self.state = self.state.wrapping_add(WY_CONST_0);
        let product = u128::from(self.state) * u128::from(self.state ^ WY_CONST_1);
        (product as u64) ^ (product >> 64) as u64
    }

    #[inline]
    fn bounded_u64(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let mut random = self.next_u64();
        let mut product = u128::from(random) * u128::from(bound);
        let mut low = product as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                random = self.next_u64();
                product = u128::from(random) * u128::from(bound);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    #[inline]
    fn bounded_usize(&mut self, bound: usize) -> usize {
        #[cfg(target_pointer_width = "64")]
        {
            self.bounded_u64(bound as u64) as usize
        }
        #[cfg(target_pointer_width = "32")]
        {
            let bound = bound as u32;
            let mut random = self.next_u64() as u32;
            let mut product = u64::from(random) * u64::from(bound);
            let mut low = product as u32;
            if low < bound {
                let threshold = bound.wrapping_neg() % bound;
                while low < threshold {
                    random = self.next_u64() as u32;
                    product = u64::from(random) * u64::from(bound);
                    low = product as u32;
                }
            }
            (product >> 32) as usize
        }
    }

    /// Generate a uniformly distributed integer in the supplied range.
    #[inline]
    pub(crate) fn usize(&mut self, range: impl RangeBounds<usize>) -> usize {
        let empty_range = || {
            panic!(
                "empty usize range: {:?}..{:?}",
                range.start_bound(),
                range.end_bound()
            )
        };
        let low = match range.start_bound() {
            Bound::Unbounded => usize::MIN,
            Bound::Included(&value) => value,
            Bound::Excluded(&value) => value.checked_add(1).unwrap_or_else(empty_range),
        };
        let high = match range.end_bound() {
            Bound::Unbounded => usize::MAX,
            Bound::Included(&value) => value,
            Bound::Excluded(&value) => value.checked_sub(1).unwrap_or_else(empty_range),
        };
        if low > high {
            empty_range();
        }
        if low == usize::MIN && high == usize::MAX {
            self.next_u64() as usize
        } else {
            let length = high.wrapping_sub(low).wrapping_add(1);
            low.wrapping_add(self.bounded_usize(length))
        }
    }

    /// Generate a uniformly distributed `u64` in the supplied range.
    #[inline]
    pub(crate) fn u64(&mut self, range: impl RangeBounds<u64>) -> u64 {
        let empty_range = || {
            panic!(
                "empty u64 range: {:?}..{:?}",
                range.start_bound(),
                range.end_bound()
            )
        };
        let low = match range.start_bound() {
            Bound::Unbounded => u64::MIN,
            Bound::Included(&value) => value,
            Bound::Excluded(&value) => value.checked_add(1).unwrap_or_else(empty_range),
        };
        let high = match range.end_bound() {
            Bound::Unbounded => u64::MAX,
            Bound::Included(&value) => value,
            Bound::Excluded(&value) => value.checked_sub(1).unwrap_or_else(empty_range),
        };
        if low > high {
            empty_range();
        }
        if low == u64::MIN && high == u64::MAX {
            self.next_u64()
        } else {
            let length = high.wrapping_sub(low).wrapping_add(1);
            low.wrapping_add(self.bounded_u64(length))
        }
    }

    #[inline]
    pub(crate) fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 0
    }

    #[inline]
    pub(crate) fn f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1_u64 << 63) as f64;
        loop {
            let value = (self.next_u64() >> 1) as f64 * SCALE;
            if value < 1.0 {
                return value;
            }
        }
    }

    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in 1..values.len() {
            let swap_index = self.usize(..=index);
            values.swap(index, swap_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn seeded_streams_are_reproducible() {
        let mut left = Rng::with_seed(42);
        let mut right = Rng::with_seed(42);
        for _ in 0..128 {
            assert_eq!(left.u64(..), right.u64(..));
        }
    }

    #[test]
    fn bounded_values_respect_exclusive_and_inclusive_ranges() {
        let mut rng = Rng::with_seed(7);
        for _ in 0..1_000 {
            assert!((3..11).contains(&rng.usize(3..11)));
            assert!((4..=9).contains(&rng.usize(4..=9)));
            assert!((10..20).contains(&rng.u64(10..20)));
        }
        assert_eq!(rng.usize(5..=5), 5);
    }

    #[test]
    fn floating_values_are_in_the_half_open_unit_interval() {
        let mut rng = Rng::with_seed(99);
        for _ in 0..1_000 {
            let value = rng.f64();
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn shuffle_preserves_all_values() {
        let mut rng = Rng::with_seed(123);
        let mut values: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut values);
        values.sort_unstable();
        assert_eq!(values, (0..100).collect::<Vec<_>>());
    }
}
