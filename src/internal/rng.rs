//! Random number generators: [`Rng`], the crate's fast seeded generator
//! (wyrand; its streams are its own), and [`RUniform`]/[`RNormal`], R's
//! default generators, for draws that must reproduce R's `set.seed` streams.

use crate::internal::dist::qnorm;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::{Bound, RangeBounds};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static SEED_SEQUENCE: AtomicU64 = AtomicU64::new(0x6a09_e667_f3bc_c909);

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

/// Words of Mersenne-Twister state (`N` in `RNG.c`).
const MT_N: usize = 624;
/// `M` in `RNG.c`.
const MT_M: usize = 397;
/// `i2_32m1 = 1/(2^32 - 1)` of `RNG.c`'s `fixup`.
const I2_32M1: f64 = 2.328306437080797e-10;

/// R's default uniform generator, `RNGkind("Mersenne-Twister")`, in the
/// state `set.seed(seed)` leaves it (`RNG_Init` in R's `src/main/RNG.c`), so
/// successive [`Self::unif_rand`] values are R's `runif(n)`.  R scrambles
/// the seed as its C type `Int32`, an unsigned 32-bit word: R's
/// `set.seed(s)` for an integer `s` is `RUniform::new(s as u32)`.
#[derive(Clone, Debug)]
pub(crate) struct RUniform {
    mt: [u32; MT_N],
    /// `mti`, the next word to temper; `MT_N` when the state is spent.
    mti: usize,
}

impl RUniform {
    pub(crate) fn new(mut seed: u32) -> Self {
        // Initial scrambling.
        for _ in 0..50 {
            seed = seed.wrapping_mul(69069).wrapping_add(1);
        }
        // `RNG_Init` fills `dummy[0]` (which holds `mti`) before the 624
        // words and `FixupSeeds` then resets it, so one value is skipped.
        seed = seed.wrapping_mul(69069).wrapping_add(1);
        let mut mt = [0; MT_N];
        for word in &mut mt {
            seed = seed.wrapping_mul(69069).wrapping_add(1);
            *word = seed;
        }
        Self { mt, mti: MT_N }
    }

    /// `MT_genrand()`: the next tempered word as a double in `[0, 1)`.
    fn genrand(&mut self) -> f64 {
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        if self.mti >= MT_N {
            // Generate N words at one time.
            for kk in 0..MT_N {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[(kk + 1) % MT_N] & LOWER_MASK);
                let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
                self.mt[kk] = self.mt[(kk + MT_M) % MT_N] ^ (y >> 1) ^ mag;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        // `y * 2^-32`, exact.
        f64::from(y) / 4_294_967_296.0
    }

    /// `unif_rand()`: [`Self::genrand`] through `fixup`, which keeps the
    /// value strictly inside `(0, 1)`.
    pub(crate) fn unif_rand(&mut self) -> f64 {
        let x = self.genrand();
        if x <= 0.0 {
            0.5 * I2_32M1
        } else if 1.0 - x <= 0.0 {
            1.0 - 0.5 * I2_32M1
        } else {
            x
        }
    }
}

/// R's default normal generator, `norm_rand()` with
/// `RNGkind(normal.kind = "Inversion")`, drawing from [`RUniform`]: after
/// `RNormal::new(seed)` successive [`Self::normal`] values are R's
/// `set.seed(seed); rnorm(n)`.
#[derive(Clone, Debug)]
pub(crate) struct RNormal {
    uniform: RUniform,
}

impl RNormal {
    /// See [`RUniform::new`] for how R's integer seed maps to `seed`.
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            uniform: RUniform::new(seed),
        }
    }

    /// `norm_rand()`: `qnorm` of a uniform refined by a second draw, since
    /// one `unif_rand()` alone is not of high enough precision.
    pub(crate) fn normal(&mut self) -> f64 {
        const BIG: f64 = 134_217_728.0; // 2^27
        let u = (BIG * self.uniform.unif_rand()).floor() + self.uniform.unif_rand();
        qnorm(u / BIG, true, false)
    }
}

#[cfg(test)]
mod tests {
    use super::{RNormal, RUniform, Rng};

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

    #[test]
    fn r_uniform_matches_r_set_seed_streams() {
        // set.seed(1); runif(3)
        let mut rng = RUniform::new(1);
        for expected in [
            0.265_508_663_142_1,
            0.372_123_899_636_790_16,
            0.572_853_363_351_896_4,
        ] {
            assert_eq!(rng.unif_rand(), expected);
        }
        // set.seed(-5); runif(2): R scrambles the seed as an unsigned word.
        let mut rng = RUniform::new(-5_i32 as u32);
        for expected in [0.734_594_767_913_222_3, 0.339_692_771_434_783_94] {
            assert_eq!(rng.unif_rand(), expected);
        }
        // set.seed(2147483647); runif(1)
        assert_eq!(
            RUniform::new(2_147_483_647).unif_rand(),
            0.689_667_426_748_201_3
        );
    }

    #[test]
    fn r_uniform_regenerates_its_state_after_624_draws() {
        // set.seed(1); runif(1250)[c(624, 625, 1249, 1250)]
        let mut rng = RUniform::new(1);
        let draws: Vec<f64> = (0..1250).map(|_| rng.unif_rand()).collect();
        assert_eq!(
            [draws[623], draws[624], draws[1248], draws[1249]],
            [
                0.136_487_586_190_924_05,
                0.324_865_140_486_508_6,
                0.754_777_192_836_627_4,
                0.065_966_630_121_693_02,
            ]
        );
    }

    #[test]
    fn r_normal_matches_r_set_seed_stream() {
        // set.seed(1); rnorm(3)
        let mut rng = RNormal::new(1);
        for expected in [
            -0.626_453_810_742_332_4,
            0.183_643_324_222_082_24,
            -0.835_628_612_410_047_2,
        ] {
            assert_eq!(rng.normal(), expected);
        }
    }
}
