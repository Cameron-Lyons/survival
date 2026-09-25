//! R's default Mersenne-Twister and inversion normal generator. Kept local to
//! Yates simulation so other stochastic APIs retain their existing streams.
use crate::internal::statistical::normal_inverse_cdf;

pub(super) struct RNormal {
    state: [u32; 624],
    index: usize,
}

impl RNormal {
    pub(super) fn new(mut seed: u32) -> Self {
        for _ in 0..50 {
            seed = seed.wrapping_mul(69069).wrapping_add(1);
        }
        // R initializes the position slot before the 624 words, then resets it.
        seed = seed.wrapping_mul(69069).wrapping_add(1);
        let mut state = [0; 624];
        for value in &mut state {
            seed = seed.wrapping_mul(69069).wrapping_add(1);
            *value = seed;
        }
        Self { state, index: 624 }
    }

    fn uniform(&mut self) -> f64 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        (f64::from(y) / 4294967296.0).clamp(0.5 / 4294967295.0, 1.0 - 0.5 / 4294967295.0)
    }

    pub(super) fn normal(&mut self) -> f64 {
        let high = (134217728.0 * self.uniform()).floor();
        normal_inverse_cdf((high + self.uniform()) / 134217728.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_r_default_seed_one_stream() {
        let mut rng = RNormal::new(1);
        for expected in [0.2655086631421, 0.37212389963679016, 0.5728533633518964] {
            assert!((rng.uniform() - expected).abs() < 1e-13);
        }
        let mut rng = RNormal::new(1);
        for expected in [-0.626453810742332, 0.183643324222082, -0.835628612410047] {
            assert!((rng.normal() - expected).abs() < 1e-13);
        }
    }
}
