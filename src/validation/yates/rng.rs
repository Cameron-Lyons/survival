//! R's default Mersenne-Twister and inversion normal generator, which the
//! Yates simulation draws from so `seed` reproduces R's `set.seed` stream.
pub(super) use crate::internal::rng::RNormal;
