use burn::prelude::Backend;
use burn::tensor::Tensor;
use fastrand::Rng;
use std::f64::consts::SQRT_2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrainValidationSplit {
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct EarlyStopping<T> {
    best_metric: f64,
    best_state: Option<T>,
    patience: Option<usize>,
    epochs_without_improvement: usize,
}

impl<T> EarlyStopping<T> {
    pub(crate) fn new(patience: Option<usize>) -> Self {
        Self {
            best_metric: f64::INFINITY,
            best_state: None,
            patience,
            epochs_without_improvement: 0,
        }
    }

    pub(crate) fn record<F>(&mut self, metric: f64, build_state: F)
    where
        F: FnOnce() -> T,
    {
        if metric < self.best_metric {
            self.best_metric = metric;
            self.epochs_without_improvement = 0;
            self.best_state = Some(build_state());
        } else {
            self.epochs_without_improvement += 1;
        }
    }

    pub(crate) fn should_stop(&self) -> bool {
        self.patience
            .is_some_and(|patience| self.epochs_without_improvement >= patience)
    }

    pub(crate) fn into_best_state(self) -> Option<T> {
        self.best_state
    }
}

#[inline]
pub(crate) fn shuffle_indices(indices: &mut [usize], rng: &mut Rng) {
    for i in (1..indices.len()).rev() {
        let j = rng.usize(0..=i);
        indices.swap(i, j);
    }
}

pub(crate) fn train_validation_split_indices(
    n_obs: usize,
    validation_fraction: f64,
    rng: &mut Rng,
) -> TrainValidationSplit {
    let mut shuffled_indices: Vec<usize> = (0..n_obs).collect();
    shuffle_indices(&mut shuffled_indices, rng);

    let n_val = ((n_obs as f64) * validation_fraction)
        .floor()
        .clamp(0.0, n_obs as f64) as usize;
    let n_train = n_obs.saturating_sub(n_val);

    TrainValidationSplit {
        train_indices: shuffled_indices[..n_train].to_vec(),
        val_indices: shuffled_indices[n_train..].to_vec(),
    }
}

pub(crate) fn shuffled_epoch_indices(train_indices: &[usize], rng: &mut Rng) -> Vec<usize> {
    let mut epoch_indices = train_indices.to_vec();
    shuffle_indices(&mut epoch_indices, rng);
    epoch_indices
}

#[inline]
pub(crate) fn gelu_cpu(x: f64) -> f64 {
    x * 0.5 * (1.0 + crate::internal::statistical::erf(x / SQRT_2))
}

#[inline]
pub(crate) fn layer_norm_cpu(x: &[f64], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / n as f64;
    let std = (var + eps as f64).sqrt();

    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let g = if i < gamma.len() {
                gamma[i] as f64
            } else {
                1.0
            };
            let b = if i < beta.len() { beta[i] as f64 } else { 0.0 };
            (xi - mean) / std * g + b
        })
        .collect()
}

pub(crate) fn tensor_to_vec_f32<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    let [rows, cols] = t.dims();
    let data = t.into_data();
    data.to_vec().unwrap_or_else(|_| vec![0.0; rows * cols])
}

#[inline]
pub(crate) fn linear_forward(
    x: &[f64],
    w: &[f32],
    b: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; out_dim];
    for j in 0..out_dim {
        let mut sum = if j < b.len() { b[j] as f64 } else { 0.0 };
        for k in 0..in_dim.min(x.len()) {
            sum += x[k] * w[j * in_dim + k] as f64;
        }
        result[j] = sum;
    }
    result
}

#[inline]
pub(crate) fn relu_vec(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

pub(crate) fn compute_duration_bins(times: &[f64], num_durations: usize) -> (Vec<usize>, Vec<f64>) {
    let n = times.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cuts = Vec::with_capacity(num_durations + 1);
    cuts.push(0.0);

    for i in 1..num_durations {
        let idx = (i * n / num_durations).min(n - 1);
        cuts.push(times[sorted_indices[idx]]);
    }
    cuts.push(times[sorted_indices[n - 1]] * 1.001);

    let duration_bins: Vec<usize> = times
        .iter()
        .map(|&t| {
            match cuts[1..]
                .binary_search_by(|cut| cut.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(idx) => (idx + 1).min(num_durations - 1),
                Err(idx) => idx.min(num_durations - 1),
            }
        })
        .collect();

    (duration_bins, cuts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let x = vec![1.0, 2.0];
        let w = vec![0.5f32, 0.5, 0.25, 0.25];
        let b = vec![0.1f32, 0.2];
        let result = linear_forward(&x, &w, &b, 2, 2);
        assert!((result[0] - 1.6).abs() < 1e-6);
        assert!((result[1] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_relu_vec() {
        let mut x = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        relu_vec(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_compute_duration_bins() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (bins, cuts) = compute_duration_bins(&times, 5);

        assert_eq!(bins.len(), 10);
        assert_eq!(cuts.len(), 6);

        for &bin in &bins {
            assert!(bin < 5);
        }
    }

    #[test]
    fn test_train_validation_split_is_reproducible() {
        let mut rng_a = Rng::with_seed(42);
        let mut rng_b = Rng::with_seed(42);

        let split_a = train_validation_split_indices(10, 0.2, &mut rng_a);
        let split_b = train_validation_split_indices(10, 0.2, &mut rng_b);

        assert_eq!(split_a, split_b);
        assert_eq!(split_a.train_indices.len(), 8);
        assert_eq!(split_a.val_indices.len(), 2);
    }

    #[test]
    fn test_early_stopping_tracks_best_metric() {
        let mut early_stopping = EarlyStopping::new(Some(2));
        early_stopping.record(3.0, || 30);
        early_stopping.record(2.0, || 20);
        early_stopping.record(2.5, || 25);
        early_stopping.record(2.6, || 26);

        assert!(early_stopping.should_stop());
        assert_eq!(early_stopping.into_best_state(), Some(20));
    }
}
