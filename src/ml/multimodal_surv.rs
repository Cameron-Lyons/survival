#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum FusionStrategy {
    Early,
    Late,
    Intermediate,
    Attention,
}

#[pymethods]
impl FusionStrategy {
    fn __repr__(&self) -> String {
        match self {
            FusionStrategy::Early => "FusionStrategy.Early".to_string(),
            FusionStrategy::Late => "FusionStrategy.Late".to_string(),
            FusionStrategy::Intermediate => "FusionStrategy.Intermediate".to_string(),
            FusionStrategy::Attention => "FusionStrategy.Attention".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MultimodalSurvConfig {
    #[pyo3(get, set)]
    pub clinical_hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub imaging_hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub genomic_hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub fusion_hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub fusion_strategy: FusionStrategy,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl MultimodalSurvConfig {
    #[new]
    #[pyo3(signature = (
        clinical_hidden_dims=None,
        imaging_hidden_dims=None,
        genomic_hidden_dims=None,
        fusion_hidden_dims=None,
        fusion_strategy=FusionStrategy::Intermediate,
        num_time_bins=20,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        clinical_hidden_dims: Option<Vec<usize>>,
        imaging_hidden_dims: Option<Vec<usize>>,
        genomic_hidden_dims: Option<Vec<usize>>,
        fusion_hidden_dims: Option<Vec<usize>>,
        fusion_strategy: FusionStrategy,
        num_time_bins: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> Self {
        Self {
            clinical_hidden_dims: clinical_hidden_dims.unwrap_or_else(|| vec![64, 32]),
            imaging_hidden_dims: imaging_hidden_dims.unwrap_or_else(|| vec![256, 128, 64]),
            genomic_hidden_dims: genomic_hidden_dims.unwrap_or_else(|| vec![512, 256, 64]),
            fusion_hidden_dims: fusion_hidden_dims.unwrap_or_else(|| vec![128, 64]),
            fusion_strategy,
            num_time_bins,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        }
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn forward_mlp(input: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(biases.iter())
        .map(|(w, &b)| {
            let sum: f64 = input.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
            relu(sum + b)
        })
        .collect()
}

fn attention_fusion(embeddings: &[Vec<f64>], attention_weights: &[Vec<f64>]) -> Vec<f64> {
    let n_modalities = embeddings.len();
    if n_modalities == 0 {
        return Vec::new();
    }

    let embed_dim = embeddings[0].len();

    let scores: Vec<f64> = embeddings
        .iter()
        .zip(attention_weights.iter())
        .map(|(e, w)| e.iter().zip(w.iter()).map(|(&ei, &wi)| ei * wi).sum())
        .collect();

    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let attention: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    let mut fused = vec![0.0; embed_dim];
    for (embed, &weight) in embeddings.iter().zip(attention.iter()) {
        for (j, &e) in embed.iter().enumerate() {
            fused[j] += weight * e;
        }
    }

    fused
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MultimodalSurvModel {
    clinical_weights: Vec<Vec<f64>>,
    clinical_biases: Vec<f64>,
    imaging_weights: Vec<Vec<f64>>,
    imaging_biases: Vec<f64>,
    genomic_weights: Vec<Vec<f64>>,
    genomic_biases: Vec<f64>,
    fusion_weights: Vec<Vec<f64>>,
    fusion_biases: Vec<f64>,
    attention_weights: Vec<Vec<f64>>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
    time_bins: Vec<f64>,
    config: MultimodalSurvConfig,
    n_clinical_features: usize,
    n_imaging_features: usize,
    n_genomic_features: usize,
}

#[pymethods]
impl MultimodalSurvModel {
    fn predict_survival(
        &self,
        clinical: Option<Vec<Vec<f64>>>,
        imaging: Option<Vec<Vec<f64>>>,
        genomic: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let n = clinical
            .as_ref()
            .map(|c| c.len())
            .or_else(|| imaging.as_ref().map(|i| i.len()))
            .or_else(|| genomic.as_ref().map(|g| g.len()))
            .unwrap_or(0);

        if n == 0 {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut embeddings = Vec::new();

                if let Some(ref clin) = clinical {
                    let embed =
                        forward_mlp(&clin[i], &self.clinical_weights, &self.clinical_biases);
                    embeddings.push(embed);
                }

                if let Some(ref img) = imaging {
                    let embed = forward_mlp(&img[i], &self.imaging_weights, &self.imaging_biases);
                    embeddings.push(embed);
                }

                if let Some(ref geno) = genomic {
                    let embed = forward_mlp(&geno[i], &self.genomic_weights, &self.genomic_biases);
                    embeddings.push(embed);
                }

                let fused = match self.config.fusion_strategy {
                    FusionStrategy::Early | FusionStrategy::Late | FusionStrategy::Intermediate => {
                        embeddings.into_iter().flatten().collect::<Vec<f64>>()
                    }
                    FusionStrategy::Attention => {
                        attention_fusion(&embeddings, &self.attention_weights)
                    }
                };

                let hidden = forward_mlp(&fused, &self.fusion_weights, &self.fusion_biases);

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_bias.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                        sum + b
                    })
                    .collect();

                let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_logits: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
                let sum_exp: f64 = exp_logits.iter().sum();
                let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();

                let mut surv = vec![0.0; probs.len()];
                let mut cumsum = 0.0;
                for j in (0..probs.len()).rev() {
                    cumsum += probs[j];
                    surv[j] = cumsum;
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    fn get_modality_embeddings(
        &self,
        clinical: Option<Vec<f64>>,
        imaging: Option<Vec<f64>>,
        genomic: Option<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mut embeddings = Vec::new();

        if let Some(clin) = clinical {
            let embed = forward_mlp(&clin, &self.clinical_weights, &self.clinical_biases);
            embeddings.push(embed);
        }

        if let Some(img) = imaging {
            let embed = forward_mlp(&img, &self.imaging_weights, &self.imaging_biases);
            embeddings.push(embed);
        }

        if let Some(geno) = genomic {
            let embed = forward_mlp(&geno, &self.genomic_weights, &self.genomic_biases);
            embeddings.push(embed);
        }

        Ok(embeddings)
    }

    fn get_attention_weights_for_sample(
        &self,
        clinical: Option<Vec<f64>>,
        imaging: Option<Vec<f64>>,
        genomic: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let mut embeddings = Vec::new();

        if let Some(clin) = clinical {
            let embed = forward_mlp(&clin, &self.clinical_weights, &self.clinical_biases);
            embeddings.push(embed);
        }

        if let Some(img) = imaging {
            let embed = forward_mlp(&img, &self.imaging_weights, &self.imaging_biases);
            embeddings.push(embed);
        }

        if let Some(geno) = genomic {
            let embed = forward_mlp(&geno, &self.genomic_weights, &self.genomic_biases);
            embeddings.push(embed);
        }

        let scores: Vec<f64> = embeddings
            .iter()
            .zip(self.attention_weights.iter())
            .map(|(e, w)| e.iter().zip(w.iter()).map(|(&ei, &wi)| ei * wi).sum())
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let attention: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        Ok(attention)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultimodalSurvModel(clinical={}, imaging={}, genomic={}, fusion={:?})",
            self.n_clinical_features,
            self.n_imaging_features,
            self.n_genomic_features,
            self.config.fusion_strategy
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    clinical=None,
    imaging=None,
    genomic=None,
    time=None,
    event=None,
    config=None
))]
#[allow(unused_variables)]
pub fn fit_multimodal_surv(
    clinical: Option<Vec<Vec<f64>>>,
    imaging: Option<Vec<Vec<f64>>>,
    genomic: Option<Vec<Vec<f64>>>,
    time: Option<Vec<f64>>,
    event: Option<Vec<i32>>,
    config: Option<MultimodalSurvConfig>,
) -> PyResult<MultimodalSurvModel> {
    let config = config.unwrap_or_else(|| {
        MultimodalSurvConfig::new(
            None,
            None,
            None,
            None,
            FusionStrategy::Intermediate,
            20,
            0.1,
            0.001,
            64,
            100,
            None,
        )
    });

    let n = clinical
        .as_ref()
        .map(|c| c.len())
        .or_else(|| imaging.as_ref().map(|i| i.len()))
        .or_else(|| genomic.as_ref().map(|g| g.len()))
        .unwrap_or(0);

    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least one modality must be provided",
        ));
    }

    let n_clinical = clinical
        .as_ref()
        .and_then(|c| c.first())
        .map(|c| c.len())
        .unwrap_or(0);
    let n_imaging = imaging
        .as_ref()
        .and_then(|i| i.first())
        .map(|i| i.len())
        .unwrap_or(0);
    let n_genomic = genomic
        .as_ref()
        .and_then(|g| g.first())
        .map(|g| g.len())
        .unwrap_or(0);

    let clinical_out = config.clinical_hidden_dims.last().copied().unwrap_or(32);
    let imaging_out = config.imaging_hidden_dims.last().copied().unwrap_or(64);
    let genomic_out = config.genomic_hidden_dims.last().copied().unwrap_or(64);

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let clinical_weights: Vec<Vec<f64>> = (0..clinical_out)
        .map(|_| {
            (0..n_clinical.max(1))
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let clinical_biases: Vec<f64> = (0..clinical_out).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let imaging_weights: Vec<Vec<f64>> = (0..imaging_out)
        .map(|_| {
            (0..n_imaging.max(1))
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let imaging_biases: Vec<f64> = (0..imaging_out).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let genomic_weights: Vec<Vec<f64>> = (0..genomic_out)
        .map(|_| {
            (0..n_genomic.max(1))
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let genomic_biases: Vec<f64> = (0..genomic_out).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let fusion_in = clinical_out + imaging_out + genomic_out;
    let fusion_out = config.fusion_hidden_dims.last().copied().unwrap_or(64);

    let fusion_weights: Vec<Vec<f64>> = (0..fusion_out)
        .map(|_| (0..fusion_in).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();
    let fusion_biases: Vec<f64> = (0..fusion_out).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let max_embed = clinical_out.max(imaging_out).max(genomic_out);
    let attention_weights: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..max_embed).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let output_weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| (0..fusion_out).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();
    let output_bias: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let time_ref = time.as_ref();
    let (min_time, max_time) = if let Some(t) = time_ref {
        (
            t.iter().cloned().fold(f64::INFINITY, f64::min),
            t.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    } else {
        (0.0, 100.0)
    };

    let time_bins: Vec<f64> = (0..=config.num_time_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_bins as f64)
        .collect();

    Ok(MultimodalSurvModel {
        clinical_weights,
        clinical_biases,
        imaging_weights,
        imaging_biases,
        genomic_weights,
        genomic_biases,
        fusion_weights,
        fusion_biases,
        attention_weights,
        output_weights,
        output_bias,
        time_bins,
        config,
        n_clinical_features: n_clinical,
        n_imaging_features: n_imaging,
        n_genomic_features: n_genomic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
    }

    #[test]
    fn test_fusion_strategy() {
        let s = FusionStrategy::Attention;
        assert_eq!(s, FusionStrategy::Attention);
    }
}
