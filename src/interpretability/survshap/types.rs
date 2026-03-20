

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

pub enum AggregationMethod {
    Mean,
    Integral,
    MaxAbsolute,
    TimeWeighted,
}

#[pymethods]
impl AggregationMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "mean" => Ok(AggregationMethod::Mean),
            "integral" => Ok(AggregationMethod::Integral),
            "max_absolute" | "maxabsolute" => Ok(AggregationMethod::MaxAbsolute),
            "time_weighted" | "timeweighted" => Ok(AggregationMethod::TimeWeighted),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown aggregation method. Use 'mean', 'integral', 'max_absolute', or 'time_weighted'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvShapConfig {
    #[pyo3(get, set)]
    pub n_coalitions: usize,
    #[pyo3(get, set)]
    pub n_background: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub parallel: bool,
}

#[pymethods]
impl SurvShapConfig {
    #[new]
    #[pyo3(signature = (n_coalitions=2048, n_background=100, seed=None, parallel=true))]
    pub fn new(
        n_coalitions: usize,
        n_background: usize,
        seed: Option<u64>,
        parallel: bool,
    ) -> PyResult<Self> {
        if n_coalitions < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_coalitions must be at least 2",
            ));
        }
        if n_background == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_background must be positive",
            ));
        }

        Ok(SurvShapConfig {
            n_coalitions,
            n_background,
            seed,
            parallel,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FeatureImportance {
    #[pyo3(get)]
    pub feature_idx: usize,
    #[pyo3(get)]
    pub importance: f64,
    #[pyo3(get)]
    pub importance_std: Option<f64>,
}

#[pymethods]
impl FeatureImportance {
    fn __repr__(&self) -> String {
        match self.importance_std {
            Some(std) => format!(
                "FeatureImportance(idx={}, importance={:.4} ± {:.4})",
                self.feature_idx, self.importance, std
            ),
            None => format!(
                "FeatureImportance(idx={}, importance={:.4})",
                self.feature_idx, self.importance
            ),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvShapResult {
    #[pyo3(get)]
    pub shap_values: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub base_value: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub aggregated_importance: Option<Vec<f64>>,
}

#[pymethods]
impl SurvShapResult {
    fn __repr__(&self) -> String {
        let n_samples = self.shap_values.len();
        let n_features = if n_samples > 0 {
            self.shap_values[0].len()
        } else {
            0
        };
        let n_times = self.time_points.len();
        format!(
            "SurvShapResult(samples={}, features={}, time_points={})",
            n_samples, n_features, n_times
        )
    }

    fn get_sample_shap(&self, sample_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        if sample_idx >= self.shap_values.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "sample_idx out of bounds",
            ));
        }
        Ok(self.shap_values[sample_idx].clone())
    }

    fn get_feature_shap(&self, feature_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n_features = self.shap_values[0].len();
        if feature_idx >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "feature_idx out of bounds",
            ));
        }
        let result: Vec<Vec<f64>> = self
            .shap_values
            .iter()
            .map(|sample| sample[feature_idx].clone())
            .collect();
        Ok(result)
    }

    fn get_shap_at_time(&self, time_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n_times = self.time_points.len();
        if time_idx >= n_times {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "time_idx out of bounds",
            ));
        }
        let n_features = self.shap_values[0].len();
        let result: Vec<Vec<f64>> = self
            .shap_values
            .iter()
            .map(|sample| (0..n_features).map(|f| sample[f][time_idx]).collect())
            .collect();
        Ok(result)
    }

    #[pyo3(signature = (method=AggregationMethod::Mean, top_k=None))]
    fn feature_ranking(
        &self,
        method: AggregationMethod,
        top_k: Option<usize>,
    ) -> PyResult<Vec<FeatureImportance>> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        let n_features = self.shap_values[0].len();
        let n_times = self.time_points.len();

        let importance = aggregate_shap_values(
            &self.shap_values,
            &self.time_points,
            method,
            n_features,
            n_times,
        );

        let time_diffs: Vec<f64> = if n_times >= 2 {
            self.time_points.windows(2).map(|w| w[1] - w[0]).collect()
        } else {
            Vec::new()
        };

        let total_time =
            self.time_points.last().unwrap_or(&1.0) - self.time_points.first().unwrap_or(&0.0);
        let time_weights: Vec<f64> = if total_time > 0.0 {
            self.time_points
                .iter()
                .map(|&t| 1.0 - (t - self.time_points[0]) / total_time)
                .collect()
        } else {
            Vec::new()
        };

        let mut stds = vec![0.0; n_features];
        for f in 0..n_features {
            let sample_importances: Vec<f64> = self
                .shap_values
                .iter()
                .map(|sample| {
                    let shap_t = &sample[f];
                    match method {
                        AggregationMethod::Mean => {
                            shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                        }
                        AggregationMethod::MaxAbsolute => {
                            shap_t.iter().map(|v| v.abs()).fold(0.0, f64::max)
                        }
                        AggregationMethod::Integral => {
                            if n_times < 2 {
                                shap_t.first().copied().unwrap_or(0.0).abs()
                            } else {
                                let mut integral = 0.0;
                                for i in 0..time_diffs.len() {
                                    let avg = (shap_t[i + 1].abs() + shap_t[i].abs()) / 2.0;
                                    integral += avg * time_diffs[i];
                                }
                                integral
                            }
                        }
                        AggregationMethod::TimeWeighted => {
                            if total_time <= 0.0 {
                                shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                            } else {
                                let mut weighted_sum = 0.0;
                                for (i, &weight) in time_weights.iter().enumerate() {
                                    weighted_sum += shap_t[i].abs() * weight;
                                }
                                weighted_sum / n_times as f64
                            }
                        }
                    }
                })
                .collect();

            let mean = importance[f];
            let variance: f64 = sample_importances
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            stds[f] = variance.sqrt();
        }

        let mut ranking: Vec<FeatureImportance> = (0..n_features)
            .map(|f| FeatureImportance {
                feature_idx: f,
                importance: importance[f],
                importance_std: Some(stds[f]),
            })
            .collect();

        ranking.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = top_k {
            ranking.truncate(k);
        }

        Ok(ranking)
    }

    fn mean_absolute_shap(&self) -> Vec<f64> {
        let n_samples = self.shap_values.len();
        if n_samples == 0 {
            return Vec::new();
        }
        let n_features = self.shap_values[0].len();
        let n_times = self.time_points.len();

        (0..n_features)
            .map(|f| {
                let total: f64 = self
                    .shap_values
                    .iter()
                    .flat_map(|sample| sample[f].iter())
                    .map(|v| v.abs())
                    .sum();
                total / (n_samples * n_times) as f64
            })
            .collect()
    }

    fn check_additivity(&self, predictions: Vec<f64>, tolerance: f64) -> PyResult<Vec<bool>> {
        let n_samples = self.shap_values.len();
        let n_times = self.time_points.len();

        if predictions.len() != n_samples * n_times {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "predictions length must equal n_samples * n_times",
            ));
        }

        let n_features = if n_samples > 0 {
            self.shap_values[0].len()
        } else {
            0
        };

        let mut results = Vec::with_capacity(n_samples * n_times);

        for i in 0..n_samples {
            for t in 0..n_times {
                let shap_sum: f64 = (0..n_features).map(|f| self.shap_values[i][f][t]).sum();
                let reconstructed = self.base_value[t] + shap_sum;
                let actual = predictions[i * n_times + t];
                results.push((reconstructed - actual).abs() <= tolerance);
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvShapExplanation {
    #[pyo3(get)]
    pub shap_values: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub base_value: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub feature_values: Vec<f64>,
    #[pyo3(get)]
    pub aggregated_importance: Option<Vec<f64>>,
}

#[pymethods]
impl SurvShapExplanation {
    fn __repr__(&self) -> String {
        let n_features = self.shap_values.len();
        let n_times = self.time_points.len();
        format!(
            "SurvShapExplanation(features={}, time_points={})",
            n_features, n_times
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct BootstrapSurvShapResult {
    #[pyo3(get)]
    pub shap_values_mean: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub shap_values_std: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub shap_values_lower: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub shap_values_upper: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub base_value: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub n_bootstrap: usize,
    #[pyo3(get)]
    pub confidence_level: f64,
}

#[pymethods]
impl BootstrapSurvShapResult {
    fn __repr__(&self) -> String {
        let n_samples = self.shap_values_mean.len();
        let n_features = if n_samples > 0 {
            self.shap_values_mean[0].len()
        } else {
            0
        };
        let n_times = self.time_points.len();
        format!(
            "BootstrapSurvShapResult(samples={}, features={}, time_points={}, n_bootstrap={}, confidence={:.0}%)",
            n_samples,
            n_features,
            n_times,
            self.n_bootstrap,
            self.confidence_level * 100.0
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct PermutationImportanceResult {
    #[pyo3(get)]
    pub importance: Vec<f64>,
    #[pyo3(get)]
    pub importance_std: Vec<f64>,
    #[pyo3(get)]
    pub baseline_score: f64,
    #[pyo3(get)]
    pub n_repeats: usize,
}

#[pymethods]
impl PermutationImportanceResult {
    fn __repr__(&self) -> String {
        format!(
            "PermutationImportanceResult(n_features={}, n_repeats={}, baseline={:.4})",
            self.importance.len(),
            self.n_repeats,
            self.baseline_score
        )
    }

    fn feature_ranking(&self, top_k: Option<usize>) -> Vec<FeatureImportance> {
        let mut ranking: Vec<FeatureImportance> = self
            .importance
            .iter()
            .zip(self.importance_std.iter())
            .enumerate()
            .map(|(idx, (&imp, &std))| FeatureImportance {
                feature_idx: idx,
                importance: imp,
                importance_std: Some(std),
            })
            .collect();

        ranking.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(k) = top_k {
            ranking.truncate(k);
        }

        ranking
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ShapInteractionResult {
    #[pyo3(get)]
    pub interaction_values: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub aggregated_interactions: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl ShapInteractionResult {
    fn __repr__(&self) -> String {
        let n_features = self.interaction_values.len();
        let n_times = self.time_points.len();
        format!(
            "ShapInteractionResult(features={}, time_points={})",
            n_features, n_times
        )
    }

    fn get_interaction(&self, feature_i: usize, feature_j: usize) -> PyResult<Vec<f64>> {
        let n_features = self.interaction_values.len();
        if feature_i >= n_features || feature_j >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "feature index out of bounds",
            ));
        }
        Ok(self.interaction_values[feature_i][feature_j].clone())
    }
    fn top_interactions(&self, top_k: usize) -> Vec<(usize, usize, f64)> {
        let n_features = self.interaction_values.len();
        let mut interactions: Vec<(usize, usize, f64)> = Vec::new();

        if let Some(ref agg) = self.aggregated_interactions {
            for (i, row) in agg.iter().enumerate().take(n_features) {
                for (j, &value) in row.iter().enumerate().take(n_features).skip(i + 1) {
                    interactions.push((i, j, value));
                }
            }
        } else {
            let n_times = self.time_points.len();
            for (i, row) in self.interaction_values.iter().enumerate().take(n_features) {
                for (j, values) in row.iter().enumerate().take(n_features).skip(i + 1) {
                    let mean_interaction: f64 =
                        values.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64;
                    interactions.push((i, j, mean_interaction));
                }
            }
        }

        interactions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        interactions.truncate(top_k);
        interactions
    }
}

