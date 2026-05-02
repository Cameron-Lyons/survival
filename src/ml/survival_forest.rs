use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

type NelsonAalenCurve = (Vec<f64>, Vec<f64>);
type SplitCandidate = (usize, f64, Vec<usize>, Vec<usize>);
type TreeWithOob = (TreeNode, Vec<usize>);
type OobPrediction = (usize, Vec<f64>);

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalForestInput {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_vars: usize,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
}

#[pymethods]
impl SurvivalForestInput {
    #[new]
    #[pyo3(signature = (x, n_obs, n_vars, time, status))]
    pub fn new(
        x: Vec<f64>,
        n_obs: usize,
        n_vars: usize,
        time: Vec<f64>,
        status: Vec<i32>,
    ) -> PyResult<Self> {
        let input = Self {
            x,
            n_obs,
            n_vars,
            time,
            status,
        };
        input.validate()?;
        Ok(input)
    }
}

impl SurvivalForestInput {
    fn validate(&self) -> PyResult<()> {
        if self.x.len() != self.n_obs * self.n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x length must equal n_obs * n_vars",
            ));
        }
        if self.time.len() != self.n_obs || self.status.len() != self.n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and status must have length n_obs",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct SurvivalForestData<'a> {
    x: &'a [f64],
    n_obs: usize,
    n_vars: usize,
    time: &'a [f64],
    status: &'a [i32],
}

impl<'a> From<&'a SurvivalForestInput> for SurvivalForestData<'a> {
    fn from(input: &'a SurvivalForestInput) -> Self {
        Self {
            x: &input.x,
            n_obs: input.n_obs,
            n_vars: input.n_vars,
            time: &input.time,
            status: &input.status,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum SplitRule {
    LogRank,
    LogRankScore,
    Conservation,
    Hellinger,
}

#[pymethods]
impl SplitRule {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "logrank" | "log_rank" => Ok(SplitRule::LogRank),
            "logrankscore" | "log_rank_score" => Ok(SplitRule::LogRankScore),
            "conservation" | "cons" => Ok(SplitRule::Conservation),
            "hellinger" => Ok(SplitRule::Hellinger),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown split rule",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalForestConfig {
    #[pyo3(get, set)]
    pub n_trees: usize,
    #[pyo3(get, set)]
    pub max_depth: Option<usize>,
    #[pyo3(get, set)]
    pub min_node_size: usize,
    #[pyo3(get, set)]
    pub mtry: Option<usize>,
    #[pyo3(get, set)]
    pub sample_fraction: f64,
    #[pyo3(get, set)]
    pub split_rule: SplitRule,
    #[pyo3(get, set)]
    pub n_random_splits: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub oob_error: bool,
}

#[pymethods]
impl SurvivalForestConfig {
    #[new]
    #[pyo3(signature = (**kwargs))]
    pub fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut config = SurvivalForestConfig::default();

        if let Some(kwargs) = kwargs {
            let keys = kwargs.keys();
            for key in keys.iter() {
                let key: String = key.extract()?;
                if ![
                    "n_trees",
                    "max_depth",
                    "min_node_size",
                    "mtry",
                    "sample_fraction",
                    "split_rule",
                    "n_random_splits",
                    "seed",
                    "oob_error",
                ]
                .contains(&key.as_str())
                {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "unexpected keyword argument '{key}'"
                    )));
                }
            }

            if let Some(value) = kwargs.get_item("n_trees")? {
                config.n_trees = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("max_depth")? {
                config.max_depth = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("min_node_size")? {
                config.min_node_size = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("mtry")? {
                config.mtry = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("sample_fraction")? {
                config.sample_fraction = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("split_rule")? {
                config.split_rule = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("n_random_splits")? {
                config.n_random_splits = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("seed")? {
                config.seed = value.extract()?;
            }
            if let Some(value) = kwargs.get_item("oob_error")? {
                config.oob_error = value.extract()?;
            }
        }

        config.validate()?;
        Ok(config)
    }
}

impl Default for SurvivalForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 500,
            max_depth: None,
            min_node_size: 15,
            mtry: None,
            sample_fraction: 0.632,
            split_rule: SplitRule::LogRank,
            n_random_splits: 10,
            seed: None,
            oob_error: true,
        }
    }
}

impl SurvivalForestConfig {
    fn validate(&self) -> PyResult<()> {
        validate_survival_forest_config(self.n_trees, self.sample_fraction, self.n_random_splits)
    }
}

fn validate_survival_forest_config(
    n_trees: usize,
    sample_fraction: f64,
    n_random_splits: usize,
) -> PyResult<()> {
    if n_trees == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_trees must be positive",
        ));
    }
    if sample_fraction <= 0.0 || sample_fraction > 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sample_fraction must be in (0, 1]",
        ));
    }
    if n_random_splits == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_random_splits must be positive",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct TreeNode {
    split_var: Option<usize>,
    split_value: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    chf: Vec<f64>,
    _unique_times: Vec<f64>,
    _n_samples: usize,
}

impl TreeNode {
    fn new_leaf(times: &[f64], status: &[i32], all_times: &[f64]) -> Self {
        let (unique_times, chf) = compute_nelson_aalen(times, status, all_times);
        TreeNode {
            split_var: None,
            split_value: None,
            left: None,
            right: None,
            chf,
            _unique_times: unique_times,
            _n_samples: times.len(),
        }
    }
}

fn compute_nelson_aalen(times: &[f64], status: &[i32], all_times: &[f64]) -> NelsonAalenCurve {
    if times.is_empty() {
        return (all_times.to_vec(), vec![0.0; all_times.len()]);
    }

    let mut events: std::collections::HashMap<i64, (f64, f64)> = std::collections::HashMap::new();

    for (i, &t) in times.iter().enumerate() {
        let key = (t * 1e6) as i64;
        let entry = events.entry(key).or_insert((0.0, 0.0));
        entry.1 += 1.0;
        if status[i] == 1 {
            entry.0 += 1.0;
        }
    }

    let n = times.len() as f64;
    let mut sorted_keys: Vec<i64> = events.keys().copied().collect();
    sorted_keys.sort();

    let mut chf = Vec::with_capacity(all_times.len());
    let mut cum_haz = 0.0;
    let mut at_risk = n;

    let mut event_idx = 0;

    for &t in all_times {
        let t_key = (t * 1e6) as i64;

        while event_idx < sorted_keys.len() && sorted_keys[event_idx] <= t_key {
            let key = sorted_keys[event_idx];
            if let Some(&(d, n_i)) = events.get(&key) {
                if at_risk > 0.0 {
                    cum_haz += d / at_risk;
                }
                at_risk -= n_i;
            }
            event_idx += 1;
        }

        chf.push(cum_haz);
    }

    (all_times.to_vec(), chf)
}

fn log_rank_split_score(
    times_left: &[f64],
    status_left: &[i32],
    times_right: &[f64],
    status_right: &[i32],
) -> f64 {
    if times_left.is_empty() || times_right.is_empty() {
        return f64::NEG_INFINITY;
    }

    let n_left = times_left.len() as f64;
    let n_right = times_right.len() as f64;
    let n_total = n_left + n_right;

    let d_left: f64 = status_left.iter().map(|&s| s as f64).sum();
    let d_right: f64 = status_right.iter().map(|&s| s as f64).sum();
    let d_total = d_left + d_right;

    if d_total == 0.0 {
        return f64::NEG_INFINITY;
    }

    let expected_left = d_total * n_left / n_total;

    let variance = d_total * (n_left / n_total) * (n_right / n_total) * (n_total - d_total)
        / (n_total - 1.0).max(1.0);

    if variance <= 0.0 {
        return f64::NEG_INFINITY;
    }

    (d_left - expected_left).powi(2) / variance
}

fn find_best_split(
    data: &SurvivalForestData<'_>,
    indices: &[usize],
    mtry: usize,
    min_node_size: usize,
    n_random_splits: usize,
    rng: &mut fastrand::Rng,
    split_rule: &SplitRule,
) -> Option<SplitCandidate> {
    if indices.len() < 2 * min_node_size {
        return None;
    }

    let mut candidate_vars: Vec<usize> = (0..data.n_vars).collect();
    for i in (1..candidate_vars.len()).rev() {
        let j = rng.usize(0..=i);
        candidate_vars.swap(i, j);
    }
    candidate_vars.truncate(mtry);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_split: Option<SplitCandidate> = None;

    for &var in &candidate_vars {
        let mut values: Vec<f64> = indices
            .iter()
            .map(|&i| data.x[i * data.n_vars + var])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();

        if values.len() < 2 {
            continue;
        }

        let split_candidates: Vec<f64> = if values.len() <= n_random_splits {
            (0..values.len() - 1)
                .map(|i| (values[i] + values[i + 1]) / 2.0)
                .collect()
        } else {
            (0..n_random_splits)
                .map(|_| {
                    let idx = rng.usize(0..values.len() - 1);
                    (values[idx] + values[idx + 1]) / 2.0
                })
                .collect()
        };

        for &split_value in &split_candidates {
            let left_idx: Vec<usize> = indices
                .iter()
                .filter(|&&i| data.x[i * data.n_vars + var] <= split_value)
                .copied()
                .collect();
            let right_idx: Vec<usize> = indices
                .iter()
                .filter(|&&i| data.x[i * data.n_vars + var] > split_value)
                .copied()
                .collect();

            if left_idx.len() < min_node_size || right_idx.len() < min_node_size {
                continue;
            }

            let times_left: Vec<f64> = left_idx.iter().map(|&i| data.time[i]).collect();
            let status_left: Vec<i32> = left_idx.iter().map(|&i| data.status[i]).collect();
            let times_right: Vec<f64> = right_idx.iter().map(|&i| data.time[i]).collect();
            let status_right: Vec<i32> = right_idx.iter().map(|&i| data.status[i]).collect();

            let score = match split_rule {
                SplitRule::LogRank | SplitRule::LogRankScore => {
                    log_rank_split_score(&times_left, &status_left, &times_right, &status_right)
                }
                _ => log_rank_split_score(&times_left, &status_left, &times_right, &status_right),
            };

            if score > best_score {
                best_score = score;
                best_split = Some((var, split_value, left_idx, right_idx));
            }
        }
    }

    best_split
}

fn build_tree(
    data: &SurvivalForestData<'_>,
    indices: &[usize],
    all_times: &[f64],
    config: &SurvivalForestConfig,
    depth: usize,
    rng: &mut fastrand::Rng,
) -> TreeNode {
    let node_times: Vec<f64> = indices.iter().map(|&i| data.time[i]).collect();
    let node_status: Vec<i32> = indices.iter().map(|&i| data.status[i]).collect();

    if indices.len() < 2 * config.min_node_size {
        return TreeNode::new_leaf(&node_times, &node_status, all_times);
    }

    if let Some(max_d) = config.max_depth
        && depth >= max_d
    {
        return TreeNode::new_leaf(&node_times, &node_status, all_times);
    }

    let mtry = config
        .mtry
        .unwrap_or((data.n_vars as f64).sqrt().ceil() as usize)
        .max(1);

    let best_split = find_best_split(
        data,
        indices,
        mtry,
        config.min_node_size,
        config.n_random_splits,
        rng,
        &config.split_rule,
    );

    match best_split {
        Some((split_var, split_value, left_idx, right_idx)) => {
            let left_child = build_tree(data, &left_idx, all_times, config, depth + 1, rng);
            let right_child = build_tree(data, &right_idx, all_times, config, depth + 1, rng);

            let (unique_times, chf) = compute_nelson_aalen(&node_times, &node_status, all_times);

            TreeNode {
                split_var: Some(split_var),
                split_value: Some(split_value),
                left: Some(Box::new(left_child)),
                right: Some(Box::new(right_child)),
                chf,
                _unique_times: unique_times,
                _n_samples: indices.len(),
            }
        }
        None => TreeNode::new_leaf(&node_times, &node_status, all_times),
    }
}

fn predict_tree<'a>(node: &'a TreeNode, x_row: &[f64]) -> &'a [f64] {
    match (&node.split_var, &node.split_value) {
        (Some(var), Some(val)) => {
            if x_row[*var] <= *val {
                if let Some(ref left) = node.left {
                    return predict_tree(left, x_row);
                }
            } else if let Some(ref right) = node.right {
                return predict_tree(right, x_row);
            }
            &node.chf
        }
        _ => &node.chf,
    }
}

fn fit_survival_forest_inner(
    data: &SurvivalForestData<'_>,
    config: &SurvivalForestConfig,
) -> SurvivalForest {
    let mut unique_times: Vec<f64> = data.time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let sample_size = (data.n_obs as f64 * config.sample_fraction).ceil() as usize;

    let base_seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let results: Vec<TreeWithOob> = (0..config.n_trees)
        .into_par_iter()
        .map(|tree_idx| {
            let mut rng = fastrand::Rng::with_seed(base_seed.wrapping_add(tree_idx as u64));

            let mut bootstrap_indices: Vec<usize> = Vec::with_capacity(sample_size);
            let mut in_bag = vec![false; data.n_obs];

            for _ in 0..sample_size {
                let idx = rng.usize(0..data.n_obs);
                bootstrap_indices.push(idx);
                in_bag[idx] = true;
            }

            let oob: Vec<usize> = (0..data.n_obs).filter(|&i| !in_bag[i]).collect();

            let tree = build_tree(data, &bootstrap_indices, &unique_times, config, 0, &mut rng);

            (tree, oob)
        })
        .collect();

    let (trees, oob_indices): (Vec<TreeNode>, Vec<Vec<usize>>) = results.into_iter().unzip();

    let oob_error = if config.oob_error {
        Some(compute_oob_error(&trees, &oob_indices, data))
    } else {
        None
    };

    let variable_importance = compute_variable_importance(&trees, &oob_indices, data);

    SurvivalForest {
        trees,
        unique_times,
        oob_error,
        variable_importance,
        n_vars: data.n_vars,
        _oob_indices: oob_indices,
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalForest {
    trees: Vec<TreeNode>,
    unique_times: Vec<f64>,
    #[pyo3(get)]
    pub oob_error: Option<f64>,
    #[pyo3(get)]
    pub variable_importance: Vec<f64>,
    n_vars: usize,
    _oob_indices: Vec<Vec<usize>>,
}

#[pymethods]
impl SurvivalForest {
    #[staticmethod]
    #[pyo3(signature = (input, config))]
    pub fn fit_typed(
        py: Python<'_>,
        input: &SurvivalForestInput,
        config: &SurvivalForestConfig,
    ) -> PyResult<Self> {
        input.validate()?;
        let input = input.clone();
        let config = config.clone();
        Ok(py.detach(move || {
            let data = SurvivalForestData::from(&input);
            fit_survival_forest_inner(&data, &config)
        }))
    }

    #[staticmethod]
    #[pyo3(signature = (x, n_obs, n_vars, time, status, config))]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        n_obs: usize,
        n_vars: usize,
        time: Vec<f64>,
        status: Vec<i32>,
        config: &SurvivalForestConfig,
    ) -> PyResult<Self> {
        let input = SurvivalForestInput::new(x, n_obs, n_vars, time, status)?;
        Self::fit_typed(py, &input, config)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_cumulative_hazard(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        if x_new.len() != n_new * self.n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_new dimensions don't match",
            ));
        }

        let n_times = self.unique_times.len();
        let n_trees = self.trees.len() as f64;
        let n_vars = self.n_vars;

        let predictions: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let x_row = &x_new[i * n_vars..(i + 1) * n_vars];

                let mut avg_chf = vec![0.0; n_times];
                for tree in &self.trees {
                    let pred = predict_tree(tree, x_row);
                    for (t, &val) in pred.iter().enumerate() {
                        if t < n_times {
                            avg_chf[t] += val;
                        }
                    }
                }

                for val in &mut avg_chf {
                    *val /= n_trees;
                }

                avg_chf
            })
            .collect();

        Ok(predictions)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_survival(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<Vec<f64>>> {
        let chf = self.predict_cumulative_hazard(x_new, n_new)?;

        let survival: Vec<Vec<f64>> = chf
            .into_iter()
            .map(|h| h.into_iter().map(|val| (-val).exp()).collect())
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_risk(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<f64>> {
        let chf = self.predict_cumulative_hazard(x_new, n_new)?;

        let risk: Vec<f64> = chf
            .into_iter()
            .map(|h| h.last().copied().unwrap_or(0.0))
            .collect();

        Ok(risk)
    }

    #[getter]
    pub fn get_unique_times(&self) -> Vec<f64> {
        self.unique_times.clone()
    }

    #[getter]
    pub fn get_n_trees(&self) -> usize {
        self.trees.len()
    }

    #[pyo3(signature = (x_new, n_new, percentile=0.5))]
    pub fn predict_survival_time(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        percentile: f64,
    ) -> PyResult<Vec<Option<f64>>> {
        if !(0.0..=1.0).contains(&percentile) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "percentile must be between 0 and 1",
            ));
        }

        let survival = self.predict_survival(x_new, n_new)?;

        let times: Vec<Option<f64>> = survival
            .par_iter()
            .map(|surv| {
                for (i, &s) in surv.iter().enumerate() {
                    if s <= percentile && i < self.unique_times.len() {
                        return Some(self.unique_times[i]);
                    }
                }
                None
            })
            .collect();

        Ok(times)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_median_survival_time(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Option<f64>>> {
        self.predict_survival_time(x_new, n_new, 0.5)
    }
}

fn compute_oob_error(
    trees: &[TreeNode],
    oob_indices: &[Vec<usize>],
    data: &SurvivalForestData<'_>,
) -> f64 {
    let n_times = trees.first().map(|t| t.chf.len()).unwrap_or(0);

    let oob_results: Vec<OobPrediction> = trees
        .par_iter()
        .zip(oob_indices.par_iter())
        .flat_map(|(tree, oob)| {
            oob.iter()
                .map(|&i| {
                    let x_row = &data.x[i * data.n_vars..(i + 1) * data.n_vars];
                    let pred = predict_tree(tree, x_row);
                    (i, pred.to_vec())
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut oob_chf: Vec<Vec<f64>> = vec![vec![0.0; n_times]; data.n_obs];
    let mut oob_count = vec![0usize; data.n_obs];

    for (i, pred) in oob_results {
        if oob_count[i] == 0 {
            oob_chf[i] = pred;
        } else {
            for (t, &val) in pred.iter().enumerate() {
                if t < oob_chf[i].len() {
                    oob_chf[i][t] += val;
                }
            }
        }
        oob_count[i] += 1;
    }

    for i in 0..data.n_obs {
        if oob_count[i] > 0 {
            let count = oob_count[i] as f64;
            for val in &mut oob_chf[i] {
                *val /= count;
            }
        }
    }

    let risk_scores: Vec<f64> = oob_chf
        .iter()
        .map(|chf| chf.last().copied().unwrap_or(0.0))
        .collect();

    let (concordant, comparable) = (0..data.n_obs)
        .into_par_iter()
        .filter(|&i| oob_count[i] > 0 && data.status[i] == 1)
        .map(|i| {
            let risk_i = risk_scores[i];
            let mut conc = 0.0;
            let mut comp = 0.0;
            for j in 0..data.n_obs {
                if i == j || oob_count[j] == 0 {
                    continue;
                }
                if data.time[i] < data.time[j] {
                    let risk_j = risk_scores[j];
                    comp += 1.0;
                    if risk_i > risk_j {
                        conc += 1.0;
                    } else if (risk_i - risk_j).abs() < 1e-10 {
                        conc += 0.5;
                    }
                }
            }
            (conc, comp)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    if comparable > 0.0 {
        1.0 - concordant / comparable
    } else {
        0.5
    }
}

fn compute_variable_importance(
    trees: &[TreeNode],
    oob_indices: &[Vec<usize>],
    data: &SurvivalForestData<'_>,
) -> Vec<f64> {
    let base_error = compute_oob_error(trees, oob_indices, data);

    let importance: Vec<f64> = (0..data.n_vars)
        .into_par_iter()
        .map(|var| {
            let mut x_permuted = data.x.to_vec();

            let mut rng = fastrand::Rng::with_seed(var as u64);
            let mut perm: Vec<usize> = (0..data.n_obs).collect();
            for i in (1..data.n_obs).rev() {
                let j = rng.usize(0..=i);
                perm.swap(i, j);
            }

            for i in 0..data.n_obs {
                x_permuted[i * data.n_vars + var] = data.x[perm[i] * data.n_vars + var];
            }

            let permuted_data = SurvivalForestData {
                x: &x_permuted,
                ..*data
            };
            let permuted_error = compute_oob_error(trees, oob_indices, &permuted_data);

            permuted_error - base_error
        })
        .collect();

    importance
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config=None))]
pub fn survival_forest(
    py: Python<'_>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: Option<&SurvivalForestConfig>,
) -> PyResult<SurvivalForest> {
    let cfg = config.cloned().unwrap_or_default();

    let input = SurvivalForestInput::new(x, n_obs, n_vars, time, status)?;
    SurvivalForest::fit_typed(py, &input, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SurvivalForestConfig {
            n_trees: 100,
            min_node_size: 10,
            ..SurvivalForestConfig::default()
        };
        assert_eq!(config.n_trees, 100);
        assert_eq!(config.min_node_size, 10);
    }

    #[test]
    fn test_nelson_aalen() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];
        let all_times = vec![1.0, 2.0, 3.0, 4.0];

        let (ut, chf) = compute_nelson_aalen(&times, &status, &all_times);
        assert_eq!(ut.len(), chf.len());
        assert!(chf.iter().all(|&h| h >= 0.0));
    }

    #[test]
    fn test_survival_forest_basic() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];

        let config = SurvivalForestConfig {
            n_trees: 10,
            max_depth: Some(3),
            min_node_size: 2,
            mtry: None,
            sample_fraction: 0.8,
            split_rule: SplitRule::LogRank,
            n_random_splits: 5,
            seed: Some(42),
            oob_error: false,
        };

        let data = SurvivalForestData {
            x: &x,
            n_obs: 6,
            n_vars: 2,
            time: &time,
            status: &status,
        };
        let forest = fit_survival_forest_inner(&data, &config);
        assert_eq!(forest.get_n_trees(), 10);
    }
}
