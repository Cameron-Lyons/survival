use pyo3::prelude::*;
use std::collections::HashMap;

/// Data for drawing a state space figure.
///
/// This provides structured data that can be used with external plotting
/// libraries to create box-and-arrow diagrams showing states and transitions
/// in multi-state survival models.
#[derive(Debug, Clone)]
#[pyclass]
pub struct StateFigData {
    /// Names of the states
    #[pyo3(get)]
    pub states: Vec<String>,
    /// Positions for each state (x, y coordinates)
    #[pyo3(get)]
    pub positions: Vec<(f64, f64)>,
    /// Edges: (from_state_idx, to_state_idx, count)
    #[pyo3(get)]
    pub edges: Vec<(usize, usize, usize)>,
    /// Box dimensions (width, height) for each state
    #[pyo3(get)]
    pub box_sizes: Vec<(f64, f64)>,
    /// Layout specification used
    #[pyo3(get)]
    pub layout: Vec<usize>,
}

/// Generate data for a state space figure.
///
/// This function computes layout positions and edge information for
/// visualizing a multi-state model. The actual plotting should be done
/// with a graphics library (matplotlib, plotly, etc.) using this data.
///
/// # Arguments
/// * `states` - Names of the states
/// * `transitions` - HashMap of (from_state, to_state) -> count
/// * `layout` - Optional layout specification (states per row)
///
/// # Returns
/// * `StateFigData` with positions and edges for plotting
#[pyfunction]
#[pyo3(signature = (states, transitions, layout=None))]
pub fn statefig(
    states: Vec<String>,
    transitions: HashMap<(String, String), usize>,
    layout: Option<Vec<usize>>,
) -> PyResult<StateFigData> {
    let n_states = states.len();

    if n_states == 0 {
        return Ok(StateFigData {
            states: vec![],
            positions: vec![],
            edges: vec![],
            box_sizes: vec![],
            layout: vec![],
        });
    }

    let state_idx: HashMap<String, usize> = states
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let mut edges = Vec::new();
    for ((from, to), count) in &transitions {
        if let (Some(&from_idx), Some(&to_idx)) = (state_idx.get(from), state_idx.get(to))
            && *count > 0
        {
            edges.push((from_idx, to_idx, *count));
        }
    }

    let layout_spec = layout.unwrap_or_else(|| compute_default_layout(n_states, &edges));
    let positions = compute_positions(&layout_spec, n_states);
    let box_sizes = vec![(1.0, 0.5); n_states];

    Ok(StateFigData {
        states,
        positions,
        edges,
        box_sizes,
        layout: layout_spec,
    })
}

/// Compute default layout based on number of states and transitions
fn compute_default_layout(n_states: usize, edges: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut out_degree = vec![0usize; n_states];
    let mut in_degree = vec![0usize; n_states];

    for &(from, to, _) in edges {
        out_degree[from] += 1;
        in_degree[to] += 1;
    }

    let mut state_scores: Vec<(usize, i32)> = (0..n_states)
        .map(|i| (i, out_degree[i] as i32 - in_degree[i] as i32))
        .collect();
    state_scores.sort_by(|a, b| b.1.cmp(&a.1));

    let n_cols = (n_states as f64).sqrt().ceil() as usize;
    let n_rows = n_states.div_ceil(n_cols);

    let mut layout = Vec::new();
    let mut remaining = n_states;

    for _ in 0..n_rows {
        let row_size = remaining.min(n_cols);
        layout.push(row_size);
        remaining -= row_size;
    }

    layout
}

/// Compute positions for states based on layout
fn compute_positions(layout: &[usize], n_states: usize) -> Vec<(f64, f64)> {
    let mut positions = vec![(0.0, 0.0); n_states];

    let n_rows = layout.len();
    let mut state_idx = 0;

    for (row, &n_in_row) in layout.iter().enumerate() {
        let y = 1.0 - (row as f64 + 0.5) / n_rows as f64;

        for col in 0..n_in_row {
            if state_idx >= n_states {
                break;
            }

            let x = (col as f64 + 0.5) / n_in_row as f64;
            positions[state_idx] = (x, y);
            state_idx += 1;
        }
    }

    positions
}

/// Generate matplotlib-compatible plot code for the state figure
#[pyfunction]
pub fn statefig_matplotlib_code(data: &StateFigData) -> String {
    let mut code = String::new();

    code.push_str("import matplotlib.pyplot as plt\n");
    code.push_str("import matplotlib.patches as mpatches\n");
    code.push_str("from matplotlib.patches import FancyArrowPatch\n\n");

    code.push_str("fig, ax = plt.subplots(figsize=(10, 8))\n");
    code.push_str("ax.set_xlim(-0.1, 1.1)\n");
    code.push_str("ax.set_ylim(-0.1, 1.1)\n");
    code.push_str("ax.set_aspect('equal')\n");
    code.push_str("ax.axis('off')\n\n");

    for (i, (state, &(x, y))) in data.states.iter().zip(data.positions.iter()).enumerate() {
        let (w, h) = data.box_sizes[i];
        code.push_str(&format!(
            "rect = mpatches.FancyBboxPatch(({:.3} - {:.3}/2, {:.3} - {:.3}/2), {:.3}, {:.3}, ",
            x,
            w * 0.15,
            y,
            h * 0.15,
            w * 0.15,
            h * 0.15
        ));
        code.push_str("boxstyle='round,pad=0.01', facecolor='lightblue', edgecolor='black')\n");
        code.push_str("ax.add_patch(rect)\n");
        code.push_str(&format!(
            "ax.text({:.3}, {:.3}, '{}', ha='center', va='center', fontsize=10)\n\n",
            x, y, state
        ));
    }

    for &(from, to, count) in &data.edges {
        let (x1, y1) = data.positions[from];
        let (x2, y2) = data.positions[to];

        code.push_str(&format!(
            "arrow = FancyArrowPatch(({:.3}, {:.3}), ({:.3}, {:.3}), ",
            x1, y1, x2, y2
        ));
        code.push_str(
            "arrowstyle='->', mutation_scale=15, color='black', connectionstyle='arc3,rad=0.1')\n",
        );
        code.push_str("ax.add_patch(arrow)\n");

        let mid_x = (x1 + x2) / 2.0;
        let mid_y = (y1 + y2) / 2.0;
        code.push_str(&format!(
            "ax.text({:.3}, {:.3}, '{}', ha='center', va='center', fontsize=8, color='red')\n\n",
            mid_x, mid_y, count
        ));
    }

    code.push_str("plt.title('State Transition Diagram')\n");
    code.push_str("plt.tight_layout()\n");
    code.push_str("plt.show()\n");

    code
}

/// Create transition matrix from edge list
#[pyfunction]
pub fn statefig_transition_matrix(data: &StateFigData) -> Vec<Vec<usize>> {
    let n = data.states.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for &(from, to, count) in &data.edges {
        matrix[from][to] = count;
    }

    matrix
}

/// Validate state transitions against allowed patterns
#[pyfunction]
pub fn statefig_validate(
    data: &StateFigData,
    allowed_transitions: HashMap<(String, String), bool>,
) -> PyResult<Vec<String>> {
    let mut issues = Vec::new();

    let _state_idx: HashMap<String, usize> = data
        .states
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    for &(from_idx, to_idx, count) in &data.edges {
        if count > 0 {
            let from_state = &data.states[from_idx];
            let to_state = &data.states[to_idx];

            let key = (from_state.clone(), to_state.clone());
            if !allowed_transitions.get(&key).copied().unwrap_or(true) {
                issues.push(format!(
                    "Invalid transition: {} -> {} ({} occurrences)",
                    from_state, to_state, count
                ));
            }
        }
    }

    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statefig_basic() {
        let states = vec![
            "Healthy".to_string(),
            "Sick".to_string(),
            "Dead".to_string(),
        ];
        let mut transitions = HashMap::new();
        transitions.insert(("Healthy".to_string(), "Sick".to_string()), 50);
        transitions.insert(("Healthy".to_string(), "Dead".to_string()), 10);
        transitions.insert(("Sick".to_string(), "Dead".to_string()), 40);

        let result = statefig(states, transitions, None).unwrap();

        assert_eq!(result.states.len(), 3);
        assert_eq!(result.positions.len(), 3);
        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_statefig_with_layout() {
        let states = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let transitions = HashMap::new();
        let layout = vec![1, 2]; // 1 state in first row, 2 in second

        let result = statefig(states, transitions, Some(layout.clone())).unwrap();

        assert_eq!(result.layout, layout);
    }

    #[test]
    fn test_statefig_empty() {
        let states: Vec<String> = vec![];
        let transitions = HashMap::new();

        let result = statefig(states, transitions, None).unwrap();

        assert!(result.states.is_empty());
    }

    #[test]
    fn test_transition_matrix() {
        let data = StateFigData {
            states: vec!["A".to_string(), "B".to_string()],
            positions: vec![(0.0, 0.0), (1.0, 0.0)],
            edges: vec![(0, 1, 5)],
            box_sizes: vec![(1.0, 0.5), (1.0, 0.5)],
            layout: vec![2],
        };

        let matrix = statefig_transition_matrix(&data);

        assert_eq!(matrix[0][1], 5);
        assert_eq!(matrix[1][0], 0);
    }
}
