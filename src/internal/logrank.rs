const LOGRANK_VARIANCE_TOLERANCE: f64 = 1e-12;

pub(crate) fn logrank_statistic_from_flat_covariance(
    observed: &[f64],
    expected: &[f64],
    variance: &[f64],
    n_groups: usize,
) -> (f64, usize) {
    if observed.len() < n_groups
        || expected.len() < n_groups
        || variance.len() < n_groups * n_groups
    {
        return (0.0, 0);
    }

    let active_groups: Vec<usize> = (0..n_groups)
        .filter(|&idx| {
            expected[idx] > 0.0
                && variance[idx * n_groups + idx].is_finite()
                && variance[idx * n_groups + idx] > LOGRANK_VARIANCE_TOLERANCE
        })
        .collect();
    let degrees_of_freedom = active_groups.len().saturating_sub(1);
    if degrees_of_freedom == 0 {
        return (0.0, 0);
    }

    let mut matrix = vec![0.0; degrees_of_freedom * degrees_of_freedom];
    let mut contrast = vec![0.0; degrees_of_freedom];
    for row in 0..degrees_of_freedom {
        let source_row = active_groups[row];
        contrast[row] = observed[source_row] - expected[source_row];
        for col in 0..degrees_of_freedom {
            let source_col = active_groups[col];
            matrix[row * degrees_of_freedom + col] = variance[source_row * n_groups + source_col];
        }
    }

    match solve_linear_system(matrix, contrast.clone(), degrees_of_freedom) {
        Some(solution) => {
            let statistic = contrast
                .iter()
                .zip(solution.iter())
                .map(|(&lhs, &rhs)| lhs * rhs)
                .sum::<f64>();
            if statistic.is_finite() && statistic > 0.0 {
                (statistic, degrees_of_freedom)
            } else {
                (0.0, degrees_of_freedom)
            }
        }
        None => (0.0, degrees_of_freedom),
    }
}

fn solve_linear_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(Vec::new());
    }

    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let pivot_tolerance = LOGRANK_VARIANCE_TOLERANCE * scale;

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = matrix[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate_abs = matrix[row * n + col].abs();
            if candidate_abs > pivot_abs {
                pivot_abs = candidate_abs;
                pivot_row = row;
            }
        }
        if pivot_abs <= pivot_tolerance || !pivot_abs.is_finite() {
            return None;
        }
        if pivot_row != col {
            for swap_col in 0..n {
                matrix.swap(col * n + swap_col, pivot_row * n + swap_col);
            }
            rhs.swap(col, pivot_row);
        }

        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / matrix[col * n + col];
            matrix[row * n + col] = 0.0;
            for inner_col in (col + 1)..n {
                matrix[row * n + inner_col] -= factor * matrix[col * n + inner_col];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..n {
            value -= matrix[row * n + col] * solution[col];
        }
        let diag = matrix[row * n + row];
        if diag.abs() <= pivot_tolerance || !diag.is_finite() {
            return None;
        }
        solution[row] = value / diag;
        if !solution[row].is_finite() {
            return None;
        }
    }

    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_covariance_statistic_matches_two_group_shortcut() {
        let observed = vec![2.0, 1.0];
        let expected = vec![0.8333333333333333, 2.1666666666666665];
        let variance = vec![
            0.4722222222222222,
            -0.4722222222222222,
            -0.4722222222222222,
            0.4722222222222222,
        ];

        let (statistic, df) =
            logrank_statistic_from_flat_covariance(&observed, &expected, &variance, 2);

        assert_eq!(df, 1);
        assert!((statistic - 2.8823529411764715).abs() < 1e-12);
    }

    #[test]
    fn flat_covariance_statistic_handles_degenerate_inputs() {
        let (statistic, df) =
            logrank_statistic_from_flat_covariance(&[0.0, 0.0], &[0.0, 0.0], &[0.0; 4], 2);

        assert_eq!(statistic, 0.0);
        assert_eq!(df, 0);
    }
}
