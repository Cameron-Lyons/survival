pub(crate) fn basis_row(knots: &[f64], x: f64, order: usize) -> Vec<f64> {
    let n_basis = knots.len() - order;
    let mut values = vec![0.0; knots.len() - 1];
    for idx in 0..knots.len() - 1 {
        if (knots[idx] <= x && x < knots[idx + 1])
            || (x == knots[knots.len() - 1] && knots[idx] <= x && x <= knots[idx + 1])
        {
            values[idx] = 1.0;
        }
    }

    for current_order in 2..=order {
        let mut next_values = vec![0.0; knots.len() - current_order];
        for idx in 0..next_values.len() {
            let left_denominator = knots[idx + current_order - 1] - knots[idx];
            let right_denominator = knots[idx + current_order] - knots[idx + 1];
            let left = if left_denominator == 0.0 {
                0.0
            } else {
                (x - knots[idx]) / left_denominator * values[idx]
            };
            let right = if right_denominator == 0.0 {
                0.0
            } else {
                (knots[idx + current_order] - x) / right_denominator * values[idx + 1]
            };
            next_values[idx] = left + right;
        }
        values = next_values;
    }
    values.truncate(n_basis);
    values
}

pub(crate) fn derivative_row(knots: &[f64], x: f64, order: usize, derivative: usize) -> Vec<f64> {
    if derivative == 0 {
        return basis_row(knots, x, order);
    }
    if order <= derivative {
        return vec![0.0; knots.len().saturating_sub(order)];
    }

    let lower_order = derivative_row(knots, x, order - 1, derivative - 1);
    let n_basis = knots.len() - order;
    (0..n_basis)
        .map(|idx| {
            let left_denominator = knots[idx + order - 1] - knots[idx];
            let right_denominator = knots[idx + order] - knots[idx + 1];
            let left = if left_denominator == 0.0 {
                0.0
            } else {
                (order - 1) as f64 / left_denominator * lower_order[idx]
            };
            let right = if right_denominator == 0.0 {
                0.0
            } else {
                (order - 1) as f64 / right_denominator * lower_order[idx + 1]
            };
            left - right
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_boundary_knots_have_expected_values_and_derivatives() {
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 7.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(
            basis_row(&knots, 0.0, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            basis_row(&knots, 8.0, 4),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            derivative_row(&knots, 0.0, 4, 2),
            vec![6.0, -7.5, 1.5, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            derivative_row(&knots, 8.0, 4, 2),
            vec![0.0, 0.0, 0.0, 0.0, 1.5, -7.5, 6.0]
        );
    }
}
