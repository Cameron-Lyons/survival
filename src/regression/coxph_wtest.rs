use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

#[derive(Debug)]
struct CoxphWtestFactorization {
    factors: Vec<f64>,
    n: usize,
    rank: usize,
    contracted_updates: bool,
}

impl CoxphWtestFactorization {
    fn decompose(matrix: &[Vec<f64>], toler_chol: f64, contracted_updates: bool) -> PyResult<Self> {
        let n = matrix.len();
        if matrix.iter().any(|row| row.len() != n) {
            return Err(value_error("First argument must be a square matrix"));
        }
        if matrix.iter().flatten().any(|value| !value.is_finite()) {
            return Err(value_error("infinite argument in coxph.wtest"));
        }
        if !toler_chol.is_finite() {
            return Err(value_error("toler_chol must be finite"));
        }

        let mut factors = vec![0.0; n * n];
        let mut epsilon = 0.0_f64;
        for row in 0..n {
            epsilon = epsilon.max(matrix[row][row]);
            for col in 0..=row {
                factors[row * n + col] = matrix[row][col];
            }
        }
        epsilon = if epsilon == 0.0 {
            toler_chol
        } else {
            epsilon * toler_chol
        };

        for pivot_col in 0..n {
            let pivot_index = pivot_col * n + pivot_col;
            let pivot = factors[pivot_index];
            if !pivot.is_finite() || pivot < epsilon {
                factors[pivot_index] = 0.0;
                continue;
            }

            for row in (pivot_col + 1)..n {
                let row_pivot_index = row * n + pivot_col;
                let multiplier = factors[row_pivot_index] / pivot;
                factors[row_pivot_index] = multiplier;
                if contracted_updates {
                    factors[row * n + row] =
                        (-(multiplier * multiplier)).mul_add(pivot, factors[row * n + row]);
                } else {
                    factors[row * n + row] -= multiplier * multiplier * pivot;
                }
                for target_row in (row + 1)..n {
                    let target_index = target_row * n + row;
                    if contracted_updates {
                        factors[target_index] = (-multiplier)
                            .mul_add(factors[target_row * n + pivot_col], factors[target_index]);
                    } else {
                        factors[target_index] -= multiplier * factors[target_row * n + pivot_col];
                    }
                }
            }
        }

        let rank = (0..n)
            .filter(|&index| factors[index * n + index] > 0.0)
            .count();

        Ok(Self {
            factors,
            n,
            rank,
            contracted_updates,
        })
    }

    fn solve(&self, rhs: &[f64]) -> PyResult<Vec<f64>> {
        if rhs.len() != self.n {
            return Err(value_error("Argument lengths do not match"));
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(value_error("infinite argument in coxph.wtest"));
        }

        let mut solution = rhs.to_vec();
        for row in 0..self.n {
            let mut value = solution[row];
            if self.contracted_updates {
                let rounded_end = row / 8 * 8;
                for (col, &known) in solution.iter().take(rounded_end).enumerate() {
                    value += -known * self.factors[row * self.n + col];
                }
                for (col, &known) in solution.iter().enumerate().take(row).skip(rounded_end) {
                    value = (-known).mul_add(self.factors[row * self.n + col], value);
                }
            } else {
                for (col, &known) in solution.iter().take(row).enumerate() {
                    value -= known * self.factors[row * self.n + col];
                }
            }
            solution[row] = value;
        }
        for row in (0..self.n).rev() {
            let diagonal = self.factors[row * self.n + row];
            if diagonal == 0.0 {
                solution[row] = 0.0;
                continue;
            }
            let mut value = solution[row] / diagonal;
            for (col, &known) in solution.iter().enumerate().skip(row + 1) {
                if self.contracted_updates {
                    value = (-known).mul_add(self.factors[col * self.n + row], value);
                } else {
                    value -= known * self.factors[col * self.n + row];
                }
            }
            solution[row] = value;
        }
        Ok(solution)
    }
}

fn coxph_wtest_core_with_updates(
    matrix: &[Vec<f64>],
    rhs_columns: &[Vec<f64>],
    toler_chol: f64,
    contracted_updates: bool,
) -> PyResult<(Vec<f64>, usize, Vec<Vec<f64>>)> {
    let factorization = CoxphWtestFactorization::decompose(matrix, toler_chol, contracted_updates)?;
    let mut tests = Vec::with_capacity(rhs_columns.len());
    let mut solve_rows = vec![vec![0.0; rhs_columns.len()]; matrix.len()];

    for (column_index, rhs) in rhs_columns.iter().enumerate() {
        let solution = factorization.solve(rhs)?;
        let test = if factorization.contracted_updates {
            let rounded_end = rhs.len() / 8 * 8;
            let mut test = 0.0;
            for (&value, &coefficient) in rhs[..rounded_end]
                .iter()
                .zip(solution[..rounded_end].iter())
            {
                test += value * coefficient;
            }
            for (&value, &coefficient) in rhs[rounded_end..]
                .iter()
                .zip(solution[rounded_end..].iter())
            {
                test = value.mul_add(coefficient, test);
            }
            test
        } else {
            rhs.iter()
                .zip(solution.iter())
                .map(|(&value, &coefficient)| value * coefficient)
                .sum()
        };
        tests.push(test);
        for (row, &value) in solution.iter().enumerate() {
            solve_rows[row][column_index] = value;
        }
    }

    Ok((tests, factorization.rank, solve_rows))
}

pub(crate) fn coxph_wtest_core(
    matrix: &[Vec<f64>],
    rhs_columns: &[Vec<f64>],
    toler_chol: f64,
) -> PyResult<(Vec<f64>, usize, Vec<Vec<f64>>)> {
    coxph_wtest_core_with_updates(matrix, rhs_columns, toler_chol, true)
}

#[pyfunction]
#[pyo3(signature = (matrix, rhs_columns, toler_chol=1e-9))]
pub fn coxph_wtest(
    matrix: Vec<Vec<f64>>,
    rhs_columns: Vec<Vec<f64>>,
    toler_chol: f64,
) -> PyResult<(Vec<f64>, usize, Vec<Vec<f64>>)> {
    coxph_wtest_core(&matrix, &rhs_columns, toler_chol)
}

#[pyfunction]
#[pyo3(signature = (matrix, rhs_columns, toler_chol=1e-9))]
pub fn _coxph_wtest_stable(
    matrix: Vec<Vec<f64>>,
    rhs_columns: Vec<Vec<f64>>,
    toler_chol: f64,
) -> PyResult<(Vec<f64>, usize, Vec<Vec<f64>>)> {
    coxph_wtest_core_with_updates(&matrix, &rhs_columns, toler_chol, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    fn assert_rel_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            let tolerance = expected.abs().max(1.0) * 1e-12;
            assert!(
                (actual - expected).abs() < tolerance,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn matches_reference_full_rank_and_multiple_rhs() {
        let (tests, rank, solve) = coxph_wtest_core(
            &[vec![2.0, 0.5], vec![0.5, 1.0]],
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            1e-9,
        )
        .expect("factorization should succeed");

        assert_eq!(rank, 2);
        assert_close(&tests, &[4.0, 16.571428571428573]);
        assert_close(&solve[0], &[0.0, 0.5714285714285714]);
        assert_close(&solve[1], &[2.0, 3.7142857142857144]);
    }

    #[test]
    fn matches_reference_singular_and_indefinite_semantics() {
        let cases = [
            (
                vec![
                    vec![1.0, 2.0, 3.0],
                    vec![2.0, 4.0, 6.0],
                    vec![3.0, 6.0, 9.0],
                ],
                vec![1.0, 2.0, 3.0],
                1,
                vec![1.0, 0.0, 0.0],
            ),
            (
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 2.0, 0.0],
                    vec![0.0, 0.0, 3.0],
                ],
                vec![1.0, 2.0, 3.0],
                2,
                vec![0.0, 1.0, 1.0],
            ),
            (
                vec![vec![1.0, 2.0], vec![2.0, 1.0]],
                vec![1.0, 2.0],
                1,
                vec![1.0, 0.0],
            ),
        ];

        for (matrix, rhs, expected_rank, expected_solve) in cases {
            let (_, rank, solve) =
                coxph_wtest_core(&matrix, &[rhs], 1e-9).expect("factorization should succeed");
            assert_eq!(rank, expected_rank);
            assert_close(
                &solve.iter().map(|row| row[0]).collect::<Vec<_>>(),
                &expected_solve,
            );
        }
    }

    #[test]
    fn matches_r_column_major_lower_triangle_semantics() {
        let (tests, rank, solve) =
            coxph_wtest_core(&[vec![2.0, 0.25], vec![7.0, 1.0]], &[vec![1.0, 2.0]], 1e-9)
                .expect("factorization should succeed");

        assert_eq!(rank, 1);
        assert_close(&tests, &[0.5]);
        assert_close(
            &solve.iter().map(|row| row[0]).collect::<Vec<_>>(),
            &[0.5, 0.0],
        );
    }

    #[test]
    fn rejects_malformed_inputs() {
        assert!(coxph_wtest_core(&[vec![1.0, 2.0]], &[vec![1.0]], 1e-9).is_err());
        assert!(coxph_wtest_core(&[vec![1.0]], &[vec![1.0, 2.0]], 1e-9).is_err());
        assert!(coxph_wtest_core(&[vec![f64::INFINITY]], &[vec![1.0]], 1e-9).is_err());
    }

    #[test]
    fn matches_reference_negative_tolerance_zero_pivot_semantics() {
        let (tests, rank, solve) = coxph_wtest_core(
            &[
                vec![6.0, 0.0, 2.0],
                vec![0.0, 0.0, 0.0],
                vec![2.0, 0.0, 3.0],
            ],
            &[
                vec![-1.0, -1.0, 0.0],
                vec![-4.0, -2.0, 2.0],
                vec![-2.0, -1.0, 1.0],
            ],
            -1.0,
        )
        .expect("factorization should succeed");

        assert_eq!(rank, 1);
        assert_close(&tests, &[1.0 / 6.0, 8.0 / 3.0, 2.0 / 3.0]);
        assert_close(&solve[0], &[-1.0 / 6.0, -2.0 / 3.0, -1.0 / 3.0]);
        assert_close(&solve[1], &[0.0, 0.0, 0.0]);
        assert_close(&solve[2], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn matches_reference_fused_near_singular_updates() {
        let (tests, rank, solve) = coxph_wtest_core(
            &[
                vec![9.0, -3.0, 3.0, 9.0],
                vec![-3.0, 1.0, -1.0, -3.0],
                vec![3.0, -1.0, 1.0, 3.0],
                vec![9.0, -3.0, 3.0, 9.0],
            ],
            &[vec![3.0, 1.0, -2.0, 4.0], vec![3.0, -6.0, 3.0, 2.0]],
            -1e-9,
        )
        .expect("factorization should succeed");

        assert_eq!(rank, 2);
        assert_rel_close(
            &tests,
            &[72_057_594_037_927_936.0, 450_359_962_737_049_600.0],
        );
        assert_rel_close(
            &solve[0],
            &[12_009_599_006_321_322.0, -30_023_997_515_803_304.0],
        );
        assert_rel_close(
            &solve[1],
            &[36_028_797_018_963_968.0, -90_071_992_547_409_920.0],
        );
        assert_close(&solve[2], &[0.0, 0.0]);
        assert_close(&solve[3], &[0.0, 0.0]);
    }
}
