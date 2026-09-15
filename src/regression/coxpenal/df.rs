//! `coxpenal.df` (`R/coxpenal.df.R`): the effective degrees of freedom of
//! each term of a penalised Cox model, after Gray (1992), together with the
//! two variance estimates `H^{-1}` (`var`) and `H^{-1} I H^{-1}` (`var2`),
//! where `H` is the penalised information and `I` the information of the
//! partial likelihood alone.
//!
//! The inputs are the dense slices of the Cholesky factor of `H` and of its
//! inverse (`hmat`, `hinv`: `nvar` rows, `nfrail + nvar` columns, as
//! [`super::kernel`] returns them — R's matrices transposed), `D^{-1}` of
//! the factorisation (`fdiag`) and the penalty second derivatives.  With a
//! sparse frailty term only the dense corner of the inverse is formed.

use super::kernel::PenaltyShape;
use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::coxph_wtest::wald_tests;
use ndarray::Array2;

/// What `coxpenal.df` returns.
#[derive(Debug, Clone)]
pub(super) struct TermDf {
    /// `H^{-1}` for the dense coefficients (empty without them).
    pub var: Array2<f64>,
    /// `H^{-1} I H^{-1}`.
    pub var2: Array2<f64>,
    /// Degrees of freedom per model term.
    pub df: Vec<f64>,
    /// Trace of the `H^{-1}` block per model term.
    pub trh: Vec<f64>,
    /// The diagonal of the frailty block of `H^{-1}` (`fvar`).
    pub fvar: Option<Vec<f64>>,
}

/// The inputs of [`coxpenal_df`].
pub(super) struct DfInput<'a> {
    pub hmat: &'a Array2<f64>,
    pub hinv: &'a Array2<f64>,
    pub fdiag: &'a [f64],
    /// Design columns (in the dense matrix) of each model term; the sparse
    /// term's entry is not read.
    pub assign: &'a [Vec<usize>],
    pub shape: PenaltyShape,
    /// The sparse term's second derivative (`coxlist1$second`).
    pub pen1: &'a [f64],
    /// The dense terms' second derivative (`coxlist2$second`: `nvar` or
    /// `nvar * nvar` entries).
    pub pen2: &'a [f64],
    /// Position of the sparse term in `assign`.
    pub sparse_term: Option<usize>,
}

/// `coxph.wtest(var, b)$solve` with R's default tolerance: the generalised
/// solve `var^{-1} b`.
fn generalised_solve(var: &Array2<f64>, b: &Array2<f64>) -> SurvivalResult<Array2<f64>> {
    let solve = wald_tests(var, b, 1e-9)?.solve;
    Array2::from_shape_vec(
        (b.nrows(), b.ncols()),
        solve.into_iter().flatten().collect(),
    )
    .map_err(|err| SurvivalError::computation(err.to_string()))
}

fn submatrix(matrix: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), rows.len()), |(i, j)| {
        matrix[(rows[i], rows[j])]
    })
}

fn trace(matrix: &Array2<f64>) -> f64 {
    matrix.diag().sum()
}

/// `sum(diag(C_jj^{-1} var2_jj))`: the degrees of freedom of one term.
fn term_df(c: &Array2<f64>, var2: &Array2<f64>, columns: &[usize]) -> SurvivalResult<f64> {
    let solve = generalised_solve(&submatrix(c, columns), &submatrix(var2, columns))?;
    Ok(trace(&solve))
}

/// `pen2` as an `nvar x nvar` penalty matrix: the diagonal when it has
/// `nvar` entries, else R's column-major `matrix(pen2, nvar)`.
fn penalty_matrix(pen2: &[f64], nvar: usize) -> Array2<f64> {
    if pen2.len() == nvar {
        Array2::from_diag(&ndarray::Array1::from_vec(pen2.to_vec()))
    } else {
        Array2::from_shape_fn((nvar, nvar), |(i, j)| pen2[j * nvar + i])
    }
}

pub(super) fn coxpenal_df(input: DfInput<'_>) -> SurvivalResult<TermDf> {
    let nvar = input.hmat.nrows();
    let nvar2 = input.fdiag.len();
    let nf = nvar2 - nvar;
    let fdiag = input.fdiag;
    let (hmat, hinv) = (input.hmat, input.hinv);
    let dinv: Vec<f64> = fdiag
        .iter()
        .map(|&d| if d == 0.0 { 0.0 } else { 1.0 / d })
        .collect();
    // t(hmat) %*% (dinv * hmat): the dense block of H (the full matrix when
    // there is no sparse term).
    let h22 = Array2::from_shape_fn((nvar, nvar), |(i, k)| {
        (0..nvar2)
            .map(|j| hmat[(i, j)] * dinv[j] * hmat[(k, j)])
            .sum::<f64>()
    });

    if input.shape.sparse && nvar == 0 {
        // Only the sparse term: everything is diagonal.
        let hdiag: Vec<f64> = fdiag.iter().map(|d| 1.0 / d).collect();
        let df = hdiag
            .iter()
            .zip(input.pen1)
            .zip(fdiag)
            .map(|((h, p), d)| (h - p) * d)
            .sum();
        return Ok(TermDf {
            var: Array2::zeros((0, 0)),
            var2: Array2::zeros((0, 0)),
            df: vec![df],
            trh: vec![fdiag.iter().sum()],
            fvar: Some(fdiag.to_vec()),
        });
    }

    if !input.shape.sparse {
        // Dense terms only: H^{-1} = hinv D^{-1} hinv'.
        let hinv_full = Array2::from_shape_fn((nvar, nvar), |(j, k)| {
            (0..nvar)
                .map(|i| hinv[(i, j)] * fdiag[i] * hinv[(i, k)])
                .sum::<f64>()
        });
        let imat = &h22 - &penalty_matrix(input.pen2, nvar);
        let var2 = hinv_full.dot(&imat).dot(&hinv_full);
        let (df, trh) = if input.assign.len() == 1 {
            (vec![(&imat * &hinv_full).sum()], vec![trace(&hinv_full)])
        } else {
            let mut df = Vec::new();
            let mut trh = Vec::new();
            for columns in input.assign {
                df.push(term_df(&hinv_full, &var2, columns)?);
                trh.push(columns.iter().map(|&c| hinv_full[(c, c)]).sum());
            }
            (df, trh)
        };
        return Ok(TermDf {
            var: hinv_full,
            var2,
            df,
            trh,
            fvar: None,
        });
    }

    // A sparse term plus other variables (see the notation of Gray's paper):
    // A.diag = diag of the frailty block of H^{-1}, B its cross block, C
    // the dense corner.
    let d1 = &fdiag[..nf];
    let d2 = &fdiag[nf..];
    let a_diag: Vec<f64> = (0..nf)
        .map(|j| d1[j] + (0..nvar).map(|i| hinv[(i, j)].powi(2) * d2[i]).sum::<f64>())
        .collect();
    let b = Array2::from_shape_fn((nf, nvar), |(j, k)| {
        (0..nvar)
            .map(|i| hinv[(i, j)] * d2[i] * hinv[(i, nf + k)])
            .sum::<f64>()
    });
    let c = Array2::from_shape_fn((nvar, nvar), |(l, k)| {
        (0..nvar)
            .map(|i| hinv[(i, nf + l)] * d2[i] * hinv[(i, nf + k)])
            .sum::<f64>()
    });
    let mut var2 = Array2::from_shape_fn((nvar, nvar), |(l, k)| {
        c[(l, k)]
            - (0..nf)
                .map(|j| b[(j, l)] * input.pen1[j] * b[(j, k)])
                .sum::<f64>()
    });
    // trace[B' A^{-1} B P2], zero without dense penalties.
    let mut temp2 = 0.0;
    if input.shape.dense {
        let h22_inverse = generalised_solve(&h22, &Array2::eye(nvar))?;
        let temp = &c - &h22_inverse;
        let p2 = penalty_matrix(input.pen2, nvar);
        var2 = &var2 - &c.dot(&p2).dot(&c);
        // R: sum(diag(temp) * pen2) for a diagonal penalty and
        // sum(diag(temp * pen2)) (an elementwise product) for a full one;
        // both are the diagonal-by-diagonal sum.
        temp2 = (0..nvar).map(|l| temp[(l, l)] * p2[(l, l)]).sum();
    }
    let mut df = Vec::new();
    let mut trh = Vec::new();
    for (term, columns) in input.assign.iter().enumerate() {
        if input.sparse_term == Some(term) {
            let penalised: f64 = a_diag.iter().zip(input.pen1).map(|(a, p)| a * p).sum();
            df.push(nf as f64 - (penalised + temp2));
            trh.push(a_diag.iter().sum());
        } else {
            df.push(term_df(&c, &var2, columns)?);
            trh.push(columns.iter().map(|&j| c[(j, j)]).sum());
        }
    }
    Ok(TermDf {
        var: c,
        var2,
        df,
        trh,
        fvar: Some(a_diag),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_df_is_the_trace_of_the_smoother() {
        // H = [[2, 0.5], [0.5, 3]] = F D F' with a penalty of 1 on each
        // coefficient: I = H - P.
        let h = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 3.0]).unwrap();
        let f21 = 0.5 / 2.0;
        let d = [2.0, 3.0 - f21 * f21 * 2.0];
        let hmat = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, f21, 1.0]).unwrap();
        let hinv = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, -f21, 1.0]).unwrap();
        let fdiag = [1.0 / d[0], 1.0 / d[1]];
        let out = coxpenal_df(DfInput {
            hmat: &hmat,
            hinv: &hinv,
            fdiag: &fdiag,
            assign: &[vec![0, 1]],
            shape: PenaltyShape {
                sparse: false,
                dense: true,
                full_imat: false,
            },
            pen1: &[],
            pen2: &[1.0, 1.0],
            sparse_term: None,
        })
        .unwrap();
        let hinv_full = Array2::from_shape_vec(
            (2, 2),
            vec![3.0 / 5.75, -0.5 / 5.75, -0.5 / 5.75, 2.0 / 5.75],
        )
        .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out.var[(i, j)] - hinv_full[(i, j)]).abs() < 1e-12);
            }
        }
        let imat = &h - &Array2::<f64>::eye(2);
        let expected_df = (&imat * &hinv_full).sum();
        assert!((out.df[0] - expected_df).abs() < 1e-12);
        assert!((out.trh[0] - trace(&hinv_full)).abs() < 1e-12);
        let var2 = hinv_full.dot(&imat).dot(&hinv_full);
        assert!((out.var2[(0, 1)] - var2[(0, 1)]).abs() < 1e-12);

        // Two terms: the per-term solve reproduces the same total.
        let split = coxpenal_df(DfInput {
            hmat: &hmat,
            hinv: &hinv,
            fdiag: &fdiag,
            assign: &[vec![0], vec![1]],
            shape: PenaltyShape {
                sparse: false,
                dense: true,
                full_imat: false,
            },
            pen1: &[],
            pen2: &[1.0, 1.0],
            sparse_term: None,
        })
        .unwrap();
        assert_eq!(split.df.len(), 2);
        assert!((split.df[0] - var2[(0, 0)] / hinv_full[(0, 0)]).abs() < 1e-12);
    }

    #[test]
    fn sparse_only_df_uses_the_diagonal() {
        let empty = Array2::zeros((0, 2));
        let out = coxpenal_df(DfInput {
            hmat: &empty,
            hinv: &empty,
            fdiag: &[0.5, 0.25],
            assign: &[vec![0]],
            shape: PenaltyShape {
                sparse: true,
                dense: false,
                full_imat: false,
            },
            pen1: &[1.0, 1.0],
            pen2: &[],
            sparse_term: Some(0),
        })
        .unwrap();
        // df = sum((1/fdiag - pen1) * fdiag) = (2 - 1) 0.5 + (4 - 1) 0.25.
        assert!((out.df[0] - 1.25).abs() < 1e-12);
        assert!((out.trh[0] - 0.75).abs() < 1e-12);
        assert_eq!(out.fvar, Some(vec![0.5, 0.25]));
    }
}
