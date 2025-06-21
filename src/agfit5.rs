// use extendr_api::prelude::*;
use std::f64::{EPSILON, INFINITY};

struct CoxState {
    covar: Vec<Vec<f64>>,
    cmat: Vec<Vec<f64>>,
    cmat2: Vec<Vec<f64>>,
    a: Vec<f64>,
    oldbeta: Vec<f64>,
    a2: Vec<f64>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    event: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    tmean: Vec<f64>,
    ptype: i32,
    pdiag: i32,
    ipen: Vec<f64>,
    upen: Vec<f64>,
    zflag: Vec<i32>,
    frail: Vec<i32>,
    score: Vec<f64>,
}

impl CoxState {
    fn new(
        nused: usize,
        nvar: usize,
        nfrail: usize,
        yy: &[f64],
        covar2: &[f64],
        offset2: &[f64],
        weights2: &[f64],
        strata: &[i32],
        sort: &[i32],
        ptype: i32,
        pdiag: i32,
        frail2: &[i32],
    ) -> Self {
        // Initialization logic similar to agfit5a
        let mut covar = vec![vec![0.0; nused]; nvar];
        let mut k = 0;
        for i in 0..nvar {
            for j in 0..nused {
                covar[i][j] = covar2[k];
                k += 1;
            }
        }

        let mut state = CoxState {
            covar,
            cmat: vec![vec![0.0; nvar + 1]; nvar + nfrail],
            cmat2: vec![vec![0.0; nvar + 1]; nvar + nfrail],
            a: vec![0.0; 4 * (nvar + nfrail) + 5 * nused],
            oldbeta: vec![0.0; nvar + nfrail],
            a2: vec![0.0; nvar + nfrail],
            offset: offset2.to_vec(),
            weights: weights2.to_vec(),
            event: yy[2 * nused..3 * nused].iter().map(|&x| x as i32).collect(),
            start: yy[0..nused].to_vec(),
            stop: yy[nused..2 * nused].to_vec(),
            sort1: sort[0..nused].iter().map(|&x| (x - 1) as usize).collect(),
            sort2: sort[nused..2 * nused]
                .iter()
                .map(|&x| (x - 1) as usize)
                .collect(),
            tmean: vec![0.0; nvar + nfrail],
            ptype,
            pdiag,
            ipen: vec![0.0; nfrail.max(nvar * nvar)],
            upen: vec![0.0; nfrail.max(nvar)],
            zflag: vec![0; nvar.max(2)],
            frail: frail2.to_vec(),
            score: vec![0.0; nused],
        };

        // Center covariates
        for i in 0..nvar {
            let mean = state.covar[i].iter().sum::<f64>() / nused as f64;
            for val in &mut state.covar[i] {
                *val -= mean;
            }
        }

        state
    }

    fn update(&mut self, beta: &mut [f64], u: &mut [f64], imat: &mut [f64], loglik: &mut f64) {
        // Main iteration logic similar to agfit5b
        let nvar = beta.len();
        let nfrail = self.frail.len();
        let nvar2 = nvar + nfrail;

        // Reset working arrays
        self.a.iter_mut().for_each(|x| *x = 0.0);
        self.a2.iter_mut().for_each(|x| *x = 0.0);

        // Compute scores and update risk sets
        for person in 0..self.weights.len() {
            let mut zbeta = self.offset[person];
            for i in 0..nvar {
                zbeta += beta[i] * self.covar[i][person];
            }
            self.score[person] = zbeta;
        }

        // Main accumulation loop
        let mut istrat = 0;
        let mut indx2 = 0;
        let mut denom = 0.0;

        for person in 0..self.weights.len() {
            // Risk set updates and calculations
            // ... (detailed risk set management)
        }

        // Update beta using Newton-Raphson
        let mut imat_matrix = vec![0.0; nvar2 * nvar2];
        // ... matrix operations and Cholesky decomposition

        // Handle convergence checks
        // ... step halving and iteration control
    }
}

// #[extendr]
// fn agfit5a(
//     nused: i32,
//     nvar: i32,
//     yy: Vec<f64>,
//     covar2: Vec<f64>,
//     offset2: Vec<f64>,
//     weights2: Vec<f64>,
//     strata: Vec<i32>,
//     sort: Vec<i32>,
//     mut means: Vec<f64>,
//     mut beta: Vec<f64>,
//     u: Vec<f64>,
//     loglik: f64,
//     method: i32,
//     ptype: i32,
//     pdiag: i32,
//     nfrail: i32,
//     frail2: Vec<i32>,
//     docenter: Vec<i32>,
// ) -> Robj {
//     let nused = nused as usize;
//     let nvar = nvar as usize;
//     let nfrail = nfrail as usize;
//
//     let state = CoxState::new(
//         nused, nvar, nfrail, &yy, &covar2, &offset2, &weights2, &strata, &sort, ptype, pdiag,
//         &frail2,
//     );
//
//     // Compute initial log-likelihood
//     let mut loglik = 0.0;
//     for person in 0..nused {
//         let mut zbeta = state.offset[person];
//         for i in 0..nvar {
//             zbeta += beta[i] * state.covar[i][person];
//         }
//         loglik += state.weights[person] * zbeta;
//     }
//
//     // Return state as external pointer
//     let external_ptr = ExternalPtr::new(state);
//     external_ptr.into()
// }

// #[extendr]
// fn agfit5b(
//     mut state_ptr: ExternalPtr<CoxState>,
//     maxiter: i32,
//     eps: f64,
//     tolerch: f64,
//     method: i32,
//     mut beta: Vec<f64>,
//     mut u: Vec<f64>,
//     mut imat: Vec<f64>,
//     mut loglik: f64,
//     mut flag: i32,
//     mut fbeta: Vec<f64>,
//     mut fdiag: Vec<f64>,
// ) -> Robj {
//     let state = state_ptr.as_mut();
//     let nvar = beta.len();
//     let nfrail = fbeta.len();
//
//     for iter in 0..maxiter {
//         state.update(&mut beta, &mut u, &mut imat, &mut loglik);
//
//         // Check convergence
//         if (state
//             .oldbeta
//             .iter()
//             .zip(&beta)
//             .all(|(a, b)| (a - b).abs() < eps))
//         {
//             flag = 0;
//             break;
//         }
//     }
//
//     list!(
//         beta = beta,
//         u = u,
//         imat = imat,
//         loglik = loglik,
//         flag = flag,
//         fbeta = fbeta,
//         fdiag = fdiag
//     )
//     .into()
// }

// #[extendr]
// fn agfit5c(state: ExternalPtr<CoxState>) {
//     // Rust's drop mechanism will automatically clean up the state
// }

// extendr_module! {
//     mod coxph5;
//     fn agfit5a;
//     fn agfit5b;
//     fn agfit5c;
// }
