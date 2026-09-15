//! Penalised Cox models: R survival's `coxpenal.fit` (`R/coxpenal.fit.R`),
//! the fitter behind `coxph()` when the formula has `ridge()`, `pspline()`
//! or `frailty()` terms, producing the `coxph.penal` object.
//!
//! The pieces follow the R sources: [`penalty`] holds the terms and their
//! `pfun`s, [`control`] the `cfun`s that choose each term's smoothing
//! parameter, [`kernel`] the C Newton-Raphson kernels (`coxfit5.c`,
//! `agfit5.c` with `cholesky3`/`chsolve3`/`chinv3`) and [`df`] the
//! degrees-of-freedom computation (`coxpenal.df`).  This module is the outer
//! loop: it removes a sparse frailty column from the design, composes the
//! penalty callbacks (`f.expr1`/`f.expr2`, the persistent `coxlist1`/
//! `coxlist2`), iterates the inner fit and the `cfun`s over `theta`, restarts
//! each inner fit from the solution of the closest earlier `theta`, and
//! assembles the fit (`coxph.penal` plus the `coxph()` post-processing of
//! `R/coxph.R`: `n`, `nevent`, the Wald test).
//!
//! The fitted dense part is kept as a [`CoxPHFit`] so that `predict()`,
//! the residual types and `survfit()` of a `coxph.penal` object come from
//! the same code as for a plain Cox model; `survfit.coxph` drops the sparse
//! frailty from the risk scores of a frailty model, which
//! [`CoxpenalFit::survfit`] reproduces.

mod control;
mod df;
mod kernel;
mod penalty;

#[cfg(feature = "python")]
pub use penalty::CallbackPenalty;
pub use penalty::{
    CoxPenaltyTerms, FrailtyFamily, FrailtyMethod, FrailtyPenalty, PenaltyTerm, PsplineMethod,
    PsplinePenalty, RidgePenalty,
};

use self::control::{Control, ControlInput, ControlState};
use self::df::{DfInput, TermDf, coxpenal_df};
use self::kernel::{InnerFit, Kernel, KernelData, PenaltyCallback, PenaltyShape};
use self::penalty::Pparm;
use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_finite;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::{
    Basehaz, CoxNewData, CoxPHFit, CoxSurvfitCurve, CoxphData, CoxphOptions, SurvfitOptions,
};
use crate::regression::coxph_diagnostics::martingale_residuals;
use crate::regression::coxph_wtest::wald_statistic;
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// `coxph.control()$outer.max`.
pub const COXPENAL_OUTER_MAX: usize = 10;

/// One model term: its design columns (R's `assign` entry) and, for a
/// penalised term, its penalty (`pcols`/`pattr`).
#[derive(Debug, Clone)]
pub struct ModelTerm {
    pub columns: Vec<usize>,
    pub penalty: Option<PenaltyTerm>,
}

/// Validated inputs of a penalised Cox fit.  `x` holds every design column,
/// including the single column of group codes of a sparse frailty term.
#[derive(Debug, Clone)]
pub struct CoxpenalData {
    pub time: Vec<f64>,
    pub entry: Option<Vec<f64>>,
    pub status: Vec<i32>,
    pub x: Array2<f64>,
    pub weights: Option<Vec<f64>>,
    pub strata: Option<Vec<i32>>,
    pub offset: Option<Vec<f64>>,
    /// The model terms in formula order; every column belongs to exactly
    /// one term and at least one term is penalised.
    pub terms: Vec<ModelTerm>,
}

impl CoxpenalData {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        time: Vec<f64>,
        entry: Option<Vec<f64>>,
        status: Vec<i32>,
        x: Array2<f64>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        terms: Vec<ModelTerm>,
    ) -> SurvivalResult<Self> {
        let base = CoxphData::try_new(time, entry, status, x, weights, strata, offset)?;
        let ncol = base.x.ncols();
        let mut owner = vec![None; ncol];
        for (t, term) in terms.iter().enumerate() {
            if term.columns.is_empty() {
                return Err(SurvivalError::invalid_input(format!(
                    "term {t} has no columns"
                )));
            }
            for &column in &term.columns {
                if column >= ncol {
                    return Err(SurvivalError::invalid_input(format!(
                        "term {t} refers to column {column}, but x has {ncol}"
                    )));
                }
                if owner[column].replace(t).is_some() {
                    return Err(SurvivalError::invalid_input(format!(
                        "column {column} belongs to more than one term"
                    )));
                }
            }
        }
        if let Some(column) = owner.iter().position(Option::is_none) {
            return Err(SurvivalError::invalid_input(format!(
                "column {column} belongs to no term"
            )));
        }
        if terms.iter().all(|term| term.penalty.is_none()) {
            return Err(SurvivalError::invalid_input("Invalid pcols or pattr arg"));
        }
        let sparse: Vec<&ModelTerm> = terms
            .iter()
            .filter(|term| term.penalty.as_ref().is_some_and(PenaltyTerm::is_sparse))
            .collect();
        if sparse.len() > 1 {
            return Err(SurvivalError::invalid_input(
                "Only one sparse penalty term allowed",
            ));
        }
        if sparse.first().is_some_and(|term| term.columns.len() > 1) {
            return Err(SurvivalError::invalid_input(
                "Sparse term must be single column",
            ));
        }
        Ok(Self {
            time: base.time,
            entry: base.entry,
            status: base.status,
            x: base.x,
            weights: base.weights,
            strata: base.strata,
            offset: base.offset,
            terms,
        })
    }

    pub fn n(&self) -> usize {
        self.time.len()
    }
}

/// Fitting options: `coxph.control()` (`iter.max`, `outer.max`, `eps`,
/// `toler.chol`), `init` and `nocenter`.
#[derive(Debug, Clone)]
pub struct CoxpenalOptions {
    /// Breslow or Efron; the exact method has no penalised kernel.
    pub method: TieMethod,
    /// Initial coefficients: one per dense column, optionally followed by
    /// one per frailty group.
    pub init: Option<Vec<f64>>,
    /// Inner (Newton) iterations per outer step.
    pub iter_max: usize,
    /// Outer iterations over the smoothing parameters.
    pub outer_max: usize,
    pub eps: f64,
    pub toler_chol: f64,
    /// As in [`CoxphOptions`]: columns whose values all lie in this set are
    /// not centred.  (R evaluates the rule on the full design and reads the
    /// flags of the dense columns by position, which misaligns them when a
    /// sparse frailty column is not the last one; this port evaluates the
    /// rule on the dense columns themselves.)
    pub nocenter: Option<Vec<f64>>,
}

impl Default for CoxpenalOptions {
    fn default() -> Self {
        Self {
            method: TieMethod::Efron,
            init: None,
            iter_max: COX_MAX_ITER,
            outer_max: COXPENAL_OUTER_MAX,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler_chol: COX_RANK_TOLERANCE,
            nocenter: Some(vec![-1.0, 0.0, 1.0]),
        }
    }
}

/// The search history of one penalised term (`fit$history[[i]]`).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct PenaltyHistory {
    /// Index of the model term.
    #[pyo3(get)]
    pub term: usize,
    /// The last theta the search proposed (the fit used the previous one
    /// when the search stopped).
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub done: bool,
    /// One row per outer iteration (plus the known starting points of a
    /// `df` search), columns named by `columns`.
    #[pyo3(get)]
    pub history: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub columns: Vec<String>,
    /// The corrected log likelihood of a gamma frailty (`c.loglik`).
    #[pyo3(get)]
    pub c_loglik: Option<f64>,
    /// `frailty.controldf`'s bisection counter.
    #[pyo3(get)]
    pub half: Option<i64>,
}

/// A fitted penalised Cox model (R's `coxph.penal` object).
///
/// The dense coefficients live in `coxph`, a [`CoxPHFit`] whose
/// coefficients, variance (`H^{-1}`), means, linear predictors (including a
/// sparse frailty) and martingale residuals are those of the penalised fit;
/// its `predict()` and residual methods are R's for a `coxph.penal` object.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct CoxpenalFit {
    pub coxph: CoxPHFit,
    /// The same fit with the frailty removed from the linear predictors,
    /// present for a sparse frailty model: the risk scores `survfit.coxph`
    /// uses.
    curve_fit: Option<CoxPHFit>,
    /// `H^{-1} I H^{-1}` for the dense coefficients (`var2`).
    pub var2: Array2<f64>,
    /// Outer iterations and total inner iterations.
    #[pyo3(get)]
    pub iter: [usize; 2],
    /// Outer iterations whose inner loop used up `iter.max` (R warns about
    /// them).
    #[pyo3(get)]
    pub inner_failures: Vec<usize>,
    /// Effective degrees of freedom of each model term.
    #[pyo3(get)]
    pub df: Vec<f64>,
    /// The penalty at the first and the final outer iteration.
    #[pyo3(get)]
    pub penalty: [f64; 2],
    /// Per model term: 0 ordinary, 1 penalised, 2 sparse.
    #[pyo3(get)]
    pub pterms: Vec<u8>,
    /// R's `assign2`: the columns of each term in the dense design (a
    /// sparse term keeps its original column).
    #[pyo3(get)]
    pub assign2: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub history: Vec<PenaltyHistory>,
    /// The fitted frailties of a sparse term (`frail`) and the diagonal
    /// of their variance (`fvar`).
    #[pyo3(get)]
    pub frail: Option<Vec<f64>>,
    #[pyo3(get)]
    pub fvar: Option<Vec<f64>>,
    /// 0-based frailty group of each observation.
    #[pyo3(get)]
    pub frail_index: Option<Vec<usize>>,
    /// The last penalty evaluations (`coxlist1` for the sparse term,
    /// `coxlist2` for the dense ones); `cox.zph` reads the dense second
    /// derivative.
    #[pyo3(get)]
    pub coxlist1: Option<CoxPenaltyTerms>,
    #[pyo3(get)]
    pub coxlist2: Option<CoxPenaltyTerms>,
}

/// A penalised term with its fit-time state: `pparm`, `cfun`, the current
/// `theta` and the search state.
struct TermState<'a> {
    /// Position in `data.terms`.
    index: usize,
    term: &'a PenaltyTerm,
    /// Columns in the dense design (empty for the sparse term).
    columns: Vec<usize>,
    pparm: Pparm,
    control: Control,
    state: ControlState,
    /// Events per group (frailty terms).
    events_by_group: Vec<f64>,
}

/// The penalty callbacks `f.expr1`/`f.expr2` with their persistent
/// `coxlist1`/`coxlist2`.
struct Composer<'a> {
    terms: Vec<TermState<'a>>,
    sparse: Option<usize>,
    full_imat: bool,
    coxlist1: CoxPenaltyTerms,
    coxlist2: CoxPenaltyTerms,
}

impl PenaltyCallback for Composer<'_> {
    fn call(&mut self, which: i32, coef: &mut [f64]) -> SurvivalResult<&CoxPenaltyTerms> {
        if which == 1 {
            let sparse = self.sparse.expect("a sparse term exists");
            let term = &self.terms[sparse];
            let nfrail = coef.len();
            let value = term.term.evaluate(coef, term.state.theta, &term.pparm, 1)?;
            let list = &mut self.coxlist1;
            list.coef = value.coef;
            if !value.flag {
                list.first = value.first.iter().map(|v| -v).collect();
                list.second = value.second;
            }
            list.penalty = -value.penalty;
            list.flag = vec![value.flag];
            if list.coef.len() != nfrail
                || list.first.len() != nfrail
                || list.second.len() != nfrail
            {
                return Err(SurvivalError::computation("Incorrect length in coxlist1"));
            }
            coef.copy_from_slice(&list.coef);
            return Ok(&self.coxlist1);
        }
        let nvar = coef.len();
        let list = &mut self.coxlist2;
        list.coef = coef.to_vec();
        let mut pentot = 0.0;
        for term in self.terms.iter().filter(|term| !term.term.is_sparse()) {
            let pen_col = &term.columns;
            let p = pen_col.len();
            let coef_term: Vec<f64> = pen_col.iter().map(|&c| list.coef[c]).collect();
            let value = term
                .term
                .evaluate(&coef_term, term.state.theta, &term.pparm, 2)?;
            if value.coef.len() != p {
                return Err(SurvivalError::computation("Length error in coxlist2"));
            }
            for (&c, &b) in pen_col.iter().zip(&value.coef) {
                list.coef[c] = b;
            }
            if value.flag {
                for &c in pen_col {
                    list.flag[c] = true;
                }
            } else {
                if value.first.len() != p {
                    return Err(SurvivalError::computation("Length error in coxlist2"));
                }
                for (k, &c) in pen_col.iter().enumerate() {
                    list.flag[c] = false;
                    list.first[c] = -value.first[k];
                }
                let recycled = |k: usize| value.second[k % value.second.len()];
                if value.second.is_empty()
                    || (self.full_imat && (p * p) % value.second.len() != 0)
                    || (!self.full_imat && p % value.second.len() != 0)
                {
                    return Err(SurvivalError::computation("Length error in coxlist2"));
                }
                if self.full_imat {
                    // R's tmat[pen.col, pen.col] <- second fills the block
                    // column-major, recycling a diagonal-only vector (an R
                    // quirk kept for fidelity: such a term combined with a
                    // full-matrix term gets its diagonal replicated across
                    // the block).
                    for col in 0..p {
                        for row in 0..p {
                            list.second[pen_col[col] * nvar + pen_col[row]] =
                                recycled(col * p + row);
                        }
                    }
                } else {
                    for (k, &c) in pen_col.iter().enumerate() {
                        list.second[c] = recycled(k);
                    }
                }
            }
            pentot -= value.penalty;
        }
        list.penalty = pentot;
        coef.copy_from_slice(&list.coef);
        Ok(&self.coxlist2)
    }
}

/// The `cfun` and `cparm` of a term; `ncols` is the number of design
/// columns of the term, `n` the number of observations and `eps2` R's
/// fallback tolerance `sqrt(control$eps)`.
fn control_of(term: &PenaltyTerm, ncols: usize, n: usize, eps2: f64) -> SurvivalResult<Control> {
    let control = match term {
        PenaltyTerm::Ridge(ridge) => match ridge.theta {
            Some(theta) => Control::Fixed { theta },
            None => Control::Df {
                df: ridge.df.unwrap_or(ncols as f64 / 2.0),
                eps: ridge.eps,
                thetas: vec![0.0],
                dfs: vec![ncols as f64],
                guess: 1.0,
                gamma_correction: false,
            },
        },
        PenaltyTerm::Pspline(spline) => match spline.method {
            PsplineMethod::Fixed(theta) => Control::Fixed { theta },
            PsplineMethod::Df(df) => Control::Df {
                df,
                eps: spline.eps,
                thetas: vec![1.0, 0.0],
                dfs: vec![1.0, spline.nterm as f64],
                guess: 1.0 - df / spline.nterm as f64,
                gamma_correction: false,
            },
            PsplineMethod::Aic => Control::Aic {
                eps: spline.eps,
                init: vec![0.5, 0.95],
                lower: 0.0,
                upper: Some(1.0),
                caic: false,
                gamma_correction: false,
            },
        },
        PenaltyTerm::Frailty(frailty) => {
            let eps = frailty.eps.unwrap_or(eps2);
            let gamma = frailty.distribution == FrailtyFamily::Gamma;
            if let Some(init) = &frailty.init
                && init.len() != 2
            {
                return Err(SurvivalError::invalid_input(
                    "frailty init must hold two starting values for theta",
                ));
            }
            let df_control = |df: f64| Control::Df {
                df,
                eps,
                thetas: vec![0.0],
                dfs: vec![0.0],
                guess: 3.0 * df / n as f64,
                gamma_correction: gamma,
            };
            match frailty.method {
                FrailtyMethod::Fixed if gamma => Control::Gamma {
                    theta: frailty.theta,
                    eps,
                    init: frailty.init.clone(),
                },
                FrailtyMethod::Fixed => Control::Fixed {
                    theta: frailty.theta.expect("fixed frailty has theta"),
                },
                FrailtyMethod::Em => Control::Gamma {
                    theta: None,
                    eps,
                    init: frailty.init.clone(),
                },
                FrailtyMethod::Reml => Control::Gauss {
                    eps,
                    init: frailty.init.clone(),
                },
                FrailtyMethod::Aic => Control::Aic {
                    eps,
                    init: vec![0.1, 1.0],
                    lower: 0.0,
                    upper: None,
                    caic: frailty.caic,
                    gamma_correction: gamma,
                },
                FrailtyMethod::Df => df_control(frailty.df.expect("df method has df")),
            }
        }
        #[cfg(feature = "python")]
        PenaltyTerm::Callback(_) => Control::Fixed { theta: f64::NAN },
    };
    Ok(control)
}

/// `tapply(status, group, sum)`: events per group in ascending group order.
fn events_by_group(groups: &[i64], status: &[i32]) -> Vec<f64> {
    let mut sums = BTreeMap::new();
    for (&g, &s) in groups.iter().zip(status) {
        *sums.entry(g).or_insert(0.0) += f64::from(s);
    }
    sums.into_values().collect()
}

fn dot_row(x: &Array2<f64>, row: usize, coef: &[f64]) -> f64 {
    x.row(row).iter().zip(coef).map(|(v, b)| v * b).sum()
}

impl CoxpenalFit {
    /// `coxpenal.fit` followed by the `coxph()` post-processing.
    pub fn fit(data: CoxpenalData, options: CoxpenalOptions) -> SurvivalResult<Self> {
        if options.method == TieMethod::Exact {
            return Err(SurvivalError::invalid_input(
                "penalised Cox models support ties = 'breslow' or 'efron' only",
            ));
        }
        if options.outer_max == 0 {
            return Err(SurvivalError::invalid_input("invalid value for outer.max"));
        }
        let n = data.n();
        let ncol = data.x.ncols();
        let nevent = data.status.iter().filter(|&&s| s == 1).count();
        let n_eff = nevent as f64;
        let weights = data.weights.clone().unwrap_or_else(|| vec![1.0; n]);
        let offset = data.offset.clone().unwrap_or_else(|| vec![0.0; n]);
        let eps2 = options.eps.sqrt();

        // pterms: 0 ordinary, 1 penalised, 2 sparse.
        let pterms: Vec<u8> = data
            .terms
            .iter()
            .map(|term| match &term.penalty {
                None => 0,
                Some(penalty) if penalty.is_sparse() => 2,
                Some(_) => 1,
            })
            .collect();
        let sparse_term = pterms.iter().position(|&p| p == 2);
        let shape = PenaltyShape {
            sparse: sparse_term.is_some(),
            dense: pterms.contains(&1),
            full_imat: !data
                .terms
                .iter()
                .filter_map(|term| term.penalty.as_ref())
                .filter(|penalty| !penalty.is_sparse())
                .all(PenaltyTerm::is_diagonal),
        };

        // Remove the sparse term's column from the design.
        let fcol = sparse_term.map(|t| data.terms[t].columns[0]);
        let (xx, assign2, frailx, nfrail) = match fcol {
            Some(fcol) => {
                let keep: Vec<usize> = (0..ncol).filter(|&c| c != fcol).collect();
                let xx = Array2::from_shape_fn((n, ncol - 1), |(i, j)| data.x[(i, keep[j])]);
                let assign2: Vec<Vec<usize>> = data
                    .terms
                    .iter()
                    .map(|term| {
                        term.columns
                            .iter()
                            .map(|&c| if c > fcol { c - 1 } else { c })
                            .collect()
                    })
                    .collect();
                // match(x, sort(unique(x))), 0-based.
                let mut levels: Vec<f64> = data.x.column(fcol).to_vec();
                levels.sort_by(f64::total_cmp);
                levels.dedup();
                let frailx: Vec<usize> = data
                    .x
                    .column(fcol)
                    .iter()
                    .map(|v| levels.binary_search_by(|l| l.total_cmp(v)).expect("level"))
                    .collect();
                (xx, assign2, Some(frailx), levels.len())
            }
            None => (
                data.x.clone(),
                data.terms.iter().map(|term| term.columns.clone()).collect(),
                None,
                0,
            ),
        };
        let nvar = xx.ncols();

        // Initial values.
        let (init, finit) = match &options.init {
            None => (vec![0.0; nvar], vec![0.0; nfrail]),
            Some(init) if init.len() == nvar => (init.clone(), vec![0.0; nfrail]),
            Some(init) if init.len() == nvar + nfrail => {
                (init[..nvar].to_vec(), init[nvar..].to_vec())
            }
            Some(_) => {
                return Err(SurvivalError::invalid_input(
                    "Wrong length for inital values",
                ));
            }
        };
        validate_finite(&init, "init")?;
        validate_finite(&finit, "init")?;

        // The penalised terms: pparm, cfun and the first theta.
        let mut terms = Vec::new();
        for (index, term) in data.terms.iter().enumerate() {
            let Some(penalty) = &term.penalty else {
                continue;
            };
            let columns: Vec<usize> = if penalty.is_sparse() {
                Vec::new()
            } else {
                assign2[index].clone()
            };
            let x_term = Array2::from_shape_fn((n, columns.len()), |(i, j)| xx[(i, columns[j])]);
            let events = if let PenaltyTerm::Frailty(_) = penalty {
                let groups: Vec<i64> = match &frailx {
                    Some(frailx) if penalty.is_sparse() => {
                        frailx.iter().map(|&g| g as i64).collect()
                    }
                    _ => (0..n)
                        .map(|i| {
                            // c(group %*% 1:ncol(group)) for an indicator matrix.
                            x_term
                                .row(i)
                                .iter()
                                .enumerate()
                                .map(|(j, v)| v * (j + 1) as f64)
                                .sum::<f64>()
                                .round() as i64
                        })
                        .collect(),
                };
                events_by_group(&groups, &data.status)
            } else {
                Vec::new()
            };
            let control = control_of(
                penalty,
                if penalty.is_sparse() {
                    nfrail
                } else {
                    columns.len()
                },
                n,
                eps2,
            )?;
            let state = control.initial();
            terms.push(TermState {
                index,
                term: penalty,
                columns,
                pparm: penalty.pparm(&x_term),
                control,
                state,
                events_by_group: events,
            });
        }
        let need_df = terms.iter().any(|term| term.control.needs_df());
        let second2 = if shape.full_imat { nvar * nvar } else { nvar };
        let mut composer = Composer {
            sparse: terms.iter().position(|term| term.term.is_sparse()),
            terms,
            full_imat: shape.full_imat,
            coxlist1: CoxPenaltyTerms::zeros(nfrail, nfrail, 1),
            coxlist2: CoxPenaltyTerms::zeros(nvar, second2, nvar),
        };

        let docenter: Vec<bool> = (0..nvar)
            .map(|col| {
                !options
                    .nocenter
                    .as_ref()
                    .is_some_and(|values| xx.column(col).iter().all(|value| values.contains(value)))
            })
            .collect();
        let mut kernel = Kernel::new(KernelData {
            stop: &data.time,
            start: data.entry.as_deref(),
            status: &data.status,
            x: &xx,
            weights: &weights,
            offset: &offset,
            strata: data.strata.as_deref(),
            frail: frailx.as_deref(),
            nfrail,
            efron: options.method == TieMethod::Efron,
            docenter: &docenter,
        });
        let means = kernel.means().to_vec();

        // coxfit5_a / agfit5a: the log likelihood at the initial values
        // without the frailty, plus the dense penalty at the first thetas.
        let mut loglik0 = kernel.initial_loglik(&init);
        if shape.dense {
            let mut beta = init.clone();
            loglik0 += composer.call(2, &mut beta)?.penalty;
        }

        // The outer loop over theta.
        let mut iter2 = 0;
        let mut iter = 0;
        let mut inner_failures = Vec::new();
        let mut theta_save: Vec<Vec<f64>> = Vec::new();
        let mut coef_save: Vec<Vec<f64>> = Vec::new();
        let mut fcoef_save: Vec<Vec<f64>> = Vec::new();
        let mut init = init;
        let mut finit = finit;
        let mut penalty0 = 0.0;
        let mut penalty = 0.0;
        let mut coxfit: Option<InnerFit> = None;
        let mut beta = Vec::new();
        let mut fbeta = Vec::new();
        let mut fdiag = Vec::new();
        let mut dftemp: Option<TermDf> = None;
        for outer in 1..=options.outer_max {
            let thetas: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            beta = init.clone();
            fbeta = finit.clone();
            let inner = kernel.newton(
                options.iter_max,
                &mut beta,
                &mut fbeta,
                options.eps,
                options.toler_chol,
                shape,
                &mut composer,
            )?;
            iter = outer;
            iter2 += inner.iter;
            if inner.iter >= options.iter_max {
                inner_failures.push(outer);
            }
            theta_save.push(thetas);
            coef_save.push(beta.clone());
            fcoef_save.push(fbeta.clone());
            // An infinite penalty made the C code set fdiag = 1 in
            // self-defence; those coefficients are zero and so is their
            // variance.
            fdiag = inner.fdiag.clone();
            if nfrail > 0 && composer.coxlist1.flag[0] {
                fdiag[..nfrail].fill(0.0);
            }
            if shape.dense {
                for i in 0..nvar {
                    if composer.coxlist2.flag[i] {
                        fdiag[nfrail + i] = 0.0;
                    }
                }
            }
            if need_df {
                dftemp = Some(coxpenal_df(DfInput {
                    hmat: &inner.hmat,
                    hinv: &inner.hinv,
                    fdiag: &fdiag,
                    assign: &assign2,
                    shape,
                    pen1: &composer.coxlist1.second,
                    pen2: &composer.coxlist2.second,
                    sparse_term,
                })?);
            }
            penalty = 0.0;
            if nfrail > 0 {
                penalty -= composer.coxlist1.penalty;
            }
            if shape.dense {
                penalty -= composer.coxlist2.penalty;
            }
            // The C code returns PL - penalty.
            let loglik1 = inner.loglik + penalty;
            if outer == 1 {
                penalty0 = penalty;
            }

            // The control functions.
            let mut done = true;
            for term in composer.terms.iter_mut() {
                let coef_term: Vec<f64> = if term.term.is_sparse() {
                    fbeta.clone()
                } else {
                    term.columns.iter().map(|&c| beta[c]).collect()
                };
                let (df, trh) = match &dftemp {
                    Some(df) => (df.df[term.index], df.trh[term.index]),
                    None => (f64::NAN, f64::NAN),
                };
                let input = ControlInput {
                    iter: outer,
                    plik: loglik1,
                    loglik: inner.loglik,
                    neff: n_eff,
                    df,
                    trh,
                    events_by_group: &term.events_by_group,
                    coef: &coef_term,
                };
                term.state = term.control.update(&term.state, input)?;
                done &= term.state.done;
            }
            coxfit = Some(inner);
            if done {
                break;
            }

            // Starting values for the next iteration: the solution of the
            // closest earlier theta (the first of equally close ones).
            let next: Vec<f64> = composer.terms.iter().map(|term| term.state.theta).collect();
            let mut which = 0;
            let mut best = f64::INFINITY;
            for (k, saved) in theta_save.iter().enumerate() {
                let distance: f64 = saved.iter().zip(&next).map(|(a, b)| (a - b).powi(2)).sum();
                if distance < best {
                    best = distance;
                    which = k;
                }
            }
            init = coef_save[which].clone();
            finit = fcoef_save[which].clone();
        }
        let coxfit = coxfit.expect("outer.max >= 1");

        // Linear predictors (with the frailty) and the martingale residuals.
        let center: f64 = means.iter().zip(&beta).map(|(m, b)| m * b).sum();
        let lp_no_frailty: Vec<f64> = (0..n)
            .map(|i| offset[i] + dot_row(&xx, i, &beta) - center)
            .collect();
        let lp: Vec<f64> = match &frailx {
            Some(frailx) => lp_no_frailty
                .iter()
                .zip(frailx)
                .map(|(lp, &g)| lp + fbeta[g])
                .collect(),
            None => lp_no_frailty.clone(),
        };

        let dftemp = match dftemp {
            Some(df) => df,
            None => coxpenal_df(DfInput {
                hmat: &coxfit.hmat,
                hinv: &coxfit.hinv,
                fdiag: &fdiag,
                assign: &assign2,
                shape,
                pen1: &composer.coxlist1.second,
                pen2: &composer.coxlist2.second,
                sparse_term,
            })?,
        };
        let coefficients: Vec<f64> = (0..nvar)
            .map(|i| {
                if fdiag[nfrail + i] == 0.0 {
                    f64::NAN
                } else {
                    beta[i]
                }
            })
            .collect();
        let coef_or_zero: Vec<f64> = coefficients
            .iter()
            .map(|b| if b.is_nan() { 0.0 } else { *b })
            .collect();

        // The dense part as a Cox model: the engine of coxph.fit at the
        // penalised coefficients, then the penalised summaries in place.
        let mut coxph = CoxPHFit::fit(
            CoxphData::try_new(
                data.time.clone(),
                data.entry.clone(),
                data.status.clone(),
                xx,
                data.weights.clone(),
                data.strata.clone(),
                data.offset.clone(),
            )?,
            CoxphOptions {
                method: options.method,
                init: Some(coef_or_zero.clone()),
                iter_max: 0,
                eps: options.eps,
                toler_chol: options.toler_chol,
                nocenter: options.nocenter.clone(),
                cluster: None,
                robust: Some(false),
            },
        )?;
        coxph.coefficients = coefficients;
        coxph.var = dftemp.var.clone();
        coxph.loglik = [loglik0, coxfit.loglik + penalty];
        coxph.score = f64::NAN;
        coxph.iter = iter2;
        coxph.flag = coxfit.flag;
        coxph.means = means;
        coxph.first = coxfit.u[nfrail..].to_vec();
        coxph.linear_predictors = lp;
        coxph.residuals = if data.entry.is_some() {
            martingale_residuals(&coxph, &coxph.linear_predictors)
        } else {
            let expected = kernel.expected_events();
            data.status
                .iter()
                .zip(&expected)
                .map(|(&s, e)| f64::from(s) - e)
                .collect()
        };
        let shift: Vec<f64> = coef_or_zero
            .iter()
            .enumerate()
            .map(|(i, b)| b - options.init.as_ref().map_or(0.0, |init| init[i]))
            .collect();
        coxph.wald_test = wald_statistic(&coxph.var, &shift, options.toler_chol)?;
        let curve_fit = frailx.is_some().then(|| {
            let mut curve = coxph.clone();
            curve.linear_predictors = lp_no_frailty;
            curve
        });

        let history = composer
            .terms
            .iter()
            .map(|term| PenaltyHistory {
                term: term.index,
                theta: term.state.theta,
                done: term.state.done,
                history: term.state.history.clone(),
                columns: term
                    .control
                    .history_columns()
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
                c_loglik: term.state.c_loglik,
                half: term.state.half,
            })
            .collect();
        Ok(Self {
            coxph,
            curve_fit,
            var2: dftemp.var2,
            iter: [iter, iter2],
            inner_failures,
            df: dftemp.df,
            penalty: [penalty0, penalty],
            pterms,
            assign2,
            history,
            frail: frailx.is_some().then_some(fbeta),
            fvar: dftemp.fvar,
            frail_index: frailx,
            coxlist1: sparse_term.map(|_| composer.coxlist1),
            coxlist2: shape.dense.then_some(composer.coxlist2),
        })
    }

    /// The fit `survfit.coxph` works from: for a sparse frailty model the
    /// risk scores omit the frailty.
    fn curve_source(&self) -> &CoxPHFit {
        self.curve_fit.as_ref().unwrap_or(&self.coxph)
    }

    /// `basehaz(fit, centered)`.
    pub fn basehaz(&self, centered: bool) -> SurvivalResult<Basehaz> {
        self.curve_source().basehaz(centered)
    }

    /// `survfit(fit, newdata, ...)`; new data cannot be used with a frailty
    /// term, as in R.
    pub fn survfit(
        &self,
        newdata: Option<&CoxNewData>,
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        if newdata.is_some() && self.frail.is_some() {
            return Err(SurvivalError::invalid_input(
                "Newdata cannot be used when a model has frailty terms",
            ));
        }
        self.curve_source().survfit(newdata, options)
    }
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.outer_iter().map(|row| row.to_vec()).collect()
}

fn matrix_from_rows(rows: &[Vec<f64>], name: &str) -> SurvivalResult<Array2<f64>> {
    let ncols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must be rectangular"
        )));
    }
    Array2::from_shape_vec(
        (rows.len(), ncols),
        rows.iter().flatten().copied().collect(),
    )
    .map_err(|err| SurvivalError::invalid_input(err.to_string()))
}

/// A penalty term for [`coxpenal_fit`]: `ridge()`, `pspline()`, `frailty()`
/// or a Python callback.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct CoxPenalty {
    pub term: PenaltyTerm,
}

#[pymethods]
impl CoxPenalty {
    /// `ridge(..., theta, df, eps, scale)`.
    #[staticmethod]
    #[pyo3(signature = (theta=None, df=None, eps=0.1, scale=true))]
    fn ridge(theta: Option<f64>, df: Option<f64>, eps: f64, scale: bool) -> PyResult<Self> {
        Ok(Self {
            term: PenaltyTerm::ridge(theta, df, eps, scale)?,
        })
    }

    /// `pspline(x, df, theta, nterm, eps, method, intercept)` without the
    /// basis: build the columns with `pspline_basis(x, nterm, degree,
    /// boundary_knots)` (dropping the first column unless `intercept`).
    #[staticmethod]
    #[pyo3(signature = (df=4.0, theta=None, nterm=None, eps=None, method=None, intercept=false))]
    fn pspline(
        df: f64,
        theta: Option<f64>,
        nterm: Option<f64>,
        eps: Option<f64>,
        method: Option<&str>,
        intercept: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            term: PenaltyTerm::pspline(df, theta, nterm, eps, method, intercept)?,
        })
    }

    /// `frailty(x, distribution, sparse, theta, df, eps, method, tdf, caic,
    /// init)`: a sparse term is one column of group codes, a dense one the
    /// indicator matrix of the groups.
    #[staticmethod]
    #[pyo3(signature = (distribution="gamma", sparse=true, theta=None, df=None, eps=None, method=None, tdf=5.0, caic=false, init=None))]
    #[allow(clippy::too_many_arguments)]
    fn frailty(
        distribution: &str,
        sparse: bool,
        theta: Option<f64>,
        df: Option<f64>,
        eps: Option<f64>,
        method: Option<&str>,
        tdf: f64,
        caic: bool,
        init: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let distribution = FrailtyFamily::parse(distribution, tdf)?;
        Ok(Self {
            term: PenaltyTerm::frailty(distribution, sparse, theta, df, eps, method, caic, init)?,
        })
    }

    /// A user-defined penalty: `fexpr(coef, which=which)` returns the
    /// `coxlist` of `cox_callback` for the term's coefficients.
    #[cfg(feature = "python")]
    #[staticmethod]
    #[pyo3(signature = (fexpr, diag=true, sparse=false))]
    fn callback(fexpr: Py<PyAny>, diag: bool, sparse: bool) -> Self {
        Self {
            term: PenaltyTerm::Callback(CallbackPenalty {
                fexpr: std::sync::Arc::new(fexpr),
                diag,
                sparse,
            }),
        }
    }

    /// `"ridge"`, `"pspline"`, `"frailty"` or `"callback"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.term.r_name()
    }

    #[getter]
    fn sparse(&self) -> bool {
        self.term.is_sparse()
    }

    #[getter]
    fn diag(&self) -> bool {
        self.term.is_diagonal()
    }

    /// The `nterm` of a `pspline` term (its basis size is `nterm + degree`).
    #[getter]
    fn nterm(&self) -> Option<usize> {
        match &self.term {
            PenaltyTerm::Pspline(spline) => Some(spline.nterm),
            _ => None,
        }
    }
}

#[pymethods]
impl CoxpenalFit {
    /// The dense part of the fit as a Cox model: `predict()`, the residual
    /// types and `survfit()` with new data come from here.
    #[getter(coxph)]
    fn coxph_getter(&self) -> CoxPHFit {
        self.coxph.clone()
    }

    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.coxph.coefficients.clone()
    }

    #[getter]
    fn var(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.coxph.var)
    }

    #[getter(var2)]
    fn var2_getter(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var2)
    }

    #[getter]
    fn loglik(&self) -> [f64; 2] {
        self.coxph.loglik
    }

    #[getter]
    fn linear_predictors(&self) -> Vec<f64> {
        self.coxph.linear_predictors.clone()
    }

    #[getter]
    fn residuals(&self) -> Vec<f64> {
        self.coxph.residuals.clone()
    }

    #[getter]
    fn means(&self) -> Vec<f64> {
        self.coxph.means.clone()
    }

    #[getter]
    fn method(&self) -> TieMethod {
        self.coxph.method
    }

    #[getter]
    fn n(&self) -> usize {
        self.coxph.n
    }

    #[getter]
    fn nevent(&self) -> usize {
        self.coxph.nevent
    }

    #[getter]
    fn wald_test(&self) -> f64 {
        self.coxph.wald_test
    }

    /// Rank flag of the last inner fit (1000: it did not converge).
    #[getter]
    fn flag(&self) -> i32 {
        self.coxph.flag
    }

    /// `basehaz(fit, centered)`.
    #[pyo3(name = "basehaz", signature = (centered = true))]
    fn basehaz_py(&self, centered: bool) -> PyResult<Basehaz> {
        Ok(self.basehaz(centered)?)
    }

    /// `survfit(fit, newdata, stype, ctype, se.fit, censor)`.
    #[pyo3(name = "survfit", signature = (newdata = None, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_py(
        &self,
        newdata: Option<Vec<Vec<f64>>>,
        new_strata: Option<Vec<i32>>,
        new_offset: Option<Vec<f64>>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = match newdata {
            Some(x) => {
                let x = if x.is_empty() {
                    Array2::zeros((0, self.coxph.nvar()))
                } else {
                    matrix_from_rows(&x, "newdata")?
                };
                Some(CoxNewData::try_new(x, new_strata, new_offset, None, None)?)
            }
            None => {
                if new_strata.is_some() || new_offset.is_some() {
                    return Err(SurvivalError::invalid_input(
                        "new_strata and new_offset require newdata",
                    )
                    .into());
                }
                None
            }
        };
        Ok(self.survfit(
            newdata.as_ref(),
            SurvfitOptions {
                stype,
                ctype,
                se_fit,
                censor,
            },
        )?)
    }
}

/// `coxph()` with penalised terms on explicit data: `penalties[i]` applies
/// to the columns `pcols[i]` of `x`; `assign` lists the columns of every
/// model term (each `pcols` entry must be one of them) and defaults to the
/// penalised groups plus one term per remaining column, in column order.
/// A sparse frailty term is a single column of group codes.
#[pyfunction]
#[pyo3(signature = (time, status, x, penalties, pcols, assign=None, entry=None, strata=None, weights=None, offset=None, method="efron", init=None, iter_max=None, outer_max=None, eps=None, toler_chol=None, nocenter=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxpenal_fit(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<Vec<f64>>,
    penalties: Vec<CoxPenalty>,
    pcols: Vec<Vec<usize>>,
    assign: Option<Vec<Vec<usize>>>,
    entry: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    method: &str,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    outer_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
) -> PyResult<CoxpenalFit> {
    if x.len() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
            "x has {} rows but time has {}",
            x.len(),
            time.len()
        ))
        .into());
    }
    let x = matrix_from_rows(&x, "x")?;
    if penalties.len() != pcols.len() {
        return Err(SurvivalError::invalid_input("Invalid pcols or pattr arg").into());
    }
    let assign = assign.unwrap_or_else(|| {
        let mut terms: Vec<Vec<usize>> = pcols.clone();
        for column in 0..x.ncols() {
            if !pcols.iter().any(|group| group.contains(&column)) {
                terms.push(vec![column]);
            }
        }
        terms.sort_by_key(|columns| columns.first().copied());
        terms
    });
    let terms = assign
        .into_iter()
        .map(|columns| {
            let penalty = pcols
                .iter()
                .position(|group| *group == columns)
                .map(|k| penalties[k].term.clone());
            ModelTerm { columns, penalty }
        })
        .collect::<Vec<_>>();
    if terms.iter().filter(|term| term.penalty.is_some()).count() != penalties.len() {
        return Err(SurvivalError::invalid_input("pcols and assign arguments disagree").into());
    }
    let data = CoxpenalData::try_new(time, entry, status, x, weights, strata, offset, terms)?;
    let defaults = CoxpenalOptions::default();
    let options = CoxpenalOptions {
        method: TieMethod::parse(Some(method))?,
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        outer_max: outer_max.unwrap_or(defaults.outer_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter: nocenter.or(defaults.nocenter),
    };
    Ok(CoxpenalFit::fit(data, options)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kidney_like() -> (Vec<f64>, Vec<i32>, Array2<f64>) {
        // Four groups of three, one covariate.
        let time = vec![
            8.0, 16.0, 23.0, 13.0, 22.0, 28.0, 30.0, 12.0, 24.0, 15.0, 7.0, 9.0,
        ];
        let status = vec![1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0];
        let rows: Vec<f64> = (0..12)
            .flat_map(|i| [(i % 5) as f64 * 0.7 - 1.0, (i / 3 + 1) as f64])
            .collect();
        (time, status, Array2::from_shape_vec((12, 2), rows).unwrap())
    }

    fn fit_terms(terms: Vec<ModelTerm>, options: CoxpenalOptions) -> CoxpenalFit {
        let (time, status, x) = kidney_like();
        let data = CoxpenalData::try_new(time, None, status, x, None, None, None, terms).unwrap();
        CoxpenalFit::fit(data, options).unwrap()
    }

    #[test]
    fn ridge_with_a_tiny_theta_matches_the_unpenalised_fit() {
        let (time, status, x) = kidney_like();
        let plain = CoxPHFit::fit(
            CoxphData::try_new(
                time.clone(),
                None,
                status.clone(),
                x.slice(ndarray::s![.., ..1]).to_owned(),
                None,
                None,
                None,
            )
            .unwrap(),
            CoxphOptions::default(),
        )
        .unwrap();
        let data = CoxpenalData::try_new(
            time,
            None,
            status,
            x.slice(ndarray::s![.., ..1]).to_owned(),
            None,
            None,
            None,
            vec![ModelTerm {
                columns: vec![0],
                penalty: Some(PenaltyTerm::ridge(Some(1e-10), None, 0.1, false).unwrap()),
            }],
        )
        .unwrap();
        let fit = CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap();
        assert!((fit.coxph.coefficients[0] - plain.coefficients[0]).abs() < 1e-6);
        assert!((fit.coxph.loglik[1] - plain.loglik[1]).abs() < 1e-8);
        assert!((fit.coxph.loglik[0] - plain.loglik[0]).abs() < 1e-12);
        assert_eq!(fit.iter[0], 1);
        assert_eq!(fit.pterms, vec![1]);
        assert!((fit.df[0] - 1.0).abs() < 1e-6);
        assert!(fit.frail.is_none());
        let total: f64 = fit.coxph.residuals.iter().sum();
        assert!(total.abs() < 1e-10);
        for (a, b) in fit.coxph.residuals.iter().zip(&plain.residuals) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(fit.history.len(), 1);
        assert!(fit.history[0].done);
        let curves = fit.survfit(None, SurvfitOptions::default()).unwrap();
        assert_eq!(curves.len(), 1);
    }

    #[test]
    fn sparse_gamma_frailty_fits_and_reports_the_frailty_pieces() {
        let fit = fit_terms(
            vec![
                ModelTerm {
                    columns: vec![0],
                    penalty: None,
                },
                ModelTerm {
                    columns: vec![1],
                    penalty: Some(
                        PenaltyTerm::frailty(
                            FrailtyFamily::Gamma,
                            true,
                            Some(0.5),
                            None,
                            None,
                            None,
                            false,
                            None,
                        )
                        .unwrap(),
                    ),
                },
            ],
            CoxpenalOptions::default(),
        );
        assert_eq!(fit.pterms, vec![0, 2]);
        assert_eq!(fit.coxph.nvar(), 1);
        let frail = fit.frail.as_ref().unwrap();
        assert_eq!(frail.len(), 4);
        // The gamma penalty recentres so that mean(exp(frail)) is one.
        let mean_exp: f64 = frail.iter().map(|f| f.exp()).sum::<f64>() / 4.0;
        assert!((mean_exp - 1.0).abs() < 1e-8);
        assert_eq!(fit.fvar.as_ref().unwrap().len(), 4);
        assert_eq!(fit.df.len(), 2);
        assert!(fit.df[1] > 0.0 && fit.df[1] < 4.0);
        assert_eq!(fit.assign2, vec![vec![0], vec![1]]);
        assert!(fit.history[0].c_loglik.is_some());
        assert!(fit.history[0].done);
        // The linear predictor carries the frailty; survfit does not.
        let lp = &fit.coxph.linear_predictors;
        let index = fit.frail_index.as_ref().unwrap();
        for i in 0..12 {
            let without = lp[i] - frail[index[i]];
            assert!((without - fit.curve_fit.as_ref().unwrap().linear_predictors[i]).abs() < 1e-12);
        }
        let total: f64 = fit.coxph.residuals.iter().sum();
        assert!(total.abs() < 1e-10);
        assert!(fit.coxlist1.is_some() && fit.coxlist2.is_none());
        assert!(fit.survfit(None, SurvfitOptions::default()).is_ok());
    }

    /// A ridge penalty with a negligible theta reproduces `coxph` on the
    /// same data; the residuals exercise the stratum reset of `coxfit5_c`
    /// (right-censored) and `agmart3` ((start, stop]), the weighted Efron
    /// fit the `efron_wt` of `agfit5b`.
    #[test]
    fn negligible_penalty_matches_coxph_with_strata_weights_and_entry() {
        let (time, status, x) = kidney_like();
        let x = x.slice(ndarray::s![.., ..1]).to_owned();
        let entry: Vec<f64> = time.iter().map(|t| t * 0.25).collect();
        let strata = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let weights = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        for (entry, strata, weights) in [
            (None, Some(strata.clone()), None),
            (Some(entry.clone()), None, Some(weights.clone())),
            (Some(entry), Some(strata), Some(weights)),
        ] {
            let plain = CoxPHFit::fit(
                CoxphData::try_new(
                    time.clone(),
                    entry.clone(),
                    status.clone(),
                    x.clone(),
                    weights.clone(),
                    strata.clone(),
                    None,
                )
                .unwrap(),
                CoxphOptions::default(),
            )
            .unwrap();
            let data = CoxpenalData::try_new(
                time.clone(),
                entry,
                status.clone(),
                x.clone(),
                weights,
                strata,
                None,
                vec![ModelTerm {
                    columns: vec![0],
                    penalty: Some(PenaltyTerm::ridge(Some(1e-12), None, 0.1, false).unwrap()),
                }],
            )
            .unwrap();
            let fit = CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap();
            assert!((fit.coxph.coefficients[0] - plain.coefficients[0]).abs() < 1e-6);
            assert!((fit.coxph.loglik[1] - plain.loglik[1]).abs() < 1e-8);
            for (a, b) in fit.coxph.residuals.iter().zip(&plain.residuals) {
                assert!((a - b).abs() < 1e-6, "{a} != {b}");
            }
        }
    }

    #[test]
    fn a_frailty_alone_is_a_null_model_with_random_effects() {
        let (time, status, x) = kidney_like();
        let data = CoxpenalData::try_new(
            time,
            None,
            status,
            x.slice(ndarray::s![.., 1..]).to_owned(),
            None,
            None,
            None,
            vec![ModelTerm {
                columns: vec![0],
                penalty: Some(
                    PenaltyTerm::frailty(
                        FrailtyFamily::Gaussian,
                        true,
                        Some(0.4),
                        None,
                        None,
                        None,
                        false,
                        None,
                    )
                    .unwrap(),
                ),
            }],
        )
        .unwrap();
        let fit = CoxpenalFit::fit(data, CoxpenalOptions::default()).unwrap();
        assert!(fit.coxph.coefficients.is_empty());
        assert_eq!(fit.coxph.var.shape(), &[0, 0]);
        assert_eq!(fit.pterms, vec![2]);
        assert_eq!(fit.df.len(), 1);
        let frail = fit.frail.as_ref().unwrap();
        assert_eq!(frail.len(), 4);
        // The Gaussian penalty recentres the frailties at zero.
        assert!(frail.iter().sum::<f64>().abs() < 1e-8);
        assert_eq!(fit.coxph.wald_test, 0.0);
        assert!(fit.history[0].done);
        assert!(fit.history[0].c_loglik.is_none());
    }

    #[test]
    fn invalid_terms_are_rejected() {
        let (time, status, x) = kidney_like();
        let ridge = PenaltyTerm::ridge(Some(1.0), None, 0.1, true).unwrap();
        let frailty = |sparse: bool| {
            PenaltyTerm::frailty(
                FrailtyFamily::Gamma,
                sparse,
                Some(1.0),
                None,
                None,
                None,
                false,
                None,
            )
            .unwrap()
        };
        let make = |terms: Vec<ModelTerm>| {
            CoxpenalData::try_new(
                time.clone(),
                None,
                status.clone(),
                x.clone(),
                None,
                None,
                None,
                terms,
            )
        };
        // No penalty at all.
        assert!(
            make(vec![ModelTerm {
                columns: vec![0, 1],
                penalty: None
            }])
            .is_err()
        );
        // A column in two terms.
        assert!(
            make(vec![
                ModelTerm {
                    columns: vec![0, 1],
                    penalty: Some(ridge.clone())
                },
                ModelTerm {
                    columns: vec![1],
                    penalty: None
                }
            ])
            .is_err()
        );
        // Two sparse terms, and a multi-column sparse term.
        assert!(
            make(vec![
                ModelTerm {
                    columns: vec![0],
                    penalty: Some(frailty(true))
                },
                ModelTerm {
                    columns: vec![1],
                    penalty: Some(frailty(true))
                }
            ])
            .is_err()
        );
        assert!(
            make(vec![ModelTerm {
                columns: vec![0, 1],
                penalty: Some(frailty(true))
            }])
            .is_err()
        );
        let data = make(vec![ModelTerm {
            columns: vec![0, 1],
            penalty: Some(ridge),
        }])
        .unwrap();
        let exact = CoxpenalOptions {
            method: TieMethod::Exact,
            ..CoxpenalOptions::default()
        };
        assert!(CoxpenalFit::fit(data.clone(), exact).is_err());
        let bad_init = CoxpenalOptions {
            init: Some(vec![0.0]),
            ..CoxpenalOptions::default()
        };
        assert!(CoxpenalFit::fit(data, bad_init).is_err());
    }
}
