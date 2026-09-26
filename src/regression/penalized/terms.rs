//! The outer-loop pieces `coxpenal.fit` (`R/coxpenal.fit.R`) and
//! `survpenal.fit` (`R/survpenal.fit.R`) share, in the order both functions
//! use them: the model terms and their checks, `pterms`/`ptype`, the removal
//! of a sparse frailty column from the design, the penalised terms' `pparm`,
//! `cfun` and first `theta`, the penalty callbacks `f.expr1`/`f.expr2` with
//! their persistent `coxlist1`/`coxlist2`, the `cfun` calls after each inner
//! fit, the restart from the closest earlier `theta` and the `history`.

use super::control::{Control, ControlInput, ControlState};
use super::df::TermDf;
use super::penalty::{
    CoxPenaltyTerms, FrailtyFamily, FrailtyMethod, PenaltyTerm, Pparm, PsplineMethod,
};
use crate::error::{SurvivalError, SurvivalResult};
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One model term: its design columns (R's `assign` entry) and, for a
/// penalised term, its penalty (`pcols`/`pattr`).
#[derive(Debug, Clone)]
pub struct ModelTerm {
    pub columns: Vec<usize>,
    pub penalty: Option<PenaltyTerm>,
}

/// The model terms of a design with `ncol` columns from R's `pcols`
/// (the columns of each penalised term), `pattr` (their penalties) and
/// `assign` (the columns of every term; by default the penalised groups
/// plus one term per remaining column, in column order).
pub(crate) fn model_terms(
    ncol: usize,
    penalties: Vec<PenaltyTerm>,
    pcols: Vec<Vec<usize>>,
    assign: Option<Vec<Vec<usize>>>,
) -> SurvivalResult<Vec<ModelTerm>> {
    if penalties.len() != pcols.len() {
        return Err(SurvivalError::invalid_input("Invalid pcols or pattr arg"));
    }
    let assign = assign.unwrap_or_else(|| {
        let mut terms: Vec<Vec<usize>> = pcols.clone();
        for column in 0..ncol {
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
                .map(|k| penalties[k].clone());
            ModelTerm { columns, penalty }
        })
        .collect::<Vec<_>>();
    if terms.iter().filter(|term| term.penalty.is_some()).count() != penalties.len() {
        return Err(SurvivalError::invalid_input(
            "pcols and assign arguments disagree",
        ));
    }
    Ok(terms)
}

/// The checks of the model terms against a design with `ncol` columns:
/// every column belongs to exactly one term, at least one term is
/// penalised, and there is at most one sparse term, of a single column.
pub(crate) fn validate_terms(ncol: usize, terms: &[ModelTerm]) -> SurvivalResult<()> {
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
    Ok(())
}

/// The search history of one penalised term (`fit$history[[i]]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", from_py_object)]
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

crate::internal::pickle::picklable!(PenaltyHistory);

/// R's `cox_callback`: evaluates the penalty functions at `coef` (which the
/// callback may recentre in place) and returns the C-level `coxlist` —
/// `first` and `penalty` already negated, `second` diagonal or full.
/// `which` is 1 for the sparse frailty term and 2 for the dense terms.
pub(crate) trait PenaltyCallback {
    fn call(&mut self, which: i32, coef: &mut [f64]) -> SurvivalResult<&CoxPenaltyTerms>;
}

/// The terms present, R's `ptype`: 1 or 3 with a sparse term, 2 or 3 with
/// dense penalised terms.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PenaltyShape {
    pub sparse: bool,
    pub dense: bool,
    /// The dense second derivative is a full matrix (R's `full.imat`,
    /// `pdiag`), else a diagonal.
    pub full_imat: bool,
}

/// `pterms` (per model term: 0 ordinary, 1 penalised, 2 sparse), the
/// position of the sparse term and the [`PenaltyShape`].
pub(crate) fn penalty_shape(terms: &[ModelTerm]) -> (Vec<u8>, Option<usize>, PenaltyShape) {
    let pterms: Vec<u8> = terms
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
        full_imat: !terms
            .iter()
            .filter_map(|term| term.penalty.as_ref())
            .filter(|penalty| !penalty.is_sparse())
            .all(PenaltyTerm::is_diagonal),
    };
    (pterms, sparse_term, shape)
}

/// Removes the sparse term's column from the design `x`: the dense design,
/// R's `assign2` (the columns of each term in it; the sparse term keeps its
/// original column), the 0-based group of every row
/// (`match(x, sort(unique(x)))`) and the number of groups.
#[allow(clippy::type_complexity)]
pub(crate) fn drop_sparse_column(
    x: ArrayView2<'_, f64>,
    terms: &[ModelTerm],
) -> (Array2<f64>, Vec<Vec<usize>>, Option<Vec<usize>>, usize) {
    let (n, ncol) = x.dim();
    let fcol = terms
        .iter()
        .find(|term| term.penalty.as_ref().is_some_and(PenaltyTerm::is_sparse))
        .map(|term| term.columns[0]);
    match fcol {
        Some(fcol) => {
            let keep: Vec<usize> = (0..ncol).filter(|&c| c != fcol).collect();
            let xx = Array2::from_shape_fn((n, ncol - 1), |(i, j)| x[(i, keep[j])]);
            let assign2: Vec<Vec<usize>> = terms
                .iter()
                .map(|term| {
                    term.columns
                        .iter()
                        .map(|&c| if c > fcol { c - 1 } else { c })
                        .collect()
                })
                .collect();
            let mut levels: Vec<f64> = x.column(fcol).to_vec();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            let frailx: Vec<usize> = x
                .column(fcol)
                .iter()
                .map(|v| levels.binary_search_by(|l| l.total_cmp(v)).expect("level"))
                .collect();
            (xx, assign2, Some(frailx), levels.len())
        }
        None => (
            x.to_owned(),
            terms.iter().map(|term| term.columns.clone()).collect(),
            None,
            0,
        ),
    }
}

/// A penalised term with its fit-time state: `pparm`, `cfun`, the current
/// `theta` and the search state.
pub(crate) struct TermState<'a> {
    /// Position in the data's `terms`.
    pub index: usize,
    pub term: &'a PenaltyTerm,
    /// Columns in the dense design (empty for the sparse term).
    pub columns: Vec<usize>,
    pub pparm: Pparm,
    pub control: Control,
    pub state: ControlState,
    /// Events per group (frailty terms).
    pub events_by_group: Vec<f64>,
}

/// The penalised terms of `terms` with their `pparm`, `cfun` and first
/// `theta` (the `cfun(parms, iter = 0)` calls).  `xx` is the dense design,
/// `assign2` its columns per term, `frailx`/`nfrail` the sparse groups,
/// `status` the event codes the frailty `cfun`s sum per group and `eps2`
/// the fallback tolerance `sqrt(eps)` R appends to every `cparm`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_term_states<'a>(
    terms: &'a [ModelTerm],
    assign2: &[Vec<usize>],
    xx: &Array2<f64>,
    frailx: Option<&[usize]>,
    nfrail: usize,
    n: usize,
    status: &[i32],
    eps2: f64,
) -> SurvivalResult<Vec<TermState<'a>>> {
    let mut states = Vec::new();
    for (index, term) in terms.iter().enumerate() {
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
            let groups: Vec<i64> = match frailx {
                Some(frailx) if penalty.is_sparse() => frailx.iter().map(|&g| g as i64).collect(),
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
            events_by_group(&groups, status)
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
        states.push(TermState {
            index,
            term: penalty,
            columns,
            pparm: penalty.pparm(&x_term)?,
            control,
            state,
            events_by_group: events,
        });
    }
    Ok(states)
}

/// The penalty callbacks `f.expr1`/`f.expr2` with their persistent
/// `coxlist1`/`coxlist2`.
pub(crate) struct Composer<'a> {
    pub terms: Vec<TermState<'a>>,
    pub sparse: Option<usize>,
    pub full_imat: bool,
    /// survpenal.fit's `f.expr1` zeroes `first` and `second` of a flagged
    /// sparse term; coxpenal.fit's keeps the previous values.
    pub zero_flagged_sparse: bool,
    pub coxlist1: CoxPenaltyTerms,
    pub coxlist2: CoxPenaltyTerms,
}

impl<'a> Composer<'a> {
    /// R's initial `coxlist1` (`nfrail` groups) and `coxlist2` (`nvar`
    /// dense coefficients, a full or diagonal second derivative).
    pub(crate) fn new(
        terms: Vec<TermState<'a>>,
        nfrail: usize,
        nvar: usize,
        full_imat: bool,
        zero_flagged_sparse: bool,
    ) -> Self {
        let second2 = if full_imat { nvar * nvar } else { nvar };
        Self {
            sparse: terms.iter().position(|term| term.term.is_sparse()),
            terms,
            full_imat,
            zero_flagged_sparse,
            coxlist1: CoxPenaltyTerms::zeros(nfrail, nfrail, 1),
            coxlist2: CoxPenaltyTerms::zeros(nvar, second2, nvar),
        }
    }
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
            } else if self.zero_flagged_sparse {
                list.first = vec![0.0; nfrail];
                list.second = vec![0.0; nfrail];
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
pub(crate) fn control_of(
    term: &PenaltyTerm,
    ncols: usize,
    n: usize,
    eps2: f64,
) -> SurvivalResult<Control> {
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
                guess: 3.0 * df / frailty.n.unwrap_or(n) as f64,
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
pub(crate) fn events_by_group(groups: &[i64], status: &[i32]) -> Vec<f64> {
    let mut sums = BTreeMap::new();
    for (&g, &s) in groups.iter().zip(status) {
        *sums.entry(g).or_insert(0.0) += f64::from(s);
    }
    sums.into_values().collect()
}

/// The `cfun` calls after outer iteration `outer`: each term gets `plik`
/// (the likelihood without the penalty), `loglik` (with it), `neff`, its
/// df and trace from `df` (when the `cfun`s need them) and its
/// coefficients (`fbeta` for the sparse term, its columns of `beta_dense`
/// otherwise).  Returns whether every search is done.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_controls(
    terms: &mut [TermState<'_>],
    outer: usize,
    plik: f64,
    loglik: f64,
    neff: f64,
    df: Option<&TermDf>,
    beta_dense: &[f64],
    fbeta: &[f64],
) -> SurvivalResult<bool> {
    let mut done = true;
    for term in terms.iter_mut() {
        let coef_term: Vec<f64> = if term.term.is_sparse() {
            fbeta.to_vec()
        } else {
            term.columns.iter().map(|&c| beta_dense[c]).collect()
        };
        let (df, trh) = match df {
            Some(df) => (df.df[term.index], df.trh[term.index]),
            None => (f64::NAN, f64::NAN),
        };
        let input = ControlInput {
            iter: outer,
            plik,
            loglik,
            neff,
            df,
            trh,
            events_by_group: &term.events_by_group,
            coef: &coef_term,
        };
        term.state = term.control.update(&term.state, input)?;
        done &= term.state.done;
    }
    Ok(done)
}

/// The saved fit whose `theta` is closest to `next` (squared distance),
/// the first of equally close ones: R's
/// `min((1:iter)[howclose == min(howclose)])`.
pub(crate) fn closest_saved(theta_save: &[Vec<f64>], next: &[f64]) -> usize {
    let mut which = 0;
    let mut best = f64::INFINITY;
    for (k, saved) in theta_save.iter().enumerate() {
        let distance: f64 = saved.iter().zip(next).map(|(a, b)| (a - b).powi(2)).sum();
        if distance < best {
            best = distance;
            which = k;
        }
    }
    which
}

/// The fit's `history`: every penalised term's search.
pub(crate) fn histories(terms: &[TermState<'_>]) -> Vec<PenaltyHistory> {
    terms
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
        .collect()
}
