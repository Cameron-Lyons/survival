use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[doc = " What a `cfun` returns: R's `iterlist[[i]]`."]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", frozen, get_all, skip_from_py_object)]
pub struct PenaltyControlState {
    #[doc = " The next `theta` to fit (the final entry is the proposal made when"]
    #[doc = " the search declared itself done)."]
    pub theta: f64,
    pub done: bool,
    #[doc = " The search history, one row per outer iteration (columns as in"]
    #[doc = " [`super::PenaltyController::columns`]); the `df` searches start from the"]
    #[doc = " known `(theta, df)` pairs of the term."]
    pub history: Vec<Vec<f64>>,
    #[doc = " The corrected log likelihood of a gamma frailty."]
    pub c_loglik: Option<f64>,
    #[doc = " `frailty.controldf`'s bisection counter."]
    pub half: Option<i64>,
    #[doc = " Zero-based df history row whose label the next theta inherits in R."]
    #[doc = " Bisection proposals have no source label."]
    pub theta_history_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PenaltyControlInput<'a> {
    pub iter: usize,
    pub plik: f64,
    pub loglik: f64,
    pub neff: f64,
    pub df: f64,
    pub trh: f64,
    pub events_by_group: &'a [f64],
    pub coef: &'a [f64],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PenaltyControl {
    Fixed {
        theta: f64,
    },
    Gamma {
        theta: Option<f64>,
        eps: f64,
        init: Option<Vec<f64>>,
    },
    Df {
        df: f64,
        eps: f64,
        thetas: Vec<f64>,
        dfs: Vec<f64>,
        guess: f64,
        gamma_correction: bool,
    },
    Aic {
        eps: f64,
        init: Vec<f64>,
        lower: f64,
        upper: Option<f64>,
        caic: bool,
        gamma_correction: bool,
    },
    Gauss {
        eps: f64,
        init: Option<Vec<f64>>,
    },
}

pub(crate) type Control = PenaltyControl;
pub(crate) type ControlState = PenaltyControlState;
pub(crate) type ControlInput<'a> = PenaltyControlInput<'a>;

impl Control {
    pub(crate) fn history_columns(&self) -> &'static [&'static str] {
        match self {
            Self::Fixed { .. } => &[],
            Self::Gamma { .. } => &["theta", "loglik", "c.loglik"],
            Self::Df { .. } => &["thetas", "dfs"],
            Self::Aic { .. } => &["theta", "loglik", "df", "aic", "aicc"],
            Self::Gauss { .. } => &["theta", "resid", "fsum", "trace"],
        }
    }

    pub(crate) fn needs_df(&self) -> bool {
        matches!(
            self,
            Self::Df { .. } | Self::Aic { .. } | Self::Gauss { .. }
        )
    }

    pub(crate) fn initial(&self) -> ControlState {
        let state = |theta: f64, done: bool| ControlState {
            theta,
            done,
            history: Vec::new(),
            c_loglik: None,
            half: None,
            theta_history_index: None,
        };
        match self {
            Self::Fixed { theta } => state(*theta, true),
            Self::Gamma { theta, init, .. } => state(
                theta.or_else(|| init.as_ref().map(|i| i[0])).unwrap_or(0.0),
                false,
            ),
            Self::Df {
                thetas, dfs, guess, ..
            } => ControlState {
                theta: *guess,
                done: false,
                history: thetas.iter().zip(dfs).map(|(&t, &d)| vec![t, d]).collect(),
                c_loglik: None,
                half: None,
                theta_history_index: None,
            },
            Self::Aic { init, .. } => state(init.first().copied().unwrap_or(0.005), false),
            Self::Gauss { init, .. } => state(init.as_ref().map_or(1.0, |i| i[0]), false),
        }
    }

    pub(crate) fn update(
        &self,
        old: &ControlState,
        input: ControlInput<'_>,
    ) -> SurvivalResult<ControlState> {
        match self {
            Self::Fixed { theta } => Ok(ControlState {
                theta: *theta,
                done: true,
                history: Vec::new(),
                c_loglik: None,
                half: None,
                theta_history_index: None,
            }),
            Self::Gamma { theta, eps, init } => {
                control_gamma(theta.is_some(), *eps, init.as_deref(), old, input)
            }
            Self::Df {
                df,
                eps,
                gamma_correction,
                ..
            } => {
                let mut state = control_df(*df, *eps, old, input)?;
                if *gamma_correction {
                    state.c_loglik = Some(input.loglik + gamma_correction_of(old.theta, &input));
                }
                Ok(state)
            }
            Self::Aic {
                eps,
                init,
                lower,
                upper,
                caic,
                gamma_correction,
            } => {
                let mut state = control_aic(*eps, init, *lower, *upper, *caic, old, input)?;
                if *gamma_correction {
                    state.c_loglik = Some(input.loglik + gamma_correction_of(old.theta, &input));
                }
                Ok(state)
            }
            Self::Gauss { eps, init } => Ok(control_gauss(*eps, init.as_deref(), old, input)),
        }
    }
}

fn gamma_correction_of(theta: f64, input: &ControlInput<'_>) -> f64 {
    if theta == 0.0 {
        0.0
    } else {
        frailty_gammacon(input.events_by_group, 1.0 / theta)
    }
}

fn control_gamma(
    fixed: bool,
    eps: f64,
    init: Option<&[f64]>,
    old: &ControlState,
    input: ControlInput<'_>,
) -> SurvivalResult<ControlState> {
    let theta = old.theta;
    let correct = gamma_correction_of(theta, &input);
    let c_loglik = input.loglik + correct;
    if fixed {
        return Ok(ControlState {
            theta,
            done: true,
            history: Vec::new(),
            c_loglik: Some(c_loglik),
            half: None,
            theta_history_index: None,
        });
    }
    let iter = input.iter;
    let mut history = old.history.clone();
    history.push(vec![theta, input.loglik, c_loglik]);
    let (theta, done) = if iter == 1 {
        (init.map_or(1.0, |i| i[1]), false)
    } else if iter == 2 {
        let theta = if history[1][2] < history[0][2] + 1.0 {
            (history[0][0] + history[1][0]) / 2.0
        } else {
            2.0 * history[1][0]
        };
        (theta, false)
    } else {
        let done = (1.0 - history[iter - 1][2] / history[iter - 2][2]).abs() < eps;
        let x: Vec<f64> = history.iter().map(|row| row[0]).collect();
        let y: Vec<f64> = history.iter().map(|row| row[2]).collect();
        let ymax = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let xmax = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let theta = if y[iter - 1] == ymax && x[iter - 1] == xmax {
            2.0 * xmax
        } else {
            let sqrt_x: Vec<f64> = x.iter().map(|v| v.sqrt()).collect();
            frailty_brent(&sqrt_x, &y, Some(0.0), None)?.powi(2)
        };
        (theta, done)
    };
    Ok(ControlState {
        theta,
        done,
        history,
        c_loglik: Some(c_loglik),
        half: None,
        theta_history_index: None,
    })
}

fn control_df(
    target_df: f64,
    eps: f64,
    old: &ControlState,
    input: ControlInput<'_>,
) -> SurvivalResult<ControlState> {
    let iter = input.iter;
    let mut history = old.history.clone();
    history.push(vec![old.theta, input.df]);
    let thetas: Vec<f64> = history.iter().map(|row| row[0]).collect();
    let dfs: Vec<f64> = history.iter().map(|row| row[1]).collect();
    let nx = thetas.len();
    if nx == 2 {
        let mut theta =
            thetas[0] + (thetas[1] - thetas[0]) * (target_df - dfs[0]) / (dfs[1] - dfs[0]);
        if target_df > input.df {
            theta *= 1.5;
        }
        return Ok(ControlState {
            theta,
            done: false,
            history,
            c_loglik: None,
            half: Some(0),
            theta_history_index: Some(0),
        });
    }
    if !(dfs[nx - 1].is_finite() && dfs[nx - 2].is_finite()) {
        return Err(SurvivalError::computation(
            "degrees-of-freedom search failed: the penalized fit has a non-finite df",
        ));
    }
    let done = iter > 1 && (dfs[nx - 1] - target_df).abs() < eps;
    let doing_well = ((dfs[nx - 1] - target_df) / (dfs[nx - 2] - target_df)).abs() <= 0.6;
    let mut ord: Vec<usize> = (0..nx).collect();
    ord.sort_by(|&l, &r| thetas[l].total_cmp(&thetas[r]));
    let x: Vec<f64> = ord.iter().map(|&i| thetas[i]).collect();
    let (y, target): (Vec<f64>, f64) = if (thetas[0] - thetas[1]) * (dfs[0] - dfs[1]) > 0.0 {
        (ord.iter().map(|&i| dfs[i]).collect(), target_df)
    } else {
        (ord.iter().map(|&i| -dfs[i]).collect(), -target_df)
    };
    let mut b1 = if y.iter().all(|&v| v > target) {
        0
    } else if y.iter().all(|&v| v < target) {
        nx - 3
    } else {
        let b1 = (0..nx).rev().find(|&i| y[i] <= target).ok_or_else(|| {
            SurvivalError::computation("degrees-of-freedom search has no lower bracket")
        })?;
        if b1 + 1 == nx {
            return Err(SurvivalError::computation(
                "degrees-of-freedom search has no upper bracket",
            ));
        }
        if !doing_well && old.half.is_none_or(|half| half < 2) {
            return Ok(ControlState {
                theta: (x[b1] + x[b1 + 1]) / 2.0,
                done,
                history,
                c_loglik: None,
                half: Some(old.half.unwrap_or(0).max(0) + 1),
                theta_history_index: None,
            });
        }
        if b1 + 2 == nx || (b1 > 0 && (target - y[b1]) < (y[b1 + 1] - target)) {
            b1 - 1
        } else {
            b1
        }
    };
    if b1 + 2 >= nx {
        b1 = nx - 3;
    }
    let xx = [(x[b1 + 1] - x[b1]).ln(), (x[b1 + 2] - x[b1]).ln()];
    let yy = [(y[b1 + 1] - y[b1]).ln(), (y[b1 + 2] - y[b1]).ln()];
    let power = (yy[1] - yy[0]) / (xx[1] - xx[0]);
    let a = yy[0] - power * xx[0];
    let newx = ((target - y[b1]).ln() - a) / power;
    Ok(ControlState {
        theta: x[b1] + newx.exp(),
        done,
        history,
        c_loglik: None,
        half: Some(0),
        theta_history_index: Some(ord[b1]),
    })
}

fn control_aic(
    eps: f64,
    init: &[f64],
    lower: f64,
    upper: Option<f64>,
    caic: bool,
    old: &ControlState,
    input: ControlInput<'_>,
) -> SurvivalResult<ControlState> {
    let iter = input.iter;
    let (n, df, loglik) = (input.neff, input.df, input.plik);
    let dfc = if n < df + 2.0 {
        (df - n) + (df + 1.0) * df / 2.0 - 1.0
    } else {
        -1.0 + (df + 1.0) / (1.0 - ((df + 2.0) / n))
    };
    let mut history = old.history.clone();
    history.push(vec![old.theta, loglik, df, loglik - df, loglik - dfc]);
    if iter == 1 {
        return Ok(ControlState {
            theta: init.get(1).copied().unwrap_or(1.0),
            done: false,
            history,
            c_loglik: None,
            half: None,
            theta_history_index: None,
        });
    }
    if iter == 2 {
        let theta = history.iter().map(|row| row[0]).sum::<f64>() / history.len() as f64;
        return Ok(ControlState {
            theta,
            done: false,
            history,
            c_loglik: None,
            half: None,
            theta_history_index: None,
        });
    }
    let column = if caic { 4 } else { 3 };
    let aic: Vec<f64> = history.iter().map(|row| row[column]).collect();
    let done = (1.0 - aic[iter - 1] / aic[iter - 2]).abs() < eps;
    let x: Vec<f64> = history.iter().map(|row| row[0]).collect();
    let xmax = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let aic_max = aic.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let theta = if x[iter - 1] == aic_max && x[iter - 1] == xmax {
        2.0 * xmax
    } else {
        frailty_brent(&x, &aic, Some(lower), upper)?
    };
    Ok(ControlState {
        theta,
        done,
        history,
        c_loglik: None,
        half: None,
        theta_history_index: None,
    })
}

fn control_gauss(
    eps: f64,
    init: Option<&[f64]>,
    old: &ControlState,
    input: ControlInput<'_>,
) -> ControlState {
    let iter = input.iter;
    let nfrail = input.coef.len() as f64;
    let fsum: f64 = input.coef.iter().map(|b| b * b).sum();
    let theta = old.theta;
    let resid = fsum / (nfrail - input.trh / theta) - theta;
    let mut history = old.history.clone();
    history.push(vec![theta, resid, fsum, input.trh]);
    let state = |theta: f64, done: bool, history: Vec<Vec<f64>>| ControlState {
        theta,
        done,
        history,
        c_loglik: None,
        half: None,
        theta_history_index: None,
    };
    if iter == 1 {
        let next = match init {
            None => {
                if resid > 0.0 {
                    theta * 3.0
                } else {
                    theta / 3.0
                }
            }
            Some(init) => init[1],
        };
        return state(next, false, history);
    }
    if iter == 2 {
        let next = if history.iter().all(|row| row[1] > 0.0) {
            history[1][0] * 2.0
        } else if history.iter().all(|row| row[1] < 0.0) {
            history[1][0] / 2.0
        } else {
            (history[0][0] + history[1][0]) / 2.0
        };
        return state(next, false, history);
    }
    let done = history[iter - 1][1].abs() < eps;
    let mut ord: Vec<usize> = (0..iter).collect();
    ord.sort_by(|&l, &r| history[l][0].total_cmp(&history[r][0]));
    let tempy: Vec<f64> = ord.iter().map(|&i| history[i][1]).collect();
    let tempx: Vec<f64> = ord.iter().map(|&i| history[i][0]).collect();
    let newtheta = if tempy.iter().all(|&v| v > 0.0) {
        2.0 * tempx[iter - 1]
    } else if tempy.iter().all(|&v| v < 0.0) {
        0.5 * tempx[0]
    } else {
        let mut b1 = ord.iter().position(|&i| i == iter - 1).expect("latest row");
        if b1 == 0 {
            b1 = 1;
        } else if b1 == iter - 1 {
            b1 = iter - 2;
        }
        let r = tempy[b1] / tempy[b1 + 1];
        let s = tempy[b1] / tempy[b1 - 1];
        let u = r / s;
        let p = s
            * (u * (r - u) * (tempx[b1 + 1] - tempx[b1]) - (1.0 - r) * (tempx[b1] - tempx[b1 - 1]));
        let q = (u - 1.0) * (r - 1.0) * (s - 1.0);
        let mut newtheta = tempx[b1] + p / q;
        if newtheta > tempx[b1 + 1] {
            newtheta = (tempx[b1] + tempx[b1 + 1]) / 2.0;
        }
        if newtheta < tempx[b1 - 1] {
            newtheta = (tempx[b1] + tempx[b1 - 1]) / 2.0;
        }
        newtheta
    };
    state(newtheta, done, history)
}

pub(crate) fn frailty_brent(
    x: &[f64],
    y: &[f64],
    lower: Option<f64>,
    upper: Option<f64>,
) -> SurvivalResult<f64> {
    let n = x.len();
    if y.len() != n {
        return Err(SurvivalError::invalid_input("Length mismatch for x and y"));
    }
    if n < 3 {
        return Ok(x.iter().sum::<f64>() / n as f64);
    }
    let mut ord: Vec<usize> = (0..n).collect();
    ord.sort_by(|&l, &r| x[l].total_cmp(&x[r]));
    let xx: Vec<f64> = ord.iter().map(|&i| x[i]).collect();
    let yy: Vec<f64> = ord.iter().map(|&i| y[i]).collect();
    let ymax = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let best: Vec<usize> = (0..n).filter(|&i| yy[i] == ymax).collect();
    let [best] = best.as_slice() else {
        return Err(SurvivalError::computation("Ties for max(y), I surrender"));
    };
    let best = *best;
    if best == 0 {
        let mut new = xx[0] - 3.0 * (xx[1] - xx[0]);
        if let Some(lower) = lower
            && new < lower
        {
            let above = xx
                .iter()
                .copied()
                .filter(|&v| v > lower)
                .fold(f64::INFINITY, f64::min);
            new = lower + (above - lower) / 10.0;
        }
        return Ok(new);
    }
    if best == n - 1 {
        let mut new = xx[n - 1] + 3.0 * (xx[n - 1] - xx[n - 2]);
        if let Some(upper) = upper
            && new > upper
        {
            let below = xx
                .iter()
                .copied()
                .filter(|&v| v < upper)
                .fold(f64::NEG_INFINITY, f64::max);
            new = upper + (below - upper) / 10.0;
        }
        return Ok(new);
    }
    let xx = &xx[best - 1..=best + 1];
    let yy = &yy[best - 1..=best + 1];
    let temp1 =
        (xx[1] - xx[0]).powi(2) * (yy[1] - yy[2]) - (xx[1] - xx[2]).powi(2) * (yy[1] - yy[0]);
    let temp2 = (xx[1] - xx[0]) * (yy[1] - yy[2]) - (xx[1] - xx[2]) * (yy[1] - yy[0]);
    let new = xx[1] - 0.5 * temp1 / temp2;
    if new < xx[0] || new > xx[2] || (n > 4 && (new - x[n - 1]) > 0.5 * (x[n - 2] - x[n - 3]).abs())
    {
        if (xx[1] - xx[0]) > (xx[2] - xx[1]) {
            Ok(xx[1] - 0.38 * (xx[1] - xx[0]))
        } else {
            Ok(xx[1] + 0.32 * (xx[2] - xx[1]))
        }
    } else {
        Ok(new)
    }
}

pub(crate) fn frailty_gammacon(d: &[f64], nu: f64) -> f64 {
    let maxd = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let term1 = if nu > 1e7 * maxd {
        d.iter().map(|v| v * v).sum::<f64>() / nu
    } else {
        d.iter().map(|v| v + nu * (nu / (nu + v)).ln()).sum()
    };
    let maxd = maxd.max(0.0) as usize;
    let mut tbl = vec![0usize; maxd + 1];
    for &v in d {
        if v > 0.0 && v.fract() == 0.0 && (v as usize) <= maxd {
            tbl[v as usize] += 1;
        }
    }
    let mut term2 = 0.0;
    let mut ctbl = 0usize;
    for level in (1..=maxd).rev() {
        ctbl += tbl[level];
        term2 += ctbl as f64 * (nu + level as f64 - 1.0).ln();
        term2 -= (tbl[level] * level) as f64 * (nu + level as f64).ln();
    }
    term1 + term2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(iter: usize, df: f64, events: &'a [f64], coef: &'a [f64]) -> ControlInput<'a> {
        ControlInput {
            iter,
            plik: -100.0,
            loglik: -102.0,
            neff: 50.0,
            df,
            trh: 3.0,
            events_by_group: events,
            coef,
        }
    }

    #[test]
    fn gammacon_matches_r_formula() {
        let d = [2.0, 0.0, 1.0, 2.0];
        let nu = 4.0_f64;
        let term1: f64 = d.iter().map(|v| v + nu * (nu / (nu + v)).ln()).sum();
        let term2 = 3.0 * nu.ln() + 2.0 * (nu + 1.0).ln() - (nu + 1.0).ln() - 4.0 * (nu + 2.0).ln();
        assert!((frailty_gammacon(&d, nu) - (term1 + term2)).abs() < 1e-12);
        let nu = 1e9_f64;
        let term2 = 3.0 * nu.ln() + 2.0 * (nu + 1.0).ln() - (nu + 1.0).ln() - 4.0 * (nu + 2.0).ln();
        assert!((frailty_gammacon(&d, nu) - (9.0 / nu + term2)).abs() < 1e-12);
    }

    #[test]
    fn brent_brackets_then_interpolates() {
        assert_eq!(
            frailty_brent(&[1.0, 3.0], &[0.0, 1.0], None, None).unwrap(),
            2.0
        );
        let out = frailty_brent(
            &[0.5, 0.95, 0.725],
            &[-747.0, -745.0, -746.0],
            Some(0.0),
            Some(1.0),
        )
        .unwrap();
        assert!((out - (1.0 + (0.95 - 1.0) / 10.0)).abs() < 1e-12);
        let out = frailty_brent(&[0.1, 0.5, 1.0], &[3.0, 2.0, 1.0], Some(0.0), None).unwrap();
        assert!((out - (0.0 + 0.1 / 10.0)).abs() < 1e-12);
        let out = frailty_brent(&[0.0, 2.0, 1.0], &[0.0, 0.0, 1.0], None, None).unwrap();
        assert!((out - 1.0).abs() < 1e-12);
        assert!(frailty_brent(&[0.0, 1.0, 2.0], &[1.0, 1.0, 0.0], None, None).is_err());
    }

    #[test]
    fn df_control_follows_r_steps() {
        let control = Control::Df {
            df: 2.0,
            eps: 0.1,
            thetas: vec![0.0],
            dfs: vec![3.0],
            guess: 1.0,
            gamma_correction: false,
        };
        let state0 = control.initial();
        assert_eq!(state0.theta, 1.0);
        assert_eq!(state0.history, vec![vec![0.0, 3.0]]);
        let state1 = control.update(&state0, input(1, 2.98, &[], &[])).unwrap();
        let expected = 0.0 + 1.0 * (2.0 - 3.0) / (2.98 - 3.0);
        assert!((state1.theta - expected).abs() < 1e-12);
        assert!(!state1.done);
        assert_eq!(state1.half, Some(0));
        let state2 = control.update(&state1, input(2, 2.25, &[], &[])).unwrap();
        assert_eq!(state2.history.len(), 3);
        assert!(state2.theta > state1.theta);
        assert!(!state2.done);
    }

    #[test]
    fn df_control_reports_a_non_finite_df_as_an_error() {
        let control = Control::Df {
            df: 5.0,
            eps: 0.1,
            thetas: vec![0.0],
            dfs: vec![1.0],
            guess: 1.0,
            gamma_correction: false,
        };
        let state1 = control
            .update(&control.initial(), input(1, 0.99, &[], &[]))
            .unwrap();
        let error = control
            .update(&state1, input(2, f64::NAN, &[], &[]))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite df"));
    }

    #[test]
    fn gamma_control_alternates_and_reports_c_loglik() {
        let control = Control::Gamma {
            theta: None,
            eps: 1e-5,
            init: None,
        };
        let events = [2.0, 1.0, 3.0];
        let state0 = control.initial();
        assert_eq!(state0.theta, 0.0);
        let state1 = control
            .update(&state0, input(1, 1.0, &events, &[]))
            .unwrap();
        assert_eq!(state1.theta, 1.0);
        assert_eq!(state1.c_loglik, Some(-102.0));
        let state2 = control
            .update(&state1, input(2, 1.0, &events, &[]))
            .unwrap();
        let correct = frailty_gammacon(&events, 1.0);
        assert!((state2.c_loglik.unwrap() - (-102.0 + correct)).abs() < 1e-12);
        assert_eq!(state2.history.len(), 2);
        let fixed = Control::Gamma {
            theta: Some(0.5),
            eps: 1e-5,
            init: None,
        };
        let state = fixed
            .update(&fixed.initial(), input(1, 1.0, &events, &[]))
            .unwrap();
        assert!(state.done);
        assert_eq!(state.theta, 0.5);
    }

    #[test]
    fn aic_and_gauss_controls_step_as_in_r() {
        let control = Control::Aic {
            eps: 1e-5,
            init: vec![0.1, 1.0],
            lower: 0.0,
            upper: None,
            caic: false,
            gamma_correction: false,
        };
        let state0 = control.initial();
        assert_eq!(state0.theta, 0.1);
        let state1 = control.update(&state0, input(1, 3.0, &[], &[])).unwrap();
        assert_eq!(state1.theta, 1.0);
        assert_eq!(state1.history[0][3], -103.0);
        let state2 = control.update(&state1, input(2, 5.0, &[], &[])).unwrap();
        assert!((state2.theta - 0.55).abs() < 1e-12);

        let gauss = Control::Gauss {
            eps: 1e-4,
            init: None,
        };
        let coef = [1.0, -1.0, 0.5];
        let state0 = gauss.initial();
        assert_eq!(state0.theta, 1.0);
        let state1 = gauss.update(&state0, input(1, 1.0, &[], &coef)).unwrap();
        assert_eq!(state1.theta, 3.0);
        assert_eq!(state1.history[0][2], 2.25);
    }
}
