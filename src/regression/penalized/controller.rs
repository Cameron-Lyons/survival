//! Validated, stateless access to the same searches used by the fitters.

use super::control::{Control, PenaltyControlInput, PenaltyControlState};
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A reusable smoothing-parameter search. Each fit owns its returned state;
/// the controller never retains coefficients, observations, or Python objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", frozen, skip_from_py_object)]
pub struct PenaltyController {
    control: Control,
}

fn require(condition: bool, message: &str) -> SurvivalResult<()> {
    if condition {
        Ok(())
    } else {
        Err(SurvivalError::invalid_input(message))
    }
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

impl PenaltyController {
    /// Validate a controller configuration before starting a search.
    pub fn new(control: Control) -> SurvivalResult<Self> {
        let this = Self { control };
        this.validate()?;
        Ok(this)
    }

    fn validate(&self) -> SurvivalResult<()> {
        let epsilon = |eps: f64| {
            require(
                eps.is_finite() && eps > 0.0,
                "eps must be finite and positive",
            )
        };
        let starts = |init: &Option<Vec<f64>>| {
            require(
                init.as_ref()
                    .is_none_or(|v| v.len() >= 2 && v.iter().all(|&x| nonnegative(x))),
                "init must contain at least two finite nonnegative values",
            )
        };
        match &self.control {
            Control::Fixed { theta } => {
                require(nonnegative(*theta), "theta must be finite and nonnegative")
            }
            Control::Gamma { theta, eps, init } => {
                epsilon(*eps)?;
                require(
                    theta.is_none_or(nonnegative),
                    "theta must be finite and nonnegative",
                )?;
                starts(init)
            }
            Control::Gauss { eps, init } => {
                epsilon(*eps)?;
                starts(init)
            }
            Control::Df {
                df,
                eps,
                thetas,
                dfs,
                guess,
                ..
            } => {
                epsilon(*eps)?;
                require(
                    nonnegative(*df) && nonnegative(*guess),
                    "target_df and guess must be finite and nonnegative",
                )?;
                require(
                    !thetas.is_empty() && thetas.len() == dfs.len(),
                    "thetas and dfs must have the same nonzero length",
                )?;
                require(
                    thetas.iter().chain(dfs).all(|&x| nonnegative(x)),
                    "thetas and dfs must be finite and nonnegative",
                )
            }
            Control::Aic {
                eps,
                init,
                lower,
                upper,
                ..
            } => {
                epsilon(*eps)?;
                require(
                    init.iter().all(|&x| nonnegative(x)),
                    "init must be finite and nonnegative",
                )?;
                require(
                    nonnegative(*lower) && upper.is_none_or(|u| u.is_finite() && u > *lower),
                    "bounds must be finite, nonnegative, and increasing",
                )
            }
        }
    }

    /// History column names, in the order used by R's penalty controllers.
    pub fn columns(&self) -> &'static [&'static str] {
        self.control.history_columns()
    }

    /// Whether the search needs the fitted degrees of freedom or Hessian trace.
    pub fn needs_df(&self) -> bool {
        self.control.needs_df()
    }

    /// The initial theta and, for a df search, its known starting points.
    pub fn initial(&self) -> SurvivalResult<PenaltyControlState> {
        self.validate()?;
        Ok(self.control.initial())
    }

    /// Propose the next theta from a caller-owned previous state and fit inputs.
    /// Iterations are consecutive and one-based. An unusable numerical proposal
    /// is an error, including R boundary cases that would return NA or NaN.
    pub fn step(
        &self,
        old: &PenaltyControlState,
        input: PenaltyControlInput<'_>,
    ) -> SurvivalResult<PenaltyControlState> {
        self.validate()?;
        require(
            input.iter > 0,
            "iter must be positive; use initial() for iteration zero",
        )?;
        require(
            nonnegative(old.theta),
            "old theta must be finite and nonnegative",
        )?;
        require(old.half.is_none_or(|h| h >= 0), "half must be nonnegative")?;
        require(
            old.history
                .iter()
                .all(|row| row.len() == self.columns().len() && row.iter().all(|v| v.is_finite())),
            "history must have finite values and the controller's column count",
        )?;
        let expected = match &self.control {
            Control::Fixed { .. } | Control::Gamma { theta: Some(_), .. } => 0,
            Control::Df { thetas, .. } => thetas
                .len()
                .checked_add(input.iter - 1)
                .ok_or_else(|| SurvivalError::invalid_input("iter is too large"))?,
            _ => input.iter - 1,
        };
        require(
            old.history.len() == expected,
            "history row count does not match iter",
        )?;
        if matches!(self.control, Control::Df { .. } | Control::Aic { .. }) {
            require(nonnegative(input.df), "df must be finite and nonnegative")?;
        }
        if matches!(self.control, Control::Aic { .. }) {
            require(
                nonnegative(input.neff) && input.plik.is_finite(),
                "neff must be finite and nonnegative and plik must be finite",
            )?;
        }
        if matches!(self.control, Control::Gauss { .. }) {
            require(
                !input.coef.is_empty()
                    && input.coef.iter().all(|x| x.is_finite())
                    && input.trh.is_finite(),
                "coef must be nonempty and finite and trh must be finite",
            )?;
        }
        if matches!(
            self.control,
            Control::Gamma { .. }
                | Control::Df {
                    gamma_correction: true,
                    ..
                }
                | Control::Aic {
                    gamma_correction: true,
                    ..
                }
        ) {
            require(input.loglik.is_finite(), "loglik must be finite")?;
            // The shared correction uses a histogram indexed by event count.
            // Limit public allocations to 80 MB, independently of caller data.
            require(
                input
                    .events_by_group
                    .iter()
                    .all(|&d| nonnegative(d) && d.fract() == 0.0 && d <= 10_000_000.0),
                "events_by_group must contain integer counts between 0 and 10000000",
            )?;
        }
        let state = self.control.update(old, input)?;
        if !nonnegative(state.theta)
            || state.history.iter().flatten().any(|x| !x.is_finite())
            || state.c_loglik.is_some_and(|x| !x.is_finite())
        {
            return Err(SurvivalError::computation(
                "penalty controller produced a nonfinite or negative proposal",
            ));
        }
        Ok(state)
    }
}

#[cfg(feature = "python")]
mod python {
    use super::*;
    use crate::internal::numpy_utils::{FloatRows, FloatVec};

    #[pymethods]
    impl PenaltyControlState {
        #[new]
        #[pyo3(signature = (theta, *, done=false, history=None, c_loglik=None, half=None))]
        fn py_new(
            theta: f64,
            done: bool,
            history: Option<FloatRows>,
            c_loglik: Option<f64>,
            half: Option<i64>,
        ) -> PyResult<Self> {
            require(nonnegative(theta), "theta must be finite and nonnegative")?;
            require(half.is_none_or(|h| h >= 0), "half must be nonnegative")?;
            let history = history.map(FloatRows::into_inner).unwrap_or_default();
            require(
                history.iter().flatten().all(|v| v.is_finite())
                    && c_loglik.is_none_or(|x| x.is_finite()),
                "state must contain finite values",
            )?;
            Ok(Self {
                theta,
                done,
                history,
                c_loglik,
                half,
                theta_history_index: None,
            })
        }

        fn __reduce__<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<crate::internal::pickle::Reduced<'py>> {
            crate::internal::pickle::reduce(py, self)
        }
    }

    #[pymethods]
    impl PenaltyController {
        /// Methods: fixed, gamma, df, aic, gaussian. Init contains starting theta
        /// values; a df search also requires target_df, thetas, dfs and guess.
        /// Gamma correction accepts integer counts up to ten million per group.
        #[new]
        #[pyo3(signature = (method, *, theta=None, eps=1e-5, init=None, target_df=None, thetas=None, dfs=None, guess=None, lower=0.0, upper=None, caic=false, gamma_correction=false))]
        #[allow(clippy::too_many_arguments)]
        fn py_new(
            method: &str,
            theta: Option<f64>,
            eps: f64,
            init: Option<FloatVec>,
            target_df: Option<f64>,
            thetas: Option<FloatVec>,
            dfs: Option<FloatVec>,
            guess: Option<f64>,
            lower: f64,
            upper: Option<f64>,
            caic: bool,
            gamma_correction: bool,
        ) -> PyResult<Self> {
            let required = |name: &str| {
                SurvivalError::invalid_input(format!("{name} is required for {method}"))
            };
            require(
                theta.is_none() || matches!(method, "fixed" | "gamma"),
                "theta is only used by fixed and gamma controllers",
            )?;
            require(
                method == "df"
                    || (target_df.is_none()
                        && thetas.is_none()
                        && dfs.is_none()
                        && guess.is_none()),
                "target_df, thetas, dfs, and guess are only used by df controllers",
            )?;
            require(
                method == "aic" || (lower == 0.0 && upper.is_none() && !caic),
                "bounds and caic are only used by aic controllers",
            )?;
            require(
                !gamma_correction || matches!(method, "df" | "aic"),
                "gamma_correction is only used by df and aic controllers",
            )?;
            require(
                init.is_none() || matches!(method, "gamma" | "gaussian" | "aic"),
                "init is only used by gamma, gaussian, and aic controllers",
            )?;
            let init = init.map(FloatVec::into_inner);
            let control = match method {
                "fixed" => Control::Fixed {
                    theta: theta.ok_or_else(|| required("theta"))?,
                },
                "gamma" => Control::Gamma { theta, eps, init },
                "gaussian" => Control::Gauss { eps, init },
                "df" => Control::Df {
                    df: target_df.ok_or_else(|| required("target_df"))?,
                    eps,
                    thetas: thetas.ok_or_else(|| required("thetas"))?.into_inner(),
                    dfs: dfs.ok_or_else(|| required("dfs"))?.into_inner(),
                    guess: guess.ok_or_else(|| required("guess"))?,
                    gamma_correction,
                },
                "aic" => Control::Aic {
                    eps,
                    init: init.unwrap_or_default(),
                    lower,
                    upper,
                    caic,
                    gamma_correction,
                },
                _ => {
                    return Err(SurvivalError::invalid_input(
                        "method must be fixed, gamma, df, aic, or gaussian",
                    )
                    .into());
                }
            };
            Ok(Self::new(control)?)
        }

        #[getter(columns)]
        fn py_columns(&self) -> Vec<String> {
            self.columns().iter().map(|s| s.to_string()).collect()
        }

        #[getter(needs_df)]
        fn py_needs_df(&self) -> bool {
            self.needs_df()
        }

        #[pyo3(name = "initial")]
        fn py_initial(&self) -> PyResult<PenaltyControlState> {
            Ok(self.initial()?)
        }

        /// Advance one fit iteration; inputs are copied before releasing the GIL.
        #[pyo3(name = "step", signature = (old, iter, *, plik=0.0, loglik=0.0, neff=0.0, df=0.0, trh=0.0, events_by_group=None, coef=None))]
        #[allow(clippy::too_many_arguments)]
        fn py_step(
            &self,
            py: Python<'_>,
            old: &PenaltyControlState,
            iter: usize,
            plik: f64,
            loglik: f64,
            neff: f64,
            df: f64,
            trh: f64,
            events_by_group: Option<FloatVec>,
            coef: Option<FloatVec>,
        ) -> PyResult<PenaltyControlState> {
            let events = events_by_group
                .map(FloatVec::into_inner)
                .unwrap_or_default();
            let coef = coef.map(FloatVec::into_inner).unwrap_or_default();
            Ok(py.detach(|| {
                self.step(
                    old,
                    PenaltyControlInput {
                        iter,
                        plik,
                        loglik,
                        neff,
                        df,
                        trh,
                        events_by_group: &events,
                        coef: &coef,
                    },
                )
            })?)
        }

        fn __reduce__<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<crate::internal::pickle::Reduced<'py>> {
            crate::internal::pickle::reduce(py, self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn df_boundary_is_an_error_instead_of_a_panic() {
        let search = PenaltyController::new(Control::Df {
            df: 2.0,
            eps: 0.1,
            thetas: vec![0.0],
            dfs: vec![0.0],
            guess: 1.0,
            gamma_correction: false,
        })
        .unwrap();
        let state = PenaltyControlState {
            theta: 2.0,
            done: false,
            history: vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            c_loglik: None,
            half: None,
            theta_history_index: None,
        };
        let result = search.step(
            &state,
            PenaltyControlInput {
                iter: 2,
                df: 2.0,
                ..Default::default()
            },
        );
        assert!(result.unwrap_err().to_string().contains("upper bracket"));
    }

    #[test]
    fn malformed_history_and_initial_values_are_rejected() {
        assert!(
            PenaltyController::new(Control::Gauss {
                eps: 1e-5,
                init: Some(vec![])
            })
            .is_err()
        );
        let search = PenaltyController::new(Control::Gamma {
            theta: None,
            eps: 1e-5,
            init: None,
        })
        .unwrap();
        let mut old = search.initial().unwrap();
        assert!(
            search
                .step(
                    &old,
                    PenaltyControlInput {
                        iter: 2,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        old.history = vec![vec![0.0]];
        assert!(
            search
                .step(
                    &old,
                    PenaltyControlInput {
                        iter: 2,
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn controller_does_not_modify_the_previous_state() {
        let search = PenaltyController::new(Control::Gamma {
            theta: None,
            eps: 1e-5,
            init: None,
        })
        .unwrap();
        let old = search.initial().unwrap();
        let next = search
            .step(
                &old,
                PenaltyControlInput {
                    iter: 1,
                    loglik: -5.0,
                    events_by_group: &[2.0, 1.0],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(old.theta, 0.0);
        assert!(old.history.is_empty());
        assert_eq!(next.history, vec![vec![0.0, -5.0, -5.0]]);
    }
}
