use crate::concordance::ConcordanceFit;
use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::core::strata_order::{grouped_sum, order_within_strata, validate_intervals};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::cox_optimizer::TieMethod;
use crate::surv_analysis::agsurv::AgsurvCurve;
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct CoxphData {
    pub time: Vec<f64>,
    pub entry: Option<Vec<f64>>,
    pub status: Vec<i32>,
    pub x: Array2<f64>,
    pub weights: Option<Vec<f64>>,
    pub strata: Option<Vec<i32>>,
    pub offset: Option<Vec<f64>>,
}

impl CoxphData {
    pub fn try_new(
        time: Vec<f64>,
        entry: Option<Vec<f64>>,
        status: Vec<i32>,
        x: Array2<f64>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        let data = Self {
            time,
            entry,
            status,
            x,
            weights,
            strata,
            offset,
        };
        data.validate()?;
        Ok(data)
    }

    pub(crate) fn validate(&self) -> SurvivalResult<()> {
        let n = self.n();
        if n == 0 {
            return Err(SurvivalError::invalid_input(
                "No (non-missing) observations",
            ));
        }
        validate_finite(&self.time, "time")?;
        validate_length(n, self.status.len(), "status")?;
        validate_binary_i32(&self.status, "status")?;
        validate_length(n, self.x.nrows(), "x")?;
        if let Some(entry) = &self.entry {
            validate_length(n, entry.len(), "entry")?;
            validate_finite(entry, "entry")?;
            validate_intervals(entry, &self.time)?;
        }
        if let Some(weights) = &self.weights {
            validate_length(n, weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
        }
        if let Some(strata) = &self.strata {
            validate_length(n, strata.len(), "strata")?;
        }
        if let Some(offset) = &self.offset {
            validate_length(n, offset.len(), "offset")?;
            validate_finite(offset, "offset")?;
        }
        Ok(())
    }

    pub fn n(&self) -> usize {
        self.time.len()
    }

    pub fn check_fit_input(&self) -> SurvivalResult<()> {
        if let Some(value) = self.x.iter().find(|value| !value.is_finite()) {
            return Err(SurvivalError::invalid_input(format!(
                "x contains non-finite value {value}"
            )));
        }
        if self
            .weights
            .as_ref()
            .is_some_and(|weights| weights.iter().any(|&w| w <= 0.0))
        {
            return Err(SurvivalError::invalid_input("Invalid weights, must be >0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CoxphOptions {
    pub method: TieMethod,
    pub init: Option<Vec<f64>>,
    pub iter_max: usize,
    pub eps: f64,
    pub toler_chol: f64,
    pub nocenter: Option<Vec<f64>>,
    pub cluster: Option<Vec<i32>>,
    pub robust: Option<bool>,
}

impl Default for CoxphOptions {
    fn default() -> Self {
        Self {
            method: TieMethod::Efron,
            init: None,
            iter_max: COX_MAX_ITER,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler_chol: COX_RANK_TOLERANCE,
            nocenter: Some(vec![-1.0, 0.0, 1.0]),
            cluster: None,
            robust: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SortedRows {
    pub order: Vec<usize>,
    pub bounds: Vec<(usize, usize)>,
    pub codes: Vec<i32>,
    pub stratum_index: Vec<usize>,
}

impl SortedRows {
    pub(super) fn new(order: Vec<usize>, strata: Option<&[i32]>) -> Self {
        let n = order.len();
        let mut bounds = Vec::new();
        let mut codes = Vec::new();
        let mut stratum_index = vec![0; n];
        let mut start = 0;
        for position in 0..n {
            let code = strata.map_or(0, |s| s[order[position]]);
            let last = position + 1 == n || strata.is_some_and(|s| s[order[position + 1]] != code);
            if last {
                bounds.push((start, position + 1));
                codes.push(code);
                for &row in &order[start..=position] {
                    stratum_index[row] = codes.len() - 1;
                }
                start = position + 1;
            }
        }
        Self {
            order,
            bounds,
            codes,
            stratum_index,
        }
    }

    pub(crate) fn by_time(time: &[f64], strata: Option<&[i32]>) -> Self {
        let order = match strata {
            Some(strata) => order_within_strata(strata, |a, b| time[a].total_cmp(&time[b])),
            None => {
                let mut order: Vec<usize> = (0..time.len()).collect();
                order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
                order
            }
        };
        Self::new(order, strata)
    }

    pub(crate) fn nstrata(&self) -> usize {
        self.codes.len()
    }

    pub(crate) fn position_of(&self, code: i32) -> Option<usize> {
        self.codes.binary_search(&code).ok()
    }
}

#[doc = " A fitted Cox model (R's `coxph` object).  As in R, the coefficient of a"]
#[doc = " redundant (aliased) covariate is `NaN` (R's `NA`) with a zero row and"]
#[doc = " column in `var`; every computation on the fit treats it as 0."]
#[pyclass(module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxPHFit {
    #[doc = " Coefficients; `NaN` marks an aliased covariate."]
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    pub var: Array2<f64>,
    pub naive_var: Option<Array2<f64>>,
    #[doc = " Log partial likelihood at the initial and the final coefficients."]
    #[pyo3(get)]
    pub loglik: [f64; 2],
    #[doc = " Score test at the initial coefficients."]
    #[pyo3(get)]
    pub score: f64,
    #[doc = " Robust score test (present with the robust variance)."]
    #[pyo3(get)]
    pub rscore: Option<f64>,
    #[pyo3(get)]
    pub wald_test: f64,
    #[pyo3(get)]
    pub iter: usize,
    #[doc = " Rank of the information matrix (`< nvar` marks aliased columns),"]
    #[doc = " `-2` converged while step halving, `1000` did not converge."]
    #[pyo3(get)]
    pub flag: i32,
    #[doc = " `agreg.fit`'s `info` for a (start, stop] Breslow or Efron fit: the"]
    #[doc = " rank of the information matrix at the initial coefficients, the"]
    #[doc = " number of recentrings of the risk scores, the number of step"]
    #[doc = " halvings, and 1 when the iterations ran out.  `None` for the other"]
    #[doc = " fitters."]
    #[pyo3(get)]
    pub info: Option<[i32; 4]>,
    #[doc = " `x %*% coef + offset - sum(coef * means)`."]
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[doc = " Martingale residuals."]
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[doc = " Column centres (weighted means for right-censored data, plain means"]
    #[doc = " for counting-process data; 0 for `nocenter` columns)."]
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[doc = " Score vector at the final coefficients (`agreg.fit`'s `first`)."]
    #[pyo3(get)]
    pub first: Vec<f64>,
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub nevent: usize,
    #[pyo3(get)]
    pub method: TieMethod,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub entry: Option<Vec<f64>>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    pub x: Array2<f64>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
    #[pyo3(get)]
    pub offset: Vec<f64>,
    #[doc = " Columns that were neither centred nor scaled (`nocenter`)."]
    #[pyo3(get)]
    pub nocenter: Vec<bool>,
    #[pyo3(get)]
    pub cluster: Option<Vec<i32>>,
    #[doc = " The concordance of the linear predictors with the outcome, as"]
    #[doc = " `coxph()` computes it: `concordancefit(Y, lp, strata, weights,"]
    #[doc = " cluster, reverse = TRUE, timefix = FALSE)`.  R's `fit$concordance`"]
    #[doc = " vector is `(colSums(count), concordance, sqrt(var))` of this object,"]
    #[doc = " and `summary(fit)$concordance` its last two entries."]
    #[pyo3(get)]
    pub concordance: ConcordanceFit,
    pub(crate) sorted: SortedRows,
    #[serde(skip)]
    pub(super) curves: OnceLock<Vec<AgsurvCurve>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictReference {
    Strata,
    Sample,
    Zero,
}

impl PredictReference {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "strata" => Ok(Self::Strata),
            "sample" => Ok(Self::Sample),
            "zero" => Ok(Self::Zero),
            other => Err(SurvivalError::invalid_input(format!(
                "reference must be 'strata', 'sample' or 'zero', got '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoxNewData {
    pub x: Array2<f64>,
    pub strata: Option<Vec<i32>>,
    pub offset: Option<Vec<f64>>,
    pub time: Option<Vec<f64>>,
    pub entry: Option<Vec<f64>>,
}

impl CoxNewData {
    pub fn try_new(
        x: Array2<f64>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        time: Option<Vec<f64>>,
        entry: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        Self::validated(x, strata, offset, time, entry, false)
    }

    pub fn try_new_prediction(
        x: Array2<f64>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        time: Option<Vec<f64>>,
        entry: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        Self::validated(x, strata, offset, time, entry, true)
    }

    pub(super) fn validated(
        x: Array2<f64>,
        strata: Option<Vec<i32>>,
        offset: Option<Vec<f64>>,
        time: Option<Vec<f64>>,
        entry: Option<Vec<f64>>,
        allow_missing: bool,
    ) -> SurvivalResult<Self> {
        let data = Self {
            x,
            strata,
            offset,
            time,
            entry,
        };
        data.validate(allow_missing)?;
        Ok(data)
    }

    pub(super) fn validate(&self, allow_missing: bool) -> SurvivalResult<()> {
        let m = self.nrows();
        if m == 0 {
            return Err(SurvivalError::invalid_input(
                "newdata must have at least one row",
            ));
        }
        if let Some(value) = self
            .x
            .iter()
            .find(|value| value.is_infinite() || (!allow_missing && value.is_nan()))
        {
            return Err(SurvivalError::invalid_input(format!(
                "newdata contains non-finite value {value}"
            )));
        }
        if let Some(strata) = &self.strata {
            validate_length(m, strata.len(), "newdata strata")?;
        }
        if let Some(offset) = &self.offset {
            validate_length(m, offset.len(), "newdata offset")?;
            if let Some(value) = offset
                .iter()
                .find(|value| value.is_infinite() || (!allow_missing && value.is_nan()))
            {
                return Err(SurvivalError::invalid_input(format!(
                    "newdata offset contains non-finite value {value}"
                )));
            }
        }
        if let Some(time) = &self.time {
            validate_length(m, time.len(), "newdata time")?;
            validate_finite(time, "newdata time")?;
        }
        if let Some(entry) = &self.entry {
            validate_length(m, entry.len(), "newdata entry")?;
            validate_finite(entry, "newdata entry")?;
        }
        if let (Some(entry), Some(time)) = (&self.entry, &self.time) {
            validate_intervals(entry, time)?;
        }
        Ok(())
    }

    pub(super) fn nrows(&self) -> usize {
        self.x.nrows()
    }
}

#[doc = " `predict(type = \"lp\" | \"risk\" | \"expected\" | \"survival\")` output."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxPrediction {
    #[pyo3(get)]
    pub fit: Vec<f64>,
    #[pyo3(get)]
    pub se_fit: Option<Vec<f64>>,
}

impl CoxPrediction {
    pub fn collapse(self, group: &[i32]) -> SurvivalResult<Self> {
        let sum = |values: &[f64], squares| {
            let column = ArrayView2::from_shape((values.len(), 1), values)
                .expect("a column vector has a valid shape");
            grouped_sum(column, group, squares).map(|result| result.column(0).to_vec())
        };
        Ok(Self {
            fit: sum(&self.fit, false)?,
            se_fit: self.se_fit.as_deref().map(|se| sum(se, true)).transpose()?,
        })
    }
}

#[doc = " `predict(type = \"terms\")` output: one column per term."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxTermsPrediction {
    #[doc = " Column count retained independently of the row count."]
    #[pyo3(get)]
    pub n_columns: usize,
    #[pyo3(get)]
    pub fit: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub se_fit: Option<Vec<Vec<f64>>>,
    #[doc = " `sum(coefficients * means)`, R's `attr(pred, \"constant\")`."]
    #[pyo3(get)]
    pub constant: f64,
}

#[doc = " `basehaz(fit)`: the cumulative hazard at each event or censoring time"]
#[doc = " per stratum, rows concatenated in stratum order."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct Basehaz {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    #[doc = " Stratum code of each row (absent for an unstratified fit)."]
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
}

#[doc = " One curve of `survfit(fit, newdata)`: `surv`, `cumhaz` and `std_err`"]
#[doc = " have one row per time and one column per new observation."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxSurvfitCurve {
    #[doc = " Stratum code the curve belongs to."]
    #[pyo3(get)]
    pub stratum: i32,
    #[doc = " Number of observations in the stratum."]
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub surv: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    #[doc = " Standard error of the cumulative hazard (`std.err` with `logse`)."]
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SurvfitOptions {
    pub stype: u8,
    pub ctype: Option<u8>,
    pub se_fit: bool,
    pub censor: bool,
    pub start_time: Option<f64>,
}

impl Default for SurvfitOptions {
    fn default() -> Self {
        Self {
            stype: 2,
            ctype: None,
            se_fit: true,
            censor: true,
            start_time: None,
        }
    }
}
