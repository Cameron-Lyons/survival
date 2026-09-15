//! Expected survival curves from a population rate table: the data side of
//! R's `survexp` (`R/survexp.R`) once the model frame has been evaluated
//! and matched to the table with `match_ratetable`.  A Cox model used as a
//! rate table (`survexp.cfit`) is not handled here.

use super::pyears::rows_to_matrix;
use super::ratetable::RateTable;
use super::survexp_fit::survexp_fit;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use ndarray::Array2;
use pyo3::prelude::*;

/// R's `method` argument of `survexp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvexpMethod {
    /// Ederer I: everyone is followed to the last time point.
    Ederer,
    /// Hakulinen: the cohort's expected survival given the observed follow-up.
    Hakulinen,
    /// Conditional expected survival.
    Conditional,
    /// One expected cumulative hazard per subject at its own follow-up.
    IndividualH,
    /// One expected survival per subject at its own follow-up.
    IndividualS,
}

impl SurvexpMethod {
    /// Parse R's `match.arg` choices.
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value {
            "ederer" => Ok(Self::Ederer),
            "hakulinen" => Ok(Self::Hakulinen),
            "conditional" => Ok(Self::Conditional),
            "individual.h" => Ok(Self::IndividualH),
            "individual.s" => Ok(Self::IndividualS),
            _ => Err(SurvivalError::invalid_input(
                "method must be 'ederer', 'hakulinen', 'conditional', 'individual.h' or 'individual.s'",
            )),
        }
    }

    /// R's historical defaults when `method` is not given: `conditional`
    /// and `cohort` decide, then whether a response is present.
    pub fn default_for(cohort: bool, conditional: bool, has_response: bool) -> Self {
        if !cohort {
            Self::IndividualS
        } else if conditional {
            Self::Conditional
        } else if has_response {
            Self::Hakulinen
        } else {
            Self::Ederer
        }
    }

    fn is_individual(self) -> bool {
        matches!(self, Self::IndividualH | Self::IndividualS)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ederer => "ederer",
            Self::Hakulinen => "hakulinen",
            Self::Conditional => "conditional",
            Self::IndividualH => "individual.h",
            Self::IndividualS => "individual.s",
        }
    }
}

/// The arguments of R's `survexp` that survive `model.frame`.
pub struct SurvexpInput<'a> {
    /// `match_ratetable`'s matrix, one row per subject.
    pub positions: &'a Array2<f64>,
    /// The response follow-up (`Surv(time, status)` reduced to `time`), or
    /// `None` for a formula without a response.
    pub y: Option<&'a [f64]>,
    /// Zero-based curve number of each subject (`strata(mf[ovars])`), or
    /// `None` for a single curve.
    pub group: Option<&'a [usize]>,
    /// Requested output times.
    pub times: Option<&'a [f64]>,
    /// R's `method` argument, or `None` to use the historical defaults.
    pub method: Option<SurvexpMethod>,
    /// R's `cohort` and `conditional` arguments.
    pub cohort: bool,
    pub conditional: bool,
    /// Divisor applied to the output times.
    pub scale: f64,
}

/// Expected survival curves (`ntime x ngroup`) or, for the individual
/// methods, one value per subject in `surv[i][0]` with empty `time` and
/// `n_risk`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvExpResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub surv: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_risk: Vec<Vec<f64>>,
    /// R's label: `Ederer` without a response, `conditional` when the
    /// `conditional` argument was set, `cohort` otherwise; the individual
    /// methods report their own name.
    #[pyo3(get)]
    pub method: String,
}

/// R's `survexp` for a rate table.
pub fn survexp(ratetable: &RateTable, input: SurvexpInput<'_>) -> SurvivalResult<SurvExpResult> {
    let n = input.positions.nrows();
    if n == 0 {
        return Err(SurvivalError::invalid_input("Data set has 0 rows"));
    }
    if input.scale.is_nan() || input.scale <= 0.0 || !input.scale.is_finite() {
        return Err(SurvivalError::invalid_input("scale must be a value > 0"));
    }
    if let Some(times) = input.times {
        validate_finite(times, "times")?;
        if times.iter().any(|&t| t < 0.0) {
            return Err(SurvivalError::invalid_input("Invalid time point requested"));
        }
        if times.windows(2).any(|w| w[1] < w[0]) {
            return Err(SurvivalError::invalid_input(
                "Times must be in increasing order",
            ));
        }
    }
    if let Some(y) = input.y {
        validate_length(n, y.len(), "y")?;
        validate_finite(y, "y")?;
        validate_non_negative(y, "y")?;
    }
    let method = input.method.unwrap_or_else(|| {
        SurvexpMethod::default_for(input.cohort, input.conditional, input.y.is_some())
    });
    if input.y.is_none() && method != SurvexpMethod::Ederer {
        return Err(SurvivalError::invalid_input(
            "a response is required in the formula unless method='ederer'",
        ));
    }
    let group = match input.group {
        Some(group) => {
            validate_length(n, group.len(), "group")?;
            group.to_vec()
        }
        None => vec![0; n],
    };

    if method.is_individual() {
        let y = input.y.ok_or_else(|| {
            SurvivalError::invalid_input(
                "for individual survival an observation time must be given",
            )
        })?;
        let max_y = y.iter().copied().fold(0.0, f64::max);
        let ids: Vec<usize> = (0..n).collect();
        let fit = survexp_fit(&ids, input.positions, Some(y), &[max_y], true, ratetable)?;
        let surv = fit
            .surv
            .row(0)
            .iter()
            .map(|&s| {
                if method == SurvexpMethod::IndividualS {
                    vec![s]
                } else {
                    vec![-s.ln()]
                }
            })
            .collect();
        return Ok(SurvExpResult {
            time: Vec::new(),
            surv,
            n_risk: Vec::new(),
            method: method.name().to_string(),
        });
    }

    // newtime: the requested times plus, with a response, its unique
    // follow-up times before the last request.
    let (newtime, y) = match (input.y, input.times) {
        (None, None) => {
            return Err(SurvivalError::invalid_input(
                "either a times argument or a response is needed",
            ));
        }
        (None, Some(times)) => {
            let max_time = times.iter().copied().fold(0.0, f64::max);
            (sorted_unique(times), vec![max_time; n])
        }
        (Some(y), None) => (sorted_unique(y), y.to_vec()),
        (Some(y), Some(times)) => {
            let max_time = times.iter().copied().fold(0.0, f64::max);
            let mut all = times.to_vec();
            all.extend(y.iter().copied().filter(|&t| t < max_time));
            (sorted_unique(&all), y.to_vec())
        }
    };
    let fit = survexp_fit(
        &group,
        input.positions,
        Some(&y),
        &newtime,
        method == SurvexpMethod::Conditional,
        ratetable,
    )?;
    let n_groups = fit.surv.ncols();

    let (time, surv, n_risk) = match input.times {
        None => (
            fit.times.clone(),
            fit.surv.rows().into_iter().map(|r| r.to_vec()).collect(),
            fit.n
                .rows()
                .into_iter()
                .map(|r| r.iter().map(|&v| v as f64).collect())
                .collect(),
        ),
        Some(times) => {
            // keep <- match(times, newtime); surv <- rbind(1, surv)[keep + 1, ]
            let mut surv = Vec::with_capacity(times.len());
            let mut n_risk = Vec::with_capacity(times.len());
            for &t in times {
                let keep = fit
                    .times
                    .iter()
                    .position(|&value| value == t)
                    .ok_or_else(|| SurvivalError::computation("requested time missing from fit"))?;
                surv.push(fit.surv.row(keep).to_vec());
                n_risk.push(fit.n.row(keep).iter().map(|&v| v as f64).collect());
            }
            (times.to_vec(), surv, n_risk)
        }
    };
    debug_assert!(surv.iter().all(|row| row.len() == n_groups));
    let label = if input.y.is_none() {
        "Ederer"
    } else if input.conditional {
        "conditional"
    } else {
        "cohort"
    };
    Ok(SurvExpResult {
        time: time.iter().map(|t| t / input.scale).collect(),
        surv,
        n_risk,
        method: label.to_string(),
    })
}

fn sorted_unique(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted.dedup();
    sorted
}

/// Python entry point of [`survexp`]: `positions` is `match_ratetable(...).r`,
/// `group` zero-based curve numbers, `method` one of R's choices or `None`.
#[pyfunction(name = "survexp")]
#[pyo3(signature = (ratetable, positions, y=None, group=None, times=None, method=None, cohort=true, conditional=false, scale=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn survexp_py(
    ratetable: &RateTable,
    positions: Vec<Vec<f64>>,
    y: Option<Vec<f64>>,
    group: Option<Vec<usize>>,
    times: Option<Vec<f64>>,
    method: Option<&str>,
    cohort: bool,
    conditional: bool,
    scale: f64,
) -> PyResult<SurvExpResult> {
    let positions = rows_to_matrix(&positions, positions.len(), ratetable.ndim(), "positions")?;
    let method = method.map(SurvexpMethod::parse).transpose()?;
    Ok(survexp(
        ratetable,
        SurvexpInput {
            positions: &positions,
            y: y.as_deref(),
            group: group.as_deref(),
            times: times.as_deref(),
            method,
            cohort,
            conditional,
            scale,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::ratetable::DimType;
    use crate::population::ratetable_data::survexp_us_table;

    fn table() -> RateTable {
        RateTable::try_new(
            vec![2],
            vec!["age".into()],
            vec![vec!["0".into(), "10".into()]],
            vec![Some(vec![0.0, 10.0])],
            vec![DimType::Continuous],
            vec![0.01, 0.03],
        )
        .unwrap()
    }

    fn input<'a>(
        positions: &'a Array2<f64>,
        y: Option<&'a [f64]>,
        times: Option<&'a [f64]>,
        method: Option<SurvexpMethod>,
    ) -> SurvexpInput<'a> {
        SurvexpInput {
            positions,
            y,
            group: None,
            times,
            method,
            cohort: true,
            conditional: false,
            scale: 1.0,
        }
    }

    #[test]
    fn ederer_follows_everyone_to_the_last_time() {
        let table = table();
        let positions = ndarray::arr2(&[[0.0], [5.0]]);
        let out = survexp(&table, input(&positions, None, Some(&[0.0, 10.0]), None)).unwrap();
        assert_eq!(out.method, "Ederer");
        assert_eq!(out.time, vec![0.0, 10.0]);
        assert_eq!(out.surv[0], vec![1.0]);
        let expected = ((-0.1f64).exp() + (-(0.05f64 + 0.15)).exp()) / 2.0;
        assert!((out.surv[1][0] - expected).abs() < 1e-12);
        assert_eq!(out.n_risk, vec![vec![2.0], vec![2.0]]);
    }

    #[test]
    fn hakulinen_uses_observed_follow_up_and_requested_times() {
        let table = table();
        let positions = ndarray::arr2(&[[0.0], [0.0]]);
        let y = [5.0, 20.0];
        let out = survexp(
            &table,
            input(
                &positions,
                Some(&y),
                Some(&[0.0, 10.0]),
                Some(SurvexpMethod::Hakulinen),
            ),
        )
        .unwrap();
        assert_eq!(out.method, "cohort");
        assert_eq!(out.time, vec![0.0, 10.0]);
        // newtime is 0, 5, 10: subject 1 leaves after day 5.
        assert_eq!(out.n_risk, vec![vec![2.0], vec![1.0]]);
        assert!((out.surv[1][0] - (-0.1f64).exp()).abs() < 1e-12);

        let default_times = survexp(&table, input(&positions, Some(&y), None, None)).unwrap();
        assert_eq!(default_times.time, vec![5.0, 20.0]);
        assert_eq!(default_times.n_risk, vec![vec![2.0], vec![1.0]]);
    }

    #[test]
    fn conditional_label_follows_the_argument_not_the_method() {
        let table = table();
        let positions = ndarray::arr2(&[[0.0]]);
        let y = [5.0];
        let mut args = input(
            &positions,
            Some(&y),
            Some(&[5.0]),
            Some(SurvexpMethod::Conditional),
        );
        assert_eq!(survexp(&table, args).unwrap().method, "cohort");
        args = input(&positions, Some(&y), Some(&[5.0]), None);
        args.conditional = true;
        let out = survexp(&table, args).unwrap();
        assert_eq!(out.method, "conditional");
        assert!((out.surv[0][0] - (-0.05f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn individual_methods_return_one_value_per_subject() {
        let table = table();
        let positions = ndarray::arr2(&[[0.0], [5.0]]);
        let y = [10.0, 10.0];
        let mut args = input(&positions, Some(&y), None, None);
        args.cohort = false;
        let out = survexp(&table, args).unwrap();
        assert_eq!(out.method, "individual.s");
        assert!(out.time.is_empty());
        assert!((out.surv[0][0] - (-0.1f64).exp()).abs() < 1e-12);
        assert!((out.surv[1][0] - (-0.2f64).exp()).abs() < 1e-12);
        let hazard = survexp(
            &table,
            input(&positions, Some(&y), None, Some(SurvexpMethod::IndividualH)),
        )
        .unwrap();
        assert!((hazard.surv[1][0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn groups_give_one_column_each_and_scale_divides_time() {
        let table = table();
        let positions = ndarray::arr2(&[[0.0], [5.0]]);
        let group = [1usize, 0];
        let mut args = input(&positions, None, Some(&[10.0]), None);
        args.group = Some(&group);
        args.scale = 10.0;
        let out = survexp(&table, args).unwrap();
        assert_eq!(out.time, vec![1.0]);
        assert!((out.surv[0][1] - (-0.1f64).exp()).abs() < 1e-12);
        assert!((out.surv[0][0] - (-0.2f64).exp()).abs() < 1e-12);
        assert_eq!(out.n_risk, vec![vec![1.0, 1.0]]);
    }

    #[test]
    fn rejects_invalid_requests() {
        let table = table();
        let positions = ndarray::arr2(&[[0.0]]);
        assert!(survexp(&table, input(&positions, None, None, None)).is_err());
        assert!(
            survexp(
                &table,
                input(
                    &positions,
                    None,
                    Some(&[1.0]),
                    Some(SurvexpMethod::Hakulinen)
                )
            )
            .is_err()
        );
        assert!(survexp(&table, input(&positions, None, Some(&[2.0, 1.0]), None)).is_err());
        assert!(survexp(&table, input(&positions, None, Some(&[-1.0]), None)).is_err());
        // The reviewer's case: a sex code survexp.us does not have.
        let off_table = ndarray::arr2(&[[109.0 * 365.25, 5.0, 18262.0]]);
        let message = survexp(
            survexp_us_table(),
            input(&off_table, None, Some(&[0.0, 100.0]), None),
        )
        .unwrap_err()
        .to_string();
        assert!(
            message.contains("The variable sex is out of range"),
            "{message}"
        );
        assert!(SurvexpMethod::parse("individual").is_err());
        assert_eq!(
            SurvexpMethod::default_for(true, false, true),
            SurvexpMethod::Hakulinen
        );
    }
}
