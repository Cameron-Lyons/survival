//! Combine transition-specific curves, as R's `survfit.matrix` does.
//!
//! Each transition contains one KM result per prediction column, with strata
//! stacked inside each result. Curves are scanned once over the union of event
//! times: no time-by-transition jump matrix or repeated curve summaries.

use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;

use super::{SurvfitAJResult, SurvfitKMResult};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::expm::survexpm;
use crate::internal::numpy_utils::{FloatMatrix, IntVec};
use crate::internal::validation::{validate_finite, validate_length};

/// How a cumulative-hazard increment updates the state probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvfitMatrixMethod {
    /// Multiply by `I + dA`, with R's cap on the diagonal departure probability.
    Discrete,
    /// Multiply by `exp(dA)`.
    MatrixExponential,
}

impl SurvfitMatrixMethod {
    pub fn parse(method: &str) -> SurvivalResult<Self> {
        match method {
            "discrete" => Ok(Self::Discrete),
            "matexp" => Ok(Self::MatrixExponential),
            _ => Err(SurvivalError::invalid_input(
                "method must be 'discrete' or 'matexp'",
            )),
        }
    }
}

/// A nonempty cell of the transition matrix. State indices are zero-based.
/// `curves` contains one result per prediction column (one for KM curves).
#[derive(Debug, Clone, Copy)]
pub struct SurvfitMatrixTransition<'a> {
    pub from: usize,
    pub to: usize,
    pub curves: &'a [SurvfitKMResult],
}

/// Join transition curves into a multistate curve. Strata vary fastest, then
/// prediction columns. `p0` has either one row, reused for every output curve,
/// or one row per output curve. The default puts everyone in the first state.
///
/// As in R, event times are unioned across transitions and times at or before
/// `start_time` are excluded. The default start is `min(0, event times)`.
/// Risk and event counts use R's right-continuous lookup, including its reuse
/// of a transition's previous event count at another transition's event time.
/// The last transition in column-major state order supplies each source's risk
/// count. No standard errors or observation-level influence are inferred.
///
/// The result also retains transition cumulative hazards. `n_censor` is zero:
/// the matrix method does not reconstruct censoring histories. `n_id` uses the
/// first transition's sample sizes to keep the shared curve representation;
/// these are not estimates of unique subjects across transitions.
pub fn survfit_matrix(
    transitions: &[SurvfitMatrixTransition<'_>],
    states: &[String],
    p0: Option<ArrayView2<'_, f64>>,
    method: SurvfitMatrixMethod,
    start_time: Option<f64>,
) -> SurvivalResult<SurvfitAJResult> {
    let invalid = SurvivalError::invalid_input;
    let ns = states.len();
    if ns < 2 || transitions.len() < 2 {
        return Err(invalid(
            "input must have at least 2 states and 2 transitions",
        ));
    }
    if states.iter().any(|s| s.is_empty())
        || states
            .iter()
            .enumerate()
            .any(|(i, s)| states[..i].contains(s))
    {
        return Err(invalid("state names must be nonempty and unique"));
    }
    if let Some(start) = start_time {
        validate_finite(&[start], "start_time")?;
    }
    // R traverses a matrix in column-major order, which matters for counts
    // when several transitions depart from the same state.
    let mut transitions = transitions.to_vec();
    transitions.sort_by_key(|t| (t.to, t.from));
    for (i, t) in transitions.iter().enumerate() {
        if t.from >= ns || t.to >= ns {
            return Err(invalid("transition state is out of bounds"));
        }
        if i > 0 && (t.from, t.to) == (transitions[i - 1].from, transitions[i - 1].to) {
            return Err(invalid("duplicate transition"));
        }
    }
    let columns = transitions[0].curves.len();
    if columns == 0 {
        return Err(invalid("each transition needs at least one curve"));
    }
    let first = &transitions[0].curves[0];
    let groups = first.n_curves();
    if groups == 0 {
        return Err(invalid("each transition needs at least one stratum"));
    }
    // Public Rust result fields can be edited, so check every field we index.
    for t in &transitions {
        if t.curves.len() != columns {
            return Err(invalid("all curves must be of the same dimension"));
        }
        for curve in t.curves {
            if curve.n_curves() != groups || curve.strata.is_some() != first.strata.is_some() {
                return Err(invalid("all curves must be of the same dimension"));
            }
            validate_length(groups, curve.n.len(), "curve sample sizes")?;
            for (name, values) in [
                ("n_risk", &curve.n_risk),
                ("n_event", &curve.n_event),
                ("cumhaz", &curve.cumhaz),
            ] {
                validate_length(curve.time.len(), values.len(), name)?;
                validate_finite(values, name)?;
                if values.iter().any(|&v| v < 0.0) {
                    return Err(invalid(&format!("{name} must be nonnegative")));
                }
            }
            validate_finite(&curve.time, "time")?;
            if let Some(sizes) = &curve.strata
                && sizes.iter().try_fold(0usize, |sum, n| sum.checked_add(*n))
                    != Some(curve.time.len())
            {
                return Err(invalid("strata sizes must sum to the number of times"));
            }
            for range in curve.curve_ranges() {
                if curve.time[range.clone()].windows(2).any(|w| w[0] >= w[1]) {
                    return Err(invalid("times must be strictly increasing within a curve"));
                }
                if curve.cumhaz[range].windows(2).any(|w| w[0] > w[1]) {
                    return Err(invalid("cumulative hazards must be nondecreasing"));
                }
            }
        }
    }
    let ncurves = columns * groups;
    let initial: Vec<Vec<f64>> = match p0 {
        Some(p0) => {
            if p0.ncols() != ns || ![1, ncurves].contains(&p0.nrows()) {
                return Err(invalid(
                    "p0 must have one row or one per curve, and one column per state",
                ));
            }
            for row in p0.rows() {
                if row.iter().any(|v| !v.is_finite() || *v < 0.0) || (row.sum() - 1.0).abs() > 1e-8
                {
                    return Err(invalid("invalid elements in p0"));
                }
            }
            (0..ncurves)
                .map(|i| p0.row(i % p0.nrows()).to_vec())
                .collect()
        }
        None => {
            let mut row = vec![0.0; ns];
            row[0] = 1.0;
            vec![row; ncurves]
        }
    };
    let t0 = start_time.unwrap_or_else(|| {
        transitions
            .iter()
            .flat_map(|t| t.curves)
            .flat_map(|c| {
                c.time
                    .iter()
                    .zip(&c.n_event)
                    .filter_map(|(&t, &d)| (d > 0.0).then_some(t))
            })
            .fold(0.0, f64::min)
    });
    let nt = transitions.len();
    let mut out = SurvfitAJResult {
        n: Vec::with_capacity(ncurves),
        time: Vec::new(),
        n_risk: Vec::new(),
        n_event: Vec::new(),
        n_censor: Vec::new(),
        n_enter: None,
        n_transition: Vec::new(),
        counts: None,
        pstate: Vec::new(),
        cumhaz: Vec::new(),
        std_err: None,
        std_chaz: None,
        std_auc: None,
        se0: None,
        lower: None,
        upper: None,
        p0: initial,
        strata: Some(Vec::with_capacity(ncurves)),
        strata_codes: None,
        n_id: Vec::with_capacity(ncurves),
        states: states.to_vec(),
        transitions: Vec::new(),
        hazard_from: transitions.iter().map(|t| t.from).collect(),
        hazard_to: transitions.iter().map(|t| t.to).collect(),
        logse: false,
        conf_int: 0.95,
        conf_type: "none".into(),
        type_: if first.type_ == "counting" {
            "mcounting"
        } else {
            "mright"
        }
        .into(),
        t0,
        start_time,
        influence_pstate: None,
    };
    for column in 0..columns {
        let ranges: Vec<_> = transitions
            .iter()
            .map(|t| t.curves[column].curve_ranges())
            .collect();
        for group in 0..groups {
            let mut times: Vec<_> = transitions
                .iter()
                .enumerate()
                .flat_map(|(k, t)| {
                    let c = &t.curves[column];
                    ranges[k][group]
                        .clone()
                        .filter_map(|i| (c.n_event[i] > 0.0).then_some(c.time[i]))
                })
                .collect();
            times.sort_by(f64::total_cmp);
            times.dedup();
            let curve_start =
                start_time.unwrap_or_else(|| times.first().copied().unwrap_or(0.0).min(0.0));
            let mut cursor: Vec<_> = ranges.iter().map(|r| r[group].start).collect();
            let mut previous = vec![0.0; nt];
            let mut prob = out.p0[column * groups + group].clone();
            let mut next = vec![0.0; ns];
            let mut departure = vec![0.0; ns];
            let mut matrix =
                (method == SurvfitMatrixMethod::MatrixExponential).then(|| Array2::zeros((ns, ns)));
            let before = out.time.len();
            for time in times {
                let mut risk = vec![0.0; ns];
                let mut events = vec![0.0; ns];
                let mut transition_events = vec![0.0; nt];
                let mut hazards = vec![0.0; nt];
                next.fill(0.0);
                departure.fill(0.0);
                if let Some(matrix) = &mut matrix {
                    matrix.fill(0.0);
                }
                for (k, t) in transitions.iter().enumerate() {
                    let c = &t.curves[column];
                    let range = &ranges[k][group];
                    risk[t.from] = 0.0;
                    while cursor[k] < range.end && c.time[cursor[k]] <= time {
                        cursor[k] += 1;
                    }
                    if cursor[k] > range.start {
                        let i = cursor[k] - 1;
                        risk[t.from] = c.n_risk[i];
                        events[t.to] += c.n_event[i];
                        transition_events[k] = c.n_event[i];
                        hazards[k] = c.cumhaz[i];
                    }
                    let jump = hazards[k] - previous[k];
                    previous[k] = hazards[k];
                    if t.from != t.to {
                        departure[t.from] += jump;
                        next[t.to] += prob[t.from] * jump;
                        if let Some(matrix) = &mut matrix {
                            matrix[(t.from, t.to)] = jump;
                        }
                    }
                }
                // Update the hazard cursors even for excluded times so that
                // the first retained jump does not include earlier hazards.
                if time <= curve_start {
                    continue;
                }
                match method {
                    SurvfitMatrixMethod::Discrete => {
                        for s in 0..ns {
                            next[s] += prob[s] * (1.0 - departure[s].min(1.0));
                        }
                    }
                    SurvfitMatrixMethod::MatrixExponential => {
                        let matrix = matrix.as_mut().expect("matrix exponential workspace");
                        for s in 0..ns {
                            matrix[(s, s)] = -departure[s];
                        }
                        let exponential = survexpm(matrix)?;
                        next.fill(0.0);
                        for i in 0..ns {
                            for j in 0..ns {
                                next[j] += prob[i] * exponential[(i, j)];
                            }
                        }
                    }
                }
                std::mem::swap(&mut prob, &mut next);
                out.time.push(time);
                out.pstate.push(prob.clone());
                out.n_risk.push(risk);
                out.n_event.push(events);
                out.n_transition.push(transition_events);
                out.cumhaz.push(hazards);
                out.n_censor.push(vec![0.0; ns]);
            }
            out.strata.as_mut().unwrap().push(out.time.len() - before);
            let n = transitions[0].curves[column].n[group];
            out.n.push(n);
            out.n_id.push(n);
        }
    }
    Ok(out)
}

#[pyfunction(name = "survfit_matrix")]
#[pyo3(signature = (curves, from_state, to_state, states, p0=None, method="discrete", start_time=None))]
#[allow(clippy::too_many_arguments)]
pub fn survfit_matrix_py(
    py: Python<'_>,
    curves: Vec<Vec<SurvfitKMResult>>,
    from_state: IntVec,
    to_state: IntVec,
    states: Vec<String>,
    p0: Option<FloatMatrix>,
    method: &str,
    start_time: Option<f64>,
) -> PyResult<SurvfitAJResult> {
    validate_length(curves.len(), from_state.len(), "from_state")?;
    validate_length(curves.len(), to_state.len(), "to_state")?;
    let method = SurvfitMatrixMethod::parse(method)?;
    let transitions: Vec<_> = curves
        .iter()
        .enumerate()
        .map(|(i, curves)| SurvfitMatrixTransition {
            from: from_state[i] as usize,
            to: to_state[i] as usize,
            curves,
        })
        .collect();
    Ok(py.detach(|| {
        survfit_matrix(
            &transitions,
            &states,
            p0.as_ref().map(|p| p.view()),
            method,
            start_time,
        )
    })?)
}

#[cfg(test)]
mod tests;
