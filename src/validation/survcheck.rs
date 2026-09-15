//! Consistency checks for multi-state survival data.
//!
//! Faithful port of `survcheck2` (R survival `R/survcheck.R`) and the C
//! helper `multicheck` (`src/multicheck.c`).  The formula/model-frame layer
//! of R's `survcheck` (response construction, `na.action`, row renumbering)
//! belongs to the caller; this module receives the already-built response.

use crate::data_prep::aeq_surv;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

/// The initial state of every row, given as codes into a level set (an R
/// factor).  Codes are 1-based like `as.integer(factor)`.
#[derive(Debug, Clone)]
pub struct SurvCheckIstate<'a> {
    pub codes: &'a [i32],
    pub levels: &'a [String],
}

/// Inputs of [`survcheck`]: a (multi-state) survival response plus the
/// subject identifier.  `time1` is `None` for right-censored data, in which
/// case every observation starts at time 0 (R's `ncol(y) == 2` branch).
#[derive(Debug, Clone)]
pub struct SurvCheckInput<'a> {
    pub id: &'a [i64],
    pub time1: Option<&'a [f64]>,
    pub time2: &'a [f64],
    /// `0` = censored, `k` = the `k`-th entry of `states` (1-based), i.e. the
    /// integer codes of R's multi-state `Surv` status column.
    pub status: &'a [i32],
    /// Names of the event states (`attr(y, "states")`).
    pub states: &'a [String],
    pub istate: Option<SurvCheckIstate<'a>>,
    /// Name of the initial state when `istate` is absent (R `istate0`).
    pub istate0: &'a str,
    /// Label of the censoring column of the transitions table (without the
    /// parentheses R adds).
    pub censor_label: &'a str,
    /// Apply R's `aeqSurv` near-tie rounding to the times first.
    pub timefix: bool,
}

/// Rows and subject ids of one kind of data problem (R's `overlap`, `gap`,
/// `jump` and `teleport` components).  Rows are 0-based indices into the
/// input; `id` lists the distinct subjects in order of first appearance.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvCheckProblem {
    pub row: Vec<usize>,
    pub id: Vec<i64>,
}

/// Counts of each problem type (R's `flag` vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvCheckFlags {
    pub overlap: usize,
    pub gap: usize,
    pub jump: usize,
    pub teleport: usize,
    pub duplicate: usize,
}

/// R's `transitions` table: `from` states by `to` states (plus the
/// censoring column), rows and columns that are entirely zero removed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvCheckTransitions {
    pub from_states: Vec<String>,
    pub to_states: Vec<String>,
    /// `counts[i][j]` = number of transitions from `from_states[i]` to
    /// `to_states[j]`.
    pub counts: Vec<Vec<usize>>,
}

/// R's `events` table: for each state (and `(any)` when there is more than
/// one event state) the number of subjects with `count[j]` visits.
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvCheckEvents {
    pub states: Vec<String>,
    /// The distinct visit counts labelling the columns.
    pub count: Vec<usize>,
    pub subjects: Vec<Vec<usize>>,
}

/// Result of [`survcheck`], mirroring R's `survcheck` object.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvCheckResult {
    /// All state names, initial states that are never destinations first.
    pub states: Vec<String>,
    pub transitions: SurvCheckTransitions,
    /// `None` when there are no events at all (R returns `NULL`).
    pub events: Option<SurvCheckEvents>,
    pub flag: SurvCheckFlags,
    /// The current state at the start of every observation, chained
    /// forward from the initial state (R's `istate` component).
    pub istate: Vec<String>,
    pub n_id: usize,
    pub n_observations: usize,
    pub n_transitions: usize,
    pub overlap: Option<SurvCheckProblem>,
    pub gap: Option<SurvCheckProblem>,
    pub jump: Option<SurvCheckProblem>,
    pub teleport: Option<SurvCheckProblem>,
}

/// Output of the `multicheck` C routine, one entry per input row.
struct MultiCheck {
    /// 2 when the row is the last observation of its subject, else 0
    /// (the C code adds 2 to the previous subject's final row).
    dupid: Vec<i32>,
    /// -1: starts before the prior interval ended, 1: after, 0: contiguous
    /// or first row of the subject.
    gap: Vec<i32>,
    /// Current state code (1-based into `states`).
    cstate: Vec<i32>,
}

/// Port of `src/multicheck.c`: walk the rows in `sort` order and chain the
/// state forward, ignoring censored rows as a state change.
fn multicheck(
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    id: &[usize],
    istate: &[i32],
    sort: &[usize],
) -> MultiCheck {
    let n = id.len();
    let mut dupid = vec![0; n];
    let mut gap = vec![0; n];
    let mut cstate = vec![0; n];
    let mut old_id: Option<usize> = None;
    let mut old_ii = 0usize;
    for (position, &ii) in sort.iter().enumerate() {
        if old_id == Some(id[ii]) {
            dupid[ii] = 0;
            gap[ii] = if time1[ii] == time2[old_ii] {
                0
            } else if time1[ii] > time2[old_ii] {
                1
            } else {
                -1
            };
            cstate[ii] = if status[old_ii] > 0 {
                status[old_ii]
            } else {
                cstate[old_ii]
            };
        } else {
            old_id = Some(id[ii]);
            dupid[ii] = 0;
            gap[ii] = 0;
            cstate[ii] = istate[ii];
            if position > 0 {
                dupid[old_ii] += 2;
            }
        }
        old_ii = ii;
    }
    if n > 0 {
        dupid[old_ii] += 2;
    }
    MultiCheck { dupid, gap, cstate }
}

/// R's `order(id, time2, time1)` (or `order(id, time1)` for right-censored
/// data, where `time1` is all zero): a stable sort, 0-based.
fn observation_order(id: &[i64], time1: &[f64], time2: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..id.len()).collect();
    order.sort_by(|&a, &b| {
        id[a]
            .cmp(&id[b])
            .then_with(|| time2[a].total_cmp(&time2[b]))
            .then_with(|| time1[a].total_cmp(&time1[b]))
    });
    order
}

/// Distinct ids in order of first appearance (R's `unique`).
fn unique_ids(id: &[i64]) -> Vec<i64> {
    let mut seen = HashMap::new();
    let mut unique = Vec::new();
    for &value in id {
        if seen.insert(value, ()).is_none() {
            unique.push(value);
        }
    }
    unique
}

fn problem(rows: Vec<usize>, id: &[i64]) -> Option<SurvCheckProblem> {
    if rows.is_empty() {
        return None;
    }
    let ids = unique_ids(&rows.iter().map(|&row| id[row]).collect::<Vec<_>>());
    Some(SurvCheckProblem { row: rows, id: ids })
}

/// The `events` table: number of subjects by number of visits to each state.
fn events_table(id: &[i64], status: &[i32], event_states: &[String]) -> Option<SurvCheckEvents> {
    let n_states = event_states.len();
    // table(id, factor(status, 0:nstate))[, -1]: rows in sorted id order.
    let mut per_subject: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (&subject, &state) in id.iter().zip(status) {
        let counts = per_subject
            .entry(subject)
            .or_insert_with(|| vec![0; n_states]);
        if state > 0 {
            counts[(state - 1) as usize] += 1;
        }
    }
    let tab1: Vec<Vec<usize>> = per_subject.into_values().collect();
    if tab1.iter().flatten().all(|&count| count == 0) {
        return None;
    }
    if n_states == 1 {
        // table(tab1): the distinct visit counts that occur, in order.
        let mut by_count: BTreeMap<usize, usize> = BTreeMap::new();
        for counts in &tab1 {
            *by_count.entry(counts[0]).or_insert(0) += 1;
        }
        return Some(SurvCheckEvents {
            states: vec![event_states[0].clone()],
            count: by_count.keys().copied().collect(),
            subjects: vec![by_count.values().copied().collect()],
        });
    }
    // Add the "(any)" column, tabulate every column over the union of the
    // observed counts, and drop states nobody ever visits.
    let with_any: Vec<Vec<usize>> = tab1
        .iter()
        .map(|counts| {
            let mut row = counts.clone();
            row.push(counts.iter().sum());
            row
        })
        .collect();
    let mut levels: Vec<usize> = with_any.iter().flatten().copied().collect();
    levels.sort_unstable();
    levels.dedup();
    let mut states = Vec::with_capacity(n_states + 1);
    let mut subjects = Vec::with_capacity(n_states + 1);
    for column in 0..=n_states {
        let row: Vec<usize> = levels
            .iter()
            .map(|&level| with_any.iter().filter(|r| r[column] == level).count())
            .collect();
        let visited = row
            .iter()
            .zip(&levels)
            .any(|(&subjects, &level)| level > 0 && subjects > 0);
        if !visited {
            continue;
        }
        states.push(if column == n_states {
            "(any)".to_string()
        } else {
            event_states[column].clone()
        });
        subjects.push(row);
    }
    Some(SurvCheckEvents {
        states,
        count: levels,
        subjects,
    })
}

/// The `transitions` table, compacted like R: from-states with neither
/// arrivals nor departures and all-zero columns are dropped.
fn transitions_table(
    states: &[String],
    censor_column: &str,
    from: &[i32],
    to: &[i32],
    keep: &[bool],
) -> SurvCheckTransitions {
    let n_states = states.len();
    let mut counts = vec![vec![0usize; n_states + 1]; n_states];
    for ((&from_code, &to_code), &keep) in from.iter().zip(to).zip(keep) {
        if !keep {
            continue;
        }
        let row = (from_code - 1) as usize;
        let column = if to_code == 0 {
            n_states
        } else {
            (to_code - 1) as usize
        };
        counts[row][column] += 1;
    }
    let keep_row: Vec<bool> = (0..n_states)
        .map(|row| {
            let out: usize = counts[row].iter().sum();
            let into: usize = counts.iter().map(|r| r[row]).sum();
            out + into > 0
        })
        .collect();
    let keep_column: Vec<bool> = (0..=n_states)
        .map(|column| counts.iter().map(|r| r[column]).sum::<usize>() > 0)
        .collect();
    let to: Vec<String> = (0..=n_states)
        .filter(|&column| keep_column[column])
        .map(|column| {
            if column == n_states {
                format!("({censor_column})")
            } else {
                states[column].clone()
            }
        })
        .collect();
    let from: Vec<String> = (0..n_states)
        .filter(|&row| keep_row[row])
        .map(|row| states[row].clone())
        .collect();
    let counts = (0..n_states)
        .filter(|&row| keep_row[row])
        .map(|row| {
            (0..=n_states)
                .filter(|&column| keep_column[column])
                .map(|column| counts[row][column])
                .collect()
        })
        .collect();
    SurvCheckTransitions {
        from_states: from,
        to_states: to,
        counts,
    }
}

fn validate(input: &SurvCheckInput<'_>) -> SurvivalResult<()> {
    let n = input.id.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, input.time2.len(), "time2")?;
    validate_length(n, input.status.len(), "status")?;
    if let Some(time1) = input.time1 {
        validate_length(n, time1.len(), "time1")?;
    }
    let n_states = input.states.len() as i32;
    if let Some((index, &value)) = input
        .status
        .iter()
        .enumerate()
        .find(|(_, value)| **value < 0 || **value > n_states)
    {
        return Err(SurvivalError::invalid_input(format!(
            "status[{index}] = {value} is not a code into the {n_states} states"
        )));
    }
    for (index, &value) in input.time2.iter().enumerate() {
        if !value.is_finite() {
            return Err(SurvivalError::invalid_input(format!(
                "time2[{index}] must be finite"
            )));
        }
    }
    if let Some(time1) = input.time1 {
        for (index, (&start, &stop)) in time1.iter().zip(input.time2).enumerate() {
            if !start.is_finite() {
                return Err(SurvivalError::invalid_input(format!(
                    "time1[{index}] must be finite"
                )));
            }
            if start >= stop {
                return Err(SurvivalError::invalid_input(format!(
                    "Stop time must be > start time (row {index})"
                )));
            }
        }
    }
    if let Some(istate) = &input.istate {
        if istate.codes.len() != n {
            return Err(SurvivalError::invalid_input("wrong length for istate"));
        }
        let n_levels = istate.levels.len() as i32;
        if let Some((index, &code)) = istate
            .codes
            .iter()
            .enumerate()
            .find(|(_, code)| **code < 1 || **code > n_levels)
        {
            return Err(SurvivalError::invalid_input(format!(
                "istate[{index}] = {code} is not a code into the {n_levels} istate levels"
            )));
        }
    }
    Ok(())
}

/// Apply `aeqSurv` to both time columns at once, as R does for a
/// counting-process response.
fn timefix_times(time1: Option<&[f64]>, time2: &[f64]) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
    let n = time2.len();
    let fixed = aeq_surv(time2, time1, None)?;
    Ok((fixed.time2.unwrap_or_else(|| vec![0.0; n]), fixed.time))
}

/// Check a multi-state (or plain) survival response for consistency, as
/// R's `survcheck` does: build the current-state vector, the transitions
/// and events tables, and flag overlapping, gapped, jumped and teleported
/// intervals.
pub fn survcheck(input: &SurvCheckInput<'_>) -> SurvivalResult<SurvCheckResult> {
    validate(input)?;
    let n = input.id.len();
    let (time1, time2) = if input.timefix {
        timefix_times(input.time1, input.time2)?
    } else {
        (
            input.time1.map_or_else(|| vec![0.0; n], <[f64]>::to_vec),
            input.time2.to_vec(),
        )
    };
    let counting = input.time1.is_some();
    if counting && let Some(index) = (0..n).find(|&i| time1[i] >= time2[i]) {
        return Err(SurvivalError::invalid_input(format!(
            "Stop time must be > start time after timefix (row {index})"
        )));
    }

    // The full state list: initial states that are not destinations first,
    // then the destination states in the user's order.
    let (istate_levels, istate_codes): (Vec<String>, Vec<i32>) = match &input.istate {
        Some(istate) => {
            // istate[, drop = TRUE]: only the levels actually used, in order.
            let used: Vec<usize> = (0..istate.levels.len())
                .filter(|&level| istate.codes.iter().any(|&code| code as usize == level + 1))
                .collect();
            let levels = used
                .iter()
                .map(|&level| istate.levels[level].clone())
                .collect();
            let codes = istate
                .codes
                .iter()
                .map(|&code| {
                    used.iter()
                        .position(|&level| level + 1 == code as usize)
                        .map_or(0, |position| position as i32 + 1)
                })
                .collect();
            (levels, codes)
        }
        None => (vec![input.istate0.to_string()], vec![1; n]),
    };
    let mut states: Vec<String> = istate_levels
        .iter()
        .filter(|level| !input.states.contains(level))
        .cloned()
        .collect();
    states.extend(input.states.iter().cloned());
    let state_index = |name: &str| {
        states
            .iter()
            .position(|s| s == name)
            .map_or(0, |position| position as i32 + 1)
    };
    // cstate2 <- factor(cstate, states)
    let cstate2: Vec<i32> = istate_codes
        .iter()
        .map(|&code| state_index(&istate_levels[(code - 1) as usize]))
        .collect();
    // stat2: status re-indexed into the full state list.
    let sindx: Vec<i32> = input.states.iter().map(|name| state_index(name)).collect();
    let stat2: Vec<i32> = input
        .status
        .iter()
        .map(|&code| {
            if code == 0 {
                0
            } else {
                sindx[(code - 1) as usize]
            }
        })
        .collect();

    let events = events_table(input.id, input.status, input.states);

    let unique = unique_ids(input.id);
    let id_code: HashMap<i64, usize> = unique
        .iter()
        .enumerate()
        .map(|(code, &id)| (id, code))
        .collect();
    let id2: Vec<usize> = input.id.iter().map(|id| id_code[id]).collect();
    let order = observation_order(input.id, &time1, &time2);
    let check = multicheck(&time1, &time2, &stat2, &id2, &cstate2, &order);

    // Without a user istate on (start, stop] data, the chained state is the
    // current state; for right-censored data every row starts at istate0.
    let cstate2 = if input.istate.is_none() && counting {
        check.cstate.clone()
    } else {
        cstate2
    };
    let keep: Vec<bool> = (0..n)
        .map(|i| stat2[i] != 0 || check.dupid[i] > 1)
        .collect();
    let transitions = transitions_table(&states, input.censor_label, &cstate2, &stat2, &keep);
    let censor_column = format!("({})", input.censor_label);
    let n_transitions = transitions
        .to_states
        .iter()
        .enumerate()
        .filter(|(_, name)| **name != censor_column)
        .map(|(column, _)| {
            transitions
                .counts
                .iter()
                .map(|row| row[column])
                .sum::<usize>()
        })
        .sum();

    let mismatch: Vec<bool> = (0..n).map(|i| cstate2[i] != check.cstate[i]).collect();
    let rows_where =
        |pred: &dyn Fn(usize) -> bool| -> Vec<usize> { (0..n).filter(|&i| pred(i)).collect() };
    let overlap_rows = rows_where(&|i| check.gap[i] < 0);
    let gap_rows = rows_where(&|i| check.gap[i] > 0 && !mismatch[i]);
    let jump_rows = rows_where(&|i| check.gap[i] > 0 && mismatch[i]);
    let teleport_rows = rows_where(&|i| check.gap[i] == 0 && mismatch[i]);
    let flag = SurvCheckFlags {
        overlap: overlap_rows.len(),
        gap: gap_rows.len(),
        jump: jump_rows.len(),
        teleport: teleport_rows.len(),
        duplicate: 0,
    };
    let istate = check
        .cstate
        .iter()
        .map(|&code| states[(code - 1) as usize].clone())
        .collect();

    Ok(SurvCheckResult {
        states,
        transitions,
        events,
        flag,
        istate,
        n_id: unique.len(),
        n_observations: n,
        n_transitions,
        overlap: problem(overlap_rows, input.id),
        gap: problem(gap_rows, input.id),
        jump: problem(jump_rows, input.id),
        teleport: problem(teleport_rows, input.id),
    })
}

/// Python entry point.  `istate` holds 1-based codes into `istate_levels`
/// (defaulting to the sorted distinct codes, as labels).
#[pyfunction(name = "survcheck")]
#[pyo3(signature = (id, time2, status, states, time1=None, istate=None, istate_levels=None, istate0="(s0)", censor_label="censored", timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn survcheck_py(
    id: Vec<i64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    states: Vec<String>,
    time1: Option<Vec<f64>>,
    istate: Option<Vec<i32>>,
    istate_levels: Option<Vec<String>>,
    istate0: &str,
    censor_label: &str,
    timefix: bool,
) -> PyResult<SurvCheckResult> {
    let default_levels = istate.as_ref().map(|codes| {
        let mut unique = codes.clone();
        unique.sort_unstable();
        unique.dedup();
        unique
            .iter()
            .map(|code| code.to_string())
            .collect::<Vec<_>>()
    });
    let istate_levels = istate_levels.or(default_levels);
    let istate = match (&istate, &istate_levels) {
        (Some(codes), Some(levels)) => Some(SurvCheckIstate { codes, levels }),
        _ => None,
    };
    let result = survcheck(&SurvCheckInput {
        id: &id,
        time1: time1.as_deref(),
        time2: &time2,
        status: &status,
        states: &states,
        istate,
        istate0,
        censor_label,
        timefix,
    })?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    fn check(
        id: &[i64],
        time1: Option<&[f64]>,
        time2: &[f64],
        status: &[i32],
        states: &[&str],
        istate: Option<(&[i32], &[&str])>,
    ) -> SurvCheckResult {
        let states = names(states);
        let levels = istate.map(|(_, levels)| names(levels));
        let istate = istate.map(|(codes, _)| SurvCheckIstate {
            codes,
            levels: levels.as_deref().unwrap(),
        });
        survcheck(&SurvCheckInput {
            id,
            time1,
            time2,
            status,
            states: &states,
            istate,
            istate0: "(s0)",
            censor_label: "censored",
            timefix: true,
        })
        .unwrap()
    }

    #[test]
    fn chains_states_forward_and_tabulates_transitions() {
        // subject 1: (0,10,b] (10,15,censor] (15,20,c] gives states a, b, b
        let result = check(
            &[1, 1, 1, 2],
            Some(&[0.0, 10.0, 15.0, 0.0]),
            &[10.0, 15.0, 20.0, 5.0],
            &[1, 0, 2, 0],
            &["b", "c"],
            Some((&[1, 2, 2, 1], &["a", "b"])),
        );
        assert_eq!(result.states, names(&["a", "b", "c"]));
        assert_eq!(result.istate, names(&["a", "b", "b", "a"]));
        assert_eq!(result.n_id, 2);
        assert_eq!(result.n_observations, 4);
        assert_eq!(result.n_transitions, 2);
        // a destination state keeps its (empty) row, like R
        assert_eq!(result.transitions.from_states, names(&["a", "b", "c"]));
        assert_eq!(
            result.transitions.to_states,
            names(&["b", "c", "(censored)"])
        );
        assert_eq!(
            result.transitions.counts,
            vec![vec![1, 0, 1], vec![0, 1, 0], vec![0, 0, 0]]
        );
        assert_eq!(
            result.flag,
            SurvCheckFlags {
                overlap: 0,
                gap: 0,
                jump: 0,
                teleport: 0,
                duplicate: 0
            }
        );
        let events = result.events.unwrap();
        assert_eq!(events.states, names(&["b", "c", "(any)"]));
        assert_eq!(events.count, vec![0, 1, 2]);
        assert_eq!(
            events.subjects,
            vec![vec![1, 1, 0], vec![1, 1, 0], vec![1, 0, 1]]
        );
    }

    #[test]
    fn flags_overlap_gap_jump_and_teleport() {
        let result = check(
            &[1, 1, 2, 2, 3, 3, 4, 4],
            Some(&[0.0, 4.0, 0.0, 6.0, 0.0, 6.0, 0.0, 5.0]),
            &[5.0, 8.0, 4.0, 9.0, 4.0, 9.0, 5.0, 9.0],
            &[1, 2, 1, 2, 0, 2, 0, 2],
            &["b", "c"],
            Some((&[1, 2, 1, 2, 1, 2, 1, 2], &["a", "b"])),
        );
        assert_eq!(result.flag.overlap, 1);
        assert_eq!(result.flag.gap, 1);
        assert_eq!(result.flag.jump, 1);
        assert_eq!(result.flag.teleport, 1);
        assert_eq!(
            result.overlap,
            Some(SurvCheckProblem {
                row: vec![1],
                id: vec![1]
            })
        );
        assert_eq!(
            result.gap,
            Some(SurvCheckProblem {
                row: vec![3],
                id: vec![2]
            })
        );
        assert_eq!(
            result.jump,
            Some(SurvCheckProblem {
                row: vec![5],
                id: vec![3]
            })
        );
        assert_eq!(
            result.teleport,
            Some(SurvCheckProblem {
                row: vec![7],
                id: vec![4]
            })
        );
    }

    #[test]
    fn right_censored_data_uses_istate0_and_a_single_event_state() {
        let result = check(
            &[1, 2, 3],
            None,
            &[1.0, 2.0, 3.0],
            &[1, 0, 1],
            &["event"],
            None,
        );
        assert_eq!(result.states, names(&["(s0)", "event"]));
        assert_eq!(result.transitions.from_states, names(&["(s0)", "event"]));
        assert_eq!(
            result.transitions.to_states,
            names(&["event", "(censored)"])
        );
        assert_eq!(result.transitions.counts, vec![vec![2, 1], vec![0, 0]]);
        let events = result.events.unwrap();
        assert_eq!(events.states, names(&["event"]));
        assert_eq!(events.count, vec![0, 1]);
        assert_eq!(events.subjects, vec![vec![1, 2]]);
        assert_eq!(result.n_transitions, 2);
    }

    #[test]
    fn no_events_gives_no_events_table() {
        let result = check(&[1, 2], None, &[1.0, 2.0], &[0, 0], &["event"], None);
        assert!(result.events.is_none());
        assert_eq!(result.n_transitions, 0);
        assert_eq!(result.transitions.to_states, names(&["(censored)"]));
    }

    #[test]
    fn rejects_invalid_intervals_and_codes() {
        let states = names(&["event"]);
        let bad = survcheck(&SurvCheckInput {
            id: &[1],
            time1: Some(&[2.0]),
            time2: &[1.0],
            status: &[1],
            states: &states,
            istate: None,
            istate0: "(s0)",
            censor_label: "censored",
            timefix: true,
        });
        assert!(bad.is_err());
        let bad_code = survcheck(&SurvCheckInput {
            id: &[1],
            time1: None,
            time2: &[1.0],
            status: &[2],
            states: &states,
            istate: None,
            istate0: "(s0)",
            censor_label: "censored",
            timefix: true,
        });
        assert!(bad_code.is_err());
    }
}
