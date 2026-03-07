use std::collections::{BTreeSet, HashMap};

#[derive(Debug)]
pub struct CoxHazResult {
    pub time: Vec<f64>,
    pub nrisk: Vec<Vec<f64>>,
    pub nevent: Vec<Vec<f64>>,
    pub haz: Vec<Vec<f64>>,
    pub cumhaz: Vec<Vec<f64>>,
    pub states: Vec<usize>,
    pub pstate: Vec<Vec<f64>>,
}

pub fn coxhaz(
    intervals: &[(f64, f64, usize)],
    ids: &[usize],
    risk: &[Vec<f64>],
    weights: Option<&[f64]>,
    expm: bool,
) -> Result<CoxHazResult, String> {
    let n = intervals.len();
    validate_inputs(n, ids, risk, weights)?;

    let weights = weights.unwrap_or(&vec![1.0; n]);
    let (grouped, mut istate) = process_groups(intervals, ids, weights, risk);

    let events = collect_events(intervals, ids);
    let transitions = extract_transitions(&events, &istate);
    let (dtime, time_index) = get_unique_event_times(&events);

    let (nevent, nrisk) = compute_nevent_nrisk(
        &events,
        &istate,
        &transitions,
        &dtime,
        &time_index,
        risk,
        weights,
    );

    let haz = compute_haz(&nevent, &nrisk);
    let cumhaz = compute_cumhaz(&haz);

    let states = collect_states(&transitions);
    let pstate = compute_pstate(&haz, &transitions, &states, expm, dtime.len());

    Ok(CoxHazResult {
        time: dtime,
        nrisk,
        nevent,
        haz,
        cumhaz,
        states,
        pstate,
    })
}

fn validate_inputs(
    n: usize,
    ids: &[usize],
    risk: &[Vec<f64>],
    weights: Option<&[f64]>,
) -> Result<(), String> {
    if ids.len() != n {
        return Err("id length must match intervals".into());
    }
    if let Some(wt) = weights {
        if wt.len() != n || wt.iter().any(|&w| w <= 0.0) {
            return Err("invalid weights".into());
        }
    }
    if risk.iter().any(|r| r.is_empty()) {
        return Err("risk matrix has empty rows".into());
    }
    let n_trans = risk[0].len();
    if risk.iter().any(|r| r.len() != n_trans) {
        return Err("risk matrix inconsistent columns".into());
    }
    Ok(())
}

fn process_groups<'a>(
    intervals: &'a [(f64, f64, usize)],
    ids: &'a [usize],
    weights: &'a [f64],
    risk: &'a [Vec<f64>],
) -> (
    HashMap<usize, Vec<(usize, &'a (f64, f64, usize), &'a f64, &'a Vec<f64>)>>,
    Vec<usize>,
) {
    let mut groups = HashMap::new();
    for (i, id) in ids.iter().enumerate() {
        groups
            .entry(*id)
            .or_insert(vec![])
            .push((i, &intervals[i], &weights[i], &risk[i]));
    }

    let mut istate = vec![0; intervals.len()];
    let mut current_state = 1;

    for (_, group) in groups.iter_mut() {
        group.sort_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap());
        current_state = 1;
        for (i, (idx, _, _, _)) in group.iter().enumerate() {
            istate[*idx] = current_state;
            if group[i].1 .2 != 0 {
                current_state = group[i].1 .2;
            }
        }
    }

    (groups, istate)
}

fn collect_events(
    intervals: &[(f64, f64, usize)],
    ids: &[usize],
) -> Vec<(usize, f64, usize, usize)> {
    intervals
        .iter()
        .enumerate()
        .filter(|(_, (_, _, event))| *event != 0)
        .map(|(i, (start, stop, event))| (i, *stop, *event, ids[i]))
        .collect()
}

fn extract_transitions(
    events: &[(usize, f64, usize, usize)],
    istate: &[usize],
) -> HashMap<(usize, usize), usize> {
    let mut transitions = HashMap::new();
    for (idx, _, event, _) in events {
        let from = istate[*idx];
        let to = *event;
        transitions.entry((from, to)).or_insert(transitions.len());
    }
    transitions
}

fn get_unique_event_times(
    events: &[(usize, f64, usize, usize)],
) -> (Vec<f64>, HashMap<f64, usize>) {
    let mut times: BTreeSet<f64> = events.iter().map(|(_, time, _, _)| *time).collect();
    let mut dtime: Vec<f64> = times.into_iter().collect();
    dtime.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let time_index: HashMap<_, _> = dtime.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    (dtime, time_index)
}

fn compute_nevent_nrisk(
    events: &[(usize, f64, usize, usize)],
    istate: &[usize],
    transitions: &HashMap<(usize, usize), usize>,
    dtime: &[f64],
    time_index: &HashMap<f64, usize>,
    risk: &[Vec<f64>],
    weights: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_time = dtime.len();
    let n_trans = transitions.len();
    let mut nevent = vec![vec![0.0; n_trans]; n_time];
    let mut nrisk = vec![vec![0.0; n_trans]; n_time];

    for (idx, time, event, id) in events {
        let from = istate[*idx];
        let to = *event;
        if let Some(&trans_idx) = transitions.get(&(from, to)) {
            let &t_idx = time_index.get(time).unwrap();
            nevent[t_idx][trans_idx] += weights[*idx] * risk[*idx][trans_idx];
        }
    }

    for (i, interval) in intervals.iter().enumerate() {
        let (start, stop, _) = *interval;
        let from = istate[i];
        for (trans, &trans_idx) in transitions {
            if trans.0 == from {
                for (t_idx, &time) in dtime.iter().enumerate() {
                    if start < time && time <= stop {
                        nrisk[t_idx][trans_idx] += risk[i][trans_idx] * weights[i];
                    }
                }
            }
        }
    }

    (nevent, nrisk)
}

fn compute_haz(nevent: &[Vec<f64>], nrisk: &[Vec<f64>]) -> Vec<Vec<f64>> {
    nevent
        .iter()
        .zip(nrisk.iter())
        .map(|(ne_row, nr_row)| {
            ne_row
                .iter()
                .zip(nr_row.iter())
                .map(|(&ne, &nr)| if nr == 0.0 { 0.0 } else { ne / nr })
                .collect()
        })
        .collect()
}

fn compute_cumhaz(haz: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut cumhaz = vec![vec![0.0; haz[0].len()]; haz.len()];
    for t in 0..haz[0].len() {
        let mut cum = 0.0;
        for (i, row) in haz.iter().enumerate() {
            cum += row[t];
            cumhaz[i][t] = cum;
        }
    }
    cumhaz
}

fn collect_states(transitions: &HashMap<(usize, usize), usize>) -> Vec<usize> {
    let mut states = BTreeSet::new();
    for &(from, to) in transitions.keys() {
        states.insert(from);
        states.insert(to);
    }
    states.into_iter().collect()
}

fn compute_pstate(
    haz: &[Vec<f64>],
    transitions: &HashMap<(usize, usize), usize>,
    states: &[usize],
    expm: bool,
    n_time: usize,
) -> Vec<Vec<f64>> {
    let n_states = states.len();
    let state_map: HashMap<_, _> = states.iter().enumerate().map(|(i, &s)| (s, i)).collect();

    let mut pstate = vec![vec![0.0; n_states]; n_time + 1];
    pstate[0][state_map[&1]] = 1.0; 

    for (t_idx, haz_row) in haz.iter().enumerate() {
        let mut tmat = vec![vec![0.0; n_states]; n_states];
        for ((from, to), &trans_idx) in transitions {
            let from_idx = state_map[from];
            let to_idx = state_map[to];
            tmat[from_idx][to_idx] += haz_row[trans_idx];
        }

        if expm {
            for i in 0..n_states {
                let sum: f64 = tmat[i].iter().sum();
                tmat[i][i] = -sum;
            }
            let exp_mat = matrix_exp(&tmat);
            pstate[t_idx + 1] = mat_vec_mult(&pstate[t_idx], &exp_mat);
        } else {
            for i in 0..n_states {
                let sum: f64 = tmat[i].iter().sum();
                tmat[i][i] = 1.0 - sum;
            }
            pstate[t_idx + 1] = mat_vec_mult(&pstate[t_idx], &tmat);
        }
    }

    pstate.into_iter().skip(1).collect()
}

fn mat_vec_mult(vec: &[f64], mat: &[Vec<f64>]) -> Vec<f64> {
    let n = mat.len();
    let mut res = vec![0.0; n];
    for (i, row) in mat.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            res[j] += vec[i] * val;
        }
    }
    res
}

fn matrix_exp(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut exp = vec![vec![0.0; n]; n];
    let mut term = vec![vec![0.0; n]; n];
    for i in 0..n {
        term[i][i] = 1.0;
        exp[i][i] = 1.0;
    }

    for k in 1..10 {
        term = mat_mult(&term, mat);
        term = mat_scale(&term, 1.0 / k as f64);
        exp = mat_add(&exp, &term);
    }
    exp
}

fn mat_mult(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut res = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    res
}

fn mat_scale(mat: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
    mat.iter()
        .map(|row| row.iter().map(|&v| v * scalar).collect())
        .collect()
}

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&a, &b)| a + b).collect())
        .collect()
}
