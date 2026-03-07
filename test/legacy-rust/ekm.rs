use std::collections::{BTreeSet, HashMap};
use approx::assert_abs_diff_eq;

#[derive(Debug, Clone)]
struct SurvivalData {
    id: u32,
    tstart: f64,
    time: f64,
    status: u32,
    trt: u32,
    wt: f32,
    pos: u32, 
}

#[derive(Debug)]
struct ByhandResult {
    n_id: usize,
    n: usize,
    time: Vec<f64>,
    n_enter: Vec<f32>,
    n_risk: Vec<f32>,
    n_event: Vec<f32>,
    n_censor: Vec<f32>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    u: Vec<Vec<f64>>,
    c: Vec<Vec<f64>>,
}

fn main() {
    let mut tdata = create_test_data();
    surv_split(&mut tdata, &[9.0, 17.0, 30.0]);
    assign_trt_and_weights(&mut tdata);
    force_tied_event(&mut tdata);

    let true_results = byhand(&tdata);

    assert_eq!(true_results[&1].n_id, 5); 
    assert_abs_diff_eq!(true_results[&1].surv[0], 0.8, epsilon = 0.001);
}

fn create_test_data() -> Vec<SurvivalData> {
    vec![
        SurvivalData { id: 1, tstart: 0.0, time: 9.0, status: 1, trt: 0, wt: 0.0, pos: 0 },
        SurvivalData { id: 2, tstart: 0.0, time: 13.0, status: 1, trt: 0, wt: 0.0, pos: 0 },
    ]
}

fn surv_split(data: &mut Vec<SurvivalData>, cuts: &[f64]) {
    let mut new_data = Vec::new();
    for d in data.drain(..) {
        let mut intervals = vec![];
        let mut current_start = d.tstart;
        let mut cuts = cuts.to_vec();
        cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        for &cut in &cuts {
            if cut < d.time {
                intervals.push(SurvivalData {
                    id: d.id,
                    tstart: current_start,
                    time: cut,
                    status: 0,
                    trt: d.trt,
                    wt: d.wt,
                    pos: 0,
                });
                current_start = cut;
            }
        }
        intervals.push(SurvivalData {
            id: d.id,
            tstart: current_start,
            time: d.time,
            status: d.status,
            trt: d.trt,
            wt: d.wt,
            pos: 0,
        });
        new_data.extend(intervals);
    }
    *data = new_data;
}

fn assign_trt_and_weights(data: &mut Vec<SurvivalData>) {
    let trt_cycle = vec![1, 1, 2, 2, 2];
    let wt_cycle = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    
    for (i, d) in data.iter_mut().enumerate() {
        d.trt = trt_cycle[i % trt_cycle.len()];
        d.wt = wt_cycle[i % wt_cycle.len()];
    }
}

fn force_tied_event(data: &mut Vec<SurvivalData>) {
    for d in data.iter_mut() {
        if d.time == 13.0 {
            d.status = 1;
        }
    }
}

fn byhand(data: &[SurvivalData]) -> HashMap<u32, ByhandResult> {
    let mut results = HashMap::new();
    let groups: Vec<u32> = data.iter().map(|d| d.trt).collect::<BTreeSet<_>>().into_iter().collect();

    for &grp in &groups {
        let group_data: Vec<&SurvivalData> = data.iter().filter(|d| d.trt == grp).collect();
        let mut unique_times = BTreeSet::new();
        for d in &group_data {
            unique_times.insert(d.tstart);
            unique_times.insert(d.time);
        }
        let time_points: Vec<f64> = unique_times.into_iter().collect();

        let mut result = ByhandResult {
            n_id: 0,
            n: group_data.len(),
            time: time_points.clone(),
            n_enter: vec![0.0; time_points.len()],
            n_risk: vec![0.0; time_points.len()],
            n_event: vec![0.0; time_points.len()],
            n_censor: vec![0.0; time_points.len()],
            surv: vec![1.0; time_points.len()],
            cumhaz: vec![0.0; time_points.len()],
            u: vec![vec![0.0; time_points.len()]; group_data.len()],
            c: vec![vec![0.0; time_points.len()]; group_data.len()],
        };

        for (j, &time_j) in time_points.iter().enumerate() {
            let mut n_risk = 0.0;
            let mut n_event = 0.0;
            let mut n_censor = 0.0;
            let mut n_enter = 0.0;

            for (i, d) in group_data.iter().enumerate() {
                if d.tstart < time_j && d.time >= time_j {
                    n_risk += d.wt;
                }
                if d.time == time_j {
                    if d.status == 1 {
                        n_event += d.wt;
                    } else if d.pos > 1 {
                        n_censor += d.wt;
                    }
                }
                if d.tstart == time_j && d.pos % 2 == 1 {
                    n_enter += d.wt;
                }
            }

            result.n_risk[j] = n_risk;
            result.n_event[j] = n_event;
            result.n_censor[j] = n_censor;
            result.n_enter[j] = n_enter;

            if n_risk > 0.0 {
                let hazard = n_event as f64 / n_risk as f64;
                result.surv[j] = if j > 0 {
                    result.surv[j-1] * (1.0 - hazard)
                } else {
                    1.0 - hazard
                };
                result.cumhaz[j] = if j > 0 {
                    result.cumhaz[j-1] + hazard
                } else {
                    hazard
                };
            }

            for (i, d) in group_data.iter().enumerate() {
            }
        }

        results.insert(grp, result);
    }

    results
}
