use nalgebra::{DMatrix, DVector, Dynamic, MatrixN, U5};
use std::collections::{BTreeSet, HashMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subjects = vec![Subject {
        id: 1,
        tstart: 0.0,
        tstop: 182.0,
        bili4: "1-2".to_string(),
        bstat: "censor".to_string(),
        age: 50.0,
    }];

    let coefficients = vec![0.05, 0.6, -1.0, 1.1];
    let x0 = 50.0;
    let initial_p0 = DVector::from_vec(vec![0.4, 0.3, 0.2, 0.1, 0.0]);

    let survival_result = calculate_survival(&subjects, &coefficients, x0, initial_p0)?;

    println!("Survival probabilities over time:");
    for (i, time) in survival_result.time.iter().enumerate() {
        println!("Time {}: {:?}", time, survival_result.pstate[i]);
    }

    Ok(())
}

struct SurvivalResult {
    time: Vec<f64>,
    pstate: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct Subject {
    id: u32,
    tstart: f64,
    tstop: f64,
    bili4: String,
    bstat: String,
    age: f64,
}

fn calculate_survival(
    data: &[Subject],
    coefficients: &[f64],
    x0: f64,
    initial_p0: DVector<f64>,
) -> Result<SurvivalResult, Box<dyn std::error::Error>> {
    let state_order = vec!["normal", "1-2", "2-4", ">4", "death"];
    let mut event_times: Vec<f64> = data
        .iter()
        .filter(|s| s.bstat != "censor")
        .map(|s| s.tstop)
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    event_times.dedup();

    let n_states = state_order.len();
    let mut pstate = initial_p0.clone();
    let mut result_pstate = vec![initial_p0.as_slice().to_vec()];

    for &time in &event_times {
        let mut hmat = DMatrix::zeros(n_states, n_states);

        let at_risk: Vec<&Subject> = data
            .iter()
            .filter(|s| s.tstart < time && s.tstop >= time)
            .collect();

        if !at_risk.is_empty() {
            let from_state = state_order.iter().position(|&s| s == "1-2").unwrap();
            let to_state = state_order.iter().position(|&s| s == "death").unwrap();

            let base_hazard = 0.1; 
            hmat[(from_state, to_state)] = base_hazard;
            hmat[(from_state, from_state)] = -base_hazard;
        }

        let tmat = hmat.exp();
        pstate = pstate * tmat;
        result_pstate.push(pstate.as_slice().to_vec());
    }

    Ok(SurvivalResult {
        time: event_times,
        pstate: result_pstate,
    })
}
