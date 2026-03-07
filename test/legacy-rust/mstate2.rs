use ndarray::{Array, Array2, Array3, Axis, Ix2};
use std::collections::{BTreeSet, HashMap};

struct SurvivalData {
    id: Vec<usize>,
    t1: Vec<f64>,
    t2: Vec<f64>,
    state: Vec<usize>,
    istate: Vec<usize>,
    weight: Vec<f64>,
    states: Vec<usize>,
    nstate: usize,
}

struct SurvivalFit {
    time: Vec<f64>,
    pstate: Array2<f64>,
    influence: Array3<f64>,
    cumhaz: Array2<f64>,
}

impl SurvivalData {
    fn new(
        id: Vec<usize>,
        t1: Vec<f64>,
        t2: Vec<f64>,
        state: Vec<usize>,
        istate: Vec<usize>,
        weight: Vec<f64>,
    ) -> Self {
        let states = Self::survival_check(&id, &t1, &t2, &state, &istate);
        let nstate = states.len();
        
        SurvivalData {
            id, t1, t2, state, istate, weight,
            states, nstate,
        }
    }

    fn survival_check(
        id: &[usize],
        t1: &[f64],
        t2: &[f64],
        state: &[usize],
        istate: &[usize],
    ) -> Vec<usize> {
        vec![]
    }

    fn fit(&self) -> SurvivalFit {
        let mut fit = SurvivalFit::new(self.nstate);
        
        let p0 = self.initial_probabilities();
        fit.initialize(p0);
        
        let event_times = self.get_event_times();
        
        for (it, &time) in event_times.iter().enumerate() {
            if it == 0 { continue; } 
            
            let at_risk = self.get_at_risk(time);
            
            let events = self.get_events(time);
            
            self.update_matrices(&mut fit, it, &at_risk, &events);
        }
        
        fit
    }

}
