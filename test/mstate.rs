use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum State {
    Entry,
    A,
    B,
    C,
}

#[derive(Debug, Clone, Copy)]
enum Event {
    Transition(State),
    Censor,
}

#[derive(Debug, Clone)]
struct Interval {
    t1: f64,
    t2: f64,
    event: Event,
}

struct Subject {
    id: u32,
    intervals: Vec<Interval>,
    current_interval: usize,
    current_state: State,
    under_observation: bool,
}

impl Subject {
    fn new(id: u32, intervals: Vec<Interval>) -> Self {
        let mut sorted_intervals = intervals;
        sorted_intervals.sort_by(|a, b| a.t1.partial_cmp(&b.t1).unwrap());
        Self {
            id,
            intervals: sorted_intervals,
            current_interval: 0,
            current_state: State::Entry,
            under_observation: true,
        }
    }

    fn update_state(&mut self, t: f64) {
        while self.under_observation && self.current_interval < self.intervals.len() {
            let interval = &self.intervals[self.current_interval];
            if interval.t2 <= t {
                if let Event::Transition(to_state) = interval.event {
                    self.current_state = to_state;
                } else {
                    self.under_observation = false;
                }
                self.current_interval += 1;
            } else {
                break;
            }
        }
    }

    fn get_state_at(&mut self, t: f64) -> Option<State> {
        if !self.under_observation {
            return None;
        }
        self.update_state(t);
        if self.current_interval >= self.intervals.len() {
            return None;
        }
        let interval = &self.intervals[self.current_interval];
        if interval.t1 <= t && t < interval.t2 {
            Some(self.current_state)
        } else {
            None
        }
    }
}

struct MultiStateModel {
    subjects: Vec<Subject>,
    event_times: Vec<f64>,
    states: Vec<State>,
    pub n_risk: Vec<Vec<f64>>,
    pub pstate: Vec<Vec<f64>>,
    pub n_transition: Vec<Vec<Vec<f64>>>,
    pub times: Vec<f64>,
}

impl MultiStateModel {
    fn new(mut subjects: Vec<Subject>) -> Self {
        let mut event_times = BTreeSet::new();
        for subject in &subjects {
            for interval in &subject.intervals {
                event_times.insert(interval.t2);
            }
        }
        let event_times: Vec<f64> = event_times.into_iter().collect();
        Self {
            subjects,
            event_times: event_times.clone(),
            states: vec![State::Entry, State::A, State::B, State::C],
            n_risk: Vec::new(),
            pstate: Vec::new(),
            n_transition: Vec::new(),
            times: event_times,
        }
    }

    fn state_to_index(&self, state: State) -> usize {
        match state {
            State::Entry => 0,
            State::A => 1,
            State::B => 2,
            State::C => 3,
        }
    }

    fn compute(&mut self) {
        self.pstate.clear();
        self.n_risk.clear();
        self.n_transition.clear();

        self.pstate.push(vec![1.0, 0.0, 0.0, 0.0]);

        for &t in &self.event_times {
            let mut current_n_risk = vec![0.0; 4];
            let mut transitions = vec![vec![0.0; 4]; 4];
            let mut subjects_at_risk = Vec::new();

            for subject in &mut self.subjects {
                if let Some(state) = subject.get_state_at(t) {
                    let state_idx = self.state_to_index(state);
                    current_n_risk[state_idx] += 1.0;
                    subjects_at_risk.push((subject.id, state));
                }
            }

            for subject in &mut self.subjects {
                if subject.current_interval < subject.intervals.len() {
                    let interval = &subject.intervals[subject.current_interval];
                    if (interval.t2 - t).abs() < 1e-9 {
                        if let Event::Transition(to_state) = interval.event {
                            let from_state = subject.current_state;
                            let from_idx = self.state_to_index(from_state);
                            let to_idx = self.state_to_index(to_state);
                            transitions[from_idx][to_idx] += 1.0;
                        }
                    }
                }
            }

            let prev_p = self.pstate.last().unwrap().clone();
            let mut new_p = vec![0.0; 4];

            for from in 0..4 {
                let risk = current_n_risk[from];
                if risk == 0.0 {
                    new_p[from] += prev_p[from];
                    continue;
                }
                let total_transitions: f64 = transitions[from].iter().sum();
                if total_transitions == 0.0 {
                    new_p[from] += prev_p[from];
                    continue;
                }
                let remain = 1.0 - (total_transitions / risk);
                new_p[from] += prev_p[from] * remain;
                for to in 0..4 {
                    if transitions[from][to] > 0.0 {
                        new_p[to] += prev_p[from] * (transitions[from][to] / risk);
                    }
                }
            }

            self.pstate.push(new_p);
            self.n_risk.push(current_n_risk);
            self.n_transition.push(transitions);
        }

        self.times = self.event_times.clone();
    }
}

fn aeq(x: &[Vec<f64>], y: &[Vec<f64>], eps: f64) -> bool {
    if x.len() != y.len() {
        return false;
    }
    for (row_x, row_y) in x.iter().zip(y.iter()) {
        if row_x.len() != row_y.len() {
            return false;
        }
        for (a, b) in row_x.iter().zip(row_y.iter()) {
            if (a - b).abs() > eps {
                return false;
            }
        }
    }
    true
}

fn main() {
    let subjects = vec![
        Subject::new(1, vec![
            Interval { t1: 0.0, t2: 4.0, event: Event::Transition(State::A) },
            Interval { t1: 4.0, t2: 9.0, event: Event::Transition(State::B) },
            Interval { t1: 9.0, t2: 10.0, event: Event::Transition(State::A) },
        ]),
        Subject::new(2, vec![
            Interval { t1: 0.0, t2: 5.0, event: Event::Transition(State::B) },
        ]),
        Subject::new(3, vec![
            Interval { t1: 2.0, t2: 9.0, event: Event::Transition(State::C) },
        ]),
        Subject::new(4, vec![
            Interval { t1: 0.0, t2: 2.0, event: Event::Transition(State::A) },
            Interval { t1: 2.0, t2: 8.0, event: Event::Transition(State::C) },
            Interval { t1: 8.0, t2: 9.0, event: Event::Censor },
        ]),
        Subject::new(5, vec![
            Interval { t1: 1.0, t2: 3.0, event: Event::Transition(State::B) },
            Interval { t1: 3.0, t2: 11.0, event: Event::Censor },
        ]),
    ];

    let mut model = MultiStateModel::new(subjects);
    model.compute();

    let expected_n_risk = vec![
        vec![4.0, 0.0, 0.0, 0.0],  
        vec![4.0, 1.0, 0.0, 0.0],  
        vec![3.0, 1.0, 0.0, 0.0],  
        vec![2.0, 2.0, 0.0, 0.0],  
        vec![1.0, 2.0, 1.0, 0.0],  
        vec![0.0, 1.0, 2.0, 1.0],  
        vec![0.0, 0.0, 1.0, 0.0],  
        vec![0.0, 0.0, 0.0, 0.0],  
    ];

    let expected_pstate = vec![
        vec![24.0/32.0, 8.0/32.0, 0.0/32.0, 0.0/32.0],
        vec![18.0/32.0, 8.0/32.0, 6.0/32.0, 0.0/32.0],
        vec![12.0/32.0, 14.0/32.0, 6.0/32.0, 0.0/32.0],
        vec![6.0/32.0, 14.0/32.0, 12.0/32.0, 0.0/32.0],
        vec![6.0/32.0, 7.0/32.0, 12.0/32.0, 7.0/32.0],
        vec![0.0/32.0, 0.0/32.0, 19.0/32.0, 13.0/32.0],
        vec![0.0/32.0, 9.5/32.0, 9.5/32.0, 13.0/32.0],
        vec![0.0/32.0, 9.5/32.0, 9.5/32.0, 13.0/32.0],
    ];

    assert_eq!(model.n_risk.len(), expected_n_risk.len());
    for (i, (actual, expected)) in model.n_risk.iter().zip(expected_n_risk.iter()).enumerate() {
        assert!(
            aeq(&[actual.clone()], &[expected.clone()], 1e-6),
            "Mismatch in n_risk at index {}: {:?} vs {:?}",
            i, actual, expected
        );
    }

    assert_eq!(model.pstate.len() - 1, expected_pstate.len());
    for (i, (actual, expected)) in model.pstate[1..].iter().zip(expected_pstate.iter()).enumerate() {
        assert!(
            aeq(&[actual.clone()], &[expected.clone()], 1e-6),
            "Mismatch in pstate at index {}: {:?} vs {:?}",
            i, actual, expected
        );
    }

    println!("All tests passed!");
}
