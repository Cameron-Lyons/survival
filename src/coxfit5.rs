use libc::{calloc, free, malloc};
use std::ffi::c_void;
use std::mem::{forget, size_of};
use std::ptr::null_mut;

pub struct CoxFit5 {
    covar: Vec<Vec<f64>>,
    cmat: Vec<Vec<f64>>,
    cmat2: Vec<Vec<f64>>,
    mark: Vec<f64>,
    wtave: Vec<f64>,
    a: Vec<f64>,
    oldbeta: Vec<f64>,
    a2: Vec<f64>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    status: Vec<i32>,
    sort: Vec<i32>,
    ttime: Vec<f64>,
    tmean: Vec<f64>,
    ptype: i32,
    pdiag: i32,
    ipen: Vec<f64>,
    upen: Vec<f64>,
    logpen: f64,
    zflag: Vec<i32>,
    frail: Option<Vec<i32>>,
    score: Vec<f64>,
}

impl CoxFit5 {
    pub fn new() -> Self {
        CoxFit5 {
            covar: Vec::new(),
            cmat: Vec::new(),
            cmat2: Vec::new(),
            mark: Vec::new(),
            wtave: Vec::new(),
            a: Vec::new(),
            oldbeta: Vec::new(),
            a2: Vec::new(),
            offset: Vec::new(),
            weights: Vec::new(),
            status: Vec::new(),
            sort: Vec::new(),
            ttime: Vec::new(),
            tmean: Vec::new(),
            ptype: 0,
            pdiag: 0,
            ipen: Vec::new(),
            upen: Vec::new(),
            logpen: 0.0,
            zflag: Vec::new(),
            frail: None,
            score: Vec::new(),
        }
    }

    pub fn coxfit5_a(&mut self, params: &CoxParams, data: &CoxData) -> CoxResult {
        let nused = data.nused;
        let nvar = data.nvar;
        let nf = params.nfrail;
        let nvar2 = nvar + nf;

        if nvar > 0 {
            self.covar = vec![vec![0.0; nused]; nvar];
            self.cmat = vec![vec![0.0; nvar + 1]; nvar2];
            self.cmat2 = vec![vec![0.0; nvar + 1]; nvar2];
        }

        self.a = vec![0.0; 4 * nvar2 + 6 * nused];

        CoxResult {
            means: vec![0.0; nvar],
            beta: vec![0.0; nvar],
            u: vec![0.0; nvar],
            imat: vec![vec![0.0; nvar]; nvar],
            loglik: 0.0,
            flag: 0,
            maxiter: 0,
            fbeta: vec![0.0; nf],
            fdiag: vec![0.0; nvar2],
            jmat: vec![vec![0.0; nvar2]; nvar2],
            expect: vec![0.0; nused],
        }
    }

    pub fn coxfit5_b(&mut self, params: &mut CoxParams, data: &CoxData) -> CoxResult {
        CoxResult {}
    }

    pub fn coxfit5_c(self, data: &CoxData) -> Vec<f64> {
        vec![0.0; data.nused]
    }
}

pub struct CoxParams {
    maxiter: i32,
    nused: usize,
    nvar: usize,
    strata: Vec<i32>,
    method: i32,
    ptype: i32,
    pdiag: i32,
    nfrail: usize,
    eps: f64,
    tolerch: f64,
}

pub struct CoxData {
    y: Vec<f64>,
    covar2: Vec<f64>,
    offset2: Vec<f64>,
    weights2: Vec<f64>,
    sorted: Vec<i32>,
    frail2: Vec<i32>,
    docenter: Vec<i32>,
}

pub struct CoxResult {
    pub means: Vec<f64>,
    pub beta: Vec<f64>,
    pub u: Vec<f64>,
    pub imat: Vec<Vec<f64>>,
    pub loglik: f64,
    pub flag: i32,
    pub maxiter: i32,
    pub fbeta: Vec<f64>,
    pub fdiag: Vec<f64>,
    pub jmat: Vec<Vec<f64>>,
    pub expect: Vec<f64>,
}

fn coxsafe(zbeta: f64) -> f64 {
    if zbeta > 20.0 {
        20.0
    } else if zbeta < -350.0 {
        -350.0
    } else {
        zbeta
    }
}
