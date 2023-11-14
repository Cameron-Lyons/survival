struct SurvivalData {
    start_time: f64,
    stop_time: f64,
    status: i32,
}

struct Covariate {
    values: Vec<f64>,
}

struct SurvivalModel {
    maxiter: usize,
    nused: usize,
    nvar: usize,
    yy: Vec<SurvivalData>,
    covar: Vec<Covariate>,
    strata: Vec<usize>,
    sort: Vec<(usize, usize)>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    eps: f64,
    tolerch: f64,
    method: u8,
    ptype: u8,
    nfrail: usize,
    frail: Vec<f64>,
    fbeta: Vec<f64>,
    pdiag: u8,
    // Returned parameters
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Vec<Vec<f64>>,
    loglik: f64,
    flag: i32,
    fdiag: Vec<f64>,
    jmat: Vec<Vec<f64>>,
    expect: Vec<f64>,
}
