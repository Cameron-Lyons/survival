use ndarray::prelude::*;
use approx::assert_abs_diff_eq;

#[derive(Debug, Clone)]
struct Observation {
    start: f64,
    stop: f64,
    event: f64,
    x: f64,
}

#[derive(Debug)]
struct ByhandResult {
    loglik: f64,
    u: f64,
    imat: f64,
    haz: Array1<f64>,
    xbar: Array1<f64>,
    mart: Array1<f64>,
    score: Array1<f64>,
    scho: Array1<f64>,
    surv: Array1<f64>,
    var: Array1<f64>,
}

fn byhand(beta: f64, newx: f64) -> ByhandResult {
    let r = beta.exp();
    
    let loglik = 4.0 * beta 
        - (r + 1.0).ln() 
        - (r + 2.0).ln() 
        - 3.0 * (3.0 * r + 2.0).ln() 
        - 2.0 * (3.0 * r + 1.0).ln();
    
    let u = 1.0 / (r + 1.0) 
        + 1.0 / (3.0 * r + 1.0) 
        + 4.0 / (3.0 * r + 2.0) 
        - (r / (r + 2.0) 
        + 3.0 * r / (3.0 * r + 2.0) 
        + 3.0 * r / (3.0 * r + 1.0));
    
    let imat = r / (r + 1.0).powi(2) 
        + 2.0 * r / (r + 2.0).powi(2) 
        + 6.0 * r / (3.0 * r + 2.0).powi(2) 
        + 3.0 * r / (3.0 * r + 1.0).powi(2) 
        + 3.0 * r / (3.0 * r + 1.0).powi(2) 
        + 12.0 * r / (3.0 * r + 2.0).powi(2);
    
    let haz = array![
        1.0 / (r + 1.0),
        1.0 / (r + 2.0),
        1.0 / (3.0 * r + 2.0),
        1.0 / (3.0 * r + 1.0),
        1.0 / (3.0 * r + 1.0),
        2.0 / (3.0 * r + 2.0)
    ];
    
    let xbar = array![
        r / (r + 1.0),
        r / (r + 2.0),
        3.0 * r / (3.0 * r + 2.0),
        3.0 * r / (3.0 * r + 1.0),
        3.0 * r / (3.0 * r + 1.0),
        3.0 * r / (3.0 * r + 2.0)
    ];
    
    let mut wtmat = Array2::zeros((10, 6));
    wtmat[[0, 0]] = 1.0;
    wtmat[[3, 0]] = 1.0;
    wtmat[[4, 0]] = 1.0;
    
    wtmat[[1, 1]] = 1.0;
    wtmat[[3, 1]] = 1.0;
    wtmat[[4, 1]] = 1.0;
    
    wtmat[[2, 2]] = 1.0;
    wtmat[[3, 2]] = 1.0;
    wtmat[[4, 2]] = 1.0;
    wtmat[[6, 2]] = 1.0;
    wtmat[[7, 2]] = 1.0;
    
    wtmat[[3, 3]] = 1.0;
    wtmat[[4, 3]] = 1.0;
    wtmat[[6, 3]] = 1.0;
    wtmat[[7, 3]] = 1.0;
    
    wtmat[[4, 4]] = 1.0;
    wtmat[[5, 4]] = 1.0;
    wtmat[[6, 4]] = 1.0;
    wtmat[[7, 4]] = 1.0;
    
    wtmat[[5, 5]] = 1.0;
    wtmat[[6, 5]] = 1.0;
    wtmat[[7, 5]] = 1.0;
    wtmat[[8, 5]] = 1.0;
    wtmat[[9, 5]] = 1.0;
    
    let diag = array![r, 1.0, 1.0, r, 1.0, r, r, r, 1.0, 1.0];
    let wtmat_scaled = &wtmat * &diag.slice(s![.., NewAxis]);
    
    let x = array![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let status = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    
    let hazmat = &wtmat_scaled * &haz.slice(s![NewAxis, ..]);
    let mut dM = -hazmat.clone();
    
    for i in 0..6 {
        dM[[i, i]] += 1.0;
    }
    dM[[6, 5]] += 1.0; 
    
    let mart = dM.sum_axis(Axis(1));
    
    let mut resid = Array2::zeros((10, 6));
    for ((i, j), &hij) in hazmat.indexed_iter() {
        resid[[i, j]] = (x[i] - xbar[j]) * hij;
    }
    
    let score = resid.sum_axis(Axis(1));
    let scho_full = resid.sum_axis(Axis(0));
    let mut scho = scho_full.to_vec();
    scho[5] /= 2.0;
    scho.insert(5, scho[5]);
    
    let cum_haz: Vec<f64> = haz.iter().scan(0.0, |acc, &h| { *acc += h; Some(*acc) }).collect();
    let surv = (-Array::from(cum_haz) * r * newx).mapv(|v| v.exp());
    
    let var_g: Vec<f64> = haz.iter().enumerate().scan(0.0, |acc, (i, &h)| {
        *acc += h.powi(2) / if i == 5 { 2.0 } else { 1.0 };
        Some(*acc)
    }).collect();
    
    let var_d: Vec<f64> = haz.iter().zip(xbar.iter()).scan(0.0, |acc, (&h, &xb)| {
        *acc += (xb - newx) * h;
        Some(*acc)
    }).collect();
    
    let varhaz = (&Array::from(var_g) + &(&Array::from(var_d).mapv(|d| d.powi(2)) / imat) * (2.0 * beta * newx).exp();
    
    ByhandResult {
        loglik,
        u,
        imat,
        haz,
        xbar,
        mart,
        score,
        scho: Array::from(scho),
        surv,
        var: varhaz,
    }
}

fn main() {
    let truth0 = byhand(0.0, 0.0);
    
    let expected_loglik = 4.0 * 0.0 
        - 1.0_f64.ln() 
        - 2.0_f64.ln() 
        - 3.0 * 2.0_f64.ln() 
        - 2.0 * 1.0_f64.ln();
    assert_abs_diff_eq!(truth0.loglik, expected_loglik, epsilon = 1e-10);
    
    let r = 1.0;
    let expected_imat = r/(r+1.0).powi(2) 
        + 2.0*r/(r+2.0).powi(2) 
        + 6.0*r/(3.0*r+2.0).powi(2) 
        + 3.0*r/(3.0*r+1.0).powi(2)*2.0 
        + 12.0*r/(3.0*r+2.0).powi(2);
    assert_abs_diff_eq!(truth0.imat, expected_imat, epsilon = 1e-10);
    
    let expected_mart = array![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ];
    for (actual, expected) in truth0.mart.iter().zip(expected_mart.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }
    
    println!("All tests passed!");
}
