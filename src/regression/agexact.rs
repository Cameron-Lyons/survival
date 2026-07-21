use std::convert::Infallible;

use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::cox_optimizer::{CoxFit, Method};

pub(crate) struct AgexactData {
    pub start: Vec<f64>,
    pub stop: Vec<f64>,
    pub event: Vec<i32>,
    pub covar: Vec<f64>,
    pub offset: Vec<f64>,
    pub strata: Vec<i32>,
    pub nocenter: Vec<i32>,
}

pub(crate) struct AgexactState {
    pub means: Vec<f64>,
    pub beta: Vec<f64>,
    pub u: Vec<f64>,
    pub imat: Vec<f64>,
    pub loglik: Vec<f64>,
    pub work: Vec<f64>,
    pub work2: Vec<i32>,
}

pub(crate) struct AgexactParams {
    pub maxiter: i32,
    pub nused: i32,
    pub nvar: i32,
    pub eps: f64,
    pub tol_chol: f64,
}

struct AgexactResult {
    maxiter: i32,
    covar: Vec<f64>,
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Vec<f64>,
    loglik: Vec<f64>,
    flag: i32,
    sctest: f64,
}

fn validate_agexact_inputs(
    data: &AgexactData,
    state: &AgexactState,
    params: &AgexactParams,
) -> PyResult<(usize, usize)> {
    if params.maxiter < 0 || params.nused < 0 || params.nvar < 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "maxiter, nused, and nvar must be non-negative",
        ));
    }

    let n = params.nused as usize;
    let p = params.nvar as usize;

    if data.start.len() != n
        || data.stop.len() != n
        || data.event.len() != n
        || data.offset.len() != n
        || data.strata.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "start, stop, event, offset, and strata must all have length nused",
        ));
    }

    let covar_len = p.checked_mul(n).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("nused * nvar is too large")
    })?;
    if data.covar.len() != covar_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covar must have length nused * nvar",
        ));
    }

    if state.means.len() != p
        || state.beta.len() != p
        || state.u.len() != p
        || data.nocenter.len() != p
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "means, beta, u, and nocenter must all have length nvar",
        ));
    }

    let imat_len = p.checked_mul(p).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("nvar * nvar is too large")
    })?;
    if state.imat.len() != imat_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "imat must have length nvar * nvar",
        ));
    }

    let extra_work = p.checked_mul(3).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("required work length overflows usize")
    })?;
    let min_work_len = imat_len
        .checked_add(extra_work)
        .and_then(|len| len.checked_add(n))
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("required work length overflows usize")
        })?;
    if state.work.len() < min_work_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "work must have length at least {}",
            min_work_len
        )));
    }

    let min_work2_len = n.checked_mul(2).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("required work2 length overflows usize")
    })?;
    if state.work2.len() < min_work2_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "work2 must have length at least {}",
            min_work2_len
        )));
    }

    Ok((n, p))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn agexact(
    maxiter: i32,
    nused: i32,
    nvar: i32,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    covar: Vec<f64>,
    offset: Vec<f64>,
    strata: Vec<i32>,
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Vec<f64>,
    loglik: Vec<f64>,
    work: Vec<f64>,
    work2: Vec<i32>,
    eps: f64,
    tol_chol: f64,
    nocenter: Vec<i32>,
) -> PyResult<Py<PyDict>> {
    let data = AgexactData {
        start,
        stop,
        event,
        covar,
        offset,
        strata,
        nocenter,
    };
    let state = AgexactState {
        means,
        beta,
        u,
        imat,
        loglik,
        work,
        work2,
    };
    let params = AgexactParams {
        maxiter,
        nused,
        nvar,
        eps,
        tol_chol,
    };
    validate_agexact_inputs(&data, &state, &params)?;
    agexact_impl(data, state, params)
}

fn take_infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

fn center_column_major(covar: &mut [f64], means: &mut [f64], nocenter: &[i32], n: usize) {
    for (variable, mean) in means.iter_mut().enumerate() {
        if nocenter[variable] == 0 || n == 0 {
            *mean = 0.0;
            continue;
        }

        let column = &mut covar[variable * n..(variable + 1) * n];
        *mean = column.iter().sum::<f64>() / n as f64;
        for value in column {
            *value -= *mean;
        }
    }
}

fn column_major_to_row_major(covar: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut row_major = Vec::with_capacity(n * p);
    for person in 0..n {
        for variable in 0..p {
            row_major.push(covar[variable * n + person]);
        }
    }
    row_major
}

fn flatten_matrix(matrix: &Array2<f64>) -> Vec<f64> {
    let mut flat = Vec::with_capacity(matrix.len());
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            flat.push(matrix[(row, column)]);
        }
    }
    flat
}

fn fit_agexact(data: AgexactData, state: AgexactState, params: AgexactParams) -> AgexactResult {
    let AgexactData {
        start,
        stop,
        event,
        mut covar,
        offset,
        mut strata,
        nocenter,
    } = data;
    let AgexactState {
        mut means,
        beta,
        u: _,
        imat: _,
        loglik: _loglik,
        work: _,
        work2: _,
    } = state;
    let AgexactParams {
        maxiter,
        nused,
        nvar,
        eps,
        tol_chol,
    } = params;
    let n = nused as usize;
    let p = nvar as usize;

    center_column_major(&mut covar, &mut means, &nocenter, n);
    let row_major = column_major_to_row_major(&covar, n, p);
    let covariates = Array2::from_shape_vec((n, p), row_major)
        .expect("validated exact Cox covariates have a valid matrix shape");

    // The original entry point treats the final row as a stratum boundary even
    // when the caller leaves the marker vector at its all-zero default.
    if let Some(last) = strata.last_mut() {
        *last = 1;
    }

    // Centering is performed above to preserve the compatibility output. The
    // shared optimizer therefore receives already-centered, unscaled columns.
    let mut fit = take_infallible(CoxFit::new_with_entry_times(
        Array1::from_vec(stop),
        Array1::from_vec(event),
        covariates,
        Some(Array1::from_vec(start)),
        Array1::from_vec(strata),
        Array1::from_vec(offset),
        Array1::ones(n),
        Method::Exact,
        maxiter as usize,
        eps,
        tol_chol,
        vec![false; p],
        beta,
    ));
    take_infallible(fit.fit_agexact_compatibility());
    let (beta, _optimizer_means, u, imat, loglik, sctest, optimizer_flag, iterations) =
        fit.results();
    let flag = if maxiter == 0 { 0 } else { optimizer_flag };

    AgexactResult {
        maxiter: iterations as i32,
        covar,
        means,
        beta,
        u,
        imat: flatten_matrix(&imat),
        loglik: loglik.to_vec(),
        flag,
        sctest,
    }
}

fn agexact_impl(
    data: AgexactData,
    state: AgexactState,
    params: AgexactParams,
) -> PyResult<Py<PyDict>> {
    let result = fit_agexact(data, state, params);
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("maxiter", result.maxiter)?;
        dict.set_item("covar", result.covar)?;
        dict.set_item("means", result.means)?;
        dict.set_item("beta", result.beta)?;
        dict.set_item("u", result.u)?;
        dict.set_item("imat", result.imat)?;
        dict.set_item("loglik", result.loglik)?;
        dict.set_item("flag", result.flag)?;
        dict.set_item("sctest", result.sctest)?;
        Ok(dict.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn sample_params(nused: i32, nvar: i32) -> AgexactParams {
        AgexactParams {
            maxiter: 1,
            nused,
            nvar,
            eps: 1e-9,
            tol_chol: 1e-9,
        }
    }

    fn workspace(n: usize, p: usize) -> AgexactState {
        AgexactState {
            means: vec![0.0; p],
            beta: vec![0.0; p],
            u: vec![0.0; p],
            imat: vec![0.0; p * p],
            loglik: vec![0.0; 2],
            work: vec![0.0; p * p + 3 * p + n],
            work2: vec![0; 2 * n],
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    fn tied_data(offset: Vec<f64>) -> AgexactData {
        AgexactData {
            start: vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0],
            stop: vec![2.0, 3.0, 3.0, 3.0, 4.0, 4.5, 5.0, 5.0, 5.0, 6.0],
            event: vec![1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            covar: vec![
                -0.8, 0.3, 1.1, -0.2, 0.7, 1.4, -1.0, 0.5, 1.0, -0.4, 0.2, 1.3, -0.5, 0.8, -1.1,
                0.4, 1.2, -0.7, 0.6, 1.0,
            ],
            offset,
            strata: vec![0; 10],
            nocenter: vec![1, 1],
        }
    }

    #[test]
    fn validate_agexact_rejects_short_workspace() {
        Python::initialize();

        let data = AgexactData {
            start: vec![0.0, 0.0],
            stop: vec![1.0, 2.0],
            event: vec![1, 0],
            covar: vec![1.0, 2.0],
            offset: vec![0.0, 0.0],
            strata: vec![0, 1],
            nocenter: vec![1],
        };
        let state = AgexactState {
            means: vec![0.0],
            beta: vec![0.0],
            u: vec![0.0],
            imat: vec![1.0],
            loglik: vec![0.0, 0.0],
            work: vec![0.0, 0.0, 0.0],
            work2: vec![0, 0, 0, 0],
        };
        let params = sample_params(2, 1);

        let err = validate_agexact_inputs(&data, &state, &params).unwrap_err();

        assert!(err.to_string().contains("work must have length at least"));
    }

    #[test]
    fn validate_agexact_rejects_length_mismatches() {
        Python::initialize();

        let data = AgexactData {
            start: vec![0.0],
            stop: vec![1.0, 2.0],
            event: vec![1, 0],
            covar: vec![1.0, 2.0],
            offset: vec![0.0, 0.0],
            strata: vec![0, 1],
            nocenter: vec![1],
        };
        let state = workspace(2, 1);
        let params = sample_params(2, 1);

        let err = validate_agexact_inputs(&data, &state, &params).unwrap_err();

        assert!(
            err.to_string()
                .contains("start, stop, event, offset, and strata must all have length nused")
        );
    }

    #[test]
    fn exact_counting_process_fit_matches_reference() {
        let n = 6;
        let data = AgexactData {
            start: vec![0.0; n],
            stop: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            event: vec![1, 0, 1, 1, 0, 1],
            covar: vec![0.2, 1.1, -0.4, 0.8, 1.5, -0.2],
            offset: vec![0.0; n],
            strata: vec![0; n],
            nocenter: vec![1],
        };
        let mut params = sample_params(n as i32, 1);
        params.maxiter = 20;

        let result = fit_agexact(data, workspace(n, 1), params);

        assert!((result.beta[0] - -0.716_230_334_1).abs() < 1e-9);
        assert!((result.loglik[0] - -4.276_666_119_0).abs() < 1e-9);
        assert!((result.loglik[1] - -3.923_517_065_7).abs() < 1e-9);
        assert!((result.sctest - 0.677_003_624_6).abs() < 1e-9);
        assert!((result.imat[0] - 0.809_942_243_8).abs() < 1e-9);
        assert_eq!(result.maxiter, 4);
        assert_eq!(result.flag, 1);
        assert_eq!(result.means, vec![0.5]);
        for (actual, expected) in result.covar.iter().zip([-0.3, 0.6, -0.9, 0.3, 1.0, -0.7]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn tied_exact_fit_matches_reference() {
        let n = 10;
        let mut params = sample_params(n as i32, 2);
        params.maxiter = 20;

        let result = fit_agexact(tied_data(vec![0.0; n]), workspace(n, 2), params);

        for (actual, expected) in result
            .beta
            .iter()
            .zip([-0.115_430_521_359_233_32, -0.213_265_368_013_505_2])
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result
            .loglik
            .iter()
            .zip([-8.679_312_040_892_672, -8.627_187_346_318_79])
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.imat.iter().zip([
            0.441_606_639_173_011_4,
            0.230_754_719_570_179_73,
            0.230_754_719_570_179_73,
            0.432_843_140_053_174_1,
        ]) {
            assert_close(*actual, expected, 1e-10);
        }
        assert_close(result.sctest, 0.105_617_248_301_556_97, 1e-10);
        assert_eq!(result.maxiter, 3);
        assert_eq!(result.flag, 2);
    }

    #[test]
    fn stratified_exact_fit_matches_reference() {
        let n = 12;
        let data = AgexactData {
            start: vec![0.0, 0.2, 0.5, 1.0, 1.2, 2.0, 0.0, 0.1, 0.4, 0.8, 1.5, 2.2],
            stop: vec![1.5, 2.5, 3.0, 3.0, 4.5, 5.5, 1.0, 2.0, 2.8, 3.8, 4.2, 5.0],
            event: vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            covar: vec![
                -0.7, 0.4, 1.2, -0.1, 0.8, 1.5, 0.5, -1.1, 0.9, 1.3, -0.4, 0.2, 1.0, -0.5, 0.3,
                1.4, -0.8, 0.6, -1.2, 0.7, 1.1, -0.2, 0.5, 1.6,
            ],
            offset: vec![0.0; n],
            strata: vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            nocenter: vec![1, 1],
        };
        let mut params = sample_params(n as i32, 2);
        params.maxiter = 20;

        let result = fit_agexact(data, workspace(n, 2), params);

        for (actual, expected) in result
            .beta
            .iter()
            .zip([-1.491_575_625_028_017_2, 0.651_994_589_955_765_1])
        {
            assert_close(*actual, expected, 1e-8);
        }
        for (actual, expected) in result
            .loglik
            .iter()
            .zip([-6.866_933_284_461_882, -4.782_313_934_490_155])
        {
            assert_close(*actual, expected, 1e-9);
        }
        for (actual, expected) in result.imat.iter().zip([
            1.110_235_034_638_227_7,
            -0.217_479_161_166_214_63,
            -0.217_479_161_166_214_63,
            0.811_063_057_618_197_3,
        ]) {
            assert_close(*actual, expected, 1e-8);
        }
        assert_eq!(result.maxiter, 5);
        assert_eq!(result.flag, 2);
    }

    #[test]
    fn zero_iteration_fit_preserves_initial_coefficients() {
        let n = 10;
        let offset = vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.25, 0.2, -0.05, 0.1];
        let mut state = workspace(n, 2);
        state.beta = vec![0.25, -0.15];
        let mut params = sample_params(n as i32, 2);
        params.maxiter = 0;

        let result = fit_agexact(tied_data(offset), state, params);

        assert_eq!(result.beta, vec![0.25, -0.15]);
        assert_eq!(result.maxiter, 0);
        assert_eq!(result.flag, 0);
        assert_eq!(result.loglik[0], result.loglik[1]);
        assert_close(result.loglik[0], -8.984_551_377_142_534, 1e-10);
        assert_close(result.sctest, 0.527_340_441_191_537_9, 1e-10);
        for (actual, expected) in result
            .u
            .iter()
            .zip([-1.198_072_004_593_938_6, 0.582_535_379_350_987_8])
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.imat.iter().zip([
            0.453_757_442_304_586_64,
            0.194_510_928_931_332_07,
            0.194_510_928_931_332_07,
            0.434_756_546_357_564_44,
        ]) {
            assert_close(*actual, expected, 1e-10);
        }
    }

    #[test]
    fn one_iteration_from_nonzero_initial_values_matches_reference() {
        let n = 10;
        let offset = vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.25, 0.2, -0.05, 0.1];
        let mut state = workspace(n, 2);
        state.beta = vec![0.25, -0.15];
        let mut params = sample_params(n as i32, 2);
        params.maxiter = 1;

        let result = fit_agexact(tied_data(offset), state, params);

        for (actual, expected) in result
            .beta
            .iter()
            .zip([-0.180_324_590_728_348, -0.129_777_028_882_461])
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result
            .loglik
            .iter()
            .zip([-8.984_551_377_142_53, -8.735_543_533_097_94])
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result.imat.iter().zip([
            0.445_743_377_908_897,
            0.234_376_633_135_133,
            0.234_376_633_135_133,
            0.452_933_208_616_838,
        ]) {
            assert_close(*actual, expected, 1e-10);
        }
        assert_eq!(result.maxiter, 1);
        assert_eq!(result.flag, 1_000);
    }

    #[test]
    fn delayed_entry_exact_fit_avoids_risk_sum_cancellation() {
        let n = 3;
        let data = AgexactData {
            start: vec![0.0, 0.0, 1.0],
            stop: vec![1.0, 2.0, 2.0],
            event: vec![1, 0, 0],
            covar: vec![0.0, 0.0, 100.0],
            offset: vec![0.0; n],
            strata: vec![0; n],
            nocenter: vec![1],
        };
        let mut state = workspace(n, 1);
        state.beta = vec![1.0];
        let mut params = sample_params(n as i32, 1);
        params.maxiter = 0;

        let result = fit_agexact(data, state, params);

        assert_close(result.loglik[0], -std::f64::consts::LN_2, 1e-12);
        assert_eq!(result.loglik[0], result.loglik[1]);
        assert_eq!(result.u, vec![0.0]);
        assert_eq!(result.imat, vec![0.0]);
        assert_eq!(result.sctest, 0.0);
        assert_eq!(result.flag, 0);
    }

    #[test]
    fn complete_tied_risk_set_has_zero_conditional_information() {
        let n = 3;
        let data = AgexactData {
            start: vec![0.0; n],
            stop: vec![1.0; n],
            event: vec![1; n],
            covar: (0..n)
                .map(|value| ((value as f64 - 0.37).powi(2)) / 7.0)
                .collect(),
            offset: vec![0.0; n],
            strata: vec![0; n],
            nocenter: vec![1],
        };
        let mut state = workspace(n, 1);
        state.beta = vec![0.7];
        let mut params = sample_params(n as i32, 1);
        params.maxiter = 0;

        let result = fit_agexact(data, state, params);

        assert_eq!(result.loglik, vec![0.0, 0.0]);
        assert_eq!(result.u, vec![0.0]);
        assert_eq!(result.imat, vec![0.0]);
        assert_eq!(result.sctest, 0.0);
        assert_eq!(result.flag, 0);
    }

    #[test]
    fn step_halving_uses_exact_compatibility_policy() {
        let n = 10;
        let data = AgexactData {
            start: vec![0.0, 0.0, 0.0, 2.0, 6.0, 5.0, 8.0, 1.0, 5.0, 10.0],
            stop: vec![1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 10.0, 10.0, 11.0, 11.0],
            event: vec![1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            covar: vec![
                -0.794_901_757_722_206_4,
                -0.971_412_993_595_188,
                -0.897_364_024_938_572_6,
                -0.311_732_919_901_538_56,
                -0.951_374_812_217_756_5,
                1.204_533_321_642_762,
                2.883_351_093_087_066,
                -0.885_194_161_791_435_7,
                0.982_157_001_943_561_7,
                0.251_309_986_747_342,
            ],
            offset: vec![
                0.321_143_620_716_472_3,
                -0.170_347_621_201_529_4,
                0.275_770_030_571_546_9,
                0.222_191_560_017_322_24,
                0.125_748_557_191_099_7,
                -0.185_935_466_774_804_63,
                0.381_606_917_637_173_1,
                -0.694_971_257_421_976,
                0.496_755_599_155_065,
                0.170_148_434_302_88,
            ],
            strata: vec![0; n],
            nocenter: vec![1],
        };
        let mut state = workspace(n, 1);
        state.beta = vec![5.853_544_164_651_22];
        let mut params = sample_params(n as i32, 1);
        params.maxiter = 40;

        let result = fit_agexact(data, state, params);

        assert_close(result.beta[0], 4.633_619_571_940_176, 1e-9);
        assert_close(result.loglik[0], -2.034_562_528_770_813, 1e-9);
        assert_close(result.loglik[1], -2.030_219_922_496_827_6, 1e-9);
        assert_close(result.imat[0], 161.077_101_202_556_04, 1e-7);
        assert_eq!(result.maxiter, 3);
        assert_eq!(result.flag, 1);
    }

    #[test]
    fn nonconverged_exact_fit_returns_final_trial_state() {
        let n = 7;
        let data = AgexactData {
            start: vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0],
            stop: vec![1.0, 1.0, 1.0, 3.0, 3.0, 4.0, 5.0],
            event: vec![1, 0, 0, 0, 0, 1, 0],
            covar: vec![
                -0.106_275_907_728_014_89,
                -1.415_108_390_302_514,
                -0.598_261_907_922_483_6,
                3.279_520_010_161_916,
                -1.334_405_338_827_207,
                2.496_179_020_159_636_3,
                0.189_703_669_111_627_2,
            ],
            offset: vec![
                1.488_417_295_240_173_7,
                -0.376_803_280_793_356_45,
                -0.310_856_512_288_097_7,
                -1.085_016_692_155_569_3,
                1.234_396_256_848_731,
                0.427_128_664_347_841_25,
                -0.159_616_115_770_802_56,
            ],
            strata: vec![0; n],
            nocenter: vec![1],
        };
        let mut state = workspace(n, 1);
        state.beta = vec![0.708_344_326_291_033_2];
        let mut params = sample_params(n as i32, 1);
        params.maxiter = 20;

        let result = fit_agexact(data, state, params);

        assert_close(result.beta[0], 36.550_680_230_661_33, 5e-7);
        assert_close(result.loglik[0], -0.266_892_412_518_194_4, 1e-12);
        assert_close(result.loglik[1], -2.564_007_672_845_036_7e-9, 1e-12);
        assert_close(result.imat[0], 1_611_299_433.054_453_4, 300.0);
        assert_eq!(result.maxiter, 20);
        assert_eq!(result.flag, 1_000);
    }

    #[test]
    fn large_tied_risk_set_uses_dynamic_programming() {
        let n = 64;
        let deaths = 32;
        let data = AgexactData {
            start: vec![0.0; n],
            stop: vec![1.0; n],
            event: (0..n).map(|person| i32::from(person < deaths)).collect(),
            covar: (0..n).map(|value| value as f64).collect(),
            offset: vec![0.0; n],
            strata: vec![0; n],
            nocenter: vec![0],
        };
        let mut params = sample_params(n as i32, 1);
        params.maxiter = 0;

        let result = fit_agexact(data, workspace(n, 1), params);

        assert_close(result.loglik[0], -42.052_280_570_411_12, 1e-10);
        assert_close(result.u[0], -512.0, 1e-10);
        assert_close(result.imat[0], 3.0 / 16_640.0, 1e-15);
        assert_close(result.sctest, 3_072.0 / 65.0, 1e-10);
        assert_eq!(result.maxiter, 0);
        assert_eq!(result.flag, 0);
    }
}
