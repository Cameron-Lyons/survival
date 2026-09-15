//! `agexact.fit`: the exact partial likelihood for (start, stop] data, R
//! survival's `R/agexact.fit.R` on top of `src/agexact.c`.
//!
//! The computation is the Newton-Raphson engine of `cox_optimizer` with
//! `TieMethod::Exact` and entry times; this module keeps the C routine's
//! calling convention (per-column centring, no case weights) as a typed
//! function and its Python binding.  `coxph_fit(method = "exact", entry =
//! ...)` reaches the same engine and adds the `coxph` post-processing.

use crate::constants::{COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::regression::cox_optimizer::{CoxFitBuilder, TieMethod};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

/// What `agexact.c` returns.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct AgexactFit {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// Column means subtracted before fitting (0 for `nocenter` columns).
    #[pyo3(get)]
    pub means: Vec<f64>,
    /// Score vector at the final coefficients.
    #[pyo3(get)]
    pub u: Vec<f64>,
    /// Inverse information matrix.
    #[pyo3(get)]
    pub var: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub loglik: [f64; 2],
    #[pyo3(get)]
    pub sctest: f64,
    /// Rank of the information matrix, `0` when no iterations were run,
    /// `1000` when the fit did not converge.
    #[pyo3(get)]
    pub flag: i32,
    #[pyo3(get)]
    pub iter: usize,
}

/// Inputs of [`agexact_fit`], in any row order.
#[derive(Debug, Clone)]
pub struct AgexactData {
    pub start: Vec<f64>,
    pub stop: Vec<f64>,
    pub event: Vec<i32>,
    pub x: Array2<f64>,
    pub offset: Option<Vec<f64>>,
    /// Stratum codes.
    pub strata: Option<Vec<i32>>,
}

/// Fitting controls of [`agexact_fit`] (`coxph.control` plus `init` and
/// `nocenter`).
#[derive(Debug, Clone)]
pub struct AgexactOptions {
    pub init: Option<Vec<f64>>,
    pub iter_max: usize,
    pub eps: f64,
    pub toler_chol: f64,
    /// Columns whose values all lie in this set are not centred
    /// (`agexact.fit`'s `nocenter`; `None` centres every column).
    pub nocenter: Option<Vec<f64>>,
}

impl Default for AgexactOptions {
    fn default() -> Self {
        Self {
            init: None,
            iter_max: COX_MAX_ITER,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler_chol: COX_RANK_TOLERANCE,
            nocenter: None,
        }
    }
}

/// Port of `agexact.fit` (without the residuals, which `CoxPHFit` provides).
pub fn agexact_fit(data: &AgexactData, options: &AgexactOptions) -> SurvivalResult<AgexactFit> {
    let n = data.stop.len();
    let nvar = data.x.ncols();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, data.start.len(), "start")?;
    validate_length(n, data.event.len(), "event")?;
    validate_length(n, data.x.nrows(), "x")?;
    validate_finite(&data.start, "start")?;
    validate_finite(&data.stop, "stop")?;
    validate_binary_i32(&data.event, "event")?;
    if let Some(index) = (0..n).find(|&i| data.start[i] >= data.stop[i]) {
        return Err(SurvivalError::invalid_input(format!(
            "Stop time must be > start time (row {index})"
        )));
    }
    if let Some(value) = data.x.iter().find(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "x contains non-finite value {value}"
        )));
    }
    if let Some(offset) = &data.offset {
        validate_length(n, offset.len(), "offset")?;
        validate_finite(offset, "offset")?;
    }
    if let Some(strata) = &data.strata {
        validate_length(n, strata.len(), "strata")?;
    }
    if let Some(init) = &options.init {
        validate_finite(init, "init")?;
    }
    let doscale: Vec<bool> = (0..nvar)
        .map(|col| {
            !options.nocenter.as_ref().is_some_and(|values| {
                data.x
                    .column(col)
                    .iter()
                    .all(|value| values.contains(value))
            })
        })
        .collect();
    let mut builder = CoxFitBuilder::new(
        Array1::from_vec(data.stop.clone()),
        Array1::from_vec(data.event.clone()),
        data.x.clone(),
    )
    .entry_times(Array1::from_vec(data.start.clone()))
    .method(TieMethod::Exact)
    .max_iter(options.iter_max)
    .eps(options.eps)
    .toler(options.toler_chol)
    .doscale(doscale)
    .initial_beta(options.init.clone().unwrap_or_else(|| vec![0.0; nvar]));
    if let Some(offset) = &data.offset {
        builder = builder.offset(Array1::from_vec(offset.clone()));
    }
    if let Some(strata) = &data.strata {
        builder = builder.strata(Array1::from_vec(strata.clone()));
    }
    let mut engine = builder.build()?;
    engine.fit();
    let results = engine.results();
    Ok(AgexactFit {
        coefficients: results.coefficients,
        means: results.means,
        u: results.score,
        var: results.var.outer_iter().map(|row| row.to_vec()).collect(),
        loglik: results.loglik,
        sctest: results.sctest,
        flag: results.flag,
        iter: results.iter,
    })
}

/// `agexact.fit(x, Surv(start, stop, event), strata, offset, init, control)`.
#[pyfunction(name = "agexact")]
#[pyo3(signature = (start, stop, event, x, offset=None, strata=None, init=None, iter_max=None, eps=None, toler_chol=None, nocenter=None))]
#[allow(clippy::too_many_arguments)]
pub fn agexact_py(
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    x: Vec<Vec<f64>>,
    offset: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
) -> PyResult<AgexactFit> {
    let ncols = x.first().map_or(0, Vec::len);
    if x.len() != stop.len() || x.iter().any(|row| row.len() != ncols) {
        return Err(SurvivalError::invalid_input(
            "x must have one rectangular row per observation",
        )
        .into());
    }
    let x = Array2::from_shape_vec((x.len(), ncols), x.into_iter().flatten().collect())
        .map_err(|err| SurvivalError::invalid_input(err.to_string()))?;
    let defaults = AgexactOptions::default();
    let options = AgexactOptions {
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter,
    };
    Ok(agexact_fit(
        &AgexactData {
            start,
            stop,
            event,
            x,
            offset,
            strata,
        },
        &options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    fn matrix(rows: usize, values: Vec<f64>) -> Array2<f64> {
        let cols = values.len() / rows;
        Array2::from_shape_vec((rows, cols), values).unwrap()
    }

    fn assert_var_close(actual: &[Vec<f64>], expected: &[[f64; 2]; 2], tolerance: f64) {
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                assert_close(*actual, *expected, tolerance);
            }
        }
    }

    fn tied_data(offset: Vec<f64>) -> AgexactData {
        AgexactData {
            start: vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0],
            stop: vec![2.0, 3.0, 3.0, 3.0, 4.0, 4.5, 5.0, 5.0, 5.0, 6.0],
            event: vec![1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            x: matrix(
                10,
                vec![
                    -0.8, 0.3, 1.1, -0.2, 0.7, 1.4, -1.0, 0.5, 1.0, -0.4, 0.2, 1.3, -0.5, 0.8,
                    -1.1, 0.4, 1.2, -0.7, 0.6, 1.0,
                ],
            ),
            offset: Some(offset),
            strata: None,
        }
    }

    fn options(iter_max: usize, init: Option<Vec<f64>>) -> AgexactOptions {
        AgexactOptions {
            init,
            iter_max,
            eps: 1e-9,
            toler_chol: 1e-9,
            nocenter: None,
        }
    }

    #[test]
    fn rejects_length_mismatches() {
        let data = AgexactData {
            start: vec![0.0],
            stop: vec![1.0, 2.0],
            event: vec![1, 0],
            x: matrix(2, vec![1.0, 2.0]),
            offset: None,
            strata: None,
        };
        assert!(agexact_fit(&data, &AgexactOptions::default()).is_err());
    }

    #[test]
    fn exact_counting_process_fit_matches_reference() {
        let data = AgexactData {
            start: vec![0.0; 6],
            stop: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            event: vec![1, 0, 1, 1, 0, 1],
            x: matrix(6, vec![0.2, 1.1, -0.4, 0.8, 1.5, -0.2]),
            offset: None,
            strata: None,
        };
        let result = agexact_fit(&data, &options(20, None)).unwrap();
        assert_close(result.coefficients[0], -0.716_230_334_1, 1e-9);
        assert_close(result.loglik[0], -4.276_666_119, 1e-9);
        assert_close(result.loglik[1], -3.923_517_065_7, 1e-9);
        assert_close(result.sctest, 0.677_003_624_6, 1e-9);
        assert_close(result.var[0][0], 0.809_942_243_8, 1e-9);
        assert_eq!(result.iter, 4);
        assert_eq!(result.flag, 1);
        assert_eq!(result.means, vec![0.5]);
    }

    #[test]
    fn tied_exact_fit_matches_reference() {
        // R: coxph(Surv(start, stop, event) ~ x, ties = "exact",
        //          control = coxph.control(eps = 1e-9, toler.chol = 1e-9))
        let result = agexact_fit(&tied_data(vec![0.0; 10]), &options(20, None)).unwrap();
        for (actual, expected) in result
            .coefficients
            .iter()
            .zip([-0.177_472_815_838_517, 0.595_407_041_347_299_2])
        {
            assert_close(*actual, expected, 1e-9);
        }
        for (actual, expected) in result
            .loglik
            .iter()
            .zip([-8.679_312_040_892_672, -8.205_532_976_098_844])
        {
            assert_close(*actual, expected, 1e-10);
        }
        let expected_var = [[0.324_676_83, 0.062_176_43], [0.062_176_43, 0.577_180_73]];
        assert_var_close(&result.var, &expected_var, 1e-7);
        assert_close(result.sctest, 0.859_122_386_159_976_7, 1e-10);
        assert_eq!(result.iter, 4);
        assert_eq!(result.flag, 2);
    }

    #[test]
    fn stratified_exact_fit_matches_reference() {
        let data = AgexactData {
            start: vec![0.0, 0.2, 0.5, 1.0, 1.2, 2.0, 0.0, 0.1, 0.4, 0.8, 1.5, 2.2],
            stop: vec![1.5, 2.5, 3.0, 3.0, 4.5, 5.5, 1.0, 2.0, 2.8, 3.8, 4.2, 5.0],
            event: vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            x: matrix(
                12,
                vec![
                    -0.7, 0.4, 1.2, -0.1, 0.8, 1.5, 0.5, -1.1, 0.9, 1.3, -0.4, 0.2, 1.0, -0.5, 0.3,
                    1.4, -0.8, 0.6, -1.2, 0.7, 1.1, -0.2, 0.5, 1.6,
                ],
            ),
            offset: None,
            strata: Some(vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        };
        // R: coxph(Surv(start, stop, event) ~ x + strata(s), ties = "exact")
        let result = agexact_fit(&data, &options(20, None)).unwrap();
        for (actual, expected) in result
            .coefficients
            .iter()
            .zip([-0.323_521_723_795_500_04, -0.218_609_476_153_718_56])
        {
            assert_close(*actual, expected, 1e-9);
        }
        for (actual, expected) in result
            .loglik
            .iter()
            .zip([-6.866_933_284_461_882, -6.658_585_236_999_502])
        {
            assert_close(*actual, expected, 1e-10);
        }
        let expected_var = [
            [0.393_679_287_349_937_1, 0.025_162_371_815_660_22],
            [0.025_162_371_815_660_22, 0.278_430_177_762_925_3],
        ];
        assert_var_close(&result.var, &expected_var, 1e-9);
        assert_close(result.sctest, 0.414_867_096_227_653_66, 1e-10);
        assert_eq!(
            result.means,
            vec![0.266_666_666_666_666_66, 0.483_333_333_333_333_4]
        );
        assert_eq!(result.iter, 3);
        assert_eq!(result.flag, 2);
    }

    #[test]
    fn zero_iteration_fit_preserves_initial_coefficients() {
        let offset = vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.25, 0.2, -0.05, 0.1];
        let result = agexact_fit(&tied_data(offset), &options(0, Some(vec![0.25, -0.15]))).unwrap();
        assert_eq!(result.coefficients, vec![0.25, -0.15]);
        assert_eq!(result.iter, 0);
        assert_eq!(result.flag, 0);
        // R: coxph(... ~ x + offset(off), ties = "exact", init = c(0.25, -0.15), iter.max = 0)
        assert_eq!(result.loglik[0], result.loglik[1]);
        assert_close(result.loglik[0], -9.337_812_385_162_364, 1e-10);
        assert_close(result.sctest, 1.695_251_548_863_118_8, 1e-10);
        let expected_var = [
            [0.327_584_856_845_065_5, 0.169_053_740_439_498_47],
            [0.169_053_740_439_498_47, 0.480_080_478_066_216_6],
        ];
        assert_var_close(&result.var, &expected_var, 1e-10);
    }

    #[test]
    fn one_iteration_from_nonzero_initial_values_matches_reference() {
        let offset = vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.25, 0.2, -0.05, 0.1];
        let result = agexact_fit(&tied_data(offset), &options(1, Some(vec![0.25, -0.15]))).unwrap();
        // R: as above with iter.max = 1
        for (actual, expected) in result
            .coefficients
            .iter()
            .zip([-0.053_520_938_263_236_784, 0.438_672_769_734_091_1])
        {
            assert_close(*actual, expected, 1e-10);
        }
        for (actual, expected) in result
            .loglik
            .iter()
            .zip([-9.337_812_385_162_364, -8.424_552_914_296_775])
        {
            assert_close(*actual, expected, 1e-10);
        }
        let expected_var = [
            [0.312_533_319_083_736_8, 0.080_356_614_895_959_37],
            [0.080_356_614_895_959_37, 0.531_495_187_396_132_9],
        ];
        assert_var_close(&result.var, &expected_var, 1e-10);
        assert_eq!(result.iter, 1);
        assert_eq!(result.flag, 1_000);
    }

    #[test]
    fn delayed_entry_exact_fit_avoids_risk_sum_cancellation() {
        let data = AgexactData {
            start: vec![0.0, 0.0, 1.0],
            stop: vec![1.0, 2.0, 2.0],
            event: vec![1, 0, 0],
            x: matrix(3, vec![0.0, 0.0, 100.0]),
            offset: None,
            strata: None,
        };
        let result = agexact_fit(&data, &options(0, Some(vec![1.0]))).unwrap();
        assert_close(result.loglik[0], -std::f64::consts::LN_2, 1e-12);
        assert_eq!(result.loglik[0], result.loglik[1]);
        assert_eq!(result.u, vec![0.0]);
        assert_eq!(result.var, vec![vec![0.0]]);
        assert_eq!(result.sctest, 0.0);
        assert_eq!(result.flag, 0);
    }

    #[test]
    fn complete_tied_risk_set_has_zero_conditional_information() {
        let data = AgexactData {
            start: vec![0.0; 3],
            stop: vec![1.0; 3],
            event: vec![1; 3],
            x: matrix(
                3,
                (0..3).map(|v| ((v as f64 - 0.37).powi(2)) / 7.0).collect(),
            ),
            offset: None,
            strata: None,
        };
        let result = agexact_fit(&data, &options(0, Some(vec![0.7]))).unwrap();
        assert_eq!(result.loglik, [0.0, 0.0]);
        assert_eq!(result.u, vec![0.0]);
        assert_eq!(result.var, vec![vec![0.0]]);
        assert_eq!(result.sctest, 0.0);
        assert_eq!(result.flag, 0);
    }

    #[test]
    fn step_halving_uses_the_exact_policy() {
        let data = AgexactData {
            start: vec![0.0, 0.0, 0.0, 2.0, 6.0, 5.0, 8.0, 1.0, 5.0, 10.0],
            stop: vec![1.0, 2.0, 2.0, 6.0, 7.0, 7.0, 10.0, 10.0, 11.0, 11.0],
            event: vec![1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            x: matrix(
                10,
                vec![
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
            ),
            offset: Some(vec![
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
            ]),
            strata: None,
        };
        let result = agexact_fit(&data, &options(40, Some(vec![5.853_544_164_651_22]))).unwrap();
        assert_close(result.coefficients[0], 4.633_619_571_940_176, 1e-9);
        assert_close(result.loglik[0], -2.034_562_528_770_813, 1e-9);
        assert_close(result.loglik[1], -2.030_219_922_496_827_6, 1e-9);
        assert_close(result.var[0][0], 161.077_101_202_556_04, 1e-7);
        assert_eq!(result.iter, 3);
        assert_eq!(result.flag, 1);
    }

    #[test]
    fn nonconverged_exact_fit_returns_final_trial_state() {
        let data = AgexactData {
            start: vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0],
            stop: vec![1.0, 1.0, 1.0, 3.0, 3.0, 4.0, 5.0],
            event: vec![1, 0, 0, 0, 0, 1, 0],
            x: matrix(
                7,
                vec![
                    -0.106_275_907_728_014_89,
                    -1.415_108_390_302_514,
                    -0.598_261_907_922_483_6,
                    3.279_520_010_161_916,
                    -1.334_405_338_827_207,
                    2.496_179_020_159_636_3,
                    0.189_703_669_111_627_2,
                ],
            ),
            offset: Some(vec![
                1.488_417_295_240_173_7,
                -0.376_803_280_793_356_45,
                -0.310_856_512_288_097_7,
                -1.085_016_692_155_569_3,
                1.234_396_256_848_731,
                0.427_128_664_347_841_25,
                -0.159_616_115_770_802_56,
            ]),
            strata: None,
        };
        let result = agexact_fit(&data, &options(20, Some(vec![0.708_344_326_291_033_2]))).unwrap();
        assert_close(result.coefficients[0], 36.550_680_230_661_33, 5e-7);
        assert_close(result.loglik[0], -0.266_892_412_518_194_4, 1e-12);
        assert_close(result.loglik[1], -2.564_007_672_845_036_7e-9, 1e-12);
        assert_close(result.var[0][0], 1_611_299_433.054_453_4, 300.0);
        assert_eq!(result.iter, 20);
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
            x: matrix(n, (0..n).map(|value| value as f64).collect()),
            offset: None,
            strata: None,
        };
        let mut options = options(0, None);
        options.nocenter = Some((0..n).map(|value| value as f64).collect());
        let result = agexact_fit(&data, &options).unwrap();
        assert_close(result.loglik[0], -42.052_280_570_411_12, 1e-10);
        assert_close(result.u[0], -512.0, 1e-10);
        assert_close(result.var[0][0], 3.0 / 16_640.0, 1e-15);
        assert_close(result.sctest, 3_072.0 / 65.0, 1e-10);
        assert_eq!(result.iter, 0);
        assert_eq!(result.flag, 0);
    }
}
