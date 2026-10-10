//! Direct Cox curves for prepared matrices (`coxsurv.fit`), without a fitted model.

use super::agsurv::{
    AgsurvCurve, CoxSurvCurve, CoxSurvType, IndividualInterval, agsurv_rows, check_baseline,
    expand_curve_validated, individual_curve_validated, prepare_baseline,
};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatMatrix, FloatVec, IntVec};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use ndarray::{Array1, Array2, ArrayView2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Original response/design, weights and relative risks. Stratum codes are
/// zero-based; `nstrata` retains declared levels with no observations.
pub struct CoxSurvData<'a> {
    pub y: ArrayView2<'a, f64>,
    pub x: ArrayView2<'a, f64>,
    pub weights: &'a [f64],
    pub risk: &'a [f64],
    pub strata: &'a [i32],
    pub nstrata: usize,
}

/// Prediction rows. `id` requests individual curves in first-appearance order;
/// `y` then supplies (start, stop] and `strata` zero-based interval strata.
pub struct CoxSurvNewData<'a> {
    pub x: ArrayView2<'a, f64>,
    pub risk: &'a [f64],
    pub y: Option<ArrayView2<'a, f64>>,
    pub strata: Option<&'a [i32]>,
    pub id: Option<&'a [i32]>,
}

/// Optional ordinary-curve components retained for the unflattened R layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxSurvBaselineDetails {
    pub hazard: Array1<f64>,
    pub varhaz: Array1<f64>,
    pub ndeath: Array1<f64>,
    pub xbar: Array2<f64>,
}

/// Stacked numerical curves. Matrix columns are prediction rows, or one
/// column for individual trajectories. No response or design data are retained.
#[pyclass(frozen, module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoxSurvRawResult {
    #[pyo3(get)]
    pub n: Vec<usize>,
    #[pyo3(get)]
    pub lengths: Vec<usize>,
    pub time: Array1<f64>,
    pub n_risk: Array1<f64>,
    pub n_event: Array1<f64>,
    pub n_censor: Array1<f64>,
    pub surv: Array2<f64>,
    pub cumhaz: Array2<f64>,
    pub std_err: Option<Array2<f64>>,
    pub details: Option<CoxSurvBaselineDetails>,
}

fn check_matrix(x: ArrayView2<'_, f64>, name: &str) -> SurvivalResult<()> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must be finite"
        )));
    }
    Ok(())
}

fn check_strata(codes: &[i32], n: usize, nstrata: usize, name: &str) -> SurvivalResult<()> {
    validate_length(n, codes.len(), name)?;
    if nstrata == 0
        || codes
            .iter()
            .any(|&code| code < 0 || code as usize >= nstrata)
    {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must contain zero-based indices below nstrata"
        )));
    }
    Ok(())
}

struct CurveStack {
    n: Vec<usize>,
    lengths: Vec<usize>,
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_err: Option<Vec<f64>>,
}

impl CurveStack {
    fn new(se: bool) -> Self {
        Self {
            n: Vec::new(),
            lengths: Vec::new(),
            time: Vec::new(),
            n_risk: Vec::new(),
            n_event: Vec::new(),
            n_censor: Vec::new(),
            surv: Vec::new(),
            cumhaz: Vec::new(),
            std_err: se.then(Vec::new),
        }
    }

    fn push(&mut self, curve: CoxSurvCurve) {
        self.n.push(curve.n);
        self.lengths.push(curve.time.len());
        self.time.extend(curve.time);
        self.n_risk.extend(curve.n_risk);
        self.n_event.extend(curve.n_event);
        self.n_censor.extend(curve.n_censor);
        // Both shared kernels return standard-layout owned matrices.
        self.surv.extend(curve.surv);
        self.cumhaz.extend(curve.cumhaz);
        if let (Some(out), Some(se)) = (&mut self.std_err, curve.std_err) {
            out.extend(se);
        }
    }

    fn finish(self, columns: usize, details: Option<CoxSurvBaselineDetails>) -> CoxSurvRawResult {
        let rows = self.time.len();
        let matrix = |values| {
            Array2::from_shape_vec((rows, columns), values)
                .expect("each curve has the same prediction width")
        };
        CoxSurvRawResult {
            n: self.n,
            lengths: self.lengths,
            time: self.time.into(),
            n_risk: self.n_risk.into(),
            n_event: self.n_event.into(),
            n_censor: self.n_censor.into(),
            surv: matrix(self.surv),
            cumhaz: matrix(self.cumhaz),
            std_err: self.std_err.map(matrix),
            details,
        }
    }
}

fn baseline_details(curves: Vec<AgsurvCurve>, nvar: usize) -> CoxSurvBaselineDetails {
    let mut hazard = Vec::new();
    let mut varhaz = Vec::new();
    let mut ndeath = Vec::new();
    let mut xbar = Vec::new();
    for curve in curves {
        hazard.extend(curve.hazard);
        varhaz.extend(curve.varhaz);
        ndeath.extend(curve.ndeath.into_iter().map(|n| n as f64));
        xbar.extend(curve.xbar);
    }
    let ntime = hazard.len();
    CoxSurvBaselineDetails {
        hazard: hazard.into(),
        varhaz: varhaz.into(),
        ndeath: ndeath.into(),
        xbar: Array2::from_shape_vec((ntime, nvar), xbar).expect("one xbar row per baseline time"),
    }
}

/// Assemble all strata in one call, sharing the kernels used by fitted models.
/// `varmat` requests standard errors. `keep_details` retains ordinary baseline
/// increments for R's list output; individual curves have no baseline extras.
pub fn coxsurv_fit(
    data: &CoxSurvData<'_>,
    newdata: &CoxSurvNewData<'_>,
    stype: u8,
    ctype: u8,
    varmat: Option<&Array2<f64>>,
    keep_details: bool,
) -> SurvivalResult<CoxSurvRawResult> {
    let survtype = CoxSurvType::from_stype_ctype(stype, ctype)?;
    let prepared = prepare_baseline(data.y, data.x, data.weights, data.risk)?;
    check_strata(data.strata, data.y.nrows(), data.nstrata, "strata")?;
    let m = newdata.x.nrows();
    let p = data.x.ncols();
    if m == 0 || newdata.x.ncols() != p {
        return Err(SurvivalError::invalid_input(
            "x2 needs at least one row and the same columns as x",
        ));
    }
    check_matrix(newdata.x, "x2")?;
    validate_length(m, newdata.risk.len(), "risk2")?;
    validate_finite(newdata.risk, "risk2")?;
    validate_non_negative(newdata.risk, "risk2")?;
    if let Some(v) = varmat {
        if v.dim() != (p, p) {
            return Err(SurvivalError::invalid_input(format!(
                "varmat must have shape ({p}, {p})"
            )));
        }
        check_matrix(v.view(), "varmat")?;
    }
    // Validate individual inputs before computing any baseline curves.
    if let Some(id) = newdata.id {
        validate_length(m, id.len(), "id2")?;
        let y = newdata
            .y
            .ok_or_else(|| SurvivalError::invalid_input("y2 is required with id2"))?;
        if y.nrows() != m || !(2..=3).contains(&y.ncols()) {
            return Err(SurvivalError::invalid_input(
                "y2 must have one row per x2 row and 2 or 3 columns",
            ));
        }
        for row in y.rows() {
            if !row[0].is_finite() || !row[1].is_finite() || row[0] >= row[1] {
                return Err(SurvivalError::invalid_input(
                    "y2 requires finite start < stop",
                ));
            }
        }
        if let Some(codes) = newdata.strata {
            check_strata(codes, m, data.nstrata, "strata2")?;
        } else if data.nstrata != 1 {
            return Err(SurvivalError::invalid_input(
                "strata2 is required with multiple strata",
            ));
        }
    }
    let mut rows = vec![Vec::new(); data.nstrata];
    for (i, &code) in data.strata.iter().enumerate() {
        rows[code as usize].push(i);
    }
    let input = prepared.data(data.x, data.weights, data.risk);
    let mut curves = Vec::with_capacity(data.nstrata);
    for rows in rows {
        let curve = agsurv_rows(&input, &rows, survtype, survtype)?;
        check_baseline(&curve)?;
        curves.push(curve);
    }
    let mut result = CurveStack::new(varmat.is_some());
    if let Some(id) = newdata.id {
        let x2 = newdata.x.as_standard_layout();
        let y2 = newdata.y.expect("individual response validated");
        let mut positions = HashMap::new();
        let mut groups: Vec<Vec<IndividualInterval<'_>>> = Vec::new();
        let flat_x = x2.as_slice().expect("standard layout");
        for (i, &subject) in id.iter().enumerate() {
            let position = *positions.entry(subject).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[position].push(IndividualInterval {
                start: y2[(i, 0)],
                stop: y2[(i, 1)],
                stratum: newdata.strata.map_or(0, |codes| codes[i] as usize),
                x2: &flat_x[i * p..(i + 1) * p],
                risk2: newdata.risk[i],
            });
        }
        for intervals in groups {
            result.push(individual_curve_validated(
                &curves, survtype, &intervals, varmat,
            )?);
        }
        Ok(result.finish(1, None))
    } else {
        for curve in &curves {
            result.push(expand_curve_validated(
                curve,
                survtype,
                newdata.x,
                newdata.risk,
                varmat,
            )?);
        }
        let details = keep_details.then(|| baseline_details(curves, p));
        Ok(result.finish(m, details))
    }
}

#[pyfunction(name = "coxsurv_fit")]
#[pyo3(signature = (y, x, weights, risk, strata, nstrata, x2, risk2, stype=2, ctype=1, varmat=None, y2=None, strata2=None, id2=None, keep_details=false))]
#[allow(clippy::too_many_arguments)]
pub fn coxsurv_fit_py(
    py: Python<'_>,
    y: FloatMatrix,
    x: FloatMatrix,
    weights: FloatVec,
    risk: FloatVec,
    strata: IntVec,
    nstrata: usize,
    x2: FloatMatrix,
    risk2: FloatVec,
    stype: u8,
    ctype: u8,
    varmat: Option<FloatMatrix>,
    y2: Option<FloatMatrix>,
    strata2: Option<IntVec>,
    id2: Option<IntVec>,
    keep_details: bool,
) -> PyResult<CoxSurvRawResult> {
    Ok(py.detach(|| {
        coxsurv_fit(
            &CoxSurvData {
                y: y.view(),
                x: x.view(),
                weights: &weights,
                risk: &risk,
                strata: &strata,
                nstrata,
            },
            &CoxSurvNewData {
                x: x2.view(),
                risk: &risk2,
                y: y2.as_ref().map(|y| y.view()),
                strata: strata2.as_deref(),
                id: id2.as_deref(),
            },
            stype,
            ctype,
            varmat.as_deref(),
            keep_details,
        )
    })?)
}

#[cfg(feature = "python")]
#[pymethods]
impl CoxSurvRawResult {
    #[getter(time)]
    fn time_view<'py>(this: Bound<'py, Self>) -> Bound<'py, numpy::PyArray1<f64>> {
        // SAFETY: this frozen result owns the array for the lifetime of its view.
        unsafe { crate::internal::numpy_utils::readonly_view(&this.get().time, this.as_any()) }
    }
    #[getter(n_risk)]
    fn n_risk_view<'py>(this: Bound<'py, Self>) -> Bound<'py, numpy::PyArray1<f64>> {
        // SAFETY: this frozen result owns the array for the lifetime of its view.
        unsafe { crate::internal::numpy_utils::readonly_view(&this.get().n_risk, this.as_any()) }
    }
    #[getter(n_event)]
    fn n_event_view<'py>(this: Bound<'py, Self>) -> Bound<'py, numpy::PyArray1<f64>> {
        // SAFETY: this frozen result owns the array for the lifetime of its view.
        unsafe { crate::internal::numpy_utils::readonly_view(&this.get().n_event, this.as_any()) }
    }
    #[getter(n_censor)]
    fn n_censor_view<'py>(this: Bound<'py, Self>) -> Bound<'py, numpy::PyArray1<f64>> {
        // SAFETY: this frozen result owns the array for the lifetime of its view.
        unsafe { crate::internal::numpy_utils::readonly_view(&this.get().n_censor, this.as_any()) }
    }
    #[getter(surv)]
    fn surv_view<'py>(this: Bound<'py, Self>) -> Bound<'py, numpy::PyArray2<f64>> {
        // SAFETY: this frozen result owns the array for the lifetime of its view.
        unsafe { crate::internal::numpy_utils::readonly_view(&this.get().surv, this.as_any()) }
    }
    #[getter(cumhaz)]
    fn cumhaz_view<'py>(this: Bound<'py, Self>) -> Bound<'py, numpy::PyArray2<f64>> {
        // SAFETY: this frozen result owns the array for the lifetime of its view.
        unsafe { crate::internal::numpy_utils::readonly_view(&this.get().cumhaz, this.as_any()) }
    }
    #[getter(std_err)]
    fn std_err_view<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, numpy::PyArray2<f64>>> {
        this.get().std_err.as_ref().map(|values| {
            // SAFETY: the frozen owner keeps this array alive and unchanged.
            unsafe { crate::internal::numpy_utils::readonly_view(values, this.as_any()) }
        })
    }
    #[getter(hazard)]
    fn hazard_view<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, numpy::PyArray1<f64>>> {
        this.get().details.as_ref().map(|values| {
            // SAFETY: the frozen owner keeps this array alive and unchanged.
            unsafe { crate::internal::numpy_utils::readonly_view(&values.hazard, this.as_any()) }
        })
    }
    #[getter(varhaz)]
    fn varhaz_view<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, numpy::PyArray1<f64>>> {
        this.get().details.as_ref().map(|values| {
            // SAFETY: the frozen owner keeps this array alive and unchanged.
            unsafe { crate::internal::numpy_utils::readonly_view(&values.varhaz, this.as_any()) }
        })
    }
    #[getter(ndeath)]
    fn ndeath_view<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, numpy::PyArray1<f64>>> {
        this.get().details.as_ref().map(|values| {
            // SAFETY: the frozen owner keeps this array alive and unchanged.
            unsafe { crate::internal::numpy_utils::readonly_view(&values.ndeath, this.as_any()) }
        })
    }
    #[getter(xbar)]
    fn xbar_view<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, numpy::PyArray2<f64>>> {
        this.get().details.as_ref().map(|values| {
            // SAFETY: the frozen owner keeps this array alive and unchanged.
            unsafe { crate::internal::numpy_utils::readonly_view(&values.xbar, this.as_any()) }
        })
    }
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, s};

    #[test]
    fn direct_curves_stack_empty_levels_and_multiple_predictions() {
        let y = arr2(&[[1.0, 1.0], [2.0, 1.0], [3.0, 0.0]]);
        let x = Array2::zeros((3, 2));
        let data = CoxSurvData {
            y: y.view(),
            x: x.view(),
            weights: &[1.0; 3],
            risk: &[1.0; 3],
            strata: &[1, 1, 1],
            nstrata: 2,
        };
        let x2 = Array2::zeros((2, 2));
        let newdata = CoxSurvNewData {
            x: x2.view(),
            risk: &[1.0, 2.0],
            y: None,
            strata: None,
            id: None,
        };
        let covariance = Array2::zeros((2, 2));
        let result = coxsurv_fit(&data, &newdata, 2, 1, Some(&covariance), true).unwrap();
        assert_eq!(result.n, vec![0, 3]);
        assert_eq!(result.lengths, vec![0, 3]);
        assert_eq!(result.surv.dim(), (3, 2));
        assert_eq!(result.time.to_vec(), vec![1.0, 2.0, 3.0]);
        assert!((result.cumhaz[(1, 1)] - 5.0 / 3.0).abs() < 1e-14);
        assert!(
            (result.std_err.unwrap()[(2, 1)] - 2.0 * (1.0_f64 / 9.0 + 0.25).sqrt()).abs() < 1e-14
        );
        let details = result.details.unwrap();
        assert_eq!(details.xbar.dim(), (3, 2));
        assert_eq!(details.ndeath.to_vec(), vec![1.0, 1.0, 0.0]);
        assert!(
            coxsurv_fit(&data, &newdata, 2, 1, None, false)
                .unwrap()
                .details
                .is_none()
        );
    }

    #[test]
    fn individual_curves_accept_strided_designs_and_keep_first_appearance_order() {
        let y = arr2(&[[1.0, 1.0], [2.0, 1.0], [3.0, 0.0]]);
        let x = Array2::zeros((3, 2));
        let data = CoxSurvData {
            y: y.view(),
            x: x.view(),
            weights: &[1.0; 3],
            risk: &[1.0; 3],
            strata: &[0; 3],
            nstrata: 1,
        };
        let x2 = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let y2 = arr2(&[[0.0, 1.0], [0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]);
        let newdata = CoxSurvNewData {
            x: x2.slice(s![.., ..;2]),
            risk: &[1.0, 2.0, 1.0, 2.0],
            y: Some(y2.view()),
            strata: None,
            id: Some(&[8, -1, 8, -1]),
        };
        let result = coxsurv_fit(&data, &newdata, 2, 1, None, true).unwrap();
        assert_eq!(result.n, vec![3, 3]);
        assert_eq!(result.lengths, vec![3, 3]);
        assert_eq!(result.time.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        for i in 0..3 {
            assert_eq!(result.cumhaz[(i + 3, 0)], 2.0 * result.cumhaz[(i, 0)]);
        }
        assert!(result.details.is_none());
        assert!(result.std_err.is_none());
    }

    #[test]
    fn direct_boundary_validates_stratum_indices_and_prediction_dimensions() {
        let y = arr2(&[[1.0, 1.0]]);
        let x = arr2(&[[0.0]]);
        let mut data = CoxSurvData {
            y: y.view(),
            x: x.view(),
            weights: &[1.0],
            risk: &[1.0],
            strata: &[0],
            nstrata: 0,
        };
        let mut newdata = CoxSurvNewData {
            x: x.view(),
            risk: &[1.0],
            y: None,
            strata: None,
            id: None,
        };
        assert!(
            coxsurv_fit(&data, &newdata, 2, 1, None, false)
                .unwrap_err()
                .to_string()
                .contains("strata")
        );
        data.nstrata = 1;
        newdata.id = Some(&[1]);
        assert!(
            coxsurv_fit(&data, &newdata, 2, 1, None, false)
                .unwrap_err()
                .to_string()
                .contains("y2 is required")
        );
        newdata.y = Some(y.view());
        assert!(
            coxsurv_fit(&data, &newdata, 2, 1, None, false)
                .unwrap_err()
                .to_string()
                .contains("start < stop")
        );
        newdata.id = None;
        let wrong_covariance = Array2::zeros((0, 0));
        assert!(
            coxsurv_fit(&data, &newdata, 2, 1, Some(&wrong_covariance), false)
                .unwrap_err()
                .to_string()
                .contains("varmat")
        );
    }
}
