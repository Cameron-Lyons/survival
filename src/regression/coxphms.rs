use std::collections::BTreeMap;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder};
use pyo3::prelude::*;

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatMatrix, FloatVec, IntVec};
use crate::internal::validation::validate_length;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::{CoxPHFit, CoxphData, CoxphOptions};

#[derive(Debug, Clone)]
pub(crate) struct MsDesign {
    pub cmap: Array2<i32>,
    pub nx: usize,
    pub baseline: Vec<i32>,
    pub strata_use: Array2<bool>,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

impl MsDesign {
    pub(crate) fn ntrans(&self) -> usize {
        self.cmap.ncols()
    }

    pub(crate) fn ncoef(&self) -> usize {
        self.cmap.iter().copied().max().unwrap_or(0).max(0) as usize
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MsStack {
    pub x: Array2<f64>,
    pub rindex: Vec<usize>,
    pub status: Vec<i32>,
    pub block: Vec<usize>,
    pub hazard: Vec<usize>,
    pub strata: Vec<i32>,
}

pub(crate) fn stacker(
    design: &MsDesign,
    istate: &[usize],
    endpoint: &[usize],
    x: ArrayView2<f64>,
    strata_codes: &[Vec<i32>],
    dropzero: bool,
) -> SurvivalResult<MsStack> {
    let n = istate.len();
    validate_length(n, endpoint.len(), "endpoint")?;
    validate_length(n, x.nrows(), "x")?;
    validate_length(design.nx, x.ncols(), "x columns")?;
    for codes in strata_codes {
        validate_length(n, codes.len(), "strata")?;
    }
    let ntrans = design.ntrans();
    let nrow_cmap = design.cmap.nrows();
    let ncoef = design.ncoef();

    let keep_col: Vec<bool> = (0..ntrans)
        .map(|k| !dropzero || design.cmap.column(k).iter().any(|&c| c > 0))
        .collect();
    let mut groups: Vec<i32> = Vec::new();
    for k in (0..ntrans).filter(|&k| keep_col[k]) {
        if !groups.contains(&design.baseline[k]) {
            groups.push(design.baseline[k]);
        }
    }
    let blocks: Vec<Vec<usize>> = groups
        .iter()
        .map(|&group| {
            (0..ntrans)
                .filter(|&k| design.baseline[k] == group)
                .collect()
        })
        .collect();

    let nstate = istate.iter().copied().max().unwrap_or(0);
    let mut at_risk: Vec<Vec<usize>> = vec![Vec::new(); nstate + 1];
    for (row, &state) in istate.iter().enumerate() {
        at_risk[state].push(row);
    }
    let rows_of = |k: usize| -> &[usize] {
        at_risk
            .get(design.from[k])
            .map_or(&[] as &[usize], Vec::as_slice)
    };
    let n2: usize = blocks.iter().flatten().map(|&k| rows_of(k).len()).sum();

    let mut xs = Array2::<f64>::zeros((n2, ncoef));
    let mut rindex = Vec::with_capacity(n2);
    let mut status = Vec::with_capacity(n2);
    let mut block = Vec::with_capacity(n2);
    let mut hazard = Vec::with_capacity(n2);
    let mut r = 0;
    for (b, transitions) in blocks.iter().enumerate() {
        for &k in transitions {
            let cmap_k = design.cmap.column(k);
            for &i in rows_of(k) {
                for c in (0..nrow_cmap).filter(|&c| cmap_k[c] > 0) {
                    let column = cmap_k[c] as usize - 1;
                    xs[(r, column)] = if c < design.nx { x[(i, c)] } else { 1.0 };
                }
                rindex.push(i);
                status.push(i32::from(endpoint[i] == design.to[k]));
                block.push(b + 1);
                hazard.push(k);
                r += 1;
            }
        }
    }

    let mut keep: Vec<bool> = (0..n2)
        .map(|r| xs.row(r).iter().all(|value| !value.is_nan()))
        .collect();
    let strata = if design.strata_use.nrows() == 0 {
        block.iter().map(|&b| b as i32 - 1).collect()
    } else {
        let terms: Vec<Vec<usize>> = blocks
            .iter()
            .map(|transitions| {
                (0..design.strata_use.nrows())
                    .filter(|&t| design.strata_use[(t, transitions[0])])
                    .collect()
            })
            .collect();
        let keys: Vec<Option<(usize, Vec<i32>)>> = (0..n2)
            .map(|r| {
                let codes: Vec<i32> = terms[block[r] - 1]
                    .iter()
                    .map(|&t| strata_codes[t][rindex[r]])
                    .collect();
                (!codes.contains(&-1)).then_some((block[r], codes))
            })
            .collect();
        let mut rank: BTreeMap<&(usize, Vec<i32>), i32> =
            keys.iter().flatten().map(|key| (key, 0)).collect();
        for (code, value) in rank.values_mut().enumerate() {
            *value = code as i32;
        }
        keys.iter()
            .zip(keep.iter_mut())
            .map(|(key, keep)| match key {
                Some(key) => rank[key],
                None => {
                    *keep = false;
                    -1
                }
            })
            .collect()
    };

    let stack = MsStack {
        x: xs,
        rindex,
        status,
        block,
        hazard,
        strata,
    };
    Ok(if keep.iter().all(|&k| k) {
        stack
    } else {
        stack.select(&keep)
    })
}

impl MsStack {
    fn select(self, keep: &[bool]) -> Self {
        let rows: Vec<usize> = (0..keep.len()).filter(|&r| keep[r]).collect();
        Self {
            x: self.x.select(Axis(0), &rows),
            rindex: take(&self.rindex, &rows),
            status: take(&self.status, &rows),
            block: take(&self.block, &rows),
            hazard: take(&self.hazard, &rows),
            strata: take(&self.strata, &rows),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MsShare {
    pub vtype: Vec<i32>,
    pub scale: Vec<f64>,
}

pub(crate) fn share(
    design: &MsDesign,
    stack: &MsStack,
    coefficients: &[f64],
    x: ArrayView2<f64>,
    id: &[i32],
    x_assign: &[i32],
) -> Option<MsShare> {
    let mut seen = Vec::with_capacity(design.baseline.len());
    let duplicated = design.baseline.iter().any(|b| {
        let dup = seen.contains(b);
        seen.push(*b);
        dup
    });
    if !duplicated {
        return None;
    }
    let nid = id.iter().copied().max().unwrap_or(0).max(0) as usize;
    let tdvar: Vec<bool> = (0..design.nx)
        .map(|c| {
            let mut first = vec![f64::NAN; nid + 1];
            x.column(c).iter().zip(id).any(|(&value, &subject)| {
                let reference = &mut first[subject as usize];
                if value.is_nan() {
                    false
                } else if reference.is_nan() {
                    *reference = value;
                    false
                } else {
                    value != *reference
                }
            })
        })
        .collect();
    let nrow = design.cmap.nrows();
    let tdvar2: Vec<bool> = (0..nrow).map(|k| k >= design.nx || tdvar[k]).collect();
    if !tdvar2.iter().any(|&v| v) {
        return None;
    }
    let ntrans = design.ntrans();
    let mut hazard_rows: Vec<Vec<usize>> = vec![Vec::new(); ntrans];
    for (r, &k) in stack.hazard.iter().enumerate() {
        hazard_rows[k].push(r);
    }
    let multipliers = |isgamma: &mut Vec<bool>| -> Vec<f64> {
        let mut scale = vec![1.0; ntrans];
        for (j, rows) in hazard_rows.iter().enumerate() {
            let candidates: Vec<usize> = (0..nrow)
                .filter(|&k| isgamma[k] && design.cmap[(k, j)] > 0)
                .collect();
            for k in candidates {
                let column = design.cmap[(k, j)] as usize - 1;
                let values = stack.x.column(column);
                let first = rows.first().map_or(f64::NAN, |&r| values[r]);
                if rows.iter().all(|&r| values[r] == first) {
                    scale[j] *= (first * coefficients[column]).exp();
                } else {
                    isgamma[k] = false;
                }
            }
        }
        scale
    };
    let mut isgamma = tdvar2.clone();
    let mut scale = multipliers(&mut isgamma);
    let mut all_of_term: BTreeMap<i32, bool> = BTreeMap::new();
    for c in (0..design.nx).filter(|&c| tdvar[c]) {
        *all_of_term.entry(x_assign[c]).or_insert(true) &= isgamma[c];
    }
    let mut changed = false;
    let mut next = isgamma.clone();
    for c in (0..design.nx).filter(|&c| tdvar[c]) {
        next[c] = isgamma[c] && all_of_term[&x_assign[c]];
        changed |= isgamma[c] && !next[c];
    }
    if changed {
        isgamma = next;
        scale = multipliers(&mut isgamma);
    }
    isgamma.iter().any(|&v| v).then(|| MsShare {
        vtype: tdvar2
            .iter()
            .zip(&isgamma)
            .map(|(&td, &gamma)| i32::from(td) + i32::from(gamma))
            .collect(),
        scale,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct MsData {
    pub time: Vec<f64>,
    pub entry: Option<Vec<f64>>,
    pub endpoint: Vec<usize>,
    pub istate: Vec<usize>,
    pub x: Array2<f64>,
    pub strata_terms: Vec<Vec<i32>>,
    pub id: Vec<i32>,
    pub x_assign: Vec<i32>,
    pub weights: Option<Vec<f64>>,
    pub offset: Option<Vec<f64>>,
    pub cluster: Option<Vec<i32>>,
}

#[derive(Debug, Clone)]
pub(crate) struct MsFit {
    pub fit: CoxPHFit,
    pub stack: MsStack,
    pub share: Option<MsShare>,
}

fn take<T: Copy>(values: &[T], rows: &[usize]) -> Vec<T> {
    rows.iter().map(|&r| values[r]).collect()
}

pub(crate) fn coxphms_fit_data(
    data: MsData,
    design: &MsDesign,
    options: CoxphOptions,
) -> SurvivalResult<MsFit> {
    let n = data.time.len();
    for (values, name) in [
        (data.entry.as_ref().map(Vec::len), "entry"),
        (data.weights.as_ref().map(Vec::len), "weights"),
        (data.offset.as_ref().map(Vec::len), "offset"),
        (data.cluster.as_ref().map(Vec::len), "cluster"),
        (Some(data.istate.len()), "istate"),
        (Some(data.id.len()), "id"),
    ] {
        if let Some(len) = values {
            validate_length(n, len, name)?;
        }
    }
    if data.id.iter().any(|&subject| subject < 1) {
        return Err(SurvivalError::invalid_input("id codes must be 1-based"));
    }
    if data.istate.contains(&0) {
        return Err(SurvivalError::invalid_input("istate codes must be 1-based"));
    }
    let nstate = data
        .istate
        .iter()
        .chain(&design.to)
        .copied()
        .max()
        .unwrap_or(0);
    if data.endpoint.iter().any(|&state| state > nstate) {
        return Err(SurvivalError::invalid_input(
            "endpoint codes must name a state",
        ));
    }
    validate_length(design.nx, data.x_assign.len(), "x_assign")?;
    if design.strata_use.nrows() != data.strata_terms.len() {
        return Err(SurvivalError::invalid_input(
            "strata_use needs one row per strata term",
        ));
    }
    let stack = stacker(
        design,
        &data.istate,
        &data.endpoint,
        data.x.view(),
        &data.strata_terms,
        true,
    )?;
    if stack.x.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "data contains an infinite predictor",
        ));
    }
    let rows = &stack.rindex;
    let offset = data.offset.as_deref().map(|offset| take(offset, rows));
    if let Some(init) = &options.init {
        let mean = data
            .offset
            .as_ref()
            .map_or(0.0, |all| all.iter().sum::<f64>() / n as f64);
        let centred: Option<Vec<f64>> = offset
            .as_ref()
            .map(|stacked| stacked.iter().map(|value| value - mean).collect());
        check_init(init, &stack.x, centred.as_deref())?;
    }
    let fit_data = CoxphData::try_new(
        take(&data.time, rows),
        data.entry.as_deref().map(|entry| take(entry, rows)),
        stack.status.clone(),
        stack.x.clone(),
        data.weights.as_deref().map(|weights| take(weights, rows)),
        Some(stack.strata.clone()),
        offset,
    )?;
    let options = CoxphOptions {
        cluster: data.cluster.as_deref().map(|cluster| take(cluster, rows)),
        ..options
    };
    let fit = CoxPHFit::fit(fit_data, options)?;
    let share = share(
        design,
        &stack,
        &fit.coefficients,
        data.x.view(),
        &data.id,
        &data.x_assign,
    );
    Ok(MsFit { fit, stack, share })
}

fn check_init(init: &[f64], x: &Array2<f64>, offset: Option<&[f64]>) -> SurvivalResult<()> {
    if init.len() != x.ncols() {
        return Err(SurvivalError::invalid_input(
            "wrong length for init argument",
        ));
    }
    let n = x.nrows();
    let center: f64 = x
        .columns()
        .into_iter()
        .zip(init)
        .map(|(column, b)| column.sum() / n as f64 * b)
        .sum();
    let risks: Vec<f64> = (0..n)
        .map(|i| {
            let eta = x.row(i).dot(&ArrayView1::from(init)) - center
                + offset.map_or(0.0, |offset| offset[i]);
            eta.exp()
        })
        .collect();
    if risks.iter().any(|risk| risk.is_infinite()) || risks.iter().all(|&risk| risk == 0.0) {
        return Err(SurvivalError::invalid_input(
            "initial values lead to overflow or underflow of the exp function",
        ));
    }
    Ok(())
}

pub(crate) fn to_index(values: IntVec, name: &str) -> SurvivalResult<Vec<usize>> {
    values
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| SurvivalError::invalid_input(format!("{name} must not be negative")))
        })
        .collect()
}

fn column_major(values: IntVec, nrow: usize, name: &str) -> SurvivalResult<Array2<i32>> {
    let ncol = values.len().checked_div(nrow).unwrap_or(0);
    if nrow * ncol != values.len() {
        return Err(SurvivalError::invalid_input(format!(
            "{name} does not have {nrow} rows"
        )));
    }
    Array2::from_shape_vec((nrow, ncol).f(), values.into_inner())
        .map_err(|err| SurvivalError::invalid_input(err.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ms_design(
    nx: usize,
    cmap: IntVec,
    cmap_nrow: usize,
    baseline: IntVec,
    trans_from: IntVec,
    trans_to: IntVec,
    strata_use: Option<IntVec>,
    nterm: usize,
) -> SurvivalResult<MsDesign> {
    let cmap = column_major(cmap, cmap_nrow, "cmap")?;
    let strata_use = match strata_use {
        Some(values) if nterm > 0 => {
            column_major(values, nterm, "strata_use")?.mapv(|value| value != 0)
        }
        _ => Array2::from_elem((0, cmap.ncols()), false),
    };
    let design = MsDesign {
        nx,
        baseline: baseline.into_inner(),
        strata_use,
        from: to_index(trans_from, "trans_from")?,
        to: to_index(trans_to, "trans_to")?,
        cmap,
    };
    for (len, name) in [
        (design.baseline.len(), "baseline"),
        (design.from.len(), "trans_from"),
        (design.to.len(), "trans_to"),
    ] {
        validate_length(design.ntrans(), len, name)?;
    }
    if design.cmap.nrows() < design.nx {
        return Err(SurvivalError::invalid_input(
            "cmap needs a row per column of x",
        ));
    }
    Ok(design)
}

#[doc = " The fit of `coxph()` for a multi-state response, from the unstacked data"]
#[doc = " and the transition maps (R's `cmap` and `smap`)."]
#[doc = ""]
#[doc = " `endpoint` is the 1-based state each row moves to (0 when censored),"]
#[doc = " `istate` its 1-based current state; `cmap` (flat, column-major, with"]
#[doc = " `cmap_nrow` rows) maps covariates to coefficients per transition,"]
#[doc = " `baseline` is `smap[1, ]`, `strata_use` (flat, column-major, one row per"]
#[doc = " entry of `strata_terms`) is `smap[-1, ]`, and `trans_from`/`trans_to`"]
#[doc = " are the 1-based states of each transition.  `id` are 1-based subject"]
#[doc = " codes and `x_assign` the term of each column of `x`; `cluster` holds"]
#[doc = " integer codes.  Every per-row vector is unstacked."]
#[doc = ""]
#[doc = " Returns the fit of the stacked data, then, per stacked row, its 0-based"]
#[doc = " source row, 1-based block, 0-based transition and stratum, and"]
#[doc = " `share$vtype` and `share$scale` when the model has them."]
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (time, endpoint, istate, x, cmap, cmap_nrow, baseline, trans_from, trans_to, id, x_assign, entry=None, strata_terms=None, strata_use=None, weights=None, offset=None, cluster=None, method="breslow", init=None, iter_max=None, eps=None, toler_chol=None, nocenter=None, robust=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxphms_fit(
    py: Python<'_>,
    time: FloatVec,
    endpoint: IntVec,
    istate: IntVec,
    x: FloatMatrix,
    cmap: IntVec,
    cmap_nrow: usize,
    baseline: IntVec,
    trans_from: IntVec,
    trans_to: IntVec,
    id: IntVec,
    x_assign: IntVec,
    entry: Option<FloatVec>,
    strata_terms: Option<Vec<IntVec>>,
    strata_use: Option<IntVec>,
    weights: Option<FloatVec>,
    offset: Option<FloatVec>,
    cluster: Option<IntVec>,
    method: &str,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
    robust: Option<bool>,
) -> PyResult<(
    CoxPHFit,
    IntVec,
    IntVec,
    IntVec,
    IntVec,
    Option<IntVec>,
    Option<FloatVec>,
)> {
    let strata_terms: Vec<Vec<i32>> = strata_terms
        .unwrap_or_default()
        .into_iter()
        .map(IntVec::into_inner)
        .collect();
    let design = ms_design(
        x.ncol(),
        cmap,
        cmap_nrow,
        baseline,
        trans_from,
        trans_to,
        strata_use,
        strata_terms.len(),
    )?;
    let data = MsData {
        time: time.into_inner(),
        entry: entry.map(FloatVec::into_inner),
        endpoint: to_index(endpoint, "endpoint")?,
        istate: to_index(istate, "istate")?,
        x: x.into_inner(),
        strata_terms,
        id: id.into_inner(),
        x_assign: x_assign.into_inner(),
        weights: weights.map(FloatVec::into_inner),
        offset: offset.map(FloatVec::into_inner),
        cluster: cluster.map(IntVec::into_inner),
    };
    let defaults = CoxphOptions::default();
    let options = CoxphOptions {
        method: TieMethod::parse(Some(method))?,
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter: nocenter.or(defaults.nocenter),
        cluster: None,
        robust,
    };
    let MsFit { fit, stack, share } =
        py.detach(move || coxphms_fit_data(data, &design, options))?;
    let as_codes = |values: Vec<usize>| IntVec(values.into_iter().map(|v| v as i32).collect());
    let (vtype, scale) = match share {
        Some(share) => (Some(IntVec(share.vtype)), Some(FloatVec(share.scale))),
        None => (None, None),
    };
    Ok((
        fit,
        as_codes(stack.rindex),
        as_codes(stack.block),
        as_codes(stack.hazard),
        IntVec(stack.strata),
        vtype,
        scale,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mtest() -> (Vec<usize>, Vec<usize>, Array2<f64>) {
        let istate = vec![1, 2, 3, 1, 1, 1, 2, 4, 1, 3];
        let endpoint = vec![2, 3, 2, 3, 4, 2, 4, 0, 3, 0];
        let x =
            Array2::from_shape_vec((10, 1), vec![0., 0., 0., 1., 1., 0., 0., 0., 2., 2.]).unwrap();
        (istate, endpoint, x)
    }

    fn mtest_design() -> MsDesign {
        MsDesign {
            cmap: Array2::from_shape_vec((1, 6), vec![1, 2, 3, 4, 5, 6]).unwrap(),
            nx: 1,
            baseline: vec![1, 2, 3, 4, 5, 6],
            strata_use: Array2::from_elem((0, 6), false),
            from: vec![1, 3, 1, 2, 1, 2],
            to: vec![2, 2, 3, 3, 4, 4],
        }
    }

    #[test]
    fn stacker_builds_one_block_per_transition() {
        let (istate, endpoint, x) = mtest();
        let stack = stacker(&mtest_design(), &istate, &endpoint, x.view(), &[], true).unwrap();
        assert_eq!(stack.rindex.len(), 21);
        let mut sizes = vec![0; 6];
        for &b in &stack.block {
            sizes[b - 1] += 1;
        }
        assert_eq!(sizes, [5, 2, 5, 2, 5, 2]);
        assert_eq!(&stack.rindex[..5], &[0, 3, 4, 5, 8]);
        assert_eq!(&stack.status[..5], &[1, 0, 0, 1, 0]);
        assert_eq!(&stack.strata[..7], &[0, 0, 0, 0, 0, 1, 1]);
        assert_eq!(stack.x[(3, 0)], 0.0);
        assert_eq!(stack.x[(4, 0)], 2.0);
        assert_eq!(stack.rindex[5..7], [2, 9]);
        assert_eq!(stack.x[(6, 1)], 2.0);
        assert_eq!(stack.x[(6, 0)], 0.0);
        assert_eq!(stack.hazard[6], 1);
    }

    #[test]
    fn dropzero_skips_transitions_without_coefficients() {
        let (istate, endpoint, x) = mtest();
        let mut design = mtest_design();
        design.cmap[(0, 1)] = 0;
        let kept = stacker(&design, &istate, &endpoint, x.view(), &[], true).unwrap();
        assert_eq!(kept.rindex.len(), 19);
        assert!(!kept.hazard.contains(&1));
        let all = stacker(&design, &istate, &endpoint, x.view(), &[], false).unwrap();
        assert_eq!(all.rindex.len(), 21);
    }

    #[test]
    fn stacker_drops_rows_with_a_missing_covariate_they_use() {
        let (istate, endpoint, mut x) = mtest();
        x[(1, 0)] = f64::NAN;
        let stack = stacker(&mtest_design(), &istate, &endpoint, x.view(), &[], true).unwrap();
        assert_eq!(stack.rindex.len(), 19);
        assert!(!stack.rindex.contains(&1));
        let design = MsDesign {
            cmap: Array2::from_shape_vec((2, 2), vec![1, 0, 0, 2]).unwrap(),
            nx: 2,
            baseline: vec![1, 2],
            strata_use: Array2::from_elem((0, 2), false),
            from: vec![1, 1],
            to: vec![2, 3],
        };
        let x2 = Array2::from_shape_vec((3, 2), vec![1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let stack = stacker(&design, &[1, 1, 1], &[2, 3, 0], x2.view(), &[], true).unwrap();
        assert_eq!(stack.rindex, [0, 1, 2, 1, 2]);
    }

    #[test]
    fn strata_codes_rank_block_and_term_codes() {
        let design = MsDesign {
            cmap: Array2::from_shape_vec((1, 2), vec![1, 2]).unwrap(),
            nx: 1,
            baseline: vec![1, 2],
            strata_use: Array2::from_shape_vec((2, 2), vec![true, true, true, false]).unwrap(),
            from: vec![1, 1],
            to: vec![2, 3],
        };
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let codes = vec![vec![1, 0, 1, 0], vec![0, 1, -1, 1]];
        let stack = stacker(
            &design,
            &[1, 1, 1, 1],
            &[2, 3, 0, 2],
            x.view(),
            &codes,
            true,
        )
        .unwrap();
        assert_eq!(stack.rindex, [0, 1, 3, 0, 1, 2, 3]);
        assert_eq!(stack.strata, [1, 0, 0, 3, 2, 3, 2]);
    }

    #[test]
    fn misaligned_block_uses_its_first_transition_strata() {
        let design = MsDesign {
            cmap: Array2::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap(),
            nx: 1,
            baseline: vec![1, 1, 2],
            strata_use: Array2::from_shape_vec((1, 3), vec![false, false, true]).unwrap(),
            from: vec![1, 1, 2],
            to: vec![2, 3, 3],
        };
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let codes = vec![vec![0, 1, 0, 1]];
        let stack = stacker(
            &design,
            &[1, 1, 2, 2],
            &[2, 0, 3, 0],
            x.view(),
            &codes,
            true,
        )
        .unwrap();
        assert_eq!(stack.block, [1, 1, 1, 1, 2, 2]);
        assert_eq!(stack.strata, [0, 0, 0, 0, 1, 2]);
    }

    #[test]
    fn fit_data_rejects_misaligned_istate_and_bad_codes() {
        let (istate, endpoint, x) = mtest();
        let data = MsData {
            time: vec![1.0; 10],
            entry: None,
            endpoint,
            istate,
            x,
            strata_terms: Vec::new(),
            id: vec![1, 1, 1, 2, 3, 4, 4, 4, 5, 5],
            x_assign: vec![1],
            weights: None,
            offset: None,
            cluster: None,
        };
        let error = |data: MsData| {
            coxphms_fit_data(data, &mtest_design(), CoxphOptions::default())
                .unwrap_err()
                .to_string()
        };
        let mut long = data.clone();
        long.istate.push(1);
        assert!(error(long).contains("istate"));
        let mut zero = data.clone();
        zero.id[0] = 0;
        assert!(error(zero).contains("id codes must be 1-based"));
        let mut no_state = data.clone();
        no_state.istate[0] = 0;
        assert!(error(no_state).contains("istate codes must be 1-based"));
        let mut past_last = data;
        past_last.endpoint[0] = 5;
        assert!(error(past_last).contains("endpoint codes must name a state"));
    }

    #[test]
    fn share_marks_constant_ph_covariates_and_scales_the_hazard() {
        let design = MsDesign {
            cmap: Array2::from_shape_vec((2, 2), vec![1, 2, 0, 3]).unwrap(),
            nx: 1,
            baseline: vec![1, 1],
            strata_use: Array2::from_elem((0, 2), false),
            from: vec![1, 1],
            to: vec![2, 3],
        };
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let stack = stacker(&design, &[1, 1, 1], &[2, 3, 0], x.view(), &[], true).unwrap();
        let share = share(
            &design,
            &stack,
            &[0.5, 0.2, -1.5],
            x.view(),
            &[1, 2, 3],
            &[1],
        )
        .unwrap();
        assert_eq!(share.vtype, [0, 2]);
        assert_eq!(share.scale, [1.0, (-1.5f64).exp()]);
        let separate = MsDesign {
            baseline: vec![1, 2],
            ..design
        };
        assert!(share_of(&separate, &x).is_none());
    }

    fn share_of(design: &MsDesign, x: &Array2<f64>) -> Option<MsShare> {
        let stack = stacker(design, &[1, 1, 1], &[2, 3, 0], x.view(), &[], true).unwrap();
        share(
            design,
            &stack,
            &[0.5, 0.2, -1.5],
            x.view(),
            &[1, 2, 3],
            &[1],
        )
    }
}
