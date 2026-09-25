//! Robust variance without materializing individual influence vectors.
//!
//! For right-censored data with one row per cluster, everyone still at risk
//! has influence `weight * a`, with a shared coefficient `a` for each estimate.
//! After leaving the risk set, hazard influence is constant and survival
//! influence is multiplied by each subsequent product-limit factor. Keeping
//! the sum of squared departed influences therefore replaces a scan of every
//! cluster at every event with a linear sweep. Repeated clusters, delayed
//! entry and requests for the influence matrix use the general kernel.

use super::{CurveRows, HazardType, KernelData, KernelOptions, SurvType};

/// Derivatives with respect to event and risk weights, shared with the
/// general infinitesimal-jackknife kernel (including the FH tie correction).
pub(super) fn hazard_influence_terms(
    d0: f64,
    d1: f64,
    nrisk: f64,
    ctype: HazardType,
) -> (f64, f64) {
    match ctype {
        HazardType::NelsonAalen => (1.0 / nrisk, (d1 / nrisk) / nrisk),
        HazardType::FlemingHarrington => {
            let mut denominator = 0.0;
            let mut event_derivative = 0.0;
            let mut risk_derivative = 0.0;
            let remaining = nrisk - d1;
            let mut k = d0.floor();
            while k > 0.0 {
                let fraction = k / d0;
                let inverse = 1.0 / (remaining + fraction * d1);
                denominator += inverse;
                event_derivative += inverse * inverse * fraction;
                risk_derivative += inverse * inverse;
                k -= 1.0;
            }
            denominator /= d0;
            if d1 != d0 {
                event_derivative *= d1 / d0;
                risk_derivative *= d1 / d0;
            }
            (
                denominator + risk_derivative - event_derivative,
                risk_derivative,
            )
        }
    }
}

pub(super) fn independent_variance(
    data: &KernelData<'_>,
    rows: &CurveRows,
    options: KernelOptions,
    times: &[f64],
    event_terms: impl Fn(usize) -> (f64, f64, f64),
    std_surv: &mut [f64],
    std_chaz: &mut [f64],
) {
    let order = &rows.sort2;
    let scale = order
        .iter()
        .map(|&row| data.wt[row])
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        std_surv.fill(0.0);
        std_chaz.fill(0.0);
        return;
    }
    // Sum backwards, avoiding cancellation as the last subjects leave.
    // Normalizing weights also avoids overflowing their squares.
    let mut risk_squares = vec![0.0; order.len() + 1];
    for k in (0..order.len()).rev() {
        let weight = data.wt[order[k]] / scale;
        risk_squares[k] = risk_squares[k + 1] + weight * weight;
    }
    let mut person = 0;
    let (mut a_surv, mut a_chaz) = (0.0_f64, 0.0_f64);
    let (mut departed_surv, mut departed_chaz) = (0.0, 0.0);
    let (mut var_surv, mut var_chaz) = (0.0_f64, 0.0_f64);
    let mut km = 1.0;
    for (i, &time) in times.iter().enumerate() {
        // In reverse curves deaths leave before censorings at the same time.
        while person < order.len() {
            let row = order[person];
            if !(data.time2[row] < time
                || (options.reverse && data.time2[row] == time && data.status[row] == 1))
            {
                break;
            }
            let weight = data.wt[row] / scale;
            departed_surv += (weight * a_surv).powi(2);
            departed_chaz += (weight * a_chaz).powi(2);
            person += 1;
        }
        let (d0, d1, nrisk) = event_terms(i);
        if d0 > 0.0 && d1 > 0.0 {
            let d1 = d1 / scale;
            let nrisk = nrisk / scale;
            let hazard = d1 / nrisk;
            let factor = 1.0 - hazard;
            let (event_derivative, risk_derivative) =
                hazard_influence_terms(d0, d1, nrisk, options.ctype);
            a_surv = a_surv * factor + km * hazard / nrisk;
            a_chaz -= risk_derivative;
            departed_surv *= factor * factor;

            while person < order.len() && data.time2[order[person]] <= time {
                let row = order[person];
                let weight = data.wt[row] / scale;
                let event = (data.status[row] == 0) == options.reverse;
                let surv = a_surv - if event { km / nrisk } else { 0.0 };
                let chaz = a_chaz + if event { event_derivative } else { 0.0 };
                departed_surv += (weight * surv).powi(2);
                departed_chaz += (weight * chaz).powi(2);
                person += 1;
            }
            km *= factor;
            var_surv = departed_surv + risk_squares[person] * a_surv * a_surv;
            var_chaz = departed_chaz + risk_squares[person] * a_chaz * a_chaz;
        }
        std_surv[i] = match options.stype {
            SurvType::KaplanMeier => var_surv.sqrt(),
            SurvType::ExpCumhaz => var_chaz.sqrt(),
        };
        std_chaz[i] = var_chaz.sqrt();
    }
}
