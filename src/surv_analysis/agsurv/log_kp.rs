//! Fitted K-P point increments, retaining log(-log(S)) before baseline rounding.
use super::{AgsurvData, group_index, reverse_cumsum};

#[allow(clippy::too_many_arguments)]
pub(super) fn increments(
    data: &AgsurvData<'_>,
    rows: &[usize],
    deaths: &[usize],
    time: &[f64],
    ndeath: &[usize],
    nrisk: &[f64],
    erisk: &[f64],
    log_risk: &[f64],
) -> Option<Vec<f64>> {
    let mut censor_risk = vec![0.0; time.len()];
    let mut active = data.start.map(|_| vec![0usize; time.len()]);
    for &row in rows {
        let g = group_index(time, data.stop[row]);
        if data.status[row] == 0 {
            censor_risk[g] += data.weights[row] * data.risk[row];
        }
        if let Some(counts) = &mut active {
            counts[g] += 1;
        }
    }
    if let (Some(start), Some(active)) = (data.start, &mut active) {
        let mut total = 0;
        for count in active.iter_mut().rev() {
            total += *count;
            *count = total;
        }
        // An interval beginning after a grid point is not yet at risk there.
        let mut entrants = vec![0.0; time.len() + 1];
        for &row in rows {
            entrants[time.partition_point(|&t| t <= start[row])] += 1.0;
        }
        reverse_cumsum(&mut entrants);
        for (g, count) in active.iter_mut().enumerate() {
            *count -= entrants[g + 1] as usize;
        }
    }
    let mut logs = vec![f64::NEG_INFINITY; time.len()];
    let mut first = 0;
    for (g, &d) in ndeath.iter().enumerate() {
        if d == 0 {
            continue;
        }
        let end = first + d;
        let nondeath = if data.start.is_none() {
            // Both terms are positive: don't subtract a large death mass from
            // a nearly equal denominator and erase the surviving risk.
            censor_risk[g] + nrisk.get(g + 1).copied().unwrap_or(0.0)
        } else if active.as_ref().expect("counting risk counts")[g] == d {
            0.0
        } else {
            let difference = nrisk[g] - erisk[g];
            if difference <= nrisk[g] * 1e-6 {
                return None;
            }
            difference
        };
        logs[g] = if nondeath == 0.0 {
            f64::INFINITY
        } else {
            ordinary_root(
                data,
                &deaths[first..end],
                log_risk,
                nrisk[g],
                erisk[g],
                nondeath,
            )
        };
        first = end;
    }
    Some(logs)
}

fn ordinary_root(
    data: &AgsurvData<'_>,
    deaths: &[usize],
    log_risk: &[f64],
    denom: f64,
    death_mass: f64,
    nondeath: f64,
) -> f64 {
    let reference = log_risk[deaths[0]];
    if deaths.iter().all(|&row| log_risk[row] == reference) {
        return (death_mass / nondeath).ln_1p().ln() - reference;
    }
    // Solve in u=q*reference_risk. Relative risks remain modest on this path;
    // the general log sweep handles unsafe spreads and weights instead.
    let terms: Vec<(f64, f64)> = deaths
        .iter()
        .map(|&row| {
            (
                (log_risk[row] - reference).exp(),
                data.weights[row] * data.risk[row] / denom,
            )
        })
        .collect();
    let event_weight: f64 = deaths.iter().map(|&row| data.weights[row]).sum();
    let target = nondeath / denom;
    let mut low = (event_weight.ln() - denom.ln() + reference).exp();
    let mut high = (event_weight.ln() - nondeath.ln() + reference).exp();
    let mut u = low;
    for _ in 0..80 {
        let mut value = -target;
        let mut derivative = 0.0;
        for &(relative, fraction) in &terms {
            let z = u * relative;
            let divisor = -(-z).exp_m1();
            let term = fraction * (-z).exp() / divisor;
            value += term;
            derivative -= term * relative / divisor;
        }
        if value == 0.0 {
            break;
        }
        if value > 0.0 {
            low = u;
        } else {
            high = u;
        }
        let next = u - value / derivative;
        if next.is_finite() && (next - u).abs() <= 8.0 * f64::EPSILON * u {
            u = next;
            break;
        }
        let next = if next > low && next < high {
            next
        } else {
            low + (high - low) / 2.0
        };
        if next == u || next == low || next == high {
            break;
        }
        u = next;
    }
    u.ln() - reference
}
