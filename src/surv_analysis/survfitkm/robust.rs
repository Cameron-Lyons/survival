use super::{CurveRows, HazardType, KernelData, KernelOptions, SurvType};

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

/// Whether `row` has left the risk set by the event time `time`: it ended earlier or, for
/// the censoring curve (`reverse`), it died then, a death coming before a censoring it ties.
fn left_before(data: &KernelData<'_>, reverse: bool, row: usize, time: f64) -> bool {
    data.time2[row] < time || (reverse && data.time2[row] == time && data.status[row] == 1)
}

/// Whether `row`, ending at an event time, has the curve's event: a death or, for the
/// censoring curve (`reverse`), a censoring.
fn has_event(data: &KernelData<'_>, reverse: bool, row: usize) -> bool {
    (data.status[row] == 0) == reverse
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
        while person < order.len() && left_before(data, options.reverse, order[person], time) {
            let weight = data.wt[order[person]] / scale;
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
                let event = has_event(data, options.reverse, row);
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

/// Whether [`cluster_variance`] should replace the dense update of every cluster at every
/// event time: it pays off when `nid x ntime` dwarfs the rows, and its expanded sums are
/// only free of cancellation with many clusters (a handful of large ones loses digits),
/// which is also when the dense loop is slow.
pub(super) fn use_cluster_sums(nid: usize, ntime: usize, nused: usize) -> bool {
    nid > 64 && (nid as f64) * (ntime as f64) > 32.0 * nused as f64
}

/// The clustered robust variance of right-censored data without influence matrices, in
/// `O(n + ntime + nid log n)` rather than the dense loop's `O(ntime x nid)`.
///
/// A cluster's influence is `W a + E`: `W` its weight still at risk (divided by the
/// largest weight, against overflow), `a` the influence of a unit of weight at risk
/// throughout ([`independent_variance`]'s recurrence) and `E` what its departed rows left
/// behind.  Every event time multiplies all the survival `E` by `f = 1 - hazard`, so they
/// are kept divided by the running product of `f`; the variance `sum (W a + E)^2` then
/// expands into running sums of `W^2`, `W E` and `E^2` that change only when a row leaves.
/// The sums are recomputed whenever `sum W^2` halves, so that the residue of the
/// subtractions stays small against what is left, and a variance that cancels to below a
/// millionth of its terms is summed cluster by cluster.  A time at which everyone at risk
/// dies (`f == 0`) has the dense loop's survival variance, computed as it does, so the
/// exact zero (or rounding residue) the confidence limits branch on is the same.
pub(super) fn cluster_variance(
    data: &KernelData<'_>,
    rows: &CurveRows,
    options: KernelOptions,
    times: &[f64],
    event_terms: impl Fn(usize) -> (f64, f64, f64),
    std_surv: &mut [f64],
    std_chaz: &mut [f64],
) {
    let (cluster, nid) = data.cluster.expect("the clustered variance has clusters");
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
    let kaplan_meier = options.stype == SurvType::KaplanMeier;
    let mut sums = ClusterSums::new(data, order, cluster, nid, scale, kaplan_meier);
    let mut person = 0;
    let (mut a_surv, mut a_chaz) = (0.0_f64, 0.0_f64);
    let (mut var_surv, mut var_chaz) = (0.0_f64, 0.0_f64);
    let mut km = 1.0;
    for (i, &time) in times.iter().enumerate() {
        while person < order.len() && left_before(data, options.reverse, order[person], time) {
            let row = order[person];
            sums.depart(cluster[row], data.wt[row], [a_surv, a_chaz], [0.0, 0.0]);
            person += 1;
        }
        let (d0, d1, nrisk) = event_terms(i);
        if d0 > 0.0 && d1 > 0.0 {
            let hazard = d1 / nrisk;
            let factor = 1.0 - hazard;
            let scaled_nrisk = nrisk / scale;
            let (event_derivative, risk_derivative) =
                hazard_influence_terms(d0, d1 / scale, scaled_nrisk, options.ctype);
            a_surv = a_surv * factor + km * hazard / scaled_nrisk;
            a_chaz -= risk_derivative;
            let mut dense_surv =
                (kaplan_meier && factor == 0.0).then(|| sums.dense_surv(km, hazard, nrisk));
            sums.decay_surv(factor);
            let event_influence = [km / scaled_nrisk, event_derivative];
            while person < order.len() && data.time2[order[person]] <= time {
                let row = order[person];
                let g = cluster[row];
                let event = has_event(data, options.reverse, row);
                if event && let Some(dense) = &mut dense_surv {
                    dense[g] -= km * data.wt[row] / nrisk;
                }
                let leaving = if event { event_influence } else { [0.0, 0.0] };
                sums.depart(g, data.wt[row], [a_surv, a_chaz], leaving);
                person += 1;
            }
            km *= factor;
            sums.keep_accurate();
            if kaplan_meier {
                var_surv = match dense_surv {
                    Some(dense) => dense.iter().fold(0.0, |sum, x| sum + x * x),
                    None => sums.surv_variance(a_surv),
                };
            }
            var_chaz = sums.chaz_variance(a_chaz);
        }
        std_surv[i] = match options.stype {
            SurvType::KaplanMeier => var_surv.sqrt(),
            SurvType::ExpCumhaz => var_chaz.sqrt(),
        };
        std_chaz[i] = var_chaz.sqrt();
    }
}

/// [`cluster_variance`]'s state: each cluster's weight at risk, the departed influences
/// and the running sums over the clusters (weights divided by `scale`).
struct ClusterSums {
    scale: f64,
    count: Vec<usize>,
    /// The weight at risk, unscaled and updated exactly as the dense loop updates it.
    weight: Vec<f64>,
    weight_weight: f64,
    /// `weight_weight` when the sums were last recomputed.
    refreshed: f64,
    /// The departed survival influences, which only a Kaplan-Meier curve's variance needs.
    surv: Option<Departed>,
    /// The departed cumulative-hazard influences, which no factor multiplies.
    chaz: Departed,
}

/// One influence's departed parts `E` and their running sums `sum W E` and `sum E^2`.
struct Departed {
    /// Each cluster's departed influence, divided by `decay`.
    values: Vec<f64>,
    /// The product of the factors since `values` was last rescaled.
    decay: f64,
    weight_value: f64,
    value_value: f64,
}

impl ClusterSums {
    fn new(
        data: &KernelData<'_>,
        order: &[usize],
        cluster: &[usize],
        nid: usize,
        scale: f64,
        kaplan_meier: bool,
    ) -> Self {
        let mut count = vec![0; nid];
        let mut weight = vec![0.0; nid];
        for &row in order {
            count[cluster[row]] += 1;
            weight[cluster[row]] += data.wt[row];
        }
        let mut sums = Self {
            scale,
            count,
            weight,
            weight_weight: 0.0,
            refreshed: 0.0,
            surv: kaplan_meier.then(|| Departed::new(nid)),
            chaz: Departed::new(nid),
        };
        sums.refresh();
        sums
    }

    /// A row of cluster `g` with weight `weight` leaves the risk set, each unit of its
    /// weight taking the influences `a - leaving` (survival) and `a + leaving` (cumulative
    /// hazard) with it.  The departed influence absorbs the whole change in `W a`, so that
    /// `W a + E` keeps the dense loop's value even when the weight left behind is set to an
    /// exact zero.
    fn depart(&mut self, g: usize, weight: f64, a: [f64; 2], leaving: [f64; 2]) {
        let old_weight = self.weight[g] / self.scale;
        self.count[g] -= 1;
        if self.count[g] == 0 {
            self.weight[g] = 0.0;
        } else {
            self.weight[g] -= weight;
        }
        let new_weight = self.weight[g] / self.scale;
        let moved = old_weight - new_weight;
        let unit = weight / self.scale;
        self.weight_weight += new_weight * new_weight - old_weight * old_weight;
        if let Some(surv) = &mut self.surv {
            surv.depart(
                g,
                [old_weight, new_weight],
                moved * a[0],
                -(unit * leaving[0]),
            );
        }
        self.chaz
            .depart(g, [old_weight, new_weight], moved * a[1], unit * leaving[1]);
    }

    /// The dense loop's survival influences at a time whose survival factor is zero,
    /// before the corrections of the rows that die then: each cluster's old influence
    /// times zero plus its share of the hazard.
    fn dense_surv(&self, km: f64, hazard: f64, nrisk: f64) -> Vec<f64> {
        self.weight
            .iter()
            .map(|&weight| weight * km * hazard / nrisk)
            .collect()
    }

    fn decay_surv(&mut self, factor: f64) {
        if let Some(surv) = &mut self.surv {
            surv.decay(factor);
        }
    }

    /// Folds a vanishing decay into the stored influences and recomputes the sums once
    /// the weight at risk has halved.
    fn keep_accurate(&mut self) {
        if let Some(surv) = &mut self.surv
            && surv.decay.abs() < 1e-100
        {
            for value in &mut surv.values {
                *value *= surv.decay;
            }
            surv.decay = 1.0;
        }
        if self.weight_weight < 0.5 * self.refreshed {
            self.refresh();
        }
    }

    fn refresh(&mut self) {
        let scale = self.scale;
        self.weight_weight = self.weight.iter().fold(0.0, |sum, &weight| {
            let weight = weight / scale;
            sum + weight * weight
        });
        self.refreshed = self.weight_weight;
        if let Some(surv) = &mut self.surv {
            surv.refresh(&self.weight, scale);
        }
        self.chaz.refresh(&self.weight, scale);
    }

    fn surv_variance(&self, a: f64) -> f64 {
        self.variance(
            self.surv
                .as_ref()
                .expect("a Kaplan-Meier curve keeps its survival influences"),
            a,
        )
    }

    fn chaz_variance(&self, a: f64) -> f64 {
        self.variance(&self.chaz, a)
    }

    /// `sum (W a + E)^2` from the running sums or, when their expansion cancels, cluster by
    /// cluster.
    fn variance(&self, departed: &Departed, a: f64) -> f64 {
        let squares = a * a * self.weight_weight + departed.value_value;
        let variance = squares + 2.0 * a * departed.weight_value;
        if variance >= 1e-6 * squares {
            return variance;
        }
        self.weight
            .iter()
            .zip(&departed.values)
            .map(|(&weight, &value)| {
                let influence = weight / self.scale * a + value * departed.decay;
                influence * influence
            })
            .sum()
    }
}

impl Departed {
    fn new(nid: usize) -> Self {
        Self {
            values: vec![0.0; nid],
            decay: 1.0,
            weight_value: 0.0,
            value_value: 0.0,
        }
    }

    /// Cluster `g`'s scaled weight at risk goes from `weights[0]` to `weights[1]` and its
    /// departed influence grows by `moved` (the weight that left times `a`) and `left`.
    fn depart(&mut self, g: usize, weights: [f64; 2], moved: f64, left: f64) {
        let [old_weight, new_weight] = weights;
        let old = self.values[g] * self.decay;
        let new = old + moved + left;
        self.weight_value += new_weight * new - old_weight * old;
        self.value_value += new * new - old * old;
        self.values[g] = new / self.decay;
    }

    /// Every departed influence is multiplied by `factor`; a zero factor wipes them out,
    /// which the running product could not express.
    fn decay(&mut self, factor: f64) {
        if factor == 0.0 {
            self.values.fill(0.0);
            self.decay = 1.0;
            self.weight_value = 0.0;
            self.value_value = 0.0;
        } else {
            self.decay *= factor;
            self.weight_value *= factor;
            self.value_value *= factor * factor;
        }
    }

    fn refresh(&mut self, weight: &[f64], scale: f64) {
        let (mut weight_value, mut value_value) = (0.0, 0.0);
        for (&weight, &value) in weight.iter().zip(&self.values) {
            let (weight, value) = (weight / scale, value * self.decay);
            weight_value += weight * value;
            value_value += value * value;
        }
        (self.weight_value, self.value_value) = (weight_value, value_value);
    }
}
