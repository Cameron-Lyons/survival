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
    // Equal risk totals can both be infinite after summing finite case weights.
    let risk_ratio = |smaller: f64, larger: f64| {
        if smaller == larger {
            1.0
        } else {
            smaller / larger
        }
    };
    // At each event, sum the squared shares of the risk weight of rows that stay
    // past that time. A single scale for the curve can erase every remaining
    // weight once a much heavier observation leaves.
    let mut risk_squares = vec![0.0; times.len()];
    let mut later = order.len();
    let mut later_risk = 0.0;
    let mut squares = 0.0;
    for (i, &time) in times.iter().enumerate().rev() {
        let (d0, d1, nrisk) = event_terms(i);
        if d0 > 0.0 && d1 > 0.0 {
            squares *= risk_ratio(later_risk, nrisk).powi(2);
            while later > 0 && data.time2[order[later - 1]] > time {
                later -= 1;
                squares += (data.weight(order[later]) / nrisk).powi(2);
            }
            risk_squares[i] = squares;
            later_risk = nrisk;
        }
    }
    let mut person = 0;
    // a_surv and a_chaz are the influence per unit of risk weight, multiplied by
    // the last event's risk weight. Each subject contributes its share times a.
    let mut previous_risk = f64::INFINITY;
    let (mut a_surv, mut a_chaz) = (0.0_f64, 0.0_f64);
    let (mut departed_surv, mut departed_chaz) = (0.0, 0.0);
    let (mut var_surv, mut var_chaz) = (0.0_f64, 0.0_f64);
    let mut km = 1.0;
    for (i, &time) in times.iter().enumerate() {
        while person < order.len() && left_before(data, options.reverse, order[person], time) {
            let weight = data.weight(order[person]) / previous_risk;
            departed_surv += (weight * a_surv).powi(2);
            departed_chaz += (weight * a_chaz).powi(2);
            person += 1;
        }
        let (d0, d1, nrisk) = event_terms(i);
        if d0 > 0.0 && d1 > 0.0 {
            let hazard = d1 / nrisk;
            let factor = 1.0 - hazard;
            let ratio = risk_ratio(nrisk, previous_risk);
            let (event_derivative, risk_derivative) =
                hazard_influence_terms(d0, hazard, 1.0, options.ctype);
            a_surv = a_surv * ratio * factor + km * hazard;
            a_chaz = a_chaz * ratio - risk_derivative;
            previous_risk = nrisk;
            departed_surv *= factor * factor;

            while person < order.len() && data.time2[order[person]] <= time {
                let row = order[person];
                let weight = data.weight(row) / nrisk;
                let event = has_event(data, options.reverse, row);
                let surv = a_surv - if event { km } else { 0.0 };
                let chaz = a_chaz + if event { event_derivative } else { 0.0 };
                departed_surv += (weight * surv).powi(2);
                departed_chaz += (weight * chaz).powi(2);
                person += 1;
            }
            km *= factor;
            var_surv = departed_surv + risk_squares[i] * a_surv * a_surv;
            var_chaz = departed_chaz + risk_squares[i] * a_chaz * a_chaz;
        }
        std_surv[i] = match options.stype {
            SurvType::KaplanMeier => var_surv.sqrt(),
            SurvType::ExpCumhaz => var_chaz.sqrt(),
        };
        std_chaz[i] = var_chaz.sqrt();
    }
}

/// Largest [`cluster_concentration`] at which [`cluster_variance`] is used.  Its expanded
/// sums cancel by about that factor, so below it they agree with the dense loop to about
/// 1e-11 relative; one cluster holding most of the weight would lose several more digits.
const MAX_CLUSTER_CONCENTRATION: f64 = 1_000.0;

/// Whether [`cluster_variance`] should replace the dense update of every cluster at every
/// event time: it pays off when `nid x ntime` dwarfs the rows, which is also when the
/// dense loop is slow, and its expanded sums are free of cancellation only when no cluster
/// holds a large share of the weight (a handful of large clusters, or one dominant
/// cluster, loses digits).  `concentration` is evaluated only when the counts qualify.
pub(super) fn use_cluster_sums(
    nid: usize,
    ntime: usize,
    nused: usize,
    concentration: impl FnOnce() -> f64,
) -> bool {
    nid > 64
        && (nid as f64) * (ntime as f64) > 32.0 * nused as f64
        && concentration() <= MAX_CLUSTER_CONCENTRATION
}

/// How much grouping rows into clusters inflates their sum of squared weights:
/// `sum_g W_g^2 / sum_i w_i^2` for the clusters' total weights `W_g`.  It is 1 when every
/// row is its own cluster and, for equal weights, the mean size of the cluster a random
/// row belongs to. `rows` gives each row's cluster code, below `ncodes`, and weight.
pub(super) fn cluster_concentration(
    rows: impl Iterator<Item = (usize, f64)> + Clone,
    ncodes: usize,
) -> f64 {
    let scale = rows
        .clone()
        .map(|(_, weight)| weight)
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 1.0;
    }
    let mut totals = vec![0.0; ncodes];
    let mut row_squares = 0.0;
    for (g, weight) in rows {
        let relative = weight / scale;
        totals[g] += relative;
        row_squares += relative * relative;
    }
    totals.iter().map(|total| total * total).sum::<f64>() / row_squares
}

/// How close to zero the survival factor `f = 1 - hazard` of an event time must be for
/// [`cluster_variance`] to take the dense loop's survival step there.  A cluster whose rows
/// all have the event then leaves behind the residue of terms about `1 / |f|` times larger,
/// which the running sums would hold only to `eps / |f|` of its size.  The weight at risk
/// falls by a factor of at least `1 / |f|` at such a time, so there are few of them: usually
/// only the last time of a curve, when every remaining subject has the event.
const DENSE_STEP_FACTOR: f64 = 1e-3;

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
/// millionth of its terms is summed cluster by cluster.  A time at which nearly all the
/// weight at risk has the event ([`DENSE_STEP_FACTOR`]) takes the dense loop's survival
/// step instead, from each cluster's `W a + E` and in the dense loop's order, and the sums
/// restart from its result.  Where everyone at risk has the event (`f == 0`) this is the
/// dense loop's variance bit for bit, the exact zero (or rounding residue) that the
/// confidence limits branch on.
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
        .map(|&row| data.weight(row))
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
            sums.depart(cluster[row], data.weight(row), [a_surv, a_chaz], [0.0, 0.0]);
            person += 1;
        }
        let (d0, d1, nrisk) = event_terms(i);
        if d0 > 0.0 && d1 > 0.0 {
            let hazard = d1 / nrisk;
            let factor = 1.0 - hazard;
            let scaled_nrisk = nrisk / scale;
            let (event_derivative, risk_derivative) =
                hazard_influence_terms(d0, d1 / scale, scaled_nrisk, options.ctype);
            let mut dense_surv = (kaplan_meier && factor.abs() < DENSE_STEP_FACTOR)
                .then(|| sums.dense_surv(a_surv, factor, km, hazard, nrisk));
            a_surv = a_surv * factor + km * hazard / scaled_nrisk;
            a_chaz -= risk_derivative;
            sums.decay_surv(factor);
            let event_influence = [km / scaled_nrisk, event_derivative];
            while person < order.len() && data.time2[order[person]] <= time {
                let row = order[person];
                let g = cluster[row];
                let event = has_event(data, options.reverse, row);
                let row_weight = data.weight(row);
                if event && let Some(dense) = &mut dense_surv {
                    dense[g] -= km * row_weight / nrisk;
                }
                let leaving = if event { event_influence } else { [0.0, 0.0] };
                sums.depart(g, row_weight, [a_surv, a_chaz], leaving);
                person += 1;
            }
            km *= factor;
            match dense_surv {
                Some(dense) => {
                    var_surv = dense.iter().fold(0.0, |sum, x| sum + x * x);
                    sums.restart_surv(&dense, a_surv);
                }
                None => {
                    sums.keep_accurate();
                    if kaplan_meier {
                        var_surv = sums.surv_variance(a_surv);
                    }
                }
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

const KEEPS_SURV: &str = "a Kaplan-Meier curve keeps its survival influences";

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
            weight[cluster[row]] += data.weight(row);
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

    /// The dense loop's survival influences at an event time, before the corrections of
    /// the rows with the event: each cluster's influence `W a + E` times the survival
    /// factor, plus its share of the hazard.
    fn dense_surv(&self, a: f64, factor: f64, km: f64, hazard: f64, nrisk: f64) -> Vec<f64> {
        let surv = self.surv.as_ref().expect(KEEPS_SURV);
        self.weight
            .iter()
            .zip(&surv.values)
            .map(|(&weight, &value)| {
                let influence = weight / self.scale * a + value * surv.decay;
                influence * factor + weight * km * hazard / nrisk
            })
            .collect()
    }

    /// Ends a dense step: each cluster's departed survival influence becomes what its
    /// dense influence leaves beside the weight still at risk, and the sums are recomputed,
    /// so that the running sums go on from the dense loop's values.
    fn restart_surv(&mut self, influences: &[f64], a: f64) {
        let surv = self.surv.as_mut().expect(KEEPS_SURV);
        for ((value, &weight), &influence) in
            surv.values.iter_mut().zip(&self.weight).zip(influences)
        {
            *value = influence - weight / self.scale * a;
        }
        surv.decay = 1.0;
        self.refresh();
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
        self.variance(self.surv.as_ref().expect(KEEPS_SURV), a)
    }

    fn chaz_variance(&self, a: f64) -> f64 {
        self.variance(&self.chaz, a)
    }

    /// `sum (W a + E)^2` from the running sums or, when their expansion cancels, cluster by
    /// cluster.
    fn variance(&self, departed: &Departed, a: f64) -> f64 {
        // `a` can pass 1e154 once the weight at risk is that small a share of the largest
        // weight, while `(a W)^2` stays finite.
        let squares = (a * self.weight_weight.sqrt()).powi(2) + departed.value_value;
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

    /// Every departed influence is multiplied by `factor`.  A zero factor, which only a
    /// dense step has, wipes them out (the running product could not express it) until
    /// [`ClusterSums::restart_surv`] sets them from the dense influences.
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
