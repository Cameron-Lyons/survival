//! The C kernels behind R's `concordancefit`.
//!
//! * `btree` ports the R helper of the same name: it places the sorted
//!   distinct predictor values on a balanced binary tree so that the
//!   weights below, tied with and above a value are `O(log n)` walks.
//! * `concordance_sweep` ports `concordance3.c` (`concordance3` for
//!   right-censored data and `concordance4` for (start, stop] data) and, with
//!   `std_err = false`, the count-only `concordance5`/`concordance6` of
//!   `concordance5.c`; the count statements are identical between the two
//!   files, only the influence, variance and rank work is skipped.
//! * `fastkm` ports `fastkm.c`: `S(t-)`, `G(t-)` and the number at risk
//!   at each unique event time, for the time-weight options.

/// R's `btree(n)`: the tree node (0-based) holding each of `n` sorted ranks.
pub(crate) fn btree(n: usize) -> Vec<usize> {
    fn tfun(n: usize, id: usize, power: usize, out: &mut Vec<usize>) {
        match n {
            0 => {}
            1 => out.push(id),
            2 => out.extend([2 * id + 1, id]),
            3 => out.extend([2 * id + 1, id, 2 * id + 2]),
            _ => {
                let nleft = if n == power * 2 {
                    power
                } else {
                    (power - 1).min(n - power / 2)
                };
                tfun(nleft, 2 * id + 1, power / 2, out);
                out.push(id);
                tfun(n - (nleft + 1), 2 * id + 2, power / 2, out);
            }
        }
    }
    let power = if n > 1 {
        1usize << ((n - 1).ilog2())
    } else {
        0
    };
    let mut out = Vec::with_capacity(n);
    tfun(n, 0, power, &mut out);
    out
}

/// The node weights (`nwt`) and node-plus-descendant weights (`twt`) of
/// `concordance3.c`.
struct RankTree {
    nwt: Vec<f64>,
    twt: Vec<f64>,
}

impl RankTree {
    fn new(ntree: usize) -> Self {
        Self {
            nwt: vec![0.0; ntree],
            twt: vec![0.0; ntree],
        }
    }

    /// `walkup`: weights of the nodes above, below and tied with `index`
    /// (`[greater, smaller, equal]`).
    fn walkup(&self, mut index: usize) -> [f64; 3] {
        let ntree = self.nwt.len();
        let mut sums = [0.0, 0.0, self.nwt[index]];
        let right = 2 * index + 2;
        if right < ntree {
            sums[0] += self.twt[right];
        }
        if right <= ntree {
            sums[1] += self.twt[right - 1];
        }
        while index > 0 {
            let parent = (index - 1) / 2;
            if index % 2 == 1 {
                sums[0] += self.twt[parent] - self.twt[index];
            } else {
                sums[1] += self.twt[parent] - self.twt[index];
            }
            index = parent;
        }
        sums
    }

    /// `addin`: adds `wt` at `index` (negative weights remove).
    fn addin(&mut self, mut index: usize, wt: f64) {
        self.nwt[index] += wt;
        while index > 0 {
            self.twt[index] += wt;
            index = (index - 1) / 2;
        }
        self.twt[0] += wt;
    }

    fn total(&self) -> f64 {
        self.twt[0]
    }
}

/// The `z^2` increment of an observation joining the risk set, the Cox
/// model variance term of `concordance3.c`.
fn cox_variance_term(wt: f64, wsum: [f64; 3]) -> f64 {
    wt * (wsum[0] * (wt + 2.0 * (wsum[1] + wsum[2]))
        + wsum[1] * (wt + 2.0 * (wsum[0] + wsum[2]))
        + (wsum[0] - wsum[1]) * (wsum[0] - wsum[1]))
}

/// Result of one [`concordance_sweep`].
#[derive(Debug, Clone, Default)]
pub(crate) struct SweepOutput {
    /// concordant, discordant, tied on x, tied on y, tied on both, and the
    /// Cox variance numerator (0 without `std_err`).
    pub(crate) count: [f64; 6],
    /// `n x 5` per-observation influence on the five counts (empty
    /// without `std_err`).
    pub(crate) influence: Vec<[f64; 5]>,
    /// Per event, in ascending time order: `(rank, timewt, casewt)`
    /// (empty without `ranks`).
    pub(crate) resid: Vec<[f64; 3]>,
}

/// One stratum's inputs to [`concordance_sweep`].
pub(crate) struct SweepInput<'a> {
    /// Entry times; `None` for right-censored data.
    pub(crate) start: Option<&'a [f64]>,
    pub(crate) stop: &'a [f64],
    /// 0/1 event indicators.
    pub(crate) status: &'a [i32],
    /// Tree node of each observation's predictor rank (`btree` output).
    pub(crate) node: &'a [usize],
    pub(crate) weight: &'a [f64],
    /// Time weight of each unique event time, largest time first (R passes
    /// `rev(timewt)`).
    pub(crate) timewt: &'a [f64],
    /// Observations by decreasing start time (only with `start`).
    pub(crate) sort_start: Option<&'a [usize]>,
    /// Observations by decreasing stop time, censored before deaths at
    /// tied times, then by predictor.
    pub(crate) sort_stop: &'a [usize],
}

/// `concordance3`/`concordance4` (`std_err`) or `concordance5`/`concordance6`
/// (`!std_err`): walks the observations from the largest time down, counting
/// each death against the risk set held in the rank tree.
pub(crate) fn concordance_sweep(input: &SweepInput<'_>, std_err: bool, ranks: bool) -> SweepOutput {
    let n = input.stop.len();
    let ntree = input.node.iter().map(|&x| x + 1).max().unwrap_or(0);
    let nevent = input.status.iter().filter(|&&s| s == 1).count();
    let x = input.node;
    let wt = input.weight;
    let time = input.stop;
    let status = input.status;
    let sort2 = input.sort_stop;
    // `tree` holds everyone at risk, `dtree` the (time-weighted) deaths.
    let mut tree = RankTree::new(ntree);
    let mut dtree = RankTree::new(ntree);
    let mut count = [0.0; 6];
    let mut imat = vec![[0.0; 5]; if std_err { n } else { 0 }];
    let mut resid = vec![[0.0; 3]; if std_err && ranks { nevent } else { 0 }];
    let mut nevent_left = nevent;
    let mut z2 = 0.0;
    let mut utime = 0;
    let mut i2 = 0;

    let mut i = 0;
    while i < n {
        let ii = sort2[i];
        if status[ii] == 0 {
            // Censored: simply add them into the tree.
            if std_err {
                let wsum = dtree.walkup(x[ii]);
                imat[ii][0] -= wsum[1];
                imat[ii][1] -= wsum[0];
                imat[ii][2] -= wsum[2];
                z2 += cox_variance_term(wt[ii], tree.walkup(x[ii]));
            }
            tree.addin(x[ii], wt[ii]);
            i += 1;
            continue;
        }

        // A death: first remove the subjects whose start time has passed.
        if let (Some(start), Some(sort1)) = (input.start, input.sort_start) {
            while i2 < n && start[sort1[i2]] >= time[ii] {
                let jj = sort1[i2];
                if std_err {
                    let wsum = dtree.walkup(x[jj]);
                    imat[jj][0] += wsum[1];
                    imat[jj][1] += wsum[0];
                    imat[jj][2] += wsum[2];
                }
                tree.addin(x[jj], -wt[jj]);
                if std_err {
                    z2 -= cox_variance_term(wt[jj], tree.walkup(x[jj]));
                }
                i2 += 1;
            }
        }

        let mut ndeath = 0;
        let mut dwt = 0.0;
        let mut dwt2 = 0.0;
        let mut xsave = x[ii];
        let mut j2 = i;
        let adjtimewt = input.timewt[utime];
        utime += 1;

        // Pass 1 over the tied deaths.
        let mut j = i;
        while j < n && time[sort2[j]] == time[ii] {
            let jj = sort2[j];
            ndeath += 1;
            count[3] += wt[jj] * dwt * adjtimewt;
            dwt += wt[jj];
            if x[jj] != xsave {
                // Restart the tied-on-both counts for a new predictor value.
                if std_err && wt[sort2[j2]] < dwt2 {
                    while j2 < j {
                        let kk = sort2[j2];
                        imat[kk][4] += (dwt2 - wt[kk]) * adjtimewt;
                        imat[kk][3] -= (dwt2 - wt[kk]) * adjtimewt;
                        j2 += 1;
                    }
                } else {
                    j2 = j;
                }
                dwt2 = 0.0;
                xsave = x[jj];
            }
            count[4] += wt[jj] * dwt2 * adjtimewt;
            dwt2 += wt[jj];

            let wsum = tree.walkup(x[jj]);
            for k in 0..3 {
                count[k] += wt[jj] * wsum[k] * adjtimewt;
                if std_err {
                    imat[jj][k] += wsum[k] * adjtimewt;
                }
            }
            if std_err {
                dtree.addin(x[jj], adjtimewt * wt[jj]);
            }
            j += 1;
        }
        if std_err && wt[sort2[j2]] < dwt2 {
            while j2 < j {
                let kk = sort2[j2];
                imat[kk][4] += (dwt2 - wt[kk]) * adjtimewt;
                imat[kk][3] -= (dwt2 - wt[kk]) * adjtimewt;
                j2 += 1;
            }
        }

        // Pass 2: influence, Cox variance, and add the deaths to the tree.
        for &jj in &sort2[i..i + ndeath] {
            if std_err {
                let wsum = dtree.walkup(x[jj]);
                imat[jj][0] -= wsum[1];
                imat[jj][1] -= wsum[0];
                imat[jj][2] -= wsum[2];
                imat[jj][3] += (dwt - wt[jj]) * adjtimewt;
                z2 += cox_variance_term(wt[jj], tree.walkup(x[jj]));
            }
            tree.addin(x[jj], wt[jj]);
        }
        if std_err {
            count[5] += dwt * adjtimewt * z2 / tree.total();
            if ranks {
                // Ranks use the Cox model risk set, i.e. after the deaths
                // have been added; filled from the back so the result is
                // in ascending time order.
                for &jj in &sort2[i..i + ndeath] {
                    let wsum = tree.walkup(x[jj]);
                    nevent_left -= 1;
                    resid[nevent_left] = [
                        (wsum[0] - wsum[1]) / tree.total(),
                        tree.total() * adjtimewt,
                        wt[jj],
                    ];
                }
            }
        }
        i += ndeath;
    }

    // Finish the influence of those never removed from the tree; the
    // contributions flip because time runs backwards.
    if std_err {
        let remaining: &[usize] = match input.sort_start {
            Some(sort1) => &sort1[i2..],
            None => sort2,
        };
        for &ii in remaining {
            let wsum = dtree.walkup(x[ii]);
            imat[ii][0] += wsum[1];
            imat[ii][1] += wsum[0];
            imat[ii][2] += wsum[2];
        }
    }
    // Ties on both were counted twice, once as tied on y.
    count[3] -= count[4];
    SweepOutput {
        count,
        influence: imat,
        resid,
    }
}

/// `fastkm1`/`fastkm2` output: values at each unique event time, ascending.
#[derive(Debug, Clone, Default)]
pub(crate) struct FastKm {
    pub(crate) etime: Vec<f64>,
    /// `S(t-)`, the Kaplan-Meier survival just before each event time.
    pub(crate) surv: Vec<f64>,
    /// `G(t-)`, the censoring survival; all ones for (start, stop] data,
    /// where R does not compute it.
    pub(crate) censor: Vec<f64>,
    pub(crate) nrisk: Vec<f64>,
}

/// `fastkm.c` on data ordered as for [`concordance_sweep`] (`sort_stop`
/// decreasing time with censored before deaths; `sort_start` decreasing
/// start time for (start, stop] data).
///
/// One row per unique death time, whatever the case weights: the C code
/// counts a time only when its weighted death count is positive, so a
/// death with case weight 0 leaves `concordance3.c` (which advances through
/// `timewt` at every death time) reading past the end of the vector.  A
/// time where no weight at all is at risk leaves `S` and `G` unchanged.
pub(crate) fn fastkm(
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    weight: &[f64],
    sort_start: Option<&[usize]>,
    sort_stop: &[usize],
) -> FastKm {
    let n = stop.len();
    let mut out = FastKm::default();
    if n == 0 {
        return out;
    }
    let mut ncount = vec![0.0; n];
    let mut dcount = vec![0.0; n];
    let mut ccount = vec![0.0; n];
    let mut nevent = 0;
    match (start, sort_start) {
        (Some(start), Some(sort1)) => {
            // Pass 1 (fastkm2): number at risk and deaths at each stop time.
            let mut ntemp = 0.0;
            let mut k = 0;
            let mut i = 0;
            while i < n {
                let dtime = stop[sort_stop[i]];
                while k < n && start[sort1[k]] >= dtime {
                    ntemp -= weight[sort1[k]];
                    k += 1;
                }
                let mut dtemp = 0.0;
                let mut death_here = false;
                while i < n && stop[sort_stop[i]] == dtime {
                    let p2 = sort_stop[i];
                    ntemp += weight[p2];
                    if status[p2] == 1 {
                        dtemp += weight[p2];
                        death_here = true;
                    }
                    // Only the value at the last of a set of tied times is
                    // used.
                    ncount[i] = ntemp;
                    dcount[i] = dtemp;
                    i += 1;
                }
                nevent += usize::from(death_here);
            }
        }
        _ => {
            // Pass 1 (fastkm1): running totals from the largest time down.
            let mut dtime = stop[sort_stop[0]];
            let mut ntemp = 0.0;
            let mut dtemp = 0.0;
            let mut ctemp = 0.0;
            let mut death_here = false;
            for i in 0..n {
                let p = sort_stop[i];
                if dtime != stop[p] {
                    dtemp = 0.0;
                    ctemp = 0.0;
                    dtime = stop[p];
                    nevent += usize::from(death_here);
                    death_here = false;
                }
                ntemp += weight[p];
                if status[p] == 0 {
                    ctemp += weight[p];
                } else {
                    dtemp += weight[p];
                    death_here = true;
                }
                ncount[i] = ntemp;
                dcount[i] = dtemp;
                ccount[i] = ctemp;
            }
            nevent += usize::from(death_here);
        }
    }

    // Pass 2: the lagged curves, walking from the earliest time up.
    out.etime.reserve(nevent);
    let mut stemp = 1.0;
    let mut gtemp = 1.0;
    let mut dtime = f64::NAN;
    let mut ctime = f64::NAN;
    let mut dfirst = true;
    let mut cfirst = true;
    for i in (0..n).rev() {
        if out.etime.len() == nevent {
            break;
        }
        let p = sort_stop[i];
        if status[p] == 1 && (dfirst || stop[p] != dtime) {
            dtime = stop[p];
            dfirst = false;
            out.nrisk.push(ncount[i]);
            out.surv.push(stemp);
            out.censor.push(gtemp);
            out.etime.push(dtime);
            if ncount[i] > 0.0 {
                stemp *= (ncount[i] - dcount[i]) / ncount[i];
            }
        }
        if start.is_none() && status[p] == 0 && (cfirst || stop[p] != ctime) {
            ctime = stop[p];
            cfirst = false;
            if ncount[i] > 0.0 {
                gtemp *= (ncount[i] - ccount[i]) / ncount[i];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btree_matches_r() {
        assert_eq!(btree(1), vec![0]);
        assert_eq!(btree(2), vec![1, 0]);
        assert_eq!(btree(3), vec![1, 0, 2]);
        assert_eq!(btree(4), vec![3, 1, 0, 2]);
        assert_eq!(btree(5), vec![3, 1, 4, 0, 2]);
        assert_eq!(btree(6), vec![3, 1, 4, 0, 5, 2]);
        assert_eq!(btree(7), vec![3, 1, 4, 0, 5, 2, 6]);
        assert_eq!(btree(8), vec![7, 3, 1, 4, 0, 5, 2, 6]);
        assert_eq!(btree(10), vec![7, 3, 8, 1, 9, 4, 0, 5, 2, 6]);
        assert_eq!(btree(13), vec![7, 3, 8, 1, 9, 4, 10, 0, 11, 5, 12, 2, 6]);
        // Every node index is used exactly once and the in-order traversal
        // (left subtree, node, right subtree) is the sorted order.
        for n in 1..40 {
            let tree = btree(n);
            let mut nodes = tree.clone();
            nodes.sort_unstable();
            assert_eq!(nodes, (0..n).collect::<Vec<_>>(), "btree({n})");
            let mut rank_tree = RankTree::new(n);
            for (rank, &node) in tree.iter().enumerate() {
                let sums = rank_tree.walkup(node);
                assert_eq!(sums, [0.0, rank as f64, 0.0], "btree({n}) rank {rank}");
                rank_tree.addin(node, 1.0);
            }
        }
    }

    #[test]
    fn rank_tree_walkup_partitions_weights() {
        let tree_index = btree(5);
        let mut tree = RankTree::new(5);
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0];
        for (rank, &w) in weights.iter().enumerate() {
            tree.addin(tree_index[rank], w);
        }
        assert_eq!(tree.walkup(tree_index[2]), [9.0, 3.0, 3.0]);
        assert_eq!(tree.walkup(tree_index[0]), [14.0, 0.0, 1.0]);
        assert_eq!(tree.walkup(tree_index[4]), [0.0, 10.0, 5.0]);
        tree.addin(tree_index[4], -5.0);
        assert_eq!(tree.walkup(tree_index[4]), [0.0, 10.0, 0.0]);
        assert_eq!(tree.total(), 10.0);
    }

    fn sweep_right(time: &[f64], status: &[i32], x: &[f64], std_err: bool) -> SweepOutput {
        let n = time.len();
        let mut levels = x.to_vec();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        let tree = btree(levels.len());
        let node: Vec<usize> = x
            .iter()
            .map(|v| tree[levels.partition_point(|l| l < v)])
            .collect();
        let mut sort_stop: Vec<usize> = (0..n).collect();
        sort_stop.sort_by(|&a, &b| {
            time[b]
                .total_cmp(&time[a])
                .then_with(|| status[a].cmp(&status[b]))
                .then_with(|| x[a].total_cmp(&x[b]))
        });
        let mut etimes: Vec<f64> = (0..n)
            .filter(|&i| status[i] == 1)
            .map(|i| time[i])
            .collect();
        etimes.sort_by(f64::total_cmp);
        etimes.dedup();
        let timewt = vec![1.0; etimes.len()];
        let weight = vec![1.0; n];
        concordance_sweep(
            &SweepInput {
                start: None,
                stop: time,
                status,
                node: &node,
                weight: &weight,
                timewt: &timewt,
                sort_start: None,
                sort_stop: &sort_stop,
            },
            std_err,
            true,
        )
    }

    #[test]
    fn sweep_counts_match_pairwise_enumeration() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0];
        let status = [1, 1, 0, 1, 1, 1, 0, 1];
        let x = [0.5, 0.2, 0.5, 0.9, 0.2, 0.7, 0.1, 0.9];
        let (mut c, mut d, mut tx, mut ty, mut txy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..8 {
            for j in 0..8 {
                if i == j || status[i] != 1 {
                    continue;
                }
                // i is a death: pair with everyone still at risk after it,
                // which includes the censored at the same time.
                if time[j] > time[i] || (time[j] == time[i] && status[j] == 0) {
                    if x[j] > x[i] {
                        c += 1.0;
                    } else if x[j] < x[i] {
                        d += 1.0;
                    } else {
                        tx += 1.0;
                    }
                } else if time[j] == time[i] && status[j] == 1 && j > i {
                    if x[j] == x[i] {
                        txy += 1.0;
                    } else {
                        ty += 1.0;
                    }
                }
            }
        }
        for std_err in [false, true] {
            let out = sweep_right(&time, &status, &x, std_err);
            assert_eq!(&out.count[..5], &[c, d, tx, ty, txy], "std_err = {std_err}");
        }
        let full = sweep_right(&time, &status, &x, true);
        assert_eq!(full.influence.len(), 8);
        assert_eq!(full.resid.len(), 6);
        // Column sums of the influence matrix are twice the counts.
        for k in 0..5 {
            let total: f64 = full.influence.iter().map(|row| row[k]).sum();
            assert!(
                (total - 2.0 * full.count[k]).abs() < 1e-12,
                "influence column {k}"
            );
        }
        let fast = sweep_right(&time, &status, &x, false);
        assert!(fast.influence.is_empty() && fast.resid.is_empty());
    }

    #[test]
    fn fastkm_matches_kaplan_meier() {
        let time: [f64; 6] = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let status = [1, 0, 1, 1, 0, 1];
        let weight = [1.0; 6];
        let mut sort_stop: Vec<usize> = (0..6).collect();
        sort_stop.sort_by(|&a, &b| {
            time[b]
                .total_cmp(&time[a])
                .then_with(|| status[a].cmp(&status[b]))
        });
        let km = fastkm(None, &time, &status, &weight, None, &sort_stop);
        assert_eq!(km.etime, vec![1.0, 2.0, 3.0, 5.0]);
        assert_eq!(km.nrisk, vec![6.0, 5.0, 3.0, 1.0]);
        let s2 = 5.0 / 6.0;
        let s3 = s2 * 4.0 / 5.0;
        let s5 = s3 * 2.0 / 3.0;
        for (actual, expected) in km.surv.iter().zip([1.0, s2, s3, s5]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        // G(t-) as R computes it (the censoring at 2 sees the running count
        // 4 because the tied death is added after it): 1, 1, 3/4, 3/8.
        for (actual, expected) in km.censor.iter().zip([1.0, 1.0, 0.75, 0.375]) {
            assert!((actual - expected).abs() < 1e-12);
        }

        let start: [f64; 6] = [0.0; 6];
        let mut sort_start: Vec<usize> = (0..6).collect();
        sort_start.sort_by(|&a, &b| start[b].total_cmp(&start[a]));
        let km2 = fastkm(
            Some(&start),
            &time,
            &status,
            &weight,
            Some(&sort_start),
            &sort_stop,
        );
        assert_eq!(km2.etime, km.etime);
        assert_eq!(km2.nrisk, km.nrisk);
        assert_eq!(km2.surv, km.surv);
        assert!(km2.censor.iter().all(|&g| g == 1.0));
    }

    #[test]
    fn fastkm_emits_every_death_time_whatever_the_weights() {
        // Deaths with case weight 0 at 3 (inside the data) and at 6 (the
        // largest time, where nothing with positive weight is at risk).
        let time: [f64; 8] = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0];
        let status = [1, 1, 0, 1, 1, 1, 0, 1];
        let weight = [1.0, 2.0, 0.5, 0.0, 1.0, 2.0, 0.5, 0.0];
        let mut sort_stop: Vec<usize> = (0..8).collect();
        sort_stop.sort_by(|&a, &b| {
            time[b]
                .total_cmp(&time[a])
                .then_with(|| status[a].cmp(&status[b]))
        });
        let km = fastkm(None, &time, &status, &weight, None, &sort_stop);
        assert_eq!(km.etime, vec![1.0, 2.0, 3.0, 4.0, 6.0]);
        assert_eq!(km.nrisk, vec![7.0, 6.0, 3.5, 3.5, 0.0]);
        // The zero-weight death at 3 is a unit factor in S; the time with
        // nobody at risk leaves S and G alone instead of producing 0/0.
        let s2 = 6.0 / 7.0;
        let s3 = s2 * 4.0 / 6.0;
        let s6 = s3 * 0.5 / 3.5;
        for (actual, expected) in km.surv.iter().zip([1.0, s2, s3, s3, s6]) {
            assert!((actual - expected).abs() < 1e-12, "{:?}", km.surv);
        }
        for (actual, expected) in km.censor.iter().zip([1.0, 1.0, 0.875, 0.875, 0.0]) {
            assert!((actual - expected).abs() < 1e-12, "{:?}", km.censor);
        }
        assert!(km.surv.iter().chain(&km.censor).all(|v| v.is_finite()));

        let start: [f64; 8] = [0.0; 8];
        let mut sort_start: Vec<usize> = (0..8).collect();
        sort_start.sort_by(|&a, &b| start[b].total_cmp(&start[a]));
        let km2 = fastkm(
            Some(&start),
            &time,
            &status,
            &weight,
            Some(&sort_start),
            &sort_stop,
        );
        assert_eq!(km2.etime, km.etime);
        assert_eq!(km2.nrisk, km.nrisk);
        assert_eq!(km2.surv, km.surv);
    }
}
