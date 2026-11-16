use crate::concordance::rank_tree::{PairedRankTree, RankTree};

fn cox_variance_term(wt: f64, wsum: [f64; 3]) -> f64 {
    wt * (wsum[0] * (wt + 2.0 * (wsum[1] + wsum[2]))
        + wsum[1] * (wt + 2.0 * (wsum[0] + wsum[2]))
        + (wsum[0] - wsum[1]) * (wsum[0] - wsum[1]))
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SweepOutput {
    pub(crate) count: [f64; 6],
    pub(crate) influence: Vec<[f64; 5]>,
    pub(crate) resid: Vec<[f64; 3]>,
}

pub(crate) struct SweepInput<'a> {
    pub(crate) start: Option<&'a [f64]>,
    pub(crate) stop: &'a [f64],
    pub(crate) status: &'a [i32],
    pub(crate) node: &'a [usize],
    pub(crate) weight: &'a [f64],
    pub(crate) timewt: &'a [f64],
    pub(crate) sort_start: Option<&'a [usize]>,
    /// A permutation of the rows: descending stop time, censored rows before
    /// deaths at each time.
    pub(crate) sort_stop: &'a [usize],
}

pub(crate) fn concordance_sweep(input: &SweepInput<'_>, std_err: bool, ranks: bool) -> SweepOutput {
    if std_err {
        influence_sweep(input, ranks)
    } else {
        SweepOutput {
            count: count_sweep(input),
            influence: Vec::new(),
            resid: Vec::new(),
        }
    }
}

fn tree_size(node: &[usize]) -> usize {
    node.iter().map(|&x| x + 1).max().unwrap_or(0)
}

/// The pair counts alone (`std.err = FALSE`).  They need only the risk-set
/// tree, kept in its own compact arrays: a walk then reads 8 bytes per node
/// rather than the 32 of a [`PairedRankTree`] node.
fn count_sweep(input: &SweepInput<'_>) -> [f64; 6] {
    let n = input.stop.len();
    let x = input.node;
    let wt = input.weight;
    let time = input.stop;
    let status = input.status;
    let sort2 = input.sort_stop;
    let mut tree = RankTree::new(tree_size(x));
    let mut count = [0.0; 6];
    let mut utime = 0;
    let mut i2 = 0;

    let mut i = 0;
    while i < n {
        let ii = sort2[i];
        if status[ii] == 0 {
            tree.addin(x[ii], wt[ii]);
            i += 1;
            continue;
        }

        if let (Some(start), Some(sort1)) = (input.start, input.sort_start) {
            while i2 < n && start[sort1[i2]] >= time[ii] {
                let jj = sort1[i2];
                tree.addin(x[jj], -wt[jj]);
                i2 += 1;
            }
        }

        let mut ndeath = 0;
        let mut dwt = 0.0;
        let mut dwt2 = 0.0;
        let mut xsave = x[ii];
        let adjtimewt = input.timewt[utime];
        utime += 1;

        let mut j = i;
        while j < n && time[sort2[j]] == time[ii] {
            let jj = sort2[j];
            ndeath += 1;
            count[3] += wt[jj] * dwt * adjtimewt;
            dwt += wt[jj];
            if x[jj] != xsave {
                dwt2 = 0.0;
                xsave = x[jj];
            }
            count[4] += wt[jj] * dwt2 * adjtimewt;
            dwt2 += wt[jj];

            let wsum = tree.walkup(x[jj]);
            for k in 0..3 {
                count[k] += wt[jj] * wsum[k] * adjtimewt;
            }
            j += 1;
        }
        for &jj in &sort2[i..i + ndeath] {
            tree.addin(x[jj], wt[jj]);
        }
        i += ndeath;
    }
    count[3] -= count[4];
    count
}

/// A row joining the risk set takes off the death-tree sums of that moment,
/// so that its influence counts only the deaths while it is at risk.
fn enter_risk_set(row: &mut [f64; 5], death: [f64; 3]) {
    row[0] -= death[1];
    row[1] -= death[0];
    row[2] -= death[2];
}

/// A row leaving the risk set adds back the death-tree sums of that moment.
fn leave_risk_set(row: &mut [f64; 5], death: [f64; 3]) {
    row[0] += death[1];
    row[1] += death[0];
    row[2] += death[2];
}

/// The counts with each row's influence on them, the Cox variance and, with
/// `ranks`, the per-event residuals.
fn influence_sweep(input: &SweepInput<'_>, ranks: bool) -> SweepOutput {
    let n = input.stop.len();
    let ntree = tree_size(input.node);
    let nevent = input.status.iter().filter(|&&s| s == 1).count();
    let x = input.node;
    let wt = input.weight;
    let time = input.stop;
    let status = input.status;
    let sort2 = input.sort_stop;
    let mut trees = PairedRankTree::new(ntree);
    let mut count = [0.0; 6];
    let mut imat = vec![[0.0; 5]; n];
    let mut resid = vec![[0.0; 3]; if ranks { nevent } else { 0 }];
    let mut nevent_left = nevent;
    let mut z2 = 0.0;
    let mut utime = 0;
    let mut i2 = 0;

    let mut i = 0;
    while i < n {
        let ii = sort2[i];
        if status[ii] == 0 {
            let (death, risk) = trees.walkup_both_add_risk(x[ii], wt[ii]);
            enter_risk_set(&mut imat[ii], death);
            z2 += cox_variance_term(wt[ii], risk);
            i += 1;
            continue;
        }

        if let (Some(start), Some(sort1)) = (input.start, input.sort_start) {
            while i2 < n && start[sort1[i2]] >= time[ii] {
                let jj = sort1[i2];
                let (death, risk) = trees.walkup_death_add_risk_walkup_risk(x[jj], -wt[jj]);
                leave_risk_set(&mut imat[jj], death);
                z2 -= cox_variance_term(wt[jj], risk);
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

        let mut j = i;
        while j < n && time[sort2[j]] == time[ii] {
            let jj = sort2[j];
            ndeath += 1;
            count[3] += wt[jj] * dwt * adjtimewt;
            dwt += wt[jj];
            if x[jj] != xsave {
                if wt[sort2[j2]] < dwt2 {
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

            let wsum = trees.walkup_risk_add_death(x[jj], adjtimewt * wt[jj]);
            for k in 0..3 {
                count[k] += wt[jj] * wsum[k] * adjtimewt;
                imat[jj][k] += wsum[k] * adjtimewt;
            }
            j += 1;
        }
        if wt[sort2[j2]] < dwt2 {
            while j2 < j {
                let kk = sort2[j2];
                imat[kk][4] += (dwt2 - wt[kk]) * adjtimewt;
                imat[kk][3] -= (dwt2 - wt[kk]) * adjtimewt;
                j2 += 1;
            }
        }

        for &jj in &sort2[i..i + ndeath] {
            let (death, risk) = trees.walkup_both_add_risk(x[jj], wt[jj]);
            enter_risk_set(&mut imat[jj], death);
            imat[jj][3] += (dwt - wt[jj]) * adjtimewt;
            z2 += cox_variance_term(wt[jj], risk);
        }
        let risk_weight = trees.total_risk();
        if risk_weight > 0.0 {
            count[5] += dwt * adjtimewt * z2 / risk_weight;
        }
        if ranks {
            for &jj in &sort2[i..i + ndeath] {
                let wsum = trees.walkup_risk(x[jj]);
                nevent_left -= 1;
                resid[nevent_left] = [
                    if risk_weight > 0.0 {
                        (wsum[0] - wsum[1]) / risk_weight
                    } else {
                        0.0
                    },
                    risk_weight * adjtimewt,
                    wt[jj],
                ];
            }
        }
        i += ndeath;
    }

    // The rows still at risk leave the risk set now.  The death tree is
    // final, so its sums depend only on the node: once there are at least as
    // many rows to settle as nodes, one walk per node in node order (a
    // sequential pass over the tree) beats a random walk per row.
    match input.sort_start {
        None => {
            // Right-censored rows are all still at risk, and `sort_stop` is
            // a permutation of the rows, so visit them in row order.
            let walks = trees.death_walkups();
            for (row, &node) in imat.iter_mut().zip(x) {
                leave_risk_set(row, walks[node]);
            }
        }
        Some(sort1) => {
            let remaining = &sort1[i2..];
            if remaining.len() >= ntree {
                let walks = trees.death_walkups();
                for &ii in remaining {
                    leave_risk_set(&mut imat[ii], walks[x[ii]]);
                }
            } else {
                for &ii in remaining {
                    leave_risk_set(&mut imat[ii], trees.walkup_death(x[ii]));
                }
            }
        }
    }
    count[3] -= count[4];
    SweepOutput {
        count,
        influence: imat,
        resid,
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FastKm {
    pub(crate) etime: Vec<f64>,
    pub(crate) surv: Vec<f64>,
    pub(crate) censor: Vec<f64>,
    pub(crate) nrisk: Vec<f64>,
}

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
                    ncount[i] = ntemp;
                    dcount[i] = dtemp;
                    i += 1;
                }
                nevent += usize::from(death_here);
            }
        }
        _ => {
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
    use crate::concordance::rank_tree::btree;
    use crate::internal::rng::Rng;

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

    /// The sweep before the risk-set and death trees shared one array: two
    /// separate [`RankTree`]s, walked and updated one call at a time.
    fn separate_tree_sweep(input: &SweepInput<'_>, std_err: bool, ranks: bool) -> SweepOutput {
        let n = input.stop.len();
        let ntree = input.node.iter().map(|&x| x + 1).max().unwrap_or(0);
        let nevent = input.status.iter().filter(|&&s| s == 1).count();
        let x = input.node;
        let wt = input.weight;
        let time = input.stop;
        let status = input.status;
        let sort2 = input.sort_stop;
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

            let mut j = i;
            while j < n && time[sort2[j]] == time[ii] {
                let jj = sort2[j];
                ndeath += 1;
                count[3] += wt[jj] * dwt * adjtimewt;
                dwt += wt[jj];
                if x[jj] != xsave {
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
                let risk_weight = tree.total();
                if risk_weight > 0.0 {
                    count[5] += dwt * adjtimewt * z2 / risk_weight;
                }
                if ranks {
                    for &jj in &sort2[i..i + ndeath] {
                        let wsum = tree.walkup(x[jj]);
                        nevent_left -= 1;
                        resid[nevent_left] = [
                            if risk_weight > 0.0 {
                                (wsum[0] - wsum[1]) / risk_weight
                            } else {
                                0.0
                            },
                            risk_weight * adjtimewt,
                            wt[jj],
                        ];
                    }
                }
            }
            i += ndeath;
        }

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
        count[3] -= count[4];
        SweepOutput {
            count,
            influence: imat,
            resid,
        }
    }

    fn assert_same_bits(actual: &SweepOutput, expected: &SweepOutput, case: &str) {
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&actual.count), bits(&expected.count), "count {case}");
        assert_eq!(
            bits(actual.influence.as_flattened()),
            bits(expected.influence.as_flattened()),
            "influence {case}"
        );
        assert_eq!(
            bits(actual.resid.as_flattened()),
            bits(expected.resid.as_flattened()),
            "resid {case}"
        );
    }

    #[test]
    fn sweep_matches_the_separate_tree_sweep_bit_for_bit() {
        let mut rng = Rng::with_seed(0x00c0_ffee);
        // Counting-process cases that settle the rows still at risk by
        // per-node walks and by per-row walks.
        let mut settled = [0usize; 2];
        for trial in 0..3000 {
            let n = 1 + rng.usize(0..if trial % 10 == 0 { 300 } else { 40 });
            let ntime = 1 + rng.usize(0..n + 2);
            let nx = 1 + rng.usize(0..if trial % 2 == 0 { 3 } else { n + 2 });
            let counting = trial % 3 == 0;
            let stop: Vec<f64> = (0..n).map(|_| (1 + rng.usize(0..ntime)) as f64).collect();
            let status: Vec<i32> = (0..n).map(|_| i32::from(rng.usize(0..3) != 0)).collect();
            let risk: Vec<f64> = (0..n).map(|_| rng.usize(0..nx) as f64 * 0.37).collect();
            let weight: Vec<f64> = (0..n)
                .map(|_| match (trial % 4, rng.usize(0..5)) {
                    (0, _) => 1.0,
                    (_, 0) => 0.0,
                    (_, 1) => 1.0,
                    _ => rng.f64() * 3.0,
                })
                .collect();
            let start: Option<Vec<f64>> = counting.then(|| {
                let lag = 1 + rng.usize(0..4);
                stop.iter()
                    .map(|&s| (s - rng.usize(1..=lag) as f64).max(-1.0))
                    .collect()
            });
            let mut levels = risk.clone();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            let tree = btree(levels.len());
            let node: Vec<usize> = risk
                .iter()
                .map(|v| tree[levels.partition_point(|l| l < v)])
                .collect();
            let mut sort_stop: Vec<usize> = (0..n).collect();
            sort_stop.sort_by(|&a, &b| {
                stop[b]
                    .total_cmp(&stop[a])
                    .then_with(|| status[a].cmp(&status[b]))
                    .then_with(|| risk[a].total_cmp(&risk[b]))
            });
            let sort_start: Option<Vec<usize>> = start.as_ref().map(|start| {
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| start[b].total_cmp(&start[a]));
                order
            });
            let mut etimes: Vec<f64> = (0..n)
                .filter(|&i| status[i] == 1)
                .map(|i| stop[i])
                .collect();
            etimes.sort_by(f64::total_cmp);
            etimes.dedup();
            if etimes.is_empty() {
                continue;
            }
            let timewt: Vec<f64> = etimes
                .iter()
                .map(|_| match (trial % 5, rng.usize(0..6)) {
                    (0, _) => 1.0,
                    (_, 0) => 0.0,
                    _ => rng.f64() * 2.0,
                })
                .collect();
            let input = SweepInput {
                start: start.as_deref(),
                stop: &stop,
                status: &status,
                node: &node,
                weight: &weight,
                timewt: &timewt,
                sort_start: sort_start.as_deref(),
                sort_stop: &sort_stop,
            };
            if let Some(start) = &start {
                // Rows starting at or after the earliest death time leave
                // the risk set during the sweep.
                let still_at_risk = start.iter().filter(|&&entry| entry < etimes[0]).count();
                settled[usize::from(still_at_risk >= levels.len())] += 1;
            }
            for (std_err, ranks) in [(true, true), (true, false), (false, true), (false, false)] {
                assert_same_bits(
                    &concordance_sweep(&input, std_err, ranks),
                    &separate_tree_sweep(&input, std_err, ranks),
                    &format!("trial {trial} counting {counting} std_err {std_err} ranks {ranks}"),
                );
            }
        }
        assert!(settled.iter().all(|&cases| cases > 50), "{settled:?}");
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
