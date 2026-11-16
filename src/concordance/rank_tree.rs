//! Weighted rank trees for the concordance sweep (the `walkup` and `addin`
//! of R's concordance C code).
//!
//! The predictor's sorted unique values are laid out as a balanced binary
//! tree in heap order ([`btree`]).  Each node keeps the weight at its value
//! and the total weight of its subtree, so the weight above, below and at a
//! value is one leaf-to-root walk, and adding weight is another.

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

/// A walk's sums start from the node's own weight and its children's
/// subtree totals: `[above, below, at]`.
fn walk_start(own: f64, children: impl Fn(usize) -> f64, index: usize, ntree: usize) -> [f64; 3] {
    let mut sums = [0.0, 0.0, own];
    let right = 2 * index + 2;
    if right < ntree {
        sums[0] += children(right);
    }
    if right <= ntree {
        sums[1] += children(right - 1);
    }
    sums
}

/// Adds the weight a step from `index` to its parent passes: the parent's
/// other weight lies above a left (odd) child and below a right (even) one.
fn add_step(sums: &mut [f64; 3], index: usize, weight: f64) {
    if index % 2 == 1 {
        sums[0] += weight;
    } else {
        sums[1] += weight;
    }
}

/// One tree: the weight at each node and the total of each subtree.
pub(crate) struct RankTree {
    nwt: Vec<f64>,
    twt: Vec<f64>,
}

impl RankTree {
    pub(crate) fn new(ntree: usize) -> Self {
        Self {
            nwt: vec![0.0; ntree],
            twt: vec![0.0; ntree],
        }
    }

    pub(crate) fn walkup(&self, mut index: usize) -> [f64; 3] {
        let mut sums = walk_start(self.nwt[index], |c| self.twt[c], index, self.nwt.len());
        while index > 0 {
            let parent = (index - 1) / 2;
            add_step(&mut sums, index, self.twt[parent] - self.twt[index]);
            index = parent;
        }
        sums
    }

    pub(crate) fn addin(&mut self, mut index: usize, wt: f64) {
        self.nwt[index] += wt;
        while index > 0 {
            self.twt[index] += wt;
            index = (index - 1) / 2;
        }
        self.twt[0] += wt;
    }

    /// The sweep reads the risk-set total from [`PairedRankTree`]; the
    /// tests read it here to check that tree against this one.
    #[cfg(test)]
    pub(crate) fn total(&self) -> f64 {
        self.twt[0]
    }
}

/// One node of [`PairedRankTree`]: a node of the risk-set tree and the same
/// node of the death tree.  Aligned to 32 bytes so that a node never
/// straddles a cache line.
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct PairedNode {
    risk_total: f64,
    death_total: f64,
    risk: f64,
    death: f64,
}

/// The risk-set tree and the death tree of the influence sweep, node by node.
///
/// The influence sweep walks both trees along the same leaf-to-root path,
/// often several times per observation.  Keeping the two trees in separate
/// arrays makes every step of every walk a cache miss in each array once the
/// trees outgrow the cache, so the trees share one array and the methods fuse
/// the walks and additions the sweep makes at one node into a single pass.
/// Each fused method reads every weight before it writes it and adds in the
/// same order as the separate [`RankTree`] calls it replaces, so the sums are
/// bitwise identical to theirs.
pub(crate) struct PairedRankTree {
    nodes: Vec<PairedNode>,
}

impl PairedRankTree {
    pub(crate) fn new(ntree: usize) -> Self {
        Self {
            nodes: vec![PairedNode::default(); ntree],
        }
    }

    fn start(&self, index: usize, own: f64, total: impl Fn(&PairedNode) -> f64) -> [f64; 3] {
        walk_start(own, |c| total(&self.nodes[c]), index, self.nodes.len())
    }

    /// The risk-set tree's `RankTree::walkup`.
    pub(crate) fn walkup_risk(&self, mut index: usize) -> [f64; 3] {
        let mut sums = self.start(index, self.nodes[index].risk, |node| node.risk_total);
        while index > 0 {
            let parent = (index - 1) / 2;
            add_step(
                &mut sums,
                index,
                self.nodes[parent].risk_total - self.nodes[index].risk_total,
            );
            index = parent;
        }
        sums
    }

    /// The death tree's `RankTree::walkup`.
    pub(crate) fn walkup_death(&self, mut index: usize) -> [f64; 3] {
        let mut sums = self.start(index, self.nodes[index].death, |node| node.death_total);
        while index > 0 {
            let parent = (index - 1) / 2;
            add_step(
                &mut sums,
                index,
                self.nodes[parent].death_total - self.nodes[index].death_total,
            );
            index = parent;
        }
        sums
    }

    /// Walks the risk-set tree, then adds `wt` to the death tree at `index`.
    pub(crate) fn walkup_risk_add_death(&mut self, mut index: usize, wt: f64) -> [f64; 3] {
        let mut risk = self.start(index, self.nodes[index].risk, |node| node.risk_total);
        self.nodes[index].death += wt;
        while index > 0 {
            let parent = (index - 1) / 2;
            add_step(
                &mut risk,
                index,
                self.nodes[parent].risk_total - self.nodes[index].risk_total,
            );
            self.nodes[index].death_total += wt;
            index = parent;
        }
        self.nodes[0].death_total += wt;
        risk
    }

    /// Walks the death tree and the risk-set tree, then adds `wt` to the
    /// risk-set tree at `index`: `(death sums, risk sums)` before the
    /// addition.
    pub(crate) fn walkup_both_add_risk(
        &mut self,
        mut index: usize,
        wt: f64,
    ) -> ([f64; 3], [f64; 3]) {
        let mut death = self.start(index, self.nodes[index].death, |node| node.death_total);
        let mut risk = self.start(index, self.nodes[index].risk, |node| node.risk_total);
        self.nodes[index].risk += wt;
        while index > 0 {
            let parent = (index - 1) / 2;
            let (node, up) = (self.nodes[index], self.nodes[parent]);
            add_step(&mut death, index, up.death_total - node.death_total);
            add_step(&mut risk, index, up.risk_total - node.risk_total);
            self.nodes[index].risk_total = node.risk_total + wt;
            index = parent;
        }
        self.nodes[0].risk_total += wt;
        (death, risk)
    }

    /// Walks the death tree, adds `wt` to the risk-set tree at `index` and
    /// walks the risk-set tree again: `(death sums, risk sums after the
    /// addition)`.  The risk walk reads each total on the path as
    /// `total + wt`, the value the addition stores there.
    pub(crate) fn walkup_death_add_risk_walkup_risk(
        &mut self,
        mut index: usize,
        wt: f64,
    ) -> ([f64; 3], [f64; 3]) {
        let mut death = self.start(index, self.nodes[index].death, |node| node.death_total);
        self.nodes[index].risk += wt;
        let mut risk = self.start(index, self.nodes[index].risk, |node| node.risk_total);
        while index > 0 {
            let parent = (index - 1) / 2;
            let (node, up) = (self.nodes[index], self.nodes[parent]);
            let total = node.risk_total + wt;
            add_step(&mut death, index, up.death_total - node.death_total);
            add_step(&mut risk, index, (up.risk_total + wt) - total);
            self.nodes[index].risk_total = total;
            index = parent;
        }
        self.nodes[0].risk_total += wt;
        (death, risk)
    }

    /// The death tree's walk from every node, in node order.  The walks
    /// from consecutive nodes share their ancestors, so this streams through
    /// the tree.
    pub(crate) fn death_walkups(&self) -> Vec<[f64; 3]> {
        (0..self.nodes.len())
            .map(|index| self.walkup_death(index))
            .collect()
    }

    pub(crate) fn total_risk(&self) -> f64 {
        self.nodes[0].risk_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::rng::Rng;

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

    fn bits(sums: [f64; 3]) -> [u64; 3] {
        sums.map(f64::to_bits)
    }

    #[test]
    fn paired_operations_match_separate_trees_bit_for_bit() {
        let mut rng = Rng::with_seed(0x5eed_c0de);
        for trial in 0..400 {
            let ntree = 1 + rng.usize(0..if trial % 10 == 0 { 300 } else { 20 });
            let mut risk = RankTree::new(ntree);
            let mut death = RankTree::new(ntree);
            let mut paired = PairedRankTree::new(ntree);
            for _ in 0..4 * ntree {
                let index = rng.usize(0..ntree);
                let wt = match rng.usize(0..4) {
                    0 => 0.0,
                    1 => 1.0,
                    2 => -rng.f64() * 3.0,
                    _ => rng.f64() * 3.0,
                };
                match rng.usize(0..3) {
                    0 => {
                        let expected = risk.walkup(index);
                        death.addin(index, wt);
                        let actual = paired.walkup_risk_add_death(index, wt);
                        assert_eq!(bits(actual), bits(expected));
                    }
                    1 => {
                        let expected = (death.walkup(index), risk.walkup(index));
                        risk.addin(index, wt);
                        let actual = paired.walkup_both_add_risk(index, wt);
                        assert_eq!(bits(actual.0), bits(expected.0));
                        assert_eq!(bits(actual.1), bits(expected.1));
                    }
                    _ => {
                        let before = death.walkup(index);
                        risk.addin(index, wt);
                        let expected = (before, risk.walkup(index));
                        let actual = paired.walkup_death_add_risk_walkup_risk(index, wt);
                        assert_eq!(bits(actual.0), bits(expected.0));
                        assert_eq!(bits(actual.1), bits(expected.1));
                    }
                }
                assert_eq!(paired.total_risk().to_bits(), risk.total().to_bits());
            }
            for (index, walk) in paired.death_walkups().into_iter().enumerate() {
                assert_eq!(bits(paired.walkup_risk(index)), bits(risk.walkup(index)));
                assert_eq!(bits(paired.walkup_death(index)), bits(death.walkup(index)));
                assert_eq!(bits(walk), bits(death.walkup(index)));
            }
        }
    }
}
