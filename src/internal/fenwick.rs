pub(crate) struct FenwickTree {
    tree: Vec<f64>,
    total: f64,
}

impl FenwickTree {
    #[inline]
    pub(crate) fn new(size: usize) -> Self {
        FenwickTree {
            tree: vec![0.0; size + 1],
            total: 0.0,
        }
    }

    #[inline]
    pub(crate) fn update(&mut self, index: usize, value: f64) {
        let mut idx = index.saturating_add(1);
        if idx >= self.tree.len() {
            return;
        }
        self.total += value;
        while idx < self.tree.len() {
            self.tree[idx] += value;
            idx += idx & (!idx + 1);
        }
    }

    #[inline]
    pub(crate) fn prefix_sum(&self, index: usize) -> f64 {
        let mut sum = 0.0;
        let mut idx = index.saturating_add(1).min(self.tree.len() - 1);
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx & (!idx + 1);
        }
        sum
    }

    #[inline]
    pub(crate) fn total(&self) -> f64 {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fenwick_tree_basic() {
        let mut tree = FenwickTree::new(10);
        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        assert!((tree.prefix_sum(0) - 1.0).abs() < 1e-10);
        assert!((tree.prefix_sum(1) - 3.0).abs() < 1e-10);
        assert!((tree.prefix_sum(2) - 6.0).abs() < 1e-10);
        assert!((tree.total() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_fenwick_tree_empty() {
        let tree = FenwickTree::new(5);
        assert!((tree.total()).abs() < 1e-10);
    }

    #[test]
    fn test_fenwick_tree_total_tracks_negative_updates() {
        let mut tree = FenwickTree::new(3);
        tree.update(0, 2.5);
        tree.update(1, 1.5);
        tree.update(0, -0.5);

        assert!((tree.prefix_sum(1) - 3.5).abs() < 1e-10);
        assert!((tree.total() - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_fenwick_tree_zero_sized_total_is_zero() {
        let tree = FenwickTree::new(0);
        assert!((tree.total()).abs() < 1e-10);
        assert!((tree.prefix_sum(0)).abs() < 1e-10);
    }

    #[test]
    fn test_fenwick_tree_out_of_range_update_is_ignored() {
        let mut tree = FenwickTree::new(2);
        tree.update(3, 10.0);

        assert!((tree.total()).abs() < 1e-10);
        assert!((tree.prefix_sum(1)).abs() < 1e-10);
    }

    #[test]
    fn test_fenwick_tree_out_of_range_prefix_sum_returns_total() {
        let mut tree = FenwickTree::new(2);
        tree.update(0, 2.0);
        tree.update(1, 3.0);

        assert!((tree.prefix_sum(usize::MAX) - tree.total()).abs() < 1e-10);
        assert!((tree.prefix_sum(10) - tree.total()).abs() < 1e-10);
    }
}
