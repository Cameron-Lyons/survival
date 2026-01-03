pub struct FenwickTree {
    tree: Vec<f64>,
}

impl FenwickTree {
    #[inline]
    pub fn new(size: usize) -> Self {
        FenwickTree {
            tree: vec![0.0; size + 1],
        }
    }

    #[inline]
    pub fn update(&mut self, index: usize, value: f64) {
        let mut idx = index + 1;
        while idx < self.tree.len() {
            self.tree[idx] += value;
            idx += idx & (!idx + 1);
        }
    }

    #[inline]
    pub fn prefix_sum(&self, index: usize) -> f64 {
        let mut sum = 0.0;
        let mut idx = index + 1;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx & (!idx + 1);
        }
        sum
    }

    #[inline]
    pub fn total(&self) -> f64 {
        self.prefix_sum(self.tree.len() - 2)
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
}
