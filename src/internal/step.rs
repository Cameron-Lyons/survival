pub(crate) fn find_interval(times: &[f64], t: f64, left_open: bool) -> usize {
    if left_open {
        times.partition_point(|&x| x < t)
    } else {
        times.partition_point(|&x| x <= t)
    }
}

pub(crate) fn step_at(times: &[f64], values: &[f64], t: f64, initial: f64) -> f64 {
    match find_interval(times, t, false) {
        0 => initial,
        index => values[index - 1],
    }
}

pub(crate) fn rank_average(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average = (start + 1 + end) as f64 / 2.0;
        for &row in &order[start..end] {
            ranks[row] = average;
        }
        start = end;
    }
    ranks
}

pub(crate) fn sort_unique(values: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.into_iter().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_interval_matches_r() {
        let times = [1.0, 2.0, 3.0];
        let at = [0.0, 1.0, 1.5, 3.0, 4.0];
        let closed: Vec<usize> = at
            .iter()
            .map(|&t| find_interval(&times, t, false))
            .collect();
        let open: Vec<usize> = at.iter().map(|&t| find_interval(&times, t, true)).collect();
        assert_eq!(closed, [0, 1, 1, 3, 3]);
        assert_eq!(open, [0, 0, 1, 2, 3]);
        assert_eq!(find_interval(&[], 1.0, false), 0);
    }

    #[test]
    fn step_at_reads_the_curve_from_each_time_on() {
        let times = [1.0, 2.0, 3.0];
        let values = [0.9, 0.7, 0.4];
        assert_eq!(step_at(&times, &values, 0.5, 1.0), 1.0);
        assert_eq!(step_at(&times, &values, 1.0, 1.0), 0.9);
        assert_eq!(step_at(&times, &values, 2.5, 1.0), 0.7);
        assert_eq!(step_at(&times, &values, 9.0, 1.0), 0.4);
        assert_eq!(step_at(&[], &[], 9.0, -1.0), -1.0);
    }

    #[test]
    fn rank_average_matches_r() {
        assert_eq!(
            rank_average(&[3.0, 1.0, 3.0, 2.0, 3.0, -1.0]),
            [5.0, 2.0, 5.0, 3.0, 5.0, 1.0]
        );
        assert!(rank_average(&[]).is_empty());
    }

    #[test]
    fn sort_unique_drops_exact_duplicates() {
        assert_eq!(
            sort_unique([3.0, 1.0, 3.0, 2.0, 1.0 + 1e-12]),
            [1.0, 1.0 + 1e-12, 2.0, 3.0]
        );
    }
}
