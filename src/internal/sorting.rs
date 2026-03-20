use crate::constants::PARALLEL_THRESHOLD_LARGE;
use rayon::prelude::*;

pub(crate) fn descending_time_indices(time: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..time.len()).collect();
    if time.len() > PARALLEL_THRESHOLD_LARGE {
        indices.par_sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descending_time_indices_orders_values_descending() {
        let time = vec![2.0, 5.0, 1.0, 4.0];
        let indices = descending_time_indices(&time);
        let ordered: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();

        assert_eq!(ordered, vec![5.0, 4.0, 2.0, 1.0]);

        let mut sorted_indices = indices;
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3]);
    }
}
