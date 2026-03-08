use std::collections::HashMap;

pub fn concordance(
    start: &[f64],
    stop: &[f64],
    status: &[u32],
    x: &[f64],
    weights: Option<&[f64]>,
    strata: Option<&[usize]>,
    reverse: bool,
) -> (f64, f64, f64, f64) {
    let weights = weights.unwrap_or_else(|| &vec![1.0; start.len()]);
    let strata = strata.unwrap_or_else(|| &vec![0; start.len()]);

    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut tied_x = 0.0;
    let mut tied_y = 0.0;

    let mut strata_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &s) in strata.iter().enumerate() {
        strata_indices.entry(s).or_default().push(i);
    }

    for indices in strata_indices.values() {
        let (c, d, tx, ty) =
            calculate_stratum_concordance(start, stop, status, x, weights, indices, reverse);
        concordant += c;
        discordant += d;
        tied_x += tx;
        tied_y += ty;
    }

    (concordant, discordant, tied_x, tied_y)
}

fn calculate_stratum_concordance(
    start: &[f64],
    stop: &[f64],
    status: &[u32],
    x: &[f64],
    weights: &[f64],
    indices: &[usize],
    reverse: bool,
) -> (f64, f64, f64, f64) {
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut tied_x = 0.0;
    let mut tied_y = 0.0;

    for &i in indices {
        if status[i] != 1 {
            continue;
        }

        let event_time = stop[i];
        let x_i = x[i];
        let weight_i = weights[i];

        let at_risk: Vec<usize> = indices
            .iter()
            .filter(|&&j| {
                start[j] < event_time
                    && event_time <= stop[j]
                    && (stop[j] > event_time || (stop[j] == event_time && status[j] == 0))
            })
            .cloned()
            .collect();

        let (mut conc, mut disc, mut tied) = (0.0, 0.0, 0.0);
        for &j in &at_risk {
            let diff = if reverse { x[j] - x_i } else { x_i - x[j] };

            match diff.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Greater => conc += weights[j],
                std::cmp::Ordering::Less => disc += weights[j],
                std::cmp::Ordering::Equal => tied += weights[j],
            }
        }

        concordant += weight_i * conc;
        discordant += weight_i * disc;
        tied_x += weight_i * tied;

        let tied_y_contribution: f64 = indices
            .iter()
            .filter(|&&j| stop[j] == event_time && status[j] == 1)
            .map(|&j| weights[j])
            .sum();
        tied_y += weight_i * (tied_y_contribution - weight_i) / 2.0;
    }

    (concordant, discordant, tied_x, tied_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_maintained_data() {
        let start = vec![0.0; 11];
        let stop = vec![
            9.0, 13.0, 28.0, 31.0, 7.0, 16.0, 23.0, 34.0, 45.0, 48.0, 60.0,
        ]; 
        let status = vec![1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]; 
        let x = vec![1.0, 6.0, 2.0, 7.0, 3.0, 7.0, 3.0, 8.0, 4.0, 4.0, 5.0];
        let weights = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        let (conc, disc, tx, ty) =
            concordance(&start, &stop, &status, &x, Some(&weights), None, false);

        assert_abs_diff_eq!(conc, 24.0, epsilon = 1e-6);
        assert_abs_diff_eq!(disc, 14.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx, 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ty, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_case_weights() {
        let start = vec![0.0; 11];
        let stop = vec![
            9.0, 13.0, 28.0, 31.0, 7.0, 16.0, 23.0, 34.0, 45.0, 48.0, 60.0,
        ];
        let status = vec![1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0];
        let x = vec![1.0, 6.0, 2.0, 7.0, 3.0, 7.0, 3.0, 8.0, 4.0, 4.0, 5.0];
        let weights = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        let (conc, disc, tx, ty) = concordance(
            &start,
            &stop,
            &status,
            &x,
            Some(&weights),
            None,
            true, 
        );

        assert_abs_diff_eq!(conc, 70.0, epsilon = 1e-6);
        assert_abs_diff_eq!(disc, 91.0, epsilon = 1e-6);
        assert_abs_diff_eq!(tx, 7.0, epsilon = 1e-6);
        assert_abs_diff_eq!(ty, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tied_data() {
        let start = vec![0.0; 10];
        let stop = vec![1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 2.0];
        let status = vec![1, 0, 1, 0, 1, 0, 1, 1, 0, 1];
        let x = vec![5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 7.0, 6.0, 5.0, 4.0];

        let (conc, disc, tx, ty) = concordance(&start, &stop, &status, &x, None, None, false);

        assert_abs_diff_eq!(conc + disc + tx + ty, 43.0, epsilon = 1e-6);
    }
}
