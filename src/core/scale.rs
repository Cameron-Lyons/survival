use pyo3::prelude::*;
use rayon::prelude::*;

type ScaleResult = (Vec<f64>, Option<f64>, Option<f64>);

fn fitted_center(x: &[f64]) -> f64 {
    let (sum, count) = x
        .iter()
        .filter(|value| !value.is_nan())
        .fold((0.0, 0usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    sum / count as f64
}

fn fitted_scale(x: &[f64], center: Option<f64>) -> f64 {
    let (sum_squares, count) = x.iter().fold((0.0, 0usize), |(sum_squares, count), value| {
        let centered = center.map_or(*value, |location| *value - location);
        if centered.is_nan() {
            (sum_squares, count)
        } else {
            (centered.mul_add(centered, sum_squares), count + 1)
        }
    });
    (sum_squares / count.saturating_sub(1).max(1) as f64).sqrt()
}

pub(crate) fn scale_values_core(
    x: &[f64],
    center: bool,
    scale: bool,
    center_value: Option<f64>,
    scale_value: Option<f64>,
) -> ScaleResult {
    let effective_center = center_value.or_else(|| center.then(|| fitted_center(x)));
    let effective_scale = scale_value.or_else(|| scale.then(|| fitted_scale(x, effective_center)));
    let values = x
        .par_iter()
        .map(|value| {
            let centered = effective_center.map_or(*value, |location| *value - location);
            effective_scale.map_or(centered, |divisor| centered / divisor)
        })
        .collect();
    (values, effective_center, effective_scale)
}

#[pyfunction]
#[pyo3(signature = (x, center=true, scale=true, center_value=None, scale_value=None))]
pub fn scale_values(
    x: Vec<f64>,
    center: bool,
    scale: bool,
    center_value: Option<f64>,
    scale_value: Option<f64>,
) -> ScaleResult {
    scale_values_core(&x, center, scale, center_value, scale_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitted_values_match_r_scale_fixture() {
        let x = [1.0, 2.0, f64::NAN, 4.0, 8.0];
        let (values, center, scale) = scale_values_core(&x, true, true, None, None);
        let expected = [
            -0.888330138395973,
            -0.565300997161074,
            f64::NAN,
            0.0807572853087248,
            1.37287385024832,
        ];

        assert_eq!(center, Some(3.75));
        assert!((scale.unwrap() - 3.09569593683445).abs() < 1e-14);
        for (actual, expected) in values.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn uncentered_scale_and_center_only_match_r() {
        let x = [1.0, 2.0, f64::NAN, 4.0, 8.0];
        let (uncentered, center, scale) = scale_values_core(&x, false, true, None, None);
        assert_eq!(center, None);
        assert!((scale.unwrap() - 5.32290647422377).abs() < 1e-14);
        assert!((uncentered[0] - 0.187867287325545).abs() < 1e-14);

        let (centered, center, scale) = scale_values_core(&x, true, false, None, None);
        assert_eq!(center, Some(3.75));
        assert_eq!(scale, None);
        assert_eq!(centered[0], -2.75);
        assert!(centered[2].is_nan());
    }

    #[test]
    fn supplied_state_rebuilds_new_values() {
        let (values, center, scale) =
            scale_values_core(&[3.0, 6.0], false, false, Some(4.8), Some(3.56370593624109));
        assert_eq!(center, Some(4.8));
        assert_eq!(scale, Some(3.56370593624109));
        assert!((values[0] - -0.505092179939682).abs() < 1e-14);
        assert!((values[1] - 0.336728119959788).abs() < 1e-14);
    }

    #[test]
    fn zero_scale_preserves_ieee_results() {
        let (constant, center, scale) = scale_values_core(&[2.0; 4], true, true, None, None);
        assert_eq!(center, Some(2.0));
        assert_eq!(scale, Some(0.0));
        assert!(constant.iter().all(|value| value.is_nan()));

        let (signed, _, _) =
            scale_values_core(&[1.0, 2.0, 4.0], false, false, Some(2.0), Some(0.0));
        assert_eq!(signed[0], f64::NEG_INFINITY);
        assert!(signed[1].is_nan());
        assert_eq!(signed[2], f64::INFINITY);

        let (missing_center, _, fitted) =
            scale_values_core(&[1.0, 2.0], false, true, Some(f64::NAN), None);
        assert_eq!(fitted, Some(0.0));
        assert!(missing_center.iter().all(|value| value.is_nan()));
    }
}
